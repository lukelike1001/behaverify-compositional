"""
nav_tree_generator.py

Emit a BehaVerify .tree for the NAV benchmark on an integer lattice.

The plant is continuous; the monolithic table approach needs finitely many NN
inputs, so state is stored as round(value * scale) and the dynamics are done in
integer arithmetic. Two consequences shape the generated model:

  * cos/sin are precomputed into static DEFINE arrays indexed by the heading
    lattice, rather than emitted as SMV trig. ACAS Xu does the same.
  * the network is a `regression` variable whose ONNX has been wrapped to emit
    the integer state delta directly, because dsl_to_nuxmv truncates raw
    outputs with int() and NAV's tanh outputs would all truncate to zero.

Usage (from examples/NAV/):

    python3 -m core.nav_tree_generator --network set --output tree/nav_set.tree
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "nav_domain_config.yaml"


@dataclass(frozen=True)
class NavDomain:
    """Lattice, bounds, and specification regions for the NAV benchmark."""

    scale: int
    dt_num: int
    dt_den: int
    trig_scale: int
    bounds: dict[str, tuple[float, float]]
    initial_state: dict[str, float]
    obstacle: dict[str, tuple[float, float]]
    goal: dict[str, tuple[float, float]]
    horizon_steps: int
    networks: dict[str, str]

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG) -> NavDomain:
        cfg: dict[str, Any] = yaml.safe_load(path.read_text())
        lattice = cfg["lattice"]
        return cls(
            scale=int(lattice["scale"]),
            dt_num=int(lattice["dt_num"]),
            dt_den=int(lattice["dt_den"]),
            trig_scale=int(cfg["trig_scale"]),
            bounds={k: tuple(v) for k, v in cfg["bounds"].items()},
            initial_state=dict(cfg["initial_state"]),
            obstacle={k: tuple(v) for k, v in cfg["obstacle"].items()},
            goal={k: tuple(v) for k, v in cfg["goal"].items()},
            horizon_steps=int(cfg["horizon_steps"]),
            networks=dict(cfg["networks"]),
        )

    # --- lattice helpers ---------------------------------------------------

    def to_lattice(self, value: float) -> int:
        return round(value * self.scale)

    def bound_ints(self, name: str) -> tuple[int, int]:
        low, high = self.bounds[name]
        return self.to_lattice(low), self.to_lattice(high)

    def size(self, name: str) -> int:
        low, high = self.bound_ints(name)
        return high - low + 1

    def table_entries(self) -> int:
        """How many NN evaluations the monolithic table needs."""
        total = 1
        for name in ("x1", "x2", "x3", "x4"):
            total *= self.size(name)
        return total

    def trig_table(self, fn) -> list[int]:
        """cos or sin over the heading lattice, scaled to integers."""
        low, high = self.bound_ints("x4")
        return [
            round(fn(index / self.scale) * self.trig_scale)
            for index in range(low, high + 1)
        ]


def _static_int_array(name: str, values: list[int]) -> str:
    """A DEFINE array literal in BehaVerify's iterative_assign form."""
    conditions = "".join(
        f"condition {{(eq, index_var, {i})}} assign{{result{{{v}}}}}"
        + ("\n\t" if (i + 1) % 6 == 0 else "")
        for i, v in enumerate(values)
    )
    return (
        f"    variable {{ env {name} DEFINE INT static array {len(values)} "
        f"iterative_assign, index_var\n\t{conditions}\n"
        f"\tassign{{result{{0}}}}\n    }}"
    )


def _region_condition(
    domain: NavDomain, region: dict[str, tuple[float, float]],
) -> str:
    """Prefix-notation conjunction saying the position is inside `region`."""
    clauses = []
    for axis in ("x1", "x2"):
        low, high = region[axis]
        clauses.append(f"(gte, {axis}, {domain.to_lattice(low)})")
        clauses.append(f"(lte, {axis}, {domain.to_lattice(high)})")
    condition = clauses[0]
    for clause in clauses[1:]:
        condition = f"(and, {condition}, {clause})"
    return condition


def build_tree(domain: NavDomain, network_key: str) -> str:
    """Render the whole .tree file as text."""
    onnx_path = domain.networks[network_key]
    scale = domain.scale
    x1_lo, x1_hi = domain.bound_ints("x1")
    x2_lo, x2_hi = domain.bound_ints("x2")
    x3_lo, x3_hi = domain.bound_ints("x3")
    x4_lo, x4_hi = domain.bound_ints("x4")

    cos_values = domain.trig_table(math.cos)
    sin_values = domain.trig_table(math.sin)

    # x1' = x1 + x3 * cos(x4) * dt, all on the lattice. cos is stored scaled by
    # trig_scale and dt = dt_num/dt_den, so divide by trig_scale * dt_den and
    # multiply by dt_num. Heading indexes the table from its lower bound.
    heading_index = f"(sub, x4, {x4_lo})"
    divisor = domain.trig_scale * domain.dt_den
    step_x1 = (
        f"(idiv, (mult, (mult, x3, (index, cos_table, {heading_index})), "
        f"{domain.dt_num}), {divisor})"
    )
    step_x2 = (
        f"(idiv, (mult, (mult, x3, (index, sin_table, {heading_index})), "
        f"{domain.dt_num}), {divisor})"
    )

    obstacle_condition = _region_condition(domain, domain.obstacle)
    goal_condition = _region_condition(domain, domain.goal)

    return f"""configuration {{
    #{{ NAV benchmark, ARCH-COMP 2025 AINNCS Section 3.11. }}#
    #{{ State is stored on an integer lattice: value * {scale}. }}#
    neural
}}
enumerations {{
}}
constants {{
    scale := {scale}, horizon := {domain.horizon_steps}
}} end_constants

variables {{
    variable {{ bl step VAR [0, horizon] assign{{result{{0}}}}}}
    variable {{ env x1 VAR [{x1_lo}, {x1_hi}] assign{{result{{{domain.to_lattice(domain.initial_state['x1'])}}}}}}}
    variable {{ env x2 VAR [{x2_lo}, {x2_hi}] assign{{result{{{domain.to_lattice(domain.initial_state['x2'])}}}}}}}
    variable {{ env x3 VAR [{x3_lo}, {x3_hi}] assign{{result{{{domain.to_lattice(domain.initial_state['x3'])}}}}}}}
    variable {{ env x4 VAR [{x4_lo}, {x4_hi}] assign{{result{{{domain.to_lattice(domain.initial_state['x4'])}}}}}}}

    #{{ The ONNX wrapper takes lattice ints and returns integer state deltas,
       so the tree passes plain variables. See networks/README.md. }}#
    variable {{ bl control NEURAL regression INT
	inputs {{x1, x2, x3, x4}} end_inputs
	num_outputs {{2}} end_num_outputs
	config {{
	    table
	}}
	source {{
	    '{onnx_path}'
	}}
    }}

    #{{ cos/sin over the heading lattice, scaled by {domain.trig_scale}. }}#
{_static_int_array('cos_table', cos_values)}
{_static_int_array('sin_table', sin_values)}
}} end_variables

environment_update {{
    variable_statement {{
	x1
	assign {{ result {{(max, {x1_lo}, (min, {x1_hi}, (add, x1, {step_x1})))}} }}
    }}
    variable_statement {{
	x2
	assign {{ result {{(max, {x2_lo}, (min, {x2_hi}, (add, x2, {step_x2})))}} }}
    }}
    variable_statement {{
	x3
	assign {{ result {{(max, {x3_lo}, (min, {x3_hi}, (add, x3, (index, control, 0))))}} }}
    }}
    variable_statement {{
	x4
	assign {{ result {{(max, {x4_lo}, (min, {x4_hi}, (add, x4, (index, control, 1))))}} }}
    }}
}} end_environment_update

checks {{
    check {{
	at_horizon
	arguments {{}}
	read_variables {{step}}
	condition {{(gte, step, horizon)}}
    }}
}} end_checks

environment_checks {{
}} end_environment_checks

actions {{
    action {{
	hold
	arguments{{}}
	local_variables {{}} end_local_variables
	read_variables {{}} end_read_variables
	write_variables {{}} end_write_variables
	initial_values {{}} end_initial_values
	update {{
	    return_statement {{ result {{success}} end_result }} end_return_statement
	}} end_update
    }}
    action {{
	advance
	arguments{{}}
	local_variables {{}} end_local_variables
	read_variables {{step}} end_read_variables
	write_variables {{step}} end_write_variables
	initial_values {{}} end_initial_values
	update {{
	    variable_statement {{
		step
		assign{{result{{(min, horizon, (add, step, 1))}}}}
	    }}
	    return_statement {{ result {{success}} end_result }} end_return_statement
	}} end_update
    }}
}} end_actions

sub_trees {{}} end_sub_trees

tree {{
    composite {{
	nav_control selector
	children {{
	    composite {{
		horizon_reached sequence
		children {{
		    at_horizon {{}}
		    hold {{}}
		}}
	    }}
	    advance {{}}
	}}
    }}
}} end_tree

tick_prerequisite {{ (True) }} end_tick_prerequisite

specifications {{
    #{{ Safety: never inside the obstacle. }}#
    INVARSPEC {{
	(not, {obstacle_condition})
    }}
    #{{ Bounded reach: the goal region is entered within the horizon. }}#
    CTLSPEC {{
	(always_finally, {goal_condition})
    }}
}} end_specifications
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="set", choices=("point", "set"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    domain = NavDomain.from_yaml(Path(args.config))
    text = build_tree(domain, args.network)

    output = Path(args.output) if args.output else (
        EXAMPLE_ROOT / "tree" / f"nav_{args.network}.tree"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)

    sizes = {n: domain.size(n) for n in ("x1", "x2", "x3", "x4")}
    print(f"wrote {output}")
    print(f"  lattice resolution = {1 / domain.scale}")
    print(f"  per-axis sizes     = {sizes}")
    print(f"  TABLE ENTRIES      = {domain.table_entries():,}")


if __name__ == "__main__":
    main()
