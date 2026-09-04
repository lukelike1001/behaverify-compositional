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

    position_scale: int
    speed_scale: int
    heading_scale: int
    dt_num: int
    dt_den: int
    trig_scale: int
    bounds: dict[str, tuple[float, float]]
    initial_state: dict[str, float]
    obstacle: dict[str, tuple[float, float]]
    goal: dict[str, tuple[float, float]]
    horizon_steps: int
    networks: dict[str, str]
    successor: str = "deterministic"

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG) -> NavDomain:
        cfg: dict[str, Any] = yaml.safe_load(path.read_text())
        lattice = cfg["lattice"]
        return cls(
            position_scale=int(lattice["position_scale"]),
            speed_scale=int(lattice["speed_scale"]),
            heading_scale=int(lattice["heading_scale"]),
            dt_num=int(lattice["dt_num"]),
            dt_den=int(lattice["dt_den"]),
            trig_scale=int(cfg["trig_scale"]),
            bounds={k: tuple(v) for k, v in cfg["bounds"].items()},
            initial_state=dict(cfg["initial_state"]),
            obstacle={k: tuple(v) for k, v in cfg["obstacle"].items()},
            goal={k: tuple(v) for k, v in cfg["goal"].items()},
            horizon_steps=int(cfg["horizon_steps"]),
            networks=dict(cfg["networks"]),
            successor=str(cfg.get("successor", "deterministic")),
        )

    # --- lattice helpers ---------------------------------------------------

    @property
    def dt(self) -> float:
        return self.dt_num / self.dt_den

    def scale_for(self, name: str) -> int:
        """Which lattice an axis lives on."""
        return {
            "x1": self.position_scale,
            "x2": self.position_scale,
            "x3": self.speed_scale,
            "x4": self.heading_scale,
        }[name]

    def to_lattice(self, value: float, name: str) -> int:
        return _round_half_away(value * self.scale_for(name))

    def control_levels(self, name: str) -> int:
        """Half-width of the integer control delta on this axis."""
        return max(1, round(self.scale_for(name) * self.dt))

    def bound_ints(self, name: str) -> tuple[int, int]:
        low, high = self.bounds[name]
        return self.to_lattice(low, name), self.to_lattice(high, name)

    def wrapper_path(self, network: str) -> str:
        """
        Wrapper for this scale triple, relative to the tree file.

        The wrapper bakes in all three scales (input Div, output Mul), so a
        tree and a wrapper from different configurations must never be paired.
        Encoding the scales in the filename makes a mismatch a missing-file
        error rather than a silent wrong answer.
        """
        return (
            f"../networks/nn-nav-{network}"
            f"-p{self.position_scale}v{self.speed_scale}"
            f"h{self.heading_scale}.onnx"
        )

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
            _round_half_away(fn(index / self.heading_scale) * self.trig_scale)
            for index in range(low, high + 1)
        ]


def _round_half_away(value: float) -> int:
    """
    Round half away from zero.

    Python's round() is half-to-even, so round(2.9 * 5) = round(14.5) = 14 and
    the state 2.9 would be stored as 2.8 -- a shift of one cell toward the
    obstacle. The position update already rounds half away from zero, so the
    lattice mapping and the trig table must use the same convention or the
    model mixes two.
    """
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


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
    domain: NavDomain, region: dict[str, tuple[float, float]], *, outward: bool,
) -> str:
    """
    Prefix-notation conjunction saying the position is inside `region`.

    A real interval rarely lands on lattice points, so the direction of the
    rounding decides which way the model errs, and the two regions need
    opposite directions to stay conservative:

      obstacle (outward=True)  -- model a LARGER box, so avoiding the modelled
                                  obstacle implies avoiding the real one.
      goal     (outward=False) -- model a SMALLER box, so reaching the modelled
                                  goal implies reaching the real one.

    Rounding both with the same convention is unsound in one of the two. Half
    away from zero maps the goal [-0.5, 0.5] to lattice [-3, 3] = [-0.6, 0.6],
    which is larger than the real goal and would accept states outside it.
    """
    clauses = []
    for axis in ("x1", "x2"):
        low, high = region[axis]
        axis_scale = domain.scale_for(axis)
        scaled_low, scaled_high = low * axis_scale, high * axis_scale
        if outward:
            lattice_low = math.floor(scaled_low)
            lattice_high = math.ceil(scaled_high)
        else:
            lattice_low = math.ceil(scaled_low)
            lattice_high = math.floor(scaled_high)
        clauses.append(f"(gte, {axis}, {lattice_low})")
        clauses.append(f"(lte, {axis}, {lattice_high})")
    condition = clauses[0]
    for clause in clauses[1:]:
        condition = f"(and, {condition}, {clause})"
    return condition


def _containment_condition(lattice_bounds: dict[str, tuple[int, int]]) -> str:
    """State stays strictly inside the modelled box, so clamping never fires."""
    clauses = []
    for axis, (low, high) in lattice_bounds.items():
        clauses.append(f"(gt, {axis}, {low})")
        clauses.append(f"(lt, {axis}, {high})")
    condition = clauses[0]
    for clause in clauses[1:]:
        condition = f"(and, {condition}, {clause})"
    return condition


def build_tree(domain: NavDomain, network_key: str) -> str:
    """Render the whole .tree file as text."""
    onnx_path = domain.wrapper_path(network_key)
    position_scale = domain.position_scale
    speed_scale = domain.speed_scale
    heading_scale = domain.heading_scale
    u1_levels = domain.control_levels("x3")
    u2_levels = domain.control_levels("x4")
    x1_lo, x1_hi = domain.bound_ints("x1")
    x2_lo, x2_hi = domain.bound_ints("x2")
    x3_lo, x3_hi = domain.bound_ints("x3")
    x4_lo, x4_hi = domain.bound_ints("x4")

    cos_values = domain.trig_table(math.cos)
    sin_values = domain.trig_table(math.sin)

    # x1' = x1 + x3 * cos(x4) * dt, all on the lattice. cos is stored scaled by
    # trig_scale and dt = dt_num/dt_den, so divide by trig_scale * dt_den and
    # multiply by dt_num. Heading indexes the table from its lower bound.
    # x1' = x1 + x3 * cos(x4) * dt in real units. On separate lattices the
    # scales no longer cancel: x1_lat/Sp + (x3_lat/Sv) * cos * dt, so
    #
    #     x1_lat' = x1_lat + x3_lat * cos_int * Sp * dt_num
    #                        / (Sv * trig_scale * dt_den)
    #
    # With one shared scale the Sp/Sv factor is 1 and this reduces to the old
    # expression, which is why the coincidence was easy to miss.
    heading_index = f"(sub, x4, {x4_lo})"
    divisor = speed_scale * domain.trig_scale * domain.dt_den
    numerator_factor = position_scale * domain.dt_num
    half = divisor // 2

    def _numerator(table: str) -> str:
        return (
            f"(mult, (mult, x3, (index, {table}, {heading_index})), "
            f"{numerator_factor})"
        )

    def _position_update(axis: str, table: str, lo: int, hi: int) -> str:
        """
        Clamped position update, deterministic or interval-valued.

        DETERMINISTIC rounds half away from zero. idiv truncates toward zero,
        which biases every step against motion: with plain truncation the
        discretised loop reaches the goal from 0 of 9 initial states, versus 6
        of 9 once rounded. The bias is removed by shifting the numerator half a
        divisor in its own direction before dividing. The resulting model is a
        different closed loop, neither over- nor under-approximating the plant.

        INTERVAL assigns nondeterministically over {floor(delta), ceil(delta)},
        so the successor covers every rounding of the exact Euler increment
        rather than picking one. INVARSPEC = true then means the property holds
        for ALL such roundings, which is a real over-approximation claim -- but
        only of "explicit Euler from the reconstructed cell CENTRE, with the
        tabulated cos/sin, snapped to the lattice". It does NOT cover the cell
        around that centre (at scale 5 the cell is 0.2 wide and variation in
        x3, x4 inside it moves the increment by a few tenths of a cell), and it
        does not close the sampled-time versus continuous-time gap.

        Truncating division makes floor/ceil sign-dependent:
            num >= 0:  floor = tdiv(num, D),          ceil = tdiv(num+D-1, D)
            num <  0:  floor = tdiv(num-D+1, D),      ceil = tdiv(num, D)
        When delta is exactly integral the two coincide and the choice
        collapses, so no spurious branching is introduced.
        """
        num = _numerator(table)
        clamp = lambda step: (
            f"(max, {lo}, (min, {hi}, (add, {axis}, {step})))"
        )
        frozen = f"\n\tcase {{(not, apply_plant)}} result {{{axis}}}"

        if domain.successor == "interval":
            floor_pos = f"(idiv, {num}, {divisor})"
            ceil_pos = f"(idiv, (add, {num}, {divisor - 1}), {divisor})"
            floor_neg = f"(idiv, (sub, {num}, {divisor - 1}), {divisor})"
            ceil_neg = f"(idiv, {num}, {divisor})"
            return (
                f"{frozen}"
                f"\n\tcase {{(gte, {num}, 0)}} result "
                f"{{{clamp(floor_pos)}, {clamp(ceil_pos)}}}"
                f"\n\tresult {{{clamp(floor_neg)}, {clamp(ceil_neg)}}}"
            )

        rounded_pos = f"(idiv, (add, {num}, {half}), {divisor})"
        rounded_neg = f"(idiv, (sub, {num}, {half}), {divisor})"
        return (
            f"{frozen}"
            f"\n\tcase {{(gte, {num}, 0)}} result {{{clamp(rounded_pos)}}}"
            f"\n\tresult {{{clamp(rounded_neg)}}}"
        )

    update_x1 = _position_update("x1", "cos_table", x1_lo, x1_hi)
    update_x2 = _position_update("x2", "sin_table", x2_lo, x2_hi)

    obstacle_condition = _region_condition(
        domain, domain.obstacle, outward=True,
    )
    goal_condition = _region_condition(domain, domain.goal, outward=False)
    containment_condition = _containment_condition(
        {"x1": (x1_lo, x1_hi), "x2": (x2_lo, x2_hi),
         "x3": (x3_lo, x3_hi), "x4": (x4_lo, x4_hi)},
    )

    return f"""configuration {{
    #{{ NAV benchmark, ARCH-COMP 2025 AINNCS Section 3.11. }}#
    #{{ State on three integer lattices: position * {position_scale},
       speed * {speed_scale}, heading * {heading_scale}.
       Control deltas: u1 in +/-{u1_levels}, u2 in +/-{u2_levels}. }}#
    neural
}}
enumerations {{
}}
constants {{
    horizon := {domain.horizon_steps}
}} end_constants

variables {{
    variable {{ bl step VAR [0, horizon] assign{{result{{0}}}}}}
    #{{ The control is latched into the blackboard during the tick so the
       environment update sees one consistent pre-tick state. Reading the
       NEURAL variable directly from environment_update makes BehaVerify
       resolve the dependency through staging and evaluate the network on
       mixed stage_0 / stage_1 inputs. Grid world latches `network` into
       `current_action` for the same reason. }}#
    variable {{ bl u1 VAR [-{u1_levels}, {u1_levels}] assign{{result{{0}}}}}}
    variable {{ bl u2 VAR [-{u2_levels}, {u2_levels}] assign{{result{{0}}}}}}
    #{{ environment_update runs on every tick regardless of the tree, so the
       plant must be gated explicitly or it keeps flying past the horizon.
       Gating on the step counter does not work: `step < horizon` drops the
       last control period, and `step <= horizon` never stops because `step`
       saturates. Gate on whether THIS tick actually applied a control. }}#
    variable {{ bl apply_plant VAR BOOLEAN assign{{result{{False}}}}}}
    variable {{ env x1 VAR [{x1_lo}, {x1_hi}] assign{{result{{{domain.to_lattice(domain.initial_state['x1'], 'x1')}}}}}}}
    variable {{ env x2 VAR [{x2_lo}, {x2_hi}] assign{{result{{{domain.to_lattice(domain.initial_state['x2'], 'x2')}}}}}}}
    variable {{ env x3 VAR [{x3_lo}, {x3_hi}] assign{{result{{{domain.to_lattice(domain.initial_state['x3'], 'x3')}}}}}}}
    variable {{ env x4 VAR [{x4_lo}, {x4_hi}] assign{{result{{{domain.to_lattice(domain.initial_state['x4'], 'x4')}}}}}}}

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
	assign {{{update_x1}
	}}
    }}
    variable_statement {{
	x2
	assign {{{update_x2}
	}}
    }}
    variable_statement {{
	x3
	assign {{
	case {{(not, apply_plant)}} result {{x3}}
	result {{(max, {x3_lo}, (min, {x3_hi}, (add, x3, u1)))}}
	}}
    }}
    variable_statement {{
	x4
	assign {{
	case {{(not, apply_plant)}} result {{x4}}
	result {{(max, {x4_lo}, (min, {x4_hi}, (add, x4, u2)))}}
	}}
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
	write_variables {{apply_plant}} end_write_variables
	initial_values {{}} end_initial_values
	update {{
	    variable_statement {{
		apply_plant
		assign{{result{{False}}}}
	    }}
	    return_statement {{ result {{success}} end_result }} end_return_statement
	}} end_update
    }}
    action {{
	advance
	arguments{{}}
	local_variables {{}} end_local_variables
	read_variables {{step, control}} end_read_variables
	write_variables {{step, u1, u2, apply_plant}} end_write_variables
	initial_values {{}} end_initial_values
	update {{
	    variable_statement {{
		apply_plant
		assign{{result{{True}}}}
	    }}
	    variable_statement {{
		u1
		assign{{result{{(index, control, 0)}}}}
	    }}
	    variable_statement {{
		u2
		assign{{result{{(index, control, 1)}}}}
	    }}
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
    #{{ Reach, as the benchmark states it: in the goal AT t = 6. With the plant
       frozen at the horizon this is exactly the ARCH-COMP property.

       Note AF(goal) is NOT a strengthening of this -- after the freeze it says
       only that some sample in [0, 6] was in the goal, so a trajectory that
       clips the goal at t = 4 and leaves satisfies AF and fails this. It is
       emitted below as a separate, strictly weaker "visited the goal" check. }}#
    INVARSPEC {{
	(implies, (eq, step, horizon), {goal_condition})
    }}
    #{{ WEAKER than the invariant above: visited the goal at some sample. }}#
    CTLSPEC {{
	(always_finally, {goal_condition})
    }}
    #{{ The modelled box is an assumption, and the environment update clamps at
       its edge. Clamping would silently keep an escaping robot inside, so the
       box is only sound if the state never reaches the boundary. }}#
    INVARSPEC {{
	{containment_condition}
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
    print(
        f"  resolutions        = position {1 / domain.position_scale:.3f}, "
        f"speed {1 / domain.speed_scale:.3f}, "
        f"heading {1 / domain.heading_scale:.3f}"
    )
    print(
        f"  control levels     = u1 +/-{domain.control_levels('x3')}, "
        f"u2 +/-{domain.control_levels('x4')}"
    )
    print(f"  successor          = {domain.successor}")
    print(f"  per-axis sizes     = {sizes}")
    print(f"  TABLE ENTRIES      = {domain.table_entries():,}")


if __name__ == "__main__":
    main()
