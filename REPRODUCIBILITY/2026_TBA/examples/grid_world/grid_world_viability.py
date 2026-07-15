"""
grid_world_viability.py

Viability kernel for the 1-NN grid-world NSBT: the largest set of safe cells
from which SOME action sequence stays safe forever (greatest fixpoint).

This module is the single source of truth for grid-world physics, the safety
partition, and A/G contract generation. Computed from domain config only --
no networks.

Physics (matches counter_template.tree environment_update):
  - State: (x, y) in [grid_min, grid_max]^2
  - Actions: We / Ea / No / So / XX (stay)
  - Cardinal moves clamp at grid borders
  - Unsafe = cell is an obstacle

Hover theorem: for every safe cell s, XX maps s -> s, so the greatest fixpoint
never deletes anything. V = Safe, safe_but_doomed = empty, and ∂V is exactly
the obstacle-adjacent cells. Kernel-boundary contracts are therefore the
one-step crash-avoidance contracts used by the compositional pipeline.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import yaml

# Direction index → (label, dx, dy). Matches NN class order in the DSL:
# We=0, Ea=1, No=2, So=3, XX=4 (stay is not a contract antecedent).
CARDINAL_DIRECTIONS: dict[int, tuple[str, int, int]] = {
    0: ("We", -1, 0),
    1: ("Ea", +1, 0),
    2: ("No", 0, +1),
    3: ("So", 0, -1),
}

ACTIONS: dict[str, tuple[int, int]] = {
    label: (dx, dy) for _idx, (label, dx, dy) in CARDINAL_DIRECTIONS.items()
}
ACTIONS["XX"] = (0, 0)

DIR_IDX: dict[str, int] = {
    label: idx for idx, (label, _dx, _dy) in CARDINAL_DIRECTIONS.items()
}


def load_config(path: str = "grid_world_domain_config.yaml") -> dict[str, Any]:
    """Load grid-world configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class GridWorldContract:
    """One A/G safety contract: at `source`, the NN must not output `forbidden_dir`."""

    source: tuple[int, int]
    forbidden_dir: str
    forbidden_dir_idx: int
    obstacle: tuple[int, int]

    @property
    def description(self) -> str:
        ox, oy = self.obstacle
        cx, cy = self.source
        return f"obstacle ({ox},{oy})  source ({cx},{cy})  forbid {self.forbidden_dir}"

    def identity(self) -> tuple[int, int, str, int, int]:
        """Canonical key for set comparison (source, forbid label, obstacle)."""
        return (*self.source, self.forbidden_dir, *self.obstacle)

    def to_spec_dict(self, contract_id: int | None = None) -> dict[str, Any]:
        """JSON-serializable fields shared with CROWN result records."""
        rec: dict[str, Any] = {
            "obstacle": list(self.obstacle),
            "source": list(self.source),
            "forbidden_dir": self.forbidden_dir,
            "forbidden_dir_idx": self.forbidden_dir_idx,
            "description": self.description,
        }
        if contract_id is not None:
            rec["id"] = contract_id
        return rec


@dataclass(frozen=True)
class GridWorldDomain:
    """Static map: bounds + obstacle set. Owns physics, not the viability partition."""

    grid_min: int
    grid_max: int
    obstacles: frozenset[tuple[int, int]]

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any] | None = None,
        config_path: str = "grid_world_domain_config.yaml",
    ) -> GridWorldDomain:
        if cfg is None:
            cfg = load_config(config_path)
        return cls(
            grid_min=int(cfg["grid"]["min"]),
            grid_max=int(cfg["grid"]["max"]),
            obstacles=frozenset(tuple(o) for o in cfg["obstacles"]),
        )

    @property
    def side_length(self) -> int:
        return self.grid_max - self.grid_min + 1

    @property
    def all_cells(self) -> list[tuple[int, int]]:
        return list(itertools.product(
            range(self.grid_min, self.grid_max + 1),
            range(self.grid_min, self.grid_max + 1),
        ))

    def in_bounds(self, x: int, y: int) -> bool:
        return self.grid_min <= x <= self.grid_max and self.grid_min <= y <= self.grid_max

    def is_safe(self, s: tuple[int, int]) -> bool:
        return s not in self.obstacles

    def simulate_step(self, x: int, y: int, action: str) -> tuple[int, int]:
        """One tick of environment_update for drone position under `action`."""
        dx, dy = ACTIONS[action]
        nx = min(self.grid_max, max(self.grid_min, x + dx))
        ny = min(self.grid_max, max(self.grid_min, y + dy))
        return (nx, ny)

    def build_successor_table(
        self,
    ) -> dict[tuple[int, int], dict[str, tuple[int, int]]]:
        """Precompute step(s, a) for every cell and action."""
        return {
            s: {a: self.simulate_step(s[0], s[1], a) for a in ACTIONS}
            for s in self.all_cells
        }

    def obstacle_adjacent_cells(self) -> frozenset[tuple[int, int]]:
        """
        Safe cells from which some cardinal action lands on an obstacle.

        Pure geometry (independent of the viability fixpoint). Under the hover
        theorem this equals ∂V; the inductive proof cross-checks that equality.
        """
        adjacent: set[tuple[int, int]] = set()
        for (ox, oy) in self.obstacles:
            for label, (dx, dy) in ACTIONS.items():
                if label == "XX":
                    continue
                sx, sy = ox - dx, oy - dy
                if not self.in_bounds(sx, sy) or not self.is_safe((sx, sy)):
                    continue
                if self.simulate_step(sx, sy, label) == (ox, oy):
                    adjacent.add((sx, sy))
        return frozenset(adjacent)


@dataclass(frozen=True)
class GridWorldViabilityKernel:
    """
    Safety/viability partition of the grid.

    Single responsibility: classify every cell as unsafe, safe-but-doomed,
    interior(V), or boundary(V), and expose Allowed_V(s) for s in V.
    Contract emission is a view of ∂V, not a second generator.
    """

    domain: GridWorldDomain
    V: frozenset[tuple[int, int]]
    interior: frozenset[tuple[int, int]]
    boundary: frozenset[tuple[int, int]]
    safe_but_doomed: frozenset[tuple[int, int]]
    unsafe: frozenset[tuple[int, int]]
    allowed: dict[tuple[int, int], list[str]]
    fixpoint_rounds: int

    @classmethod
    def compute(cls, domain: GridWorldDomain | None = None) -> GridWorldViabilityKernel:
        """Greatest fixpoint of V = {s safe : exists a with step(s,a) in V}."""
        if domain is None:
            domain = GridWorldDomain.from_config()

        succ = domain.build_successor_table()
        all_cells = domain.all_cells

        safe = frozenset(s for s in all_cells if domain.is_safe(s))
        unsafe = frozenset(all_cells) - safe

        kernel: set[tuple[int, int]] = set(safe)
        rounds = 0
        while True:
            doomed = {
                s for s in kernel
                if not any(succ[s][a] in kernel for a in ACTIONS)
            }
            if not doomed:
                break
            kernel -= doomed
            rounds += 1
        V = frozenset(kernel)

        allowed = {s: [a for a in ACTIONS if succ[s][a] in V] for s in V}
        interior = frozenset(
            s for s, acts in allowed.items() if len(acts) == len(ACTIONS)
        )
        boundary = V - interior
        safe_but_doomed = safe - V

        return cls(
            domain=domain,
            V=V,
            interior=interior,
            boundary=boundary,
            safe_but_doomed=safe_but_doomed,
            unsafe=unsafe,
            allowed=allowed,
            fixpoint_rounds=rounds,
        )

    def contracts_from_boundary(self) -> list[GridWorldContract]:
        """
        A/G contracts required by inductive obligation (ii) on ∂V:

            Assume drone at source s ∈ ∂V
            Guarantee NN ≠ a  for each a ∉ Allowed_V(s)

        Stay (XX) is never emitted: it always stays in V for s in V.
        """
        contracts: list[GridWorldContract] = []
        for source in sorted(self.boundary):
            allowed_here = set(self.allowed[source])
            for label in ACTIONS:
                if label == "XX" or label in allowed_here:
                    continue
                landing = self.domain.simulate_step(source[0], source[1], label)
                if landing not in self.domain.obstacles:
                    raise AssertionError(
                        f"forbidden action {label} at {source} landed on "
                        f"{landing}, expected an obstacle"
                    )
                contracts.append(GridWorldContract(
                    source=source,
                    forbidden_dir=label,
                    forbidden_dir_idx=DIR_IDX[label],
                    obstacle=landing,
                ))
        return contracts


def generate_contracts(
    obstacles: list[tuple[int, int]] | None = None,
    grid_min: int | None = None,
    grid_max: int | None = None,
    config_path: str = "grid_world_domain_config.yaml",
) -> list[GridWorldContract]:
    """
    Public entry point for the compositional pipeline.

    Builds the domain (from explicit args or config), computes V, and returns
    kernel-boundary contracts. Prefer this over hand-rolled obstacle walks.
    """
    if obstacles is None or grid_min is None or grid_max is None:
        domain = GridWorldDomain.from_config(config_path=config_path)
    else:
        domain = GridWorldDomain(
            grid_min=grid_min,
            grid_max=grid_max,
            obstacles=frozenset(tuple(o) for o in obstacles),
        )
    return GridWorldViabilityKernel.compute(domain).contracts_from_boundary()


if __name__ == "__main__":
    domain = GridWorldDomain.from_config()
    kernel = GridWorldViabilityKernel.compute(domain)
    n = domain.side_length ** 2
    print(
        f"grid = [{domain.grid_min},{domain.grid_max}]^2  ({n} cells, "
        f"{len(domain.obstacles)} obstacles)"
    )
    print(
        f"|V| = {len(kernel.V)}  interior = {len(kernel.interior)}  "
        f"boundary = {len(kernel.boundary)}  "
        f"safe-but-doomed = {len(kernel.safe_but_doomed)}  "
        f"unsafe = {len(kernel.unsafe)}"
    )
    print(f"fixpoint rounds = {kernel.fixpoint_rounds}")
    derived = kernel.contracts_from_boundary()
    print(f"contracts from ∂V = {len(derived)}")
    for i, c in enumerate(derived, start=1):
        print(f"{i:<4} {c.description}")
