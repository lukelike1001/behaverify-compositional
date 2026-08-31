"""
grid_world_domain.py

Static grid-world plant: bounds and obstacle set from
grid_world_domain_config.yaml, plus the executable physics that mirrors
counter_template.tree environment_update.

Owns the action tables, config loading, and one-step dynamics only -- not the
viability partition, contract generation, or CROWN settings (those are
separate layers under core/safety/ and core/liveness/).

Physics:
  - State: (x, y) in [grid_min, grid_max]^2
  - Actions: We / Ea / No / So / XX (stay)
  - Cardinal moves clamp at grid borders
  - Unsafe = cell is an obstacle
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import yaml

from core.paths import EXAMPLE_ROOT

DEFAULT_CONFIG_PATH = str(EXAMPLE_ROOT / "grid_world_domain_config.yaml")

# Direction index → (label, dx, dy). Matches NN class order in the DSL:
# We=0, Ea=1, No=2, So=3, XX=4 (stay is not a safety-contract antecedent).
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


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load grid-world configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        config_path: str = DEFAULT_CONFIG_PATH,
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
