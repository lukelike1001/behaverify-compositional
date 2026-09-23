"""
nav_domain.py

The discretized NAV plant: a grid over the 4-D state space, and the integer
dynamics that the generated NSBT encodes.

Owns the lattice and the physics on it. Does not own the .tree text
(nav_tree_generator), the state-space bounds (nav_boundary_finder), or contract
generation.

WHY INTEGERS IN UNITS OF THE CELL SIZE
--------------------------------------
An SMV variable's representable values *are* the lattice, so "scale factor" and
"cell size" are one knob, not two. Each state variable is therefore an integer
`i` standing for the physical value `i * cell_size`, and the grid resolution is
chosen per axis.

THE UPDATE, AND WHERE THE DIVISORS COME FROM
--------------------------------------------
Control arrives in milli-units (u_milli = 1000 * u), because the SMV table
stores integers. With control period `dt`:

  velocity     dv_phys = (u1/1000) * dt
               dv_index = dv_phys / h_v          =>  divide u1 by 1000*h_v/dt
  heading      likewise with h_th
  position     dx_phys = v_phys * cos(theta) * dt
                       = (v*h_v) * (cos_t/1000) * dt
               dx_index = dx_phys / h_x          =>  divide (v * cos_t) by
                                                     1000*h_x/(h_v*dt)

With dt = 0.2 and equal cell sizes the position divisor is exactly 5000, and
the cell size cancels -- changing resolution touches the domains and the trig
tables, not the update logic.

INTEGRATION SCHEME: EXPLICIT EULER, DELIBERATELY
------------------------------------------------
Position is updated from the *pre-update* velocity and heading. This keeps every
successor a function of stage-0 state alone, so the overflow check can be
computed before anything is clamped. Measured cost is negligible: single-step
explicit Euler lands within 0.073 of an RK4 reference over the full horizon,
while quantization error dominates at every feasible cell size
(reports/NAV/2026_09_22_nsbt_design.md, section 6.3).

CONSERVATIVE REGION ROUNDING
----------------------------
The obstacle is over-approximated (any cell touching it counts as a collision)
and the goal is under-approximated (only cells fully inside count as arrival).
Both directions make a "safe" verdict harder to obtain, which is the side to err
on. See `region_to_indices`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import yaml

from core.nav_boundary_finder import STATE_ORDER, NavBounds, NavBoundaryFinder
from core.paths import EXAMPLE_ROOT

DEFAULT_CONFIG_PATH = str(EXAMPLE_ROOT / "nav_domain_config.yaml")

# Control values are stored as integers; this is the multiplier.
CONTROL_SCALE = 1000
# cos/sin are tabulated as integers; this is their multiplier.
TRIG_SCALE = 1000


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load NAV benchmark configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def truncating_divide(numerator: int, denominator: int) -> int:
    """
    Integer division matching nuXmv's `/` and BehaVerify's meta `idiv`.

    Both truncate toward zero, so -7 / 2 is -3 rather than -4. Verified against
    nuXmv 2.1.0 directly. Using Python's `//` here would floor instead, and the
    reference dynamics would silently diverge from the generated model.
    """
    return int(numerator / denominator)


@dataclass(frozen=True)
class NavAxis:
    """One state dimension: its cell size and its inclusive index range."""

    name: str
    cell_size: float
    lower_index: int
    upper_index: int

    @classmethod
    def from_interval(
        cls, name: str, interval: tuple[float, float], cell_size: float
    ) -> NavAxis:
        if cell_size <= 0:
            raise ValueError(f"{name}: cell_size must be positive")
        return cls(
            name=name,
            cell_size=cell_size,
            lower_index=math.floor(interval[0] / cell_size),
            upper_index=math.ceil(interval[1] / cell_size),
        )

    @property
    def cell_count(self) -> int:
        return self.upper_index - self.lower_index + 1

    @property
    def lower_value(self) -> float:
        return self.lower_index * self.cell_size

    @property
    def upper_value(self) -> float:
        return self.upper_index * self.cell_size

    def to_index(self, value: float) -> int:
        return int(round(value / self.cell_size))

    def to_value(self, index: int) -> float:
        return index * self.cell_size

    def contains(self, index: int) -> bool:
        return self.lower_index <= index <= self.upper_index

    def clamp(self, index: int) -> int:
        return min(self.upper_index, max(self.lower_index, index))


@dataclass(frozen=True)
class NavGrid:
    """The 4-D lattice: one NavAxis per state dimension, in STATE_ORDER."""

    axes: tuple[NavAxis, NavAxis, NavAxis, NavAxis]

    @classmethod
    def from_bounds(
        cls, bounds: NavBounds, cell_sizes: dict[str, float]
    ) -> NavGrid:
        intervals = {
            "x": bounds.x, "y": bounds.y, "v": bounds.v, "theta": bounds.theta,
        }
        return cls(axes=tuple(  # type: ignore[arg-type]
            NavAxis.from_interval(name, intervals[name], cell_sizes[name])
            for name in STATE_ORDER
        ))

    def axis(self, name: str) -> NavAxis:
        for candidate in self.axes:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    @property
    def cell_count(self) -> int:
        """Table entries the monolithic pipeline must enumerate."""
        count = 1
        for axis in self.axes:
            count *= axis.cell_count
        return count

    def to_indices(self, state: Sequence[float]) -> tuple[int, int, int, int]:
        return tuple(  # type: ignore[return-value]
            axis.to_index(value) for axis, value in zip(self.axes, state)
        )

    def to_values(self, indices: Sequence[int]) -> tuple[float, ...]:
        return tuple(
            axis.to_value(index) for axis, index in zip(self.axes, indices)
        )

    def contains(self, indices: Sequence[int]) -> bool:
        return all(
            axis.contains(index) for axis, index in zip(self.axes, indices)
        )

    def region_to_indices(
        self, region: dict[str, Sequence[float]], *, conservative: str
    ) -> dict[str, tuple[int, int]]:
        """
        Convert a physical x/y region to inclusive index bounds.

        conservative="outward" grows the region to every cell it touches (used
        for the obstacle: more states count as collisions).
        conservative="inward" keeps only cells strictly inside (used for the
        goal: fewer states count as arrival).

        Both directions make a `true` verdict harder to obtain.
        """
        if conservative not in {"outward", "inward"}:
            raise ValueError("conservative must be 'outward' or 'inward'")
        rounded = {}
        for name, (low, high) in region.items():
            axis = self.axis(name)
            if conservative == "outward":
                lower = math.floor(low / axis.cell_size)
                upper = math.ceil(high / axis.cell_size)
            else:
                lower = math.ceil(low / axis.cell_size)
                upper = math.floor(high / axis.cell_size)
            rounded[name] = (lower, upper)
        return rounded


@dataclass(frozen=True)
class QuantizationReport:
    """How many cells a maximal one-step change moves, per axis."""

    cells_per_step: dict[str, float]

    @property
    def frozen_axes(self) -> list[str]:
        """Axes where a maximal step cannot move a single cell."""
        return [name for name, n in self.cells_per_step.items() if n < 1.0]

    @property
    def is_representable(self) -> bool:
        return not self.frozen_axes


class NavDiscreteDynamics:
    """
    The integer physics the generated tree encodes.

    This is the reference implementation: the tree and this class must agree,
    and a test pins that by construction rather than by inspection.
    """

    def __init__(
        self,
        grid: NavGrid,
        control_period: float,
        control_bound: float = 1.0,
    ) -> None:
        self.grid = grid
        self.control_period = control_period
        self.control_bound = control_bound

    # ------------------------------------------------------------- divisors

    def _rate_divisor(self, axis_name: str) -> int:
        """Divisor turning a milli-unit control into a step on a rate axis."""
        axis = self.grid.axis(axis_name)
        return int(round(CONTROL_SCALE * axis.cell_size / self.control_period))

    def _position_divisor(self, axis_name: str) -> int:
        """Divisor turning `v * trig` into a step on a position axis."""
        position = self.grid.axis(axis_name)
        speed = self.grid.axis("v")
        return int(round(
            TRIG_SCALE * position.cell_size
            / (speed.cell_size * self.control_period)
        ))

    @property
    def velocity_divisor(self) -> int:
        return self._rate_divisor("v")

    @property
    def heading_divisor(self) -> int:
        return self._rate_divisor("theta")

    @property
    def x_divisor(self) -> int:
        return self._position_divisor("x")

    @property
    def y_divisor(self) -> int:
        return self._position_divisor("y")

    # ---------------------------------------------------------- trig tables

    def _trig_table(self, function: Any) -> list[int]:
        theta = self.grid.axis("theta")
        return [
            int(round(TRIG_SCALE * function(theta.to_value(index))))
            for index in range(theta.lower_index, theta.upper_index + 1)
        ]

    @property
    def cosine_table(self) -> list[int]:
        return self._trig_table(math.cos)

    @property
    def sine_table(self) -> list[int]:
        return self._trig_table(math.sin)

    @property
    def theta_table_offset(self) -> int:
        """Added to a theta index to get a 0-based array index."""
        return -self.grid.axis("theta").lower_index

    # --------------------------------------------------------------- physics

    def raw_successor(
        self,
        indices: Sequence[int],
        control_milli: Sequence[int],
    ) -> tuple[int, int, int, int]:
        """
        Unclamped next state. Explicit Euler: position uses the old v and theta.
        """
        x, y, v, theta = indices
        u1, u2 = control_milli
        table_index = theta + self.theta_table_offset
        return (
            x + truncating_divide(v * self.cosine_table[table_index], self.x_divisor),
            y + truncating_divide(v * self.sine_table[table_index], self.y_divisor),
            v + truncating_divide(u1, self.velocity_divisor),
            theta + truncating_divide(u2, self.heading_divisor),
        )

    def step(
        self,
        indices: Sequence[int],
        control_milli: Sequence[int],
    ) -> tuple[tuple[int, int, int, int], bool]:
        """One tick: returns the clamped successor and whether it overflowed."""
        raw = self.raw_successor(indices, control_milli)
        overflowed = not self.grid.contains(raw)
        clamped = tuple(  # type: ignore[assignment]
            axis.clamp(index) for axis, index in zip(self.grid.axes, raw)
        )
        return clamped, overflowed

    # ------------------------------------------------------------ diagnostics

    def quantization(self) -> QuantizationReport:
        """
        Cells moved by a maximal one-step change, per axis.

        Below 1.0 the change truncates to zero every tick and that axis is
        frozen -- the model cannot represent the dynamics at all, and any
        verdict it produces is about a robot that does not move.
        """
        speed = self.grid.axis("v")
        max_speed = max(abs(speed.lower_value), abs(speed.upper_value))
        return QuantizationReport(cells_per_step={
            "x": max_speed * self.control_period / self.grid.axis("x").cell_size,
            "y": max_speed * self.control_period / self.grid.axis("y").cell_size,
            "v": self.control_bound * self.control_period / speed.cell_size,
            "theta": self.control_bound * self.control_period
                     / self.grid.axis("theta").cell_size,
        })

    def simulate(
        self,
        initial_indices: Sequence[int],
        control: Any,
        steps: int,
    ) -> tuple[list[tuple[int, ...]], bool]:
        """
        Roll the lattice dynamics forward, calling `control(indices) -> (u1, u2)`.

        Returns the index trajectory and whether the box was ever breached.
        """
        state = tuple(initial_indices)
        trajectory = [state]
        overflowed = False
        for _tick in range(steps):
            state, breached = self.step(state, control(state))
            overflowed = overflowed or breached
            trajectory.append(state)
        return trajectory, overflowed


def build_domain(
    cell_sizes: dict[str, float],
    cfg: dict[str, Any] | None = None,
    bounds: NavBounds | None = None,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> tuple[NavGrid, NavDiscreteDynamics, dict[str, Any]]:
    """
    Assemble the grid and dynamics from configuration.

    `bounds` defaults to the sound analytic box, which requires no network.
    """
    cfg = cfg if cfg is not None else load_config(config_path)
    if bounds is None:
        bounds = NavBoundaryFinder(cfg=cfg).calculate_boundary("analytic")
    grid = NavGrid.from_bounds(bounds, cell_sizes)
    dynamics = NavDiscreteDynamics(
        grid,
        control_period=float(cfg["control_period"]),
        control_bound=float(cfg["control_bound"]),
    )
    return grid, dynamics, cfg


def uniform_cell_sizes(cell_size: float) -> dict[str, float]:
    """The same resolution on every axis -- the simple default."""
    return {name: cell_size for name in STATE_ORDER}
