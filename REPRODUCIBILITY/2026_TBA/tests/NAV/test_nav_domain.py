"""
Tests for the NAV lattice and the integer dynamics the NSBT encodes.

Three things are pinned:

  1. Division truncates toward zero, matching nuXmv and BehaVerify's meta
     `idiv`. Python's `//` floors instead; if this reference implementation
     used it, it would silently disagree with the generated model on every
     negative value -- and `v` is negative for most of this trajectory.

  2. The discrete dynamics converge to the real closed loop as the grid is
     refined. This is what makes any verdict about the model a statement about
     the robot rather than about arithmetic.

  3. Obstacle and goal rounding are conservative in opposite directions, so
     neither makes a `true` verdict easier to obtain.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from example_imports import activate_example

EXAMPLE_ROOT = activate_example("NAV")

from core.nav_boundary_finder import NavBoundaryFinder  # noqa: E402
from core.nav_domain import (  # noqa: E402
    CONTROL_SCALE,
    NavAxis,
    NavGrid,
    build_domain,
    load_config,
    truncating_divide,
    uniform_cell_sizes,
)

SET_NETWORK = "nn-nav-set.onnx"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


# ------------------------------------------------------------------ division


@pytest.mark.parametrize("numerator,denominator,expected", [
    (-7, 2, -3),     # nuXmv truncates toward zero, not floor(-3.5) = -4
    (7, 2, 3),
    (-990, 750, -1),
    (-1, 5000, 0),
])
def test_division_truncates_toward_zero(numerator, denominator, expected) -> None:
    assert truncating_divide(numerator, denominator) == expected


def test_division_differs_from_python_floor_on_negatives() -> None:
    """The distinction this whole reference implementation depends on."""
    assert truncating_divide(-7, 2) != -7 // 2


# ---------------------------------------------------------------------- axis


def test_axis_index_value_roundtrip() -> None:
    axis = NavAxis.from_interval("x", (-1.0, 3.5), 0.25)
    for index in range(axis.lower_index, axis.upper_index + 1):
        assert axis.to_index(axis.to_value(index)) == index


def test_axis_covers_its_interval() -> None:
    axis = NavAxis.from_interval("x", (-1.0, 3.5), 0.25)
    assert axis.lower_value <= -1.0 and axis.upper_value >= 3.5


def test_axis_rejects_nonpositive_cell_size() -> None:
    with pytest.raises(ValueError, match="cell_size must be positive"):
        NavAxis.from_interval("x", (0.0, 1.0), 0.0)


def test_grid_cell_count_is_the_product(cfg: dict) -> None:
    grid, _dynamics, _ = build_domain(uniform_cell_sizes(0.5), cfg=cfg)
    expected = 1
    for axis in grid.axes:
        expected *= axis.cell_count
    assert grid.cell_count == expected


# ------------------------------------------------------------ region rounding


def test_obstacle_rounds_outward_and_goal_inward(cfg: dict) -> None:
    """
    Conservative in opposite directions: the obstacle grows, the goal shrinks.
    Both make a `true` verdict harder, which is the side to err on.
    """
    grid, _dynamics, _ = build_domain(uniform_cell_sizes(0.3), cfg=cfg)
    obstacle = grid.region_to_indices(
        {"x": (1.0, 2.0), "y": (1.0, 2.0)}, conservative="outward")
    goal = grid.region_to_indices(
        {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, conservative="inward")
    axis = grid.axis("x")
    assert axis.to_value(obstacle["x"][0]) <= 1.0
    assert axis.to_value(obstacle["x"][1]) >= 2.0
    assert axis.to_value(goal["x"][0]) >= -0.5
    assert axis.to_value(goal["x"][1]) <= 0.5


def test_region_rounding_rejects_unknown_direction(cfg: dict) -> None:
    grid, _dynamics, _ = build_domain(uniform_cell_sizes(0.5), cfg=cfg)
    with pytest.raises(ValueError, match="outward"):
        grid.region_to_indices({"x": (0.0, 1.0)}, conservative="sideways")


# -------------------------------------------------------------- the divisors


def test_divisors_match_the_derivation(cfg: dict) -> None:
    """divisor = CONTROL_SCALE * cell_size / dt, and 5000 when sizes agree."""
    cell = 0.15
    _grid, dynamics, _ = build_domain(uniform_cell_sizes(cell), cfg=cfg)
    dt = float(cfg["control_period"])
    assert dynamics.velocity_divisor == int(round(CONTROL_SCALE * cell / dt))
    assert dynamics.heading_divisor == dynamics.velocity_divisor
    assert dynamics.x_divisor == 5000     # cell size cancels when x and v agree


def test_position_divisor_tracks_the_axis_ratio(cfg: dict) -> None:
    _grid, dynamics, _ = build_domain(
        {"x": 0.3, "y": 0.3, "v": 0.15, "theta": 0.15}, cfg=cfg)
    assert dynamics.x_divisor == 10000    # 5000 * (0.3 / 0.15)


# ----------------------------------------------------------- quantization


def test_coarse_grid_is_flagged_as_frozen(cfg: dict) -> None:
    """At h = 0.5 a maximal control change truncates to zero every tick."""
    _grid, dynamics, _ = build_domain(uniform_cell_sizes(0.5), cfg=cfg)
    report = dynamics.quantization()
    assert not report.is_representable
    assert set(report.frozen_axes) == {"v", "theta"}


def test_fine_grid_is_representable(cfg: dict) -> None:
    _grid, dynamics, _ = build_domain(uniform_cell_sizes(0.1), cfg=cfg)
    assert dynamics.quantization().is_representable


def test_quantization_threshold_is_control_bound_times_period(cfg: dict) -> None:
    """The floor is |u|max * dt; 0.2 here."""
    floor = float(cfg["control_bound"]) * float(cfg["control_period"])
    _g1, just_under, _ = build_domain(uniform_cell_sizes(floor * 0.99), cfg=cfg)
    _g2, just_over, _ = build_domain(uniform_cell_sizes(floor * 1.01), cfg=cfg)
    assert just_under.quantization().is_representable
    assert not just_over.quantization().is_representable


# ------------------------------------------------------------- trig tables


def test_trig_tables_match_the_heading_axis(cfg: dict) -> None:
    grid, dynamics, _ = build_domain(uniform_cell_sizes(0.15), cfg=cfg)
    heading = grid.axis("theta")
    assert len(dynamics.cosine_table) == heading.cell_count
    for position, index in enumerate(
            range(heading.lower_index, heading.upper_index + 1)):
        expected = int(round(1000 * math.cos(heading.to_value(index))))
        assert dynamics.cosine_table[position] == expected


def test_heading_offset_makes_indices_zero_based(cfg: dict) -> None:
    grid, dynamics, _ = build_domain(uniform_cell_sizes(0.15), cfg=cfg)
    heading = grid.axis("theta")
    assert heading.lower_index + dynamics.theta_table_offset == 0


# --------------------------------------------------- fidelity to the robot


def _reference_trajectory(steps: int = 30, substeps: int = 20) -> np.ndarray:
    """RK4 on the true continuous closed loop."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(EXAMPLE_ROOT / "networks" / SET_NETWORK))
    name = session.get_inputs()[0].name

    def control(state):
        return session.run(
            None, {name: np.asarray(state, dtype=np.float32).reshape(1, 4)}
        )[0][0].astype(float)

    def derivative(state, u):
        return np.array([state[2] * math.cos(state[3]),
                         state[2] * math.sin(state[3]), u[0], u[1]])

    state = np.array([3.0, 3.0, 0.0, 0.0])
    step = 0.2 / substeps
    for _tick in range(steps):
        u = control(state)
        for _sub in range(substeps):
            k1 = derivative(state, u)
            k2 = derivative(state + step / 2 * k1, u)
            k3 = derivative(state + step / 2 * k2, u)
            k4 = derivative(state + step * k3, u)
            state = state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return state


def _discrete_final_position(cell: float, cfg: dict) -> tuple[float, float]:
    """Run the lattice dynamics with the real network in the loop."""
    import onnxruntime as ort
    bounds = NavBoundaryFinder(
        onnx_path=str(EXAMPLE_ROOT / "networks" / SET_NETWORK), cfg=cfg
    ).calculate_boundary("sampled", samples_per_axis=3)
    grid, dynamics, _ = build_domain(
        uniform_cell_sizes(cell), cfg=cfg, bounds=bounds)
    session = ort.InferenceSession(str(EXAMPLE_ROOT / "networks" / SET_NETWORK))
    name = session.get_inputs()[0].name

    def control(indices):
        physical = np.asarray(grid.to_values(indices), dtype=np.float32)
        output = session.run(None, {name: physical.reshape(1, 4)})[0][0]
        return (int(CONTROL_SCALE * output[0]), int(CONTROL_SCALE * output[1]))

    trajectory, _overflowed = dynamics.simulate(
        grid.to_indices([3.0, 3.0, 0.0, 0.0]), control, steps=30)
    final = grid.to_values(trajectory[-1])
    return final[0], final[1]


def test_discrete_dynamics_converge_to_the_real_trajectory(cfg: dict) -> None:
    """
    Refining the grid must move the discrete model toward the real robot.

    Without this the whole model is arithmetic with no claim on the benchmark.
    """
    reference = _reference_trajectory()
    errors = []
    for cell in (0.1, 0.05, 0.025):
        x, y = _discrete_final_position(cell, cfg)
        errors.append(math.hypot(x - reference[0], y - reference[1]))
    assert errors == sorted(errors, reverse=True), f"not converging: {errors}"
    assert errors[-1] < 0.2


def test_coarse_grid_leaves_the_robot_frozen(cfg: dict) -> None:
    """Below the quantization floor the model never moves -- the failure the guard exists for."""
    grid, dynamics, _ = build_domain(uniform_cell_sizes(0.5), cfg=cfg)
    start = grid.to_indices([3.0, 3.0, 0.0, 0.0])
    trajectory, _overflowed = dynamics.simulate(
        start, lambda _indices: (-CONTROL_SCALE, CONTROL_SCALE), steps=30)
    assert trajectory[-1] == start


def test_step_reports_overflow_outside_the_box(cfg: dict) -> None:
    grid, dynamics, _ = build_domain(uniform_cell_sizes(0.1), cfg=cfg)
    speed = grid.axis("v")
    at_edge = (0, 0, speed.upper_index, 0)
    _successor, overflowed = dynamics.step(at_edge, (CONTROL_SCALE, 0))
    assert overflowed


def test_step_clamps_to_the_box(cfg: dict) -> None:
    grid, dynamics, _ = build_domain(uniform_cell_sizes(0.1), cfg=cfg)
    speed = grid.axis("v")
    successor, _overflowed = dynamics.step(
        (0, 0, speed.upper_index, 0), (CONTROL_SCALE, 0))
    assert grid.contains(successor)
