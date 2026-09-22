"""
Tests for NAV state-space bounds.

These pin the two claims the monolithic pipeline rests on:

  1. The analytic bounds are sound -- every measured trajectory stays inside
     them. If this breaks, the declared SMV domains no longer cover the
     reachable set and any verdict from nuXmv is meaningless.

  2. The state order is (x, y, v, theta), per dynamics.m and against the
     ARCH-COMP prose. If this breaks, the network is being fed a permuted
     state and the whole model is wrong in a way no specification would catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from example_imports import activate_example

EXAMPLE_ROOT = activate_example("NAV")

from core.nav_boundary_finder import (  # noqa: E402
    STATE_ORDER,
    NavBoundaryFinder,
    NavBounds,
    load_config,
)

NETWORKS = ("nn-nav-set.onnx", "nn-nav-point.onnx")

# Coarse enough to keep the suite fast; soundness does not depend on density.
SAMPLES_PER_AXIS = 5
SUBSTEPS = 10


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def analytic(cfg: dict) -> NavBounds:
    return NavBoundaryFinder(cfg=cfg).calculate_boundary("analytic")


def _finder(name: str, cfg: dict) -> NavBoundaryFinder:
    return NavBoundaryFinder(
        onnx_path=str(EXAMPLE_ROOT / "networks" / name), cfg=cfg
    )


def _sampled(name: str, cfg: dict) -> NavBounds:
    return _finder(name, cfg).calculate_boundary(
        "sampled",
        samples_per_axis=SAMPLES_PER_AXIS,
        substeps_per_control_period=SUBSTEPS,
    )


# ------------------------------------------------------------------ analytic


def test_analytic_needs_no_network(cfg: dict) -> None:
    """Sound bounds come from |u| <= U alone, so no ONNX file is required."""
    bounds = NavBoundaryFinder(onnx_path=None, cfg=cfg).calculate_boundary()
    assert bounds.method == "analytic"
    assert bounds.is_sound


def test_analytic_matches_closed_form(cfg: dict, analytic: NavBounds) -> None:
    """v, theta widen by U*T; position widens by |v0|max*T + U*T^2/2."""
    u_max = float(cfg["control_bound"])
    horizon = float(cfg["horizon"])
    initial = cfg["initial_set"]

    rate_growth = u_max * horizon
    speed_max = max(abs(float(initial["v"][0])), abs(float(initial["v"][1])))
    position_growth = speed_max * horizon + 0.5 * u_max * horizon**2

    assert analytic.v == pytest.approx(
        (initial["v"][0] - rate_growth, initial["v"][1] + rate_growth)
    )
    assert analytic.theta == pytest.approx(
        (initial["theta"][0] - rate_growth, initial["theta"][1] + rate_growth)
    )
    assert analytic.x == pytest.approx(
        (initial["x"][0] - position_growth, initial["x"][1] + position_growth)
    )
    assert analytic.y == pytest.approx(
        (initial["y"][0] - position_growth, initial["y"][1] + position_growth)
    )


def test_analytic_contains_initial_set(cfg: dict, analytic: NavBounds) -> None:
    initial = cfg["initial_set"]
    for name, (lo, hi) in zip(
        STATE_ORDER, (analytic.x, analytic.y, analytic.v, analytic.theta)
    ):
        assert lo <= initial[name][0] and initial[name][1] <= hi


def test_thirty_control_steps(cfg: dict) -> None:
    assert NavBoundaryFinder(cfg=cfg).num_control_steps == 30


def test_horizon_must_divide_into_control_periods(cfg: dict) -> None:
    bad = dict(cfg, horizon=6.05)
    with pytest.raises(ValueError, match="whole number of control periods"):
        _ = NavBoundaryFinder(cfg=bad).num_control_steps


# ------------------------------------------------------------------- sampled


@pytest.mark.parametrize("name", NETWORKS)
def test_sampled_is_inside_analytic(
    name: str, cfg: dict, analytic: NavBounds
) -> None:
    """The soundness check: no measured trajectory escapes the declared box."""
    assert analytic.contains(_sampled(name, cfg))


@pytest.mark.parametrize("name", NETWORKS)
def test_sampled_is_flagged_unsound(name: str, cfg: dict) -> None:
    """Guards against a sampled envelope being declared as an SMV domain."""
    assert not _sampled(name, cfg).is_sound


def test_sampled_is_reproducible(cfg: dict) -> None:
    """Same inputs, same envelope -- no hidden randomness in the rollout."""
    first = _sampled("nn-nav-set.onnx", cfg)
    second = _sampled("nn-nav-set.onnx", cfg)
    assert first == second


def test_sampled_requires_a_network(cfg: dict) -> None:
    with pytest.raises(ValueError, match="no ONNX path"):
        NavBoundaryFinder(cfg=cfg).calculate_boundary("sampled")


def test_unknown_method_rejected(cfg: dict) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        NavBoundaryFinder(cfg=cfg).calculate_boundary("guess")


# --------------------------------------------------------------- state order


def test_documented_state_order_reaches_the_goal(cfg: dict) -> None:
    """
    Under (x, y, v, theta) the robust controller lands in the goal region.

    This is the empirical basis for preferring dynamics.m over the ARCH-COMP
    prose ordering, which the next test shows does not converge.
    """
    finder = _finder("nn-nav-set.onnx", cfg)
    initial = cfg["initial_set"]
    start = np.array(
        [initial[name][0] for name in STATE_ORDER], dtype=np.float64
    )
    start[0] = start[1] = 3.0  # centre of the initial set

    final = finder.simulate(start, substeps_per_control_period=SUBSTEPS)[-1]
    goal = cfg["goal"]
    assert goal["x"][0] <= final[0] <= goal["x"][1]
    assert goal["y"][0] <= final[1] <= goal["y"][1]


def test_prose_state_order_does_not_converge(cfg: dict) -> None:
    """
    Swapping v and theta -- the ARCH-COMP prose ordering -- diverges.

    Keeps the ordering decision falsifiable rather than a comment.
    """
    finder = _finder("nn-nav-set.onnx", cfg)

    def swapped_derivative(state: np.ndarray, u: np.ndarray) -> np.ndarray:
        # state read as (x, y, theta, v): position driven by state[3].
        return np.array(
            [
                state[3] * np.cos(state[2]),
                state[3] * np.sin(state[2]),
                u[0],
                u[1],
            ]
        )

    state = np.array([3.0, 3.0, 0.0, 0.0])
    h = finder.control_period / SUBSTEPS
    for _ in range(finder.num_control_steps):
        u = finder.control(state)
        for _sub in range(SUBSTEPS):
            k1 = swapped_derivative(state, u)
            k2 = swapped_derivative(state + 0.5 * h * k1, u)
            k3 = swapped_derivative(state + 0.5 * h * k2, u)
            k4 = swapped_derivative(state + h * k3, u)
            state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    goal = cfg["goal"]
    in_goal = (
        goal["x"][0] <= state[0] <= goal["x"][1]
        and goal["y"][0] <= state[1] <= goal["y"][1]
    )
    assert not in_goal
