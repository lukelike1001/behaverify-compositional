"""
Tests for the verifier-free Lipschitz bound on the NAV controllers.

The bound is what licenses a *sound* monolithic table: without it, the table's
one-value-per-cell entry is an unjustified assertion. Two things are pinned:

  1. The bound really does hold -- sampled pairs never violate it. If this
     breaks, a "sound" monolithic run is not sound.
  2. The crossover cell size (where the bound stops saying anything) is where
     the arithmetic says it is. The feasibility argument rests on that number.
"""

from __future__ import annotations

import numpy as np
import pytest

from example_imports import activate_example

EXAMPLE_ROOT = activate_example("NAV")

from core.nav_boundary_finder import NavBoundaryFinder, load_config  # noqa: E402
from core.compositional.nav_lipschitz_bound import (  # noqa: E402
    OUTPUT_COMPONENT_RANGE,
    STATE_DIMENSION,
    NavLipschitzBound,
    table_entries,
)

SET_NET = "nn-nav-set.onnx"
POINT_NET = "nn-nav-point.onnx"


def _bound(name: str) -> NavLipschitzBound:
    return NavLipschitzBound.from_onnx(
        str(EXAMPLE_ROOT / "networks" / name), name=name
    )


@pytest.fixture(scope="module")
def set_bound() -> NavLipschitzBound:
    return _bound(SET_NET)


@pytest.fixture(scope="module")
def point_bound() -> NavLipschitzBound:
    return _bound(POINT_NET)


# ----------------------------------------------------------------- the bound


def test_constant_is_product_of_spectral_norms(set_bound: NavLipschitzBound) -> None:
    assert set_bound.constant == pytest.approx(float(np.prod(set_bound.spectral_norms)))


def test_three_weight_matrices(set_bound: NavLipschitzBound) -> None:
    """4 -> 64 -> 32 -> 2 gives three MatMul layers."""
    assert len(set_bound.spectral_norms) == 3


@pytest.mark.parametrize("name", [SET_NET, POINT_NET])
def test_bound_holds_on_sampled_pairs(name: str) -> None:
    """
    The soundness check: ||u(a) - u(b)|| <= L ||a - b|| on random pairs.

    Sampling cannot prove the bound, but a violation would disprove it -- and
    would mean any monolithic run calling itself sound is not.
    """
    bound = _bound(name)
    finder = NavBoundaryFinder(onnx_path=str(EXAMPLE_ROOT / "networks" / name))
    rng = np.random.default_rng(0)

    # Spread over the region the robot actually operates in.
    low = np.array([-0.5, -0.5, -2.0, -1.0])
    span = np.array([4.0, 4.0, 2.5, 3.0])
    for _ in range(200):
        a = low + rng.random(STATE_DIMENSION) * span
        b = low + rng.random(STATE_DIMENSION) * span
        output_gap = np.linalg.norm(finder.control(a) - finder.control(b))
        input_gap = np.linalg.norm(a - b)
        assert output_gap <= bound.constant * input_gap + 1e-9


# --------------------------------------------------------------- the spread


def test_spread_never_exceeds_trivial_bound(set_bound: NavLipschitzBound) -> None:
    """tanh already confines each component to [-1, 1]; never report worse."""
    for cell_side in (10.0, 1.0, 0.25, 0.01):
        assert set_bound.control_spread(cell_side) <= OUTPUT_COMPONENT_RANGE


def test_spread_shrinks_with_the_cell(set_bound: NavLipschitzBound) -> None:
    sides = [0.05, 0.025, 0.01, 0.005]
    spreads = [set_bound.control_spread(h) for h in sides]
    assert spreads == sorted(spreads, reverse=True)


def test_crossover_is_where_the_bound_stops_saying_anything(
    set_bound: NavLipschitzBound,
) -> None:
    critical = set_bound.critical_cell_side()
    assert set_bound.is_informative(critical * 0.99)
    assert not set_bound.is_informative(critical * 1.01)
    assert set_bound.control_spread(critical * 1.01) == OUTPUT_COMPONENT_RANGE


def test_quarter_cell_is_uninformative(set_bound: NavLipschitzBound) -> None:
    """
    At any grid coarse enough to enumerate, the bound permits every control.

    This is the load-bearing claim of the feasibility argument.
    """
    assert not set_bound.is_informative(0.25)
    assert set_bound.control_spread(0.25) == OUTPUT_COMPONENT_RANGE


# ------------------------------------------------------ point vs set training


def test_set_training_gives_a_smaller_constant(
    set_bound: NavLipschitzBound, point_bound: NavLipschitzBound
) -> None:
    """
    The set-trained controller is provably smoother, from the weights alone.

    Independent of CORA, CROWN-Reach, and any simulation -- it corroborates the
    benchmark's own claim about set-based training.
    """
    assert set_bound.constant < point_bound.constant
    assert point_bound.critical_cell_side() < set_bound.critical_cell_side()


# ---------------------------------------------------------------- table size


def test_table_entries_is_the_product_grid() -> None:
    assert table_entries([1.0, 1.0], 0.1) == 100
    assert table_entries([2.0, 1.0, 0.5], 0.5) == 4 * 2 * 1


def test_degenerate_axis_counts_once() -> None:
    """A zero-width axis contributes one cell, not zero."""
    assert table_entries([1.0, 0.0], 0.1) == 10


# -------------------------------------------------- collapse vs informative


def test_two_thresholds_are_distinct(set_bound: NavLipschitzBound) -> None:
    """
    Informative and trivial-everywhere are different cell sizes.

    The stored interval is u(centre) +/- L*d/2 intersected with [-1, 1], so it
    swallows the whole output range only once L*d >= 4, not L*d >= 2.
    """
    assert set_bound.critical_cell_side() < set_bound.collapse_cell_side()
    assert set_bound.collapse_cell_side() == pytest.approx(
        2.0 * set_bound.critical_cell_side()
    )


def test_table_collapses_at_coarse_resolution(set_bound: NavLipschitzBound) -> None:
    """Above the collapse size every cell stores [-1, 1], so one row suffices."""
    assert set_bound.is_trivial_everywhere(0.25)
    assert set_bound.distinct_entries(0.25, cells=48_441_600) == 1


def test_table_does_not_collapse_when_informative(
    set_bound: NavLipschitzBound,
) -> None:
    assert not set_bound.is_trivial_everywhere(0.05)
    assert set_bound.distinct_entries(0.05, cells=5_370_624) == 5_370_624


def test_band_between_thresholds_is_neither(set_bound: NavLipschitzBound) -> None:
    """At h = 0.1 the bound neither constrains every cell nor collapses."""
    assert not set_bound.is_informative(0.1)
    assert not set_bound.is_trivial_everywhere(0.1)
