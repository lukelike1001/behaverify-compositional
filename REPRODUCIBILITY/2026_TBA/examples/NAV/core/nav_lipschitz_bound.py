"""
nav_lipschitz_bound.py

A sound, verifier-free bound on how much the NAV controller's output can vary
across a cell of the discretized state space, and the resulting monolithic
table size.

Owns the Lipschitz estimate and the table-size arithmetic only -- not the
NSBT, the SMV encoding, or contract generation. State-space bounds come from
nav_boundary_finder.NavBoundaryFinder.

WHY THIS EXISTS
---------------
The monolithic table stores one control value per cell: the network evaluated
at the cell's centre. That is only correct if the network is constant across
the cell, which it is not. To make the table sound you must instead store the
*range* the network can take over the cell.

Getting that range tightly requires a neural-network verifier (CROWN). This
module computes the cheapest sound alternative -- a Lipschitz constant from
the product of the weight matrices' spectral norms -- so that the monolithic
pipeline can be given its best verifier-free shot before being compared
against the compositional one.

THE BOUND
---------
A function f is L-Lipschitz when  ||f(a) - f(b)|| <= L ||a - b||  for all a, b.
For a feed-forward network, each linear layer stretches by at most its spectral
norm ||W||_2, and ReLU and tanh are both 1-Lipschitz, so

    L <= prod_k ||W_k||_2

Over a cell of side h in n dimensions the L2 diameter is h*sqrt(n), so each
output component varies by at most L * h * sqrt(n). The network's output layer
is tanh, so every component already lies in [-1, 1] -- a spread of
OUTPUT_COMPONENT_RANGE = 2.0 is therefore the trivial bound, available for free
and without looking at the weights at all.

There are two thresholds, and they are different because the stored interval is
`u(centre) +/- L*d/2` *intersected with* [-1, 1]:

  L*d <  2.0   informative for every cell, wherever its centre sits.
  L*d >= 4.0   trivial for every cell: the interval swallows [-1, 1] no matter
               the centre, since |u(centre)| <= 1. Every cell stores the same
               thing, so the table collapses to a single free variable and
               costs nothing to encode.

Between the two the bound is graded -- narrow only for cells whose centre is
near tanh saturation. Measured on the NAV trajectory at h = 0.1 (L*d = 3.42),
22 of 30 cells are still fully trivial and the average width is 1.93 out of
2.0, i.e. essentially no information.

This estimate is loose by construction: it assumes every layer stretches
maximally in the same direction at once, and ignores that ReLU zeroes many
neurons. That looseness is the point -- it is what a verifier buys you.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from core.paths import EXAMPLE_ROOT

# tanh output layer: every component of u already lies in [-1, 1].
OUTPUT_COMPONENT_RANGE = 2.0

# The stored interval is u(centre) +/- L*d/2 intersected with [-1, 1]. Since
# |u(centre)| <= 1, a half-width of 2.0 swallows [-1, 1] for every possible
# centre -- so at L*d >= 4.0 every cell stores the same thing and the table
# collapses to one free variable.
COLLAPSE_SPREAD = 2.0 * OUTPUT_COMPONENT_RANGE

STATE_DIMENSION = 4

DEFAULT_NETWORKS = ("nn-nav-set.onnx", "nn-nav-point.onnx")


def _weight_matrices(onnx_path: str) -> list[np.ndarray]:
    """Weight matrices of the MatMul layers, in graph order."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(onnx_path).graph
    initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    return [
        np.asarray(initializers[name], dtype=np.float64)
        for node in graph.node
        if node.op_type == "MatMul"
        for name in node.input
        if name in initializers
    ]


@dataclass(frozen=True)
class NavLipschitzBound:
    """
    A sound upper bound on the controller's Lipschitz constant.

    `spectral_norms` are per-layer; `constant` is their product. Both are
    computed from the weights alone -- no inputs, no sampling, no verifier.
    """

    name: str
    spectral_norms: tuple[float, ...]
    constant: float

    @classmethod
    def from_onnx(cls, onnx_path: str, name: str | None = None) -> NavLipschitzBound:
        matrices = _weight_matrices(onnx_path)
        if not matrices:
            raise ValueError(f"no MatMul weights found in {onnx_path}")
        norms = tuple(float(np.linalg.norm(w, 2)) for w in matrices)
        return cls(
            name=name if name is not None else onnx_path,
            spectral_norms=norms,
            constant=float(np.prod(norms)),
        )

    # -------------------------------------------------------------- the bound

    @staticmethod
    def cell_diameter(cell_side: float, dimension: int = STATE_DIMENSION) -> float:
        """L2 diameter of an n-dimensional cube of the given side length."""
        return cell_side * math.sqrt(dimension)

    def raw_spread(self, cell_side: float) -> float:
        """L * diameter, before intersecting with the network's own output range."""
        return self.constant * self.cell_diameter(cell_side)

    def control_spread(self, cell_side: float) -> float:
        """
        Sound bound on how much one component of u varies across a cell.

        Never worse than the trivial bound, since tanh already confines every
        component to [-1, 1].
        """
        return min(OUTPUT_COMPONENT_RANGE, self.raw_spread(cell_side))

    def is_informative(self, cell_side: float) -> bool:
        """
        True when the bound constrains every cell, wherever its centre sits.

        Conservative: between this threshold and `is_trivial_everywhere` the
        bound still narrows cells whose centre is near tanh saturation, but
        only slightly.
        """
        return self.raw_spread(cell_side) < OUTPUT_COMPONENT_RANGE

    def is_trivial_everywhere(self, cell_side: float) -> bool:
        """
        True when every cell stores the whole of [-1, 1], whatever its centre.

        At and above this cell size the table has one distinct entry, so it
        collapses to a single unconstrained variable and costs nothing to
        encode -- a small, fast model that reports `false` about a robot with
        an arbitrary controller.
        """
        return self.raw_spread(cell_side) >= COLLAPSE_SPREAD

    def critical_cell_side(self, dimension: int = STATE_DIMENSION) -> float:
        """Largest cell side at which the bound constrains every cell."""
        return OUTPUT_COMPONENT_RANGE / (self.constant * math.sqrt(dimension))

    def collapse_cell_side(self, dimension: int = STATE_DIMENSION) -> float:
        """Smallest cell side at which the bound is trivial for every cell."""
        return COLLAPSE_SPREAD / (self.constant * math.sqrt(dimension))

    def distinct_entries(self, cell_side: float, cells: int) -> int:
        """
        Table rows actually needed: `cells`, or 1 once every row is identical.
        """
        return 1 if self.is_trivial_everywhere(cell_side) else cells


def table_entries(widths: Iterable[float], cell_side: float) -> int:
    """
    Number of cells the monolithic table must enumerate.

    One entry per cell of the product grid -- each costs one ONNX forward pass
    at generation time and one `case` line in the SMV file.
    """
    count = 1
    for width in widths:
        count *= max(1, int(round(width / cell_side)))
    return count


@dataclass(frozen=True)
class FeasibilityRow:
    """One row of the resolution / soundness / cost table."""

    cell_side: float
    control_spread: float
    informative: bool
    trivial_everywhere: bool
    entries: dict[str, int]
    distinct_entries: dict[str, int]


def feasibility_table(
    bound: NavLipschitzBound,
    boxes: dict[str, dict[str, float]],
    cell_sides: Iterable[float],
) -> list[FeasibilityRow]:
    """Cross the candidate resolutions against each declared state-space box."""
    return [
        FeasibilityRow(
            cell_side=h,
            control_spread=bound.control_spread(h),
            informative=bound.is_informative(h),
            trivial_everywhere=bound.is_trivial_everywhere(h),
            entries={
                label: table_entries(widths.values(), h)
                for label, widths in boxes.items()
            },
            distinct_entries={
                label: bound.distinct_entries(
                    h, table_entries(widths.values(), h)
                )
                for label, widths in boxes.items()
            },
        )
        for h in cell_sides
    ]


def main() -> None:
    """Print the Lipschitz constants and the feasibility table."""
    import argparse

    from core.nav_boundary_finder import NavBoundaryFinder, load_config

    parser = argparse.ArgumentParser(
        description="Sound verifier-free bounds and monolithic table sizes."
    )
    parser.add_argument("--networks", nargs="+", default=list(DEFAULT_NETWORKS))
    parser.add_argument(
        "--cell-sides",
        nargs="+",
        type=float,
        default=[0.5, 0.25, 0.1, 0.05, 0.025, 0.01],
    )
    parser.add_argument("--samples-per-axis", type=int, default=9)
    args = parser.parse_args()

    cfg = load_config()
    bounds = {
        name: NavLipschitzBound.from_onnx(
            str(EXAMPLE_ROOT / "networks" / name), name=name
        )
        for name in args.networks
    }

    print("LIPSCHITZ CONSTANTS (weights only -- no inputs, no sampling)")
    for name, bound in bounds.items():
        norms = ", ".join(f"{n:.3f}" for n in bound.spectral_norms)
        print(
            f"  {name:<20} ||W|| = [{norms}]  ->  L = {bound.constant:8.2f}"
            f"   informative below h = {bound.critical_cell_side():.4f}"
            f"   trivial at or above h = {bound.collapse_cell_side():.4f}"
        )

    primary = bounds[args.networks[0]]
    finder = NavBoundaryFinder(
        onnx_path=str(EXAMPLE_ROOT / "networks" / args.networks[0]), cfg=cfg
    )
    analytic = finder.calculate_boundary("analytic")
    sampled = finder.calculate_boundary(
        "sampled", samples_per_axis=args.samples_per_axis
    )
    boxes = {"sound": analytic.widths(), "sampled": sampled.widths()}

    print()
    print(f"TABLE SIZE for {args.networks[0]}")
    print("  sound box   = analytic bounds (safe to declare)")
    print("  sampled box = measured envelope (NOT sound; optimistic reference)")
    print()
    header = (
        f"{'h':>7} {'u spread':>10} {'informative':>12} "
        f"{'cells (sound)':>18} {'cells (sampled)':>18} {'rows needed':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in feasibility_table(primary, boxes, args.cell_sides):
        rows = (
            "1 (collapses)"
            if row.trivial_everywhere
            else f"{row.entries['sampled']:,}+"
        )
        print(
            f"{row.cell_side:>7} {row.control_spread:>10.2f} "
            f"{('yes' if row.informative else 'NO'):>12} "
            f"{row.entries['sound']:>18,} {row.entries['sampled']:>18,} "
            f"{rows:>14}"
        )
    print()
    print("for scale: grid world table = 2,401;  "
          "ACAS Xu largest = 456,775 (nuXmv aborted after 10 min)")


if __name__ == "__main__":
    main()
