"""
hull_split.py

Experiment 3 of the Q1 contract gate, in its cheapest and most decisive form:
how finely must the initial set be split before an AXIS-ALIGNED BOX enclosure of
the reachable set clears the obstacle at all?

This deliberately removes CROWN and interval arithmetic. Each piece of Init is
propagated as a cloud of exact trajectories, and the piece's enclosure is the
bounding box of that cloud. That box is a *subset* of any sound over-
approximation of the piece, so the clearance reported here is an UPPER BOUND on
what any box-based method can achieve -- CROWN slack, wrapping, and the series
remainder can only make it worse.

So a negative clearance here is a genuine kill: at that split depth, no
box-based contract scheme can prove Q1, however tight its bounds. A positive
clearance is not a proof, only permission to try.

    python3 -m scripts.hull_split --pieces 1 2 4 8 16 32
"""

from __future__ import annotations

import argparse

import numpy as np

from core.nav_tree_generator import NavDomain
from scripts.init_clearance import _control, _rk4, _session

SUBSTEPS = 50  # per control period; per-period motion is ~0.076, so ~0.0015 apart
SAMPLES_PER_PIECE = 5  # per axis, so 25 trajectories per piece


def _box_clearance(
    lower: np.ndarray, upper: np.ndarray, obstacle_lo: float, obstacle_hi: float,
) -> np.ndarray:
    """
    Distance from each axis-aligned box to the obstacle square, 0 if they touch.

    lower, upper are (n, 2). Box-to-box distance is the norm of the per-axis
    separations, each clamped at 0.
    """
    gap = np.maximum(
        np.maximum(obstacle_lo - upper, lower - obstacle_hi), 0.0,
    )
    return np.linalg.norm(gap, axis=1)


def _seed(pieces: int) -> np.ndarray:
    """Sample grid for each of pieces x pieces sub-boxes of Init, piece-major."""
    edges = np.linspace(2.9, 3.1, pieces + 1)
    blocks = []
    for i in range(pieces):
        for j in range(pieces):
            axis_x = np.linspace(edges[i], edges[i + 1], SAMPLES_PER_PIECE)
            axis_y = np.linspace(edges[j], edges[j + 1], SAMPLES_PER_PIECE)
            mesh_x, mesh_y = np.meshgrid(axis_x, axis_y, indexing="ij")
            blocks.append(np.column_stack([mesh_x.ravel(), mesh_y.ravel()]))
    positions = np.concatenate(blocks)
    return np.column_stack([positions, np.zeros((len(positions), 2))])


def run(network: str, domain: NavDomain, pieces: int) -> dict:
    session, input_name = _session(network)
    states = _seed(pieces).astype(np.float64)

    n_pieces = pieces * pieces
    per_piece = SAMPLES_PER_PIECE * SAMPLES_PER_PIECE
    obstacle_lo, obstacle_hi = domain.obstacle["x1"][0], domain.obstacle["x1"][1]
    h = domain.dt / SUBSTEPS

    worst = np.inf
    worst_step = -1
    for step in range(domain.horizon_steps):
        controls = _control(session, input_name, states).astype(np.float64)
        for _ in range(SUBSTEPS):
            states = _rk4(states, controls, h)
            grouped = states[:, :2].reshape(n_pieces, per_piece, 2)
            clearance = _box_clearance(
                grouped.min(axis=1), grouped.max(axis=1),
                obstacle_lo, obstacle_hi,
            )
            here = float(clearance.min())
            if here < worst:
                worst, worst_step = here, step

    return {"pieces": n_pieces, "clearance": worst, "step": worst_step}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pieces", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--networks", nargs="+", default=["point", "set"],
    )
    args = parser.parse_args()

    domain = NavDomain.from_yaml()
    print("Best-case box enclosure of the reachable set (exact cloud hull).")
    print("Positive = a box scheme is not yet ruled out. Negative/zero = dead.\n")

    for network in args.networks:
        print(f"{network}:")
        print("  split   boxes   hull clearance   worst period")
        for pieces in args.pieces:
            result = run(network, domain, pieces)
            print(
                f"  {pieces:2d}x{pieces:<2d}  {result['pieces']:6d}"
                f"   {result['clearance']:14.6f}   {result['step']:4d}",
            )
        print()


if __name__ == "__main__":
    main()
