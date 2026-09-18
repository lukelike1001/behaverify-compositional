"""
increment_grid_budget.py

How many LRA increment cells does a *moving* NAV closed loop actually occupy?

D8 used 72 cells because (x3, x4) stayed in a tiny tube under a timid hand-picked
U. A uniform 0.05 grid of the whole modelled Box is 56 x 96 = 5,376 -- the scare
number. This script measures the demand-driven count: unique (x3, x4) cells that
the *real* net visits from Init.

Also sizes heading bins as ~eps / |x3| (fine heading only when speed is large),
which stays LRA (rectangles in the (x3, x4) plane) unlike indexing cells by the
product x3*cos(x4).

    python3 -m scripts.increment_grid_budget --grid 51
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from core.nav_tree_generator import NavDomain
from scripts.init_clearance import NETWORKS, _control, _rk4, _session

SUBSTEPS = 50  # enough to track (x3, x4); not a tube-clearance run


def _cloud(network: str, domain: NavDomain, grid: int) -> np.ndarray:
    session, input_name = _session(network)
    axis = np.linspace(2.9, 3.1, grid)
    mesh_x, mesh_y = np.meshgrid(axis, axis, indexing="ij")
    states = np.column_stack(
        [
            mesh_x.ravel(),
            mesh_y.ravel(),
            np.zeros(grid * grid),
            np.zeros(grid * grid),
        ],
    ).astype(np.float64)
    h = domain.dt / SUBSTEPS
    snapshots = [states[:, 2:4].copy()]
    for _ in range(domain.horizon_steps):
        controls = _control(session, input_name, states).astype(np.float64)
        for _ in range(SUBSTEPS):
            states = _rk4(states, controls, h)
        snapshots.append(states[:, 2:4].copy())
    return np.concatenate(snapshots, axis=0)


def _full_grid_count(x3_span: float, x4_span: float, step: float) -> int:
    n3 = max(1, math.ceil(x3_span / step - 1e-12))
    n4 = max(1, math.ceil(x4_span / step - 1e-12))
    return n3 * n4


def _occupied(pairs: np.ndarray, x3_lo: float, x4_lo: float, step: float) -> int:
    i3 = np.floor((pairs[:, 0] - x3_lo) / step).astype(int)
    i4 = np.floor((pairs[:, 1] - x4_lo) / step).astype(int)
    return len({(int(a), int(b)) for a, b in zip(i3, i4)})


def _adaptive_occupied(pairs: np.ndarray, x3_lo: float, dx3: float, eps: float) -> int:
    """Uniform x3 bins; heading bin width ~ eps / max(|x3|, dx3) in that bin."""
    i3 = np.floor((pairs[:, 0] - x3_lo) / dx3).astype(int)
    cells: set[tuple[int, int]] = set()
    for bin_id in np.unique(i3):
        sl = pairs[i3 == bin_id]
        x3_ref = max(float(np.max(np.abs(sl[:, 0]))), dx3)
        dx4 = eps / x3_ref
        i4 = np.floor(sl[:, 1] / dx4).astype(int)
        for b in np.unique(i4):
            cells.add((int(bin_id), int(b)))
    return len(cells)


def report(network: str, domain: NavDomain, pairs: np.ndarray) -> None:
    b3 = domain.bounds["x3"]
    b4 = domain.bounds["x4"]
    box_span3, box_span4 = b3[1] - b3[0], b4[1] - b4[0]
    x3_min, x3_max = float(pairs[:, 0].min()), float(pairs[:, 0].max())
    x4_min, x4_max = float(pairs[:, 1].min()), float(pairs[:, 1].max())
    reach_span3, reach_span4 = x3_max - x3_min, x4_max - x4_min

    print(f"{network}:")
    print(
        f"  reachable (x3, x4)  "
        f"[{x3_min:.3f}, {x3_max:.3f}] x [{x4_min:.3f}, {x4_max:.3f}]"
    )
    print(
        f"  modelled Box        "
        f"[{b3[0]}, {b3[1]}] x [{b4[0]}, {b4[1]}]"
    )
    print("  cells (uniform 0.05 / 0.4):")
    for step in (0.4, 0.05):
        full = _full_grid_count(box_span3, box_span4, step)
        aabb = _full_grid_count(reach_span3, reach_span4, step)
        occ = _occupied(pairs, x3_min, x4_min, step)
        print(
            f"    dx={step:4.2f}  full Box {full:6d}   "
            f"reachable AABB {aabb:5d}   occupied {occ:5d}"
        )
    adapt = _adaptive_occupied(pairs, x3_min, dx3=0.05, eps=0.05)
    print(
        f"  adaptive |x3|-scaled heading, dx3=0.05, eps=0.05: "
        f"{adapt} occupied cells"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=51)
    args = parser.parse_args()

    domain = NavDomain.from_yaml()
    print(
        f"Init grid {args.grid}x{args.grid}, {domain.horizon_steps} periods, "
        f"{SUBSTEPS} sub-steps/period (sample instants + held flow)\n"
        "Occupied = unique cells that contain at least one sampled (x3, x4).\n"
        "That is a LOWER bound on a sound demand-driven cover.\n"
    )
    for network in NETWORKS:
        pairs = _cloud(network, domain, args.grid)
        report(network, domain, pairs)


if __name__ == "__main__":
    main()
