"""
init_clearance.py

Experiment 1 of the Q1 contract gate: the true minimum obstacle clearance over
the WHOLE initial set, not a handful of sampled starts.

Q1 quantifies over Init = [2.9,3.1]^2 x {0} x {0}. If any true trajectory from
Init enters the obstacle, Q1 is false, no sound contract set can prove it, and
a refinement loop would split forever. So this runs before any contract work.

Clearance is measured on the TUBE, not at the 31 samples: the control is held
over each period and the state is sub-stepped, so a trajectory that clips the
obstacle corner between two samples is caught.

Also records the width of the true reachable cloud at each sample, which is the
budget against which interval wrapping has to be judged: if the true cloud is
already as wide as the clearance, over-approximation excess is not the issue.

    python3 -m scripts.init_clearance --grid 21
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort

from core.nav_tree_generator import NavDomain
from core.paths import EXAMPLE_ROOT

NETWORKS = ("point", "set")
SUBSTEPS = 200  # per control period, for the held-control flow


def _session(network: str) -> tuple[ort.InferenceSession, str]:
    path = EXAMPLE_ROOT / "networks" / f"nn-nav-{network}.onnx"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name


def _control(session, input_name, states: np.ndarray) -> np.ndarray:
    """Batched forward pass. states is (n, 4); returns (n, 2)."""
    return session.run(None, {input_name: states.astype(np.float32)})[0]


def _derivative(states: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """The ARCH-COMP NAV vector field, held control."""
    out = np.empty_like(states)
    out[:, 0] = states[:, 2] * np.cos(states[:, 3])
    out[:, 1] = states[:, 2] * np.sin(states[:, 3])
    out[:, 2] = controls[:, 0]
    out[:, 3] = controls[:, 1]
    return out


def _rk4(states: np.ndarray, controls: np.ndarray, h: float) -> np.ndarray:
    k1 = _derivative(states, controls)
    k2 = _derivative(states + 0.5 * h * k1, controls)
    k3 = _derivative(states + 0.5 * h * k2, controls)
    k4 = _derivative(states + h * k3, controls)
    return states + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _clearance(positions: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Signed distance from each point to the axis-aligned obstacle square.

    Positive outside, negative inside (penetration depth). positions is (n, 2).
    """
    below = lo - positions
    above = positions - hi
    outside = np.maximum(np.maximum(below, above), 0.0)
    outside_distance = np.linalg.norm(outside, axis=1)
    inside_depth = np.max(np.minimum(positions - lo, hi - positions), axis=1)
    return np.where(
        outside_distance > 0.0, outside_distance, -np.abs(inside_depth),
    )


def run(network: str, domain: NavDomain, grid: int) -> dict:
    session, input_name = _session(network)

    axis = np.linspace(2.9, 3.1, grid)
    mesh_x, mesh_y = np.meshgrid(axis, axis, indexing="ij")
    starts = np.column_stack(
        [
            mesh_x.ravel(),
            mesh_y.ravel(),
            np.zeros(grid * grid),
            np.zeros(grid * grid),
        ],
    ).astype(np.float64)

    obstacle_lo, obstacle_hi = domain.obstacle["x1"][0], domain.obstacle["x1"][1]
    dt = domain.dt
    h = dt / SUBSTEPS

    states = starts.copy()
    best = np.full(len(states), np.inf)
    best_step = np.zeros(len(states), dtype=int)
    widths = []

    initial = _clearance(states[:, :2], obstacle_lo, obstacle_hi)
    best = np.minimum(best, initial)
    widths.append(
        (0, np.ptp(states[:, 0]), np.ptp(states[:, 1]),
         np.ptp(states[:, 2]), np.ptp(states[:, 3])),
    )

    for step in range(domain.horizon_steps):
        controls = _control(session, input_name, states).astype(np.float64)
        for _ in range(SUBSTEPS):
            states = _rk4(states, controls, h)
            here = _clearance(states[:, :2], obstacle_lo, obstacle_hi)
            improved = here < best
            best_step[improved] = step
            best = np.minimum(best, here)
        widths.append(
            (step + 1, np.ptp(states[:, 0]), np.ptp(states[:, 1]),
             np.ptp(states[:, 2]), np.ptp(states[:, 3])),
        )

    goal_lo, goal_hi = domain.goal["x1"][0], domain.goal["x1"][1]
    in_goal = (
        (states[:, 0] >= goal_lo) & (states[:, 0] <= goal_hi)
        & (states[:, 1] >= goal_lo) & (states[:, 1] <= goal_hi)
    )

    worst = int(np.argmin(best))
    return {
        "network": network,
        "starts": len(states),
        "min_clearance": float(best[worst]),
        "worst_start": tuple(starts[worst, :2]),
        "worst_step": int(best_step[worst]),
        "collisions": int((best < 0).sum()),
        "goal_at_horizon": int(in_goal.sum()),
        "widths": widths,
        "final_cloud": (
            float(np.ptp(states[:, 0])), float(np.ptp(states[:, 1])),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=21)
    parser.add_argument("--widths", action="store_true")
    args = parser.parse_args()

    domain = NavDomain.from_yaml()
    print(f"Init grid {args.grid}x{args.grid} over [2.9,3.1]^2, "
          f"{domain.horizon_steps} periods, {SUBSTEPS} sub-steps/period\n")

    for network in NETWORKS:
        result = run(network, domain, args.grid)
        print(f"{network}:")
        print(f"  starts                {result['starts']}")
        print(f"  min tube clearance    {result['min_clearance']:.6f}")
        print(f"    at start            {result['worst_start']}")
        print(f"    during period       {result['worst_step']}")
        print(f"  starts colliding      {result['collisions']}")
        print(f"  in goal at k=30       {result['goal_at_horizon']}"
              f" / {result['starts']}")
        print(f"  final cloud w(x1,x2)  {result['final_cloud'][0]:.4f}, "
              f"{result['final_cloud'][1]:.4f}")
        if args.widths:
            print("  step  w(x1)   w(x2)   w(x3)   w(x4)")
            for step, w1, w2, w3, w4 in result["widths"]:
                print(f"  {step:4d}  {w1:.4f}  {w2:.4f}  {w3:.4f}  {w4:.4f}")
        print()


if __name__ == "__main__":
    main()
