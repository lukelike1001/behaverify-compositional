"""
control_ranges.py

What control range does each NAV network actually emit, per box, per step?

This is the number that decides whether the in-model nuXmv encoding gets a
proof or a spurious FALSE. A hand-picked contract can be made as tight as one
likes; a contract derived from the network cannot. So: propagate each piece of
Init independently, and at every control period record the sampled spread of
the network's output over that piece.

Sampled spread is a LOWER bound on the true range over the piece, which is in
turn a lower bound on any sound CROWN bound. So every width here is optimistic
-- if these are already too wide to be useful, CROWN cannot rescue them.

    python3 -m scripts.control_ranges --pieces 1 4 16
"""

from __future__ import annotations

import argparse

import numpy as np

from core.nav_tree_generator import NavDomain
from scripts.hull_split import SAMPLES_PER_PIECE, _seed
from scripts.init_clearance import _control, _rk4, _session

SUBSTEPS = 50


def run(network: str, domain: NavDomain, pieces: int) -> dict:
    session, input_name = _session(network)
    states = _seed(pieces).astype(np.float64)
    n_pieces, per_piece = pieces * pieces, SAMPLES_PER_PIECE**2
    h = domain.dt / SUBSTEPS

    first: tuple | None = None
    widest = np.zeros(2)
    widest_step = [0, 0]
    per_step_max = []

    for step in range(domain.horizon_steps):
        controls = _control(session, input_name, states).astype(np.float64)
        grouped = controls.reshape(n_pieces, per_piece, 2)
        spread = grouped.max(axis=1) - grouped.min(axis=1)  # (n_pieces, 2)
        worst = spread.max(axis=0)
        per_step_max.append((step, worst[0], worst[1]))
        for axis in (0, 1):
            if worst[axis] > widest[axis]:
                widest[axis] = worst[axis]
                widest_step[axis] = step
        if step == 0:
            first = (
                float(controls[:, 0].min()), float(controls[:, 0].max()),
                float(controls[:, 1].min()), float(controls[:, 1].max()),
            )
        for _ in range(SUBSTEPS):
            states = _rk4(states, controls, h)

    return {
        "pieces": n_pieces,
        "u_at_init": first,
        "widest": tuple(widest),
        "widest_step": tuple(widest_step),
        "per_step_max": per_step_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pieces", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--per-step", action="store_true")
    args = parser.parse_args()

    domain = NavDomain.from_yaml()
    for network in ("point", "set"):
        print(f"{network}:")
        for pieces in args.pieces:
            result = run(network, domain, pieces)
            u1lo, u1hi, u2lo, u2hi = result["u_at_init"]
            print(
                f"  {pieces:2d}x{pieces:<2d} ({result['pieces']:4d} boxes)  "
                f"u at Init: u1 [{u1lo:+.4f},{u1hi:+.4f}] "
                f"u2 [{u2lo:+.4f},{u2hi:+.4f}]",
            )
            print(
                f"          widest per-box spread over 30 periods: "
                f"u1 {result['widest'][0]:.4f} (step {result['widest_step'][0]}), "
                f"u2 {result['widest'][1]:.4f} (step {result['widest_step'][1]})",
            )
            if args.per_step:
                for step, w1, w2 in result["per_step_max"]:
                    print(f"            step {step:2d}  u1 {w1:.4f}  u2 {w2:.4f}")
        print()


if __name__ == "__main__":
    main()
