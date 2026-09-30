"""
What is the certified operating envelope of the geofence NSBT?

A benchmark whose safety property is FALSE proves nothing, and my first choice
of initial set was false: it admitted states already past the fence and beyond
the hard limit.  Rather than guess a smaller one, this measures the largest
initial box for which the true closed loop is safe, so the benchmark's numbers
are derived rather than asserted.

Physical sanity check on the answer: braking at 1.0 from velocity v needs
v^2/2 of stopping distance, and there is HARD_LIMIT - FENCE = 0.5 of room past
the fence, so the fastest survivable approach is v = 1.0.
"""
from __future__ import annotations

import numpy as np

from nsbt import DT, FENCE, HARD_LIMIT, Tree, simulate


def worst_over(tree, x_hi, v_hi, steps=120, n=25):
    """Largest x reached from any start in [-2, x_hi] x [-v_hi, v_hi]."""
    worst = -np.inf
    for x0 in np.linspace(-2.0, x_hi, n):
        for v0 in np.linspace(-v_hi, v_hi, n):
            for x, _, _, _ in simulate(tree, float(x0), float(v0), steps):
                worst = max(worst, x)
    return worst


def main():
    """Scan start bounds and report the largest safe envelope."""
    tree = Tree()
    print(f"fence {FENCE}, hard limit {HARD_LIMIT}, brake 1.0, dt {DT}")
    print(f"{'x0 <=':>7} {'|v0| <=':>8} {'max x':>8}  verdict")
    best = None
    for x_hi in (0.0, 0.5, 0.9, 1.0):
        for v_hi in (0.5, 0.8, 1.0, 1.2):
            worst = worst_over(tree, x_hi, v_hi)
            ok = worst <= HARD_LIMIT
            print(f"{x_hi:>7.2f} {v_hi:>8.2f} {worst:>8.4f}  "
                  f"{'safe' if ok else 'VIOLATED'}")
            if ok and (best is None or x_hi * v_hi >= best[0] * best[1]):
                best = (x_hi, v_hi, worst)
    if best:
        print(f"\nlargest safe envelope found: x0 <= {best[0]}, |v0| <= {best[1]}, "
              f"max x = {best[2]:.4f}, margin {HARD_LIMIT - best[2]:.4f}")
    else:
        print("\nno safe envelope in the scanned range")


if __name__ == "__main__":
    main()
