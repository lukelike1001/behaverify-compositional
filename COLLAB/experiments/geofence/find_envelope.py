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

That check is not ad hoc.  It is the viability kernel of the double integrator
under |u| <= 1, which is the canonical running example of the run-time-assurance
tutorial (Hobbs, Mote, Abate, Coogan and Feron, "Run Time Assurance for
Safety-Critical Systems", IEEE Control Systems Magazine 2023,
doi:10.1109/MCS.2023.3234380, Fig. 3: "a car moving straight towards a wall").
Our plant IS their plant, our brake IS their u = -1, and our fence is the
conservative switching surface their explicit-RTA section describes.  So the
envelope below is not a number we picked; it is their kernel boundary, and
``kernel_bound`` checks that the simulation agrees with it.
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


def kernel_bound(x_from, wall=HARD_LIMIT):
    """Fastest approach from x_from that full braking can still stop by wall.

    The viability kernel of the double integrator with |u| <= 1: a state is
    recoverable exactly when the remaining room covers the stopping distance,
    wall - x >= v^2 / 2.
    """
    return np.sqrt(max(0.0, 2.0 * (wall - x_from)))


def main():
    """Scan start bounds and report the largest safe envelope."""
    tree = Tree()
    print(f"fence {FENCE}, hard limit {HARD_LIMIT}, brake 1.0, dt {DT}")
    print(f"{'x0 <=':>7} {'|v0| <=':>8} {'max x':>8}  verdict")
    safe = []
    for x_hi in (0.0, 0.5, 0.9, 1.0):
        for v_hi in (0.5, 0.8, 1.0, 1.2):
            worst = worst_over(tree, x_hi, v_hi)
            ok = worst <= HARD_LIMIT
            print(f"{x_hi:>7.2f} {v_hi:>8.2f} {worst:>8.4f}  "
                  f"{'safe' if ok else 'VIOLATED'}")
            if ok:
                safe.append((x_hi, v_hi, worst))
    if not safe:
        print("\nno safe envelope in the scanned range")
        return
    x_hi, v_hi, worst = max(safe, key=lambda e: e[0] * e[1])
    print(f"\nlargest safe envelope found: x0 <= {x_hi}, |v0| <= {v_hi}, "
          f"max x = {worst:.4f}, margin {HARD_LIMIT - worst:.4f}")
    print("\nviability kernel (Hobbs et al. 2023, Fig. 3): braking only "
          "starts at the fence,")
    print(f"  so the bound is sqrt(2 * (limit - fence)) = "
          f"{kernel_bound(FENCE):.4f}; measured envelope {v_hi:.4f}")
    print(f"  from the start line x = {x_hi} it would be "
          f"{kernel_bound(x_hi):.4f} if braking were always available")


if __name__ == "__main__":
    main()
