"""
Find the smallest branch that a single interval box CANNOT represent.

Tonight's droneNew audit killed the obvious version of this argument: droneNew's
dynamics are a symmetric +/-1 walk, whose reachable set IS an interval, so a box
loses nothing (measured 1.00x for 25 steps).  A box only loses when the branch
makes the reachable set NON-CONVEX - when the two branches move the state in
ways that leave a gap in the middle.

A geofence does exactly that, and it is a real requirement rather than a
contrived one: aviation regulators require a drone to turn back at a boundary.

    if x >= FENCE:  brake hard   (a = -A)
    else:           cruise       (a = +A)

Straddle the fence and the accelerations point in OPPOSITE directions, so after
one step the velocities separate into two clusters with nothing between them.
That gap is what a box cannot express and a disjunction can.

This measures the gap before anything is built on top of it.
"""
from __future__ import annotations

import numpy as np

DT = 0.1
FENCE = 1.0
ACCEL = 2.0


def step_exact(states):
    """One step of the branching system, keeping the state set explicit."""
    out = []
    for x, v in states:
        a = -ACCEL if x >= FENCE else ACCEL      # the behavior-tree branch
        out.append((x + v * DT, v + a * DT))
    return out


def main():
    """Compare the exact reachable set against its bounding box."""
    # an initial set that straddles the fence, as a real uncertainty would
    xs = np.linspace(0.95, 1.05, 21)
    states = [(float(x), 0.0) for x in xs]

    print(f"geofence at x={FENCE}, accel +/-{ACCEL}, dt={DT}")
    print(f"{'k':>3} {'exact v clusters':>34} {'box v':>16} {'gap':>7}")
    for k in range(1, 6):
        states = step_exact(states)
        vs = sorted(v for _, v in states)
        lo, hi = vs[0], vs[-1]
        # a gap is any jump between consecutive values larger than the spacing
        gaps = [(b - a) for a, b in zip(vs, vs[1:])]
        biggest = max(gaps) if gaps else 0.0
        clusters = 1 + sum(1 for g in gaps if g > 1e-9 and g > 2 * np.median(gaps))
        occupied = sum(g for g in gaps if g <= 2 * max(np.median(gaps), 1e-12))
        print(f"{k:>3} {clusters:>2} cluster(s), v in "
              f"[{lo:>6.3f},{hi:>6.3f}] {'':>2} [{lo:>6.3f},{hi:>6.3f}] "
              f"{biggest:>7.3f}")
        if biggest > 1e-9:
            frac = 1 - occupied / max(hi - lo, 1e-12)
            print(f"      -> the box covers {frac * 100:.0f}% MORE velocity "
                  f"than the system can actually reach")


if __name__ == "__main__":
    main()
