"""
Derive the operating envelope and the property constants. Do not assert them.

The geofence is the cautionary tale: the first initial set I chose made the
safety property FALSE (max x 2.55 against a limit of 1.5), and the envelope had
to be re-derived by measuring the true closed loop. A benchmark whose property
is false proves nothing, and Gemini found two more false properties in v0.1 of
this one before any code existed.

So nothing here is chosen. The clearance bound and the recovery-tick bound are
MEASURED over a dense sweep of the initial set, and the constants that go in
the paper are those measurements with margin.

WHAT THE INITIAL SET IS, and why it is not the start of the mission. The whole
flight - cruise out, trip the hazard, brake, fly home - takes about 46 ticks,
and compositional verification dies between 40 and 60 (0.1 / 7.7 / 54.9 / 246 s
at 10 / 20 / 30 / 40 steps). Verifying the cruise phase buys nothing: it is the
RECOVERY that is safety-critical. So the initial set is the state at the moment
the hazard fires, quantified over the region and speeds where that can happen,
and the horizon covers the recovery alone.
"""
from __future__ import annotations

import numpy as np

from plant import (CLEARANCE, DT, DRAG, S_MAX, START, Tree, _clip,
                   obstacle_distance, return_law)

# The trip region must sit outside the obstacle AND far enough from its face
# that a vehicle at the speed limit can still stop: v^2/(2*a_max) = 0.25 at
# v = 1.0, plus clearance. The obstacle face is at y = 0.8, so y <= 0.45.
# My first attempt used y up to 0.85, which is INSIDE the obstacle box, and the
# sweep correctly reported the true system entering it.
# Continuous stopping distance v^2/(2*a_max) = 0.25 at v = 1.0 UNDERSTATES the
# discrete-time braking distance: deceleration is applied once per tick and the
# brake step hands over at |v| <= 0.05, so the vehicle covers 0.30 rather than
# 0.25. Measured at y <= 0.45 the worst clearance was -0.0073, i.e. a graze.
# Backing the region off to y <= 0.35 is the derived answer, not a guess.
TRIP = (0.55, 0.90, 0.20, 0.35)
HORIZON = 40


def recover(tree, x, y, vx, vy, steps=HORIZON):
    """Run the recovery branch alone, from a latched hazard. Returns the trace."""
    tree.reset()
    tree.hazards["TooFast"] = True      # latched: the mission is already lost
    trace = []
    for _ in range(steps):
        ax, ay, node, active = tree.tick(x, y, vx, vy)
        trace.append((x, y, vx, vy, node, active))
        x, y = x + vx * DT, y + vy * DT
        vx = vx + (ax - DRAG * vx) * DT
        vy = vy + (ay - DRAG * vy) * DT
    trace.append((x, y, vx, vy, "end", False))
    return trace


def sweep(n=13, speeds=5):
    """Dense sweep of the trip region at every heading, up to the speed limit."""
    tree = Tree()
    worst_gap, worst_at = np.inf, None
    longest, longest_at = 0, None
    incomplete = []
    for x in np.linspace(TRIP[0], TRIP[1], n):
        for y in np.linspace(TRIP[2], TRIP[3], n):
            for speed in np.linspace(0.2, S_MAX, speeds):
                for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    vx, vy = speed * np.cos(theta), speed * np.sin(theta)
                    trace = recover(tree, float(x), float(y), vx, vy)
                    gap = min(obstacle_distance(s[0], s[1]) for s in trace)
                    ticks = sum(1 for s in trace if s[5])
                    if gap < worst_gap:
                        worst_gap, worst_at = gap, (x, y, speed, theta)
                    if ticks > longest:
                        longest, longest_at = ticks, (x, y, speed, theta)
                    if trace[-1][4] != "end" or ticks >= HORIZON:
                        incomplete.append((x, y, speed, theta, ticks))
    return worst_gap, worst_at, longest, longest_at, incomplete


def main():
    """Report the measured bounds, and say plainly whether they are usable."""
    print(f"divert envelope: trip region {TRIP}, speeds up to {S_MAX}, "
          f"horizon {HORIZON}")
    gap, gap_at, ticks, ticks_at, bad = sweep()
    total = 13 * 13 * 5 * 8
    print(f"  swept {total} recovery starts\n")
    print(f"  worst clearance        {gap:.4f}  at x={gap_at[0]:.2f} "
          f"y={gap_at[1]:.2f} |v|={gap_at[2]:.2f} th={gap_at[3]:.2f}")
    print(f"  longest recovery       {ticks} ticks  at x={ticks_at[0]:.2f} "
          f"y={ticks_at[1]:.2f} |v|={ticks_at[2]:.2f}")
    print(f"  did not finish in {HORIZON}: {len(bad)} of {total}")
    print()
    if gap <= 0:
        print("  *** the TRUE system enters the obstacle. The benchmark is")
        print("      broken and the clearance property is false. ***")
        return
    if bad:
        print(f"  *** {len(bad)} starts do not complete recovery inside the")
        print("      horizon, so the bounded-recovery property is FALSE. ***")
    print(f"  derived delta_safe   = {gap * 0.8:.3f}  (80% of measured worst, "
          f"draft assumed {CLEARANCE})")
    print(f"  derived N            = {ticks + 3}  (measured worst + 3)")
    print(f"  both must be stated as DERIVED, with this sweep as the evidence.")


if __name__ == "__main__":
    main()
