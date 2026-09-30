"""
Does behavior-tree branching actually defeat per-step interval reachability?

This is the cheapest possible test of the project's central thesis, and it needs
no neural network and no parser change.

On NAV, plain per-step interval reachability BEAT the compositional architecture
(k=10 with 10 contracts and no solver, against k=8 with ~1,300 contracts and
Z3), because NAV's tree is a single action node: the reachable set stays
unimodal and one box tracks it.

droneNew is the opposite shape.  `loc` is nondeterministic every step
(`VAR BOOLEAN assign{result{True, False}}` - a GPS dropout), and the tree forks
on it: `loc = False` takes the emergency-landing branch and drives z DOWN, while
`loc = True` continues the mission and holds altitude.  A single interval box
must cover both, and once it straddles z = 5 the predicates `Tookoff` (z >= 5)
and `Landed` (z = 0) are BOTH ambiguous, so the admitted action set becomes
{takeoff, fly, land} over the whole box.

If a single box survives this, the thesis dies everywhere and no benchmark
engineering rescues it.  If it blows up while a disjunction-preserving encoding
does not, droneNew's structure is exactly what compositional verification is
for.

Both models are abstractions of the SAME droneNew dynamics, so the comparison
isolates the representation rather than the domain.
"""
from __future__ import annotations

import sys

import z3

# droneNew's own constants, from REPRODUCIBILITY/2026_FM/examples/BT2Fiacre
Z_MAX, Z_CRUISE, Z_TAKEOFF = 80.0, 20.0, 5.0
XY_LO, XY_HI = -30.0, 30.0
STEP = 1.0                    # the tree moves one unit per tick


def true_spread(steps):
    """Exact reachable z-interval, tracking the two branches as a SET.

    This is what a disjunction-preserving encoding represents: at each step the
    set of reachable altitudes, kept as an explicit union rather than collapsed
    into one hull.
    """
    reach = {Z_CRUISE}
    history = []
    for _ in range(steps):
        nxt = set()
        for z in reach:
            nxt.add(min(z + STEP, Z_MAX))        # loc True  -> climb/hold
            nxt.add(max(z - STEP, 0.0))          # loc False -> emergency land
        reach = nxt
        history.append((min(reach), max(reach), len(reach)))
    return history


def box_spread(steps):
    """Per-step interval reachability: ONE box, so the branches must merge.

    The box cannot say "these states land and those fly", so every step it
    takes the hull of both outcomes.
    """
    lo = hi = Z_CRUISE
    history = []
    for _ in range(steps):
        lo, hi = max(lo - STEP, 0.0), min(hi + STEP, Z_MAX)
        history.append((lo, hi))
    return history


def ambiguous(lo, hi):
    """Which tree predicates are undecided over this box?

    Each ambiguous predicate is a branch the box cannot resolve, so the
    abstraction must admit every action beneath it.
    """
    out = []
    if lo < Z_TAKEOFF < hi:
        out.append("Tookoff(z>=5)")
    if lo <= 0.0 < hi:
        out.append("Landed(z=0)")
    if lo < 60.0 < hi:
        out.append("Ceiling(z<=60)")
    return out


def main():
    """Compare the exact reachable set against the single-box hull."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    exact = true_spread(steps)
    box = box_spread(steps)
    print(f"droneNew altitude, {steps} steps, start z={Z_CRUISE}")
    print(f"{'k':>3} {'exact set':>22} {'|set|':>6} {'single box':>18} "
          f"{'box/exact':>10}  ambiguous predicates")
    for k, ((elo, ehi, n), (blo, bhi)) in enumerate(zip(exact, box), 1):
        ew, bw = ehi - elo, bhi - blo
        amb = ambiguous(blo, bhi)
        if k <= 6 or k % 5 == 0 or k == steps:
            print(f"{k:>3} [{elo:>7.1f},{ehi:>7.1f}] {n:>6} "
                  f"[{blo:>6.1f},{bhi:>6.1f}] {bw / max(ew, 1e-9):>9.2f}x  "
                  f"{', '.join(amb) if amb else '-'}")
    print()
    print("NOTE: the exact set and the box hull coincide in WIDTH here, because")
    print("droneNew's z-dynamics are a symmetric +/-1 walk whose reachable set")
    print("IS an interval. The box loses nothing on this variable.")


if __name__ == "__main__":
    main()
