"""
The attack that decides the benchmark: what if the tool splits the INITIAL set?

Every reachability baseline here starts from one box spanning x in [-2, 0.9]
and v in [-0.8, 0.8], and CROWN's bound over that box is [-1.0000, +1.0000] -
the network's entire output range.  So the baselines were never given any
information about the network at all, which is why free control reproduces
their answer exactly.  That is not a deep fact about set propagation; it is a
fact about starting from a wide box.

Splitting the initial set into a grid and propagating each piece separately is
standard practice and it makes every piece's CROWN bound tight.  If a modest
split proves the property, the geofence result collapses and we say so.  If it
takes an implausible number of pieces, the cost of matching the solver
pointwise is the measurement, and that is the honest form of the claim.

Compositional is the limit of this process - it reasons about each concrete
state - so the question is only what a finite split costs to catch up.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from compare import X0_HI, X0_LO
from polytope_baseline import CrownOracle, Polygon, run
from nsbt import HARD_LIMIT


def run_split(oracle, pieces, steps, v0_bound, merge=True):
    """Propagate each initial sub-box independently and take the worst peak."""
    xs = np.linspace(X0_LO, X0_HI, pieces + 1)
    vs = np.linspace(-v0_bound, v0_bound, pieces + 1)
    worst, total, gave_up = X0_HI, 0, False
    for i in range(pieces):
        for j in range(pieces):
            verdict, peak, sets = run(oracle, steps, v0_bound, merge=merge,
                                      start=Polygon.box(xs[i], xs[i + 1],
                                                        vs[j], vs[j + 1]))
            worst = max(worst, peak)
            total += sets
            gave_up = gave_up or verdict == "gave up"
    return ("gave up" if gave_up else
            "violated" if worst > HARD_LIMIT else "safe"), worst, total


def main():
    """Sweep the initial-set split and report where, if ever, it succeeds."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    v0_bound = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    oracle = CrownOracle(refine=1)
    print(f"initial-set splitting: {steps} steps, |v0| <= {v0_bound}, "
          f"limit {HARD_LIMIT}, true peak 1.2622")
    print("compositional proves this down to a limit of 1.270 with no split\n")
    print(f"{'split':>7} {'boxes':>7} {'verdict':>10} {'peak x':>10} "
          f"{'sets':>9} {'sec':>8}")
    for pieces in (1, 2, 4, 8, 16, 32, 64):
        t0 = time.time()
        verdict, peak, sets = run_split(oracle, pieces, steps, v0_bound)
        print(f"{pieces:>5}x{pieces:<2} {pieces**2:>7} {verdict:>10} "
              f"{peak:>10.4f} {sets:>9} {time.time() - t0:>8.1f}", flush=True)
        if verdict == "safe":
            print("\n=> a finite initial-set split PROVES it. The reachability")
            print("   baselines were handicapped and the headline is WITHDRAWN")
            print("   in its current form.")
            return
    print("\n=> no split in the scanned range proves it. The cost of matching")
    print("   the solver pointwise is the measurement.")


if __name__ == "__main__":
    main()
