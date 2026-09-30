"""
Is the hard limit of 1.5 tuned so that reachability lands on the wrong side?

The obvious attack on this benchmark: the true maximum is 1.2622, the limit is
1.5, and the strongest reachability baseline reports 1.5498.  That is a 3%
overshoot past the limit, so a reviewer can say the limit was picked to sit in
the one narrow band where the methods disagree.

This measures the band.  For each candidate limit it reports what the true
system does, what the compositional encoding can prove, and what each
reachability baseline concludes.  If the band is one or two hundredths wide the
attack lands; if it is wide the benchmark is not balanced on a knife edge.

It also measures whether the gap grows with the horizon, which distinguishes a
one-off numerical coincidence from a compounding structural loss.
"""
from __future__ import annotations

import math
import sys
import time

from compare import compositional, crown_cells, per_step_box
from conzono_baseline import run as conzono_run
from find_envelope import worst_over
from nsbt import Tree

X0_HI_ENV = 0.9


def reach_peaks(cells, steps, v0_bound):
    """Peak x each reachability baseline reaches with the early exit disabled.

    Both baselines grow monotonically, so the peak decides every limit at once:
    a limit below the peak is reported violated, one above it is reported safe.
    """
    _, box = per_step_box(cells, steps, v0_bound, limit=math.inf)
    verdict, cz, sets = conzono_run(cells, steps, v0_bound, limit=math.inf)
    return box, (None if verdict == "gave up" else cz), sets


def true_peak(steps, v0_bound):
    """Largest x the real closed loop reaches, by dense simulation."""
    return worst_over(Tree(), X0_HI_ENV, v0_bound, steps=steps)


def sweep(cells, steps, v0_bound, limits):
    """Ask Z3 to prove each candidate limit; unsat means proved safe."""
    rows = []
    for limit in limits:
        res, secs = compositional(cells, steps, v0_bound, limit=limit)
        rows.append((limit, "safe" if str(res) == "unsat" else "no proof",
                     secs))
    return rows


def horizon_growth(cells, v0_bound, horizons):
    """Does the reachability overshoot compound, or is it a fixed offset?"""
    rows = []
    for steps in horizons:
        box, cz, sets = reach_peaks(cells, steps, v0_bound)
        rows.append((steps, true_peak(steps, v0_bound), box, cz, sets))
    return rows


def _fmt(value):
    return "gave up" if value is None else f"{value:.4f}"


def _report_band(rows, true_max, box, cz):
    """Print the range of limits over which the methods disagree."""
    proved = [limit for limit, verdict, _ in rows if verdict == "safe"]
    if not proved:
        print("\ncompositional proved nothing in the scanned range")
        return
    low = min(proved)
    high = cz if cz is not None else box
    print(f"\ntrue maximum {true_max:.4f}")
    print(f"compositional proves every limit down to {low:.2f}")
    print(f"constrained zonotopes report violated up to {_fmt(cz)}")
    print(f"=> the two disagree for every limit in [{low:.2f}, {_fmt(high)}], "
          f"a band of {high - low:.2f}")
    print(f"   the benchmark's 1.5 sits {(1.5 - low) / (high - low):.0%} "
          "of the way into it")


# pylint: disable-next=too-many-locals
def main():
    """Sweep the limit, then sweep the horizon, and print both tables."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    v0_bound = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    cells = crown_cells(int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    limits = [1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.70]

    box, cz, sets = reach_peaks(cells, steps, v0_bound)
    truth = true_peak(steps, v0_bound)
    print(f"geofence limit sweep: {len(cells)} contracts, {steps} steps, "
          f"|v0| <= {v0_bound}")
    print(f"peak x  true {truth:.4f}   interval {box:.4f}   "
          f"constrained zonotope {_fmt(cz)} ({sets} sets)\n")

    print(f"{'limit':>6} {'true':>10} {'compositional':>14} "
          f"{'interval':>10} {'con-zono':>10} {'Z3 s':>7}")
    t0 = time.time()
    rows = sweep(cells, steps, v0_bound, limits)
    for limit, verdict, secs in rows:
        print(f"{limit:>6.2f} "
              f"{'safe' if truth <= limit else 'VIOLATED':>10} "
              f"{verdict:>14} "
              f"{'safe' if box <= limit else 'violated':>10} "
              f"{'safe' if cz is not None and cz <= limit else 'violated':>10} "
              f"{secs:>7.1f}")
    _report_band(rows, truth, box, cz)
    print(f"\nsweep took {time.time() - t0:.1f}s")

    print("\nhorizon growth (does the overshoot compound?)")
    print(f"{'steps':>6} {'true':>9} {'interval':>10} {'con-zono':>10} "
          f"{'sets':>6}")
    for h, truth_h, box_h, cz_h, sets_h in horizon_growth(
            cells, v0_bound, (10, 20, 30, 40, 60)):
        print(f"{h:>6} {truth_h:>9.4f} {box_h:>10.4f} {_fmt(cz_h):>10} "
              f"{sets_h:>6}")


if __name__ == "__main__":
    main()
