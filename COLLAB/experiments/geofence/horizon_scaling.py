"""
Does the compositional cost grow more slowly than the reachability cost?

The reachability side is measured in limit_sweep.py and it diverges: the peak
x it reports grows without bound while the true system's peak is constant, and
its set count fragments past any fixed budget.  That is only half an argument.
The other half is whether the compositional side stays affordable, because a
method that answers correctly but takes an hour at 60 steps is not a win.

Structurally the two should differ.  Guard splitting is MULTIPLICATIVE in sets
- every straddle doubles, and the pieces persist for the rest of the run - while
bounded model checking is ADDITIVE in constraints, one more copy of the
transition relation per step, with the solver sharing structure across the
disjunction rather than enumerating it.  This measures whether that shows up in
wall clock, and it is written to be able to REFUTE the expectation: an unknown
or a super-linear curve is reported as such rather than dropped.
"""
from __future__ import annotations

import sys
import time

from compare import compositional, crown_cells
from conzono_baseline import run as conzono_run

HORIZONS = (10, 20, 30, 40, 60, 80)
TIMEOUT_S = 1800.0


def compositional_curve(cells, v0_bound, horizons, timeout_s=TIMEOUT_S):
    """Wall clock and verdict for the compositional proof at each horizon."""
    rows = []
    for steps in horizons:
        res, secs = compositional(cells, steps, v0_bound, timeout_s=timeout_s)
        rows.append((steps, _verdict(res), secs))
        print(f"  {steps:>3} steps  {_verdict(res):>10}  {secs:>8.1f}s",
              flush=True)
    return rows


def reachability_curve(cells, v0_bound, horizons, cap=4000):
    """Peak x and peak set count for the strongest baseline at each horizon."""
    rows = []
    for steps in horizons:
        t0 = time.time()
        verdict, peak_x, sets = conzono_run(cells, steps, v0_bound, cap=cap,
                                            limit=float("inf"))
        rows.append((steps, verdict, peak_x, sets, time.time() - t0))
        print(f"  {steps:>3} steps  {verdict:>10}  peak x {peak_x:>8.4f}  "
              f"{sets:>6} sets  {time.time() - t0:>7.1f}s", flush=True)
    return rows


def _verdict(res):
    return {"unsat": "proved", "sat": "no proof"}.get(str(res), "unknown")


def _growth(rows, value):
    """Ratio of each row's cost to the previous row's, for reading the curve."""
    out = []
    for i in range(1, len(rows)):
        prev, cur = value(rows[i - 1]), value(rows[i])
        out.append(cur / prev if prev > 0 else float("inf"))
    return out


def main():
    """Run both curves and print them side by side with growth ratios."""
    v0_bound = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
    cells = crown_cells(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    print(f"horizon scaling: {len(cells)} contracts, |v0| <= {v0_bound}, "
          f"timeout {TIMEOUT_S:.0f}s per solve\n")

    print("compositional (Z3 BMC):", flush=True)
    comp = compositional_curve(cells, v0_bound, HORIZONS)
    print("\nconstrained-zonotope reachability:", flush=True)
    reach = reachability_curve(cells, v0_bound, HORIZONS)

    print(f"\n{'steps':>6} {'compositional':>14} {'sec':>9} "
          f"{'reach peak x':>13} {'sets':>7} {'sec':>8}")
    for (steps, verdict, secs), (_, _, peak_x, sets, rsecs) in zip(comp, reach):
        print(f"{steps:>6} {verdict:>14} {secs:>9.1f} {peak_x:>13.4f} "
              f"{sets:>7} {rsecs:>8.1f}")

    print("\ngrowth per horizon step (ratio to the previous row)")
    print("  compositional time  ",
          "  ".join(f"{r:.2f}" for r in _growth(comp, lambda r: r[2])))
    print("  reachability sets   ",
          "  ".join(f"{r:.2f}" for r in _growth(reach, lambda r: r[3])))
    print("  reachability peak x ",
          "  ".join(f"{r:.2f}" for r in _growth(reach, lambda r: r[2])))
    if any(v == "unknown" for _, v, _ in comp):
        print("\nAT LEAST ONE SOLVE TIMED OUT. The compositional side does not")
        print("scale to that horizon and the claim must say so.")


if __name__ == "__main__":
    main()
