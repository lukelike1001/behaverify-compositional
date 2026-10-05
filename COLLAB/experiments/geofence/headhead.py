"""
One process, one clock, one initial set. Every number in a single table.

This exists because I quoted grid world liveness's 14.5-minute CROWN figure as
if it were the geofence's, and concluded reachability beat us sixty times when
in fact we are faster. The fix is not to be more careful when quoting; it is to
stop quoting across benchmarks at all. Everything below is measured here, now,
on the same initial set and the same horizon, or it is not reported.
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")
# pylint: disable=wrong-import-position
from adaptive_split import adaptive_split
from compare import X0_HI, X0_LO, compositional, crown_cells
from nsbt import HARD_LIMIT
from polytope_baseline import CrownOracle

STEPS, V0 = 20, 0.8


def timed(label, fn):
    """Run and report wall clock, so no figure is ever quoted second hand."""
    start = time.time()
    out = fn()
    return label, out, time.time() - start


def main():
    """Measure both methods end to end at two limits and print one table."""
    print(f"geofence head to head: initial set x in [{X0_LO}, {X0_HI}], "
          f"|v0| <= {V0}, {STEPS} steps")
    print("true peak 1.2622; compositional proof floor 1.270\n")

    _, cells, t_crown = timed("crown", lambda: crown_cells(12))
    _, oracle, t_oracle = timed("oracle", lambda: CrownOracle(refine=1))
    print(f"contract generation: {len(cells)} contracts in {t_crown:.2f}s")
    print(f"oracle construction (for the splitting side): {t_oracle:.2f}s\n")

    print(f"{'limit':>7} {'method':>24} {'verdict':>10} {'solve s':>9} "
          f"{'total s':>9}")
    for limit in (HARD_LIMIT, 1.27):
        res, secs = compositional(cells, STEPS, V0, limit=limit)
        verdict = "proved" if str(res) == "unsat" else "no proof"
        print(f"{limit:>7.2f} {'compositional (Z3)':>24} {verdict:>10} "
              f"{secs:>9.2f} {secs + t_crown:>9.2f}")

        out = adaptive_split(oracle, limit, steps=STEPS, v0_bound=V0)
        avedict, _, evaluated, leaves, asecs = out
        print(f"{limit:>7.2f} {'adaptive splitting':>24} "
              f"{'proved' if avedict == 'safe' else avedict:>10} "
              f"{asecs:>9.2f} {asecs + t_oracle:>9.2f}"
              f"   ({evaluated} evaluated, {leaves} leaves)")
    print("\nContract generation is counted in the compositional total.")
    print("Oracle construction is counted in the splitting total.")
    print("Both sides call CROWN; neither figure is borrowed from elsewhere.")


if __name__ == "__main__":
    main()
