"""
The strongest reachability baseline: constrained zonotopes with exact guard cuts.

This is what CORA would do, implemented directly because CORA's constrained
zonotope operations require the MATLAB Optimization Toolbox and only base MATLAB
is licensed here.  The construction is Scott et al. 2016 (see conzono.py), and
its correlation preservation is verified: cutting x <= 0 on a set where x and v
are perfectly correlated forces v <= 0, where an axis-aligned box would still
allow v <= 1.

Baselines so far, all against a limit of 1.5 with a true maximum of 1.2622:

    interval, branch hull        1.5298   violated
    interval, guard split        1.5498   violated
    zonotope, branch hull        1.5298   violated
    zonotope, guard split        1.5861   violated
    constrained zonotope         <- this file, the one that keeps correlation
    compositional (Z3)           safe

If this also fails, every reachability baseline we can build fails, and the only
remaining question is whether CORA's own implementation does better.
"""
from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from compare import _crown_hull, crown_cells, X0_HI, X0_LO
from conzono import ConZonotope
from nsbt import BRAKE, DT, FENCE, HARD_LIMIT

A = np.array([[1.0, DT], [0.0, 1.0]])


# pylint: disable-next=too-many-locals
def run(cells, steps, v0_bound, cap=4000):
    """Propagate constrained zonotopes, cutting exactly at the guard."""
    start = ConZonotope([(X0_LO + X0_HI) / 2, 0.0],
                        [[(X0_HI - X0_LO) / 2, 0.0], [0.0, v0_bound]])
    sets = [start]
    worst, peak = X0_HI, 1
    for _ in range(steps):
        nxt = []
        for zono in sets:
            for piece, outside in _split(zono):
                span = piece.interval()
                if span is None:
                    continue
                lo, hi = span
                if outside:
                    alo = ahi = -BRAKE
                else:
                    alo, ahi = _crown_hull(cells, lo[0], hi[0], lo[1], hi[1])
                nxt.append(piece.linear_map(
                    A, [0.0, DT * (alo + ahi) / 2],
                    [[0.0], [DT * (ahi - alo) / 2]]))
        sets = nxt
        peak = max(peak, len(sets))
        for zono in sets:
            span = zono.interval()
            if span is not None:
                worst = max(worst, span[1][0])
        if worst > HARD_LIMIT:
            return "violated", worst, peak
        if len(sets) > cap:
            return "gave up", worst, peak
    return "safe", worst, peak


def _split(zono):
    """Cut at the guard, keeping correlation. Yields (piece, is_outside)."""
    span = zono.interval()
    if span is None:
        return []
    lo, hi = span
    if hi[0] <= FENCE:
        return [(zono, False)]
    if lo[0] >= FENCE:
        return [(zono, True)]
    inside = zono.intersect_halfspace([1.0, 0.0], FENCE)
    outside = zono.intersect_halfspace([-1.0, 0.0], -FENCE)
    out = []
    if inside is not None and not inside.is_empty():
        out.append((inside, False))
    if outside is not None and not outside.is_empty():
        out.append((outside, True))
    return out or [(zono, False)]


def main():
    """Run the strongest baseline and say plainly what the verdict means."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    v0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    cells = crown_cells(int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    t0 = time.time()
    verdict, worst, peak = run(cells, steps, v0)
    print(f"constrained-zonotope reachability ({len(cells)} contracts, "
          f"{steps} steps):")
    print(f"  {verdict}  max x {worst:.4f} (limit {HARD_LIMIT}, true max "
          f"1.2622), peak {peak} sets, {time.time() - t0:.1f}s")
    if verdict == "safe":
        print("  => a correlation-preserving baseline PROVES it. The benchmark")
        print("     does not discriminate and the pilot headline is WITHDRAWN.")
    else:
        print("  => even exact guard cuts fail. Every reachability baseline we")
        print("     can build loses on this system.")


if __name__ == "__main__":
    main()
