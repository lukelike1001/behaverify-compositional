"""
Adaptive initial-set splitting for the geofence benchmark.

Measures whether reachability with adaptive refinement (splitting only
violating sub-boxes rather than a uniform 32x32 grid) proves the
clearance-style property, and how many boxes / seconds it requires.
"""
from __future__ import annotations

import collections
import sys
import time

import numpy as np

from compare import X0_HI, X0_LO
from polytope_baseline import CrownOracle, Polygon, run
from nsbt import HARD_LIMIT


def adaptive_split(oracle, limit, steps=20, v0_bound=0.8, max_depth=10, merge=True):
    """Adaptively split boxes that exceed limit.

    Returns: verdict, worst_peak, num_evaluated, leaf_boxes, elapsed_time
    """
    t0 = time.time()
    # Box representation: (xlo, xhi, vlo, vhi, depth)
    queue = collections.deque([(X0_LO, X0_HI, -v0_bound, v0_bound, 0)])
    worst_peak = X0_HI
    num_evaluated = 0
    leaf_boxes = 0

    while queue:
        xlo, xhi, vlo, vhi, depth = queue.popleft()
        num_evaluated += 1
        poly = Polygon.box(xlo, xhi, vlo, vhi)
        verdict, peak, _ = run(oracle, steps, v0_bound, merge=merge, start=poly)
        worst_peak = max(worst_peak, peak)

        if peak <= limit:
            # Box is certified safe! No need to refine further
            leaf_boxes += 1
            continue

        # If violated and within depth, bisect
        if depth >= max_depth:
            # Cannot refine further; entire search violated
            return "violated", worst_peak, num_evaluated, leaf_boxes, time.time() - t0

        # Choose dimension to bisect: normalize by initial width
        dx_norm = (xhi - xlo) / (X0_HI - X0_LO)
        dv_norm = (vhi - vlo) / (2 * v0_bound)

        if dv_norm >= dx_norm:
            # Bisect along velocity
            vmid = (vlo + vhi) / 2.0
            queue.append((xlo, xhi, vlo, vmid, depth + 1))
            queue.append((xlo, xhi, vmid, vhi, depth + 1))
        else:
            # Bisect along position
            xmid = (xlo + xhi) / 2.0
            queue.append((xlo, xmid, vlo, vhi, depth + 1))
            queue.append((xmid, xhi, vlo, vhi, depth + 1))

    return "safe", worst_peak, num_evaluated, leaf_boxes, time.time() - t0


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    v0_bound = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    oracle = CrownOracle(refine=1)

    print(f"Adaptive Initial-Set Splitting: steps={steps}, |v0|<={v0_bound}")
    print(f"True peak x: 1.2622 (hard limit {HARD_LIMIT}, compositional floor 1.270)\n")
    print(f"{'limit':>8} {'verdict':>10} {'peak x':>10} {'evaluated':>10} {'leaves':>8} {'sec':>8}")

    for limit in (1.50, 1.40, 1.30, 1.27):
        verdict, peak, evals, leaves, secs = adaptive_split(
            oracle, limit=limit, steps=steps, v0_bound=v0_bound, max_depth=12
        )
        print(f"{limit:>8.2f} {verdict:>10} {peak:>10.4f} {evals:>10} {leaves:>8} {secs:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
