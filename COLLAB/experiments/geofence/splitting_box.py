"""
Is the per-step box a strawman? Give the baseline its best shot.

compare.py's per_step_box takes the HULL of both branches whenever the box
straddles the fence.  A competent hybrid-reachability tool would not do that: it
would SPLIT the set at the guard and propagate each piece under its own
dynamics, which is exactly what Flow*, CORA and SpaceEx do at mode transitions.

If guard splitting recovers the proof, the headline result is much weaker and
should be restated as "naive interval propagation loses" rather than
"reachability loses".  That is worth knowing tonight.

The cost of splitting is fragmentation: each straddle doubles the number of sets
to track.  This measures both - whether splitting proves the property, and how
many sets it takes.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

from compare import _crown_hull, crown_cells, X0_HI, X0_LO
from nsbt import BRAKE, DT, FENCE, HARD_LIMIT


def split_at_fence(box):
    """Split a box at the guard x = FENCE, returning the pieces that exist."""
    xlo, xhi, vlo, vhi = box
    if xhi < FENCE or xlo >= FENCE:
        return [box]
    return [(xlo, min(xhi, FENCE), vlo, vhi),        # inside: network
            (max(xlo, FENCE), xhi, vlo, vhi)]        # outside: brake


def step(box, cells):
    """Advance one box one tick under whichever branch its guard selects."""
    xlo, xhi, vlo, vhi = box
    if xlo >= FENCE:
        alo = ahi = -BRAKE
    else:
        alo, ahi = _crown_hull(cells, xlo, xhi, vlo, vhi)
    return (xlo + vlo * DT, xhi + vhi * DT, vlo + alo * DT, vhi + ahi * DT)


def run(cells, steps, v0_bound, cap=200000):
    """Guard-splitting reachability. Returns (verdict, max x, peak set count)."""
    boxes = [(X0_LO, X0_HI, -v0_bound, v0_bound)]
    worst, peak = X0_HI, 1
    for _ in range(steps):
        nxt = []
        for box in boxes:
            for piece in split_at_fence(box):
                nxt.append(step(piece, cells))
        # merge duplicates so the count reflects genuinely distinct sets
        boxes = [tuple(round(c, 9) for c in b) for b in nxt]
        boxes = list(dict.fromkeys(boxes))
        peak = max(peak, len(boxes))
        worst = max(worst, max(b[1] for b in boxes))
        if worst > HARD_LIMIT:
            return "violated", worst, peak
        if len(boxes) > cap:
            return "gave up (fragmentation)", worst, peak
    return "safe", worst, peak


def main():
    """Compare hull-taking against guard-splitting on the same contracts."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    v0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    grid = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    cells = crown_cells(grid)
    verdict, worst, peak = run(cells, steps, v0)
    print(f"guard-splitting reachability: {verdict}  "
          f"(max x {worst:.4f}, limit {HARD_LIMIT}, peak {peak} sets)")


if __name__ == "__main__":
    main()
