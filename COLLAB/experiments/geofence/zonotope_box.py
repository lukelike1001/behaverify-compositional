"""
A competent reachability baseline: zonotopes, not axis-aligned boxes.

Gemini refuted claim #86 and was right. splitting_box.py propagates axis-aligned
intervals, which lose the correlation between x and v every step - the wrapping
effect - on dynamics that are linear and coupled.  Attributing that failure to
"what CORA and Flow* do" was a strawman: CORA uses zonotopes, Flow* uses Taylor
models, SpaceEx uses support functions, and all three map EXACTLY through a
linear transformation.

This implements the zonotope version, so the baseline is competent:

    state as  c + sum_i  g_i * e_i ,  e_i in [-1, 1]
    linear map A:  c -> A c ,  g_i -> A g_i          (exact, no wrapping)
    input interval [alo, ahi]:  centre shifts, one new generator per step

The guard is still the hard part - a zonotope cannot be split at x = FENCE
without over-approximating - so the honest comparison is: does a method that
represents the CONTINUOUS dynamics exactly still lose on the BRANCH?

If it proves the property, the benchmark does not discriminate and the pilot's
headline must be withdrawn.  That is the outcome worth knowing.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from compare import _crown_hull, crown_cells, X0_HI, X0_LO
from nsbt import BRAKE, DT, FENCE, HARD_LIMIT

# x' = x + v dt ,  v' = v + a dt
A = np.array([[1.0, DT], [0.0, 1.0]])
B = np.array([[0.0], [DT]])


class Zonotope:
    """centre + generators, the representation CORA uses."""

    def __init__(self, centre, generators):
        self.c = np.asarray(centre, float).reshape(2)
        self.g = np.asarray(generators, float).reshape(2, -1)

    def bounds(self):
        """Interval hull, used only for reporting and guard tests."""
        radius = np.abs(self.g).sum(axis=1)
        return self.c - radius, self.c + radius

    def step(self, alo, ahi):
        """Exact linear map, plus one generator for the input interval."""
        centre = A @ self.c + B.flatten() * (alo + ahi) / 2
        mapped = A @ self.g
        new_gen = B * (ahi - alo) / 2
        return Zonotope(centre, np.hstack([mapped, new_gen]))

    def order(self):
        """Generators per dimension - the quantity CORA caps by reduction."""
        return self.g.shape[1] / 2


def run(cells, steps, v0_bound):
    """Propagate a zonotope, taking the branch hull where the set straddles."""
    zono = Zonotope([(X0_LO + X0_HI) / 2, 0.0],
                    [[(X0_HI - X0_LO) / 2, 0.0], [0.0, v0_bound]])
    worst = X0_HI
    for _ in range(steps):
        (xlo, vlo), (xhi, vhi) = zono.bounds()
        if xhi < FENCE:
            alo, ahi = _crown_hull(cells, xlo, xhi, vlo, vhi)
        elif xlo >= FENCE:
            alo = ahi = -BRAKE
        else:
            clo, chi = _crown_hull(cells, xlo, min(xhi, FENCE), vlo, vhi)
            alo, ahi = min(clo, -BRAKE), max(chi, -BRAKE)
        zono = zono.step(alo, ahi)
        worst = max(worst, zono.bounds()[1][0])
        if worst > HARD_LIMIT:
            return "violated", worst, zono.order()
    return "safe", worst, zono.order()


def main():
    """Run the zonotope baseline and say plainly what it means."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    v0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    cells = crown_cells(int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    verdict, worst, order = run(cells, steps, v0)
    print(f"  zonotope, branch hull   : {verdict:>9}  max x {worst:.4f} "
          f"(order {order:.0f})")
    sv, sw, peak = run_splitting(cells, steps, v0)
    print(f"  zonotope, guard split   : {sv:>9}  max x {sw:.4f} "
          f"(peak {peak} sets)")
    verdict = "safe" if (verdict == "safe" or sv == "safe") else "violated"
    if verdict == "safe":
        print("  => a competent baseline PROVES the property. The benchmark does")
        print("     not discriminate, and the pilot's headline must be withdrawn.")
    else:
        print("  => even with exact linear propagation the branch defeats it, so")
        print("     the loss is the GUARD and not the wrapping effect.")



def split_zonotope(zono, at):
    """Split a zonotope at the hyperplane x = `at`, over-approximating each part.

    A zonotope is not closed under halfspace intersection, so CORA and its
    relatives either move to constrained zonotopes or over-approximate. This
    takes the standard cheap route: cut the interval hull and re-fit an
    axis-aligned zonotope to each piece. That loses correlation in the piece it
    cuts, which is exactly the price a real tool pays at a guard, so it is a
    fair rather than a flattering baseline.
    """
    (xlo, vlo), (xhi, vhi) = zono.bounds()
    if xhi <= at or xlo >= at:
        return [zono]
    out = []
    for lo, hi in ((xlo, at), (at, xhi)):
        out.append(Zonotope([(lo + hi) / 2, (vlo + vhi) / 2],
                            [[(hi - lo) / 2, 0.0], [0.0, (vhi - vlo) / 2]]))
    return out


# pylint: disable-next=too-many-locals
def run_splitting(cells, steps, v0_bound, cap=100000):
    """Zonotope dynamics AND guard splitting: the strongest cheap baseline."""
    zonos = [Zonotope([(X0_LO + X0_HI) / 2, 0.0],
                      [[(X0_HI - X0_LO) / 2, 0.0], [0.0, v0_bound]])]
    worst, peak = X0_HI, 1
    for _ in range(steps):
        nxt = []
        for zono in zonos:
            for piece in split_zonotope(zono, FENCE):
                (plo, wlo), (phi, whi) = piece.bounds()
                if plo >= FENCE:
                    alo = ahi = -BRAKE
                else:
                    alo, ahi = _crown_hull(cells, plo, phi, wlo, whi)
                nxt.append(piece.step(alo, ahi))
        zonos = nxt
        peak = max(peak, len(zonos))
        worst = max(worst, max(z.bounds()[1][0] for z in zonos))
        if worst > HARD_LIMIT:
            return "violated", worst, peak
        if len(zonos) > cap:
            return "gave up", worst, peak
    return "safe", worst, peak

if __name__ == "__main__":
    main()
