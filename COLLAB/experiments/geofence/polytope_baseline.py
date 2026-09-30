"""
The strongest set-propagation baseline that can exist for this system.

Every earlier baseline could be attacked for its set representation: intervals
wrap, zonotopes cannot be cut exactly, constrained zonotopes need an LP, and my
own implementations of all three could be worse than CORA's.  This one removes
that whole line of attack by being EXACT.

In two dimensions with linear dynamics, a convex polygon is closed under every
operation the geofence needs, with no over-approximation anywhere:

    guard cut       halfspace clip                   exact
    plant step      linear map of the vertices       exact
    control input   Minkowski sum with a segment     exact (hull of two copies)

and the control bound comes from CROWN evaluated on demand over each polygon's
own bounding box, so there is no contract grid and therefore no union over
cells.  It also refines: ``refine=k`` subdivides each polygon into k-by-k
pieces before bounding the network, which is the mitigation a real tool applies
when its control interval is too wide.

What is left is the ONE approximation no set-propagation method can avoid:
within a single set the control is a RANGE, free to take any value in it
independently of where in the set the state actually is.  So if this baseline
still fails, the loss is the control abstraction rather than the geometry, and
the only way around it is to keep the branch and the control as a logical
disjunction and let a solver pick per state - which is what compositional does.

Two modes, because they separate the two things a tool must trade off:
    split   keep every piece, so the guard disjunction is exact and the set
            count grows
    merge   convex-hull the pieces each step, which is what a tool does to keep
            the set count bounded
"""
from __future__ import annotations

import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
# pylint: disable=wrong-import-position
import numpy as np
import torch

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from compare import X0_HI, X0_LO
from nsbt import BRAKE, DT, FENCE, HARD_LIMIT, NET
from train_controller import Controller

EPS = 1e-12


class Polygon:
    """A convex polygon in the (x, v) plane, stored as its vertices."""

    def __init__(self, points):
        self.points = np.asarray(points, float)

    @classmethod
    def box(cls, xlo, xhi, vlo, vhi):
        """The axis-aligned box as a polygon."""
        return cls([(xlo, vlo), (xhi, vlo), (xhi, vhi), (xlo, vhi)])

    @classmethod
    def hull(cls, points):
        """Convex hull by monotone chain; exact and dependency-free."""
        pts = sorted({(float(x), float(y)) for x, y in points})
        if len(pts) <= 2:
            return cls(pts)
        def half(seq):
            out = []
            for p in seq:
                while len(out) >= 2 and _cross(out[-2], out[-1], p) <= 0:
                    out.pop()
                out.append(p)
            return out[:-1]
        return cls(half(pts) + half(pts[::-1]))

    def is_empty(self):
        """True when the clip left nothing behind."""
        return len(self.points) == 0

    def bounds(self):
        """(xlo, xhi, vlo, vhi) of the polygon."""
        lo = self.points.min(axis=0)
        hi = self.points.max(axis=0)
        return lo[0], hi[0], lo[1], hi[1]

    def clip(self, normal, offset):
        """Intersect with the halfspace normal . p <= offset. Exact."""
        if self.is_empty():
            return self
        normal = np.asarray(normal, float)
        pts = self.points
        vals = pts @ normal - offset
        out = []
        for i, point in enumerate(pts):
            nxt = pts[(i + 1) % len(pts)]
            here, there = vals[i], vals[(i + 1) % len(pts)]
            if here <= EPS:
                out.append(tuple(point))
            if (here < -EPS and there > EPS) or (here > EPS and there < -EPS):
                out.append(tuple(point + (nxt - point) * (here /
                                                          (here - there))))
        return Polygon.hull(out) if out else Polygon([])

    def affine(self, matrix):
        """Linear map of every vertex. Exact for a convex polygon."""
        return Polygon(self.points @ np.asarray(matrix, float).T)

    def sweep(self, low, high):
        """Minkowski sum with the segment [low, high]. Exact."""
        low = np.asarray(low, float)
        high = np.asarray(high, float)
        return Polygon.hull(list(self.points + low) + list(self.points + high))


# pylint: disable-next=too-few-public-methods
class CrownOracle:
    """CROWN bounds on the network's output over an arbitrary input box."""

    def __init__(self, model=None, refine=1):
        if model is None:
            model = Controller()
            model.load_state_dict(torch.load(
                pathlib.Path(NET).with_suffix(".pt"),
                weights_only=True))
            model.eval()
        self.bounded = BoundedModule(model, torch.zeros(1, 2))
        self.refine = refine
        self.cache = {}

    def bounds(self, xlo, xhi, vlo, vhi):
        """Tightest control range CROWN gives over this box, after refining."""
        key = (round(xlo, 9), round(xhi, 9), round(vlo, 9), round(vhi, 9))
        if key not in self.cache:
            self.cache[key] = self._bounds(*key)
        return self.cache[key]

    # pylint: disable-next=too-many-locals
    def _bounds(self, xlo, xhi, vlo, vhi):
        xs = np.linspace(xlo, xhi, self.refine + 1)
        vs = np.linspace(vlo, vhi, self.refine + 1)
        los, his = [], []
        for i in range(self.refine):
            for j in range(self.refine):
                lo = np.array([[xs[i], vs[j]]], np.float32)
                hi = np.array([[xs[i + 1], vs[j + 1]]], np.float32)
                tensor = BoundedTensor(torch.tensor((lo + hi) / 2),
                                       PerturbationLpNorm(
                                           x_L=torch.tensor(lo),
                                           x_U=torch.tensor(hi)))
                low, high = self.bounded.compute_bounds(x=(tensor,),
                                                        method="CROWN")
                los.append(max(float(low[0][0]), -1.0))
                his.append(min(float(high[0][0]), 1.0))
        return min(los), max(his)


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _split_at_fence(poly):
    """Exact guard cut. Yields (piece, takes_the_brake_branch)."""
    xlo, xhi, _, _ = poly.bounds()
    if xhi <= FENCE:
        return [(poly, False)]
    if xlo >= FENCE:
        return [(poly, True)]
    inside = poly.clip((1.0, 0.0), FENCE)
    outside = poly.clip((-1.0, 0.0), -FENCE)
    out = [(p, flag) for p, flag in ((inside, False), (outside, True))
           if not p.is_empty()]
    return out or [(poly, False)]


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def run(oracle, steps, v0_bound, merge=False, cap=20000, start=None):
    """Propagate exact polygons and report the peak x over the horizon.

    ``start`` overrides the initial set, so a caller can split it and
    propagate each piece separately - the mitigation a real tool applies when
    its control bound over the whole initial box is vacuous.
    """
    matrix = np.array([[1.0, DT], [0.0, 1.0]])
    sets = [start or Polygon.box(X0_LO, X0_HI, -v0_bound, v0_bound)]
    worst, peak = X0_HI, 1
    for _ in range(steps):
        nxt = []
        for poly in sets:
            for piece, braking in _split_at_fence(poly):
                if braking:
                    alo = ahi = -BRAKE
                else:
                    alo, ahi = oracle.bounds(*piece.bounds())
                nxt.append(piece.affine(matrix).sweep((0.0, DT * alo),
                                                      (0.0, DT * ahi)))
        if merge:
            nxt = [Polygon.hull(np.vstack([p.points for p in nxt]))]
        sets = nxt
        peak = max(peak, len(sets))
        worst = max(worst, max(p.bounds()[1] for p in sets))
        if len(sets) > cap:
            return "gave up", worst, peak
    return ("violated" if worst > HARD_LIMIT else "safe"), worst, peak


def main():
    """Run both modes at several refinements and print the table."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    v0_bound = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    print(f"exact polygon reachability: {steps} steps, |v0| <= {v0_bound}, "
          f"limit {HARD_LIMIT}, true max 1.2622\n")
    print(f"{'mode':>8} {'refine':>7} {'verdict':>10} {'peak x':>10} "
          f"{'sets':>8} {'sec':>7}")
    for merge in (True, False):
        for refine in (1, 2, 4, 8):
            oracle = CrownOracle(refine=refine)
            t0 = time.time()
            verdict, worst, peak = run(oracle, steps, v0_bound, merge=merge)
            print(f"{'merge' if merge else 'split':>8} {refine:>7} "
                  f"{verdict:>10} {worst:>10.4f} {peak:>8} "
                  f"{time.time() - t0:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
