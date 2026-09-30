"""
Constrained zonotopes: the correlation-preserving guard split, without MATLAB.

CORA's decisive advantage over our earlier baselines is that constrained
zonotopes are CLOSED under halfspace intersection, so cutting a set at a guard
loses no correlation between x and v.  Our zonotope_box.py re-fits an
axis-aligned box to each piece and therefore throws that correlation away.

CORA needs the MATLAB Optimization Toolbox for these operations and only base
MATLAB is licensed on this machine, so this implements the construction
directly.  It follows Scott, Raimondo, Marseglia and Braatz, "Constrained
zonotopes: A new tool for set-based estimation and fault detection"
(Automatica 69, 2016) - Proposition 1 for the halfspace intersection.

    CZ = { c + G xi : A xi = b , ||xi||_inf <= 1 }

Support function in direction d is the linear program
    max d' (c + G xi)   s.t.  A xi = b ,  -1 <= xi <= 1
which scipy solves exactly, so the interval hull is tight rather than estimated.

If a set representation that keeps correlation through the cut STILL fails on
the geofence, the pilot's headline is as strong as it can be made without a
MATLAB licence.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


class ConZonotope:
    """{ c + G xi : A xi = b, |xi| <= 1 }, closed under halfspace intersection."""

    def __init__(self, c, g, a=None, b=None):
        self.c = np.asarray(c, float).reshape(-1)
        self.g = np.asarray(g, float).reshape(len(self.c), -1)
        n = self.g.shape[1]
        self.a = np.zeros((0, n)) if a is None else np.asarray(a, float).reshape(-1, n)
        self.b = np.zeros(0) if b is None else np.asarray(b, float).reshape(-1)

    def support(self, direction, maximise=True):
        """Exact support value via an LP over the generator coordinates."""
        d = np.asarray(direction, float).reshape(-1)
        cost = d @ self.g
        res = linprog(-cost if maximise else cost,
                      A_eq=self.a if self.a.size else None,
                      b_eq=self.b if self.b.size else None,
                      bounds=[(-1.0, 1.0)] * self.g.shape[1],
                      method="highs")
        if not res.success:
            return None                      # infeasible: the set is empty
        return float(d @ self.c + cost @ res.x)

    def interval(self):
        """Tight interval hull, computed by 2n support evaluations."""
        lo, hi = [], []
        for i in range(len(self.c)):
            e = np.zeros(len(self.c))
            e[i] = 1.0
            top = self.support(e, True)
            bot = self.support(e, False)
            if top is None or bot is None:
                return None
            lo.append(bot)
            hi.append(top)
        return np.array(lo), np.array(hi)

    def is_empty(self):
        """Empty iff the support LP is infeasible."""
        e = np.zeros(len(self.c))
        e[0] = 1.0
        return self.support(e) is None

    def linear_map(self, matrix, offset_centre, offset_gen):
        """A*S + (an interval input): exact for the linear part."""
        m = np.asarray(matrix, float)
        c = m @ self.c + np.asarray(offset_centre, float).reshape(-1)
        g = np.hstack([m @ self.g,
                       np.asarray(offset_gen, float).reshape(len(self.c), -1)])
        a = np.hstack([self.a, np.zeros((self.a.shape[0], 1))]) \
            if self.a.size else np.zeros((0, g.shape[1]))
        return ConZonotope(c, g, a, self.b)

    def intersect_halfspace(self, h, k):
        """S cap { x : h'x <= k }, EXACTLY (Scott et al. 2016, Prop. 1).

        A slack generator absorbs the slice, and one equality constraint ties
        it to the halfspace. Nothing is projected away, which is the whole
        point: the correlation between x and v survives the cut.
        """
        h = np.asarray(h, float).reshape(-1)
        lo = self.support(h, maximise=False)
        hi = self.support(h, maximise=True)
        if lo is None:
            return self
        if hi <= k:
            return self                                    # already inside
        if lo > k:
            return None                                    # empty
        span = (k - lo) / 2.0
        centre = (k + lo) / 2.0
        g = np.hstack([self.g, np.zeros((len(self.c), 1))])
        a_new = np.hstack([h @ self.g, [span]])
        a = np.vstack([np.hstack([self.a, np.zeros((self.a.shape[0], 1))]),
                       a_new]) if self.a.size else a_new.reshape(1, -1)
        b = np.concatenate([self.b, [centre - h @ self.c]])
        return ConZonotope(self.c, g, a, b)
