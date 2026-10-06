"""Sound AFFINE contracts for `divert`, and the interval arithmetic behind them.

WHY AFFINE AND NOT INTERVAL. The property `escapes_within` asks the recovery
branch to reach a home box AND come to rest, which is set-point regulation. The
thing that makes a regulator converge is the SLOPE of the control with respect
to the state: `return_law` is a PD law, a_x = -k_p (x - x_home) - k_d vx. An
interval contract `guard -> ax in [lo, hi]` discards that slope, so inside any
cell the solver may pick a CONSTANT acceleration, and a constant acceleration
cannot be forced to regulate to a point. Shrinking the cell shrinks the
interval but never recovers the slope, so the failure is INVARIANT UNDER
REFINEMENT - which is precisely the monotone-failure signature measured with
interval contracts (576 cells no proof 28.9s, 2304 cells no proof 227.5s).

An affine contract `lA.s + lb <= a <= uA.s + ub` keeps the slope and is still
linear real arithmetic, so Z3 stays in LRA rather than dropping into nlsat.

WHY THIS IS NOT A NEW DEMAND ON THE NETWORK ORACLE. alpha-beta-CROWN's
`compute_bounds` natively returns exactly this shape - a coefficient matrix and
a bias - and the previous pipeline concretized it to an interval at the contract
boundary and threw the coefficients away. Affine contracts consume what CROWN
already produces.

THE CONSTRUCTION. Fix the coefficient vector A a priori, then compute the bias
by sound interval evaluation of the residual r(s) = f(s) - A.s over the cell:

    A.s + inf r  <=  f(s)  <=  A.s + sup r        for all s in the cell

This is sound for ANY f and any choice of A, because it is just the definition
of the residual bounded by its own range. It is exact (zero width) when f is
affine with coefficients A. Choosing A badly costs tightness, never soundness,
so the a-priori choice does not beg the question: a wrong A degrades to the
interval contract it replaces.
"""
from __future__ import annotations

import math

from plant import (A_MAX, OBSTACLE, STANDOFF, START, V_RETURN, obstacle_distance)

HOME = ((START[0] + START[1]) / 2, (START[2] + START[3]) / 2)

# The a-priori slope. 6.4 is the far-branch position gain (4.0 * 1.6, the gain
# that appears once `speed = min(V_RETURN, 1.6 * span)` takes its unsaturated
# branch and `span` cancels); 4.0 is the far-branch velocity gain. The near
# branch uses 6.0 and 4.0, close enough that the residual stays narrow across
# the switch instead of blowing up at it.
SLOPE_POS, SLOPE_VEL = -6.4, -4.0


class Ivl:
    """A closed real interval with the operations this module needs.

    Every operation returns an interval CONTAINING the true result set, so a
    bound derived from these is an over-approximation and a proof under it
    holds for the real system.
    """

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = float(lo)
        self.hi = float(lo if hi is None else hi)

    def __repr__(self):
        return f"[{self.lo:.4f}, {self.hi:.4f}]"

    def __add__(self, other):
        other = _as_ivl(other)
        return Ivl(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Ivl(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-_as_ivl(other))

    def __rsub__(self, other):
        return _as_ivl(other) + (-self)

    def __mul__(self, other):
        other = _as_ivl(other)
        corners = (self.lo * other.lo, self.lo * other.hi,
                   self.hi * other.lo, self.hi * other.hi)
        return Ivl(min(corners), max(corners))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _as_ivl(other)
        if other.lo <= 0.0 <= other.hi:
            raise ZeroDivisionError("interval division straddling zero")
        return self * Ivl(1.0 / other.hi, 1.0 / other.lo)

    def __rtruediv__(self, other):
        return _as_ivl(other) / self

    def hull(self, other):
        """Smallest interval containing both. Used to merge switch branches."""
        return Ivl(min(self.lo, other.lo), max(self.hi, other.hi))

    def abs_range(self):
        """Range of |v| over the interval, as (min, max)."""
        if self.lo <= 0.0 <= self.hi:
            return 0.0, max(-self.lo, self.hi)
        return min(abs(self.lo), abs(self.hi)), max(abs(self.lo), abs(self.hi))


def _as_ivl(value):
    return value if isinstance(value, Ivl) else Ivl(value)


def hypot_ivl(a, b):
    """Sound range of hypot(a, b) over two independent intervals."""
    alo, ahi = a.abs_range()
    blo, bhi = b.abs_range()
    return Ivl(math.hypot(alo, blo), math.hypot(ahi, bhi))


def min_ivl(cap, a):
    """Sound range of min(cap, a) for a scalar cap."""
    return Ivl(min(cap, a.lo), min(cap, a.hi))


# --- the obstacle term, bounded soundly over a cell ------------------------

def _gap_range(box):
    """Sound range of `obstacle_distance` over the cell's position extent.

    The maximum uses the fact that distance to a convex set is a CONVEX
    function, so its maximum over a box is attained at a vertex; the minimum is
    the exact box-to-box separation.
    """
    xlo, xhi, ylo, yhi = box[:4]
    oxlo, oxhi, oylo, oyhi = OBSTACLE
    dx = max(oxlo - xhi, 0.0, xlo - oxhi)
    dy = max(oylo - yhi, 0.0, ylo - oyhi)
    lo = math.hypot(dx, dy)
    if lo == 0.0:                     # the cell touches the obstacle
        lo = -0.5 * min(oxhi - oxlo, oyhi - oylo)
    hi = max(obstacle_distance(px, py)
             for px in (xlo, xhi) for py in (ylo, yhi))
    return lo, hi


def _avoid_range(box, strength=8.0):
    """Sound interval for ONE component of `avoid` over the cell.

    Both components share the bound: the unit vector p has norm 1, so each of
    `0.4 px -+ py` is at most hypot(0.4, 1) in magnitude whatever the side
    choice is, which also disposes of the sign discontinuity in `side`.
    """
    gap_lo, gap_hi = _gap_range(box)
    if gap_lo >= STANDOFF:
        return Ivl(0.0)                        # exactly off over the whole cell
    urgency_hi = strength * (STANDOFF - gap_lo)
    reach = urgency_hi * math.hypot(0.4, 1.0)
    return Ivl(-reach, reach)                  # hull already contains 0


# --- the residual of `return_law` against the a-priori slope ---------------

def _residual_axis(box, axis):
    """Interval of r(s) = return_law_axis(s) - (SLOPE_POS*p + SLOPE_VEL*v).

    The affine part is cancelled SYMBOLICALLY, branch by branch, before any
    interval is taken. Subtracting two intervals instead would lose the
    correlation between f and A.s and leave the residual exactly as wide as f
    itself, which is the interval contract this module exists to replace.
    """
    plo, phi = (box[0], box[1]) if axis == 0 else (box[2], box[3])
    home = HOME[axis]
    d = Ivl(home - phi, home - plo)            # d = home - p
    other = Ivl(HOME[1 - axis] - box[3], HOME[1 - axis] - box[2]) if axis == 0 \
        else Ivl(HOME[0] - box[1], HOME[0] - box[0])
    span = hypot_ivl(d, other)

    parts = []
    if span.lo < 0.15:                         # settle branch: 6 d - 4 v
        # 6 d - 4 v - (-6.4 p - 4 v) = 0.4 p + 6 home
        parts.append(Ivl(0.4 * plo, 0.4 * phi) + 6.0 * home)
    if span.hi >= 0.15:
        cross = V_RETURN / 1.6                 # speed saturates above this span
        if span.lo < cross:                    # 6.4 d - 4 v: the slope is EXACT
            parts.append(Ivl(6.4 * home))
        if span.hi >= cross:                   # 3.8 d/span - 4 v
            sat = Ivl(max(span.lo, cross), span.hi)
            parts.append(d * (3.8 / sat - 6.4) + 6.4 * home)
    out = parts[0]
    for part in parts[1:]:
        out = out.hull(part)
    return out + _avoid_range(box)


def _unclipped_axis(box, axis):
    """Interval of the UNCLIPPED law, needed only to test for saturation."""
    vlo, vhi = (box[4], box[5]) if axis == 0 else (box[6], box[7])
    plo, phi = (box[0], box[1]) if axis == 0 else (box[2], box[3])
    return _residual_axis(box, axis) + Ivl(SLOPE_POS * phi, SLOPE_POS * plo) \
        + Ivl(SLOPE_VEL * vhi, SLOPE_VEL * vlo)


def affine_contract(box):
    """Contract for one cell, in CROWN's native shape.

    Returns {axis: (lA, lb, uA, ub)} with lA, uA in R^4 over s = (x, y, vx, vy),
    meaning   lA . s + lb  <=  a_axis(s)  <=  uA . s + ub   for all s in cell.

    lA and uA are kept INDEPENDENT because that is what `compute_bounds`
    returns: CROWN's lower and upper relaxations are different hyperplanes, and
    forcing them equal would collapse a convex polytope into a parallel strip
    and throw away tightness for nothing. The analytic placeholder happens to
    produce lA == uA wherever the law is affine, which is a property of this
    particular law and not a constraint of the interface.

    Saturation is handled per axis. `_clip` is now a per-axis box, which is
    MONOTONE in its own argument, so the image of an interval under it is
    exactly the interval of the images - a tight interval contract rather than
    a relaxed one. Where an axis saturates anywhere in the cell the slope is
    dropped and the contract degrades to that tight interval; where it does not,
    the slope survives untouched.
    """
    out = {}
    for axis in (0, 1):
        residual = _residual_axis(box, axis)
        unclipped = _unclipped_axis(box, axis)
        if -A_MAX <= unclipped.lo and unclipped.hi <= A_MAX:
            slope = [0.0, 0.0, 0.0, 0.0]
            slope[axis] = SLOPE_POS
            slope[axis + 2] = SLOPE_VEL
            out[axis] = (slope, residual.lo, list(slope), residual.hi)
        else:
            lo = max(-A_MAX, min(A_MAX, unclipped.lo))
            hi = max(-A_MAX, min(A_MAX, unclipped.hi))
            zero = [0.0, 0.0, 0.0, 0.0]
            out[axis] = (zero, lo, list(zero), hi)
    return out


GRID = 256.0            # coefficients are rounded to multiples of 1/GRID


def coarsen(cell, contract, grid=GRID):
    """Round a contract's coefficients to small rationals, OUTWARD.

    CROWN returns float32 coefficients. Z3 reads each one as an exact rational
    with a denominator near 2^24, and linear arithmetic over rationals that
    wide is ruinous - the CROWN-backed loop timed out at 22 cells and a horizon
    of 12, where the analytic contracts were running hundreds of cells at 35.

    Rounding the SLOPE alone would not be sound, so the bias absorbs the
    difference: replacing lA by lA' costs at most max over the cell of
    (lA' - lA) . s, which is computed exactly here from the cell's corners and
    subtracted. The contract after rounding is therefore implied by the one
    before it, and still a valid relaxation of the network.
    """
    out = {}
    for axis, (low_a, low_b, up_a, up_b) in contract.items():
        new_low = [math.floor(c * grid) / grid for c in low_a]
        new_up = [math.ceil(c * grid) / grid for c in up_a]
        # worst case of (new - old) . s over the cell, corner by corner
        low_slack = sum(max((n - o) * cell[2 * d], (n - o) * cell[2 * d + 1])
                        for d, (n, o) in enumerate(zip(new_low, low_a)))
        up_slack = sum(max((o - n) * cell[2 * d], (o - n) * cell[2 * d + 1])
                       for d, (n, o) in enumerate(zip(new_up, up_a)))
        out[axis] = (new_low, math.floor((low_b - low_slack) * grid) / grid,
                     new_up, math.ceil((up_b + up_slack) * grid) / grid)
    return out
