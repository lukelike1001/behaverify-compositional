"""Zonotope propagation of (set x TREE MODE), which is what actually closes.

WHY A BOX IS NOT ENOUGH, measured rather than asserted. Near home the
closed loop on one axis is

    A = [[1, dt], [-6.4 dt, 1 - drag*dt - 4 dt]]

whose eigenvalues are 0.795 +- 0.148i, spectral radius 0.8087 - contracting.
But ||A||_inf = 1.23, and interval arithmetic grows by the row sums of |A|,
not by the spectral radius. Propagating the SAME map 24 ticks from a width of
0.02:

    interval arithmetic   0.02000 -> 0.29786     (x 14.9, diverging)
    correlations kept     0.02000 -> 0.00028     (x 0.014, contracting)

a factor of a thousand apart. That is the wrapping effect, and it is why the
box version of this ran 1,280 -> 33,024 -> 204,055 regions and why grid
snapping only slowed it to 14,144 -> 94,533. The eigenvalues are complex, so
the set ROTATES, and no axis-aligned box can follow a rotating set.

AND THIS IS THE REAL ARGUMENT FOR AFFINE CONTRACTS. The matrix A above exists
only because the contract carries a SLOPE: A's bottom row is built from the
contract's coefficients on p and v. Concretise the contract to an interval -
which is what the earlier pipeline did with CROWN's output - and the slope is
zero, A becomes [[1, dt], [0, 1 - drag*dt]], which has spectral radius 1 and
does not contract at all. An interval contract cannot express a closed loop
that converges. Tightness was never the main reason to keep lA and uA; the
reason is that they COMPOSE with an affine set representation.
"""
from __future__ import annotations

import numpy as np

from plant import A_MAX, DRAG, DT, START
from verify import V_STOP

MAX_GENERATORS = 24


class Zono:
    """center + generators, paired with the tree's latching discrete state.

    The point set is {c + G e : |e|_inf <= 1}. `handed` and `done` are the
    tree's memory and are NOT functions of the continuous state, which is
    exactly what a reachability tool cannot carry for you.
    """

    __slots__ = ("c", "G", "handed", "done")

    def __init__(self, c, G, handed=False, done=False):
        self.c = np.asarray(c, dtype=float).reshape(4)
        self.G = np.asarray(G, dtype=float).reshape(4, -1)
        self.handed, self.done = handed, done

    @classmethod
    def from_box(cls, box, handed=False, done=False):
        """The zonotope of an axis-aligned box."""
        lo = np.array(box[0::2], dtype=float)
        hi = np.array(box[1::2], dtype=float)
        return cls(0.5 * (lo + hi), np.diag(0.5 * (hi - lo)), handed, done)

    def radius(self):
        """Half-widths of the bounding box."""
        return np.abs(self.G).sum(axis=1)

    def box(self):
        """Interval hull, as the flat tuple the contract generators expect."""
        rad = self.radius()
        return tuple(float(v) for d in range(4)
                     for v in (self.c[d] - rad[d], self.c[d] + rad[d]))

    def reduce(self, keep=MAX_GENERATORS):
        """Absorb the smallest generators into an aligned box.

        Generator count grows by two a tick, so without this the representation
        becomes the cost. Absorbing the SMALLEST ones keeps the directions that
        carry the set's shape and over-approximates only the rest.
        """
        if self.G.shape[1] <= keep:
            return self
        order = np.argsort(np.abs(self.G).sum(axis=0))
        small, large = order[:self.G.shape[1] - keep + 4], order[len(order) - keep + 4:]
        box = np.diag(np.abs(self.G[:, small]).sum(axis=1))
        return Zono(self.c, np.hstack([self.G[:, large], box]),
                    self.handed, self.done)


def bisect(z, axis):
    """Split along the generator most aligned with `axis`. EXACT, not a relaxation.

    Halving one factor's range splits the point set in two with no loss, which
    is what makes guard handling exact here: a box intersection would have to
    throw away the correlations the whole method exists to keep.
    """
    index = int(np.argmax(np.abs(z.G[axis])))
    column = z.G[:, index]
    halved = z.G.copy()
    halved[:, index] = 0.5 * column
    return (Zono(z.c - 0.5 * column, halved, z.handed, z.done),
            Zono(z.c + 0.5 * column, halved, z.handed, z.done))


def classify(zonos, slabs, depth=6):
    """Split each zonotope until every piece is decided by `slabs`.

    `slabs` is a list of (axis, lo, hi), always an axis-aligned box because
    every guard in this tree is one. A piece still straddling a boundary after
    `depth` splits is sent to BOTH sides, which over-approximates and so keeps
    the result sound.
    """
    inside, outside = [], []
    queue = [(z, 0) for z in zonos]
    while queue:
        z, level = queue.pop()
        rad = z.radius()
        verdicts = []
        for axis, lo, hi in slabs:
            low, high = z.c[axis] - rad[axis], z.c[axis] + rad[axis]
            verdicts.append(True if low >= lo and high <= hi
                            else False if high < lo or low > hi else None)
        if all(v is True for v in verdicts):
            inside.append(z)
        elif any(v is False for v in verdicts):
            outside.append(z)
        elif level >= depth:
            inside.append(z)
            outside.append(z)
        else:
            axis = next(a for (a, _, _), v in zip(slabs, verdicts) if v is None)
            queue.extend((piece, level + 1) for piece in bisect(z, axis))
    return inside, outside


def _affine_control(z, contract):
    """Mid-slope matrix, offset and radius of the control over the zonotope.

    The contract gives lA.s + lb <= a <= uA.s + ub. Writing the midpoint as an
    affine function and the half-gap as a bounded disturbance keeps the SLOPE
    in the linear map - which is the whole point - and puts only the gap into a
    new generator.
    """
    slope = np.zeros((2, 4))
    offset = np.zeros(2)
    radius = np.zeros(2)
    for axis in (0, 1):
        low_a, low_b, up_a, up_b = contract[axis]
        low_a, up_a = np.asarray(low_a), np.asarray(up_a)
        mid_a, mid_b = 0.5 * (low_a + up_a), 0.5 * (low_b + up_b)
        spread = 0.5 * ((up_a - low_a) @ z.c + (up_b - low_b))
        spread += 0.5 * np.abs((up_a - low_a) @ z.G).sum()
        # The contract bounds the NETWORK's output; the actuator clips it. A
        # containment check caught this: where CROWN's bound lies entirely
        # above A_MAX the clipped value the plant applies is OUTSIDE the
        # interval, and 10 of 4,972 simulated states fell out of the
        # propagated set. Clipping is monotone, so the image of the interval is
        # the interval of the images - but only the UNSATURATED case keeps the
        # slope, so saturation falls back to a tight interval. `split_to` makes
        # the pieces small enough that this costs little.
        centre = mid_a @ z.c + mid_b
        swing = float(np.abs(mid_a @ z.G).sum())
        reach = swing + abs(spread)
        # Clipping is monotone, so clip(a) stays inside [m(s) - r, m(s) + r]
        # unless that whole interval lies outside [-A_MAX, A_MAX] - only then
        # does the plant apply a value the contract does not contain. Testing
        # the looser `centre +- reach within the limit` instead dropped the
        # slope wherever the bound merely TOUCHED saturation, and the lost
        # slope let pieces grow, which forced more splitting, which is how the
        # live count went from a peak of 3,597 to 37,479.
        if (centre + swing - abs(spread) <= A_MAX
                and centre - swing + abs(spread) >= -A_MAX):
            slope[axis], offset[axis] = mid_a, mid_b
            radius[axis] = abs(spread)
        else:
            low = max(-A_MAX, min(A_MAX, centre - reach))
            high = max(-A_MAX, min(A_MAX, centre + reach))
            offset[axis] = 0.5 * (low + high)
            radius[axis] = 0.5 * (high - low)
    return slope, offset, radius


def _brake_control(z):
    """Brake as an affine map where it does not saturate, interval where it does."""
    slope = np.zeros((2, 4))
    offset = np.zeros(2)
    radius = np.zeros(2)
    rad = z.radius()
    for axis in (0, 1):
        vel, half = z.c[2 + axis], rad[2 + axis]
        if 3.0 * (abs(vel) + half) <= A_MAX:          # no saturation anywhere
            slope[axis, 2 + axis] = -3.0
        else:
            lo = max(-A_MAX, min(A_MAX, -3.0 * (vel + half)))
            hi = max(-A_MAX, min(A_MAX, -3.0 * (vel - half)))
            offset[axis], radius[axis] = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return slope, offset, radius


def advance(z, slope, offset, radius):
    """One tick of the plant as an exact affine image plus a disturbance."""
    matrix = np.zeros((4, 4))
    matrix[0, 0] = matrix[1, 1] = 1.0
    matrix[0, 2] = matrix[1, 3] = DT
    matrix[2, 2] = matrix[3, 3] = 1.0 - DRAG * DT
    matrix[2:4, :] += DT * slope
    centre = matrix @ z.c + np.concatenate([[0.0, 0.0], DT * offset])
    noise = np.zeros((4, 2))
    noise[2, 0], noise[3, 1] = DT * radius[0], DT * radius[1]
    keep = noise[:, np.abs(noise).sum(axis=0) > 0]
    gens = np.hstack([matrix @ z.G, keep]) if keep.size else matrix @ z.G
    return Zono(centre, gens, z.handed, z.done)


HOME_SLABS = ((0, START[0], START[1]), (1, START[2], START[3]))
SLOW_SLABS = ((2, -V_STOP, V_STOP), (3, -V_STOP, V_STOP))


def tick(zonos, source, depth=4):
    """One tick of the tree over a set of (zonotope, mode) pairs.

    The flag updates mirror `verify.Encoding._tick` exactly, including its one
    conservatism: the plant may hand over and succeed within a single tick
    while this takes two, so the horizon here is an upper bound on the plant's.
    """
    slow, moving = classify(split_to(zonos), SLOW_SLABS, depth)
    at_home, away = classify(slow, HOME_SLABS, depth)

    pending = [(z, z.handed, z.done) for z in moving]
    pending += [(z, True, z.done) for z in away]
    pending += [(z, True, z.done or z.handed) for z in at_home]

    returning = [z for z, _, _ in pending if z.handed and not z.done]
    contracts = {}
    if returning:
        boxes = [z.box() for z in returning]
        contracts = {id(z): c for z, (_, c) in zip(returning, source(boxes))}

    live, escaped = [], []
    for z, handed, done in pending:
        if z.done:
            escaped.append(z)
            continue
        if not z.handed:
            control = _brake_control(z)
        else:
            control = _affine_control(z, contracts[id(z)])
        nxt = advance(z, *control).reduce()
        nxt.handed, nxt.done = handed, done
        (escaped if done else live).append(nxt)
    return live, escaped


def escape_horizon(start, source, limit=60, depth=4, verbose=True):
    """Ticks until every piece has left the operating region, or None."""
    zonos = [start] if isinstance(start, Zono) else list(start)
    for step in range(1, limit + 1):
        zonos, escaped = tick(zonos, source, depth)
        if verbose:
            widest = max((z.radius().max() for z in zonos), default=0.0)
            print(f"  tick {step:>3}  {len(zonos):>6} live  {len(escaped):>5} "
                  f"escaped  widest radius {widest:.4f}", flush=True)
        if not zonos:
            return step
    return None


# CROWN's gap over this network against cell half-width: 0.012 -> 0.011,
# 0.060 -> 0.078, 0.150 -> 1.114, 0.600 -> 14.799 against A_MAX = 2.0. Pieces
# wider than about 0.06 therefore carry a contract worth nothing, which is why
# propagating B1 whole let the set grow before it could contract.
TARGET_POS, TARGET_VEL = 0.06, 0.05


def split_to(zonos, pos=TARGET_POS, vel=TARGET_VEL, cap=20000):
    """Bisect until every piece is small enough for a useful contract.

    Bisection is exact, so this costs only piece count - no accuracy. It is the
    counterpart of `reduce`, which trades accuracy for generator count.
    """
    limits = (pos, pos, vel, vel)
    out, queue = [], list(zonos)
    while queue:
        z = queue.pop()
        rad = z.radius()
        axis = max(range(4), key=lambda d: rad[d] / limits[d])
        if rad[axis] > limits[axis] and len(out) + len(queue) < cap:
            queue.extend(bisect(z, axis))
        else:
            out.append(z)
    return out
