"""Finite escape by stepwise reachability over (box x TREE MODE).

WHY NOT THE UNROLLED SMT ENCODING. Two independent measurements say it cannot
close this benchmark, and they agree.
  - CROWN's bound gap grows superlinearly with cell size: half-width 0.012
    gives a gap of 0.011, half-width 0.060 gives 0.078, half-width 0.600 gives
    14.8 against an actuator limit of 2.0. Useful contracts therefore need
    cells of half-width about 0.06, and the Return corridor is 0.6 wide in
    four dimensions - of order 10^4 cells.
  - The unrolled encoding goes `unknown` at 256 cells with a horizon of 6, and
    at 16 cells with a horizon of 14. Per-cell contracts make the solver choose
    WHICH cell at EVERY tick, so the search grows like cells^horizon.
Those two numbers do not meet: the cell count tightness demands is three orders
of magnitude past the cell count the solver tolerates.

WHAT THIS DOES INSTEAD. Propagate a set of boxes one tick at a time, re-boxing
after each step. Cost is then LINEAR in boxes and in horizon rather than
exponential, because nothing has to be decided jointly across ticks.

AND WHY IT IS NOT JUST REACHABILITY. The set being propagated is not a box, it
is a box PAIRED WITH THE TREE'S STATE - whether Brake has handed over, whether
Recovery has succeeded. Those are latching, so they are memory, not a function
of the continuous state, and a reachability tool has no notion of them. Every
tick the guards `slow` and `home` split each box, and because both are
axis-aligned boxes the pieces are still boxes and the partition stays exact.
The tree decides which controller each piece is propagated under, which is the
part of this problem no reachability tool derives for you.
"""
from __future__ import annotations

import numpy as np

from plant import A_MAX, DRAG, DT, START
from verify import V_STOP


class Region:
    """A box of continuous states together with the tree's discrete state."""

    __slots__ = ("box", "handed", "done")

    def __init__(self, box, handed=False, done=False):
        self.box = tuple(float(v) for v in box)
        self.handed, self.done = handed, done

    def __repr__(self):
        return (f"Region({[round(v, 3) for v in self.box]}, "
                f"handed={self.handed}, done={self.done})")

    def width(self, axis):
        """Full width along one axis."""
        return self.box[2 * axis + 1] - self.box[2 * axis]

    def clip_to(self, axis, lo, hi):
        """This region intersected with a slab, or None if they miss."""
        new_lo = max(self.box[2 * axis], lo)
        new_hi = min(self.box[2 * axis + 1], hi)
        if new_lo > new_hi:
            return None
        box = list(self.box)
        box[2 * axis], box[2 * axis + 1] = new_lo, new_hi
        return Region(box, self.handed, self.done)

    def split(self, axis):
        """Halve along one axis."""
        mid = 0.5 * (self.box[2 * axis] + self.box[2 * axis + 1])
        halves = []
        for lo, hi in ((self.box[2 * axis], mid), (mid, self.box[2 * axis + 1])):
            box = list(self.box)
            box[2 * axis], box[2 * axis + 1] = lo, hi
            halves.append(Region(box, self.handed, self.done))
        return halves


def _partition(region, axis, lo, hi):
    """Split a region into the part inside the slab and the parts outside."""
    inside = region.clip_to(axis, lo, hi)
    outside = []
    for low, high in ((-np.inf, lo), (hi, np.inf)):
        piece = region.clip_to(axis, low, high)
        if piece is not None and piece.width(axis) > 0:
            outside.append(piece)
    if inside is not None and inside.width(axis) == 0 and outside:
        # a zero-width face is already represented in the neighbouring pieces
        pass
    return inside, outside


def by_predicate(regions, slabs):
    """Split every region by a conjunction of slabs; returns (holds, fails).

    `slabs` is a list of (axis, lo, hi). Because every guard in this tree is an
    axis-aligned box, the pieces are exactly representable and nothing is lost
    here - the over-approximation in this method lives entirely in the one-step
    propagation, not in the guards.
    """
    holds, fails = [], []
    for region in regions:
        current = [region]
        for axis, lo, hi in slabs:
            survivors = []
            for piece in current:
                inside, outside = _partition(piece, axis, lo, hi)
                fails.extend(outside)
                if inside is not None:
                    survivors.append(inside)
            current = survivors
        holds.extend(current)
    return holds, fails


HOME_SLABS = ((0, START[0], START[1]), (1, START[2], START[3]))
SLOW_SLABS = ((2, -V_STOP, V_STOP), (3, -V_STOP, V_STOP))


def refine(regions, max_width, cap=200000):
    """Bisect until every region is narrow enough for a tight contract.

    The threshold is not arbitrary: CROWN's gap over this network is 0.011 at
    half-width 0.012 and 14.8 at half-width 0.6, so the bound is only worth
    having below roughly 0.06. Velocity axes are held to a quarter of the
    position threshold because the success test on velocity is twenty times
    tighter than the one on position.
    """
    out = []
    queue = list(regions)
    while queue:
        region = queue.pop()
        limits = [max_width, max_width, max_width / 4.0, max_width / 4.0]
        widest = max(range(4), key=lambda d: region.width(d) / limits[d])
        if region.width(widest) > limits[widest] and len(out) + len(queue) < cap:
            queue.extend(region.split(widest))
        else:
            out.append(region)
    return out


def _brake_bounds(box):
    """Exact acceleration interval for Brake over a box, saturation included.

    `clip(-3v)` is monotone decreasing in v, so the image of an interval is the
    interval of the images and nothing is over-approximated here.
    """
    bounds = []
    for axis in (0, 1):
        lo_v, hi_v = box[4 + 2 * axis], box[5 + 2 * axis]
        pair = sorted((-3.0 * hi_v, -3.0 * lo_v))
        bounds.append(tuple(max(-A_MAX, min(A_MAX, v)) for v in pair))
    return bounds


def _contract_bounds(box, contract):
    """Concretise an affine contract over the box it was taken on."""
    bounds = []
    for axis in (0, 1):
        low_a, low_b, up_a, up_b = contract[axis]
        low = low_b + sum(min(low_a[d] * box[2 * d], low_a[d] * box[2 * d + 1])
                          for d in range(4))
        high = up_b + sum(max(up_a[d] * box[2 * d], up_a[d] * box[2 * d + 1])
                          for d in range(4))
        bounds.append((max(low, -A_MAX), min(high, A_MAX)))
    return bounds


def _advance(region, bounds):
    """One tick of the plant applied to a box, given acceleration intervals."""
    box = region.box
    out = [0.0] * 8
    for axis in (0, 1):
        lo_p, hi_p = box[2 * axis], box[2 * axis + 1]
        lo_v, hi_v = box[4 + 2 * axis], box[5 + 2 * axis]
        lo_a, hi_a = bounds[axis]
        out[2 * axis], out[2 * axis + 1] = lo_p + lo_v * DT, hi_p + hi_v * DT
        # 1 - DRAG*DT > 0, so the velocity update is increasing in v
        out[4 + 2 * axis] = lo_v + (lo_a - DRAG * lo_v) * DT
        out[5 + 2 * axis] = hi_v + (hi_a - DRAG * hi_v) * DT
    return Region(out, region.handed, region.done)


def tick(regions, source, max_width, step_pos=0.05, step_vel=0.02):
    """One tick of the tree applied to a set of (box, mode) regions.

    The flag updates mirror `verify.Encoding._tick` exactly, including its one
    conservatism: the plant lets Brake hand over and Return succeed inside a
    single tick, while this takes two. That makes the horizon reported here an
    upper bound on the plant's, which is the direction a bound must err in.
    """
    regions = snap(refine(regions, max_width), step_pos, step_vel)
    slow, moving = by_predicate(regions, SLOW_SLABS)
    at_home, away = by_predicate(slow, HOME_SLABS)

    pending = []
    for region in moving:                    # not slow: flags unchanged
        pending.append((region, region.handed, region.done))
    for region in away:                      # slow, not home: Brake hands over
        pending.append((region, True, region.done))
    for region in at_home:                   # slow and home: Recovery may end
        pending.append((region, True, region.done or region.handed))

    needs_contract = [r for r, _, _ in pending if r.handed and not r.done]
    contracts = dict(zip((id(r) for r in needs_contract),
                         (c for _, c in source([r.box for r in needs_contract]))
                         )) if needs_contract else {}

    escaped, carried = [], []
    for region, handed, done in pending:
        if region.done:                      # already out of Omega_i
            escaped.append(region)
            continue
        if not region.handed:
            bounds = _brake_bounds(region.box)
        elif not region.done:
            bounds = _contract_bounds(region.box, contracts[id(region)])
        else:
            bounds = ((0.0, 0.0), (0.0, 0.0))
        nxt = _advance(region, bounds)
        nxt.handed, nxt.done = handed, done
        (escaped if done else carried).append(nxt)
    return carried, escaped


def escape_horizon(start, source, max_width=0.06, limit=60,
                   step_pos=0.05, step_vel=0.02, verbose=True):
    """Ticks until EVERY region has left the operating region, or None.

    Returning a number is a proof of the finite-escape hypothesis over the
    whole start set, because every step over-approximates: the guards are
    exact, the contracts are sound relaxations of the network, and the one-step
    box propagation contains the true image.
    """
    regions = [start] if isinstance(start, Region) else list(start)
    for step in range(1, limit + 1):
        regions, escaped = tick(regions, source, max_width,
                                step_pos, step_vel)
        if verbose:
            print(f"  tick {step:>3}  {len(regions):>7} live  "
                  f"{len(escaped):>6} escaped this tick", flush=True)
        if not regions:
            return step
    return None


def snap(regions, step_pos, step_vel):
    """Cover every region by cells of a FIXED grid, then deduplicate.

    Without this the set grows without bound - splitting every region each tick
    took it from 1,280 to 33,024 to 204,055 in three ticks, because nothing
    ever recombined two pieces that had come back together. Snapping outward to
    a shared grid is a sound over-approximation (each region is contained in
    the cells covering it) and it makes the representation CONVERGE: two
    trajectories that reach the same grid cell in the same tree state become
    one region again, so the count is bounded by the reachable part of the grid
    rather than by the number of splits performed.
    """
    steps = (step_pos, step_pos, step_vel, step_vel)
    seen = {}
    for region in regions:
        spans = []
        for axis in range(4):
            size = steps[axis]
            lo = int(np.floor(region.box[2 * axis] / size))
            hi = int(np.floor(region.box[2 * axis + 1] / size))
            spans.append(range(lo, hi + 1))
        for i in spans[0]:
            for j in spans[1]:
                for m in spans[2]:
                    for n in spans[3]:
                        key = (i, j, m, n, region.handed, region.done)
                        if key not in seen:
                            box = []
                            for axis, idx in enumerate((i, j, m, n)):
                                size = steps[axis]
                                box.extend((idx * size, (idx + 1) * size))
                            seen[key] = Region(box, region.handed, region.done)
    return list(seen.values())
