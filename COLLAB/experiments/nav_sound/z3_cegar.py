"""
Demand-driven contract refinement for NAV, in Z3.

Every model so far emits a contract for EVERY cell of the 4-D box, but the
system only ever visits a thin tube: at grid 16/16/16 the true trajectories
touch 74 of 65,536 cells.  Interval propagation cannot find that tube soundly
(it swells to cover everything within ten steps), so instead we let the SOLVER
tell us which cells matter.

  1. Emit contracts for an ACTIVE set of cells only.  Cells outside it are
     unconstrained apart from the global tanh bound, so the abstraction is
     COARSER than the fully-emitted one, never finer.  That direction is what
     makes this sound: UNSAT at any iteration is a real proof.
  2. Check.  UNSAT -> proved, stop.
  3. SAT -> read the counterexample, find the cells its states occupy, and add
     any that were not active.  If they were ALL active, the counterexample is
     genuine against these contracts and the GRID is too coarse, so refining
     the cell set cannot help and the loop reports that instead of spinning.

Usage: python3 .../z3_cegar.py <nx> <nv> <nvd> <kind> <K> [max_iters]
"""
import itertools
import math
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort
import torch
import z3

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from gen_nav_contracts import load_net, crown_bounds, floor_to, ceil_to  # noqa: E402
from gen_nav_affine import crown_affine, NET                             # noqa: E402
from gen_nav_smv import displacement_bounds                              # noqa: E402
from affine_displacement import affine_displacement                      # noqa: E402
from auto_LiRPA import BoundedModule                                     # noqa: E402

DT = 0.2
XR, VR = (-1.0, 3.5), (-2.0, 2.0)
OBS = ((1.0, 2.0), (1.0, 2.0))


def cell_index(val, lo, hi, n):
    """Grid index containing val, clamped to the grid."""
    return min(max(int((val - lo) / (hi - lo) * n), 0), n - 1)


def cell_box(key, nx, nv, nth):
    """(lo, hi) corners of the cell named by a 4-tuple of indices."""
    i, j, a, b = key
    xs = np.linspace(*XR, nx + 1)
    vs = np.linspace(*VR, nv + 1)
    ts = np.linspace(-math.pi, math.pi, nth + 1)
    return ([xs[i], xs[j], vs[a], ts[b]],
            [xs[i + 1], xs[j + 1], vs[a + 1], ts[b + 1]])


def seed_cells(nx, nv, nth, nvd=None):
    """Cells the true trajectories visit: a cheap, useful starting set.

    Returns the network cells, and the displacement cells when nvd is given.
    """
    sess = ort.InferenceSession(str(NET))
    inp = sess.get_inputs()[0].name
    seen, dseen = set(), set()
    for x0, y0 in itertools.product(np.linspace(2.9, 3.1, 15), repeat=2):
        state = np.array([x0, y0, 0.0, 0.0], np.float32)
        for _ in range(30):
            u = sess.run(None, {inp: state.reshape(1, 4)})[0].ravel()
            seen.add((cell_index(state[0], *XR, nx),
                      cell_index(state[1], *XR, nx),
                      cell_index(state[2], *VR, nv),
                      cell_index(state[3], -math.pi, math.pi, nth)))
            if nvd:
                dseen.add((cell_index(state[2], *VR, nvd),
                           cell_index(state[3], -math.pi, math.pi, nvd)))
            state = state + DT * np.array(
                [state[2] * np.cos(state[3]), state[2] * np.sin(state[3]),
                 u[0], u[1]], np.float32)
    return (seen, dseen) if nvd else seen



def with_neighbours(keys, limits):
    """Each cell plus its face neighbours, clipped to the grid.

    An uncovered cell is UNCONSTRAINED, so the solver will route the next
    counterexample straight through one if it can; adding only the cells a
    counterexample touched therefore converges one thin path at a time.
    Widening by one cell per axis fills the tube far faster, and since adding
    contracts only tightens the abstraction it cannot cost soundness.
    """
    out = set()
    for key in keys:
        out.add(key)
        for axis, limit in enumerate(limits):
            for step in (-1, 1):
                nb = list(key)
                nb[axis] += step
                if 0 <= nb[axis] < limit:
                    out.add(tuple(nb))
    return out


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def build(active, dactive, kind, bm, nx, nv, nth, nvd, horizon,
          affine_disp=False):
    """A BMC query carrying contracts for the active cells only.

    Both families are restricted: the network cells in ``active`` and the
    displacement cells in ``dactive``.  Refining only the network leaves
    nvd^2 displacement contracts dominating the query.
    """
    keys = sorted(active)
    boxes = [cell_box(k, nx, nv, nth) for k in keys]
    lo = np.array([b[0] for b in boxes], np.float32)
    hi = np.array([b[1] for b in boxes], np.float32)
    if kind == "affine":
        l_a, lbias, u_a, ubias = crown_affine(bm, lo, hi)
    else:
        clb, cub = crown_bounds(bm, lo, hi)

    s = z3.Solver()
    st = [[z3.Real(f"{n}_{k}") for n in ("x", "y", "v", "th")]
          for k in range(horizon + 1)]
    s.add(st[0][0] >= 2.9, st[0][0] <= 3.1, st[0][1] >= 2.9, st[0][1] <= 3.1,
          st[0][2] == 0.0, st[0][3] == 0.0)
    vd = np.linspace(*VR, nvd + 1)
    td = np.linspace(-math.pi, math.pi, nvd + 1)

    for k in range(horizon):
        x, y, v, th = st[k]
        u1, u2, dx, dy = (z3.Real(f"{n}_{k}") for n in ("u1", "u2", "dx", "dy"))
        s.add(u1 >= -1, u1 <= 1, u2 >= -1, u2 <= 1)
        for n in range(len(keys)):
            guard = z3.And(x >= float(lo[n][0]), x <= float(hi[n][0]),
                           y >= float(lo[n][1]), y <= float(hi[n][1]),
                           v >= float(lo[n][2]), v <= float(hi[n][2]),
                           th >= float(lo[n][3]), th <= float(hi[n][3]))
            if kind == "affine":
                terms = [x, y, v, th]
                low = sum(float(l_a[n][0][i]) * terms[i] for i in range(4)) \
                    + float(lbias[n][0])
                high = sum(float(u_a[n][0][i]) * terms[i] for i in range(4)) \
                    + float(ubias[n][0])
                s.add(z3.Implies(guard, z3.And(u1 >= low, u1 <= high)))
            else:
                s.add(z3.Implies(guard, z3.And(
                    u1 >= floor_to(max(float(clb[n][0]), -1.0)),
                    u1 <= ceil_to(min(float(cub[n][0]), 1.0)))))
        for a, b in sorted(dactive):
            g = z3.And(v >= float(vd[a]), v <= float(vd[a + 1]),
                       th >= float(td[b]), th <= float(td[b + 1]))
            if affine_disp:
                # dx = v*cos(th)*dt linearised about the cell centre with a
                # sound second-order remainder.  Couples only v and th, so it
                # stays cheap, and at nvd=32 it is TIGHTER than the
                # concretized interval at nvd=128 with 16x fewer contracts.
                cons = []
                for var, use_sin in ((dx, False), (dy, True)):
                    av, ath, c, rad = affine_displacement(
                        vd[a], vd[a + 1], td[b], td[b + 1], DT, use_sin)
                    cons += [var >= av * v + ath * th + (c - rad),
                             var <= av * v + ath * th + (c + rad)]
                s.add(z3.Implies(g, z3.And(*cons)))
            else:
                (dxl, dxh), (dyl, dyh) = displacement_bounds(
                    vd[a], vd[a + 1], td[b], td[b + 1])
                s.add(z3.Implies(g, z3.And(dx >= dxl, dx <= dxh,
                                           dy >= dyl, dy <= dyh)))
        s.add(st[k + 1][0] == x + dx, st[k + 1][1] == y + dy,
              st[k + 1][2] == v + DT * u1, st[k + 1][3] == th + DT * u2)

    s.add(z3.Or([z3.And(st[k][0] >= OBS[0][0], st[k][0] <= OBS[0][1],
                        st[k][1] >= OBS[1][0], st[k][1] <= OBS[1][1])
                 for k in range(horizon + 1)]))
    return s, st


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def refine(active, dactive, kind, bm, nx, nv, nth, nvd, horizon,
           max_iters, affine_disp=False):
    """Refine until UNSAT, until the grid is too coarse, or until max_iters.

    Returns (status, active, dactive) so a caller can carry a converged cell
    set forward to a longer horizon instead of rediscovering it from scratch.
    """
    for it in range(max_iters):
        t0 = time.time()
        solver, st = build(active, dactive, kind, bm,
                           nx, nv, nth, nvd, horizon, affine_disp)
        res = solver.check()
        dt = time.time() - t0
        if res == z3.unsat:
            total = len(active) + len(dactive)
            full = nx * nx * nv * nth + nvd * nvd
            print(f"  iter {it}: {len(active)}+{len(dactive)} cells -> "
                  f"UNSAT ({dt:.1f}s)")
            print(f"  PROVED to depth {horizon} with {total} contracts "
                  f"({full / total:.0f}x fewer than the full grid)")
            return "proved", active, dactive
        if res != z3.sat:
            print(f"  iter {it}: {res} ({dt:.1f}s)")
            return "unknown", active, dactive
        model = solver.model()
        added, dadded = set(), set()
        for k in range(horizon + 1):
            vals = [float(model.eval(st[k][i], model_completion=True)
                          .as_fraction()) for i in range(4)]
            key = (cell_index(vals[0], *XR, nx), cell_index(vals[1], *XR, nx),
                   cell_index(vals[2], *VR, nv),
                   cell_index(vals[3], -math.pi, math.pi, nth))
            if key not in active:
                added.add(key)
            dkey = (cell_index(vals[2], *VR, nvd),
                    cell_index(vals[3], -math.pi, math.pi, nvd))
            if dkey not in dactive:
                dadded.add(dkey)
        print(f"  iter {it}: {len(active)}+{len(dactive)} cells -> SAT "
              f"({dt:.1f}s), {len(added)}+{len(dadded)} new on the "
              "counterexample")
        if not added and not dadded:
            print("  counterexample stays inside the covered cells: the GRID "
                  "is too coarse, adding cells cannot help")
            return "too_coarse", active, dactive
        active |= with_neighbours(added, (nx, nx, nv, nth))
        dactive |= with_neighbours(dadded, (nvd, nvd))
    print(f"  stopped after {max_iters} iterations at "
          f"{len(active)}+{len(dactive)} cells")
    return "exhausted", active, dactive


def main():
    """Ladder the horizon, carrying each converged cell set to the next.

    Starting cold at the full horizon makes every iteration expensive while the
    cell set is still being discovered.  Proving a short horizon first is cheap
    and yields most of the cells the longer one needs.
    """
    nx, nv, nvd = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    kind, horizon = sys.argv[4], int(sys.argv[5])
    max_iters = int(sys.argv[6]) if len(sys.argv) > 6 else 25
    ladder = [int(v) for v in sys.argv[7].split(",")] if len(sys.argv) > 7 \
        else [horizon]
    # Refining the DISPLACEMENT cells makes each query cheaper but removes
    # constraints that are doing real work: an uncovered displacement cell
    # leaves dx and dy free and the solver walks straight through it.  The
    # network cells are the ones worth refining, so keep the plant's in full.
    full_disp = (sys.argv[8].lower() != "refine") if len(sys.argv) > 8 else True
    affine_disp = sys.argv[9].lower() == "affinedisp" if len(sys.argv) > 9 else False
    nth = nv
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    active, dactive = seed_cells(nx, nv, nth, nvd)
    active = with_neighbours(active, (nx, nx, nv, nth))
    dactive = ({(a, b) for a in range(nvd) for b in range(nvd)} if full_disp
               else with_neighbours(dactive, (nvd, nvd)))
    print(f"{kind} net {nx}x{nv}x{nth}, disp {nvd} "
          f"({'all' if full_disp else 'refined'}, {'affine' if affine_disp else 'interval'})")
    print(f"  full grid would be {nx*nx*nv*nth} net + {nvd*nvd} disp cells; "
          f"seeded with {len(active)} + {len(dactive)}")

    for rung in ladder:
        print(f"--- horizon {rung} ---")
        status, active, dactive = refine(
            active, dactive, kind, bm, nx, nv, nth, nvd, rung,
            max_iters, affine_disp)
        if status != "proved":
            print(f"  stopped at horizon {rung} with status {status}")
            return
    print(f"ALL RUNGS PROVED up to horizon {ladder[-1]}")


if __name__ == "__main__":
    main()
