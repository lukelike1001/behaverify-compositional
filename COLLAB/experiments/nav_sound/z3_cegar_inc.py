"""
Incremental demand-driven refinement.

z3_cegar.py rebuilds the whole solver every iteration, so each round pays for
the transition relation, the plant contracts and the property again, and throws
away everything Z3 learned.  Refinement only ever ADDS contracts, so the
assertion set is monotone and Z3 can be driven incrementally instead: build the
frame once, then assert the new cells' implications and re-check.

Soundness is unchanged - the assertions are the same ones, in the same
direction - so this is purely a speed change, and it is cross-checked against
the non-incremental loop.

Usage: python3 .../z3_cegar_inc.py <nx> <nv> <nvd> <K> [max_iters] [ladder]
"""
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
from gen_nav_affine import NET, crown_affine                             # noqa: E402
from gen_nav_smv import displacement_bounds                              # noqa: E402
from affine_displacement import affine_displacement                      # noqa: E402
from z3_cegar import cell_index, cell_box, seed_cells                   # noqa: E402
from auto_LiRPA import BoundedModule                                     # noqa: E402

DT = 0.2
XR, VR = (-1.0, 3.5), (-2.0, 2.0)
OBS = ((1.0, 2.0), (1.0, 2.0))



def cell_radii(nx, nv, nth, margin=(0.30, 0.30, 0.30, 0.30)):
    """How many cells per axis cover a fixed PHYSICAL margin.

    Expanding by one cell per axis covers a shrinking physical region as the
    grid refines, so a finer grid slips out of the covered set more easily -
    and an uncovered cell is UNCONSTRAINED, which costs more than the tighter
    bounds gain.  Keeping the margin physical rather than index-based makes
    refinement monotone in the way one expects.
    """
    widths = ((XR[1] - XR[0]) / nx, (XR[1] - XR[0]) / nx,
              (VR[1] - VR[0]) / nv, 2 * math.pi / nth)
    return tuple(max(1, int(round(m / w))) for m, w in zip(margin, widths))


def expand(keys, limits, radii):
    """Every cell within `radii` of a given cell, per axis, clipped."""
    out = set()
    for key in keys:
        for axis, (limit, rad) in enumerate(zip(limits, radii)):
            for step in range(-rad, rad + 1):
                nb = list(key)
                nb[axis] += step
                if 0 <= nb[axis] < limit:
                    out.add(tuple(nb))
        out.add(key)
    return out


# pylint: disable-next=too-many-instance-attributes
class IncrementalRefiner:
    """Holds one Z3 solver and grows its contract set in place."""

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def __init__(self, nx, nv, nth, nvd, horizon, plant_cells=None,
                 affine_plant=False, affine_net=False):
        self.nx, self.nv, self.nth = nx, nv, nth
        self.nvd, self.horizon = nvd, horizon
        self.bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
        self.solver = z3.Solver()
        self.affine_plant = affine_plant
        self.affine_net = affine_net
        self.emitted = set()          # (level, i, j, a, b)
        self.plant_emitted = set()
        self.plant_cells = (sorted(plant_cells) if plant_cells is not None
                            else [(a, b) for a in range(nvd)
                                  for b in range(nvd)])
        self.state = [[z3.Real(f"{n}_{k}") for n in ("x", "y", "v", "th")]
                      for k in range(horizon + 1)]
        self._frame()

    # pylint: disable-next=too-many-locals
    def _frame(self):
        """Initial set, transition relation, plant contracts and property."""
        st = self.state
        s = self.solver
        s.add(st[0][0] >= 2.9, st[0][0] <= 3.1,
              st[0][1] >= 2.9, st[0][1] <= 3.1,
              st[0][2] == 0.0, st[0][3] == 0.0)
        for k in range(self.horizon):
            x, y, v, th = st[k]
            u1, u2, dx, dy = (z3.Real(f"{n}_{k}")
                              for n in ("u1", "u2", "dx", "dy"))
            s.add(u1 >= -1, u1 <= 1, u2 >= -1, u2 <= 1)
            # |dx|, |dy| <= |v|*dt holds everywhere, with no cell needed.
            # Without it an UNREFINED displacement cell leaves dx and dy
            # completely free and the solver walks straight through one, which
            # is what stopped the refining loop converging.  Split on the sign
            # of v to stay linear.
            s.add(z3.Implies(v >= 0, z3.And(dx <= DT * v, dx >= -DT * v,
                                            dy <= DT * v, dy >= -DT * v)))
            s.add(z3.Implies(v <= 0, z3.And(dx <= -DT * v, dx >= DT * v,
                                            dy <= -DT * v, dy >= DT * v)))
            s.add(st[k + 1][0] == x + dx, st[k + 1][1] == y + dy,
                  st[k + 1][2] == v + DT * u1, st[k + 1][3] == th + DT * u2)
        s.add(z3.Or([z3.And(st[k][0] >= OBS[0][0], st[k][0] <= OBS[0][1],
                            st[k][1] >= OBS[1][0], st[k][1] <= OBS[1][1])
                     for k in range(self.horizon + 1)]))

    # pylint: disable-next=too-many-locals
    def add_cells(self, keys, level=0):
        """Assert the network contract for each not-yet-emitted cell.

        ``level`` L means the grid is subdivided 2^L times per axis.  A parent
        cell's contract and its children's are BOTH valid for the true network,
        so adding finer contracts without retracting coarser ones is sound and
        strictly tightens the abstraction - which is what lets refinement by
        splitting run incrementally.
        """
        scale = 2 ** level
        fresh = [(level,) + k for k in keys if (level,) + k not in self.emitted]
        if not fresh:
            return 0
        nx, nv, nth = self.nx * scale, self.nv * scale, self.nth * scale
        boxes = [cell_box(k[1:], nx, nv, nth) for k in fresh]
        lo = np.array([b[0] for b in boxes], np.float32)
        hi = np.array([b[1] for b in boxes], np.float32)
        if self.affine_net:
            l_a, lbias, u_a, ubias = crown_affine(self.bm, lo, hi)
            clb = cub = None
        else:
            clb, cub = crown_bounds(self.bm, lo, hi)
            l_a = lbias = u_a = ubias = None
        for k in range(self.horizon):
            x, y, v, th = self.state[k]
            for n, o, ctrl in ((n, o, z3.Real(f"u{o + 1}_{k}"))
                               for n in range(len(fresh)) for o in (0, 1)):
                guard = z3.And(x >= float(lo[n][0]), x <= float(hi[n][0]),
                               y >= float(lo[n][1]), y <= float(hi[n][1]),
                               v >= float(lo[n][2]), v <= float(hi[n][2]),
                               th >= float(lo[n][3]), th <= float(hi[n][3]))
                if self.affine_net:
                    # An AFFINE contract makes the bound FOLLOW the state, so
                    # the union over several spanned cells is not much wider
                    # than one cell's - which is the feedback loop that defeats
                    # interval contracts once the state spreads past a cell.
                    terms = [x, y, v, th]
                    lower = sum(float(l_a[n][o][i]) * terms[i]
                                for i in range(4)) + float(lbias[n][o])
                    upper = sum(float(u_a[n][o][i]) * terms[i]
                                for i in range(4)) + float(ubias[n][o])
                    self.solver.add(z3.Implies(
                        guard, z3.And(ctrl >= lower, ctrl <= upper)))
                else:
                    self.solver.add(z3.Implies(guard, z3.And(
                        ctrl >= floor_to(max(float(clb[n][o]), -1.0)),
                        ctrl <= ceil_to(min(float(cub[n][o]), 1.0)))))
        self.emitted.update(fresh)
        return len(fresh)

    def split_cells(self, model, level):
        """Add the counterexample's cells at a finer level.

        Called when adding same-level cells stops helping: the counterexample
        is inside covered cells, so the residual freedom is the CONTRACT WIDTH
        there, and only a finer cell reduces it.
        """
        scale = 2 ** level
        nx, nv, nth = self.nx * scale, self.nv * scale, self.nth * scale
        keys = set()
        for k in range(self.horizon + 1):
            vals = [float(model.eval(self.state[k][i], model_completion=True)
                          .as_fraction()) for i in range(4)]
            keys.add((cell_index(vals[0], *XR, nx),
                      cell_index(vals[1], *XR, nx),
                      cell_index(vals[2], *VR, nv),
                      cell_index(vals[3], -math.pi, math.pi, nth)))
        return self.add_cells(expand(keys, (nx, nx, nv, nth),
                                     cell_radii(nx, nv, nth)), level)

    # pylint: disable-next=too-many-locals
    def add_plant_cells(self, keys):
        """Assert the displacement contract for each new (v, th) cell.

        Safe to do incrementally only because the global |dx|,|dy| <= |v|*dt
        bound already constrains the cells that have not been added.
        """
        fresh = [k for k in keys if k not in self.plant_emitted]
        if not fresh:
            return 0
        vd = np.linspace(*VR, self.nvd + 1)
        td = np.linspace(-math.pi, math.pi, self.nvd + 1)
        for k in range(self.horizon):
            _, _, v, th = self.state[k]
            dx, dy = z3.Real(f"dx_{k}"), z3.Real(f"dy_{k}")
            for a, b in fresh:
                g = z3.And(v >= float(vd[a]), v <= float(vd[a + 1]),
                           th >= float(td[b]), th <= float(td[b + 1]))
                if self.affine_plant:
                    # v*cos(th)*dt is a KNOWN analytic function, so it can be
                    # linearised about the cell centre with a sound
                    # second-order remainder instead of concretized to a box.
                    # It couples only v and th, so it stays cheap.
                    cons = []
                    for var, use_sin in ((dx, False), (dy, True)):
                        av, ath, c, rad = affine_displacement(
                            vd[a], vd[a + 1], td[b], td[b + 1], DT, use_sin)
                        cons += [var >= av * v + ath * th + (c - rad),
                                 var <= av * v + ath * th + (c + rad)]
                    self.solver.add(z3.Implies(g, z3.And(*cons)))
                else:
                    (dxl, dxh), (dyl, dyh) = displacement_bounds(
                        vd[a], vd[a + 1], td[b], td[b + 1])
                    self.solver.add(z3.Implies(g, z3.And(
                        dx >= dxl, dx <= dxh, dy >= dyl, dy <= dyh)))
        self.plant_emitted.update(fresh)
        return len(fresh)

    def counterexample_plant_cells(self, model):
        """(v, th) cells the counterexample trace occupies."""
        out = set()
        for k in range(self.horizon + 1):
            v = float(model.eval(self.state[k][2], model_completion=True)
                      .as_fraction())
            th = float(model.eval(self.state[k][3], model_completion=True)
                       .as_fraction())
            out.add((cell_index(v, *VR, self.nvd),
                     cell_index(th, -math.pi, math.pi, self.nvd)))
        return out

    def counterexample_cells(self, model):
        """Cells the counterexample trace occupies, at the base level."""
        out = set()
        for k in range(self.horizon + 1):
            vals = [float(model.eval(self.state[k][i], model_completion=True)
                          .as_fraction()) for i in range(4)]
            out.add((cell_index(vals[0], *XR, self.nx),
                     cell_index(vals[1], *XR, self.nx),
                     cell_index(vals[2], *VR, self.nv),
                     cell_index(vals[3], -math.pi, math.pi, self.nth)))
        return out


# pylint: disable-next=too-many-locals

def _classify(model, ref, horizon):
    """Is the counterexample real, or an artefact of the over-approximation?

    Simulates the TRUE closed loop from the counterexample's own initial state.
    If the true run also hits the obstacle the counterexample is genuine and no
    refinement will remove it; if it stays clear, the counterexample is spurious
    and the gap is over-approximation error.
    """
    x0 = float(model.eval(ref.state[0][0], model_completion=True).as_fraction())
    y0 = float(model.eval(ref.state[0][1], model_completion=True).as_fraction())
    sess = ort.InferenceSession(str(NET))
    inp = sess.get_inputs()[0].name
    st = np.array([x0, y0, 0.0, 0.0], np.float32)
    worst, hit = 1e9, False
    for _ in range(horizon + 1):
        dxm = max(1.0 - st[0], 0.0, st[0] - 2.0)
        dym = max(1.0 - st[1], 0.0, st[1] - 2.0)
        d = float(np.hypot(dxm, dym))
        worst = min(worst, d)
        hit = hit or d == 0.0
        u = sess.run(None, {inp: st.reshape(1, 4)})[0].ravel()
        st = st + DT * np.array([st[2] * np.cos(st[3]), st[2] * np.sin(st[3]),
                                 u[0], u[1]], np.float32)
    print(f"  true run from the counterexample's start ({x0:.4f}, {y0:.4f}): "
          f"closest approach {worst:.4f}")
    print("  => COUNTEREXAMPLE IS REAL" if hit else
          "  => counterexample is SPURIOUS (true run clears the obstacle); "
          "the gap is over-approximation error, not a real violation")


# pylint: disable-next=too-many-locals
def main():
    """Refine incrementally and report the outcome."""
    nx, nv, nvd = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    horizon = int(sys.argv[4])
    max_iters = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    nth = nv
    active, dseed = seed_cells(nx, nv, nth, nvd)
    radii = cell_radii(nx, nv, nth)
    dradii = cell_radii(nvd, nvd, nvd)[2:]
    plant = expand(dseed, (nvd, nvd), dradii) if len(sys.argv) > 6 \
        and sys.argv[6].lower() == "seedplant" else None
    modes = {a.lower() for a in sys.argv[7:]}
    ref = IncrementalRefiner(nx, nv, nth, nvd, horizon, plant,
                             "affineplant" in modes, "affinenet" in modes)
    if modes:
        print(f"  contract forms: {sorted(modes)}")
    active = expand(active, (nx, nx, nv, nth), radii)
    print(f"  coverage radii (cells per axis): net {radii}, plant {dradii}")
    print(f"incremental, net {nx}x{nv}x{nth}, disp {nvd} "
          f"({'seeded' if plant else 'all'}), horizon {horizon}")
    ref.add_cells(active)
    ref.add_plant_cells(ref.plant_cells)
    print(f"  seeded with {len(ref.emitted)} network cells "
          f"+ {len(ref.plant_emitted)} plant contracts")

    level = 0
    for it in range(max_iters):
        t0 = time.time()
        res = ref.solver.check()
        dt = time.time() - t0
        if res == z3.unsat:
            print(f"  iter {it}: {len(ref.emitted)} cells -> UNSAT ({dt:.1f}s)")
            print(f"  PROVED to depth {horizon} with {len(ref.emitted)} "
                  f"network contracts of {nx*nx*nv*nth} "
                  f"({nx*nx*nv*nth/len(ref.emitted):.0f}x fewer) "
                  f"plus {len(ref.plant_emitted)} plant contracts")
            return
        if res != z3.sat:
            print(f"  iter {it}: {res} ({dt:.1f}s)")
            return
        model = ref.solver.model()
        cells = ref.counterexample_cells(model)
        seen0 = {k[1:] for k in ref.emitted if k[0] == 0}
        added = ref.add_cells(
            expand(cells - seen0, (nx, nx, nv, nth), radii))
        pcells = ref.counterexample_plant_cells(model)
        padded = ref.add_plant_cells(
            expand(pcells - ref.plant_emitted, (nvd, nvd), dradii))
        print(f"  iter {it}: {len(ref.emitted)}+{len(ref.plant_emitted)} "
              f"cells -> SAT ({dt:.1f}s), +{added}+{padded} new")
        if not added and not padded:
            level += 1
            split = ref.split_cells(model, level)
            if split:
                print(f"  iter {it}: stalled -> SPLIT to level {level} "
                      f"(cells 2^{level}x finer), +{split} contracts")
                continue
            print("  counterexample stays inside the covered cells, and "
                  "splitting added nothing further")
            ranges = []
            for i, name in enumerate(("x", "y", "v", "th")):
                vals = [float(model.eval(ref.state[k][i],
                                         model_completion=True).as_fraction())
                        for k in range(horizon + 1)]
                lim = XR if i < 2 else (VR if i == 2 else (-math.pi, math.pi))
                flag = " ESCAPES MODELLED BOX" if (
                    min(vals) < lim[0] - 1e-9 or max(vals) > lim[1] + 1e-9
                ) else ""
                ranges.append(f"    {name}: [{min(vals):.3f}, {max(vals):.3f}] "
                              f"box [{lim[0]:.3f}, {lim[1]:.3f}]{flag}")
            print("  counterexample state ranges:")
            print("\n".join(ranges))
            _classify(model, ref, horizon)
            return
    print(f"  stopped after {max_iters} iterations at {len(ref.emitted)} cells")


if __name__ == "__main__":
    main()
