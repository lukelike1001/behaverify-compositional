"""
Does compositional verification earn its keep when the behavior tree BRANCHES?

Bare NAV is one action node, so a per-step reachable box tracks it as well as
anything and the compositional machinery buys nothing (RESULTS sections 12-13).
The claim worth testing is that compositional verification pays where the tree
BRANCHES, because the reachable set is then a union over modes that a single
box cannot represent and an SMT encoding can.

So: a genuine two-branch NSBT over the same plant,

    selector( sequence(check: x < SPLIT, act: u = net_set(s)),
              act: u = net_point(s) )

verified two ways on identical contracts:

  per-step  one box per step; where the box STRADDLES the branch condition it
            must take the union of both controllers' bounds, because it cannot
            represent "these states go left and those go right".
  smt       the condition is a guard, so each state uses its own branch, and
            the disjunction is kept rather than collapsed.

Both are sound.  The question is only how much precision the box loses.

Usage: python3 .../branching_nav.py [split] [steps]
"""
import itertools
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort
import torch
import z3

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from gen_nav_contracts import load_net, crown_bounds                # noqa: E402
from auto_LiRPA import BoundedModule                                # noqa: E402
from per_step_contracts import displacement_box                     # noqa: E402

NETS = {n: f"REPRODUCIBILITY/2026_TBA/examples/NAV/networks/{n}.onnx"
        for n in ("nn-nav-set", "nn-nav-point")}
DT = 0.2
OBS = ((1.0, 2.0), (1.0, 2.0))
XR, VR = (-1.0, 3.5), (-2.0, 2.0)


def bounds_for(bmod, box):
    """CROWN control bounds for one network over a state box."""
    lo = np.array([[b[0] for b in box]], np.float32)
    hi = np.array([[b[1] for b in box]], np.float32)
    lb, ub = crown_bounds(bmod, lo, hi)
    return ((max(float(lb[0][0]), -1.0), min(float(ub[0][0]), 1.0)),
            (max(float(lb[0][1]), -1.0), min(float(ub[0][1]), 1.0)))


def hits_obstacle(box):
    """Whether a state box intersects the obstacle."""
    (xlo, xhi), (ylo, yhi) = box[0], box[1]
    return (xlo <= OBS[0][1] and xhi >= OBS[0][0]
            and ylo <= OBS[1][1] and yhi >= OBS[1][0])


# pylint: disable-next=too-many-locals
def per_step(models, split, steps, axis=0):
    """Box propagation; unions both branches wherever the box straddles."""
    box = [(2.9, 3.1), (2.9, 3.1), (0.0, 0.0), (0.0, 0.0)]
    straddles = 0
    for k in range(steps + 1):
        if hits_obstacle(box):
            return k, straddles
        lo_x, hi_x = box[axis]
        if hi_x < split:
            active = ["nn-nav-set"]
        elif lo_x >= split:
            active = ["nn-nav-point"]
        else:
            active = ["nn-nav-set", "nn-nav-point"]
            straddles += 1
        per_net = [bounds_for(models[n], box) for n in active]
        u1 = (min(b[0][0] for b in per_net), max(b[0][1] for b in per_net))
        u2 = (min(b[1][0] for b in per_net), max(b[1][1] for b in per_net))
        (dxl, dxh), (dyl, dyh) = displacement_box(*box[2], *box[3])
        box = [(box[0][0] + dxl, box[0][1] + dxh),
               (box[1][0] + dyl, box[1][1] + dyh),
               (box[2][0] + DT * u1[0], box[2][1] + DT * u1[1]),
               (box[3][0] + DT * u2[0], box[3][1] + DT * u2[1])]
    return None, straddles


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def smt(models, split, steps, nx=8, nvd=64, axis=0):
    """SMT encoding that keeps the branch as a guard instead of a union."""
    xs = np.linspace(*XR, nx + 1)
    vs = np.linspace(*VR, nx + 1)
    ts = np.linspace(-np.pi, np.pi, nx + 1)
    cells, los, his = [], [], []
    for i, j, a, b in itertools.product(range(nx), repeat=4):
        los.append([xs[i], xs[j], vs[a], ts[b]])
        his.append([xs[i + 1], xs[j + 1], vs[a + 1], ts[b + 1]])
        cells.append((i, j, a, b))
    lo_arr, hi_arr = np.array(los, np.float32), np.array(his, np.float32)
    per_net = {n: crown_bounds(models[n], lo_arr, hi_arr) for n in NETS}

    st = [[z3.Real(f"{v}_{k}") for v in ("x", "y", "v", "th")]
          for k in range(steps + 1)]
    sol = z3.Solver()
    sol.add(st[0][0] >= 2.9, st[0][0] <= 3.1, st[0][1] >= 2.9, st[0][1] <= 3.1,
            st[0][2] == 0.0, st[0][3] == 0.0)
    vd = np.linspace(*VR, nvd + 1)
    td = np.linspace(-np.pi, np.pi, nvd + 1)
    for k in range(steps):
        x, y, v, th = st[k]
        u1, u2, dx, dy = (z3.Real(f"{n}_{k}") for n in ("u1", "u2", "dx", "dy"))
        sol.add(u1 >= -1, u1 <= 1, u2 >= -1, u2 <= 1)
        for n, (lb, ub) in per_net.items():
            # the branch condition is a GUARD, so each state uses its own
            # controller rather than the union of both
            svar = st[k][axis]
            branch = (svar < split if n == "nn-nav-set"
                      else svar >= split)
            for idx, (clo, chi) in enumerate(zip(lo_arr, hi_arr)):
                guard = z3.And(branch,
                               x >= float(clo[0]), x <= float(chi[0]),
                               y >= float(clo[1]), y <= float(chi[1]),
                               v >= float(clo[2]), v <= float(chi[2]),
                               th >= float(clo[3]), th <= float(chi[3]))
                sol.add(z3.Implies(guard, z3.And(
                    u1 >= max(float(lb[idx][0]), -1.0),
                    u1 <= min(float(ub[idx][0]), 1.0),
                    u2 >= max(float(lb[idx][1]), -1.0),
                    u2 <= min(float(ub[idx][1]), 1.0))))
        for a in range(nvd):
            for b in range(nvd):
                (dxl, dxh), (dyl, dyh) = displacement_box(
                    vd[a], vd[a + 1], td[b], td[b + 1])
                sol.add(z3.Implies(
                    z3.And(v >= float(vd[a]), v <= float(vd[a + 1]),
                           th >= float(td[b]), th <= float(td[b + 1])),
                    z3.And(dx >= dxl, dx <= dxh, dy >= dyl, dy <= dyh)))
        sol.add(st[k + 1][0] == x + dx, st[k + 1][1] == y + dy,
                st[k + 1][2] == v + DT * u1, st[k + 1][3] == th + DT * u2)
    sol.add(z3.Or([z3.And(st[k][0] >= OBS[0][0], st[k][0] <= OBS[0][1],
                          st[k][1] >= OBS[1][0], st[k][1] <= OBS[1][1])
                   for k in range(steps + 1)]))
    return sol.check()


def truth(split, steps, axis=0):
    """Closest approach of the true branching closed loop to the obstacle."""
    sess = {n: ort.InferenceSession(p) for n, p in NETS.items()}
    worst = 1e9
    for x0, y0 in itertools.product(np.linspace(2.9, 3.1, 9), repeat=2):
        s = np.array([x0, y0, 0.0, 0.0], np.float32)
        for _ in range(steps + 1):
            dxm = max(1.0 - s[0], 0.0, s[0] - 2.0)
            dym = max(1.0 - s[1], 0.0, s[1] - 2.0)
            worst = min(worst, float(np.hypot(dxm, dym)))
            net = "nn-nav-set" if s[axis] < split else "nn-nav-point"
            u = sess[net].run(
                None, {sess[net].get_inputs()[0].name: s.reshape(1, 4)}
            )[0].ravel()
            s = s + DT * np.array([s[2] * np.cos(s[3]), s[2] * np.sin(s[3]),
                                   u[0], u[1]], np.float32)
    return worst


def main():
    """Compare per-step and SMT on the branching tree."""
    split = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    axis = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    name = ("x", "y", "v", "th")[axis]
    models = {n: BoundedModule(load_net(p), torch.zeros(1, 4))
              for n, p in NETS.items()}
    print(f"branching NSBT: {name} < {split} -> nn-nav-set, else nn-nav-point")
    print(f"  true closest approach over {steps} steps: "
          f"{truth(split, steps, axis):.4f}")
    hit, straddles = per_step(models, split, steps, axis)
    verdict = (f"touches obstacle at k={hit}" if hit is not None
               else f"clear through k={steps}")
    print(f"  per-step: {verdict}   "
          f"(box straddled the branch on {straddles} steps)")
    for k in range(4, steps + 1, 2):
        res = smt(models, split, k, axis=axis)
        print(f"  smt k={k:<3} {res}")
        if res == z3.sat:
            break


if __name__ == "__main__":
    main()
