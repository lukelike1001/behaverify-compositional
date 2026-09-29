"""
Direct BMC on the NAV contracts in Z3, bypassing nuXmv entirely.

nuXmv/MathSAT cannot decide the AFFINE encoding even at 514 contracts, while it
handles 5,120 INTERVAL contracts to depth 9 in 65s.  The open question is
whether that is a MathSAT weakness or something fundamental about dense
disjunctive linear arithmetic.  This asks a different solver the same question,
with the same contracts, so the comparison is like for like.

Usage: python3 COLLAB/experiments/nav_sound/z3_bmc.py <nx> <nv> <nvd> <kind> <K>
  kind: "affine" or "interval"
"""
import math
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import z3

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from gen_nav_contracts import load_net, crown_bounds, floor_to, ceil_to  # noqa: E402
from gen_nav_affine import crown_affine, NET                             # noqa: E402
from gen_nav_smv import displacement_bounds                              # noqa: E402
from auto_LiRPA import BoundedModule                                     # noqa: E402

DT = 0.2
XR, VR = (-1.0, 3.5), (-2.0, 2.0)
OBS = ((1.0, 2.0), (1.0, 2.0))


def cells(nx, nv, nth):
    """Every (lo, hi) grid cell of the 4-D state box."""
    xs = np.linspace(*XR, nx + 1)
    vs = np.linspace(*VR, nv + 1)
    ts = np.linspace(-math.pi, math.pi, nth + 1)
    lo, hi = [], []
    for i in range(nx):
        for j in range(nx):
            for a in range(nv):
                for b in range(nth):
                    lo.append([xs[i], xs[j], vs[a], ts[b]])
                    hi.append([xs[i + 1], xs[j + 1], vs[a + 1], ts[b + 1]])
    return np.array(lo, np.float32), np.array(hi, np.float32)


def main():  # pylint: disable=too-many-locals
    """Run BMC to depth K and report SAT/UNSAT with wall time."""
    nx, nv, nvd = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    kind, horizon = sys.argv[4], int(sys.argv[5])
    nth = nv
    net = load_net(NET)
    bm = BoundedModule(net, torch.zeros(1, 4))
    lo, hi = cells(nx, nv, nth)
    if kind == "affine":
        l_a, lbias, u_a, ubias = crown_affine(bm, lo, hi)
    else:
        clb, cub = crown_bounds(bm, lo, hi)

    s = z3.Solver()
    st = [[z3.Real(f"{n}_{k}") for n in ("x", "y", "v", "th")]
          for k in range(horizon + 1)]
    ct = [[z3.Real(f"{n}_{k}") for n in ("u1", "u2", "dx", "dy")]
          for k in range(horizon)]

    s.add(st[0][0] >= 2.9, st[0][0] <= 3.1, st[0][1] >= 2.9, st[0][1] <= 3.1,
          st[0][2] == 0.0, st[0][3] == 0.0)

    vd = np.linspace(*VR, nvd + 1)
    td = np.linspace(-math.pi, math.pi, nvd + 1)
    for k in range(horizon):
        x, y, v, th = st[k]
        u1, u2, dx, dy = ct[k]
        s.add(u1 >= -1, u1 <= 1, u2 >= -1, u2 <= 1)
        for n, (cl, ch) in enumerate(zip(lo, hi)):
            guard = z3.And(x >= float(cl[0]), x <= float(ch[0]),
                           y >= float(cl[1]), y <= float(ch[1]),
                           v >= float(cl[2]), v <= float(ch[2]),
                           th >= float(cl[3]), th <= float(ch[3]))
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
        for a in range(nvd):
            for b in range(nvd):
                (dxl, dxh), (dyl, dyh) = displacement_bounds(
                    vd[a], vd[a + 1], td[b], td[b + 1])
                g = z3.And(v >= float(vd[a]), v <= float(vd[a + 1]),
                           th >= float(td[b]), th <= float(td[b + 1]))
                s.add(z3.Implies(g, z3.And(dx >= dxl, dx <= dxh,
                                           dy >= dyl, dy <= dyh)))
        s.add(st[k + 1][0] == x + dx, st[k + 1][1] == y + dy,
              st[k + 1][2] == v + DT * u1, st[k + 1][3] == th + DT * u2)

    s.add(z3.Or([z3.And(st[k][0] >= OBS[0][0], st[k][0] <= OBS[0][1],
                        st[k][1] >= OBS[1][0], st[k][1] <= OBS[1][1])
                 for k in range(horizon + 1)]))

    print(f"{kind} net {nx}x{nv} disp {nvd}, horizon {horizon}: "
          f"{nx**2 * nv * nth + nvd**2} contracts/step")
    t0 = time.time()
    res = s.check()
    print(f"  {res}  ({time.time() - t0:.1f}s)   "
          f"{'counterexample exists' if res == z3.sat else 'property holds to this depth' if res == z3.unsat else ''}")


if __name__ == "__main__":
    main()
