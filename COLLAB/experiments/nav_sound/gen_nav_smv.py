"""
The full NAV model for nuXmv: abstract control + abstract displacement, both
constrained by outward-rounded interval contracts. No lookup table.

Two abstractions, each sound because the true function satisfies its bounds:
  u1,u2 = ctrl(x,y,v,th)      bounded per state cell by alpha-beta-CROWN
  dx,dy = (v cos th, v sin th)*dt   bounded per (v,th) cell by interval arithmetic

dx and dy are bounded on the SAME (v,th) cell rather than independently, which
keeps the coupling |(dx,dy)| <= |v|dt implicit. Bounding them on unrelated cells
would let the solver travel at sqrt(2)|v| (measured: 1.41x at 4 theta bands).
"""
from __future__ import annotations
import math, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
from gen_nav_contracts import load_net, crown_bounds, floor_to, ceil_to  # noqa: E402
from auto_LiRPA import BoundedModule                                      # noqa: E402

NET = pathlib.Path("REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx")
DT, STEPS = 0.2, 30
OBS = ((1.0, 2.0), (1.0, 2.0))
GOAL = ((-0.5, 0.5), (-0.5, 0.5))


def displacement_bounds(vlo, vhi, tlo, thi, samples=400):
    """Sound interval bounds on (v cos th, v sin th)*dt over one (v,th) cell."""
    ts = np.linspace(tlo, thi, samples)
    c, s = np.cos(ts), np.sin(ts)
    dx = np.concatenate([vlo * c, vhi * c]) * DT
    dy = np.concatenate([vlo * s, vhi * s]) * DT
    pad = abs(vhi - vlo) * (thi - tlo) * DT          # outward pad for sampling error
    return ((floor_to(dx.min() - pad), ceil_to(dx.max() + pad)),
            (floor_to(dy.min() - pad), ceil_to(dy.max() + pad)))


def build(nx: int, nv: int, xr=(-1.0, 3.5), vr=(-2.0, 2.0)) -> str:
    xs = np.linspace(*xr, nx + 1); vs = np.linspace(*vr, nv + 1)
    ts = np.linspace(-math.pi, math.pi, nv + 1)
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))

    los, his, cells = [], [], []
    for i in range(nx):
        for j in range(nx):
            for a in range(nv):
                for b in range(nv):
                    lo = [xs[i], xs[j], vs[a], ts[b]]
                    hi = [xs[i+1], xs[j+1], vs[a+1], ts[b+1]]
                    los.append(lo); his.append(hi); cells.append((lo, hi))
    lb, ub = crown_bounds(bm, np.array(los, np.float32), np.array(his, np.float32))

    L = ["MODULE main", "VAR",
         "  x : real;", "  y : real;", "  v : real;", "  th : real;",
         "  u1 : real;", "  u2 : real;", "  dx : real;", "  dy : real;",
         f"  k : 0..{STEPS};",
         f"INIT x >= 2.9 & x <= 3.1 & y >= 2.9 & y <= 3.1 & v = 0.0 & th = 0.0 & k = 0;",
         f"TRANS next(k) = (k < {STEPS} ? k + 1 : {STEPS});",
         f"TRANS (k < {STEPS}) -> (next(x) = x + dx & next(y) = y + dy &"
         f" next(v) = v + {DT} * u1 & next(th) = th + {DT} * u2);",
         f"TRANS (k >= {STEPS}) -> (next(x) = x & next(y) = y & next(v) = v & next(th) = th);"]

    for (lo, hi), l, u in zip(cells, lb, ub):
        g = (f"x >= {lo[0]:.6f} & x <= {hi[0]:.6f} & y >= {lo[1]:.6f} & y <= {hi[1]:.6f} & "
             f"v >= {lo[2]:.6f} & v <= {hi[2]:.6f} & th >= {lo[3]:.6f} & th <= {hi[3]:.6f}")
        L.append(f"INVAR ({g}) -> (u1 >= {floor_to(max(l[0],-1.0)):.6f} &"
                 f" u1 <= {ceil_to(min(u[0],1.0)):.6f} &"
                 f" u2 >= {floor_to(max(l[1],-1.0)):.6f} &"
                 f" u2 <= {ceil_to(min(u[1],1.0)):.6f});")

    for a in range(nv):
        for b in range(nv):
            (dxl, dxh), (dyl, dyh) = displacement_bounds(vs[a], vs[a+1], ts[b], ts[b+1])
            L.append(f"INVAR (v >= {vs[a]:.6f} & v <= {vs[a+1]:.6f} &"
                     f" th >= {ts[b]:.6f} & th <= {ts[b+1]:.6f}) ->"
                     f" (dx >= {dxl:.6f} & dx <= {dxh:.6f} &"
                     f" dy >= {dyl:.6f} & dy <= {dyh:.6f});")

    prop = sys.argv[3] if len(sys.argv) > 3 else "obstacle"
    if prop == "obstacle":
        L.append(f"INVARSPEC !(x >= {OBS[0][0]} & x <= {OBS[0][1]} &"
                 f" y >= {OBS[1][0]} & y <= {OBS[1][1]});")
    elif prop == "goal":
        L.append(f"INVARSPEC (k = {STEPS}) -> (x >= {GOAL[0][0]} & x <= {GOAL[0][1]} &"
                 f" y >= {GOAL[1][0]} & y <= {GOAL[1][1]});")
    else:                                   # progress: must be FALSE (horizon reachable)
        L.append(f"INVARSPEC k != {STEPS};")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    nx, nv = int(sys.argv[1]), int(sys.argv[2])
    text = build(nx, nv)
    out = pathlib.Path(f"COLLAB/experiments/nav_sound/nav_{nx}_{nv}_"
                       f"{sys.argv[3] if len(sys.argv)>3 else 'obstacle'}.smv")
    out.write_text(text)
    print(f"  {out.name}: {len(text.splitlines())} lines, "
          f"{text.count('INVAR (')} contracts")
