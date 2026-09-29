"""
NAV with AFFINE (linear) CROWN contracts instead of concretized intervals.

CROWN certifies  lA.s + lbias <= f(s) <= uA.s + ubias  over each cell.  Prior
models concretized this to a box, which is where almost all of the width came
from.  Keeping the linear form costs the same one constraint per cell and stays
in linear real arithmetic, so nuXmv's MathSAT handles it directly.

Soundness under float->decimal rounding: rounding a COEFFICIENT in the wrong
direction can break the bound at some point of the cell, so each rounded
coefficient's error is multiplied by the largest |s_i| the cell admits and the
total is subtracted (lower) or added (upper) as a pad.

Usage: python3 COLLAB/experiments/nav_sound/gen_nav_affine.py <nx> <nv> [prop] [nvd] [nth]
"""
from __future__ import annotations

import math
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from gen_nav_contracts import load_net, floor_to, ceil_to              # noqa: E402
from gen_nav_smv import displacement_bounds, DT, STEPS, OBS, GOAL      # noqa: E402
from affine_displacement import affine_displacement                    # noqa: E402
from auto_LiRPA import BoundedModule, BoundedTensor                    # noqa: E402
from auto_LiRPA.perturbations import PerturbationLpNorm                # noqa: E402

NET = pathlib.Path("REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx")
VARS = ("x", "y", "v", "th")
PREC = 6


def crown_affine(bm, lo, hi):
    """Linear CROWN bounds (lA, lbias, uA, ubias) for a batch of input boxes."""
    l = torch.tensor(lo, dtype=torch.float32)
    h = torch.tensor(hi, dtype=torch.float32)
    xb = BoundedTensor((l + h) / 2, PerturbationLpNorm(x_L=l, x_U=h))
    _, _, adict = bm.compute_bounds(
        x=(xb,), method="CROWN", return_A=True,
        needed_A_dict={bm.output_name[0]: [bm.input_name[0]]})
    a = adict[bm.output_name[0]][bm.input_name[0]]
    return (a["lA"].detach().numpy(), a["lbias"].detach().numpy(),
            a["uA"].detach().numpy(), a["ubias"].detach().numpy())


def affine_expr(coef, bias, lo, hi, lower: bool, names=VARS) -> str:
    """Emit a sound affine SMV expression, padding for coefficient rounding.

    For a LOWER bound the rounded form must never exceed the true one, so the
    accumulated rounding error (|dc_i| * max|s_i| plus the bias error) is
    subtracted; for an UPPER bound it is added.
    """
    rnd = np.round(coef, PREC)
    pad = float(np.sum(np.abs(coef - rnd) * np.maximum(np.abs(lo), np.abs(hi))))
    b = float(bias)
    rb = round(b, PREC)
    pad += abs(b - rb)
    const = (floor_to(rb - pad) if lower else ceil_to(rb + pad))
    terms = [f"{c:.6f} * {v}" for c, v in zip(rnd, names) if c != 0.0]
    terms.append(f"{const:.6f}")
    return " + ".join(terms)


def build(nx: int, nv: int, nvd: int | None = None, nth: int | None = None,
          affine_disp: bool = True, xr=(-1.0, 3.5), vr=(-2.0, 2.0)) -> str:
    nth = nth or nv
    xs = np.linspace(*xr, nx + 1)
    vs = np.linspace(*vr, nv + 1)
    ts = np.linspace(-math.pi, math.pi, nth + 1)
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))

    los, his = [], []
    for i in range(nx):
        for j in range(nx):
            for a in range(nv):
                for b in range(nth):
                    los.append([xs[i], xs[j], vs[a], ts[b]])
                    his.append([xs[i + 1], xs[j + 1], vs[a + 1], ts[b + 1]])
    lo_arr = np.array(los, np.float32)
    hi_arr = np.array(his, np.float32)
    lA, lbias, uA, ubias = crown_affine(bm, lo_arr, hi_arr)

    lines = ["MODULE main", "VAR",
             "  x : real;", "  y : real;", "  v : real;", "  th : real;",
             "  u1 : real;", "  u2 : real;", "  dx : real;", "  dy : real;",
             f"  k : 0..{STEPS};",
             "INIT x >= 2.9 & x <= 3.1 & y >= 2.9 & y <= 3.1 &"
             " v = 0.0 & th = 0.0 & k = 0;",
             # tanh bounds the control everywhere, cell or no cell
             "INVAR u1 >= -1.0 & u1 <= 1.0 & u2 >= -1.0 & u2 <= 1.0;",
             # |dx|,|dy| <= |v|*dt holds everywhere, including OUTSIDE the
             # contract grid.  Without it, once |v| leaves the modelled
             # range (it grows by up to dt per step, so up to 6 by k=30)
             # no displacement contract applies and the abstract system
             # can teleport.  Split on the sign of v to stay linear.
             f"INVAR (v >= 0) -> (dx <= {DT} * v & dx >= -{DT} * v &"
             f" dy <= {DT} * v & dy >= -{DT} * v);",
             f"INVAR (v <= 0) -> (dx <= -{DT} * v & dx >= {DT} * v &"
             f" dy <= -{DT} * v & dy >= {DT} * v);",
             f"TRANS next(k) = (k < {STEPS} ? k + 1 : {STEPS});",
             f"TRANS (k < {STEPS}) -> (next(x) = x + dx & next(y) = y + dy &"
             f" next(v) = v + {DT} * u1 & next(th) = th + {DT} * u2);",
             f"TRANS (k >= {STEPS}) -> (next(x) = x & next(y) = y &"
             " next(v) = v & next(th) = th);"]

    for n, (lo, hi) in enumerate(zip(lo_arr, hi_arr)):
        guard = " & ".join(f"{v} >= {lo[i]:.6f} & {v} <= {hi[i]:.6f}"
                           for i, v in enumerate(VARS))
        parts = []
        for out, name in ((0, "u1"), (1, "u2")):
            parts.append(f"{name} >= {affine_expr(lA[n][out], lbias[n][out], lo, hi, True)}")
            parts.append(f"{name} <= {affine_expr(uA[n][out], ubias[n][out], lo, hi, False)}")
        lines.append(f"INVAR ({guard}) -> ({' & '.join(parts)});")

    nvd = nvd or nv
    vd = np.linspace(*vr, nvd + 1)
    td = np.linspace(-math.pi, math.pi, nvd + 1)
    for a in range(nvd):
        for b in range(nvd):
            cell_lo = np.array([vd[a], td[b]])
            cell_hi = np.array([vd[a + 1], td[b + 1]])
            parts = []
            if affine_disp:
                for name, use_sin in (("dx", False), ("dy", True)):
                    av, ath, c, rad = affine_displacement(
                        vd[a], vd[a + 1], td[b], td[b + 1], DT, use_sin)
                    lo_e = affine_expr(np.array([av, ath]), c - rad,
                                       cell_lo, cell_hi, True, ("v", "th"))
                    hi_e = affine_expr(np.array([av, ath]), c + rad,
                                       cell_lo, cell_hi, False, ("v", "th"))
                    parts.append(f"{name} >= {lo_e} & {name} <= {hi_e}")
            else:
                (dxl, dxh), (dyl, dyh) = displacement_bounds(
                    vd[a], vd[a + 1], td[b], td[b + 1])
                parts.append(f"dx >= {dxl:.6f} & dx <= {dxh:.6f}")
                parts.append(f"dy >= {dyl:.6f} & dy <= {dyh:.6f}")
            lines.append(f"INVAR (v >= {vd[a]:.6f} & v <= {vd[a+1]:.6f} &"
                         f" th >= {td[b]:.6f} & th <= {td[b+1]:.6f}) ->"
                         f" ({' & '.join(parts)});")

    prop = sys.argv[3] if len(sys.argv) > 3 else "obstacle"
    if prop == "obstacle":
        lines.append(f"INVARSPEC !(x >= {OBS[0][0]} & x <= {OBS[0][1]} &"
                     f" y >= {OBS[1][0]} & y <= {OBS[1][1]});")
    elif prop == "goal":
        lines.append(f"INVARSPEC (k = {STEPS}) -> (x >= {GOAL[0][0]} &"
                     f" x <= {GOAL[0][1]} & y >= {GOAL[1][0]} & y <= {GOAL[1][1]});")
    else:
        lines.append(f"INVARSPEC k != {STEPS};")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    NX, NV = int(sys.argv[1]), int(sys.argv[2])
    def _arg(i):
        """Positional int argument, treating an empty string as absent."""
        return int(sys.argv[i]) if len(sys.argv) > i and sys.argv[i] else None

    NVD = _arg(4)
    NTH = _arg(5)
    # "net": affine network + interval displacement; "full": both affine
    ADISP = (sys.argv[6].lower() != "interval") if len(sys.argv) > 6 else True
    text = build(NX, NV, NVD, NTH, ADISP)
    tag = f"{NX}_{NV}" + (f"_d{NVD}" if NVD else "") + (f"_t{NTH}" if NTH else "")
    dest = pathlib.Path(f"COLLAB/experiments/nav_sound/"
                        f"affine{'full' if ADISP else 'net'}_{tag}_"
                        f"{sys.argv[3] if len(sys.argv) > 3 else 'obstacle'}.smv")
    dest.write_text(text)
    print(f"  {dest.name}: {len(text.splitlines())} lines, "
          f"{text.count('INVAR (')} contracts")
