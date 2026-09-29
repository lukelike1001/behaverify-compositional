"""
Do AFFINE contracts beat INTERVAL contracts at the same contract count?

CROWN computes a linear relaxation  lA.s + lb <= f(s) <= uA.s + ub  over each
cell and we have been CONCRETIZING it to a box, which is where the width comes
from.  Keeping the linear form costs the same one constraint per cell, stays in
linear real arithmetic, and is sound by exactly the same theorem.

This measures the tightness gain: mean affine width over the cell (sampled)
against the concretized interval width for the same cell.
"""
import math
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import sys

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
from gen_nav_contracts import load_net                                # noqa: E402
from auto_LiRPA import BoundedModule, BoundedTensor                   # noqa: E402
from auto_LiRPA.perturbations import PerturbationLpNorm               # noqa: E402

NET = "REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx"
XR, VR = (-1.0, 3.5), (-2.0, 2.0)
STATE = (2.012, 2.522, -1.554, 0.0)


def cell(val, lo, hi, n):
    edges = np.linspace(lo, hi, n + 1)
    i = min(max(int((val - lo) / (hi - lo) * n), 0), n - 1)
    return float(edges[i]), float(edges[i + 1])


def affine_and_interval(bm, lo, hi):
    """Return (lA, lbias, uA, ubias, interval_lo, interval_hi) for one box."""
    l = torch.tensor(lo, dtype=torch.float32)
    h = torch.tensor(hi, dtype=torch.float32)
    mid = (l + h) / 2
    xb = BoundedTensor(mid, PerturbationLpNorm(x_L=l, x_U=h))
    out = bm.compute_bounds(x=(xb,), method="CROWN", return_A=True,
                            needed_A_dict={bm.output_name[0]: [bm.input_name[0]]})
    lb, ub, adict = out
    a = adict[bm.output_name[0]][bm.input_name[0]]
    return (a["lA"].detach().numpy()[0], a["lbias"].detach().numpy()[0],
            a["uA"].detach().numpy()[0], a["ubias"].detach().numpy()[0],
            lb.detach().numpy()[0], ub.detach().numpy()[0])


def main():
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    print(f"{'nx/nv/nth':>12} {'contracts':>10} {'interval w':>11} "
          f"{'affine w (mean)':>16} {'gain':>7}")
    for nx, nv, nth in ((4, 4, 4), (8, 8, 8), (8, 16, 16), (16, 32, 16)):
        x, y, v, th = STATE
        box = (cell(x, *XR, nx), cell(y, *XR, nx),
               cell(v, *VR, nv), cell(th, -math.pi, math.pi, nth))
        lo = np.array([[b[0] for b in box]], np.float32)
        hi = np.array([[b[1] for b in box]], np.float32)
        lA, lbias, uA, ubias, ilo, ihi = affine_and_interval(bm, lo, hi)
        iw = float(ihi[0] - ilo[0])

        grids = [np.linspace(b[0], b[1], 8) for b in box]
        pts = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 4)
        widths = (pts @ (uA[0] - lA[0])) + (ubias[0] - lbias[0])
        aw = float(widths.mean())
        print(f"{nx}/{nv}/{nth:<6} {nx**2 * nv * nth:>10} {iw:>11.3f} "
              f"{aw:>16.3f} {iw / max(aw, 1e-9):>6.2f}x")


if __name__ == "__main__":
    main()
