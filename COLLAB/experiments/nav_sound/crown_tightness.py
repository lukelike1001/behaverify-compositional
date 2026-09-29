"""
Is the vacuous bound at the failure cell the CELL's fault or CROWN's?

Compares the CROWN interval against the true range of u1 over the same cell,
estimated by dense sampling.  Sampling under-approximates the true range, so
    true_range <= sampled_range <= CROWN
and a large CROWN/sampled ratio means the RELAXATION is loose, not the grid.
If so, a tighter bounding method (alpha-beta-CROWN with branch and bound)
would buy precision at the SAME contract count.
"""
import math
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort
import torch
import sys

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
from gen_nav_contracts import load_net, crown_bounds  # noqa: E402
from auto_LiRPA import BoundedModule                  # noqa: E402

NET = "REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx"
XR, VR = (-1.0, 3.5), (-2.0, 2.0)
STATE = (2.012, 2.522, -1.554, 0.0)   # true trajectory at k=8
SAMPLES = 12                          # per axis -> 12^4 = 20736 points


def cell(val, lo, hi, n):
    edges = np.linspace(lo, hi, n + 1)
    i = min(max(int((val - lo) / (hi - lo) * n), 0), n - 1)
    return float(edges[i]), float(edges[i + 1])


def main():
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    sess = ort.InferenceSession(NET)
    inp = sess.get_inputs()[0].name
    print(f"{'nx/nv/nth':>12} {'CROWN':>16} {'sampled true':>18} "
          f"{'CROWN w':>8} {'true w':>7} {'ratio':>6}")
    for nx, nv, nth in ((8, 8, 8), (8, 16, 16), (16, 32, 16)):
        x, y, v, th = STATE
        bx, by = cell(x, *XR, nx), cell(y, *XR, nx)
        bv = cell(v, *VR, nv)
        bt = cell(th, -math.pi, math.pi, nth)
        lo = np.array([[bx[0], by[0], bv[0], bt[0]]], np.float32)
        hi = np.array([[bx[1], by[1], bv[1], bt[1]]], np.float32)
        lb, ub = crown_bounds(bm, lo, hi)
        cw = float(ub[0][0] - lb[0][0])

        grids = [np.linspace(a, b, SAMPLES) for a, b in (bx, by, bv, bt)]
        pts = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 4)
        out = sess.run(None, {inp: pts.astype(np.float32)})[0][:, 0]
        tw = float(out.max() - out.min())
        print(f"{nx}/{nv}/{nth:<6} [{lb[0][0]:>6.3f},{ub[0][0]:>6.3f}] "
              f"  [{out.min():>6.3f},{out.max():>6.3f}]  "
              f"{cw:>8.3f} {tw:>7.3f} {cw / max(tw, 1e-9):>6.2f}x")


if __name__ == "__main__":
    main()
