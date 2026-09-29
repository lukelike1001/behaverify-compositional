"""
Where does the abstraction's error actually come from?

At every state on the TRUE trajectory, compare:
  - the true control u the network emits, and
  - the CROWN interval for the grid cell containing that state.

If the interval is wide, the bound is loose (refining the grid helps).
If the interval is tight but the true u sits near an edge, the grid is fine
and the error is compounding through the dynamics instead.

Usage: python3 COLLAB/experiments/nav_sound/slack_check.py <nx> <nv>
"""
import math
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
from gen_nav_contracts import load_net, crown_bounds  # noqa: E402
from auto_LiRPA import BoundedModule                  # noqa: E402

NET = "REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx"
DT, STEPS = 0.2, 30
XR, VR = (-1.0, 3.5), (-2.0, 2.0)


def cell_of(val, lo, hi, n):
    """Index of the grid cell containing val, clamped to the grid."""
    idx = int((val - lo) / (hi - lo) * n)
    return min(max(idx, 0), n - 1)


def main():
    nx, nv = int(sys.argv[1]), int(sys.argv[2])
    xs = np.linspace(*XR, nx + 1)
    vs = np.linspace(*VR, nv + 1)
    ts = np.linspace(-math.pi, math.pi, nv + 1)
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    sess = ort.InferenceSession(NET)
    inp = sess.get_inputs()[0].name

    state = np.array([3.0, 3.0, 0.0, 0.0], np.float32)
    print(f"grid {nx}x{nv} ({nx**2 * nv**2} network contracts)")
    print(f"{'k':>3} {'x':>7} {'y':>7} {'v':>7} {'u1_true':>8} "
          f"{'u1_lo':>7} {'u1_hi':>7} {'width':>7}")
    widths = []
    for k in range(STEPS):
        u = sess.run(None, {inp: state.reshape(1, 4)})[0].ravel()
        i = cell_of(state[0], *XR, nx)
        j = cell_of(state[1], *XR, nx)
        a = cell_of(state[2], *VR, nv)
        b = cell_of(state[3], -math.pi, math.pi, nv)
        lo = np.array([[xs[i], xs[j], vs[a], ts[b]]], np.float32)
        hi = np.array([[xs[i + 1], xs[j + 1], vs[a + 1], ts[b + 1]]], np.float32)
        lb, ub = crown_bounds(bm, lo, hi)
        w = float(ub[0][0] - lb[0][0])
        widths.append(w)
        if k < 14:
            print(f"{k:>3} {state[0]:>7.3f} {state[1]:>7.3f} {state[2]:>7.3f} "
                  f"{u[0]:>8.3f} {lb[0][0]:>7.3f} {ub[0][0]:>7.3f} {w:>7.3f}")
        state = state + DT * np.array(
            [state[2] * np.cos(state[3]), state[2] * np.sin(state[3]),
             u[0], u[1]], np.float32)
    print(f"\nmean u1 interval width on the true trajectory: {np.mean(widths):.3f}")
    print(f"max  u1 interval width on the true trajectory: {np.max(widths):.3f}")
    print(f"worst-case |v| drift after 8 steps: "
          f"{DT * sum(sorted(widths)[-8:]) / 2:.3f} "
          f"(half-width x dt, summed over the 8 widest cells)")


if __name__ == "__main__":
    main()
