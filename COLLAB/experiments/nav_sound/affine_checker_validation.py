"""
Does affine_admission.py actually detect unsound affine contracts?

A checker that passes everything proves nothing.  Corrupts the emitted bounds
three ways and confirms each is caught:
  swap    - emit the upper bound where the lower belongs (and vice versa)
  nopad   - drop the rounding pad and round coefficients the WRONG way
  shrink  - tighten both bounds toward the cell midpoint
"""
import math
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from gen_nav_contracts import load_net              # noqa: E402
from gen_nav_affine import crown_affine, affine_expr, VARS, NET  # noqa: E402
from affine_admission import edges, evaluate        # noqa: E402
from auto_LiRPA import BoundedModule                # noqa: E402

XR, VR = (-1.0, 3.5), (-2.0, 2.0)
DT, STEPS = 0.2, 30


def run(mode: str, nx=8, nv=8, nth=8) -> int:
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    sess = ort.InferenceSession(str(NET))
    inp = sess.get_inputs()[0].name
    viol = 0
    state = np.array([3.0, 3.0, 0.0, 0.0], np.float32)
    for _ in range(STEPS):
        u = sess.run(None, {inp: state.reshape(1, 4)})[0].ravel()
        box = (edges(*XR, nx, state[0]), edges(*XR, nx, state[1]),
               edges(*VR, nv, state[2]), edges(-math.pi, math.pi, nth, state[3]))
        lo = np.array([[b[0] for b in box]], np.float32)
        hi = np.array([[b[1] for b in box]], np.float32)
        lA, lbias, uA, ubias = crown_affine(bm, lo, hi)
        for out in (0, 1):
            elo = affine_expr(lA[0][out], lbias[0][out], lo[0], hi[0], True)
            ehi = affine_expr(uA[0][out], ubias[0][out], lo[0], hi[0], False)
            a, b = evaluate(elo, state), evaluate(ehi, state)
            if mode == "swap":
                a, b = b, a
            elif mode == "nopad":
                # round coefficients the wrong way with no compensating pad
                a = float(np.sum(np.ceil(lA[0][out] * 1e3) / 1e3 * state)
                          + math.ceil(lbias[0][out] * 1e3) / 1e3)
                b = float(np.sum(np.floor(uA[0][out] * 1e3) / 1e3 * state)
                          + math.floor(ubias[0][out] * 1e3) / 1e3)
            elif mode == "shrink":
                mid = (a + b) / 2
                a, b = mid - (b - a) * 0.1, mid + (b - a) * 0.1
            if not a <= u[out] <= b:
                viol += 1
        state = state + DT * np.array(
            [state[2] * np.cos(state[3]), state[2] * np.sin(state[3]),
             u[0], u[1]], np.float32)
    return viol


if __name__ == "__main__":
    print(f"{'corruption':>10} {'violations detected':>20}")
    for m in ("none", "swap", "nopad", "shrink"):
        n = run(m)
        flag = "(correct: clean)" if m == "none" and n == 0 else (
            "DETECTED" if n > 0 else "*** MISSED ***")
        print(f"{m:>10} {n:>20}  {flag}")
