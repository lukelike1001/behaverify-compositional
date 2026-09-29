"""
Soundness precondition for the AFFINE contracts: do they admit the true system?

Same role as admission_check.py but for the linear form.  At every state on
every simulated trajectory, the true network output must satisfy the emitted
affine bounds for the cell containing that state, INCLUDING the rounding pad.
This can only refute soundness, never establish it; it catches errors in
coefficient rounding direction, pad computation and cell indexing.
"""
import itertools
import math
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from gen_nav_contracts import load_net                  # noqa: E402
from gen_nav_affine import crown_affine, affine_expr, VARS, PREC, NET  # noqa: E402
from auto_LiRPA import BoundedModule                    # noqa: E402

XR, VR = (-1.0, 3.5), (-2.0, 2.0)
DT, STEPS = 0.2, 30


def edges(lo, hi, n, val):
    e = np.linspace(lo, hi, n + 1)
    i = min(max(int((val - lo) / (hi - lo) * n), 0), n - 1)
    return float(e[i]), float(e[i + 1])


def evaluate(expr: str, state, names=VARS) -> float:
    """Evaluate an emitted SMV affine expression at a concrete state."""
    env = dict(zip(names, (float(s) for s in state)))
    return float(eval(expr, {"__builtins__": {}}, env))  # noqa: S307


def main():
    nx, nv = int(sys.argv[1]), int(sys.argv[2])
    nth = int(sys.argv[3]) if len(sys.argv) > 3 else nv
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    sess = ort.InferenceSession(str(NET))
    inp = sess.get_inputs()[0].name

    viol = outside = total = 0
    for x0, y0 in itertools.product(np.linspace(2.9, 3.1, 3), repeat=2):
        state = np.array([x0, y0, 0.0, 0.0], np.float32)
        for _ in range(STEPS):
            u = sess.run(None, {inp: state.reshape(1, 4)})[0].ravel()
            box = (edges(*XR, nx, state[0]), edges(*XR, nx, state[1]),
                   edges(*VR, nv, state[2]),
                   edges(-math.pi, math.pi, nth, state[3]))
            lo = np.array([[b[0] for b in box]], np.float32)
            hi = np.array([[b[1] for b in box]], np.float32)
            if not all(b[0] - 1e-6 <= s <= b[1] + 1e-6
                       for b, s in zip(box, state)):
                outside += 1
            lA, lbias, uA, ubias = crown_affine(bm, lo, hi)
            for out in (0, 1):
                elo = affine_expr(lA[0][out], lbias[0][out], lo[0], hi[0], True)
                ehi = affine_expr(uA[0][out], ubias[0][out], lo[0], hi[0], False)
                if not (evaluate(elo, state) <= u[out] <= evaluate(ehi, state)):
                    viol += 1
            total += 1
            state = state + DT * np.array(
                [state[2] * np.cos(state[3]), state[2] * np.sin(state[3]),
                 u[0], u[1]], np.float32)
    print(f"grid {nx}x{nv}x{nth}: {total} states, "
          f"{viol} contract violations, {outside} outside the modelled box")
    print("SOUND (necessary condition holds)" if viol == 0
          else "UNSOUND - affine contracts exclude the true system")


if __name__ == "__main__":
    main()
