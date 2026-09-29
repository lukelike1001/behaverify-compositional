"""
End-to-end soundness precondition for the FULLY affine NAV model.

Checks BOTH contract families against the true closed loop at once:
  - the affine CROWN contract on the network output, and
  - the affine Taylor-with-remainder contract on the displacement.

A verdict of "is true" from nuXmv is only meaningful if the true system is
inside the abstraction, so both families must admit every true state.  As
before this can only refute soundness, never establish it.
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
from gen_nav_contracts import load_net                              # noqa: E402
from gen_nav_affine import crown_affine, affine_expr, NET           # noqa: E402
from affine_displacement import affine_displacement                 # noqa: E402
from affine_admission import edges, evaluate                        # noqa: E402
from auto_LiRPA import BoundedModule                                # noqa: E402

XR, VR = (-1.0, 3.5), (-2.0, 2.0)
DT, STEPS = 0.2, 30


def main():
    nx, nv = int(sys.argv[1]), int(sys.argv[2])
    nvd = int(sys.argv[3]) if len(sys.argv) > 3 else nv
    nth = nv
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    sess = ort.InferenceSession(str(NET))
    inp = sess.get_inputs()[0].name

    net_viol = disp_viol = total = 0
    for x0, y0 in itertools.product(np.linspace(2.9, 3.1, 3), repeat=2):
        state = np.array([x0, y0, 0.0, 0.0], np.float32)
        for _ in range(STEPS):
            u = sess.run(None, {inp: state.reshape(1, 4)})[0].ravel()

            box = (edges(*XR, nx, state[0]), edges(*XR, nx, state[1]),
                   edges(*VR, nv, state[2]),
                   edges(-math.pi, math.pi, nth, state[3]))
            lo = np.array([[b[0] for b in box]], np.float32)
            hi = np.array([[b[1] for b in box]], np.float32)
            lA, lbias, uA, ubias = crown_affine(bm, lo, hi)
            for out in (0, 1):
                elo = affine_expr(lA[0][out], lbias[0][out], lo[0], hi[0], True)
                ehi = affine_expr(uA[0][out], ubias[0][out], lo[0], hi[0], False)
                if not evaluate(elo, state) <= u[out] <= evaluate(ehi, state):
                    net_viol += 1

            dbox = (edges(*VR, nvd, state[2]),
                    edges(-math.pi, math.pi, nvd, state[3]))
            clo = np.array([dbox[0][0], dbox[1][0]])
            chi = np.array([dbox[0][1], dbox[1][1]])
            truth = (float(state[2] * np.cos(state[3]) * DT),
                     float(state[2] * np.sin(state[3]) * DT))
            for idx, use_sin in ((0, False), (1, True)):
                av, ath, c, rad = affine_displacement(
                    clo[0], chi[0], clo[1], chi[1], DT, use_sin)
                elo = affine_expr(np.array([av, ath]), c - rad, clo, chi,
                                  True, ("v", "th"))
                ehi = affine_expr(np.array([av, ath]), c + rad, clo, chi,
                                  False, ("v", "th"))
                env = np.array([state[2], state[3]])
                names = ("v", "th")
                if not (evaluate(elo, env, names) <= truth[idx]
                        <= evaluate(ehi, env, names)):
                    disp_viol += 1

            total += 1
            state = state + DT * np.array(
                [state[2] * np.cos(state[3]), state[2] * np.sin(state[3]),
                 u[0], u[1]], np.float32)

    print(f"grid net {nx}x{nv}, disp {nvd}: {total} states")
    print(f"  network contract violations:      {net_viol}")
    print(f"  displacement contract violations: {disp_viol}")
    print("SOUND (necessary condition holds)" if net_viol == disp_viol == 0
          else "*** UNSOUND ***")


if __name__ == "__main__":
    main()
