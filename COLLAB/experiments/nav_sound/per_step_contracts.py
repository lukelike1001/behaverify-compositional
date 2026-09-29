"""
One contract per STEP, fitted to the reachable set, instead of one per cell.

The feedback loop that caps the grid approach is that once the abstract state
spans several cells, the admitted control is the UNION of those cells'
contracts - and the cells OVER-COVER the reachable set, since a cell that only
partly intersects it contributes its whole contract.

This drops the grid entirely.  At each step, CROWN is evaluated on a box fitted
to the current reachable set, giving one control interval; the box is then
propagated.  Sound for the same reason as before - the true state is inside the
box at every step, so the true control is inside the bound - and it costs one
CROWN call per step rather than thousands of contracts.

The price is the wrapping effect: the box cannot represent correlation between
variables, so it inflates. Whether that costs more than the union it removes is
exactly what this measures.

Usage: python3 .../per_step_contracts.py [steps]
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from gen_nav_contracts import load_net, crown_bounds                # noqa: E402
from gen_nav_affine import NET                                      # noqa: E402
from margin_check import dist_to_box                                # noqa: E402
from auto_LiRPA import BoundedModule                                # noqa: E402

DT = 0.2
OBS = ((1.0, 2.0), (1.0, 2.0))


def displacement_box(vlo, vhi, tlo, thi, samples=600):
    """Sound interval bounds on (v cos th, v sin th)*dt over a (v, th) box."""
    ts = np.linspace(tlo, thi, samples)
    cos, sin = np.cos(ts), np.sin(ts)
    dxs = np.concatenate([vlo * cos, vhi * cos]) * DT
    dys = np.concatenate([vlo * sin, vhi * sin]) * DT
    pad = abs(vhi - vlo) * (thi - tlo) * DT
    return ((dxs.min() - pad, dxs.max() + pad),
            (dys.min() - pad, dys.max() + pad))


# pylint: disable-next=too-many-locals
def main():
    """Propagate the reachable box and report when it touches the obstacle."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    bmod = BoundedModule(load_net(NET), torch.zeros(1, 4))
    box = [(2.9, 3.1), (2.9, 3.1), (0.0, 0.0), (0.0, 0.0)]
    print(f"{'k':>3} {'x':>16} {'y':>16} {'v':>16} {'u1 width':>9} {'dist':>7}")
    for k in range(steps + 1):
        (xlo, xhi), (ylo, yhi), (vlo, vhi), (tlo, thi) = box
        corners = [(x, y) for x in (xlo, xhi) for y in (ylo, yhi)]
        dist = 0.0 if (xlo <= OBS[0][1] and xhi >= OBS[0][0]
                       and ylo <= OBS[1][1] and yhi >= OBS[1][0]) \
            else min(dist_to_box(x, y) for x, y in corners)
        lo = np.array([[xlo, ylo, vlo, tlo]], np.float32)
        hi = np.array([[xhi, yhi, vhi, thi]], np.float32)
        lb, ub = crown_bounds(bmod, lo, hi)
        u1w = float(ub[0][0] - lb[0][0])
        print(f"{k:>3} [{xlo:>6.3f},{xhi:>6.3f}] [{ylo:>6.3f},{yhi:>6.3f}] "
              f"[{vlo:>6.3f},{vhi:>6.3f}] {u1w:>9.4f} {dist:>7.3f}")
        if dist == 0.0:
            print(f"  reachable box touches the obstacle at k={k}")
            return
        if k == steps:
            break
        (dxl, dxh), (dyl, dyh) = displacement_box(vlo, vhi, tlo, thi)
        box = [(xlo + dxl, xhi + dxh), (ylo + dyl, yhi + dyh),
               (vlo + DT * max(float(lb[0][0]), -1.0),
                vhi + DT * min(float(ub[0][0]), 1.0)),
               (tlo + DT * max(float(lb[0][1]), -1.0),
                thi + DT * min(float(ub[0][1]), 1.0))]
    print(f"  obstacle avoided through k={steps}")


if __name__ == "__main__":
    main()


# pylint: disable-next=too-many-locals
def admission(steps=12):
    """Soundness precondition: every true trajectory stays inside the boxes.

    The scheme is only an over-approximation if the reachable box really does
    contain the true state at every step; if a true trajectory escapes, the
    CROWN bound was taken over the wrong region and every verdict is void.
    """
    import itertools
    import onnxruntime as ort
    bmod = BoundedModule(load_net(NET), torch.zeros(1, 4))
    sess = ort.InferenceSession(str(NET))
    inp = sess.get_inputs()[0].name

    boxes, box = [], [(2.9, 3.1), (2.9, 3.1), (0.0, 0.0), (0.0, 0.0)]
    for _ in range(steps + 1):
        boxes.append(box)
        (xlo, xhi), (ylo, yhi), (vlo, vhi), (tlo, thi) = box
        lo = np.array([[xlo, ylo, vlo, tlo]], np.float32)
        hi = np.array([[xhi, yhi, vhi, thi]], np.float32)
        lb, ub = crown_bounds(bmod, lo, hi)
        (dxl, dxh), (dyl, dyh) = displacement_box(vlo, vhi, tlo, thi)
        box = [(xlo + dxl, xhi + dxh), (ylo + dyl, yhi + dyh),
               (vlo + DT * max(float(lb[0][0]), -1.0),
                vhi + DT * min(float(ub[0][0]), 1.0)),
               (tlo + DT * max(float(lb[0][1]), -1.0),
                thi + DT * min(float(ub[0][1]), 1.0))]

    escapes = checked = 0
    for x0, y0 in itertools.product(np.linspace(2.9, 3.1, 11), repeat=2):
        st = np.array([x0, y0, 0.0, 0.0], np.float32)
        for k in range(steps + 1):
            for i in range(4):
                lo_i, hi_i = boxes[k][i]
                if not lo_i - 1e-6 <= st[i] <= hi_i + 1e-6:
                    escapes += 1
            checked += 4
            u = sess.run(None, {inp: st.reshape(1, 4)})[0].ravel()
            st = st + DT * np.array([st[2] * np.cos(st[3]),
                                     st[2] * np.sin(st[3]), u[0], u[1]],
                                    np.float32)
    print(f"admission over {checked} variable-checks: {escapes} escapes")
    print("SOUND (necessary condition holds)" if escapes == 0
          else "*** UNSOUND: a true trajectory left the reachable box ***")
