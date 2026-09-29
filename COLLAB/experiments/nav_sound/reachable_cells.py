"""
How many grid cells does NAV actually reach?

Contracts are emitted for every cell of the 4-D box, but the system only ever
visits a thin tube.  Cells outside it are dead weight: dropping them is sound
(an unconstrained cell is simply more behaviour, i.e. a coarser
over-approximation) provided we also confirm the abstraction never enters one.

This measures the ceiling on that saving two ways: the cells the TRUE
trajectories touch, and a sound interval-propagated over-approximation of the
reachable tube, which is what a contract generator could actually restrict to.
"""
import itertools
import math
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort

NET = "REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx"
DT, STEPS = 0.2, 30
XR, VR = (-1.0, 3.5), (-2.0, 2.0)


def idx(val, lo, hi, n):
    """Grid index containing val, clamped."""
    return min(max(int((val - lo) / (hi - lo) * n), 0), n - 1)


def true_cells(nx, nv, nth, sess, inp):
    """Cells visited by simulated true trajectories from the initial set."""
    seen = set()
    for x0, y0 in itertools.product(np.linspace(2.9, 3.1, 7), repeat=2):
        state = np.array([x0, y0, 0.0, 0.0], np.float32)
        for _ in range(STEPS):
            u = sess.run(None, {inp: state.reshape(1, 4)})[0].ravel()
            seen.add((idx(state[0], *XR, nx), idx(state[1], *XR, nx),
                      idx(state[2], *VR, nv),
                      idx(state[3], -math.pi, math.pi, nth)))
            state = state + DT * np.array(
                [state[2] * np.cos(state[3]), state[2] * np.sin(state[3]),
                 u[0], u[1]], np.float32)
    return seen


# pylint: disable-next=too-many-locals
def tube_cells(nx, nv, nth):
    """Sound interval over-approximation of the reachable tube, as cells.

    Propagates a box forward under |u| <= 1 and |displacement| <= |v|*dt, which
    holds for every admissible control, so no reachable state is omitted.
    """
    box = [(2.9, 3.1), (2.9, 3.1), (0.0, 0.0), (0.0, 0.0)]
    seen = set()
    for _ in range(STEPS):
        (xl, xh), (yl, yh), (vl, vh), (tl, th) = box
        for i in range(idx(xl, *XR, nx), idx(xh, *XR, nx) + 1):
            for j in range(idx(yl, *XR, nx), idx(yh, *XR, nx) + 1):
                for a in range(idx(vl, *VR, nv), idx(vh, *VR, nv) + 1):
                    for b in range(idx(tl, -math.pi, math.pi, nth),
                                   idx(th, -math.pi, math.pi, nth) + 1):
                        seen.add((i, j, a, b))
        speed = max(abs(vl), abs(vh))
        box = [(xl - speed * DT, xh + speed * DT),
               (yl - speed * DT, yh + speed * DT),
               (max(vl - DT, VR[0]), min(vh + DT, VR[1])),
               (max(tl - DT, -math.pi), min(th + DT, math.pi))]
    return seen


def main():
    """Report total, tube and true cell counts for a few grids."""
    sess = ort.InferenceSession(NET)
    inp = sess.get_inputs()[0].name
    print(f"{'grid':>12} {'all cells':>10} {'sound tube':>11} {'true':>7} "
          f"{'tube saving':>12}")
    for nx, nv, nth in ((8, 8, 8), (12, 12, 12), (16, 16, 16)):
        total = nx * nx * nv * nth
        tube = len(tube_cells(nx, nv, nth))
        true = len(true_cells(nx, nv, nth, sess, inp))
        print(f"{nx}/{nv}/{nth:<6} {total:>10} {tube:>11} {true:>7} "
              f"{total / max(tube, 1):>11.1f}x")


if __name__ == "__main__":
    main()
