"""
Adversarial stress test of the AFFINE rounding pad.

The trajectory admission check only visits states the true system reaches.  The
pad has to hold at EVERY point of EVERY cell, including cell corners where
|s_i| is largest and the coefficient rounding error is most amplified.  This
samples random cells and, within each, the corners plus random interior points.

A single violation means the emitted contracts exclude a genuine behaviour and
every verdict built on them is void.
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
from gen_nav_contracts import load_net                          # noqa: E402
from gen_nav_affine import crown_affine, affine_expr, NET       # noqa: E402
from affine_admission import evaluate                           # noqa: E402
from auto_LiRPA import BoundedModule                            # noqa: E402

XR, VR = (-1.0, 3.5), (-2.0, 2.0)


def main():  # pylint: disable=too-many-locals
    """Stress the rounding pad at cell corners and interior points."""
    nx, nv, nth = 8, 8, 8
    n_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(0)
    xs = np.linspace(*XR, nx + 1)
    vs = np.linspace(*VR, nv + 1)
    ts = np.linspace(-math.pi, math.pi, nth + 1)
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    sess = ort.InferenceSession(str(NET))
    inp = sess.get_inputs()[0].name

    viol = checked = 0
    worst = -math.inf   # most positive (lv-o, o-uv); >0 means violation
    for _ in range(n_cells):
        i, j = rng.integers(0, nx, 2)
        a = rng.integers(0, nv)
        b = rng.integers(0, nth)
        lo = np.array([[xs[i], xs[j], vs[a], ts[b]]], np.float32)
        hi = np.array([[xs[i + 1], xs[j + 1], vs[a + 1], ts[b + 1]]], np.float32)
        l_a, lbias, u_a, ubias = crown_affine(bm, lo, hi)

        corners = np.array(list(itertools.product(*zip(lo[0], hi[0]))), np.float32)
        interior = rng.uniform(lo[0], hi[0], size=(40, 4)).astype(np.float32)
        pts = np.vstack([corners, interior])
        outs = sess.run(None, {inp: pts})[0]

        for out in (0, 1):
            elo = affine_expr(l_a[0][out], lbias[0][out], lo[0], hi[0], True)
            ehi = affine_expr(u_a[0][out], ubias[0][out], lo[0], hi[0], False)
            for p, o in zip(pts, outs[:, out]):
                lv, uv = evaluate(elo, p), evaluate(ehi, p)
                worst = max(worst, lv - o, o - uv)
                if not lv <= o <= uv:
                    viol += 1
                checked += 1

    print(f"{n_cells} random cells, {checked} bound checks "
          f"({len(corners)} corners + 40 interior points per cell per output)")
    print(f"violations: {viol}")
    print(f"tightest point: {worst:+.3e} "
          f"(negative = slack remaining; positive = VIOLATION)")
    print("PAD HOLDS" if viol == 0 else "*** PAD IS UNSOUND ***")


if __name__ == "__main__":
    main()
