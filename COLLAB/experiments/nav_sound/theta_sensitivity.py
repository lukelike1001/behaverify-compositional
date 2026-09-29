"""
Is the CROWN bound blowup driven by theta resolution or by v/x resolution?

Holds the state fixed at the point where the bound went vacuous (k=8 on the
true trajectory) and varies each grid axis one at a time.
"""
import math
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import sys

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
from gen_nav_contracts import load_net, crown_bounds  # noqa: E402
from auto_LiRPA import BoundedModule                  # noqa: E402

NET = "REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx"
XR, VR = (-1.0, 3.5), (-2.0, 2.0)
STATE = (2.012, 2.522, -1.554, 0.0)   # true trajectory at k=8


def cell(val, lo, hi, n):
    """Interval [edge_lo, edge_hi] of the grid cell containing val."""
    edges = np.linspace(lo, hi, n + 1)
    i = min(max(int((val - lo) / (hi - lo) * n), 0), n - 1)
    return edges[i], edges[i + 1]


def width_at(nx, nv, nth, bm):
    """CROWN u1 interval width for the cell containing STATE."""
    x, y, v, th = STATE
    lo, hi = zip(cell(x, *XR, nx), cell(y, *XR, nx),
                 cell(v, *VR, nv), cell(th, -math.pi, math.pi, nth))
    lb, ub = crown_bounds(bm, np.array([lo], np.float32), np.array([hi], np.float32))
    return float(ub[0][0] - lb[0][0])


def main():
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    base = (8, 8, 8)
    print(f"baseline nx=8 nv=8 nth=8 -> width {width_at(*base, bm):.3f}\n")
    for axis, name in enumerate(("nx", "nv", "nth")):
        print(f"varying {name} (others held at 8):")
        for n in (4, 8, 16, 32, 64):
            cfg = list(base)
            cfg[axis] = n
            print(f"   {name}={n:<3} width {width_at(*cfg, bm):.3f}")
        print()


if __name__ == "__main__":
    main()


def combined():
    """Width and contract cost under simultaneous refinement of all three axes.

    Required width: position error over 8 steps is ~0.64*w (two integrations of
    a v-error of dt*w/2 per step), and the true trajectory's margin to the
    obstacle is 0.2445, so we need w < ~0.38.
    """
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    print(f"{'nx':>4} {'nv':>4} {'nth':>4} {'contracts':>10} {'width':>7} "
          f"{'pos err':>8}")
    for nx, nv, nth in ((4, 8, 8), (8, 8, 8), (8, 16, 8), (8, 16, 16),
                        (8, 32, 16), (16, 32, 16), (16, 32, 32),
                        (16, 64, 32), (32, 64, 32), (32, 64, 64)):
        w = width_at(nx, nv, nth, bm)
        print(f"{nx:>4} {nv:>4} {nth:>4} {nx**2 * nv * nth:>10} {w:>7.3f} "
              f"{0.64 * w:>8.3f}")


if __name__ != "__main__":
    pass
