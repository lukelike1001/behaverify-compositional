"""
What grid does a given horizon actually need?

The obstacle margin is 0.2445.  A control interval of width w admits a velocity
error of dt*w/2 per step, and velocity integrates into position, so after k
steps the position error is about dt^2 * (w/2) * k^2/2.  Setting that below the
margin gives the width a horizon can tolerate - and hence the grid it needs.

This is a back-of-envelope scaling, not a proof: it tells us which grids are
worth trying, not which will succeed.
"""
import math
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from gen_nav_contracts import load_net, crown_bounds  # noqa: E402
from affine_contracts import cell, XR, VR, STATE      # noqa: E402
from gen_nav_affine import NET                        # noqa: E402
from auto_LiRPA import BoundedModule                  # noqa: E402

DT = 0.2
MARGIN = 0.2445


def tolerable_width(k):
    """Largest control-interval width whose drift stays under the margin."""
    return MARGIN / (DT ** 2 * k ** 2 / 4)


def measured_width(bm, nx, nv, nth):
    """CROWN interval width at the cell where the abstraction first failed."""
    x, y, v, th = STATE
    box = (cell(x, *XR, nx), cell(y, *XR, nx),
           cell(v, *VR, nv), cell(th, -math.pi, math.pi, nth))
    lo = np.array([[b[0] for b in box]], np.float32)
    hi = np.array([[b[1] for b in box]], np.float32)
    lb, ub = crown_bounds(bm, lo, hi)
    return float(ub[0][0] - lb[0][0])


def main():
    """Print the width each horizon tolerates, and what each grid delivers."""
    print("horizon -> largest tolerable control width")
    for k in (4, 8, 12, 16, 20, 25, 30):
        print(f"  k={k:<3} w <= {tolerable_width(k):.4f}")
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    print("\ngrid -> measured CROWN width, and the horizon it supports")
    for n in (8, 16, 32, 64, 128):
        w = measured_width(bm, n, n, n)
        k = math.sqrt(MARGIN / (DT ** 2 * w / 4)) if w > 0 else float("inf")
        print(f"  {n}/{n}/{n:<5} w = {w:.4f}   supports k ~ {k:.0f}")


if __name__ == "__main__":
    main()
