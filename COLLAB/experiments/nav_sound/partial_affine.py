"""
Can we keep affine tightness without affine density?

A full affine contract couples the control to all four state variables, which
turns every contract into a dense linear constraint for the theory solver.  A
PARTIAL affine contract keeps the linear term only on the axes that matter and
CONCRETIZES the rest: for a dropped axis i, the term a_i*s_i is replaced by its
minimum over the cell (for a lower bound) or its maximum (for an upper bound).
That is sound - it is the same concretization that produces interval contracts,
just applied to fewer axes - and it yields a sparser constraint.

Measured at the cell where the abstraction failed with intervals.
"""
import math
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from gen_nav_contracts import load_net                  # noqa: E402
from gen_nav_affine import crown_affine, NET            # noqa: E402
from affine_contracts import cell, XR, VR, STATE        # noqa: E402
from auto_LiRPA import BoundedModule                    # noqa: E402

VARS = ("x", "y", "v", "th")


# pylint: disable-next=too-many-locals
def widths(bm, nx, nv, nth, keep):
    """Mean width over the cell when only axes in `keep` stay linear."""
    x, y, v, th = STATE
    box = (cell(x, *XR, nx), cell(y, *XR, nx),
           cell(v, *VR, nv), cell(th, -math.pi, math.pi, nth))
    lo = np.array([[b[0] for b in box]], np.float32)
    hi = np.array([[b[1] for b in box]], np.float32)
    l_a, lbias, u_a, ubias = crown_affine(bm, lo, hi)
    la, ua = l_a[0][0].copy(), u_a[0][0].copy()
    lb, ub = float(lbias[0][0]), float(ubias[0][0])

    for i in range(4):
        if VARS[i] in keep:
            continue
        # concretize axis i: worst case of a_i*s_i over the cell
        ends = (la[i] * lo[0][i], la[i] * hi[0][i])
        lb += min(ends)
        ends = (ua[i] * lo[0][i], ua[i] * hi[0][i])
        ub += max(ends)
        la[i] = ua[i] = 0.0

    grids = [np.linspace(b[0], b[1], 8) for b in box]
    pts = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 4)
    return float(((pts @ (ua - la)) + (ub - lb)).mean())


def main():  # pylint: disable=too-many-locals
    """Compare full, partial and fully-concretized contract forms."""
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
    for nx, nv, nth in ((6, 6, 6), (8, 8, 8)):
        print(f"\ngrid {nx}/{nv}/{nth}  ({nx**2 * nv * nth} contracts)")
        print(f"  {'linear axes':>22} {'width':>8} {'coupled vars':>13}")
        for keep in (VARS, ("v", "th"), ("v",), ("th",), ()):
            label = ",".join(keep) if keep else "none (= interval)"
            print(f"  {label:>22} {widths(bm, nx, nv, nth, keep):>8.3f} "
                  f"{len(keep):>13}")


if __name__ == "__main__":
    main()
