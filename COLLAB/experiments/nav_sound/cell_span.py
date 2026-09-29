"""
How many contract cells does the abstract state occupy at each step?

Refining the network grid 4x (8^3 -> 32^3) left the one-sided y gap unchanged
at 0.594 vs 0.595.  The explanation this tests: the abstract state does not sit
inside one cell.  It is spread over many, so the control it admits is the UNION
of those cells' contracts - and a finer grid simply means proportionally more,
smaller cells covering the same spread, leaving the union just as wide.

If so, the abstraction has a feedback loop - state spread widens the effective
contract, which widens the state spread - that cell size cannot break.

Usage: python3 .../cell_span.py <nx> <nv> <nvd> <K>
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import z3

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from z3_cegar import seed_cells                                     # noqa: E402
from z3_cegar_inc import IncrementalRefiner, cell_radii, expand     # noqa: E402

XR, VR = (-1.0, 3.5), (-2.0, 2.0)
AXES = ("x", "y", "v", "th")


def extent(solver, var):
    """Min and max of one Z3 variable over the current constraints."""
    out = []
    for want_max in (False, True):
        opt = z3.Optimize()
        for c in solver.assertions():
            opt.add(c)
        (opt.maximize if want_max else opt.minimize)(var)
        if opt.check() != z3.sat:
            return None
        out.append(float(opt.model().eval(
            var, model_completion=True).as_fraction()))
    return out


def main():
    """Report the abstract spread per axis, in cells, at each step."""
    nx, nv, nvd = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    horizon = int(sys.argv[4])
    nth = nv
    active, _ = seed_cells(nx, nv, nth, nvd)
    ref = IncrementalRefiner(nx, nv, nth, nvd, horizon, None)
    ref.add_cells(expand(active, (nx, nx, nv, nth), cell_radii(nx, nv, nth)))
    widths = ((XR[1] - XR[0]) / nx, (XR[1] - XR[0]) / nx,
              (VR[1] - VR[0]) / nv, 2 * np.pi / nth)
    print(f"net {nx}x{nv}x{nth}, disp {nvd}: spread in CELLS per axis")
    print(f"{'k':>3} " + " ".join(f"{a:>7}" for a in AXES))
    for k in range(horizon + 1):
        row = []
        for i in range(4):
            got = extent(ref.solver, ref.state[k][i])
            row.append((got[1] - got[0]) / widths[i] if got else float("nan"))
        print(f"{k:>3} " + " ".join(f"{c:>7.2f}" for c in row))


if __name__ == "__main__":
    main()
