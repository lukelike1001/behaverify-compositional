"""
Where does the abstraction's error actually accumulate?

The width budget in horizon_budget.py infers the error from a single
worst-case cell, and it underpredicts the 8^3 grid (it says k~3; that grid
proves k=8), so it is not to be trusted about WHY k=9 resists.

This measures the thing directly: for each step, ask Z3 for the minimum and
maximum of each state variable over ALL behaviours the abstraction admits, and
compare against the spread of the true trajectories from the same initial set.
The gap between the two is the over-approximation error, step by step, and its
growth says whether the error is injected early and integrated, or accumulated
steadily.

Usage: python3 .../drift_profile.py <nx> <nv> <nvd> <K>
"""
import itertools
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort
import z3

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
sys.path.insert(0, "COLLAB/experiments/nav_sound")
from gen_nav_affine import NET                                      # noqa: E402
from z3_cegar import seed_cells                                     # noqa: E402
from z3_cegar_inc import IncrementalRefiner, cell_radii, expand     # noqa: E402


DT = 0.2
XR, VR = (-1.0, 3.5), (-2.0, 2.0)


def true_spread(horizon):
    """Min and max of each state variable over simulated true trajectories."""
    sess = ort.InferenceSession(str(NET))
    inp = sess.get_inputs()[0].name
    traces = []
    for x0, y0 in itertools.product(np.linspace(2.9, 3.1, 9), repeat=2):
        st = np.array([x0, y0, 0.0, 0.0], np.float32)
        row = [st.copy()]
        for _ in range(horizon):
            u = sess.run(None, {inp: st.reshape(1, 4)})[0].ravel()
            st = st + DT * np.array([st[2] * np.cos(st[3]),
                                     st[2] * np.sin(st[3]), u[0], u[1]],
                                    np.float32)
            row.append(st.copy())
        traces.append(row)
    arr = np.array(traces)            # (starts, steps, 4)
    return arr.min(axis=0), arr.max(axis=0)


# pylint: disable-next=too-many-locals
def main():
    """Print abstract vs true reachable interval for y at each step."""
    nx, nv, nvd = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    horizon = int(sys.argv[4])
    nth = nv
    active, _ = seed_cells(nx, nv, nth, nvd)
    # ALL plant cells: seeding them from the true path leaves the abstraction
    # free to enter (v, th) cells the true system never visits - notably v > 0 -
    # where only the global |dy| <= |v|*dt bound applies, which is far weaker
    # than the per-cell one.
    ref = IncrementalRefiner(nx, nv, nth, nvd, horizon, None)
    ref.add_cells(expand(active, (nx, nx, nv, nth), cell_radii(nx, nv, nth)))
    ref.add_plant_cells(ref.plant_cells)

    lo_t, hi_t = true_spread(horizon)
    print(f"net {nx}x{nv}x{nth}, disp {nvd}, horizon {horizon}")
    # The quantity that decides safety is the ONE-SIDED gap between the
    # abstraction's lower bound and the true one, not the interval width: an
    # interval that is merely shifted down still enters the obstacle.
    axis = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    name = ("x", "y", "v", "th")[axis]
    print(f"{'k':>3} {'abstract ' + name:>20} {'true ' + name:>18} "
          f"{'lo gap':>8} {'excess w':>9}")
    for k in range(horizon + 1):
        bounds = []
        for want_max in (False, True):
            opt = z3.Optimize()
            for c in ref.solver.assertions():
                opt.add(c)
            if want_max:
                opt.maximize(ref.state[k][axis])
            else:
                opt.minimize(ref.state[k][axis])
            if opt.check() != z3.sat:
                bounds.append(float("nan"))
                continue
            bounds.append(float(opt.model().eval(
                ref.state[k][axis], model_completion=True).as_fraction()))
        lo_gap = float(lo_t[k][axis]) - bounds[0]
        excess = (bounds[1] - bounds[0]) - float(hi_t[k][axis] - lo_t[k][axis])
        print(f"{k:>3} [{bounds[0]:>8.3f},{bounds[1]:>8.3f}] "
              f"[{lo_t[k][axis]:>7.3f},{hi_t[k][axis]:>7.3f}] "
              f"{lo_gap:>8.3f} {excess:>9.3f}")


if __name__ == "__main__":
    main()
