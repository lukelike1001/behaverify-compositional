"""
Compositional verification vs per-step interval reachability, head to head, on
a benchmark designed so the comparison is meaningful.

On NAV, per-step reachability WON (step 10 with 10 contracts and no solver,
against step 8 with ~1,300 contracts and Z3) because NAV's tree is a single
action node: nothing ever branches, so one box tracks the system exactly.

The geofence NSBT branches on the sign of the acceleration, which makes the
reachable set non-convex (COLLAB/experiments/geofence/box_must_fail.py).  This
runs both methods on the same system, the same contracts and the same property,
so the only difference is whether the branch is kept as a DISJUNCTION or
collapsed into a HULL.

Both are sound: each admits at least every behaviour the true system has, so a
proof from either is a proof about the real closed loop.

Usage: python3 compare.py [steps] [v0_bound] [grid]
"""
from __future__ import annotations

import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import z3

from train_controller import Controller
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from nsbt import BRAKE, DT, FENCE, HARD_LIMIT, NET

X0_LO, X0_HI = -2.0, 0.9
V_RANGE = (-1.5, 1.5)          # the box the contracts are built over


# pylint: disable-next=too-many-locals
def crown_cells(grid):
    """CROWN bounds on the network's acceleration, per cell of an (x,v) grid."""
    model = Controller()
    model.load_state_dict(torch.load(
        pathlib.Path(NET).with_suffix(".pt"), weights_only=True))
    model.eval()
    bounded = BoundedModule(model, torch.zeros(1, 2))
    xs = np.linspace(X0_LO, FENCE, grid + 1)
    vs = np.linspace(*V_RANGE, grid + 1)
    cells = []
    for i in range(grid):
        for j in range(grid):
            lo = np.array([[xs[i], vs[j]]], np.float32)
            hi = np.array([[xs[i + 1], vs[j + 1]]], np.float32)
            mid = torch.tensor((lo + hi) / 2)
            xb = BoundedTensor(mid, PerturbationLpNorm(
                x_L=torch.tensor(lo), x_U=torch.tensor(hi)))
            lb, ub = bounded.compute_bounds(x=(xb,), method="CROWN")
            cells.append((float(xs[i]), float(xs[i + 1]),
                          float(vs[j]), float(vs[j + 1]),
                          max(float(lb[0][0]), -1.0), min(float(ub[0][0]), 1.0)))
    return cells


# pylint: disable-next=too-many-locals
def compositional(cells, steps, v0_bound):
    """Keep the branch as a guard, so each state uses its own controller."""
    solver = z3.Solver()
    x = [z3.Real(f"x_{k}") for k in range(steps + 1)]
    v = [z3.Real(f"v_{k}") for k in range(steps + 1)]
    solver.add(x[0] >= X0_LO, x[0] <= X0_HI,
               v[0] >= -v0_bound, v[0] <= v0_bound)
    for k in range(steps):
        a = z3.Real(f"a_{k}")
        solver.add(a >= -1, a <= 1)
        # the behavior tree: outside the fence brake, inside use the network
        solver.add(z3.Implies(x[k] >= FENCE, a == -BRAKE))
        for xlo, xhi, vlo, vhi, alo, ahi in cells:
            guard = z3.And(x[k] < FENCE,
                           x[k] >= xlo, x[k] <= xhi, v[k] >= vlo, v[k] <= vhi)
            solver.add(z3.Implies(guard, z3.And(a >= alo, a <= ahi)))
        solver.add(x[k + 1] == x[k] + v[k] * DT, v[k + 1] == v[k] + a * DT)
    solver.add(z3.Or([x[k] > HARD_LIMIT for k in range(steps + 1)]))
    t0 = time.time()
    res = solver.check()
    return res, time.time() - t0


# pylint: disable-next=too-many-locals
def per_step_box(cells, steps, v0_bound):
    """One box per step, so a straddling box must admit BOTH branches."""
    xlo, xhi, vlo, vhi = X0_LO, X0_HI, -v0_bound, v0_bound
    worst = xhi
    for _ in range(steps):
        if xhi < FENCE:                       # wholly inside: network only
            alo, ahi = _crown_hull(cells, xlo, xhi, vlo, vhi)
        elif xlo >= FENCE:                    # wholly outside: brake only
            alo = ahi = -BRAKE
        else:                                 # STRADDLES: must admit both
            clo, chi = _crown_hull(cells, xlo, min(xhi, FENCE), vlo, vhi)
            alo, ahi = min(clo, -BRAKE), max(chi, -BRAKE)
        xlo, xhi = xlo + vlo * DT, xhi + vhi * DT
        vlo, vhi = vlo + alo * DT, vhi + ahi * DT
        worst = max(worst, xhi)
        if worst > HARD_LIMIT:
            return "violated", worst
    return "safe", worst


def _crown_hull(cells, xlo, xhi, vlo, vhi):
    """Union of the contracts of every cell the box overlaps."""
    los, his = [], []
    for cxlo, cxhi, cvlo, cvhi, alo, ahi in cells:
        if cxhi >= xlo and cxlo <= xhi and cvhi >= vlo and cvlo <= vhi:
            los.append(alo)
            his.append(ahi)
    return (min(los) if los else -1.0), (max(his) if his else 1.0)


# pylint: disable-next=too-many-locals
def main():
    """Run both methods on the same system and report which proves the property."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    v0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    grid = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    cells = crown_cells(grid)
    print(f"geofence NSBT: x0 in [{X0_LO}, {X0_HI}], |v0| <= {v0}, "
          f"{steps} steps, {grid}x{grid} grid ({len(cells)} contracts)")

    verdict, worst = per_step_box(cells, steps, v0)
    print(f"  per-step box      : {verdict:>8}  (max x reached {worst:.4f}, "
          f"limit {HARD_LIMIT})")

    sanity(cells, steps, v0)

    res, secs = compositional(cells, steps, v0)
    proved = "safe" if res == z3.unsat else ("violated" if res == z3.sat else "?")
    print(f"  compositional (Z3): {proved:>8}  ({secs:.1f}s)")



# pylint: disable-next=too-many-locals
def sanity(cells, steps, v0_bound):
    """Guard against a vacuous proof.

    An UNSAT safety result is worthless if the model admits no behaviour at all:
    an over-constrained or infeasible encoding proves everything.  Gemini caught
    exactly this failure on the NAV goal property, where nuXmv reported
    "is true" for (k=30 -> FALSE) on a deadlocked model.

    Two checks:
      reachable  the model must have SOME execution (drop the property, expect SAT)
      moves      the model must actually progress (x must be able to differ from x0)
    """
    solver = z3.Solver()
    x = [z3.Real(f"x_{k}") for k in range(steps + 1)]
    v = [z3.Real(f"v_{k}") for k in range(steps + 1)]
    solver.add(x[0] >= X0_LO, x[0] <= X0_HI,
               v[0] >= -v0_bound, v[0] <= v0_bound)
    for k in range(steps):
        a = z3.Real(f"a_{k}")
        solver.add(a >= -1, a <= 1)
        solver.add(z3.Implies(x[k] >= FENCE, a == -BRAKE))
        for xlo, xhi, vlo, vhi, alo, ahi in cells:
            guard = z3.And(x[k] < FENCE,
                           x[k] >= xlo, x[k] <= xhi, v[k] >= vlo, v[k] <= vhi)
            solver.add(z3.Implies(guard, z3.And(a >= alo, a <= ahi)))
        solver.add(x[k + 1] == x[k] + v[k] * DT, v[k + 1] == v[k] + a * DT)

    solver.push()
    reachable = solver.check()
    solver.pop()
    solver.add(x[steps] != x[0])
    moves = solver.check()
    print(f"  vacuity guard     : model has executions = "
          f"{'YES' if reachable == z3.sat else '*** NO, PROOF IS VACUOUS ***'}, "
          f"state can change = "
          f"{'YES' if moves == z3.sat else '*** NO, MODEL IS FROZEN ***'}")
    return reachable == z3.sat and moves == z3.sat

if __name__ == "__main__":
    main()
