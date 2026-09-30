"""
Verify a property about WHICH BEHAVIOR-TREE NODE IS ACTIVE.

This is the property class that motivates the whole project.  A reachability
tool computes reachable sets over R^n; a network verifier proves input-output
facts about a frozen network.  Neither has any notion of node activation, so
neither can state a property like this one, let alone decide it:

    the Brake node is never active for more than N consecutive ticks

That is a real requirement - a safety envelope that engages and never releases
has failed, even if the vehicle stays inside its bounds - and it is NOT a
tautology of the guard: discharging it needs the network to actually bring the
state back, which is exactly the closed-loop coupling that makes NSBT
verification hard.

Measured on the true system, the longest consecutive Brake run over the
operating envelope is 18 ticks, so the property holds for N >= 18 and is false
for smaller N.  A verifier worth having must agree on both sides.
"""
from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")
import z3

from compare import _upper, crown_cells, V_TOP, X_TOP, X0_HI, X0_LO
from latched import MARGIN
from nsbt import BRAKE, DT, FENCE

V0, GRID = 0.8, 12


# pylint: disable-next=too-many-locals
def check_recovery(cells, steps, bound):
    """Is there a run where Brake stays active for `bound` consecutive ticks?

    Returns z3.unsat when the property holds: no such run exists.
    """
    solver = z3.Solver()
    x = [z3.Real(f"x_{k}") for k in range(steps + 1)]
    v = [z3.Real(f"v_{k}") for k in range(steps + 1)]
    braking = [z3.Bool(f"braking_{k}") for k in range(steps + 1)]
    active = []

    solver.add(x[0] >= X0_LO, x[0] <= X0_HI, v[0] >= -V0, v[0] <= V0,
               z3.Not(braking[0]))
    for k in range(steps):
        a = z3.Real(f"a_{k}")
        solver.add(a >= -1, a <= 1)
        # the tree: ShouldBrake is the guard, and it latches
        should = z3.Or(x[k] >= FENCE,
                       z3.And(braking[k], x[k] >= FENCE - MARGIN))
        active.append(should)
        solver.add(braking[k + 1] == should)
        solver.add(z3.Implies(should, a == -BRAKE))
        for xlo, xhi, vlo, vhi, alo, ahi in cells:
            guard = z3.And(z3.Not(should),
                           x[k] >= xlo, _upper(x[k], xhi, X_TOP),
                           v[k] >= vlo, _upper(v[k], vhi, V_TOP))
            solver.add(z3.Implies(guard, z3.And(a >= alo, a <= ahi)))
        solver.add(x[k + 1] == x[k] + v[k] * DT, v[k + 1] == v[k] + a * DT)

    # a counterexample is `bound` consecutive ticks with Brake active
    windows = [z3.And(*active[i:i + bound])
               for i in range(len(active) - bound + 1)]
    solver.add(z3.Or(windows))
    t0 = time.time()
    return solver.check(), time.time() - t0


def main():
    """Check the property at bounds either side of the true maximum of 18."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cells = crown_cells(GRID)
    print(f"'Brake never active for N consecutive ticks', {steps} steps, "
          f"{len(cells)} contracts")
    print(f"  true system's longest Brake run over the envelope: 18 ticks\n")
    print(f"{'N':>4} {'verdict':>10} {'means':>34} {'time':>7}")
    for bound in (12, 16, 19, 22):
        res, secs = check_recovery(cells, steps, bound)
        if res == z3.unsat:
            says, means = "proved", f"no run brakes {bound} ticks in a row"
        elif res == z3.sat:
            says, means = "refuted", f"a run brakes {bound} ticks in a row"
        else:
            says, means = "unknown", ""
        print(f"{bound:>4} {says:>10} {means:>34} {secs:>6.1f}s")


if __name__ == "__main__":
    main()
