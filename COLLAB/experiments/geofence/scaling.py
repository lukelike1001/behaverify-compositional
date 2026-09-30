"""
Does compositional still win when the tree has MANY guards?

The pilot has one branch.  Its advantage comes from keeping that branch as a
disjunction instead of a hull - but a disjunction over many guards is exactly
what makes SMT expensive, while splitting over many guards is what makes
flowpipe tools fragment.  Both sides get worse; the question is which gets worse
faster, and the pilot says nothing about it.

A realistic drone safety envelope has several guards at once - geofence,
altitude ceiling, velocity limit, battery threshold - so this scales the tree to
N nested fences, each with its own braking level, and measures both methods.

    selector
      |-- sequence[ x >= fence_N ,  Brake_N ]     hardest braking
      |-- ...
      |-- sequence[ x >= fence_1 ,  Brake_1 ]     gentlest
      +-- NeuralCruise

This is a priority ladder, which is how behavior trees express exactly this
kind of layered envelope.
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import z3

from compare import _crown_hull, crown_cells, X0_HI, X0_LO
from nsbt import DT, HARD_LIMIT

STEPS, V0 = 20, 0.8


def fences(n):
    """N fences between 0.6 and 1.0, with braking that stiffens outward."""
    xs = np.linspace(0.6, 1.0, n)
    return [(float(x), -(0.4 + 0.6 * i / max(n - 1, 1))) for i, x in enumerate(xs)]


# pylint: disable-next=too-many-locals
def compositional_n(cells, ladder, steps):
    """Keep every guard as a disjunction: each state uses its own branch."""
    solver = z3.Solver()
    x = [z3.Real(f"x_{k}") for k in range(steps + 1)]
    v = [z3.Real(f"v_{k}") for k in range(steps + 1)]
    solver.add(x[0] >= X0_LO, x[0] <= X0_HI, v[0] >= -V0, v[0] <= V0)
    lowest = ladder[0][0]
    for k in range(steps):
        a = z3.Real(f"a_{k}")
        solver.add(a >= -1, a <= 1)
        # priority ladder: the highest fence whose guard holds wins
        for i, (fx, brake) in enumerate(ladder):
            higher = [ladder[j][0] for j in range(i + 1, len(ladder))]
            guard = z3.And(x[k] >= fx, *[x[k] < h for h in higher])
            solver.add(z3.Implies(guard, a == brake))
        for xlo, xhi, vlo, vhi, alo, ahi in cells:
            guard = z3.And(x[k] < lowest,
                           x[k] >= xlo, x[k] <= xhi, v[k] >= vlo, v[k] <= vhi)
            solver.add(z3.Implies(guard, z3.And(a >= alo, a <= ahi)))
        solver.add(x[k + 1] == x[k] + v[k] * DT, v[k + 1] == v[k] + a * DT)
    solver.add(z3.Or([x[k] > HARD_LIMIT for k in range(steps + 1)]))
    t0 = time.time()
    res = solver.check()
    return res, time.time() - t0


# pylint: disable-next=too-many-locals
def splitting_n(cells, ladder, steps, cap=400000):
    """Split at every guard, as a flowpipe tool would. Returns sets at peak."""
    edges = [f for f, _ in ladder]
    boxes = [(X0_LO, X0_HI, -V0, V0)]
    worst, peak = X0_HI, 1
    t0 = time.time()
    for _ in range(steps):
        nxt = []
        for xlo, xhi, vlo, vhi in boxes:
            cuts = [e for e in edges if xlo < e < xhi]
            pieces, lo = [], xlo
            for c in sorted(cuts) + [xhi]:
                pieces.append((lo, c))
                lo = c
            for plo, phi in pieces:
                active = [b for f, b in ladder if plo >= f]
                if active:
                    alo = ahi = min(active)
                else:
                    alo, ahi = _crown_hull(cells, plo, phi, vlo, vhi)
                nxt.append((plo + vlo * DT, phi + vhi * DT,
                            vlo + alo * DT, vhi + ahi * DT))
        boxes = list(dict.fromkeys(tuple(round(c, 9) for c in b) for b in nxt))
        peak = max(peak, len(boxes))
        worst = max(worst, max(b[1] for b in boxes))
        if worst > HARD_LIMIT:
            return "violated", peak, time.time() - t0
        if len(boxes) > cap:
            return "gave up", peak, time.time() - t0
    return "safe", peak, time.time() - t0


def main():
    """Scale the number of guards and measure both methods."""
    cells = crown_cells(12)
    print(f"{'guards':>7} {'splitting':>12} {'peak sets':>10} {'time':>7}  "
          f"{'compositional':>14} {'time':>8}")
    for n in (1, 2, 3, 4, 6):
        ladder = fences(n)
        sv, peak, st = splitting_n(cells, ladder, STEPS)
        res, ct = compositional_n(cells, ladder, STEPS)
        comp = "safe" if res == z3.unsat else "violated" if res == z3.sat else "?"
        print(f"{n:>7} {sv:>12} {peak:>10} {st:>6.1f}s  {comp:>14} {ct:>7.1f}s")


if __name__ == "__main__":
    main()
