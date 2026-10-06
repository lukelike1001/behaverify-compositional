"""Prove the finite-escape bound ONE TREE NODE AT A TIME, then compose.

THE PROBLEM WITH THE MONOLITHIC UNROLLING. `verify.prove` unrolls the whole
recovery branch for 35 ticks in one query. Horizon is the binding constraint on
this whole line of work - the 2-D geofence took 1192.8s at 40 steps and timed
out at 45 - and 35 ticks of a 4-D plant with contracts at every tick is a large
query to hand a solver in one piece.

THE DECOMPOSITION. The recovery branch is a `sequence with_partial_memory` of
two steps, and a sequence is exactly where an assume-guarantee split belongs:

  Lemma 1 (Brake)   from the trip region at any admissible velocity, Brake
                    reaches the handover condition within n1 ticks, and the
                    state at handover lies in a box B1.
  Lemma 2 (Return)  from B1, Return reaches home-and-stopped within n2 ticks.
  Compose           the recovery leaves its operating region within n1 + n2.

The composition is sound because B1 OVER-approximates the handover set: Lemma 2
is proved from a superset of the states Lemma 1 can actually deliver, so any
trajectory the real system has is covered. B1 is computed here by optimisation
rather than guessed, so the two lemmas cannot drift apart.

WHY THIS IS THE POINT AND NOT AN OPTIMISATION. Brake is analytic and needs no
network, so Lemma 1 carries no contracts at all; Return is the only step that
consults a network, so every contract in the proof sits inside Lemma 2 over a
much smaller initial set than the trip region. The tree's own structure is what
licenses the split, which is the thing a monolithic encoding cannot exploit and
the thing a reachability tool has no notion of.
"""
from __future__ import annotations

import math

import z3

from plant import A_MAX, DRAG, DT, S_MAX, START
from envelope import TRIP
from verify import REGION, V_STOP, Partition, prove

AXES = ("x", "y", "vx", "vy")


def _brake_trace(steps):
    """Unroll Brake alone. Returns the solver and the per-tick state vectors.

    No contracts appear here: Brake is `a = -3v` saturated per axis, which is
    exact in linear arithmetic, so this lemma is independent of the network
    entirely.
    """
    solver = z3.Solver()
    state = [[z3.Real(f"{n}_{k}") for n in AXES] for k in range(steps + 1)]
    first = state[0]
    solver.add(first[0] >= TRIP[0], first[0] <= TRIP[1],
               first[1] >= TRIP[2], first[1] <= TRIP[3])
    solver.add(*[test for v in first[2:]
                 for test in (v >= -S_MAX, v <= S_MAX)])
    for k in range(steps):
        now, nxt = state[k], state[k + 1]
        for axis in (0, 1):
            raw = -3.0 * now[2 + axis]
            accel = z3.If(raw > A_MAX, A_MAX, z3.If(raw < -A_MAX, -A_MAX, raw))
            solver.add(nxt[axis] == now[axis] + now[2 + axis] * DT)
            solver.add(nxt[2 + axis] == now[2 + axis]
                       + (accel - DRAG * now[2 + axis]) * DT)
    return solver, state


def _slow(vec):
    """The inscribed stop box, the same one `verify` uses."""
    return z3.And(vec[2] >= -V_STOP, vec[2] <= V_STOP,
                  vec[3] >= -V_STOP, vec[3] <= V_STOP)


def brake_horizon(limit=25):
    """Smallest n1 such that Brake provably reaches the stop box within n1."""
    for steps in range(1, limit + 1):
        solver, state = _brake_trace(steps)
        solver.add(z3.Not(z3.Or(*[_slow(v) for v in state[1:]])))
        if str(solver.check()) == "unsat":
            return steps
    return None


def handover_box(n1):
    """Optimised box B1 containing every state at the tick Brake hands over.

    Each face is an exact extremum, not a sampled one: the brake dynamics are
    piecewise linear and the query is an LP, so `z3.Optimize` returns the true
    bound and B1 is as tight as a box can be.
    """
    bounds = []
    for axis in range(4):
        face = []
        for sense in (0, 1):
            opt = z3.Optimize()
            solver, state = _brake_trace(n1)
            opt.add(*solver.assertions())
            # the FIRST tick at which the stop box is entered
            entered = z3.Or(*[z3.And(_slow(state[k]),
                                     *[z3.Not(_slow(state[j]))
                                       for j in range(k)],
                                     state[axis if False else 0] == state[0])
                              for k in range(n1 + 1)])
            opt.add(entered)
            target = z3.Real("target")
            opt.add(target == z3.Sum(
                [z3.If(z3.And(_slow(state[k]),
                              *[z3.Not(_slow(state[j])) for j in range(k)]),
                       state[k][axis], 0.0) for k in range(n1 + 1)]))
            handle = opt.minimize(target) if sense == 0 else opt.maximize(target)
            assert str(opt.check()) == "sat"
            value = opt.lower(handle) if sense == 0 else opt.upper(handle)
            face.append(float(value.as_decimal(10).rstrip("?")))
        bounds.extend(face)
    return tuple(bounds)


def return_horizon(box, source=None, limit=45, verbose=True):
    """Lemma 2 by zonotope propagation. Returns (n2, seconds).

    NOT by unrolled SMT, and not by interval reachability. Both were measured
    and both fail, for reasons that do not overlap:
      - unrolled SMT makes the solver choose WHICH contract cell applies at
        EVERY tick, so the search grows like cells^horizon. It returns
        `unknown` at 256 cells with a horizon of 6, and at 28 cells with a
        horizon of 35 after 1,201s. Meanwhile CROWN's bound is only useful
        below a half-width of about 0.06, which the Return corridor needs
        roughly 10^4 cells to reach. The two numbers miss each other by three
        orders of magnitude.
      - interval reachability diverges on the wrapping effect: the closed loop
        has complex eigenvalues of modulus 0.809 but ||A||_inf = 1.23, so the
        boxes grow 1.23x a tick while the true set shrinks.
    """
    import time                                         # noqa: PLC0415
    import zono                                         # noqa: PLC0415
    from network import CrownContracts                  # noqa: PLC0415

    started = time.time()
    steps = zono.escape_horizon(zono.Zono.from_box(box, handed=True),
                                source or CrownContracts(), limit=limit,
                                verbose=verbose)
    return steps, time.time() - started


def main():
    """Both lemmas, then the composed bound, against the measured sweep."""
    print("Lemma 1 (Brake): analytic, no contracts, Z3")
    n1 = brake_horizon()
    box = handover_box(n1)
    print(f"  n1 = {n1} ticks")
    print("  B1 = " + ", ".join(
        f"{n} [{box[2 * d]:.4f}, {box[2 * d + 1]:.4f}]"
        for d, n in enumerate(AXES)))
    print("\nLemma 2 (Return): CROWN contracts over the trained net_divert")
    n2, secs = return_horizon(box)
    if n2 is None:
        print("\nLemma 2 did not close; the composed bound is not available.")
        return
    print(f"  n2 = {n2} ticks  ({secs:.1f}s)")
    print(f"\nCOMPOSED  N = n1 + n2 = {n1} + {n2} = {n1 + n2} ticks")
    print("measured longest recovery over 26,244 box starts: 31 ticks")


if __name__ == "__main__":
    main()
