"""
The Z3 encoding of `divert`, and the horizon gate that decides whether the
benchmark is verifiable at all.

WHAT IS BEING PROVED, in the notation of the theory it belongs to. Ogren's
convergence theorem (as stated in Rietz et al., arXiv:2602.06525v2, Thm 1) says
a behavior tree reaches its root's success region in finite time if, for every
leaf i, the convergence set C_i is invariant under the controller AND there is a
horizon h such that x_t in Omega_i implies x_{t+h} not in Omega_i.

Rietz et al. enforce the FIRST condition by masking invalid actions at runtime.
They do not enforce the second at all - it is left to reward shaping, and their
Table I reports 7% outright failure and 19.30% of episodes violating the one
safety condition that depends on an agent they cannot predict.

THE SECOND CONDITION IS WHAT THIS FILE PROVES. "Recovery leaves its operating
region within N ticks" is exactly `x_t in Omega_i implies x_{t+h} not in
Omega_i`. It is a BOUNDED property and therefore a safety property, not
liveness - but it is the missing hypothesis of a published liveness theorem,
which is a better thing to be than a liveness claim we cannot support.

THE GATE THIS FILE EXISTS FOR. The property needs a horizon K >= N+1 to be
decidable, and N is 36 by measurement (`envelope.py`: longest recovery 33 ticks
over 6,760 starts, plus margin). Z3's measured wall is 40 steps proved in
1192.8s on the 2-D geofence with ONE network, timing out at 45. `divert` is 4-D
with TWO networks, so the wall moves down by an unknown amount. If Z3 cannot
reach K=37 here, no amount of network training rescues the benchmark, and that
is worth an hour to find out rather than a week.

So `main()` runs the gate with ANALYTIC placeholder contracts and no trained
networks. The placeholders are sampled rather than certified and are NOT SOUND;
they exist to measure solver cost at depth. CROWN contracts replace them
through the same interface once training is done.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import z3

from plant import A_MAX, DRAG, DT, S_MAX, START, return_law
from envelope import TRIP

V_STOP = 0.05           # the Brake step hands over below this speed
HOME_R = 0.1            # Return succeeds inside this radius of the start


def _sampled_bounds(law, box, samples=5):
    """Placeholder contract: sample the analytic law over a box, pad outward.

    NOT SOUND. Sampling can miss the extremes between sample points, so a
    bound derived this way may exclude real behaviour. It is used only to give
    the solver realistically-shaped constraints while measuring cost at depth;
    CROWN replaces it through the same interface, and no verdict produced with
    these is a proof.
    """
    xlo, xhi, ylo, yhi, vxlo, vxhi, vylo, vyhi = box
    axs, ays = [], []
    for x in np.linspace(xlo, xhi, samples):
        for y in np.linspace(ylo, yhi, samples):
            for vx in np.linspace(vxlo, vxhi, samples):
                for vy in np.linspace(vylo, vyhi, samples):
                    ax, ay = law(float(x), float(y), float(vx), float(vy))
                    axs.append(ax)
                    ays.append(ay)
    pad = 0.15 * A_MAX
    return (max(min(axs) - pad, -A_MAX), min(max(axs) + pad, A_MAX),
            max(min(ays) - pad, -A_MAX), min(max(ays) + pad, A_MAX))


def placeholder_contracts(resolution=2, vel_resolution=None):
    """Contracts over the reachable box, split in ALL FOUR dimensions.

    An earlier version split only x and y, leaving every cell spanning the full
    velocity range. `return_law` is a VELOCITY-TARGETING controller, so a bound
    taken over all velocities is near-vacuous, and refining position from 4
    cells to 64 changed nothing - the same signature as the geofence, where
    refinement failed because it was applied to the axis that was already fine.
    """
    vel_resolution = resolution if vel_resolution is None else vel_resolution
    box = (TRIP[0] - 1.0, TRIP[1] + 1.0, TRIP[2] - 1.0, TRIP[3] + 1.0,
           -S_MAX * 1.5, S_MAX * 1.5, -S_MAX * 1.5, S_MAX * 1.5)
    xs = np.linspace(box[0], box[1], resolution + 1)
    ys = np.linspace(box[2], box[3], resolution + 1)
    vxs = np.linspace(box[4], box[5], vel_resolution + 1)
    vys = np.linspace(box[6], box[7], vel_resolution + 1)
    cells = []
    for i in range(resolution):
        for j in range(resolution):
            for m in range(vel_resolution):
                for n in range(vel_resolution):
                    sub = (xs[i], xs[i + 1], ys[j], ys[j + 1],
                           vxs[m], vxs[m + 1], vys[n], vys[n + 1])
                    cells.append((sub, _sampled_bounds(return_law, sub)))
    return cells


# pylint: disable-next=too-many-locals
def build(cells, steps):  # noqa: C901
    """Encode the recovery branch over `steps` ticks. Returns solver and vars.

    Only the recovery branch is encoded: the hazard has already latched, which
    is the operating region Omega_i whose escape the property is about. The
    cruise branch is irrelevant to that question and encoding it would only
    lengthen the unrolling.
    """
    s = z3.Solver()
    x = [z3.Real(f"x_{k}") for k in range(steps + 1)]
    y = [z3.Real(f"y_{k}") for k in range(steps + 1)]
    vx = [z3.Real(f"vx_{k}") for k in range(steps + 1)]
    vy = [z3.Real(f"vy_{k}") for k in range(steps + 1)]
    ax = [z3.Real(f"ax_{k}") for k in range(steps)]
    ay = [z3.Real(f"ay_{k}") for k in range(steps)]
    # step: 0 Brake, 1 Return, 2 done. Recovery is active while step < 2.
    st = [z3.Int(f"st_{k}") for k in range(steps + 1)]

    s.add(x[0] >= TRIP[0], x[0] <= TRIP[1], y[0] >= TRIP[2], y[0] <= TRIP[3])
    # Linear box rather than the circular speed bound. A box CONTAINS the
    # circle, so this admits MORE initial states than the real system has: the
    # abstraction is coarser and a proof under it still holds. The circular
    # form is nonlinear and drops Z3 into nlsat, which cost 37.3s at K=10
    # against the 2-D geofence's 0.1s.
    s.add(vx[0] >= -S_MAX, vx[0] <= S_MAX, vy[0] >= -S_MAX, vy[0] <= S_MAX)
    s.add(st[0] == 0)

    for k in range(steps):
        # Box form again. Here the box is INSIDE the circle, so the Brake step
        # hands over later than the real system would, recovery takes longer,
        # and the bound we prove is therefore conservative.
        slow = z3.And(vx[k] >= -V_STOP, vx[k] <= V_STOP,
                      vy[k] >= -V_STOP, vy[k] <= V_STOP)
        home = z3.And(x[k] >= START[0] - HOME_R, x[k] <= START[1] + HOME_R,
                      y[k] >= START[2] - HOME_R, y[k] <= START[3] + HOME_R)
        # the tree's own transition relation, straight from plant.py's tick()
        s.add(st[k + 1] == z3.If(st[k] == 0, z3.If(slow, 1, 0),
                                 z3.If(st[k] == 1,
                                       z3.If(z3.And(home, slow), 2, 1), 2)))
        # Brake is an analytic law and needs no contract
        braking = st[k] == 0
        s.add(z3.Implies(braking, z3.And(ax[k] == -3.0 * vx[k],
                                         ay[k] == -3.0 * vy[k])))
        # Return consults the network, so it is bounded by contracts
        returning = st[k] == 1
        for (bx, bounds) in cells:
            guard = z3.And(returning,
                           x[k] >= bx[0], x[k] < bx[1],
                           y[k] >= bx[2], y[k] < bx[3],
                           vx[k] >= bx[4], vx[k] < bx[5],
                           vy[k] >= bx[6], vy[k] < bx[7])
            s.add(z3.Implies(guard, z3.And(ax[k] >= bounds[0],
                                           ax[k] <= bounds[1],
                                           ay[k] >= bounds[2],
                                           ay[k] <= bounds[3])))
        # global bound, true everywhere and needing no cell: the actuator
        # saturates, so an uncovered cell can never free the control
        s.add(ax[k] >= -A_MAX, ax[k] <= A_MAX, ay[k] >= -A_MAX, ay[k] <= A_MAX)
        s.add(z3.Implies(st[k] == 2, z3.And(ax[k] == 0, ay[k] == 0)))

        s.add(x[k + 1] == x[k] + vx[k] * DT, y[k + 1] == y[k] + vy[k] * DT)
        s.add(vx[k + 1] == vx[k] + (ax[k] - DRAG * vx[k]) * DT)
        s.add(vy[k + 1] == vy[k] + (ay[k] - DRAG * vy[k]) * DT)
    return s, (x, y, vx, vy, st)


def escapes_within(cells, steps, bound, timeout_s=1800):
    """Is Omega_i left within `bound` ticks? unsat means yes, proved."""
    s, (_, _, _, _, st) = build(cells, steps)
    s.add(st[bound] < 2)          # still recovering at tick `bound`
    s.set("timeout", int(timeout_s * 1000))
    t0 = time.time()
    res = s.check()
    return str(res), time.time() - t0


def vacuity(cells, steps):
    """A model that proves everything proves nothing. Check it has runs."""
    s, (_, _, _, _, st) = build(cells, steps)
    s.push()
    feasible = s.check()
    s.pop()
    s.add(st[steps] == 2)
    completes = s.check()
    return str(feasible), str(completes)


def main():
    """The gate: how deep can Z3 go on this structure before it stops?"""
    steps_list = [int(a) for a in sys.argv[1:]] or [10, 20, 30, 37, 45]
    cells = placeholder_contracts()
    print(f"divert horizon gate | {len(cells)} placeholder contracts "
          f"(sampled, NOT sound - cost measurement only)")
    print("true longest recovery 33 ticks, property bound N = 36, "
          "so the property needs K >= 37\n")
    feas, comp = vacuity(cells, 10)
    print(f"vacuity: model has executions = {feas}, recovery can complete = "
          f"{comp}  (both must be sat)\n")
    print(f"{'K':>5} {'verdict':>10} {'escapes<=K-1':>14} {'sec':>9}")
    for steps in steps_list:
        verdict, secs = escapes_within(cells, steps, steps - 1)
        meaning = {"unsat": "PROVED", "sat": "no proof",
                   "unknown": "TIMEOUT"}.get(verdict, verdict)
        print(f"{steps:>5} {verdict:>10} {meaning:>14} {secs:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
