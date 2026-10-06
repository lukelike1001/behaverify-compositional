"""The Z3 encoding of `divert`, and the CEGAR loop that discharges it.

WHAT IS BEING PROVED, in the notation of the theory it belongs to. Ogren's
convergence theorem (as stated in Rietz et al., arXiv:2602.06525v2, Thm 1) says
a behavior tree reaches its root's success region in finite time if, for every
leaf i, the convergence set C_i is invariant under the controller AND there is
a horizon h with x_t in Omega_i implying x_{t+h} not in Omega_i.

Rietz et al. enforce the FIRST condition by masking invalid actions at runtime.
They do not enforce the second at all, and their Table I reports 7% outright
failure. THE SECOND IS WHAT THIS FILE PROVES: "recovery leaves its operating
region within N ticks" is exactly the finite-escape hypothesis. It is bounded
and therefore a SAFETY property by Alpern and Schneider, not liveness - but it
is the missing precondition of a published liveness theorem, which is a better
thing to be than a liveness claim we cannot support.

THREE SOUNDNESS DEFECTS THAT LIVED IN THE EARLIER VERSION OF THIS FILE, all
of which made the model finish EARLIER than the real system and so would have
turned a proof into a non-proof. They are recorded because each one produced
plausible numbers while being wrong.
  1. Brake was encoded as the unsaturated a = -3v together with the global
     |a| <= A_MAX. Those are jointly unsatisfiable whenever |v| > A_MAX/3, so
     every fast initial state was silently deleted from the set the proof
     quantified over. The plant now saturates per axis and the encoding is
     exact.
  2. The handover and success test hypot(v) <= 0.05 was encoded as the box
     |vx|, |vy| <= 0.05, which CONTAINS that circle, so the model came to rest
     at speeds the real system still calls moving. The box is now inscribed.
  3. The success region was the start box padded by 0.1. Tree._home is already
     a box and is now encoded exactly.

AND ONE COMPLETENESS DEFECT, which is why no amount of refinement was working.
Cell guards were half-open on the upper face, so the top face of every
dimension belonged to no cell. The solver parked the state on vy = 0.25 and
every contract fell away, leaving only |a| <= A_MAX: free control. Faces are
now closed, which makes neighbouring guards overlap - sound, because both
contracts hold on a shared face and their conjunction is tighter than either.
"""
from __future__ import annotations

import math
import sys
import time

import z3

from affine import affine_contract, coarsen
from plant import A_MAX, DRAG, DT, S_MAX, START
from envelope import TRIP

# The plant's tests are Euclidean. This box is INSCRIBED in the circle of
# radius 0.05, so the model hands over and succeeds strictly later than the
# real system and the horizon it proves is an upper bound on the true one.
V_STOP = 0.05 / math.sqrt(2.0)

# The region the contracts tile. Measured, not guessed: a sweep of 26x26
# starts x 5 speeds x 8 headings puts every Return-phase state inside
# x [0.140, 1.257], y [-0.157, 0.707], vx [-1.094, 0.069], vy [-0.815, 0.439].
# The margin below is what the abstraction, which is looser than the plant, is
# allowed to wander into before the contracts stop covering it.
REGION = (-0.3, 1.7, -0.6, 1.2, -1.5, 1.1, -1.3, 1.1)


class Partition:
    """A kd-partition of `REGION` that splits where counterexamples live.

    A uniform grid is not an option: twelve divisions per axis is 12^4 = 20,736
    cells, and at forty ticks that is roughly 830,000 guarded implications
    before the search starts. Splitting only the cells a counterexample
    actually visits keeps the encoding at the size the proof needs.

    Leaving a cell out is SOUND, not merely convenient. An uncovered state is
    constrained by nothing but the actuator limit |a| <= A_MAX, which admits
    every control the network could have produced and more, so a partial
    partition is a COARSER over-approximation than the full grid. `unsat` under
    it therefore implies `unsat` under the full grid and under the plant. What
    a missing cell costs is completeness: the proof may simply not close.
    """

    def __init__(self, region=REGION, source=None, seed=0):
        self.region = region
        self.cells = [region]
        # `source` maps a list of cells to a list of (cell, contract). The
        # default is the analytic law, which is a synthetic ablation and not a
        # result; `network.CrownContracts()` is the system under test.
        self.source = source or (lambda cells: [(c, coarsen(c, affine_contract(c)))
                                                for c in cells])
        # Starting from a single cell spends the first rounds re-solving an
        # essentially unconstrained problem: one cell spanning the whole region
        # has a contract so loose it is barely better than |a| <= A_MAX. Each
        # seed level bisects every cell on all four axes, so `seed` levels give
        # 16**seed cells before the first query.
        for _ in range(seed):
            self.split(range(len(self.cells)), axes=4)

    def contracts(self):
        """Every cell paired with its contract, from whichever source."""
        return self.source(self.cells)

    def containing(self, point):
        """Indices of the cells holding `point`. Faces overlap, so may be many."""
        return [i for i, c in enumerate(self.cells)
                if all(c[2 * d] <= point[d] <= c[2 * d + 1] for d in range(4))]

    def split(self, indices, axes=2):
        """Bisect each named cell along its `axes` widest axes, relatively.

        Widths are normalised by the region's own extent on that axis so that a
        position of order 2 and a velocity of order 1 are compared on equal
        terms rather than position always winning - the geofence failed once
        for refining the axis that was already fine.

        Two axes rather than one because a counterexample early on visits very
        few cells, so single-axis bisection adds about one cell per solve and
        the loop spends its budget re-solving an essentially unchanged problem.
        """
        for index in sorted(set(indices), reverse=True):
            parts = [self.cells.pop(index)]
            spans = [(parts[0][2 * d + 1] - parts[0][2 * d])
                     / (self.region[2 * d + 1] - self.region[2 * d])
                     for d in range(4)]
            for axis in sorted(range(4), key=spans.__getitem__)[-axes:]:
                grown = []
                for cell in parts:
                    mid = 0.5 * (cell[2 * axis] + cell[2 * axis + 1])
                    for lo, hi in ((cell[2 * axis], mid),
                                   (mid, cell[2 * axis + 1])):
                        half = list(cell)
                        half[2 * axis], half[2 * axis + 1] = lo, hi
                        grown.append(tuple(half))
                parts = grown
            self.cells.extend(parts)


class Encoding:
    """One bounded unrolling of the recovery branch, with its Z3 variables.

    Only the recovery branch is encoded. The hazard has already latched, which
    is the operating region Omega_i whose escape the property is about; the
    cruise branch cannot influence that question and encoding it would only
    lengthen the unrolling.
    """

    def __init__(self, cells, steps, initial=None, handed=False):
        # `initial` replaces the trip region when a lemma starts somewhere
        # else - compose.py enters Return from Brake's handover box - and
        # `handed` says the Brake step is already complete.
        self.initial, self.handed_at_start = initial, handed
        self.steps = steps
        self.solver = z3.Solver()
        self.x, self.y = self._reals("x", steps + 1), self._reals("y", steps + 1)
        self.vx = self._reals("vx", steps + 1)
        self.vy = self._reals("vy", steps + 1)
        self.ax, self.ay = self._reals("ax", steps), self._reals("ay", steps)
        # The tree's step index is 0 Brake, 1 Return, 2 done, and it never goes
        # backwards, so two LATCHING BOOLEANS carry it exactly: `handed` is
        # step >= 1 and `done` is step == 2. An Int was the obvious encoding and
        # the wrong one - it drags the problem out of QF_LRA into mixed integer
        # and real arithmetic, and with dense CROWN coefficients that cost a
        # timeout at nineteen cells and a horizon of twelve.
        self.handed = [z3.Bool(f"handed_{k}") for k in range(steps + 1)]
        self.done = [z3.Bool(f"done_{k}") for k in range(steps + 1)]
        self.covered = []
        self._initial()
        for k in range(steps):
            self._tick(cells, k)

    @staticmethod
    def _reals(name, count):
        return [z3.Real(f"{name}_{k}") for k in range(count)]

    def _initial(self):
        """The hazard has just latched somewhere in the measured trip region.

        The velocity box CIRCUMSCRIBES the speed circle, so it admits MORE
        initial states than the plant has - speeds up to sqrt(2) * S_MAX. The
        circular form is nonlinear and drops Z3 into nlsat, which cost 37.3s at
        K=10 against the geofence's 0.1s. The envelope sweep was re-run over
        this box rather than the circle so the measured bound and the proved
        one quantify over the same set: 26,244 starts, longest recovery 31.
        """
        box = self.initial or (TRIP[0], TRIP[1], TRIP[2], TRIP[3],
                               -S_MAX, S_MAX, -S_MAX, S_MAX)
        for d, var in enumerate(self.state(0)):
            self.solver.add(var >= box[2 * d], var <= box[2 * d + 1])
        self.solver.add(self.handed[0] if self.handed_at_start
                        else z3.Not(self.handed[0]))
        self.solver.add(z3.Not(self.done[0]))

    def state(self, k):
        """The four continuous variables at tick k."""
        return (self.x[k], self.y[k], self.vx[k], self.vy[k])

    def _tick(self, cells, k):
        solver = self.solver
        slow = z3.And(self.vx[k] >= -V_STOP, self.vx[k] <= V_STOP,
                      self.vy[k] >= -V_STOP, self.vy[k] <= V_STOP)
        home = z3.And(self.x[k] >= START[0], self.x[k] <= START[1],
                      self.y[k] >= START[2], self.y[k] <= START[3])
        # the tree's own transition relation, straight from plant.py's tick().
        # Brake hands over on `slow` alone; Return succeeds only once handed
        # over AND home AND slow, so a state that is already home when Brake
        # releases still spends a tick in Return, exactly as the plant does.
        solver.add(self.handed[k + 1] == z3.Or(self.handed[k], slow))
        solver.add(self.done[k + 1] == z3.Or(
            self.done[k], z3.And(self.handed[k], home, slow)))
        self._brake(k, z3.Not(self.handed[k]))
        self._return(cells, k, z3.And(self.handed[k], z3.Not(self.done[k])))
        solver.add(self.ax[k] >= -A_MAX, self.ax[k] <= A_MAX,
                   self.ay[k] >= -A_MAX, self.ay[k] <= A_MAX)
        solver.add(z3.Implies(self.done[k],
                              z3.And(self.ax[k] == 0, self.ay[k] == 0)))
        solver.add(self.x[k + 1] == self.x[k] + self.vx[k] * DT,
                   self.y[k + 1] == self.y[k] + self.vy[k] * DT)
        solver.add(self.vx[k + 1] == self.vx[k]
                   + (self.ax[k] - DRAG * self.vx[k]) * DT)
        solver.add(self.vy[k + 1] == self.vy[k]
                   + (self.ay[k] - DRAG * self.vy[k]) * DT)

    def _brake(self, k, active):
        """Brake is analytic, so it is encoded EXACTLY rather than bounded.

        Saturation must be encoded with it; omitting it is defect 1 above. Per
        axis the clip is a nested If and costs nothing in linear arithmetic.
        """
        for accel, vel in ((self.ax[k], self.vx[k]), (self.ay[k], self.vy[k])):
            raw = -3.0 * vel
            self.solver.add(z3.Implies(active, accel == z3.If(
                raw > A_MAX, A_MAX, z3.If(raw < -A_MAX, -A_MAX, raw))))

    def _return(self, cells, k, active):
        """Return consults the network, so it is bounded by affine contracts."""
        state = self.state(k)
        guards = []
        for cell, contract in cells:
            guard = z3.And(active, *[test for d, var in enumerate(state)
                                     for test in (var >= cell[2 * d],
                                                  var <= cell[2 * d + 1])])
            guards.append(guard)
            for axis, accel in ((0, self.ax[k]), (1, self.ay[k])):
                low_a, low_b, up_a, up_b = contract[axis]
                low = z3.Sum([c * v for c, v in zip(low_a, state)]) + low_b
                high = z3.Sum([c * v for c, v in zip(up_a, state)]) + up_b
                self.solver.add(z3.Implies(guard,
                                           z3.And(accel >= low, accel <= high)))
        self.covered.append(z3.Or(*guards) if guards else z3.BoolVal(False))

    def trace(self, model):
        """The counterexample as plain floats, one row per tick."""
        def value(expr):
            return float(model.eval(expr, model_completion=True)
                         .as_decimal(12).rstrip("?"))
        return [tuple(value(v) for v in self.state(k))
                for k in range(self.steps + 1)]


def prove(steps, bound, rounds=40, timeout_s=600, region=REGION,
          source=None, initial=None, handed=False, seed=0, verbose=True):
    """CEGAR: tighten only where the counterexample actually went.

    Returns (verdict, rounds_used, cells_used, seconds). The verdict is
    "PROVED" for unsat, "TIMEOUT" when Z3 gives up, "ESCAPED" when the
    counterexample leaves the contract region - which means the region is too
    small rather than the property false - and "EXHAUSTED" when the round
    budget runs out with a counterexample still inside the region.
    """
    part = Partition(region, source, seed)
    started = time.time()
    for round_index in range(1, rounds + 1):
        cells = part.contracts()
        enc = Encoding(cells, steps, initial, handed)
        enc.solver.add(z3.Not(enc.done[bound]))   # still recovering at `bound`
        enc.solver.set("timeout", int(timeout_s * 1000))
        tick = time.time()
        result = str(enc.solver.check())
        if verbose:
            print(f"  round {round_index:>3}  {len(part.cells):>5} cells  "
                  f"{result:>7}  {time.time() - tick:7.1f}s", flush=True)
        if result == "unsat":
            return "PROVED", round_index, len(part.cells), time.time() - started
        if result == "unknown":
            return "TIMEOUT", round_index, len(part.cells), time.time() - started
        model = enc.solver.model()
        rows = enc.trace(model)
        active = [k for k in range(steps)
                  if z3.is_true(model.eval(enc.handed[k], model_completion=True))
                  and not z3.is_true(model.eval(enc.done[k],
                                                model_completion=True))]
        visited = {i for k in active for i in part.containing(rows[k])}
        if not visited:
            outside = next((rows[k] for k in active
                            if not part.containing(rows[k])), None)
            if verbose:
                print(f"  counterexample left the contract region at {outside}")
            return "ESCAPED", round_index, len(part.cells), time.time() - started
        part.split(visited)
    return "EXHAUSTED", rounds, len(part.cells), time.time() - started


def vacuity(steps, region=REGION, source=None):
    """A model that proves everything proves nothing. Check it has runs."""
    enc = Encoding(Partition(region, source).contracts(), steps)
    enc.solver.push()
    feasible = str(enc.solver.check())
    enc.solver.pop()
    enc.solver.add(enc.done[steps])
    return feasible, str(enc.solver.check())


def main():
    """Prove the finite-escape bound, reporting tightness against the sweep."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    bound = int(sys.argv[2]) if len(sys.argv) > 2 else 34
    feasible, completes = vacuity(10)
    print(f"vacuity: model has executions = {feasible}, recovery can complete "
          f"= {completes}  (both must be sat)\n")
    print(f"divert finite escape | horizon K = {steps}, bound N = {bound}")
    print("sweep of 6,760 starts measured a longest recovery of 31 ticks\n")
    verdict, rounds, cells, secs = prove(steps, bound)
    print(f"\n{verdict} after {rounds} rounds, {cells} cells, {secs:.1f}s")


if __name__ == "__main__":
    main()
