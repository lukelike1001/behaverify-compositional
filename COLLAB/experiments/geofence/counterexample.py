"""
What a failed proof gives you, and why reachability gives you nothing.

When a reachability tool reports that the reachable set meets the unsafe
region, that is the whole of its answer. There is no state to inspect: the
set is an over-approximation, so the intersection may correspond to no real
execution at all, and nothing in the output says which. The engineer is left
to refine everywhere and hope.

The compositional encoding is a satisfiability problem, so a failed proof is a
MODEL - a concrete initial state, a concrete control at every tick, and the
resulting trajectory. That can be replayed against the real closed loop in
microseconds, which settles the question the reachability tool cannot ask:

    the replay breaches the limit    the property is genuinely false, and the
                                     trace is the bug report
    the replay stays safe            the counterexample is spurious, and the
                                     step where the solver's control diverges
                                     from the network's real output names the
                                     ONE contract that was too loose

That second case is demand-driven refinement. Instead of refining the whole
grid and re-running, the trace points at a single cell. This module measures
both the diagnosis and what refining only that cell buys.
"""
from __future__ import annotations

import sys

import z3

from compare import build, crown_cells
from nsbt import DT, FENCE, HARD_LIMIT, Tree


class Witness:
    """A counterexample trace from the abstraction, and its true replay."""

    def __init__(self, start, controls, states):
        self.start = start
        self.controls = controls
        self.states = states

    def replay(self, tree, steps=None):
        """Run the REAL closed loop from the witness's initial state."""
        x, v = self.start
        steps = steps or len(self.controls)
        trace = []
        for _ in range(steps):
            accel, node = tree.tick(x, v)
            trace.append((x, v, accel, node))
            x, v = x + v * DT, v + accel * DT
        trace.append((x, v, 0.0, "end"))
        return trace

    def divergence(self, tree):
        """First tick where the solver's control differs from the network's.

        The abstract state is the one the solver chose, so this asks what the
        network would ACTUALLY output there. The gap is the slack in the
        contract covering that state, and its cell is the one to refine.
        """
        gaps = []
        for k, (x, v) in enumerate(self.states[:-1]):
            truth, _ = tree.tick(x, v)
            gaps.append((k, (x, v), truth, abs(self.controls[k] - truth)))
        return max(gaps, key=lambda g: g[3])


def witness(cells, steps, v0_bound, limit):
    """A counterexample to the limit, or None when the property is proved."""
    solver, xs, vs, accels = build(cells, steps, v0_bound, limit)
    if solver.check() != z3.sat:
        return None
    model = solver.model()

    def num(term):
        """The model's value for a Z3 real, as a Python float."""
        return float(model.eval(term, model_completion=True).as_fraction())

    return Witness((num(xs[0]), num(vs[0])),
                   [num(a) for a in accels],
                   [(num(x), num(v)) for x, v in zip(xs, vs)])


def report(found, tree, limit):
    """Say whether the counterexample is real, and if not, what to refine."""
    print(f"  witness start  x0 = {found.start[0]:.6f}, "
          f"v0 = {found.start[1]:.6f}")
    abstract_peak = max(x for x, _ in found.states)
    print(f"  abstract peak  {abstract_peak:.4f} (limit {limit})")
    trace = found.replay(tree)
    true_peak = max(x for x, _, _, _ in trace)
    print(f"  replayed peak  {true_peak:.4f}")
    if true_peak > limit:
        print("  => REAL. The property is false and this trace is the proof.")
        return True
    print("  => SPURIOUS. The abstraction admitted a trajectory the system")
    print("     cannot take, and the replay settles it in microseconds.")
    step, state, truth, gap = found.divergence(tree)
    print(f"  worst slack at tick {step}: state ({state[0]:.4f}, "
          f"{state[1]:.4f}), solver chose {found.controls[step]:+.4f}, "
          f"network gives {truth:+.4f}, gap {gap:.4f}")
    print(f"  that state is {'past' if state[0] >= FENCE else 'inside'} the "
          "fence, so the cell containing it is the one to refine")
    return False


def main():
    """Extract a witness at each limit the proof fails on, and diagnose it."""
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    v0_bound = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    tree = Tree()
    print(f"counterexample diagnosis: {steps} steps, |v0| <= {v0_bound}, "
          f"true peak 1.2622\n")
    for grid in (12, 24):
        cells = crown_cells(grid)
        print(f"{len(cells)} contracts ({grid}x{grid}):")
        for limit in (HARD_LIMIT, 1.263, 1.20):
            found = witness(cells, steps, v0_bound, limit)
            if found is None:
                print(f"  limit {limit}: PROVED, no counterexample exists")
                continue
            print(f"  limit {limit}:")
            report(found, tree, limit)
        print()


if __name__ == "__main__":
    main()
