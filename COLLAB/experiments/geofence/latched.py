"""
The geofence NSBT with hysteresis: where node status stops being a function of
the continuous state.

In the v1 tree (nsbt.py) the active node is decided by x alone - x >= FENCE
means Brake, otherwise NeuralCruise.  A reachability tool tracking (x, v) could
therefore express "Brake is active" as the state predicate x >= FENCE, so the
expressiveness argument does not yet bite.

Real safety envelopes do not work that way.  Releasing the brake the instant the
boundary is cleared makes the controller CHATTER at the fence, so engineers add
HYSTERESIS: once braking starts it continues until the state is comfortably back
inside.  That turns the envelope into a latch.

    selector
      |-- sequence
      |     |-- ShouldBrake?   x >= FENCE  OR  (braking AND x >= FENCE - MARGIN)
      |     +-- Brake          a := -BRAKE ;  braking := True
      +-- sequence
            |-- ReleaseLatch   braking := False
            +-- NeuralCruise   a := net(x, v)

Now two executions can sit at the SAME (x, v) inside the hysteresis band and
have DIFFERENT active nodes, because one of them was braking a tick ago and the
other was not.  A reachability tool that tracks only (x, v) cannot tell them
apart - not imprecisely, but in principle.  This file demonstrates that
concretely rather than arguing it.
"""
from __future__ import annotations

import numpy as np

from nsbt import BRAKE, DT, FENCE, Tree

MARGIN = 0.3          # release the brake only once x < FENCE - MARGIN


class LatchedTree(Tree):
    """The v2 tree. Carries one bit of blackboard state: `braking`."""

    def tick_latched(self, x, v, braking):
        """One tick. Returns (accel, active_node, next_braking)."""
        should_brake = x >= FENCE or (braking and x >= FENCE - MARGIN)
        if should_brake:
            return -BRAKE, "Brake", True
        return self.tick(x, v)[0], "NeuralCruise", False


def run(tree, x, v, braking, steps):
    """Execute the latched closed loop, recording (x, v, node, braking)."""
    trace = []
    for _ in range(steps):
        a, node, braking = tree.tick_latched(x, v, braking)
        trace.append((x, v, node, braking))
        x, v = x + v * DT, v + a * DT
    return trace


def main():
    """Find two executions at the same (x, v) whose active nodes differ."""
    tree = LatchedTree()
    print(f"hysteresis band: [{FENCE - MARGIN}, {FENCE}) - inside it the active")
    print("node depends on whether the system was braking a tick ago.\n")

    # sweep initial conditions rather than hand-picking two trajectories
    seen: dict[tuple[float, float], tuple[str, bool, tuple]] = {}
    collisions = []
    for x0 in np.linspace(0.2, 1.3, 45):
        for v0 in np.linspace(-0.8, 0.8, 33):
            for x, v, node, braking in run(tree, float(x0), float(v0), False, 30):
                key = (round(x, 2), round(v, 2))
                prev = seen.get(key)
                if prev is None:
                    seen[key] = (node, braking, (x0, v0))
                elif prev[0] != node:
                    collisions.append((key, prev, (node, braking, (x0, v0))))

    print(f"  swept {45 * 33} starts x 30 steps over {len(seen)} distinct (x,v) cells")
    for key, a, b in collisions[:3]:
        print(f"\n  SAME STATE (x={key[0]}, v={key[1]}), DIFFERENT ACTIVE NODE:")
        print(f"    from start ({a[2][0]:.2f},{a[2][1]:.2f}): "
              f"node={a[0]:<13} braking={a[1]}")
        print(f"    from start ({b[2][0]:.2f},{b[2][1]:.2f}): "
              f"node={b[0]:<13} braking={b[1]}")

    # and the property that needs the latch to state at all
    worst_run = 0
    for x0 in np.linspace(0.2, 1.3, 45):
        run_len = 0
        for _, _, node, _ in run(tree, float(x0), 0.0, False, 30):
            run_len = run_len + 1 if node == "Brake" else 0
            worst_run = max(worst_run, run_len)
    print(f"\n  longest consecutive Brake activation: {worst_run} ticks")
    print("  'Brake never stays active more than N ticks' is a property about the")
    print("  TREE's state, not the drone's position, and is the shape of property")
    print("  a safety engineer actually writes.")


if __name__ == "__main__":
    main()
