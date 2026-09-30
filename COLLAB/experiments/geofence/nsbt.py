"""
The geofence NSBT: the smallest honest neuro-symbolic behavior tree with
continuous state.

    selector
      |-- sequence
      |     |-- OutsideFence?   (check:  x >= FENCE)
      |     +-- Brake           (action: a := -BRAKE)
      +-- NeuralCruise          (action: a := net(x, v))

    x' = x + v*dt
    v' = v + a*dt

WHY THIS SHAPE. It is how safety-critical robots are actually built - a learned
inner-loop controller wrapped in a hand-written safety envelope - and it is the
minimal system with the property that defeats a single interval box: the two
branches accelerate in OPPOSITE directions, so a state set straddling the fence
splits into disjoint velocity clusters (COLLAB/experiments/geofence/
box_must_fail.py).  droneNew lacked this, which is why a box was exact there.

TWO SPECIFICATIONS, deliberately of different kinds:
  safety      x never exceeds HARD_LIMIT.  A reachability tool CAN express this.
  structural  whenever x >= FENCE, the Brake node is the one that ran.  A
              reachability tool CANNOT express this - it has no notion of which
              node is active - and neither can a neural network verifier.
"""
from __future__ import annotations

import pathlib

import numpy as np
import onnxruntime as ort

DT = 0.1
FENCE = 1.0
BRAKE = 1.0
HARD_LIMIT = 1.5
# resolved against this file so the module runs from any directory
NET = str(pathlib.Path(__file__).resolve().parent / "cruise.onnx")

X0 = (-2.0, 2.0)          # the region the drone may start in
V0 = (-1.0, 1.0)


# pylint: disable-next=too-few-public-methods
class Tree:
    """The behavior tree, executed concretely. Returns (accel, active_node)."""

    def __init__(self, net_path=NET):
        self.sess = ort.InferenceSession(net_path)
        self.inp = self.sess.get_inputs()[0].name

    def tick(self, x, v):
        """One tick: the selector tries the safety branch, then the network."""
        if x >= FENCE:                                   # OutsideFence? succeeds
            return -BRAKE, "Brake"
        state = np.array([[x, v]], np.float32)
        return float(self.sess.run(None, {self.inp: state})[0].ravel()[0]), "Cruise"


def simulate(tree, x, v, steps):
    """Run the closed loop, recording every state and which node fired."""
    trace = []
    for _ in range(steps):
        a, node = tree.tick(x, v)
        trace.append((x, v, a, node))
        x, v = x + v * DT, v + a * DT
    return trace


def main():
    """Check both specifications hold of the TRUE system before proving them."""
    tree = Tree()
    worst_x, worst_at = -np.inf, None
    structural_violations = 0
    total = 0

    for x0 in np.linspace(*X0, 41):
        for v0 in np.linspace(*V0, 21):
            for x, _, _, node in simulate(tree, float(x0), float(v0), 80):
                total += 1
                if x > worst_x:
                    worst_x, worst_at = x, (x0, v0)
                # structural: outside the fence, Brake must be the active node
                if x >= FENCE and node != "Brake":
                    structural_violations += 1

    print(f"geofence NSBT: fence x={FENCE}, hard limit x={HARD_LIMIT}")
    print(f"  {total} states from {41 * 21} starts x 80 steps")
    print(f"  SAFETY     max x reached = {worst_x:.4f} "
          f"(limit {HARD_LIMIT}) from start {worst_at[0]:.2f},{worst_at[1]:.2f}")
    print(f"             {'HOLDS' if worst_x <= HARD_LIMIT else '*** VIOLATED ***'}"
          f"  margin {HARD_LIMIT - worst_x:.4f}")
    print(f"  STRUCTURAL {structural_violations} violations "
          f"{'HOLDS' if structural_violations == 0 else '*** VIOLATED ***'}")


if __name__ == "__main__":
    main()
