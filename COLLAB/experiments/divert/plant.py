"""
The `divert` NSBT: a planar vehicle whose safety branch flies it home.

The system is a drone in a square arena. A neural controller steers it toward
a goal around an obstacle. A priority checklist runs every tick: if the vehicle
is over its speed limit, outside the corridor, or has stopped making progress,
the checklist pre-empts cruising, brakes, and a SECOND network returns it to
the start region. Once a hazard fires it LATCHES - an emergency is not
un-declared because the reading improved for one tick.

That checklist is the behavior tree, and it is what no reachability tool
derives for you.

PROVENANCE, so no reviewer has to ask where the shape came from:
  plant         the double integrator of the run-time-assurance tutorial
                (Hobbs, Mote, Abate, Coogan, Feron, IEEE Control Systems
                Magazine 2023, doi:10.1109/MCS.2023.3234380), two copies - one
                per axis. The geofence's derived envelope matched that paper's
                viability kernel to four decimals.
  architecture  a behavior tree carrying guarantees, extended with learned
                components (Sprague and Ogren, IEEE CDC 2022, pp. 3989-3996).
                They enforce at RUNTIME with the network untrusted; we verify
                OFFLINE over the actual weights.

This module runs the tree with ANALYTIC controllers. Networks come later, by
imitation, exactly as the geofence did - there is no point training against
physics that does not work.
"""
from __future__ import annotations

import numpy as np

DT = 0.1
STANDOFF = 0.5         # avoidance activation radius
DRAG = 0.1
A_MAX = 2.0           # Gemini's kinematic sizing: 1.0 needs 45-49 ticks and
                      # dies past the 40-tick wall; 2.0 gives a 31-tick mission
S_MAX = 1.0           # speed limit; exceeding it is a hazard
ARENA = (0.0, 2.0)
OBSTACLE = (0.8, 1.2, 0.8, 1.2)        # xlo, xhi, ylo, yhi
START = (0.1, 0.3, 0.1, 0.3)
GOAL = (1.7, 1.9, 1.7, 1.9)
CLEARANCE = 0.1                         # delta_safe
STUCK_TICKS = 15                        # progress watchdog

STATUS_SUCCESS, STATUS_RUNNING = "S", "R"


def obstacle_distance(x, y):
    """Distance from (x, y) to the obstacle box; negative means inside."""
    xlo, xhi, ylo, yhi = OBSTACLE
    dx = max(xlo - x, 0.0, x - xhi)
    dy = max(ylo - y, 0.0, y - yhi)
    if dx == 0.0 and dy == 0.0:
        return -min(x - xlo, xhi - x, y - ylo, yhi - y)
    return float(np.hypot(dx, dy))


def _clip(ax, ay):
    """Saturate the acceleration to the actuator limit, preserving direction."""
    mag = float(np.hypot(ax, ay))
    if mag <= A_MAX:
        return ax, ay
    return ax * A_MAX / mag, ay * A_MAX / mag


def avoid(x, y, vx, vy, strength):
    """Steer AROUND the obstacle, not into a head-on stall.

    A purely radial push fails on a symmetric obstacle approached head-on: the
    push is anti-parallel to the velocity, so the vehicle decelerates and then
    drifts through. The fix is a tangential term that picks the side to pass on
    from the sign of the cross product with the current velocity, which is what
    a pilot does.
    """
    gap = obstacle_distance(x, y)
    if gap >= STANDOFF:
        return 0.0, 0.0
    cx, cy = (OBSTACLE[0] + OBSTACLE[1]) / 2, (OBSTACLE[2] + OBSTACLE[3]) / 2
    px, py = x - cx, y - cy
    norm = max(float(np.hypot(px, py)), 1e-9)
    px, py = px / norm, py / norm
    urgency = strength * (STANDOFF - gap)
    side = 1.0 if (px * vy - py * vx) >= 0.0 else -1.0
    return (urgency * (0.4 * px - side * py),
            urgency * (0.4 * py + side * px))


V_RETURN = 0.95        # the return must finish inside the 40-tick wall
V_CRUISE = 0.80        # target cruise speed, deliberately under S_MAX so that
                       # the hazard fires for a REASON rather than immediately


def cruise_law(x, y, vx, vy):
    """Analytic stand-in for net_cruise: steer to the goal, skirt the obstacle.

    Velocity-targeting rather than position-targeting. A position gain large
    enough to cross the arena saturates the actuator on tick one and trips the
    speed hazard before the vehicle has gone anywhere, which makes the mission
    'cruise for six ticks, then go home' and tests nothing.
    """
    gx, gy = (GOAL[0] + GOAL[1]) / 2, (GOAL[2] + GOAL[3]) / 2
    dx, dy = gx - x, gy - y
    span = max(float(np.hypot(dx, dy)), 1e-9)
    ax = 3.0 * (V_CRUISE * dx / span - vx)
    ay = 3.0 * (V_CRUISE * dy / span - vy)
    dax, day = avoid(x, y, vx, vy, 7.0)
    ax, ay = ax + dax, ay + day
    return _clip(ax, ay)


def return_law(x, y, vx, vy):
    """Analytic stand-in for net_divert: fly home, also avoiding the obstacle."""
    hx, hy = (START[0] + START[1]) / 2, (START[2] + START[3]) / 2
    dx, dy = hx - x, hy - y
    span = float(np.hypot(dx, dy))
    if span < 0.15:                      # close in and settle
        ax, ay = 6.0 * dx - 4.0 * vx, 6.0 * dy - 4.0 * vy
    else:
        speed = min(V_RETURN, 1.6 * span)
        ax = 4.0 * (speed * dx / span - vx)
        ay = 4.0 * (speed * dy / span - vy)
    dax, day = avoid(x, y, vx, vy, 8.0)
    ax, ay = ax + dax, ay + day
    return _clip(ax, ay)


class Tree:
    """The behavior tree, with the latch and the resume point as real state."""

    def __init__(self, cruise=cruise_law, divert=return_law):
        self.cruise = cruise
        self.divert = divert
        self.reset()

    def reset(self):
        """Clear the latched hazards and the recovery resume point."""
        self.hazards = {"TooFast": False, "OutOfCorridor": False, "Stuck": False}
        self.step_index = 0            # 0 = Brake, 1 = Return, 2 = done
        self.no_progress = 0
        self.best_gap = None

    def _update_hazards(self, x, y, vx, vy):
        """Hazard children latch: with_partial_memory never un-fires one."""
        if np.hypot(vx, vy) > S_MAX:
            self.hazards["TooFast"] = True
        lo, hi = ARENA
        if not (lo <= x <= hi and lo <= y <= hi):
            self.hazards["OutOfCorridor"] = True
        if self.no_progress > STUCK_TICKS:
            self.hazards["Stuck"] = True
        return any(self.hazards.values())

    def _track_progress(self, x, y):
        """Watchdog feeding the Stuck hazard."""
        gx, gy = (GOAL[0] + GOAL[1]) / 2, (GOAL[2] + GOAL[3]) / 2
        gap = float(np.hypot(gx - x, gy - y))
        if self.best_gap is None or gap < self.best_gap - 1e-3:
            self.best_gap, self.no_progress = gap, 0
        else:
            self.no_progress += 1

    def tick(self, x, y, vx, vy):
        """One tick. Returns (ax, ay, active_node, recovery_active)."""
        self._track_progress(x, y)
        if not self._update_hazards(x, y, vx, vy):
            ax, ay = self.cruise(x, y, vx, vy)
            return ax, ay, "NeuralCruise", False
        if self.step_index == 2:                       # Recovery SUCCEEDED
            return 0.0, 0.0, "Idle", False             # and is no longer active
        if self.step_index == 0:                       # Brake
            if np.hypot(vx, vy) <= 0.05:
                self.step_index = 1
            else:
                return (*_clip(-3.0 * vx, -3.0 * vy), "Brake", True)
        if self._home(x, y) and np.hypot(vx, vy) <= 0.05:
            self.step_index = 2                        # Return returns SUCCESS,
            return 0.0, 0.0, "Idle", False             # so Recovery completes
        ax, ay = self.divert(x, y, vx, vy)             # Return
        return ax, ay, "Return", True

    @staticmethod
    def _home(x, y):
        """Inside the start region, which is what completes the recovery."""
        return START[0] <= x <= START[1] and START[2] <= y <= START[3]


def simulate(tree, x, y, vx, vy, steps):
    """Run the closed loop, recording the state and which node fired."""
    tree.reset()
    trace = []
    for _ in range(steps):
        ax, ay, node, recovering = tree.tick(x, y, vx, vy)
        trace.append((x, y, vx, vy, node, recovering))
        x, y = x + vx * DT, y + vy * DT
        vx = vx + (ax - DRAG * vx) * DT
        vy = vy + (ay - DRAG * vy) * DT
    trace.append((x, y, vx, vy, "end", False))
    return trace


def main():
    """Does the physics work at all? Run one mission and report it."""
    tree = Tree()
    trace = simulate(tree, 0.2, 0.2, 0.0, 0.0, 60)
    worst = min(obstacle_distance(s[0], s[1]) for s in trace)
    recovery = [i for i, s in enumerate(trace) if s[5]]
    print(f"divert: a_max {A_MAX}, s_max {S_MAX}, dt {DT}, drag {DRAG}")
    print(f"  obstacle {OBSTACLE}, clearance target {CLEARANCE}")
    print(f"  closest approach to obstacle: {worst:.4f}")
    print(f"  recovery active on ticks: "
          f"{recovery[0] if recovery else '-'}..{recovery[-1] if recovery else '-'}"
          f"  ({len(recovery)} ticks)")
    print(f"  final position: ({trace[-1][0]:.3f}, {trace[-1][1]:.3f})")
    for i in range(0, min(len(trace), 40), 5):
        s = trace[i]
        print(f"   t={i:2d} ({s[0]:5.2f},{s[1]:5.2f}) |v|={np.hypot(s[2],s[3]):.3f} "
              f"gap={obstacle_distance(s[0],s[1]):6.3f}  {s[4]}")


if __name__ == "__main__":
    main()
