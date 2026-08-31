"""
grid_world_goal_distance.py

Goal-relative progress measure for the 1-NN grid-world NSBT: the least
fixpoint dual of the viability kernel.

    dist(s, g)  shortest number of ticks from cell s to goal g, moving only
                through safe cells under GridWorldDomain physics
    Dec(s, g)   the actions at s that strictly decrease dist(., g)

dist is a ranking function: every action in Dec(s, g) drops it by exactly one,
and it is bounded below by zero, so a controller confined to Dec from every
state reaches g in at most dist(s, g) ticks. That is the well-founded descent
argument behind liveness contracts, and the reason Dec -- not a single pinned
action -- is the guarantee: it constrains the network without determinizing it.

Computed from GridWorldDomain physics only -- no networks. Orthogonal to
core/safety/grid_world_viability.py: V is "can stay safe forever" (greatest
fixpoint), dist is "how far from the goal" (least fixpoint from g).

Note on XX: stay is never in Dec, since simulate_step(s, "XX") == s leaves dist
unchanged. Excluding it is what rules out the hovering path that makes the
CTL specification false under safety contracts alone.

Run from examples/grid_world/:

    python3 -m core.liveness.grid_world_goal_distance
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from core.grid_world_domain import ACTIONS, DIR_IDX, GridWorldDomain

Cell = tuple[int, int]


@dataclass(frozen=True)
class GridWorldGoalDistance:
    """
    BFS distance-to-goal table over the free cells of one grid.

    distance_by_goal[g][s] is dist(s, g). Cells with no safe route to g are
    absent from the inner dict rather than stored as infinity, so membership
    is the reachability test.
    """

    domain: GridWorldDomain
    distance_by_goal: dict[Cell, dict[Cell, int]]

    @classmethod
    def compute(cls, domain: GridWorldDomain | None = None) -> GridWorldGoalDistance:
        """One backward BFS per safe goal cell; all cells labelled per goal."""
        if domain is None:
            domain = GridWorldDomain.from_config()

        predecessors = cls._build_predecessor_table(domain)
        safe_cells = [s for s in domain.all_cells if domain.is_safe(s)]
        return cls(
            domain=domain,
            distance_by_goal={
                goal: cls._breadth_first_distances(goal, predecessors)
                for goal in safe_cells
            },
        )

    @staticmethod
    def _build_predecessor_table(
        domain: GridWorldDomain,
    ) -> dict[Cell, list[Cell]]:
        """
        Reverse of the one-step successor relation, restricted to safe cells.

        Built from simulate_step rather than raw adjacency so that border
        clamping is respected: a move that clamps is a self-loop and never
        yields a predecessor.
        """
        predecessors: dict[Cell, list[Cell]] = {
            s: [] for s in domain.all_cells if domain.is_safe(s)
        }
        for source in domain.all_cells:
            if not domain.is_safe(source):
                continue
            for action in ACTIONS:
                if action == "XX":
                    continue
                landing = domain.simulate_step(source[0], source[1], action)
                if landing == source or not domain.is_safe(landing):
                    continue
                predecessors[landing].append(source)
        return predecessors

    @staticmethod
    def _breadth_first_distances(
        goal: Cell,
        predecessors: dict[Cell, list[Cell]],
    ) -> dict[Cell, int]:
        """Distances to `goal` for every cell that can reach it."""
        distances: dict[Cell, int] = {goal: 0}
        queue: deque[Cell] = deque([goal])
        while queue:
            current = queue.popleft()
            for previous in predecessors[current]:
                if previous not in distances:
                    distances[previous] = distances[current] + 1
                    queue.append(previous)
        return distances

    # --- queries -----------------------------------------------------------

    def distance(self, source: Cell, goal: Cell) -> int | None:
        """dist(source, goal); None when no safe route exists."""
        return self.distance_by_goal[goal].get(source)

    def is_reachable(self, source: Cell, goal: Cell) -> bool:
        return self.distance(source, goal) is not None

    def decreasing_actions(self, source: Cell, goal: Cell) -> list[str]:
        """
        Dec(source, goal): actions whose landing cell is one tick closer.

        Empty at the goal itself and at cells with no route to it. Nonempty
        everywhere else, which is what makes the descent argument total.
        """
        here = self.distance(source, goal)
        if here is None or here == 0:
            return []
        decreasing: list[str] = []
        for action in ACTIONS:
            if action == "XX":
                continue
            landing = self.domain.simulate_step(source[0], source[1], action)
            if not self.domain.is_safe(landing):
                continue
            if self.distance(landing, goal) == here - 1:
                decreasing.append(action)
        return decreasing

    def decreasing_action_indices(self, source: Cell, goal: Cell) -> list[int]:
        """Dec(source, goal) as NN class indices (We=0 Ea=1 No=2 So=3)."""
        return [DIR_IDX[a] for a in self.decreasing_actions(source, goal)]

    def progress_pairs(self) -> list[tuple[Cell, Cell]]:
        """
        Every (source, goal) pair that needs a liveness contract.

        Excludes goal cells themselves (already arrived) and unreachable
        sources (the CTL specification excuses walled-off targets).
        """
        return [
            (source, goal)
            for goal, distances in sorted(self.distance_by_goal.items())
            for source, d in sorted(distances.items())
            if d > 0
        ]


def main() -> None:
    domain = GridWorldDomain.from_config()
    table = GridWorldGoalDistance.compute(domain)

    safe_cells = [s for s in domain.all_cells if domain.is_safe(s)]
    pairs = table.progress_pairs()
    widths = [len(table.decreasing_actions(s, g)) for s, g in pairs]
    unreachable = sum(
        1
        for goal in safe_cells
        for source in safe_cells
        if not table.is_reachable(source, goal)
    )

    print(
        f"grid = [{domain.grid_min},{domain.grid_max}]^2  "
        f"({len(safe_cells)} safe cells, {len(domain.obstacles)} obstacles)"
    )
    print(f"progress pairs (source != goal, reachable) = {len(pairs)}")
    print(f"unreachable (source, goal) pairs           = {unreachable}")
    print(f"|Dec| min / max                            = {min(widths)} / {max(widths)}")
    print(f"forbidden actions per pair (5 - |Dec|)     = "
          f"{5 - max(widths)} .. {5 - min(widths)}")
    print(f"never-select obligations if unmerged       = "
          f"{sum(5 - w for w in widths)}")

    example_source, example_goal = pairs[len(pairs) // 2]
    print()
    print(
        f"example: source {example_source} → goal {example_goal}  "
        f"dist = {table.distance(example_source, example_goal)}  "
        f"Dec = {table.decreasing_actions(example_source, example_goal)}"
    )


if __name__ == "__main__":
    main()
