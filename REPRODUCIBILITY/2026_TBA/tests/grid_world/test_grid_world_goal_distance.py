"""
Tests for the liveness ranking function: dist(s, g) and Dec(s, g).

These encode the descent argument that liveness contracts rest on. If any of
them fails, injecting Dec-membership guarantees no longer implies AF(goal),
regardless of what CROWN and nuXmv report.
"""

from __future__ import annotations

import pytest

from example_imports import activate_example

activate_example("grid_world")

from core.grid_world_domain import ACTIONS, DIR_IDX, GridWorldDomain  # noqa: E402
from core.liveness.grid_world_goal_distance import (  # noqa: E402
    GridWorldGoalDistance,
)

MAX_DESCENT_STEPS = 100


@pytest.fixture(scope="module")
def domain() -> GridWorldDomain:
    return GridWorldDomain.from_config()


@pytest.fixture(scope="module")
def goal_distance(domain: GridWorldDomain) -> GridWorldGoalDistance:
    return GridWorldGoalDistance.compute(domain)


@pytest.fixture(scope="module")
def progress_pairs(
    goal_distance: GridWorldGoalDistance,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return goal_distance.progress_pairs()


def test_committed_grid_shape(domain: GridWorldDomain) -> None:
    """The 7x7 / 18-obstacle instance the reported numbers describe."""
    assert (domain.grid_min, domain.grid_max) == (0, 6)
    assert len(domain.obstacles) == 18
    assert sum(1 for c in domain.all_cells if domain.is_safe(c)) == 31


def test_goal_has_distance_zero(goal_distance: GridWorldGoalDistance) -> None:
    for goal in goal_distance.distance_by_goal:
        assert goal_distance.distance(goal, goal) == 0
        assert goal_distance.decreasing_actions(goal, goal) == []


def test_every_progress_pair_has_a_decreasing_action(
    goal_distance: GridWorldGoalDistance,
    progress_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    """
    Totality of Dec.

    An empty Dec at a reachable non-goal state would make the liveness
    contract vacuously unsatisfiable there and strand the descent.
    """
    assert progress_pairs
    for source, goal in progress_pairs:
        assert goal_distance.decreasing_actions(source, goal), (
            f"empty Dec at source={source} goal={goal}"
        )


def test_decreasing_actions_drop_distance_by_exactly_one(
    domain: GridWorldDomain,
    goal_distance: GridWorldGoalDistance,
    progress_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    """Each action in Dec lands on a safe cell exactly one tick closer."""
    for source, goal in progress_pairs:
        here = goal_distance.distance(source, goal)
        for action in goal_distance.decreasing_actions(source, goal):
            landing = domain.simulate_step(source[0], source[1], action)
            assert domain.is_safe(landing)
            assert goal_distance.distance(landing, goal) == here - 1


def test_following_dec_reaches_goal_in_exactly_dist_steps(
    domain: GridWorldDomain,
    goal_distance: GridWorldGoalDistance,
    progress_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    """
    The well-founded descent argument, executed.

    A controller confined to Dec arrives in dist(s, g) ticks -- this is what
    the injected INVARs are supposed to buy on the SMV side.
    """
    for source, goal in progress_pairs:
        current, steps = source, 0
        while current != goal:
            action = goal_distance.decreasing_actions(current, goal)[0]
            current = domain.simulate_step(current[0], current[1], action)
            steps += 1
            assert steps <= MAX_DESCENT_STEPS, "descent failed to terminate"
        assert steps == goal_distance.distance(source, goal)


def test_stay_is_never_a_decreasing_action(
    goal_distance: GridWorldGoalDistance,
    progress_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    """
    XX excluded from Dec is what kills the hovering path.

    Hovering is why safety contracts alone leave the CTL specification false.
    """
    for source, goal in progress_pairs:
        assert "XX" not in goal_distance.decreasing_actions(source, goal)


def test_distance_is_at_least_manhattan(
    goal_distance: GridWorldGoalDistance,
    progress_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    """Obstacles can only lengthen a route, never shorten it."""
    for source, goal in progress_pairs:
        manhattan = abs(source[0] - goal[0]) + abs(source[1] - goal[1])
        assert goal_distance.distance(source, goal) >= manhattan


def test_free_space_is_fully_connected(
    domain: GridWorldDomain,
    goal_distance: GridWorldGoalDistance,
) -> None:
    """
    No safe cell is walled off from any other on this map.

    The CTL specification excuses goals inside obstacles; this records that
    the escape clause never fires for safe goals here.
    """
    safe_cells = [c for c in domain.all_cells if domain.is_safe(c)]
    for goal in safe_cells:
        for source in safe_cells:
            assert goal_distance.is_reachable(source, goal)


def test_obstacles_are_absent_from_the_table(
    domain: GridWorldDomain,
    goal_distance: GridWorldGoalDistance,
) -> None:
    """Obstacles are neither goals nor labelled sources."""
    for obstacle in domain.obstacles:
        assert obstacle not in goal_distance.distance_by_goal
    for distances in goal_distance.distance_by_goal.values():
        assert not (set(distances) & domain.obstacles)


def test_action_indices_match_the_dsl_class_order(
    goal_distance: GridWorldGoalDistance,
    progress_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    """
    Dec indices are what CROWN and the SMV patch consume.

    A mismatch here silently constrains the wrong output class.
    """
    assert DIR_IDX == {"We": 0, "Ea": 1, "No": 2, "So": 3}
    assert "XX" not in DIR_IDX
    for source, goal in progress_pairs[:50]:
        labels = goal_distance.decreasing_actions(source, goal)
        indices = goal_distance.decreasing_action_indices(source, goal)
        assert indices == [DIR_IDX[label] for label in labels]


def test_border_clamping_never_counts_as_progress(
    domain: GridWorldDomain,
    goal_distance: GridWorldGoalDistance,
    progress_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    """
    A move that clamps at the border is a self-loop, not a step.

    dist is built from simulate_step rather than raw adjacency precisely so
    that clamping cannot masquerade as forward motion.
    """
    for source, goal in progress_pairs:
        for action in goal_distance.decreasing_actions(source, goal):
            assert domain.simulate_step(source[0], source[1], action) != source


def test_unmerged_obligation_count_is_reported_accurately(
    goal_distance: GridWorldGoalDistance,
    progress_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    """
    Pins the cost figures the liveness plan is being judged against.

    930 progress pairs and 3596 never-select obligations are the baseline
    that region merging has to beat; a silent change to either invalidates
    the comparison against the monolithic table (7^4 = 2401 entries).
    """
    assert len(progress_pairs) == 930
    widths = [
        len(goal_distance.decreasing_actions(source, goal))
        for source, goal in progress_pairs
    ]
    assert min(widths) == 1
    assert max(widths) == 4
    assert sum(len(ACTIONS) - width for width in widths) == 3596
