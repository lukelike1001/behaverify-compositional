"""
Tests for grid-world liveness contract records and their SMV injection.

The contract layer is where the ranking function becomes something CROWN and
nuXmv can consume. Two properties matter and neither is visible from the
distance table alone: allowed/forbidden must partition the action set, and the
injected INVAR must guard on the goal as well as the drone.
"""

from __future__ import annotations

import sys

import pytest

from example_imports import activate_example

_EXAMPLE_ROOT = activate_example("grid_world")

from core.grid_world_contract import (  # noqa: E402
    GridWorldContract,
    GridWorldLivenessContract,
)
from core.grid_world_domain import ACTIONS, GridWorldDomain  # noqa: E402
from core.liveness.grid_world_goal_distance import (  # noqa: E402
    GridWorldGoalDistance,
)
from core.liveness.grid_world_liveness_contract_generator import (  # noqa: E402
    STAY_CLASS_INDEX,
    GridWorldLivenessContractGenerator,
)

_SRC_DIR = str((_EXAMPLE_ROOT / ".." / ".." / "src").resolve())
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dsl_with_contracts_to_nuxmv import (  # noqa: E402
    DEFAULT_DIR_MAP,
    build_invar_lines,
)


@pytest.fixture(scope="module")
def generator() -> GridWorldLivenessContractGenerator:
    return GridWorldLivenessContractGenerator.from_domain()


@pytest.fixture(scope="module")
def contracts(
    generator: GridWorldLivenessContractGenerator,
) -> list[GridWorldLivenessContract]:
    return generator.generate_all_contracts()


def test_one_contract_per_progress_pair(
    generator: GridWorldLivenessContractGenerator,
    contracts: list[GridWorldLivenessContract],
) -> None:
    assert len(contracts) == 930
    assert len(contracts) == len(generator.goal_distance.progress_pairs())
    identities = [c.identity() for c in contracts]
    assert len(identities) == len(set(identities))


def test_allowed_and_forbidden_partition_the_action_set(
    contracts: list[GridWorldLivenessContract],
) -> None:
    """
    Every action is either required-eligible or forbidden, never both.

    An overlap would emit an INVAR forbidding an action the contract also
    permits, making the model vacuously unsatisfiable at that cell.
    """
    for contract in contracts:
        allowed = set(contract.allowed_dirs)
        forbidden = set(contract.forbidden_dirs)
        assert allowed | forbidden == set(ACTIONS)
        assert not (allowed & forbidden)
        assert allowed


def test_stay_is_always_forbidden(
    contracts: list[GridWorldLivenessContract],
) -> None:
    """XX never makes progress, so every contract must rule it out."""
    for contract in contracts:
        assert "XX" in contract.forbidden_dirs
        assert STAY_CLASS_INDEX in contract.forbidden_dir_idxs
        assert "XX" not in contract.allowed_dirs


def test_class_indices_cover_all_five_outputs(
    contracts: list[GridWorldLivenessContract],
) -> None:
    """Indices must span the DSL order We=0 Ea=1 No=2 So=3 XX=4."""
    for contract in contracts[:100]:
        combined = set(contract.allowed_dir_idxs) | set(
            contract.forbidden_dir_idxs
        )
        assert combined == {0, 1, 2, 3, 4}


def test_distance_matches_the_ranking_table(
    generator: GridWorldLivenessContractGenerator,
    contracts: list[GridWorldLivenessContract],
) -> None:
    for contract in contracts:
        assert contract.distance == generator.goal_distance.distance(
            contract.source, contract.goal_region,
        )
        assert contract.distance > 0


def test_goal_region_is_required_and_serialized(
    contracts: list[GridWorldLivenessContract],
) -> None:
    """
    Unlike safety, a liveness contract pins one goal.

    The goal must survive into the JSON: the SMV guard needs it, and without
    it the constraint would apply for every target.
    """
    for contract in contracts[:100]:
        assert contract.goal_region is not None
        assert contract.to_spec_dict()["goal"] == list(contract.goal_region)


def test_obligation_count_is_the_reported_baseline(
    generator: GridWorldLivenessContractGenerator,
) -> None:
    assert generator.crown_obligation_count() == 3596


def test_invar_lines_guard_on_drone_and_goal() -> None:
    """
    The injected constraint must name all four coordinates.

    Dropping the goal guard is the one unsound failure mode here: the same
    action forbidden for one target is required for another.
    """
    record = {
        "source": [4, 0],
        "goal": [3, 5],
        "forbidden_dirs": ["We", "No", "So", "XX"],
    }
    lines = build_invar_lines(
        [record], "network_stage_0", "drone_x_stage_0", "drone_y_stage_0",
        DEFAULT_DIR_MAP, "destination_x_stage_0", "destination_y_stage_0",
    )
    assert len(lines) == 4
    for line in lines:
        assert "drone_x_stage_0 = 4" in line
        assert "drone_y_stage_0 = 0" in line
        assert "destination_x_stage_0 = 3" in line
        assert "destination_y_stage_0 = 5" in line
    assert {line.rsplit("!= ", 1)[1].rstrip(";") for line in lines} == {
        "left", "up", "down", "no_action",
    }


def test_invar_lines_reject_missing_goal_variables() -> None:
    """A liveness record without goal SMV names must fail loudly, not silently."""
    record = {"source": [4, 0], "goal": [3, 5], "forbidden_dirs": ["We"]}
    with pytest.raises(ValueError, match="goal_x"):
        build_invar_lines(
            [record], "network_stage_0", "drone_x_stage_0", "drone_y_stage_0",
            DEFAULT_DIR_MAP,
        )


def test_safety_records_still_inject_without_goal_variables() -> None:
    """The never_selects path must keep working unchanged."""
    record = {"source": [0, 2], "forbidden_dir": "So"}
    lines = build_invar_lines(
        [record], "network_stage_0", "drone_x_stage_0", "drone_y_stage_0",
        DEFAULT_DIR_MAP,
    )
    assert len(lines) == 1
    assert "destination" not in lines[0]
    assert lines[0].endswith("!= down;")


def test_liveness_contract_subclasses_the_shared_base() -> None:
    assert issubclass(GridWorldLivenessContract, GridWorldContract)


def test_generator_matches_a_hand_computed_pair() -> None:
    """
    Spot-check one pair end to end against the distance table.

    Guards against an index/label mix-up that the aggregate tests would miss.
    """
    domain = GridWorldDomain.from_config()
    table = GridWorldGoalDistance.compute(domain)
    generator = GridWorldLivenessContractGenerator(goal_distance=table)
    by_identity = {
        c.identity(): c for c in generator.generate_all_contracts()
    }

    source, goal = (4, 0), (3, 5)
    contract = by_identity[(*source, *goal)]
    assert contract.allowed_dirs == tuple(
        table.decreasing_actions(source, goal)
    )
    assert contract.distance == table.distance(source, goal)


def test_liveness_coverage_leaves_obstacle_goals_unconstrained(
    generator: GridWorldLivenessContractGenerator,
    contracts: list[GridWorldLivenessContract],
) -> None:
    """
    Progress contracts cover 930 of 1519 occupiable (drone, goal) states.

    The environment picks a new goal nondeterministically over the whole grid,
    obstacles included, and dist(s, g) is undefined for those, so 558 states
    carry no progress obligation. This is why a liveness-only run cannot
    establish INVARSPEC: the abstract network is free at exactly those states.
    Safety contracts, which guard on the drone cell for every goal, are what
    cover them.
    """
    domain = generator.domain
    safe_cells = [c for c in domain.all_cells if domain.is_safe(c)]
    occupiable = [(s, g) for s in safe_cells for g in domain.all_cells]
    covered = {c.identity() for c in contracts}

    uncovered = [
        (s, g) for s, g in occupiable if (*s, *g) not in covered
    ]
    goal_in_obstacle = [(s, g) for s, g in uncovered if g in domain.obstacles]
    already_at_goal = [(s, g) for s, g in uncovered if s == g]

    assert len(occupiable) == 1519
    assert len(covered) == 930
    assert len(goal_in_obstacle) == 558
    assert len(already_at_goal) == 31
    assert len(uncovered) == len(goal_in_obstacle) + len(already_at_goal)
