"""
Tests for the grid-world safety side: viability partition and ∂V contracts.

Guards the refactor that split grid_world_viability.py into core/: the
partition numbers and the 38-contract set are what the committed contract
JSONs and the reported results describe.
"""

from __future__ import annotations

import json

import pytest

from example_imports import activate_example

_EXAMPLE_ROOT = activate_example("grid_world")

from core.grid_world_contract import (  # noqa: E402
    GridWorldContract,
    GridWorldSafetyContract,
)
from core.grid_world_domain import ACTIONS, GridWorldDomain  # noqa: E402
from core.safety.grid_world_safety_contract_generator import (  # noqa: E402
    GridWorldSafetyContractGenerator,
    generate_contracts,
)
from core.safety.grid_world_viability import GridWorldViabilityKernel  # noqa: E402

_COMMITTED_CONTRACTS = (
    _EXAMPLE_ROOT
    / "contracts"
    / "discrete"
    / "safety"
    / "1000__6_18_0__0200_1_discrete.json"
)


@pytest.fixture(scope="module")
def domain() -> GridWorldDomain:
    return GridWorldDomain.from_config()


@pytest.fixture(scope="module")
def kernel(domain: GridWorldDomain) -> GridWorldViabilityKernel:
    return GridWorldViabilityKernel.compute(domain)


@pytest.fixture(scope="module")
def contracts() -> list[GridWorldSafetyContract]:
    return generate_contracts()


def test_hover_theorem_holds(
    domain: GridWorldDomain, kernel: GridWorldViabilityKernel,
) -> None:
    """
    Stay makes every safe cell viable, so V = Safe and nothing is doomed.

    This is what collapses the safety contracts to one-step crash avoidance.
    """
    safe = frozenset(c for c in domain.all_cells if domain.is_safe(c))
    assert kernel.fixpoint_rounds == 0
    assert kernel.V == safe
    assert kernel.safe_but_doomed == frozenset()
    assert all("XX" in kernel.allowed[cell] for cell in kernel.V)


def test_partition_covers_the_grid(
    domain: GridWorldDomain, kernel: GridWorldViabilityKernel,
) -> None:
    assert kernel.unsafe == domain.obstacles
    assert len(kernel.V) + len(kernel.unsafe) == domain.side_length ** 2
    assert len(kernel.V) == 31
    assert len(kernel.interior) == 6
    assert len(kernel.boundary) == 25


def test_boundary_equals_obstacle_adjacent_geometry(
    domain: GridWorldDomain, kernel: GridWorldViabilityKernel,
) -> None:
    """∂V from Allowed_V agrees with the independent geometric construction."""
    assert kernel.boundary == domain.obstacle_adjacent_cells()


def test_contract_count_and_identity_uniqueness(
    contracts: list[GridWorldSafetyContract],
) -> None:
    assert len(contracts) == 38
    identities = [c.identity() for c in contracts]
    assert len(identities) == len(set(identities))


def test_every_contract_forbids_a_real_crash(
    domain: GridWorldDomain,
    kernel: GridWorldViabilityKernel,
    contracts: list[GridWorldSafetyContract],
) -> None:
    for contract in contracts:
        assert contract.source in kernel.boundary
        assert contract.forbidden_dir not in kernel.allowed[contract.source]
        landing = domain.simulate_step(
            contract.source[0], contract.source[1], contract.forbidden_dir,
        )
        assert landing == contract.obstacle
        assert landing in domain.obstacles


def test_stay_is_never_forbidden(
    contracts: list[GridWorldSafetyContract],
) -> None:
    """XX keeps every safe cell in V, so it is never a contract consequent."""
    assert all(c.forbidden_dir != "XX" for c in contracts)
    assert all(c.forbidden_dir in ACTIONS for c in contracts)


def test_generator_agrees_with_the_public_entry_point(
    domain: GridWorldDomain, contracts: list[GridWorldSafetyContract],
) -> None:
    from_generator = GridWorldSafetyContractGenerator.from_domain(
        domain,
    ).generate_all_contracts()
    assert [c.identity() for c in from_generator] == [
        c.identity() for c in contracts
    ]


def test_safety_contracts_quantify_over_every_goal(
    contracts: list[GridWorldSafetyContract],
) -> None:
    """
    goal_region is None for safety: "never crash, wherever the target is".

    A liveness contract pins one goal instead, which is the distinction the
    shared base class exists to carry.
    """
    assert all(c.goal_region is None for c in contracts)
    assert all("goal" not in c.to_spec_dict() for c in contracts)


def test_spec_dict_schema(contracts: list[GridWorldSafetyContract]) -> None:
    record = contracts[0].to_spec_dict(contract_id=1)
    assert set(record) == {
        "id", "source", "description", "obstacle",
        "forbidden_dir", "forbidden_dir_idx",
    }
    assert record["id"] == 1
    assert contracts[0].to_spec_dict().get("id") is None


def test_contract_set_matches_committed_json(
    contracts: list[GridWorldSafetyContract],
) -> None:
    """
    Value parity with the April 2026 artifacts, order-insensitively.

    The generator's emission order differs from those files; the contract set
    must not.
    """
    committed = json.loads(_COMMITTED_CONTRACTS.read_text())["contracts"]
    fields = ("obstacle", "source", "forbidden_dir", "forbidden_dir_idx")

    def key(record: dict) -> tuple:
        return tuple(
            tuple(record[f]) if isinstance(record[f], list) else record[f]
            for f in fields
        )

    assert {key(c.to_spec_dict()) for c in contracts} == {
        key(record) for record in committed
    }


def test_safety_contract_subclasses_the_shared_base() -> None:
    assert issubclass(GridWorldSafetyContract, GridWorldContract)
    with pytest.raises(TypeError):
        GridWorldContract(source=(0, 0))  # type: ignore[abstract]
