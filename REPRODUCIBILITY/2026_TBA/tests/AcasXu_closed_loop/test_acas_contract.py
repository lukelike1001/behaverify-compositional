"""
Minimal tests for typed ACAS contract specs.

Loads committed JSON under examples/AcasXu_closed_loop/contracts/.

ACAS Xu has one contract kind -- safety never-select. See
examples/AcasXu_closed_loop/contracts/discrete/liveness/DISCLAIMER.md.
"""

from __future__ import annotations

import pytest

from example_imports import activate_example

# Both examples ship a top-level `core` package; activate this one so import
# order between test files does not decide which package `core.*` resolves to.
_EXAMPLE = activate_example("AcasXu_closed_loop")

from core.acas_contract import AcasSafetyContract  # noqa: E402

_CONTRACTS = _EXAMPLE / "contracts"


def test_load_safety_corridor() -> None:
    path = (
        _CONTRACTS / "discrete" / "safety" / "safety_corridor_contracts.json"
    )
    contracts = AcasSafetyContract.load_json(path)
    assert len(contracts) == 2
    assert all(isinstance(c, AcasSafetyContract) for c in contracts)
    assert contracts[0].forbidden_advisory
    assert contracts[0].nn_input_lower
    assert contracts[0].dangerous_xy


def test_load_safety_full_sample() -> None:
    path = _CONTRACTS / "discrete" / "safety" / "safety_full_contracts.json"
    contracts = AcasSafetyContract.load_json(path)
    assert len(contracts) > 100
    first = contracts[0]
    assert first.contract_id == 1
    assert first.forbidden_advisory_idx >= 0


def test_safety_round_trip() -> None:
    path = (
        _CONTRACTS / "discrete" / "safety" / "safety_corridor_contracts.json"
    )
    original = AcasSafetyContract.load_json(path)[0]
    restored = AcasSafetyContract.from_dict(original.to_dict())
    assert restored.contract_id == original.contract_id
    assert restored.forbidden_advisory == original.forbidden_advisory
    assert restored.forbidden_advisory_idx == original.forbidden_advisory_idx
    assert restored.dangerous_xy == original.dangerous_xy
    assert restored.nn_input_lower == original.nn_input_lower


def test_crown_input_bounds() -> None:
    path = (
        _CONTRACTS / "discrete" / "safety" / "safety_corridor_contracts.json"
    )
    contract = AcasSafetyContract.load_json(path)[0]
    lower, upper = contract.crown_input_bounds()
    assert lower == contract.nn_input_lower
    assert upper == contract.nn_input_upper
