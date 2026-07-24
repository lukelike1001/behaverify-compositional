"""
Minimal tests for typed ACAS contract specs.

Loads committed JSON under examples/AcasXu_closed_loop/contracts/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "AcasXu_closed_loop"
)
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from core.acas_contract import AcasLivenessContract, AcasSafetyContract  # noqa: E402

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


def test_load_liveness_contracts() -> None:
    path = _CONTRACTS / "discrete" / "liveness" / "liveness_contracts.json"
    contracts = AcasLivenessContract.load_json(path)
    assert len(contracts) == 52
    assert all(isinstance(c, AcasLivenessContract) for c in contracts)
    assert contracts[0].required_advisory
    assert contracts[0].x_mag >= 0


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


def test_liveness_round_trip() -> None:
    path = _CONTRACTS / "discrete" / "liveness" / "liveness_contracts.json"
    original = AcasLivenessContract.load_json(path)[0]
    restored = AcasLivenessContract.from_dict(original.to_dict())
    assert restored.contract_id == original.contract_id
    assert restored.required_advisory == original.required_advisory
    assert restored.x_mag == original.x_mag
    assert restored.y_mag == original.y_mag
    assert restored.nn_input_upper == original.nn_input_upper


def test_crown_input_bounds() -> None:
    path = _CONTRACTS / "discrete" / "liveness" / "liveness_contracts.json"
    contract = AcasLivenessContract.load_json(path)[0]
    lower, upper = contract.crown_input_bounds()
    assert lower == contract.nn_input_lower
    assert upper == contract.nn_input_upper
