"""
acas_contract.py

Typed A/G contract specs for ACAS Xu.

AcasSafetyContract -- NN must NOT select forbidden_advisory (never_selects)

ACAS Xu has no liveness specification, so there is no second contract kind
here; see contracts/discrete/liveness/DISCLAIMER.md.

Specs live under:
  contracts/discrete/safety/safety_full_contracts.json
  contracts/discrete/safety/safety_corridor_contracts.json

Pure data + serialization. One instance = one JSON object under "contracts".
CROWN execution stays on AcasSafetyContractVerifier.
"""

from __future__ import annotations

import json
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AcasContract(ABC):
    """
    Shared network / input-box identity for one local NN obligation.

    Active network is a_prev; NN inputs lie in
    [nn_input_lower, nn_input_upper] (points use equal bounds).
    """

    contract_id: int
    a_prev: str
    network_idx: int
    onnx: str
    heading_own_var: int
    x_sign: int
    y_sign: int
    nn_input_lower: list[float]
    nn_input_upper: list[float]
    description: str = ""

    def crown_input_bounds(self) -> tuple[list[float], list[float]]:
        return list(self.nn_input_lower), list(self.nn_input_upper)

    def _base_dict(self) -> dict[str, Any]:
        return {
            "id": self.contract_id,
            "a_prev": self.a_prev,
            "network_idx": self.network_idx,
            "onnx": self.onnx,
            "heading_own_var": self.heading_own_var,
            "x_sign": self.x_sign,
            "y_sign": self.y_sign,
            "nn_input_lower": list(self.nn_input_lower),
            "nn_input_upper": list(self.nn_input_upper),
            "description": self.description,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize this single contract to a JSON-compatible dict."""
        raise NotImplementedError


@dataclass
class AcasSafetyContract(AcasContract):
    """
    Safety A/G: in this region, NN never selects forbidden_advisory.

    Same schema for full-table and corridor specs (corridor = fewer rows).
    """

    forbidden_advisory: str = ""
    forbidden_advisory_idx: int = 0
    dangerous_xy: list[tuple[int, int]] = field(default_factory=list)
    n_states_covered: int = 0
    role: str | None = None
    empirical_forward_pass: dict[str, Any] | None = None
    contract_type: str = "range"

    def to_dict(self) -> dict[str, Any]:
        record = self._base_dict()
        record.update({
            "type": self.contract_type,
            "forbidden_advisory": self.forbidden_advisory,
            "forbidden_advisory_idx": self.forbidden_advisory_idx,
            "n_states_covered": self.n_states_covered or len(self.dangerous_xy),
            "dangerous_xy": [list(xy) for xy in self.dangerous_xy],
        })
        if self.role is not None:
            record["role"] = self.role
        if self.empirical_forward_pass is not None:
            record["empirical_forward_pass"] = self.empirical_forward_pass
        return record

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcasSafetyContract:
        """Build one safety contract from one JSON object."""
        dangerous = [
            (int(xy[0]), int(xy[1])) for xy in data.get("dangerous_xy", [])
        ]
        return cls(
            contract_id=int(data["id"]),
            a_prev=str(data["a_prev"]),
            network_idx=int(data["network_idx"]),
            onnx=str(data["onnx"]),
            heading_own_var=int(data["heading_own_var"]),
            x_sign=int(data["x_sign"]),
            y_sign=int(data["y_sign"]),
            nn_input_lower=[float(v) for v in data["nn_input_lower"]],
            nn_input_upper=[float(v) for v in data["nn_input_upper"]],
            description=str(data.get("description", "")),
            forbidden_advisory=str(data["forbidden_advisory"]),
            forbidden_advisory_idx=int(data["forbidden_advisory_idx"]),
            dangerous_xy=dangerous,
            n_states_covered=int(data.get("n_states_covered", len(dangerous))),
            role=(str(data["role"]) if data.get("role") is not None else None),
            empirical_forward_pass=data.get("empirical_forward_pass"),
            contract_type=str(data.get("type", "range")),
        )

    @classmethod
    def load_json(cls, path: Path | str) -> list[AcasSafetyContract]:
        """Load a safety-only specs file (full or corridor). Returns a list."""
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_list = payload["contracts"] if isinstance(payload, dict) else payload
        return [cls.from_dict(item) for item in raw_list]

    @staticmethod
    def dump_json(
        contracts: list[AcasSafetyContract],
        path: Path | str,
        *,
        description: str = "",
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        """Write a safety specs file from a list of contracts."""
        payload: dict[str, Any] = {
            "description": description,
            "contracts": [contract.to_dict() for contract in contracts],
        }
        if extra_meta:
            payload.update(extra_meta)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
