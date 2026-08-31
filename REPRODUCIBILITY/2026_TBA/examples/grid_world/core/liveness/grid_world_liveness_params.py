"""
grid_world_liveness_params.py

Shared liveness configuration from grid_world_liveness_params.yaml.
Plant physics stays on GridWorldDomain (grid_world_domain_config.yaml).

All keys are required; a missing one raises ValueError naming the YAML path
and the key, so a partial config fails at load rather than halfway through a
CROWN run.

    from core.liveness.grid_world_liveness_params import GridWorldLivenessParams

    params = GridWorldLivenessParams.from_yaml()
    params.contracts_path_for("1000__6_18_0__0200_1")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from core.paths import EXAMPLE_ROOT

DEFAULT_YAML_PATH = Path(__file__).resolve().parent / "grid_world_liveness_params.yaml"

_REQUIRED_KEYS = (
    "contracts_dir",
    "results_dir",
    "num_classes",
    "stay_class_index",
    "nn_input_eps",
    "expected_progress_pairs",
    "expected_crown_obligations",
    "verification",
)
_REQUIRED_VERIFICATION_KEYS = ("timeout_sec", "pgd_order", "device")


@dataclass(frozen=True)
class GridWorldLivenessParams:
    """Tunables for the liveness product line, loaded once from YAML."""

    contracts_dir: Path
    results_dir: Path
    num_classes: int
    stay_class_index: int
    nn_input_eps: float
    expected_progress_pairs: int
    expected_crown_obligations: int
    timeout_sec: float
    pgd_order: str
    device: str

    CONTRACTS_SUFFIX: ClassVar[str] = "_liveness.json"

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_YAML_PATH) -> GridWorldLivenessParams:
        raw: dict[str, Any] = yaml.safe_load(path.read_text())
        for key in _REQUIRED_KEYS:
            if key not in raw:
                raise ValueError(f"{path}: missing required key '{key}'")
        verification = raw["verification"]
        for key in _REQUIRED_VERIFICATION_KEYS:
            if key not in verification:
                raise ValueError(f"{path}: missing required key 'verification.{key}'")

        return cls(
            contracts_dir=EXAMPLE_ROOT / raw["contracts_dir"],
            results_dir=EXAMPLE_ROOT / raw["results_dir"],
            num_classes=int(raw["num_classes"]),
            stay_class_index=int(raw["stay_class_index"]),
            nn_input_eps=float(raw["nn_input_eps"]),
            expected_progress_pairs=int(raw["expected_progress_pairs"]),
            expected_crown_obligations=int(raw["expected_crown_obligations"]),
            timeout_sec=float(verification["timeout_sec"]),
            pgd_order=str(verification["pgd_order"]),
            device=str(verification["device"]),
        )

    def contracts_path_for(self, network_name: str) -> Path:
        """Where this network's CROWN results belong."""
        return self.contracts_dir / f"{network_name}{self.CONTRACTS_SUFFIX}"
