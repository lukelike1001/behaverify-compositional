"""
acas_liveness_contract_config.py

Shared liveness configuration from acas_liveness_params.yaml.
Plant physics stays on AcasDomain (acas_model_params.yaml).

All keys in acas_liveness_params.yaml are required. Missing keys raise
ValueError with the YAML path and key name.

    from core.liveness.acas_liveness_contract_config import AcasLivenessContractConfig

    config = AcasLivenessContractConfig.from_yaml()
    trajectory = config.load_trajectory()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from core.acas_domain import AcasDomain
from core.liveness.acas_lasso_trajectory import AcasLassoTrajectory
from core.paths import EXAMPLE_ROOT

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_YAML_PATH = _MODULE_DIR / "acas_liveness_params.yaml"

# Top-level keys that must appear in acas_liveness_params.yaml
_REQUIRED_TOP_LEVEL = (
    "trajectory_path",
    "specs_path",
    "results_path",
    "nn_input_eps",
    "expected_stem_length",
    "expected_total_states",
    "verification",
)

# Keys that must appear under verification:
_REQUIRED_VERIFICATION = (
    "timeout_sec",
    "pgd_order",
    "device",
)


@dataclass(frozen=True)
class AcasLivenessContractConfig:
    """
    Liveness product-line settings: artifact paths, nn_input_eps, CROWN, plant.

    Construct only via ``from_yaml()``. Pass the same instance into generator,
    verifier, and (later) the liveness pipeline.
    """

    DEFAULT_YAML_PATH: ClassVar[Path] = _DEFAULT_YAML_PATH

    root: Path
    trajectory_path: Path
    specs_path: Path
    results_path: Path
    nn_input_eps: float
    expected_stem_length: int
    expected_total_states: int
    timeout_sec: float
    pgd_order: str
    device: str
    domain: AcasDomain

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
        *,
        domain: AcasDomain | None = None,
    ) -> AcasLivenessContractConfig:
        yaml_path = Path(path) if path is not None else cls.DEFAULT_YAML_PATH
        with open(yaml_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if not isinstance(raw, dict):
            raise ValueError(
                f"{yaml_path}: expected a mapping at the top level, "
                f"got {type(raw).__name__}"
            )

        cls._require_keys(raw, _REQUIRED_TOP_LEVEL, yaml_path)

        verification = raw["verification"]
        if not isinstance(verification, dict):
            raise ValueError(
                f"{yaml_path}: 'verification' must be a mapping, "
                f"got {type(verification).__name__}"
            )
        cls._require_keys(
            verification, _REQUIRED_VERIFICATION, yaml_path, prefix="verification.",
        )

        plant = domain if domain is not None else AcasDomain.from_yaml()

        return cls(
            root=EXAMPLE_ROOT,
            trajectory_path=cls._resolve_path(
                EXAMPLE_ROOT, raw["trajectory_path"],
            ),
            specs_path=cls._resolve_path(EXAMPLE_ROOT, raw["specs_path"]),
            results_path=cls._resolve_path(EXAMPLE_ROOT, raw["results_path"]),
            nn_input_eps=float(raw["nn_input_eps"]),
            expected_stem_length=int(raw["expected_stem_length"]),
            expected_total_states=int(raw["expected_total_states"]),
            timeout_sec=float(verification["timeout_sec"]),
            pgd_order=str(verification["pgd_order"]),
            device=str(verification["device"]),
            domain=plant,
        )

    @staticmethod
    def _require_keys(
        mapping: dict[str, Any],
        required: tuple[str, ...],
        yaml_path: Path,
        *,
        prefix: str = "",
    ) -> None:
        missing = [f"{prefix}{key}" for key in required if key not in mapping]
        if missing:
            raise ValueError(
                f"{yaml_path}: missing required key(s): {', '.join(missing)}"
            )

    @staticmethod
    def _resolve_path(root: Path, value: Path | str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (root / path).resolve()

    def load_trajectory(self) -> AcasLassoTrajectory:
        """
        Load the precomputed lasso dump using this config's path and size checks.

        cycle_start_index is expected_stem_length for the current weight dump.
        """
        return AcasLassoTrajectory.from_json_file(
            self.trajectory_path,
            cycle_start_index=self.expected_stem_length,
            expected_total=self.expected_total_states,
        )
