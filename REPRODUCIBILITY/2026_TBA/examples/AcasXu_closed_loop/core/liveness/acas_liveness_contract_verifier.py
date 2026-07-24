"""
acas_liveness_contract_verifier.py

Liveness-side ACAS façade: discharge AcasLivenessContract via alpha-beta-CROWN
always_selects (equals / required advisory).

Uses CrownVerifier under the hood. Safety never_selects lives on
AcasSafetyContractVerifier.

Configuration: acas_liveness_params.yaml via AcasLivenessContractConfig.

Usage (from AcasXu_closed_loop/):

    python3 acas_liveness_contract_verifier.py
    python3 acas_liveness_contract_verifier.py --limit 3
    python3 acas_liveness_contract_verifier.py --timeout 30 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.paths import EXAMPLE_ROOT

_TBA = (EXAMPLE_ROOT / "../..").resolve()
if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))

from pipeline.crown_verifier import CrownVerifier  # noqa: E402

from core.acas_contract import AcasLivenessContract  # noqa: E402
from core.liveness.acas_liveness_contract_config import AcasLivenessContractConfig  # noqa: E402


@dataclass
class AcasLivenessContractVerifier:
    """
    Batch / single-contract liveness verification for AcasLivenessContract.

    Construct with from_yaml / from_config. Holds config + CrownVerifier.
    """

    config: AcasLivenessContractConfig
    crown_verifier: CrownVerifier
    onnx_margins: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
        *,
        timeout_sec: float | None = None,
        pgd_order: str | None = None,
        device: str | None = None,
        onnx_margins: dict[int, float] | None = None,
    ) -> AcasLivenessContractVerifier:
        config = AcasLivenessContractConfig.from_yaml(path)
        return cls.from_config(
            config,
            timeout_sec=timeout_sec,
            pgd_order=pgd_order,
            device=device,
            onnx_margins=onnx_margins,
        )

    @classmethod
    def from_config(
        cls,
        config: AcasLivenessContractConfig,
        *,
        timeout_sec: float | None = None,
        pgd_order: str | None = None,
        device: str | None = None,
        onnx_margins: dict[int, float] | None = None,
    ) -> AcasLivenessContractVerifier:
        """Orchestrator path: share one config; flags override CROWN knobs only."""
        active_timeout = config.timeout_sec if timeout_sec is None else float(timeout_sec)
        active_pgd = config.pgd_order if pgd_order is None else pgd_order
        active_device = config.device if device is None else device
        crown_verifier = CrownVerifier.from_timeout_and_attack_settings(
            timeout_seconds=active_timeout,
            pgd_order=active_pgd,
            device=active_device,
        )
        return cls(
            config=config,
            crown_verifier=crown_verifier,
            onnx_margins=dict(onnx_margins or {}),
        )

    @property
    def domain(self):
        return self.config.domain

    def _resolve_onnx_path(self, onnx: str) -> str:
        path = Path(onnx)
        if path.is_file():
            return str(path)
        root = self.config.root
        for _a_prev, (_idx, onnx_rel) in self.domain.a_prev_to_nn.items():
            if Path(onnx_rel).name == path.name or onnx_rel.endswith(onnx):
                return str(root / onnx_rel)
        candidate = root / onnx
        if candidate.is_file():
            return str(candidate)
        return str(root / "networks" / path.name)

    def load_liveness_contracts(
        self,
        path: Path | None = None,
        *,
        limit: int | None = None,
    ) -> list[AcasLivenessContract]:
        specs_path = path if path is not None else self.config.specs_path
        contracts = AcasLivenessContract.load_json(specs_path)
        if limit is not None:
            contracts = contracts[:limit]
        return contracts

    def verify_contract(self, contract: AcasLivenessContract) -> str:
        """Always-selects CROWN; returns SAT / UNSAT / TIMEOUT."""
        onnx_path = self._resolve_onnx_path(contract.onnx)
        lower, upper = contract.crown_input_bounds()
        status, _ = self.crown_verifier.certify_network_always_selects_class(
            onnx_path=onnx_path,
            input_lower_bounds=lower,
            input_upper_bounds=upper,
            required_class_index=contract.required_advisory_idx,
            number_of_classes=len(self.domain.advisories),
        )
        return status

    def verify_all(
        self,
        contracts: list[AcasLivenessContract],
        *,
        verbose: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Verify every liveness contract; return (records, status counts)."""
        records: list[dict[str, Any]] = []
        counts = {"SAT": 0, "UNSAT": 0, "TIMEOUT": 0}
        for contract in contracts:
            start = time.perf_counter()
            status = self.verify_contract(contract)
            wall = time.perf_counter() - start
            counts[status] = counts.get(status, 0) + 1
            margin = self.onnx_margins.get(contract.contract_id)
            if verbose:
                margin_s = f"{margin:+.4f}" if margin is not None else "  n/a"
                print(
                    f"  contract {contract.contract_id:>2}/{len(contracts)}  "
                    f"{contract.a_prev:>12} -> {contract.required_advisory:<12}  "
                    f"margin={margin_s}  {status:<8}  {wall:.2f}s"
                )
            records.append({
                "id": contract.contract_id,
                "status": status,
                "onnx_margin": margin,
                "required_advisory": contract.required_advisory,
                "a_prev": contract.a_prev,
                "wall_sec": round(wall, 3),
            })
        return records, counts

    def write_results(
        self,
        records: list[dict[str, Any]],
        path: Path | None = None,
        *,
        description: str = "CROWN always_selects results for liveness contracts",
    ) -> Path:
        out = path if path is not None else self.config.results_path
        summary = {
            status: sum(1 for record in records if record["status"] == status)
            for status in ("SAT", "UNSAT", "TIMEOUT")
        }
        payload = {
            "description": description,
            "summary": summary,
            "contracts": [
                {
                    "id": record["id"],
                    "status": record["status"],
                    "onnx_margin": record.get("onnx_margin"),
                    "required_advisory": record["required_advisory"],
                    "a_prev": record["a_prev"],
                }
                for record in records
            ],
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote liveness results: {out}")
        return out

    def write_sat_stubs(
        self,
        contracts: list[AcasLivenessContract],
        path: Path | None = None,
        *,
        onnx_margins: dict[int, float] | None = None,
    ) -> Path:
        """Mark contracts SAT without CROWN (ONNX-margin-only pipeline runs)."""
        margins = onnx_margins if onnx_margins is not None else self.onnx_margins
        records = [
            {
                "id": contract.contract_id,
                "status": "SAT",
                "onnx_margin": margins.get(contract.contract_id),
                "required_advisory": contract.required_advisory,
                "a_prev": contract.a_prev,
                "wall_sec": 0.0,
            }
            for contract in contracts
        ]
        return self.write_results(
            records,
            path,
            description=(
                "Liveness results: CROWN skipped; statuses SAT from ONNX consistency"
            ),
        )

    def run(
        self,
        *,
        contracts: list[AcasLivenessContract] | None = None,
        specs_path: Path | None = None,
        results_path: Path | None = None,
        limit: int | None = None,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        if contracts is None:
            contracts = self.load_liveness_contracts(specs_path, limit=limit)
        records, counts = self.verify_all(contracts, verbose=verbose)
        print(f"CROWN summary: {counts}")
        self.write_results(records, results_path)
        return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to acas_liveness_params.yaml",
    )
    parser.add_argument(
        "--specs",
        type=Path,
        default=None,
        help="Override specs_path from config for this run",
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=None,
        help="Override results_path from config for this run",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override verification.timeout_sec for this run",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override verification.device for this run",
    )
    parser.add_argument(
        "--pgd-order",
        default=None,
        dest="pgd_order",
        help="Override verification.pgd_order for this run",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    verifier = AcasLivenessContractVerifier.from_yaml(
        args.config,
        timeout_sec=args.timeout,
        device=args.device,
        pgd_order=args.pgd_order,
    )
    contracts = verifier.load_liveness_contracts(args.specs, limit=args.limit)
    print(f"Loaded {len(contracts)} liveness contracts")
    verifier.run(contracts=contracts, results_path=args.results_out)


if __name__ == "__main__":
    main()
