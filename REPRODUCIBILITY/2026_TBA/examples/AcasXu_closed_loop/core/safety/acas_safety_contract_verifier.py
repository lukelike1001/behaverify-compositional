"""
acas_safety_contract_verifier.py

Safety-side ACAS façade: discharge AcasSafetyContract via alpha-beta-CROWN.

Uses CrownVerifier under the hood. Liveness (equals pins) is a separate track.

  - continuous: one never-selects call on the range box
  - discrete: one never-selects call per dangerous lattice point (short-circuit UNSAT)

Configuration: acas_verifier_params.yaml

Usage (from AcasXu_closed_loop/):

    python3 acas_safety_contract_verifier.py
    python3 acas_safety_contract_verifier.py --limit 5
    python3 acas_safety_contract_verifier.py --discrete --network-idx 1
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.paths import EXAMPLE_ROOT

_TBA = (EXAMPLE_ROOT / "../..").resolve()
if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))

from pipeline.neuro.crown.crown_verifier import CrownVerifier  # noqa: E402

from core.acas_contract import AcasSafetyContract  # noqa: E402
from core.acas_domain import AcasDomain  # noqa: E402

DEFAULT_VERIFIER_PARAMS = EXAMPLE_ROOT / "core" / "acas_verifier_params.yaml"


@dataclass
class AcasSafetyContractVerifier:
    """
    Batch / single-contract safety verification for AcasSafetyContract.

    Construct with from_yaml / from_config. Holds domain + CrownVerifier.
    """

    domain: AcasDomain
    crown_verifier: CrownVerifier
    contracts_path: Path
    output_path: Path
    network_idx: int
    num_classes: int
    timeout_sec: float
    discrete: bool = False
    discrete_timeout_sec: float = 5.0
    discrete_state_eps: float = 0.0
    pgd_order: str = "skip"
    device: str = "cpu"
    raw_config: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def load_yaml(path: Path | str | None = None) -> dict[str, Any]:
        config_path = Path(path) if path is not None else DEFAULT_VERIFIER_PARAMS
        with open(config_path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
        *,
        domain: AcasDomain | None = None,
    ) -> AcasSafetyContractVerifier:
        return cls.from_config(cls.load_yaml(path), domain=domain)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        domain: AcasDomain | None = None,
    ) -> AcasSafetyContractVerifier:
        plant = domain if domain is not None else AcasDomain.from_yaml()
        verification = config.get("verification", {})
        discrete = bool(config.get("discrete", False))
        timeout_sec = float(verification.get("timeout_sec", 30.0))
        discrete_timeout_sec = float(
            config.get(
                "discrete_timeout",
                verification.get("discrete_timeout_sec", 5.0),
            )
        )
        active_timeout = discrete_timeout_sec if discrete else timeout_sec
        crown_verifier = CrownVerifier.from_timeout_and_attack_settings(
            timeout_seconds=active_timeout,
            pgd_order=str(config.get("pgd_order", "skip")),
            device=str(config.get("device", "cpu")),
        )
        return cls(
            domain=plant,
            crown_verifier=crown_verifier,
            contracts_path=Path(config["contracts_path"]),
            output_path=Path(config["output_path"]),
            network_idx=int(config["network_idx"]),
            num_classes=int(config["num_classes"]),
            timeout_sec=timeout_sec,
            discrete=discrete,
            discrete_timeout_sec=discrete_timeout_sec,
            discrete_state_eps=float(verification.get("discrete_state_eps", 0.0)),
            pgd_order=str(config.get("pgd_order", "skip")),
            device=str(config.get("device", "cpu")),
            raw_config=dict(config),
        )

    def rebuild_crown_verifier(self, timeout_seconds: float | None = None) -> None:
        """Recreate CrownVerifier (e.g. after switching discrete / timeout)."""
        if timeout_seconds is None:
            timeout_seconds = (
                self.discrete_timeout_sec if self.discrete else self.timeout_sec
            )
        self.crown_verifier = CrownVerifier.from_timeout_and_attack_settings(
            timeout_seconds=timeout_seconds,
            pgd_order=self.pgd_order,
            device=self.device,
        )

    def _resolve_onnx_path(self, onnx: str) -> str:
        path = Path(onnx)
        if path.is_file():
            return str(path)
        candidate = EXAMPLE_ROOT / onnx
        if candidate.is_file():
            return str(candidate)
        return str(path)

    # ------------------------------------------------------------------
    # Contract loading
    # ------------------------------------------------------------------

    def load_safety_contracts(
        self,
        *,
        limit: int | None = None,
        retry_from: Path | str | None = None,
    ) -> tuple[list[AcasSafetyContract], dict[int, dict[str, Any]]]:
        """
        Load AcasSafetyContract specs for self.network_idx.

        Returns (contracts_to_verify, previous_records_by_id).
        """
        contracts = [
            contract
            for contract in AcasSafetyContract.load_json(self.contracts_path)
            if contract.network_idx == self.network_idx
        ]
        if not contracts:
            raise SystemExit(
                f"No contracts found for network_idx={self.network_idx}. "
                f"Check contracts_path={self.contracts_path}."
            )

        previous_records: dict[int, dict[str, Any]] = {}
        if retry_from is not None:
            with open(retry_from, encoding="utf-8") as handle:
                previous = json.load(handle)
            previous_records = {
                record["id"]: record for record in previous["contracts"]
            }
            timeout_ids = {
                record["id"]
                for record in previous["contracts"]
                if record["status"] == "TIMEOUT"
            }
            contracts = [
                contract for contract in contracts
                if contract.contract_id in timeout_ids
            ]
            print(
                f"Retry mode: {len(contracts)} TIMEOUT contracts from {retry_from}"
            )

        if limit is not None:
            contracts = contracts[:limit]

        return contracts, previous_records

    # ------------------------------------------------------------------
    # Single-contract verification
    # ------------------------------------------------------------------

    def verify_contract(self, contract: AcasSafetyContract) -> str:
        """Verify one safety A/G contract; returns SAT / UNSAT / TIMEOUT."""
        if self.discrete:
            return self._verify_discrete(contract)
        return self._verify_continuous(contract)

    def _verify_continuous(self, contract: AcasSafetyContract) -> str:
        onnx_path = self._resolve_onnx_path(contract.onnx)
        lower, upper = contract.crown_input_bounds()
        status, _ = self.crown_verifier.certify_network_never_selects_class(
            onnx_path=onnx_path,
            input_lower_bounds=lower,
            input_upper_bounds=upper,
            forbidden_class_index=contract.forbidden_advisory_idx,
            number_of_classes=self.num_classes,
        )
        return status

    def _verify_discrete(self, contract: AcasSafetyContract) -> str:
        """
        One CROWN call per dangerous lattice point; short-circuit on UNSAT.
        """
        onnx_path = self._resolve_onnx_path(contract.onnx)
        timeout_seen = False
        eps = self.discrete_state_eps

        for x_mag, y_mag in contract.dangerous_xy:
            exact = self.domain.compute_nn_inputs(
                x_mag, y_mag, contract.x_sign, contract.y_sign,
                contract.heading_own_var,
            )
            if eps == 0.0:
                lower = exact
                upper = exact
            else:
                lower = [value - eps for value in exact]
                upper = [value + eps for value in exact]
            status, _ = self.crown_verifier.certify_network_never_selects_class(
                onnx_path=onnx_path,
                input_lower_bounds=lower,
                input_upper_bounds=upper,
                forbidden_class_index=contract.forbidden_advisory_idx,
                number_of_classes=self.num_classes,
            )
            if status == "UNSAT":
                return "UNSAT"
            if status == "TIMEOUT":
                timeout_seen = True

        return "TIMEOUT" if timeout_seen else "SAT"

    def verify_point_state(
        self,
        *,
        x_mag: int,
        y_mag: int,
        x_sign: int,
        y_sign: int,
        heading_own_var: int,
        forbidden_advisory: str,
        a_prev: str,
        onnx_path: str | None = None,
    ) -> str:
        """Ad-hoc single lattice point via a one-cell AcasSafetyContract."""
        if forbidden_advisory not in self.domain.adv_idx:
            raise ValueError(f"unknown forbidden_advisory={forbidden_advisory!r}")
        if a_prev not in self.domain.a_prev_to_nn:
            raise ValueError(f"unknown a_prev={a_prev!r}")

        network_idx, onnx_rel = self.domain.a_prev_to_nn[a_prev]
        if onnx_path is None:
            onnx_path = str(EXAMPLE_ROOT / onnx_rel)

        exact = self.domain.compute_nn_inputs(
            x_mag, y_mag, x_sign, y_sign, heading_own_var,
        )
        synthetic = AcasSafetyContract(
            contract_id=0,
            a_prev=a_prev,
            network_idx=network_idx,
            onnx=onnx_path,
            heading_own_var=heading_own_var,
            x_sign=x_sign,
            y_sign=y_sign,
            nn_input_lower=list(exact),
            nn_input_upper=list(exact),
            forbidden_advisory=forbidden_advisory,
            forbidden_advisory_idx=self.domain.adv_idx[forbidden_advisory],
            dangerous_xy=[(x_mag, y_mag)],
            n_states_covered=1,
            contract_type="range",
        )
        was_discrete = self.discrete
        self.discrete = True
        self.rebuild_crown_verifier(self.discrete_timeout_sec)
        try:
            return self._verify_discrete(synthetic)
        finally:
            self.discrete = was_discrete
            self.rebuild_crown_verifier()

    # ------------------------------------------------------------------
    # Batch run + report
    # ------------------------------------------------------------------

    @staticmethod
    def result_marker(status: str) -> str:
        if status == "SAT":
            return "✓"
        if status == "UNSAT":
            return "✗  <- VIOLATION"
        return "?  <- TIMEOUT (inconclusive)"

    @staticmethod
    def print_summary(records: list[dict[str, Any]]) -> None:
        counts = {
            status: sum(1 for record in records if record["status"] == status)
            for status in ("SAT", "UNSAT", "TIMEOUT")
        }
        print(
            f"\nSummary: {counts['SAT']} SAT, {counts['UNSAT']} UNSAT, "
            f"{counts['TIMEOUT']} TIMEOUT out of {len(records)} contracts"
        )

    def mode_description(self) -> str:
        if self.discrete:
            return (
                f"discrete, EPS={self.discrete_state_eps}, "
                f"timeout={self.discrete_timeout_sec}s per state"
            )
        return "continuous"

    def verify_all(
        self,
        contracts: list[AcasSafetyContract],
        *,
        previous_records: dict[int, dict[str, Any]] | None = None,
        retry_from: Path | str | None = None,
    ) -> tuple[list[dict[str, Any]], float, str]:
        """
        Verify every safety contract; return (records, total_wall_sec, mode_str).

        Result records stay plain dicts for JSON reports.
        """
        onnx_path = contracts[0].onnx
        mode_str = self.mode_description()
        print(
            f"Verifying {len(contracts)} contracts for NN_{self.network_idx} "
            f"({onnx_path})"
        )
        if self.discrete:
            print(f"Mode: {mode_str}\n")
        else:
            print(f"Timeout: {self.timeout_sec}s per contract\n")

        print(
            f"{'#':<5} {'Heading':>7} {'Quad':>6} {'Forbidden':<14} "
            f"{'States':>6} {'Sec':>6} {'Status':<10} Marker"
        )
        print("-" * 80)

        new_records: list[dict[str, Any]] = []
        run_start = time.perf_counter()

        for index, contract in enumerate(contracts):
            t0 = time.perf_counter()
            status = self.verify_contract(contract)
            wall_sec = time.perf_counter() - t0

            def sign_label(value: int) -> str:
                return "+" if value == 1 else "-"

            quad = (
                f"({sign_label(contract.x_sign)},"
                f"{sign_label(contract.y_sign)})"
            )
            print(
                f"{index + 1:<5} {contract.heading_own_var:>7} {quad:>6} "
                f"{contract.forbidden_advisory:<14} "
                f"{contract.n_states_covered:>6} "
                f"{wall_sec:>6.1f} {status:<10} {self.result_marker(status)}"
            )
            sys.stdout.flush()

            new_records.append({
                "id": contract.contract_id,
                "heading_own_var": contract.heading_own_var,
                "x_sign": contract.x_sign,
                "y_sign": contract.y_sign,
                "forbidden_advisory": contract.forbidden_advisory,
                "forbidden_advisory_idx": contract.forbidden_advisory_idx,
                "n_states_covered": contract.n_states_covered,
                "dangerous_xy": [list(xy) for xy in contract.dangerous_xy],
                "wall_sec": round(wall_sec, 3),
                "status": status,
            })

        total_wall = time.perf_counter() - run_start

        if previous_records is not None and retry_from is not None:
            merged = dict(previous_records)
            merged.update({record["id"]: record for record in new_records})
            records = sorted(merged.values(), key=lambda record: record["id"])
            improved = sum(
                1 for record in new_records if record["status"] == "SAT"
            )
            print(
                f"\nRetry improved {improved}/{len(new_records)} contracts to SAT"
            )
        else:
            records = new_records

        print("-" * 80)
        self.print_summary(records)
        print(f"Total wall time: {total_wall:.1f}s  ({total_wall / 60:.1f} min)")
        return records, total_wall, mode_str

    def build_report(
        self,
        records: list[dict[str, Any]],
        onnx_path: str,
        total_wall_sec: float,
        mode: str,
    ) -> dict[str, Any]:
        counts = {
            status: sum(1 for record in records if record["status"] == status)
            for status in ("SAT", "UNSAT", "TIMEOUT")
        }
        sat_times = [
            record["wall_sec"] for record in records if record["status"] == "SAT"
        ]
        return {
            "network_idx": self.network_idx,
            "onnx_path": onnx_path,
            "mode": mode,
            "timestamp": datetime.datetime.now().isoformat(),
            "timeout_sec": self.timeout_sec,
            "total_wall_sec": round(total_wall_sec, 3),
            "avg_sat_sec": (
                round(sum(sat_times) / len(sat_times), 3) if sat_times else None
            ),
            "summary": {**counts, "total": len(records)},
            "contracts": records,
        }

    def write_report(self, report: dict[str, Any]) -> None:
        parent = os.path.dirname(self.output_path) or "."
        os.makedirs(parent, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Results saved to {self.output_path}")

    def run(
        self,
        *,
        limit: int | None = None,
        retry_from: Path | str | None = None,
        timeout_override: float | None = None,
    ) -> list[dict[str, Any]]:
        """Load safety contracts, verify all, write report."""
        if timeout_override is not None:
            self.timeout_sec = float(timeout_override)
            if not self.discrete:
                self.rebuild_crown_verifier(self.timeout_sec)
        if self.discrete:
            self.rebuild_crown_verifier(self.discrete_timeout_sec)

        contracts, previous_records = self.load_safety_contracts(
            limit=limit,
            retry_from=retry_from,
        )
        records, total_wall, mode_str = self.verify_all(
            contracts,
            previous_records=previous_records or None,
            retry_from=retry_from,
        )
        report = self.build_report(
            records, contracts[0].onnx, total_wall, mode_str,
        )
        self.write_report(report)
        return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify ACAS Xu safety A/G contracts via alpha-beta-CROWN.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_VERIFIER_PARAMS),
        help="Path to acas_verifier_params.yaml",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Verify only the first N contracts (pilot runs)",
    )
    parser.add_argument(
        "--retry-from", default=None, dest="retry_from",
        help="Previous results JSON; re-verify only TIMEOUT contracts",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Override timeout_sec from YAML for continuous mode",
    )
    parser.add_argument(
        "--network-idx", type=int, default=None, dest="network_idx",
        help="Override network_idx (1=clear … 5=strong_left)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override output_path from YAML",
    )
    parser.add_argument(
        "--discrete", action="store_true",
        help="Discrete mode: one CROWN call per dangerous lattice point",
    )
    parser.add_argument(
        "--discrete-timeout", type=float, default=None, dest="discrete_timeout",
        help="Per-state timeout for discrete mode",
    )
    parser.add_argument(
        "--point", action="store_true",
        help="Verify a single lattice point instead of the contracts JSON batch",
    )
    parser.add_argument("--x_mag", type=int, default=None)
    parser.add_argument("--y_mag", type=int, default=None)
    parser.add_argument("--x_sign", type=int, default=None, choices=[-1, 1])
    parser.add_argument("--y_sign", type=int, default=None, choices=[-1, 1])
    parser.add_argument("--heading", type=int, default=None, dest="heading")
    parser.add_argument(
        "--advisory", default=None,
        help="a_prev / network for --point mode",
    )
    parser.add_argument(
        "--forbidden", default=None,
        help="Forbidden advisory for --point mode",
    )
    args = parser.parse_args()

    config = AcasSafetyContractVerifier.load_yaml(args.config)
    if args.network_idx is not None:
        config["network_idx"] = args.network_idx
    if args.output is not None:
        config["output_path"] = args.output
    if args.discrete or args.point:
        config["discrete"] = True
        if args.discrete_timeout is not None:
            config["discrete_timeout"] = args.discrete_timeout

    verifier = AcasSafetyContractVerifier.from_config(config)

    if args.point:
        required = [
            args.x_mag, args.y_mag, args.x_sign, args.y_sign,
            args.heading, args.advisory, args.forbidden,
        ]
        if any(value is None for value in required):
            parser.error(
                "--point requires --x_mag --y_mag --x_sign --y_sign "
                "--heading --advisory --forbidden"
            )
        status = verifier.verify_point_state(
            x_mag=args.x_mag,
            y_mag=args.y_mag,
            x_sign=args.x_sign,
            y_sign=args.y_sign,
            heading_own_var=args.heading,
            forbidden_advisory=args.forbidden,
            a_prev=args.advisory,
        )
        print(
            f"point state=({args.x_mag},{args.y_mag},sx={args.x_sign},"
            f"sy={args.y_sign},h={args.heading}) a_prev={args.advisory} "
            f"forbidden={args.forbidden} status={status}"
        )
        return

    verifier.run(
        limit=args.limit,
        retry_from=args.retry_from,
        timeout_override=args.timeout,
    )


if __name__ == "__main__":
    main()
