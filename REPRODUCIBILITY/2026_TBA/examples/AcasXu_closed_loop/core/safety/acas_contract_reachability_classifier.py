"""
acas_contract_reachability_classifier.py

Classify discrete-mode CROWN result contracts by whether they cover a
reachable dangerous state under nondeterministic AcasReachableSet.

Standalone analysis tool — no pipeline reads the report. Regenerates
results/discrete/safety/contract_reachability_report.json for research sizing
(projected INVAR ceiling after pruning unreachable cells).

Usage (from AcasXu_closed_loop/):

    python3 acas_contract_reachability_classifier.py
    python3 acas_contract_reachability_classifier.py \\
        --contracts-dir contracts/discrete/safety/archive \\
        --report-path results/discrete/safety/contract_reachability_report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.acas_domain import AcasDomain
from core.acas_reachability import AcasReachableSet
from core.paths import EXAMPLE_ROOT

_DEFAULT_CONTRACTS_DIR = (
    EXAMPLE_ROOT / "contracts/discrete/safety/archive"
)
_DEFAULT_REPORT_PATH = (
    EXAMPLE_ROOT / "results/discrete/safety/contract_reachability_report.json"
)


@dataclass
class AcasContractReachabilityClassifier:
    """
    Post-process discrete CROWN results: reachable vs unreachable dangerous cells.

    Construct with an explicit domain and paths (CLI supplies defaults once).
    """

    domain: AcasDomain
    contracts_dir: Path
    report_path: Path

    def network_idx_to_advisory(self) -> dict[int, str]:
        return {
            network_idx: name
            for name, (network_idx, _onnx) in self.domain.a_prev_to_nn.items()
        }

    def classify(
        self,
        reachable_set: AcasReachableSet | None = None,
    ) -> dict[str, Any]:
        """
        Build the reachability classification report.

        If ``reachable_set`` is omitted, compute it from ``self.domain``.
        """
        active_set = (
            reachable_set
            if reachable_set is not None
            else AcasReachableSet.compute(self.domain)
        )
        idx_to_advisory = self.network_idx_to_advisory()

        per_network: dict[str, Any] = {}
        reachable_dangerous_state_contracts: list[dict[str, Any]] = []
        projected_invar_ceiling = 0

        result_files = sorted(
            self.contracts_dir.glob("aprev_*_crown_results.json")
        )
        if not result_files:
            raise FileNotFoundError(
                f"no aprev_*_crown_results.json under {self.contracts_dir}"
            )

        for result_file in result_files:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            network_idx = int(data["network_idx"])
            if network_idx not in idx_to_advisory:
                raise KeyError(
                    f"{result_file}: unknown network_idx={network_idx}"
                )
            advisory = idx_to_advisory[network_idx]

            per_network[advisory] = {}
            for status in ("SAT", "UNSAT"):
                contracts = [
                    contract
                    for contract in data["contracts"]
                    if contract["status"] == status
                ]
                reachable_states_by_id = {
                    contract["id"]: active_set.dangerous_xy(contract, advisory)
                    for contract in contracts
                }

                reachable_dangerous_state_contracts.extend(
                    {
                        "network": advisory,
                        "contract_id": contract_id,
                        "status": status,
                    }
                    for contract_id, states in reachable_states_by_id.items()
                    if states
                )
                if status == "SAT":
                    projected_invar_ceiling += sum(
                        len(states)
                        for states in reachable_states_by_id.values()
                    )

                per_network[advisory][status] = {
                    "total": len(contracts),
                    "unreachable_explained": sum(
                        not states
                        for states in reachable_states_by_id.values()
                    ),
                    "reachable_dangerous_state": sum(
                        bool(states)
                        for states in reachable_states_by_id.values()
                    ),
                }

        return {
            "per_network": per_network,
            "reachable_dangerous_state_contracts": (
                reachable_dangerous_state_contracts
            ),
            "projected_invar_ceiling": projected_invar_ceiling,
        }

    def write_report(
        self,
        report: dict[str, Any],
        path: Path | None = None,
    ) -> Path:
        out = path if path is not None else self.report_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return out

    def print_summary(self, report: dict[str, Any]) -> None:
        print(json.dumps(report["per_network"], indent=2))
        print(
            f"\nProjected INVAR ceiling after pruning (was 8,982): "
            f"{report['projected_invar_ceiling']}"
        )

    def run(self) -> dict[str, Any]:
        """Classify, write report, print summary."""
        report = self.classify()
        out = self.write_report(report)
        self.print_summary(report)
        print(f"Report written to {out}")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=_DEFAULT_CONTRACTS_DIR,
        help="Directory of aprev_*_crown_results.json files",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=_DEFAULT_REPORT_PATH,
        help="Output JSON path for the classification report",
    )
    args = parser.parse_args()

    classifier = AcasContractReachabilityClassifier(
        domain=AcasDomain.from_yaml(),
        contracts_dir=args.contracts_dir.resolve(),
        report_path=args.report_path.resolve(),
    )
    classifier.run()


if __name__ == "__main__":
    main()
