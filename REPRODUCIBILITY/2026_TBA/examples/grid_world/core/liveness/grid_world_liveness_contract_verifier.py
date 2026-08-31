"""
grid_world_liveness_contract_verifier.py

Discharge grid-world liveness (progress) contracts with alpha-beta-CROWN.

One contract asserts membership: at (source, goal), the NN must output some
action in Dec(source, goal). CROWN has no "argmax in set" query, so each
contract expands into one never-select call per action in the complement --
membership holds exactly when every one of those calls returns SAT.

    Assume    (x_d, y_d) == source  AND  (x_g, y_g) == goal
    Guarantee argmax NN != a   for every a not in Dec(source, goal)

Both halves of the assumption are pinned to lattice points (eps = 0 by
default), because the SMV guard matches exact cells. That makes these point
queries -- as strong as the monolithic table at integer inputs, and no
stronger.

Usage (from examples/grid_world/):

    python3 -m core.liveness.grid_world_liveness_contract_verifier \\
        --onnx networks/1000__6_18_0__0200_1.onnx \\
        --output contracts/discrete/liveness/1000__6_18_0__0200_1_liveness.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_TBA = _HERE.parents[4]
if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))

from pipeline.crown_verifier import CrownVerifier  # noqa: E402
from pipeline.process_memory import ProcessMemory  # noqa: E402

from core.grid_world_contract import GridWorldLivenessContract  # noqa: E402
from core.grid_world_domain import GridWorldDomain  # noqa: E402
from core.liveness.grid_world_liveness_contract_generator import (  # noqa: E402
    GridWorldLivenessContractGenerator,
)
from core.liveness.grid_world_liveness_params import (  # noqa: E402
    GridWorldLivenessParams,
)
from core.paths import EXAMPLE_ROOT  # noqa: E402

_PARAMS = GridWorldLivenessParams.from_yaml()

DEFAULT_TIMEOUT_SEC: float = _PARAMS.timeout_sec
DEFAULT_NUM_CLASSES: int = _PARAMS.num_classes
# Lattice points: the SMV guard is an exact cell, so no ball is needed.
DEFAULT_EPS: float = _PARAMS.nn_input_eps


def _status_marker(status: str) -> str:
    if status == "SAT":
        return "✓"
    if status == "UNSAT":
        return "✗  ← STALL / WRONG TURN"
    return "?  ← TIMEOUT (inconclusive)"


@dataclass
class GridWorldLivenessContractVerifier:
    """
    CROWN-based certifier for progress contracts.

    Owns neural verification only -- the ranking function lives on
    GridWorldGoalDistance, contract emission on
    GridWorldLivenessContractGenerator.
    """

    domain: GridWorldDomain
    onnx_path: str
    output_path: str
    num_classes: int = DEFAULT_NUM_CLASSES
    eps: float = DEFAULT_EPS
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    pgd_order: str = _PARAMS.pgd_order
    device: str = _PARAMS.device
    contracts: list[GridWorldLivenessContract] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        onnx_path: str,
        output_path: str,
        domain: GridWorldDomain | None = None,
        **overrides: Any,
    ) -> GridWorldLivenessContractVerifier:
        if domain is None:
            domain = GridWorldDomain.from_config()
        generator = GridWorldLivenessContractGenerator.from_domain(domain)
        return cls(
            domain=domain,
            onnx_path=onnx_path,
            output_path=output_path,
            contracts=generator.generate_all_contracts(),
            **overrides,
        )

    def _make_crown_verifier(self) -> CrownVerifier:
        return CrownVerifier.from_timeout_and_attack_settings(
            timeout_seconds=self.timeout_sec,
            pgd_order=self.pgd_order,
            device=self.device,
        )

    def mode_description(self) -> str:
        return (
            f"point queries, drone EPS={self.eps}, goal EPS={self.eps}, "
            f"timeout={self.timeout_sec}s per obligation"
        )

    def _input_bounds(
        self, contract: GridWorldLivenessContract,
    ) -> tuple[list[float], list[float]]:
        """The assumption box: source cell and goal cell, both pinned."""
        sx, sy = contract.source
        gx, gy = contract.goal_region
        lower = [sx - self.eps, sy - self.eps, gx - self.eps, gy - self.eps]
        upper = [sx + self.eps, sy + self.eps, gx + self.eps, gy + self.eps]
        return lower, upper

    def certify(
        self,
        contract: GridWorldLivenessContract,
        crown_verifier: CrownVerifier | None = None,
    ) -> tuple[str, str | None, int]:
        """
        Discharge one membership contract.

        Returns (status, failing_direction, calls_made). Short-circuits on the
        first UNSAT: one forbidden action the network might still select is
        enough to break the descent at this state.
        """
        if crown_verifier is None:
            crown_verifier = self._make_crown_verifier()
        lower, upper = self._input_bounds(contract)
        timeout_seen: str | None = None
        calls = 0

        for direction, class_index in zip(
            contract.forbidden_dirs, contract.forbidden_dir_idxs,
        ):
            status, _result = crown_verifier.certify_network_never_selects_class(
                onnx_path=self.onnx_path,
                input_lower_bounds=lower,
                input_upper_bounds=upper,
                forbidden_class_index=class_index,
                number_of_classes=self.num_classes,
            )
            calls += 1
            if status == "UNSAT":
                return "UNSAT", direction, calls
            if status == "TIMEOUT":
                timeout_seen = direction

        if timeout_seen is not None:
            return "TIMEOUT", timeout_seen, calls
        return "SAT", None, calls

    def certify_all(
        self, *, write_json: bool = True, verbose: bool = True,
    ) -> dict[str, Any]:
        """Certify every progress contract; optionally write CROWN JSON."""
        if not self.contracts:
            self.contracts = GridWorldLivenessContractGenerator.from_domain(
                self.domain,
            ).generate_all_contracts()

        if verbose:
            total_calls = sum(len(c.forbidden_dirs) for c in self.contracts)
            print(
                f"Generated {len(self.contracts)} liveness contracts "
                f"({total_calls} never-select obligations)"
            )
            print(f"  {self.mode_description()}\n")
            print(f"{'#':<5} {'Description':<52} {'Status':<10} {'Marker'}")
            print("-" * 88)

        crown_verifier = self._make_crown_verifier()
        tracemalloc.start()
        start = time.perf_counter()
        records: list[dict[str, Any]] = []
        calls_made = 0

        for index, contract in enumerate(self.contracts):
            status, failing, calls = self.certify(contract, crown_verifier)
            calls_made += calls
            if verbose and (status != "SAT" or (index + 1) % 50 == 0):
                print(
                    f"{index + 1:<5} {contract.description:<52} "
                    f"{status:<10} {_status_marker(status)}"
                )
                if failing is not None and status != "SAT":
                    print(f"        network may select forbidden {failing}")
                sys.stdout.flush()

            record = contract.to_spec_dict(contract_id=index + 1)
            record["status"] = status
            record["failing_dir"] = failing
            record["crown_calls"] = calls
            records.append(record)

        wall_sec = time.perf_counter() - start
        peak_rss_kb = ProcessMemory.peak_self_rss_kilobytes()
        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        counts = {
            status: sum(1 for r in records if r["status"] == status)
            for status in ("SAT", "UNSAT", "TIMEOUT")
        }
        if verbose:
            print("-" * 88)
            print(
                f"\nSummary: {counts['SAT']} SAT, {counts['UNSAT']} UNSAT, "
                f"{counts['TIMEOUT']} TIMEOUT out of {len(records)} contracts"
            )
            print(f"CROWN calls: {calls_made}  |  wall {wall_sec / 60:.1f} min")

        if write_json:
            output = Path(self.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "onnx_path": self.onnx_path,
                "timestamp": datetime.datetime.now().isoformat(),
                "kind": "liveness",
                "guarantee_type": "in_set",
                "mode": self.mode_description(),
                "timeout_sec": self.timeout_sec,
                "summary": {**counts, "total": len(records),
                            "crown_calls": calls_made},
                "contracts": records,
            }, indent=2))
            if verbose:
                print(f"\nWrote {output}")

        return {
            "wall_sec": round(wall_sec, 3),
            "peak_rss_kb": peak_rss_kb,
            "peak_traced_bytes": peak_traced,
            "crown_calls": calls_made,
            **counts,
            "total": len(records),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="certify only the first N contracts (smoke test)",
    )
    args = parser.parse_args()

    verifier = GridWorldLivenessContractVerifier.build(
        onnx_path=args.onnx,
        output_path=args.output,
        timeout_sec=args.timeout,
        eps=args.eps,
        device=args.device,
    )
    if args.limit is not None:
        verifier.contracts = verifier.contracts[: args.limit]
    verifier.certify_all()


if __name__ == "__main__":
    main()


__all__ = [
    "EXAMPLE_ROOT",
    "GridWorldLivenessContractVerifier",
]
