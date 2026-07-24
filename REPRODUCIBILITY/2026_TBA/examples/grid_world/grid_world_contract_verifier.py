"""
grid_world_contract_verifier.py

Discharge grid-world A/G contracts with alpha-beta-CROWN.

Owns neural verification only -- not the viability kernel (see
grid_world_viability.py) and not inductive partition checks (see
grid_world_inductive_proof.py).

Modes:
  CONTINUOUS (default): one CROWN call per contract; goal ranges over the full
    grid [grid_min, grid_max]^2; drone in an EPS-ball around the source cell.
  DISCRETE: one CROWN call per integer goal in the grid, short-circuit on UNSAT.

Class index mapping (DSL / NN order): We=0 Ea=1 No=2 So=3 XX=4

Run from examples/grid_world/:

    python3 grid_world_contract_verifier.py \\
        --onnx networks/1000__6_18_0__0100_1.onnx \\
        --output contracts/continuous_goals/enabled_pgd/out.json
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

_HERE = Path(__file__).parent.resolve()
_TBA = (_HERE / "../../").resolve()
if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))

from pipeline.crown_verifier import CrownVerifier  # noqa: E402
from pipeline.process_memory import ProcessMemory  # noqa: E402

from grid_world_viability import (  # noqa: E402
    GridWorldContract,
    GridWorldDomain,
    generate_contracts,
    load_config,
)

DISCRETE_GOAL_DEFAULT_TIMEOUT_SEC: float = 5.0
# eps=0.0 is safe in practice when PGD resolves before BaB; use 1e-5 if NaNs appear.
DISCRETE_GOAL_EPS: float = 0.0


def _extract_counterexample(result: Any) -> list[float] | None:
    adv = result.stats.get("attack_examples")
    if adv is None:
        adv = result.stats.get("all_adv_candidates")
    if adv is None:
        return None
    try:
        ce = adv.view(-1)[:4].tolist()
        return [round(v, 6) for v in ce]
    except Exception:
        return None


def _status_marker(status: str) -> str:
    if status == "SAT":
        return "✓"
    if status == "UNSAT":
        return "✗  ← VIOLATION"
    return "?  ← TIMEOUT (inconclusive)"


@dataclass
class GridWorldContractVerifier:
    """
    CROWN-based certifier for kernel-boundary contracts.

    Construct via from_config() (pipeline / CLI) or pass domain + paths explicitly.
    """

    domain: GridWorldDomain
    onnx_path: str
    output_path: str
    num_classes: int
    eps: float
    timeout_sec: float
    pgd_order: str = "before"
    discrete: bool = False
    discrete_timeout: float = DISCRETE_GOAL_DEFAULT_TIMEOUT_SEC
    device: str = "cpu"
    contracts: list[GridWorldContract] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> GridWorldContractVerifier:
        """
        Build a verifier from a domain YAML dict plus runtime keys:

            onnx_path, output_path  (required)
            discrete, discrete_timeout, pgd_order  (optional)
        """
        domain = GridWorldDomain.from_config(cfg)
        discrete = bool(cfg.get("discrete", False))
        eps = 0.0 if discrete else float(cfg["verification"]["eps"])
        contracts = generate_contracts(
            list(domain.obstacles), domain.grid_min, domain.grid_max,
        )
        return cls(
            domain=domain,
            onnx_path=str(cfg["onnx_path"]),
            output_path=str(cfg["output_path"]),
            num_classes=int(cfg["num_classes"]),
            eps=eps,
            timeout_sec=float(cfg["verification"]["timeout_sec"]),
            pgd_order=str(cfg.get("pgd_order", "before")),
            discrete=discrete,
            discrete_timeout=float(
                cfg.get("discrete_timeout", DISCRETE_GOAL_DEFAULT_TIMEOUT_SEC)
            ),
            contracts=contracts,
        )

    def _make_crown_verifier(self) -> CrownVerifier:
        timeout_seconds = self.discrete_timeout if self.discrete else self.timeout_sec
        return CrownVerifier.from_timeout_and_attack_settings(
            timeout_seconds=timeout_seconds,
            pgd_order=self.pgd_order,
            device=self.device,
        )

    def mode_description(self) -> str:
        d = self.domain
        if self.discrete:
            n_goals = d.side_length ** 2
            return (
                f"discrete, {n_goals} integer goals, "
                f"drone EPS={self.eps}, goal EPS={DISCRETE_GOAL_EPS}, "
                f"timeout={self.discrete_timeout}s per goal"
            )
        return (
            f"single-call, goal=[{d.grid_min},{d.grid_max}]^2, "
            f"drone EPS={self.eps}"
        )

    def certify_continuous(
        self,
        contract: GridWorldContract,
        crown_verifier: CrownVerifier | None = None,
    ) -> tuple[str, list[float] | None]:
        """One CROWN call: drone near source, goal over the full continuous grid."""
        if crown_verifier is None:
            crown_verifier = self._make_crown_verifier()
        cx, cy = contract.source
        d = self.domain
        lower = [cx - self.eps, cy - self.eps, d.grid_min, d.grid_min]
        upper = [cx + self.eps, cy + self.eps, d.grid_max, d.grid_max]
        status, result = crown_verifier.certify_network_never_selects_class(
            onnx_path=self.onnx_path,
            input_lower_bounds=lower,
            input_upper_bounds=upper,
            forbidden_class_index=contract.forbidden_dir_idx,
            number_of_classes=self.num_classes,
        )
        ce = _extract_counterexample(result) if status == "UNSAT" else None
        return status, ce

    def certify_discrete(
        self,
        contract: GridWorldContract,
        crown_verifier: CrownVerifier | None = None,
    ) -> tuple[str, list[float] | None]:
        """
        CROWN once per integer goal; short-circuit on first UNSAT.
        TIMEOUT if no UNSAT but at least one call timed out.
        """
        if crown_verifier is None:
            crown_verifier = self._make_crown_verifier()
        cx, cy = contract.source
        d = self.domain
        timeout_seen = False
        for gx in range(d.grid_min, d.grid_max + 1):
            for gy in range(d.grid_min, d.grid_max + 1):
                lower = [
                    cx - self.eps, cy - self.eps,
                    gx - DISCRETE_GOAL_EPS, gy - DISCRETE_GOAL_EPS,
                ]
                upper = [
                    cx + self.eps, cy + self.eps,
                    gx + DISCRETE_GOAL_EPS, gy + DISCRETE_GOAL_EPS,
                ]
                status, result = crown_verifier.certify_network_never_selects_class(
                    onnx_path=self.onnx_path,
                    input_lower_bounds=lower,
                    input_upper_bounds=upper,
                    forbidden_class_index=contract.forbidden_dir_idx,
                    number_of_classes=self.num_classes,
                )
                if status == "UNSAT":
                    return "UNSAT", _extract_counterexample(result)
                if status == "TIMEOUT":
                    timeout_seen = True
        return ("TIMEOUT" if timeout_seen else "SAT"), None

    def certify(
        self,
        contract: GridWorldContract,
        crown_verifier: CrownVerifier | None = None,
    ) -> tuple[str, list[float] | None]:
        """Discharge one contract in the verifier's configured mode."""
        if self.discrete:
            return self.certify_discrete(contract, crown_verifier)
        return self.certify_continuous(contract, crown_verifier)

    def certify_all(self, *, write_json: bool = True, verbose: bool = True) -> dict[str, Any]:
        """
        Certify every kernel-boundary contract; optionally write CROWN JSON.

        Returns pipeline metrics: wall_sec, peak_rss_kb, sat/unsat/timeout counts.
        """
        if not self.contracts:
            self.contracts = generate_contracts(
                list(self.domain.obstacles),
                self.domain.grid_min,
                self.domain.grid_max,
            )

        mode_str = self.mode_description()
        if verbose:
            mode_note = (
                f"discrete mode: {self.domain.side_length ** 2} integer goals, "
                f"timeout={self.discrete_timeout}s per goal"
                if self.discrete
                else (
                    f"drone EPS={self.eps}, "
                    f"goal=[{self.domain.grid_min},{self.domain.grid_max}]^2, "
                    f"timeout={self.timeout_sec}s"
                )
            )
            print(f"Generated {len(self.contracts)} contracts  ({mode_note})\n")
            print(f"{'#':<4} {'Description':<45} {'Status':<10} {'Marker'}")
            print("-" * 75)

        crown_verifier = self._make_crown_verifier()
        tracemalloc.start()
        t0 = time.perf_counter()
        records: list[dict[str, Any]] = []

        for i, contract in enumerate(self.contracts):
            status, counterexample = self.certify(contract, crown_verifier)
            if verbose:
                print(
                    f"{i + 1:<4} {contract.description:<45} "
                    f"{status:<10} {_status_marker(status)}"
                )
                if counterexample is not None:
                    gx, gy = counterexample[2], counterexample[3]
                    is_int = (
                        abs(gx - round(gx)) < 0.01 and abs(gy - round(gy)) < 0.01
                    )
                    print(
                        f"       CE: drone=({counterexample[0]:.4f},"
                        f"{counterexample[1]:.4f}) "
                        f"goal=({gx:.4f},{gy:.4f})  goal_is_integer={is_int}"
                    )
                sys.stdout.flush()

            rec = contract.to_spec_dict(contract_id=i + 1)
            rec["status"] = status
            rec["counterexample"] = counterexample
            records.append(rec)

        wall_sec = time.perf_counter() - t0
        rss_after = ProcessMemory.peak_self_rss_kilobytes()
        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        counts = {
            s: sum(1 for r in records if r["status"] == s)
            for s in ("SAT", "UNSAT", "TIMEOUT")
        }
        if verbose:
            print("-" * 75)
            print(
                f"\nSummary: {counts['SAT']} SAT, {counts['UNSAT']} UNSAT, "
                f"{counts['TIMEOUT']} TIMEOUT out of {len(records)} contracts"
            )

        if write_json:
            self.write_report(records, mode_str)

        return {
            "wall_sec": round(wall_sec, 3),
            "peak_rss_kb": rss_after,
            "peak_traced_bytes": peak_traced,
            "sat": counts["SAT"],
            "unsat": counts["UNSAT"],
            "timeout": counts["TIMEOUT"],
            "total": len(records),
            "skipped": False,
        }

    def write_report(self, records: list[dict[str, Any]], mode_str: str) -> None:
        counts = {
            s: sum(1 for r in records if r["status"] == s)
            for s in ("SAT", "UNSAT", "TIMEOUT")
        }
        report = {
            "onnx_path": self.onnx_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": mode_str,
            "timeout_sec": self.timeout_sec,
            "summary": {**counts, "total": len(records)},
            "contracts": records,
        }
        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Results saved to {self.output_path}")


def run_verification(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pipeline entry point: build verifier from cfg and certify all contracts."""
    return GridWorldContractVerifier.from_config(cfg).certify_all()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify grid-world A/G contracts via alpha-beta-CROWN "
                    "(GridWorldContractVerifier)."
    )
    parser.add_argument(
        "--config",
        default="grid_world_domain_config.yaml",
        help="Path to YAML config (default: grid_world_domain_config.yaml)",
    )
    parser.add_argument("--onnx", required=True, help="Path to the ONNX network file")
    parser.add_argument("--output", required=True, help="Path to write the contracts JSON")
    parser.add_argument(
        "--no-pgd",
        action="store_true",
        help="Disable PGD attack (BaB only); sets pgd_order=skip",
    )
    parser.add_argument(
        "--discrete",
        action="store_true",
        help=(
            "Discrete verification mode: check each integer goal position "
            "individually instead of the full continuous range."
        ),
    )
    parser.add_argument(
        "--discrete-timeout",
        type=float,
        default=DISCRETE_GOAL_DEFAULT_TIMEOUT_SEC,
        help=(
            f"Per-goal timeout in seconds for discrete mode "
            f"(default: {DISCRETE_GOAL_DEFAULT_TIMEOUT_SEC}s)."
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg["onnx_path"] = args.onnx
    cfg["output_path"] = args.output
    if args.no_pgd:
        cfg["pgd_order"] = "skip"
    if args.discrete:
        cfg["discrete"] = True
        cfg["discrete_timeout"] = args.discrete_timeout

    run_verification(cfg)


if __name__ == "__main__":
    main()
