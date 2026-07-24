"""
scripts/discrete/run_acas_safety_pipeline.py

Safety product-line driver (compositional NSBT method):

  1. [TREE]     Generate tree/acas_closed_loop.tree if needed
  2. [SMV]      Generate symbolic/smv/acas_closed_loop.smv if needed
  3. [GENERATE] AcasSafetyContractGenerator → specs JSON
  4. [CROWN]    AcasSafetyContractVerifier (never_selects) → results JSON
  5. [PATCH]    AcasSmvContractPatcher injects SAT contracts
  6. [NUXMV]    Symbolic INVARSPEC check
  7. [REPORT]   pipeline_report.json

Neural stages are skippable when specs/results already exist
(--skip-contracts). Tree/SMV skip when artifacts exist and flags set.

Usage (from AcasXu_closed_loop/):

    # Full discrete safety for one network (generate + CROWN + symbolic)
    python3 scripts/discrete/run_acas_safety_pipeline.py \\
        --output results/discrete/safety/nn1 \\
        --network-idx 1 --discrete --run-crown

    # Symbolic only (pre-verified corridor — April-style step 3)
    python3 scripts/discrete/run_acas_safety_pipeline.py \\
        --specs contracts/crown/discrete/safety/safety_corridor_contracts.json \\
        --results contracts/crown/discrete/safety/safety_corridor_contract_results.json \\
        --output results/discrete/safety/corridor \\
        --skip-contracts --skip-tree --skip-smv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent  # scripts/discrete/
_EXAMPLE = _SCRIPTS.parent.parent  # AcasXu_closed_loop/
_TBA = (_EXAMPLE / "../..").resolve()
_REPO_SRC = (_TBA / "src").resolve()

if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from pipeline.pipeline_report_writer import PipelineReportWriter  # noqa: E402
from pipeline.process_memory import ProcessMemory  # noqa: E402
from pipeline.symbolic.nuxmv.nuxmv_verifier import NuxmvVerifier  # noqa: E402

from core.acas_contract import AcasSafetyContract  # noqa: E402
from core.acas_smv_contract_patcher import AcasSmvContractPatcher  # noqa: E402
from core.safety.acas_safety_contract_generator import (  # noqa: E402
    AcasSafetyContractGenerator,
)
from core.safety.acas_safety_contract_verifier import (  # noqa: E402
    AcasSafetyContractVerifier,
)

DEFAULT_NUXMV = _TBA / "nuXmv_DL/bin/nuXmv"
DEFAULT_NUXMV_CMD = _TBA / "commands/nuxmv_commands/command_invar"
DEFAULT_METAMODEL = _TBA / "metamodel/behaverify.tx"
DEFAULT_VERIFIER_YAML = _EXAMPLE / "core" / "acas_verifier_params.yaml"
DEFAULT_SPECS = (
    _EXAMPLE / "contracts/crown/discrete/safety/safety_full_contracts.json"
)
DEFAULT_RESULTS = (
    _EXAMPLE / "contracts/crown/discrete/safety/safety_full_results.json"
)
DEFAULT_OUTPUT = _EXAMPLE / "results/discrete/safety/full"
DEFAULT_TREE = _EXAMPLE / "tree/acas_closed_loop.tree"
DEFAULT_BASE_SMV = _EXAMPLE / "symbolic/smv/acas_closed_loop.smv"


def _ensure_tree(*, skip: bool) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("[TREE] generation")
    print("=" * 60)
    if skip and DEFAULT_TREE.exists():
        print(f"  Skipped — reusing {DEFAULT_TREE.relative_to(_EXAMPLE)}")
        return {"wall_sec": 0.0, "skipped": True}
    from core.acas_tree_generator import AcasTreeGenerator  # noqa: PLC0415

    start = time.perf_counter()
    AcasTreeGenerator(output_path=DEFAULT_TREE).generate()
    wall = time.perf_counter() - start
    print(f"  Generated {DEFAULT_TREE.relative_to(_EXAMPLE)} ({wall:.1f}s)")
    return {"wall_sec": round(wall, 3), "skipped": False}


def _ensure_smv(*, skip: bool) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("[SMV] base model generation")
    print("=" * 60)
    if skip and DEFAULT_BASE_SMV.exists():
        print(f"  Skipped — reusing {DEFAULT_BASE_SMV.relative_to(_EXAMPLE)}")
        return {"wall_sec": 0.0, "skipped": True}
    if not DEFAULT_TREE.is_file():
        raise FileNotFoundError(f"tree missing: {DEFAULT_TREE}")
    src_dir = str(_REPO_SRC)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import dsl_to_nuxmv as dsl  # noqa: PLC0415

    DEFAULT_BASE_SMV.parent.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    start = time.perf_counter()
    previous_cwd = os.getcwd()
    os.chdir(str(_EXAMPLE))
    try:
        dsl.dsl_to_nuxmv(
            str(DEFAULT_METAMODEL),
            str(DEFAULT_TREE.relative_to(_EXAMPLE)),
            str(DEFAULT_BASE_SMV.relative_to(_EXAMPLE)),
            False, False, False, False,
            10000, False, True, None,
        )
    finally:
        os.chdir(previous_cwd)
    wall = time.perf_counter() - start
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    lines = DEFAULT_BASE_SMV.read_text(encoding="utf-8").count("\n")
    print(f"  Generated {DEFAULT_BASE_SMV.relative_to(_EXAMPLE)} ({wall:.1f}s)")
    return {
        "wall_sec": round(wall, 3),
        "smv_lines": lines,
        "peak_traced_bytes": peak_traced,
        "peak_rss_kb": ProcessMemory.peak_self_rss_kilobytes(),
        "skipped": False,
    }


def _generate_contracts(specs_path: Path, eps: float) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("[GENERATE] safety contracts")
    print("=" * 60)
    start = time.perf_counter()
    generator = AcasSafetyContractGenerator.from_yaml()
    pairs = generator.enumerate_dangerous_pairs()
    contracts = generator.group_range_contracts(pairs, eps=eps)
    AcasSafetyContract.dump_json(
        contracts,
        specs_path,
        description="ACAS Xu closed-loop A/G safety contract specs (range-based)",
    )
    wall = time.perf_counter() - start
    print(f"  Wrote {len(contracts)} contracts → {specs_path}")
    return {
        "wall_sec": round(wall, 3),
        "n_contracts": len(contracts),
        "n_pairs": len(pairs),
    }


def _run_crown(
    *,
    verifier_yaml: Path,
    specs_path: Path,
    results_path: Path,
    network_idx: int | None,
    discrete: bool,
    discrete_timeout: float | None,
    timeout: float | None,
    limit: int | None,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("[CROWN] safety never_selects")
    print("=" * 60)
    start = time.perf_counter()
    verifier = AcasSafetyContractVerifier.from_yaml(verifier_yaml)
    verifier.contracts_path = specs_path
    verifier.output_path = results_path
    if network_idx is not None:
        verifier.network_idx = network_idx
    if discrete:
        verifier.discrete = True
        if discrete_timeout is not None:
            verifier.discrete_timeout_sec = discrete_timeout
        verifier.rebuild_crown_verifier(verifier.discrete_timeout_sec)
    records = verifier.run(limit=limit, timeout_override=timeout)
    wall = time.perf_counter() - start
    counts = {
        status: sum(1 for record in records if record["status"] == status)
        for status in ("SAT", "UNSAT", "TIMEOUT")
    }
    print(f"  CROWN summary: {counts}")
    return {
        "wall_sec": round(wall, 3),
        "n_verified": len(records),
        **counts,
        "network_idx": verifier.network_idx,
        "discrete": verifier.discrete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specs",
        type=Path,
        default=DEFAULT_SPECS,
        help="Safety contract specs JSON",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="CROWN results JSON (written by --run-crown or pre-existing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for patched SMV, nuXmv log, pipeline_report.json",
    )
    parser.add_argument("--nuxmv", type=Path, default=DEFAULT_NUXMV)
    parser.add_argument("--nuxmv-cmd", type=Path, default=DEFAULT_NUXMV_CMD)
    parser.add_argument(
        "--verifier-config",
        type=Path,
        default=DEFAULT_VERIFIER_YAML,
        help="YAML for SMV var names + CROWN defaults",
    )
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--network-idx", type=int, default=None)
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--discrete-timeout", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--run-crown",
        action="store_true",
        help="Run AcasSafetyContractVerifier before patching",
    )
    parser.add_argument(
        "--skip-contracts",
        action="store_true",
        help="Skip generate + CROWN; use existing --specs and --results",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generator; keep existing specs (still may run CROWN)",
    )
    parser.add_argument("--skip-tree", action="store_true")
    parser.add_argument("--skip-smv", action="store_true")
    args = parser.parse_args()

    specs_path = args.specs if args.specs.is_absolute() else _EXAMPLE / args.specs
    results_path = (
        args.results if args.results.is_absolute() else _EXAMPLE / args.results
    )
    output_dir = (
        args.output if args.output.is_absolute() else _EXAMPLE / args.output
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_path = specs_path.resolve()
    results_path = results_path.resolve()

    pipeline_start = time.perf_counter()
    tree_metrics = _ensure_tree(skip=args.skip_tree)
    smv_metrics = _ensure_smv(skip=args.skip_smv)

    generate_metrics: dict[str, Any]
    crown_metrics: dict[str, Any]

    if args.skip_contracts:
        if not specs_path.is_file() or not results_path.is_file():
            raise FileNotFoundError(
                "--skip-contracts requires existing --specs and --results files"
            )
        print("\n[CONTRACTS] skipped (using pre-verified JSON)")
        generate_metrics = {"skipped": True}
        crown_metrics = {"skipped": True}
    else:
        if args.skip_generate and specs_path.is_file():
            generate_metrics = {"skipped": True, "specs": str(specs_path)}
            print(f"\n[GENERATE] skipped — reusing {specs_path}")
        else:
            generate_metrics = _generate_contracts(specs_path, args.eps)

        if args.run_crown:
            crown_metrics = _run_crown(
                verifier_yaml=args.verifier_config,
                specs_path=specs_path,
                results_path=results_path,
                network_idx=args.network_idx,
                discrete=args.discrete,
                discrete_timeout=args.discrete_timeout,
                timeout=args.timeout,
                limit=args.limit,
            )
        elif results_path.is_file():
            print(f"\n[CROWN] skipped — reusing {results_path}")
            crown_metrics = {"skipped": True, "results": str(results_path)}
        else:
            raise FileNotFoundError(
                f"no CROWN results at {results_path}; pass --run-crown "
                f"or --skip-contracts with existing results"
            )

    print("\n" + "=" * 60)
    print("[PATCH] SMV contract injection")
    print("=" * 60)
    patcher = AcasSmvContractPatcher.from_verifier_yaml(args.verifier_config)
    sat_contracts = AcasSmvContractPatcher.load_sat_contracts(
        specs_path, results_path,
    )
    patched_smv = output_dir / "acas_closed_loop_safety.smv"
    patch_metrics = patcher.patch_file(
        DEFAULT_BASE_SMV, patched_smv, sat_contracts,
    )
    print(
        f"  Injected {patch_metrics['invar_lines']} INVARs from "
        f"{patch_metrics['sat_contracts']} SAT contracts → {patched_smv.name}"
    )

    nuxmv_ctx = {
        "nuxmv_bin": args.nuxmv.resolve(),
        "nuxmv_cmd": args.nuxmv_cmd.resolve(),
        "smv_path": patched_smv,
        "nuxmv_out_path": output_dir / "nuxmv_output.txt",
    }
    nuxmv_metrics = NuxmvVerifier.from_pipeline_context(
        nuxmv_ctx,
    ).run_and_collect_metrics()

    total = time.perf_counter() - pipeline_start
    PipelineReportWriter.from_path(output_dir / "pipeline_report.json").write(
        steps={
            "tree": tree_metrics,
            "smv": smv_metrics,
            "generate": generate_metrics,
            "crown": crown_metrics,
            "smv_patch": patch_metrics,
            "nuxmv": nuxmv_metrics,
        },
        total_wall_seconds=total,
        extra_fields={
            "mode": "safety_compositional",
            "specs": str(specs_path),
            "results": str(results_path),
        },
    )
    print(f"\nDone. Report: {output_dir / 'pipeline_report.json'}")


if __name__ == "__main__":
    main()
