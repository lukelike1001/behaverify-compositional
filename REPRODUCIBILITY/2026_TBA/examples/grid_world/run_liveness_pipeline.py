#!/usr/bin/env python3
"""
run_liveness_pipeline.py

End-to-end compositional LIVENESS pipeline for one grid-world network:

    1. CROWN   discharge progress contracts (skippable with --skip-contracts)
    2. merge   safety + liveness SAT contracts into one injection file
    3. SMV     replace the NN table with the merged INVAR constraints
    4. nuXmv   check INVARSPEC and CTLSPEC on the abstract model
    5. report  write pipeline_report.json

The question this answers: does constraining the abstract network to
Dec(source, goal) make CTLSPEC true compositionally, matching the monolithic
result that safety contracts alone cannot reproduce?

Safety contracts are injected ALONGSIDE liveness by default, and both are
needed for a sound joint verdict.

Liveness contracts exist only where dist(source, goal) is defined -- 930 of the
1519 (drone, goal) states this model can occupy. The environment picks a new
goal nondeterministically over the whole grid, obstacle cells included, so 558
states have a goal inside an obstacle and carry NO progress obligation; the
abstract network is unconstrained there and free to crash. The CTL
specification excuses those states explicitly (its "target in Obs" disjunct);
INVARSPEC does not. Safety contracts guard on the drone cell for every goal,
which is exactly the cover liveness cannot provide.

  --liveness-only  ablation: isolates the liveness contribution to CTLSPEC.
                   Expect INVARSPEC=false, for the coverage reason above and
                   not because progress contracts failed.

Usage (from examples/grid_world/):

    python3 run_liveness_pipeline.py \\
        --onnx networks/1000__6_18_0__0200_1.onnx \\
        --output results/liveness/1000__6_18_0__0200_1 \\
        --liveness-contracts contracts/discrete/liveness/1000__6_18_0__0200_1_liveness.json \\
        --safety-contracts contracts/discrete/safety/1000__6_18_0__0200_1_discrete.json \\
        --skip-contracts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).parent.resolve()
_TBA = (_HERE / "../../").resolve()
if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))

from pipeline.nuxmv_verifier import NuxmvVerifier  # noqa: E402
from pipeline.pipeline_report_writer import PipelineReportWriter  # noqa: E402

from core.grid_world_domain import GridWorldDomain  # noqa: E402
from core.liveness.grid_world_liveness_contract_verifier import (  # noqa: E402
    GridWorldLivenessContractVerifier,
)
from grid_world_pipeline_context import GridWorldPipelineContext  # noqa: E402
from grid_world_smv_builder import GridWorldSmvBuilder  # noqa: E402

_COUNTER_TEMPLATE = _HERE / "counter_template.tree"
_PIPELINE_CONFIG = _HERE / "pipeline_filepaths_config.yaml"


def _load_pipeline_config(path: Path = _PIPELINE_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_CFG = _load_pipeline_config()
_PATHS = _CFG["paths"]
_SMV_CFG = {**_CFG["smv"], "src_dir": str((_TBA / "src").resolve())}

_DEFAULT_NUXMV = (_HERE / _PATHS["nuxmv_bin"]).resolve()
_DEFAULT_NUXMV_CMD = (_HERE / _PATHS["nuxmv_cmd"]).resolve()
_DEFAULT_METAMODEL = (_HERE / _PATHS["metamodel"]).resolve()
_DEFAULT_CONFIG = (_HERE / _PATHS["contracts_config"]).resolve()


def _summarize(path: Path) -> dict[str, Any]:
    """SAT/UNSAT/TIMEOUT counts from a CROWN result file."""
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    return {
        "path": str(path),
        "sat": summary.get("SAT", 0),
        "unsat": summary.get("UNSAT", 0),
        "timeout": summary.get("TIMEOUT", 0),
        "total": summary.get("total", len(data.get("contracts", []))),
    }


def merge_contract_files(
    liveness_path: Path,
    safety_path: Path | None,
    merged_path: Path,
) -> dict[str, Any]:
    """
    Concatenate SAT contracts from both kinds into one injection file.

    Records keep their own shape -- safety rows carry "forbidden_dir", liveness
    rows carry "forbidden_dirs" + "goal" -- and build_invar_lines dispatches on
    that. Non-SAT rows are dropped here as well as downstream, so the counts in
    the report describe what was actually injected.
    """
    records: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}

    liveness = json.loads(liveness_path.read_text())
    liveness_sat = [c for c in liveness["contracts"] if c["status"] == "SAT"]
    records.extend(liveness_sat)
    sources["liveness"] = _summarize(liveness_path)

    if safety_path is not None:
        safety = json.loads(safety_path.read_text())
        safety_sat = [c for c in safety["contracts"] if c["status"] == "SAT"]
        records.extend(safety_sat)
        sources["safety"] = _summarize(safety_path)

    for index, record in enumerate(records):
        record["id"] = index + 1

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(json.dumps({
        "description": (
            "Merged SAT contracts for the compositional liveness run: "
            "liveness (in_set, goal-guarded) plus safety (never_selects)."
        ),
        "summary": {
            "SAT": len(records), "UNSAT": 0, "TIMEOUT": 0,
            "total": len(records),
        },
        "sources": sources,
        "contracts": records,
    }, indent=2))

    return {
        "merged_path": str(merged_path),
        "liveness_sat": len(liveness_sat),
        "safety_sat": len(records) - len(liveness_sat),
        "injected_contracts": len(records),
        "sources": sources,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--tree", default=None)
    p.add_argument("--liveness-contracts", default=None, dest="liveness_contracts")
    p.add_argument("--safety-contracts", default=None, dest="safety_contracts")
    p.add_argument(
        "--liveness-only", action="store_true", dest="liveness_only",
        help=(
            "ablation: inject only progress contracts. Leaves the 558 "
            "obstacle-goal states unconstrained, so INVARSPEC is expected to "
            "fail for coverage reasons unrelated to liveness."
        ),
    )
    p.add_argument(
        "--skip-contracts", action="store_true",
        help="reuse an existing liveness CROWN result instead of re-running",
    )
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--config", default=str(_DEFAULT_CONFIG))
    p.add_argument("--nuxmv", default=str(_DEFAULT_NUXMV))
    p.add_argument("--nuxmv-cmd", default=str(_DEFAULT_NUXMV_CMD), dest="nuxmv_cmd")
    p.add_argument("--metamodel", default=str(_DEFAULT_METAMODEL))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    pipeline_start = time.perf_counter()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    network_name = Path(args.onnx).stem

    liveness_path = Path(
        args.liveness_contracts
        or output_dir / f"{network_name}_liveness.json"
    ).resolve()

    # --- step 1: CROWN ----------------------------------------------------
    if args.skip_contracts:
        print("\n[1/4] LIVENESS CONTRACTS — skipped (reusing existing JSON)")
        contracts_metrics = {**_summarize(liveness_path), "skipped": True}
    else:
        verifier = GridWorldLivenessContractVerifier.build(
            onnx_path=os.path.relpath(Path(args.onnx).resolve()),
            output_path=os.path.relpath(liveness_path),
            domain=GridWorldDomain.from_config(config_path=args.config),
            timeout_sec=args.timeout,
        )
        contracts_metrics = verifier.certify_all()

    # --- step 2: merge ----------------------------------------------------
    if not args.liveness_only and not args.safety_contracts:
        raise SystemExit(
            "the default configuration injects safety contracts too; pass "
            "--safety-contracts <CROWN results>, or --liveness-only to run "
            "the ablation without them"
        )
    safety_path = (
        None if args.liveness_only else Path(args.safety_contracts).resolve()
    )
    merged_path = output_dir / f"{network_name}_merged_contracts.json"
    merge_metrics = merge_contract_files(liveness_path, safety_path, merged_path)
    print(
        f"\n[2/4] MERGE — {merge_metrics['injected_contracts']} SAT contracts "
        f"({merge_metrics['liveness_sat']} liveness, "
        f"{merge_metrics['safety_sat']} safety)"
    )

    # --- steps 3-4: SMV + nuXmv -------------------------------------------
    context = GridWorldPipelineContext.from_cli_arguments(
        argparse.Namespace(
            onnx=args.onnx,
            output=str(output_dir),
            tree=args.tree,
            config=args.config,
            nuxmv=args.nuxmv,
            nuxmv_cmd=args.nuxmv_cmd,
            metamodel=args.metamodel,
            skip_contracts=True,
            contracts=str(merged_path),
        ),
        _COUNTER_TEMPLATE,
    )
    ctx = context.as_dict()

    smv_metrics = GridWorldSmvBuilder.from_pipeline_ctx(ctx, _SMV_CFG).generate()
    nuxmv_metrics = NuxmvVerifier.from_pipeline_context(ctx).run_and_collect_metrics()

    PipelineReportWriter.from_path(context.report_path).write(
        steps={
            "liveness_contracts": contracts_metrics,
            "contract_merge": merge_metrics,
            "smv_generation": smv_metrics,
            "nuxmv_verification": nuxmv_metrics,
        },
        total_wall_seconds=time.perf_counter() - pipeline_start,
        extra_fields={
            "network": network_name,
            "onnx_path": str(context.onnx_path),
            "tree_path": str(context.tree_path),
            "mode": "liveness_only" if args.liveness_only else "liveness+safety",
        },
    )

    print("\n" + "=" * 60)
    print(
        f"INVARSPEC = {nuxmv_metrics.get('invarspec')}   "
        f"CTLSPEC = {nuxmv_metrics.get('ctlspec')}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
