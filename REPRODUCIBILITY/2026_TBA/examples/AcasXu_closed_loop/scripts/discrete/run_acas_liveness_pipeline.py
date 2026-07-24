"""
scripts/discrete/run_acas_liveness_pipeline.py

Liveness product-line driver (compositional NSBT method) — symmetric with
run_acas_safety_pipeline.py:

  1. [TREE]     Generate tree/acas_closed_loop.tree if needed
  2. [SMV]      Generate symbolic/smv/acas_closed_loop.smv if needed
  3. [GENERATE] AcasLivenessContractGenerator → equals-pin specs
  4. [CROWN]    AcasLivenessContractVerifier (always_selects), or SAT stubs
  5. [PATCH]    AcasSmvContractPatcher + optional CTL eventually-far
  6. [NUXMV]    Symbolic check
  7. [REPORT]   pipeline_report.json

Contract paths default from core/liveness/acas_liveness_params.yaml.

Usage (from AcasXu_closed_loop/):

    python3 scripts/discrete/run_acas_liveness_pipeline.py --skip-tree --skip-smv --no-ctl
    python3 scripts/discrete/run_acas_liveness_pipeline.py --run-crown
    python3 scripts/discrete/run_acas_liveness_pipeline.py --skip-contracts --skip-tree --skip-smv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent  # scripts/discrete/
_EXAMPLE = _SCRIPTS.parent.parent  # AcasXu_closed_loop/
_TBA = (_EXAMPLE / "../..").resolve()

if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from pipeline.pipeline_report_writer import PipelineReportWriter  # noqa: E402
from pipeline.nuxmv_verifier import NuxmvVerifier  # noqa: E402

from core.acas_artifact_builder import AcasArtifactBuilder  # noqa: E402
from core.acas_smv_contract_patcher import AcasSmvContractPatcher  # noqa: E402
from core.liveness.acas_liveness_contract_config import (  # noqa: E402
    AcasLivenessContractConfig,
)
from core.liveness.acas_liveness_contract_generator import (  # noqa: E402
    AcasLivenessContractGenerator,
)
from core.liveness.acas_liveness_contract_verifier import (  # noqa: E402
    AcasLivenessContractVerifier,
)

DEFAULT_NUXMV = _TBA / "nuXmv_DL/bin/nuXmv"
DEFAULT_NUXMV_CMD_COMBO = _TBA / "commands/nuxmv_commands/command_combo_invar_ctl"
DEFAULT_TREE = _EXAMPLE / "tree/acas_closed_loop.tree"
DEFAULT_BASE_SMV = _EXAMPLE / "symbolic/smv/acas_closed_loop.smv"
DEFAULT_OUTPUT = _EXAMPLE / "results/discrete/liveness"


def _inject_ctl_eventually_far(smv: str, distance_threshold: int) -> str:
    """CTLSPEC AG AF (distance_stage_0 >= threshold) for far-away liveness goal."""
    marker = "--------------SPECIFICATIONS\n"
    if marker not in smv:
        raise ValueError("SPECIFICATIONS marker not found")
    ctl_line = (
        f"CTLSPEC NAME liveness_eventually_far := "
        f"AG AF (system.distance_stage_0 >= {distance_threshold});\n"
    )
    if "liveness_eventually_far" in smv:
        return smv
    invar_marker = "INVARSPEC"
    idx = smv.find(invar_marker)
    if idx < 0:
        return smv.replace(marker, marker + ctl_line, 1)
    semi = smv.find(";", idx)
    if semi < 0:
        return smv.replace(marker, marker + ctl_line, 1)
    return smv[: semi + 1] + "\n" + ctl_line + smv[semi + 1 :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to acas_liveness_params.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for patched SMV, nuXmv output, pipeline report",
    )
    parser.add_argument("--base-smv", type=Path, default=DEFAULT_BASE_SMV)
    parser.add_argument("--nuxmv", type=Path, default=DEFAULT_NUXMV)
    parser.add_argument(
        "--nuxmv-cmd",
        type=Path,
        default=DEFAULT_NUXMV_CMD_COMBO,
        help="command_combo_invar_ctl for CTL; command_invar for INVAR only",
    )
    parser.add_argument(
        "--skip-contracts",
        action="store_true",
        help="Skip generate + CROWN; use existing specs/results from config",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse specs if present (still may run CROWN)",
    )
    parser.add_argument(
        "--run-crown",
        action="store_true",
        help="Run AcasLivenessContractVerifier (always_selects)",
    )
    parser.add_argument("--skip-tree", action="store_true")
    parser.add_argument("--skip-smv", action="store_true")
    parser.add_argument("--no-ctl", action="store_true")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = AcasLivenessContractConfig.from_yaml(args.config)
    specs_path = config.specs_path
    results_path = config.results_path
    ctl_distance_threshold = config.domain.max_dist

    output_dir = (
        args.output if args.output.is_absolute() else _EXAMPLE / args.output
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.perf_counter()

    builder = AcasArtifactBuilder(tree_path=DEFAULT_TREE, smv_path=DEFAULT_BASE_SMV)
    tree_metrics = builder.ensure_tree(reuse_existing=args.skip_tree)
    smv_metrics = builder.ensure_smv(reuse_existing=args.skip_smv)

    if args.skip_contracts:
        if not specs_path.is_file() or not results_path.is_file():
            raise FileNotFoundError(
                "--skip-contracts requires existing specs and results "
                f"({specs_path}, {results_path})"
            )
        print("\n[CONTRACTS] skipped (using pre-verified liveness JSON)")
        generate_metrics: dict[str, Any] = {
            "skipped": True,
            "n_contracts": len(
                json.loads(specs_path.read_text(encoding="utf-8"))["contracts"]
            ),
            "ctl_distance_threshold": ctl_distance_threshold,
        }
    else:
        reuse_specs = args.skip_generate and specs_path.is_file()
        margins: dict[int, float] = {}
        contracts = None
        if reuse_specs:
            print(f"\n[GENERATE] skipped — reusing {specs_path}")
            generate_metrics = {
                "skipped": True,
                "n_contracts": len(
                    json.loads(specs_path.read_text(encoding="utf-8"))["contracts"]
                ),
                "ctl_distance_threshold": ctl_distance_threshold,
            }
        else:
            print("\n" + "=" * 60)
            print("[GENERATE] liveness contracts")
            print("=" * 60)
            generator = AcasLivenessContractGenerator.from_config(config)
            contracts = generator.generate_all_contracts()
            margins = generator.fill_onnx_margins(contracts)
            generator.write_specs(contracts, specs_path)
            generate_metrics = {
                "wall_sec": 0.0,
                "n_contracts": len(contracts),
                "ctl_distance_threshold": ctl_distance_threshold,
                "max_rho_on_onnx_cycle": (
                    generator.trajectory.max_distance_on_cycle()
                ),
                "min_onnx_margin": min(margins.values()) if margins else None,
                "nn_input_eps": config.nn_input_eps,
            }

        verifier = AcasLivenessContractVerifier.from_config(
            config,
            timeout_sec=args.timeout,
            device=args.device,
            onnx_margins=margins,
        )
        if contracts is None:
            contracts = verifier.load_liveness_contracts(specs_path)
        if args.run_crown:
            print("\n" + "=" * 60)
            print("[CROWN] liveness always_selects")
            print("=" * 60)
            verifier.run(contracts=contracts, results_path=results_path)
            generate_metrics["crown_ran"] = True
        else:
            verifier.write_sat_stubs(
                contracts, results_path, onnx_margins=margins,
            )
            print(
                "  CROWN skipped (ONNX-backed SAT stubs); "
                "pass --run-crown for real checks"
            )
            generate_metrics["crown_ran"] = False

    print("\n" + "=" * 60)
    print("[PATCH] SMV equals-INVAR injection")
    print("=" * 60)
    patcher = AcasSmvContractPatcher.from_verifier_yaml()
    sat_contracts = AcasSmvContractPatcher.load_sat_contracts(
        specs_path, results_path,
    )
    seed_contracts = [
        contract for contract in sat_contracts
        if contract.get("dangerous_xy") == [[7, 6]]
        and contract.get("a_prev") == "clear"
    ]
    print(f"[CHECK] seed clear contracts: {len(seed_contracts)} (expect >= 1)")

    smv_path = output_dir / "acas_closed_loop_liveness.smv"
    base_smv = (
        args.base_smv if args.base_smv.is_absolute() else _EXAMPLE / args.base_smv
    )
    patch_start = time.perf_counter()
    patch_info = patcher.patch_file(base_smv, smv_path, sat_contracts)
    if not args.no_ctl:
        patched = smv_path.read_text(encoding="utf-8")
        patched = _inject_ctl_eventually_far(patched, ctl_distance_threshold)
        smv_path.write_text(patched, encoding="utf-8")
        patch_info["ctl_injected"] = True
        patch_info["distance_threshold"] = ctl_distance_threshold
    else:
        patch_info["ctl_injected"] = False
        patch_info["distance_threshold"] = None
    patch_info["wall_sec"] = round(time.perf_counter() - patch_start, 3)
    print(
        f"  Injected {patch_info['invar_lines']} equals-INVARs "
        f"(CTL={patch_info['ctl_injected']})"
    )

    nuxmv_ctx = {
        "nuxmv_bin": args.nuxmv.resolve(),
        "nuxmv_cmd": args.nuxmv_cmd.resolve(),
        "smv_path": smv_path,
        "nuxmv_out_path": output_dir / "nuxmv_output.txt",
    }
    nuxmv_metrics = NuxmvVerifier.from_pipeline_context(
        nuxmv_ctx,
    ).run_and_collect_metrics()

    PipelineReportWriter.from_path(output_dir / "pipeline_report.json").write(
        steps={
            "tree": tree_metrics,
            "smv": smv_metrics,
            "liveness_contracts": generate_metrics,
            "smv_patch": patch_info,
            "nuxmv": nuxmv_metrics,
        },
        total_wall_seconds=time.perf_counter() - pipeline_start,
        extra_fields={
            "specs": str(specs_path),
            "results": str(results_path),
            "mode": "liveness_compositional",
            "config": str(
                args.config or AcasLivenessContractConfig.DEFAULT_YAML_PATH
            ),
        },
    )
    print(f"\nDone. Report: {output_dir / 'pipeline_report.json'}")


if __name__ == "__main__":
    main()
