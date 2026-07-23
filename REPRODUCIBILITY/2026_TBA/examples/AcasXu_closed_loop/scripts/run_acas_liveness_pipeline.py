"""
run_acas_liveness_pipeline.py

Liveness contracts → equals-INVARs in SMV → optional CTL → nuXmv.

Generate/verify via AcasLivenessContractGenerator and
AcasLivenessContractVerifier. SMV injection via AcasSmvContractPatcher.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_EXAMPLE = _SCRIPTS.parent
_HERE = _EXAMPLE  # AcasXu_closed_loop root (compat alias)
_TBA = (_HERE / "../../").resolve()
if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from pipeline.pipeline_report_writer import PipelineReportWriter
from pipeline.symbolic.nuxmv.nuxmv_verifier import NuxmvVerifier

from core.liveness.acas_liveness_contract_config import AcasLivenessContractConfig
from core.liveness.acas_liveness_contract_generator import AcasLivenessContractGenerator
from core.liveness.acas_liveness_contract_verifier import AcasLivenessContractVerifier
from core.acas_smv_contract_patcher import AcasSmvContractPatcher

DEFAULT_NUXMV = _HERE / "../../nuXmv_DL/bin/nuXmv"
DEFAULT_NUXMV_CMD_COMBO = _HERE / "../../commands/nuxmv_commands/command_combo_invar_ctl"
DEFAULT_BASE_SMV = _HERE / "symbolic/smv/acas_360.smv"
DEFAULT_OUTPUT = _HERE / "results/discrete/liveness"


def _inject_ctl_eventually_far(smv: str, distance_threshold: int) -> str:
    """
    CTLSPEC AG AF (distance_stage_0 >= threshold).

    Use domain max_dist (active cutoff), not ONNX cycle max rho: SMV freezes
    when distance >= max_dist (acas.active false).
    """
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
        help="Use command_combo_invar_ctl for CTL; command_invar for safety only",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse existing specs/results from config paths",
    )
    parser.add_argument(
        "--run-crown",
        action="store_true",
        help="Run AcasLivenessContractVerifier (always_selects)",
    )
    parser.add_argument("--no-ctl", action="store_true")
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
    args = parser.parse_args()

    config = AcasLivenessContractConfig.from_yaml(args.config)
    specs_path = config.specs_path
    results_path = config.results_path
    ctl_distance_threshold = config.domain.max_dist

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.perf_counter()

    if not args.skip_generate or not specs_path.exists():
        generator = AcasLivenessContractGenerator.from_config(config)
        contracts = generator.generate_all_contracts()
        margins = generator.fill_onnx_margins(contracts)
        generator.write_specs(contracts, specs_path)

        verifier = AcasLivenessContractVerifier.from_config(
            config,
            timeout_sec=args.timeout,
            device=args.device,
            onnx_margins=margins,
        )
        if args.run_crown:
            verifier.run(contracts=contracts, results_path=results_path)
        else:
            verifier.write_sat_stubs(
                contracts, results_path, onnx_margins=margins,
            )
            print("  CROWN skipped (ONNX margins only); statuses set SAT")

        generate_metrics = {
            "wall_sec": 0.0,
            "n_contracts": len(contracts),
            "ctl_distance_threshold": ctl_distance_threshold,
            "max_rho_on_onnx_cycle": (
                generator.trajectory.max_distance_on_cycle()
            ),
            "min_onnx_margin": min(margins.values()) if margins else None,
            "nn_input_eps": config.nn_input_eps,
            "crown_ran": bool(args.run_crown),
        }
    else:
        specs = json.loads(specs_path.read_text(encoding="utf-8"))
        generate_metrics = {
            "wall_sec": 0.0,
            "n_contracts": len(specs["contracts"]),
            "ctl_distance_threshold": ctl_distance_threshold,
            "skipped": True,
        }

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

    smv_path = output_dir / "acas_360_liveness.smv"
    patch_start = time.perf_counter()
    patch_info = patcher.patch_file(
        args.base_smv, smv_path, sat_contracts,
    )
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
        f"  Patched SMV with {patch_info['invar_lines']} equals-INVARs "
        f"(CTL={patch_info['ctl_injected']})"
    )

    ctx = {
        "nuxmv_bin": args.nuxmv.resolve(),
        "nuxmv_cmd": args.nuxmv_cmd.resolve(),
        "smv_path": smv_path,
        "nuxmv_out_path": output_dir / "nuxmv_output.txt",
    }
    nuxmv_metrics = NuxmvVerifier.from_pipeline_context(ctx).run_and_collect_metrics()

    PipelineReportWriter.from_path(output_dir / "pipeline_report.json").write(
        steps={
            "liveness_contracts": generate_metrics,
            "smv_patch": patch_info,
            "nuxmv": nuxmv_metrics,
        },
        total_wall_seconds=time.perf_counter() - pipeline_start,
        extra_fields={
            "specs": str(specs_path),
            "results": str(results_path),
            "mode": "liveness_equals_contracts",
            "config": str(
                args.config or AcasLivenessContractConfig.DEFAULT_YAML_PATH
            ),
        },
    )


if __name__ == "__main__":
    main()
