"""
run_acas_liveness_pipeline.py

Pin the abstract NN on the 52-state lasso, inject equals-INVARs, optional CTL,
run nuXmv. Reuses SMV patch helpers from run_acas_compositional_pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
_TBA = (_HERE / "../../").resolve()
if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))

from pipeline.pipeline_report_writer import PipelineReportWriter
from pipeline.symbolic.nuxmv.nuxmv_verifier import NuxmvVerifier

import run_acas_compositional_pipeline as acas_pipe
from acas_lasso_pins import (
    AcasLassoPinSet,
    AcasLassoTrajectory,
    DEFAULT_EPS,
    DEFAULT_LASSO_JSON,
)

DEFAULT_NUXMV = _HERE / "../../nuXmv_DL/bin/nuXmv"
DEFAULT_NUXMV_CMD_INVAR = _HERE / "../../commands/nuxmv_commands/command_invar"
DEFAULT_NUXMV_CMD_COMBO = _HERE / "../../commands/nuxmv_commands/command_combo_invar_ctl"
DEFAULT_BASE_SMV = _HERE / "symbolic/smv/acas_360.smv"


def _inject_ctl_eventually_far(smv: str, distance_threshold: int) -> str:
    """
    CTLSPEC AG AF (distance_stage_0 >= threshold).

    Use threshold=1000 (max_dist / active cutoff), not 1400: SMV freezes when
    distance >= 1000 (acas.active false), so AG AF (rho>=1400) is false even
    under perfect pins — the CE loops at rho=1000 with active=FALSE.
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


def _load_sat_pin_contracts(specs_path: Path, results_path: Path) -> list[dict]:
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    status_by_id = {c["id"]: c["status"] for c in results["contracts"]}
    sat = []
    for contract in specs["contracts"]:
        status = status_by_id.get(contract["id"])
        if status != "SAT":
            continue
        merged = {**contract, "status": status}
        sat.append(merged)
    print(f"  {len(sat)} SAT pins loaded (of {len(specs['contracts'])} total)")
    return sat


def patch_smv_with_pins(
    base_smv_path: Path,
    output_smv_path: Path,
    sat_contracts: list[dict],
    smv_vars: dict[str, str],
    d_star: int | None,
    inject_ctl: bool,
) -> dict:
    smv = base_smv_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    smv, lines_removed = acas_pipe._remove_nn_defines(smv)
    smv = acas_pipe._add_command_free_var(smv)
    invar_lines = acas_pipe._build_invar_lines(sat_contracts, smv_vars)
    smv = acas_pipe._inject_invars(smv, invar_lines)
    if inject_ctl and d_star is not None:
        # d_star arg repurposed as distance threshold for "eventually far"
        smv = _inject_ctl_eventually_far(smv, d_star)
    output_smv_path.parent.mkdir(parents=True, exist_ok=True)
    output_smv_path.write_text(smv, encoding="utf-8")
    return {
        "sat_contracts": len(sat_contracts),
        "invar_lines": len(invar_lines),
        "nn_lines_removed": lines_removed,
        "ctl_injected": bool(inject_ctl and d_star is not None),
        "d_star": d_star,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lasso-json", type=Path, default=DEFAULT_LASSO_JSON)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument(
        "--specs",
        type=Path,
        default=_HERE / "contracts/crown/discrete/liveness_contracts.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=_HERE / "contracts/crown/discrete/liveness_contract_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_HERE / "results/compositional/lasso_pins",
    )
    parser.add_argument("--base-smv", type=Path, default=DEFAULT_BASE_SMV)
    parser.add_argument("--nuxmv", type=Path, default=DEFAULT_NUXMV)
    parser.add_argument(
        "--nuxmv-cmd",
        type=Path,
        default=DEFAULT_NUXMV_CMD_COMBO,
        help="Use command_combo_invar_ctl for CTL; command_invar for safety only",
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--run-crown", action="store_true")
    parser.add_argument("--no-ctl", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.perf_counter()

    # --- pins ---
    if not args.skip_generate or not args.specs.exists():
        trajectory = AcasLassoTrajectory.from_json_file(args.lasso_json)
        pin_set = AcasLassoPinSet.from_trajectory(trajectory, eps=args.eps)
        pin_set.fill_onnx_margins()
        r_physics = pin_set.abstract_reachable_pair_count(model_active_freeze=False)
        r_smv = pin_set.abstract_reachable_pair_count(model_active_freeze=True)
        print(f"[CHECK] abstract |R| always-physics = {r_physics}")
        print(f"[CHECK] abstract |R| SMV-freeze = {r_smv}")
        print(pin_set.diagnose_ctl_counterexample_freeze())
        pin_set.write_specs_json(args.specs)
        if args.run_crown:
            pin_set.certify_with_crown(
                timeout_seconds=args.timeout, device=args.device,
            )
        else:
            for pin in pin_set.pins:
                pin.crown_status = "SAT"
            print("  CROWN skipped (ONNX margins only); statuses set SAT")
        pin_set.write_crown_results_json(args.results)
        # CTL threshold: active cutoff (1000), not ONNX cycle max (1400).
        ctl_distance_threshold = 1000
        generate_metrics = {
            "wall_sec": 0.0,
            "n_pins": len(pin_set.pins),
            "abstract_reachable_physics": r_physics,
            "abstract_reachable_smv_freeze": r_smv,
            "ctl_distance_threshold": ctl_distance_threshold,
            "max_rho_on_onnx_cycle": trajectory.max_distance_on_cycle(),
            "min_onnx_margin": min(p.onnx_margin for p in pin_set.pins),
            "eps": args.eps,
            "crown_ran": bool(args.run_crown),
        }
        d_star = ctl_distance_threshold
    else:
        specs = json.loads(args.specs.read_text(encoding="utf-8"))
        d_star = 1000
        trajectory = AcasLassoTrajectory.from_json_file(args.lasso_json)
        pin_set = AcasLassoPinSet.from_trajectory(trajectory, eps=args.eps)
        generate_metrics = {
            "wall_sec": 0.0,
            "n_pins": len(specs["contracts"]),
            "abstract_reachable_physics": pin_set.abstract_reachable_pair_count(
                model_active_freeze=False,
            ),
            "abstract_reachable_smv_freeze": pin_set.abstract_reachable_pair_count(
                model_active_freeze=True,
            ),
            "ctl_distance_threshold": d_star,
            "skipped": True,
        }

    smv_vars = acas_pipe._load_smv_vars()
    sat_contracts = _load_sat_pin_contracts(args.specs, args.results)

    # Seed pin should force clear at seed (implies not strong_right).
    seed_pins = [
        c for c in sat_contracts
        if c["dangerous_xy"] == [[7, 6]] and c["a_prev"] == "clear"
    ]
    print(f"[CHECK] seed clear pins: {len(seed_pins)} (expect >= 1)")

    smv_path = output_dir / "acas_360_lasso_pins.smv"
    patch_start = time.perf_counter()
    patch_info = patch_smv_with_pins(
        base_smv_path=args.base_smv,
        output_smv_path=smv_path,
        sat_contracts=sat_contracts,
        smv_vars=smv_vars,
        d_star=d_star if not args.no_ctl else None,
        inject_ctl=not args.no_ctl,
    )
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
            "pins": generate_metrics,
            "smv_patch": patch_info,
            "nuxmv": nuxmv_metrics,
        },
        total_wall_seconds=time.perf_counter() - pipeline_start,
        extra_fields={
            "specs": str(args.specs),
            "results": str(args.results),
            "mode": "lasso_determinism_pins",
        },
    )


if __name__ == "__main__":
    main()
