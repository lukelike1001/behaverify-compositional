"""
run_acas_compositional_pipeline.py

End-to-end compositional verification pipeline for the ACAS Xu 5-NN closed-loop NSBT.

Stages:
  1. [TREE]     Generate acas_360.tree from acas_template_360.tree via generate_acas_tree.py
  2. [SMV]      Convert .tree → base nuXmv SMV via dsl_to_nuxmv.py
  3. [PATCH]    Replace 5 NN lookup-table DEFINE blocks with non-deterministic VAR +
                INVAR constraints derived from the verified A/G contracts JSON
  4. [VERIFY]   Run nuXmv to check INVARSPEC (distance >= 200)
                Delegates to pipeline/symbolic/nuxmv/nuxmv_verifier.py (NuxmvVerifier)
  5. [REPORT]   Write JSON report with per-step timing and verdicts
                Delegates to pipeline/pipeline_report_writer.py (PipelineReportWriter)

SMV variable names are read from acas_verifier_params.yaml (smv_variables section)
rather than hardcoded in this script.

SMV file locations:
  Base SMV  : symbolic/smv/acas_360.smv   (generated once, reused with --skip-smv)
  Patched SMV: <output_dir>/acas_360_contracts.smv

Usage (from AcasXu_closed_loop/):
  python run_acas_compositional_pipeline.py \\
      --contracts contracts/crown/continuous/enabled_pgd/aprev_clear_crown_results.json \\
      --output    results/continuous/enabled_pgd/aprev_clear \\
      [--nuxmv    ../../nuXmv_DL/bin/nuXmv] \\
      [--nuxmv-cmd ../../commands/nuxmv_commands/command_invar] \\
      [--skip-tree]   # reuse existing tree/acas_360.tree
      [--skip-smv]    # reuse existing symbolic/smv/acas_360.smv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_EXAMPLE = _SCRIPTS.parent
_HERE = _EXAMPLE  # AcasXu_closed_loop root (compat alias)
_TBA  = (_HERE / "../../").resolve()

if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from pipeline.symbolic.nuxmv.nuxmv_verifier import NuxmvVerifier
from pipeline.pipeline_report_writer import PipelineReportWriter
from pipeline.process_memory import ProcessMemory

from core.acas_smv_contract_patcher import AcasSmvContractPatcher

try:
    import resource as _resource
except ImportError:
    _resource = None  # Windows

DEFAULT_NUXMV     = _HERE / "../../nuXmv_DL/bin/nuXmv"
DEFAULT_NUXMV_CMD = _HERE / "../../commands/nuxmv_commands/command_invar"
DEFAULT_METAMODEL = _HERE / "../../metamodel/behaverify.tx"
DEFAULT_SRC       = _HERE / "../../src"
DEFAULT_CONFIG    = _HERE / "core" / "acas_verifier_params.yaml"


# ---------------------------------------------------------------------------
# SMV variable names (read from config)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 1 — Tree generation
# ---------------------------------------------------------------------------

def run_tree_generation(ctx: dict) -> dict:
    import subprocess
    print("\n" + "=" * 60)
    print("[1/4] TREE GENERATION")
    print("=" * 60)

    if ctx["skip_tree"] and ctx["tree_path"].exists():
        print(f"  Skipped — reusing {ctx['tree_path']}")
        return {"wall_sec": 0.0, "skipped": True}

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(_HERE / "generate_acas_tree.py")],
        cwd=str(_HERE), capture_output=True, text=True, check=False,
    )
    wall_sec = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  ERROR: generate_acas_tree.py failed:\n{result.stderr}")
        raise RuntimeError("Tree generation failed")

    print(f"  Generated {ctx['tree_path']}  ({wall_sec:.1f}s)")
    return {"wall_sec": round(wall_sec, 3), "skipped": False}


# ---------------------------------------------------------------------------
# Step 2 — Base SMV generation
# ---------------------------------------------------------------------------

def run_smv_generation(ctx: dict) -> dict:
    print("\n" + "=" * 60)
    print("[2/4] BASE SMV GENERATION")
    print("=" * 60)

    if ctx["skip_smv"] and ctx["base_smv_path"].exists():
        print(f"  Skipped — reusing {ctx['base_smv_path']}")
        return {"wall_sec": 0.0, "skipped": True}

    src_dir = str(DEFAULT_SRC.resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import dsl_to_nuxmv as _dsl  # noqa: PLC0415

    tracemalloc.start()
    t0 = time.perf_counter()

    _orig_cwd = os.getcwd()
    os.chdir(str(ctx["tree_path"].parent))
    try:
        _dsl.dsl_to_nuxmv(
            str(ctx["metamodel"]),
            str(ctx["tree_path"]),
            str(ctx["base_smv_path"]),
            False, False, False, False,
            10000, False, True, None,
        )
    finally:
        os.chdir(_orig_cwd)

    wall_sec = time.perf_counter() - t0
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = ProcessMemory.peak_self_rss_kilobytes()

    smv_lines = ctx["base_smv_path"].read_text().count("\n")
    print(f"  Generated {ctx['base_smv_path']}  ({wall_sec:.1f}s, {smv_lines} lines)")
    return {
        "wall_sec":          round(wall_sec, 3),
        "peak_rss_kb":       rss,
        "peak_traced_bytes": peak_traced,
        "smv_lines":         smv_lines,
        "skipped":           False,
    }


# ---------------------------------------------------------------------------
# Step 3 — SMV patching (AcasSmvContractPatcher)
# ---------------------------------------------------------------------------

def run_smv_patch(ctx: dict, patcher: AcasSmvContractPatcher) -> dict:
    print("\n" + "=" * 60)
    print("[3/4] SMV PATCHING (contract injection)")
    print("=" * 60)

    # --contracts is CROWN results; --spec is original specs (IDs must align).
    sat_contracts = AcasSmvContractPatcher.load_sat_contracts(
        ctx["spec_path"],
        ctx["contracts_path"],
    )
    metrics = patcher.patch_file(
        ctx["base_smv_path"],
        ctx["smv_path"],
        sat_contracts,
    )
    print(f"  Removed 5 NN DEFINE blocks ({metrics['nn_lines_removed']} lines)")
    print("  Replaced NN table outputs with non-deterministic advisory domain")
    print(
        f"  Injected {metrics['invar_lines']} INVAR constraints from "
        f"{metrics['sat_contracts']} SAT contracts"
    )
    print(f"  Patched SMV: {ctx['smv_path']}  ({metrics['wall_sec']:.1f}s)")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="End-to-end compositional verification pipeline for ACAS Xu 5-NN NSBT."
    )
    p.add_argument("--contracts",  required=True,
                   help="Path to verified contracts JSON (e.g. contracts/crown/continuous/enabled_pgd/aprev_clear_crown_results.json)")
    p.add_argument("--spec",       default="contracts/crown/discrete/safety/safety_full_contracts.json",
                   help="Path to original contract spec JSON (default: contracts/crown/discrete/safety/safety_full_contracts.json)")
    p.add_argument("--output",     required=True,
                   help="Output directory for patched SMV, nuXmv output, and report")
    p.add_argument("--nuxmv",      default=str(DEFAULT_NUXMV),
                   help=f"nuXmv binary path (default: {DEFAULT_NUXMV})")
    p.add_argument("--nuxmv-cmd",  default=str(DEFAULT_NUXMV_CMD), dest="nuxmv_cmd",
                   help=f"nuXmv command file (default: {DEFAULT_NUXMV_CMD})")
    p.add_argument("--metamodel",  default=str(DEFAULT_METAMODEL),
                   help=f"behaverify.tx path (default: {DEFAULT_METAMODEL})")
    p.add_argument("--config",     default=str(DEFAULT_CONFIG),
                   help=f"Config YAML for SMV variable names (default: {DEFAULT_CONFIG})")
    p.add_argument("--skip-tree",  action="store_true",
                   help="Skip tree generation; reuse tree/acas_360.tree if it exists")
    p.add_argument("--skip-smv",   action="store_true",
                   help="Skip base SMV generation; reuse symbolic/smv/acas_360.smv if it exists")
    args = p.parse_args()

    patcher = AcasSmvContractPatcher.from_verifier_yaml(Path(args.config))
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure tree/ and symbolic/smv/ dirs exist
    (_HERE / "tree").mkdir(exist_ok=True)
    (_HERE / "symbolic" / "smv").mkdir(parents=True, exist_ok=True)

    ctx = {
        "contracts_path":  Path(args.contracts).resolve(),
        "spec_path":       Path(args.spec).resolve(),
        "tree_path":       _HERE / "tree" / "acas_360.tree",
        "base_smv_path":   _HERE / "symbolic" / "smv" / "acas_360.smv",
        "smv_path":        output_dir / "acas_360_contracts.smv",   # patched SMV
        "nuxmv_out_path":  output_dir / "nuxmv_output.txt",
        "report_path":     output_dir / "pipeline_report.json",
        "metamodel":       Path(args.metamodel).resolve(),
        "nuxmv_bin":       Path(args.nuxmv).resolve(),
        "nuxmv_cmd":       Path(args.nuxmv_cmd).resolve(),
        "skip_tree":       args.skip_tree,
        "skip_smv":        args.skip_smv,
    }

    t_start = time.perf_counter()

    tree_metrics = run_tree_generation(ctx)
    smv_metrics = run_smv_generation(ctx)
    patch_metrics = run_smv_patch(ctx, patcher)
    nuxmv_metrics = NuxmvVerifier.from_pipeline_context(ctx).run_and_collect_metrics()

    total = time.perf_counter() - t_start

    PipelineReportWriter.from_path(ctx["report_path"]).write(
        steps={
            "tree": tree_metrics,
            "smv": smv_metrics,
            "smv_patch": patch_metrics,
            "nuxmv": nuxmv_metrics,
        },
        total_wall_seconds=total,
        extra_fields={"contracts_path": str(ctx["contracts_path"])},
    )


if __name__ == "__main__":
    main()
