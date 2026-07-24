"""
scripts/run_acas_monolithic_pipeline.py

Monolithic verification of the ACAS Xu closed-loop SMV (networks left in the
model — no contract INVAR injection).

Stages:
  1. [TREE]  Generate tree/acas_closed_loop.tree if missing (AcasTreeGenerator)
  2. [SMV]   Generate symbolic/smv/acas_closed_loop.smv if missing (dsl_to_nuxmv)
  3. [NUXMV] Run nuXmv all-invar on the base SMV, or load the 2025_NEUS
             reference with --skip-monolithic (~9.6 GB RAM if run live)

Output (under results/monolithic/):
  nuxmv_output.txt      — nuXmv log (live run only)
  pipeline_report.json  — timing + INVARSPEC verdict

Usage (from AcasXu_closed_loop/):

    python3 scripts/run_acas_monolithic_pipeline.py
    python3 scripts/run_acas_monolithic_pipeline.py --skip-monolithic
    python3 scripts/run_acas_monolithic_pipeline.py --nuxmv /path/to/nuXmv

WARNING: a full monolithic nuXmv run needs ~12 GB free RAM. Prefer
--skip-monolithic on smaller machines (uses 2025_NEUS invar.txt).
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
_EXAMPLE = _SCRIPTS.parent
_TBA = (_EXAMPLE / "../..").resolve()

if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from pipeline.pipeline_report_writer import PipelineReportWriter  # noqa: E402
from pipeline.nuxmv_verifier import NuxmvVerifier  # noqa: E402

from core.acas_artifact_builder import AcasArtifactBuilder  # noqa: E402

DEFAULT_NUXMV = _TBA / "nuXmv_DL/bin/nuXmv"
DEFAULT_NUXMV_CMD = _TBA / "commands/nuxmv_commands/command_all_invar"
DEFAULT_NEUS_REFERENCE = (
    _EXAMPLE / "../../../2025_NEUS/examples/AcasXu_closed_loop/invar.txt"
).resolve()
DEFAULT_OUTPUT_DIR = _EXAMPLE / "results/monolithic"
DEFAULT_TREE = _EXAMPLE / "tree/acas_closed_loop.tree"
DEFAULT_SMV = _EXAMPLE / "symbolic/smv/acas_closed_loop.smv"


def _parse_neus_reference(reference_path: Path) -> dict[str, Any]:
    text = reference_path.read_text(encoding="utf-8", errors="replace")

    invar_match = re.search(r"\[Invar\s+(True|False)\b", text, re.IGNORECASE)
    invarspec = invar_match.group(1).lower() if invar_match else None

    user_match = re.search(r"User time\s+(\S+)\s+seconds", text)
    user_sec = float(user_match.group(1)) if user_match else None

    rss_match = re.search(r"Maximum resident size\s+=\s+(\d+)\s*K", text)
    peak_rss_kb = int(rss_match.group(1)) if rss_match else None

    return {
        "wall_sec": round(user_sec, 3) if user_sec is not None else None,
        "user_sec": round(user_sec, 3) if user_sec is not None else None,
        "peak_rss_kb": peak_rss_kb,
        "invarspec": invarspec,
        "source": str(reference_path),
        "skipped_live_nuxmv": True,
    }


def _run_live_nuxmv(
    *,
    nuxmv_bin: Path,
    nuxmv_cmd: Path,
    smv_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("[3/3] MONOLITHIC NUXMV")
    print("=" * 60)
    print(f"  Model : {smv_path}")
    print("  WARNING: expects ~9.6 GB peak RSS; use --skip-monolithic if tight on RAM.")

    if not nuxmv_bin.is_file():
        raise FileNotFoundError(f"nuXmv binary not found: {nuxmv_bin}")
    if not nuxmv_cmd.is_file():
        raise FileNotFoundError(f"nuXmv command file not found: {nuxmv_cmd}")
    if not smv_path.is_file():
        raise FileNotFoundError(f"SMV model not found: {smv_path}")

    ctx = {
        "nuxmv_bin": nuxmv_bin.resolve(),
        "nuxmv_cmd": nuxmv_cmd.resolve(),
        "smv_path": smv_path.resolve(),
        "nuxmv_out_path": log_path.resolve(),
    }
    metrics = NuxmvVerifier.from_pipeline_context(ctx).run_and_collect_metrics()
    metrics["skipped_live_nuxmv"] = False
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nuxmv",
        type=Path,
        default=DEFAULT_NUXMV,
        help="Path to nuXmv binary",
    )
    parser.add_argument(
        "--nuxmv-cmd",
        type=Path,
        default=DEFAULT_NUXMV_CMD,
        help="nuXmv -source command file (default: command_all_invar)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: results/monolithic)",
    )
    parser.add_argument(
        "--neus-reference",
        type=Path,
        default=DEFAULT_NEUS_REFERENCE,
        help="2025_NEUS invar.txt used with --skip-monolithic",
    )
    parser.add_argument(
        "--skip-monolithic",
        action="store_true",
        help="Do not run nuXmv; load verdict/timing from 2025_NEUS reference",
    )
    args = parser.parse_args()

    output_dir = args.output
    if not output_dir.is_absolute():
        output_dir = (_EXAMPLE / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_start = time.perf_counter()
    # Generate tree/SMV only when missing (same as the old shell script).
    builder = AcasArtifactBuilder(tree_path=DEFAULT_TREE, smv_path=DEFAULT_SMV)
    tree_metrics = builder.ensure_tree(
        reuse_existing=True, stage_label="[1/3] TREE GENERATION"
    )
    smv_metrics = builder.ensure_smv(
        reuse_existing=True, stage_label="[2/3] BASE SMV GENERATION"
    )

    report_path = output_dir / "pipeline_report.json"
    log_path = output_dir / "nuxmv_output.txt"

    if args.skip_monolithic:
        print("\n" + "=" * 60)
        print("[3/3] MONOLITHIC NUXMV (2025_NEUS reference)")
        print("=" * 60)
        reference = args.neus_reference
        if not reference.is_file():
            raise FileNotFoundError(
                f"--skip-monolithic requires reference file: {reference}"
            )
        nuxmv_metrics = _parse_neus_reference(reference)
        print(f"  INVARSPEC : {nuxmv_metrics.get('invarspec')}")
        if nuxmv_metrics.get("user_sec") is not None:
            print(f"  User time : {nuxmv_metrics['user_sec']:.1f}s")
        if nuxmv_metrics.get("peak_rss_kb") is not None:
            print(
                f"  Peak RSS  : "
                f"{nuxmv_metrics['peak_rss_kb'] / 1024**2:.2f} GB"
            )
        print(f"  Source    : {reference}")
    else:
        nuxmv_metrics = _run_live_nuxmv(
            nuxmv_bin=args.nuxmv,
            nuxmv_cmd=args.nuxmv_cmd,
            smv_path=DEFAULT_SMV,
            log_path=log_path,
        )

    total_wall = time.perf_counter() - pipeline_start
    PipelineReportWriter.from_path(report_path).write(
        steps={
            "tree": tree_metrics,
            "smv": smv_metrics,
            "nuxmv": nuxmv_metrics,
        },
        total_wall_seconds=total_wall,
        extra_fields={
            "mode": "monolithic",
            "smv_path": str(DEFAULT_SMV.relative_to(_EXAMPLE)),
            "output_dir": str(output_dir),
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            **(
                {"source": nuxmv_metrics["source"]}
                if nuxmv_metrics.get("source")
                else {}
            ),
        },
    )
    print(f"\nDone. Report: {report_path}")


if __name__ == "__main__":
    main()
