"""
pipeline.symbolic.nuxmv.nuxmv_verifier — shared nuXmv subprocess adapter.

Runs nuXmv on an SMV model with a command file, writes combined stdout/stderr,
parses INVARSPEC / CTLSPEC verdicts, and returns pipeline metrics.

Example pipelines build paths (binary, command file, SMV, log) and call:

    NuxmvVerifier.from_pipeline_context(ctx).run_and_collect_metrics()

This module has no example-specific physics or contract logic.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.process_memory import ProcessMemory

_INVARIANT_VERDICT_PATTERN = re.compile(r"-- invariant .+ is (true|false)")
_SPECIFICATION_VERDICT_PATTERN = re.compile(r"-- specification .+ is (true|false)")


@dataclass
class NuxmvVerifier:
    """
    Thin adapter around the nuXmv binary for one SMV + command-file run.

    Holds the four paths needed for a symbolic check and records wall time,
    approximate child RSS delta, return code, and parsed verdicts.
    """

    nuxmv_binary_path: Path
    nuxmv_command_file_path: Path
    smv_model_path: Path
    output_log_path: Path

    @classmethod
    def from_pipeline_context(cls, pipeline_context: dict[str, Any]) -> NuxmvVerifier:
        """
        Build from the shared pipeline ctx dict used by grid world and ACAS.

        Required keys:
            nuxmv_bin       — nuXmv executable
            nuxmv_cmd       — command file (-source)
            smv_path        — SMV model to check
            nuxmv_out_path  — where to write combined stdout+stderr
        """
        return cls(
            nuxmv_binary_path=Path(pipeline_context["nuxmv_bin"]),
            nuxmv_command_file_path=Path(pipeline_context["nuxmv_cmd"]),
            smv_model_path=Path(pipeline_context["smv_path"]),
            output_log_path=Path(pipeline_context["nuxmv_out_path"]),
        )

    def parse_invar_and_ctl_verdicts(
        self, nuxmv_output_text: str,
    ) -> dict[str, str | None]:
        """Extract INVARSPEC and CTLSPEC true/false lines from nuXmv output."""
        invariant_match = _INVARIANT_VERDICT_PATTERN.search(nuxmv_output_text)
        specification_match = _SPECIFICATION_VERDICT_PATTERN.search(nuxmv_output_text)
        return {
            "invarspec": (
                invariant_match.group(1) if invariant_match is not None else None
            ),
            "ctlspec": (
                specification_match.group(1) if specification_match is not None else None
            ),
        }

    def build_nuxmv_command(self) -> list[str]:
        """nuXmv argv: binary -source <cmd> <smv>."""
        return [
            str(self.nuxmv_binary_path),
            "-source",
            str(self.nuxmv_command_file_path),
            str(self.smv_model_path),
        ]

    def run_and_collect_metrics(self) -> dict[str, Any]:
        """
        Run nuXmv, write the log file, and return metrics for pipeline_report.json.

        Metrics keys:
            wall_sec, peak_rss_kb, returncode, invarspec, ctlspec
        """
        print("\n" + "=" * 60)
        print("[nuXmv] SYMBOLIC VERIFICATION")
        print("=" * 60)

        command = self.build_nuxmv_command()
        print(f"  Command: {' '.join(command)}")

        rss_before_kb = ProcessMemory.peak_children_rss_kilobytes()
        start_time = time.perf_counter()

        completed_process = subprocess.run(
            command, capture_output=True, text=True, check=False,
        )

        wall_seconds = time.perf_counter() - start_time
        rss_after_kb = ProcessMemory.peak_children_rss_kilobytes()

        combined_output_text = completed_process.stdout + completed_process.stderr
        self.output_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_log_path.write_text(combined_output_text, encoding="utf-8")

        verdicts = self.parse_invar_and_ctl_verdicts(combined_output_text)
        metrics: dict[str, Any] = {
            "wall_sec": round(wall_seconds, 3),
            "peak_rss_kb": rss_after_kb - rss_before_kb,
            "returncode": completed_process.returncode,
            **verdicts,
        }
        print(
            f"\n  [{wall_seconds:.1f}s]  INVARSPEC={verdicts['invarspec']}  "
            f"CTLSPEC={verdicts['ctlspec']}"
        )
        print(f"  Output saved to: {self.output_log_path}")
        return metrics
