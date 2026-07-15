"""
pipeline.pipeline_report_writer — JSON report + console summary for E2E runs.

Example-agnostic: step names and extra fields are supplied by the caller
(grid world, ACAS Xu, or a future NSBT pipeline).
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineReportWriter:
    """
    Writes pipeline_report.json and prints a short timing/verdict table.

    Holds only the destination path; step metrics are passed into write(...).
    """

    report_path: Path

    @classmethod
    def from_path(cls, report_path: Path | str) -> PipelineReportWriter:
        return cls(report_path=Path(report_path))

    def write(
        self,
        steps: dict[str, dict[str, Any]],
        total_wall_seconds: float,
        extra_fields: dict[str, Any] | None = None,
    ) -> str:
        """
        Serialise the report and print a summary.

        The first step metrics dict that contains "invarspec" supplies the
        verdict (INVAR / optional CTL).

        Returns the verdict string written into the JSON.
        """
        invar_verdict = None
        ctl_verdict = None
        for step_metrics in steps.values():
            if "invarspec" in step_metrics:
                invar_verdict = step_metrics["invarspec"]
                ctl_verdict = step_metrics.get("ctlspec")
                break

        if ctl_verdict is not None:
            verdict = f"INVAR={invar_verdict} CTL={ctl_verdict}"
        else:
            verdict = f"INVAR={invar_verdict}"

        report_body: dict[str, Any] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "steps": steps,
            "total_wall_sec": round(total_wall_seconds, 3),
            "verdict": verdict,
            **(extra_fields or {}),
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(report_body, indent=2), encoding="utf-8",
        )
        self._print_summary(steps, verdict, total_wall_seconds)
        return verdict

    def _print_summary(
        self,
        steps: dict[str, dict[str, Any]],
        verdict: str,
        total_wall_seconds: float,
    ) -> None:
        timing_parts = [
            f"{step_name}={step_metrics.get('wall_sec', 0.0):.1f}s"
            for step_name, step_metrics in steps.items()
        ]
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"  Timing  : {' | '.join(timing_parts)}")
        print(f"            total={total_wall_seconds:.1f}s")
        print(f"  Verdict : {verdict}")
        print(f"  Report  : {self.report_path}")
        print("=" * 60)
