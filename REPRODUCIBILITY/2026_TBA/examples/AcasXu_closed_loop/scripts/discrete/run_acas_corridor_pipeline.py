"""
scripts/discrete/run_acas_corridor_pipeline.py

Reproduce the inductive-invariant corridor result end-to-end (report Section 8):

  1. [DISCOVERY]  run_acas_inductive_invariant_check.py
                  (R, V, corridor, candidate-injection CHECKs — no networks)
  2. [CROWN]      point certificate for Q2 at the seed
                  (a_prev=clear forbids strong_right at (7,6,+,+,h=10))
  3. [PATCH+NUXMV] run_acas_safety_pipeline.py with pre-verified corridor JSON

Usage (from AcasXu_closed_loop/):

    python3 scripts/discrete/run_acas_corridor_pipeline.py
    python3 scripts/discrete/run_acas_corridor_pipeline.py --nuxmv /path/to/nuXmv

Output:
    results/discrete/safety/corridor/pipeline_report.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DISCRETE = Path(__file__).resolve().parent
_EXAMPLE = _SCRIPTS_DISCRETE.parent.parent
_TBA = (_EXAMPLE / "../..").resolve()

DEFAULT_NUXMV = _TBA / "nuXmv_DL/bin/nuXmv"
DEFAULT_CORRIDOR_SPECS = (
    _EXAMPLE / "contracts/crown/discrete/safety/safety_corridor_contracts.json"
)
DEFAULT_CORRIDOR_RESULTS = (
    _EXAMPLE
    / "contracts/crown/discrete/safety/safety_corridor_contract_results.json"
)
DEFAULT_OUTPUT = _EXAMPLE / "results/discrete/safety/corridor"


def _run(label: str, command: list[str]) -> None:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print("  $ " + " ".join(command))
    completed = subprocess.run(command, cwd=str(_EXAMPLE), check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"{label} failed with exit code {completed.returncode}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nuxmv",
        type=Path,
        default=DEFAULT_NUXMV,
        help="Path to nuXmv binary",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for child scripts",
    )
    args = parser.parse_args()

    python = args.python
    inductive = _SCRIPTS_DISCRETE / "run_acas_inductive_invariant_check.py"
    verifier = (
        _EXAMPLE / "core/safety/acas_safety_contract_verifier.py"
    )
    safety_pipeline = _SCRIPTS_DISCRETE / "run_acas_safety_pipeline.py"

    _run(
        "[1/3] DISCOVERY (viability kernel, corridor trace)",
        [python, str(inductive)],
    )

    _run(
        "[2/3] CROWN CERTIFICATE (Q2, operative seed contract)",
        [
            python,
            str(verifier),
            "--point",
            "--advisory", "clear",
            "--forbidden", "strong_right",
            "--x_mag", "7",
            "--y_mag", "6",
            "--x_sign", "1",
            "--y_sign", "1",
            "--heading", "10",
        ],
    )

    _run(
        "[3/3] SMV PATCH + NUXMV (corridor SAT contracts)",
        [
            python,
            str(safety_pipeline),
            "--specs", str(DEFAULT_CORRIDOR_SPECS.relative_to(_EXAMPLE)),
            "--results", str(DEFAULT_CORRIDOR_RESULTS.relative_to(_EXAMPLE)),
            "--output", str(DEFAULT_OUTPUT.relative_to(_EXAMPLE)),
            "--nuxmv", str(args.nuxmv),
            "--skip-contracts",
            "--skip-tree",
            "--skip-smv",
        ],
    )

    report = DEFAULT_OUTPUT / "pipeline_report.json"
    print(f"\nDone. Report: {report.relative_to(_EXAMPLE)}")


if __name__ == "__main__":
    main()
