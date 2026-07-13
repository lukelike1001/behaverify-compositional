#!/usr/bin/env bash
#
# run_acas_corridor_pipeline.sh
#
# Reproduces the inductive-invariant corridor result end-to-end: discovery (viability
# kernel + corridor trace, no networks) -> CROWN certificate for the one operative
# contract (Q2) -> SMV patch with that single INVAR line -> nuXmv. See
# reports/Acas_Xu_closed_loop/2026_07_12_inductive_invariant_stress_test.md, Section 8.
#
# Stages:
#   1. [DISCOVERY]      acas_inductive_analysis.py -- viability kernel, corridor,
#                        candidate-injection checks
#   2. [CROWN]           verify_single_state.py -- formal certificate for the seed
#                        contract (Q2); re-verifies contracts/crown/corridor_crown_results.json
#   3. [PATCH+VERIFY]    run_acas_compositional_pipeline.py -- inject the 1 INVAR line,
#                        run nuXmv
#
# Usage (from AcasXu_closed_loop/):
#   ./run_acas_corridor_pipeline.sh
#   ./run_acas_corridor_pipeline.sh --nuxmv /path/to/nuXmv
#
# Output:
#   results/compositional/discrete_goals/corridor/pipeline_report.json
#
# Prerequisites:
#   pip install -e .   (alpha-beta-CROWN, see root README.md)
#   nuXmv binary at ../../nuXmv_DL/bin/nuXmv (or override with --nuxmv)

set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUXMV="${_HERE}/../../nuXmv_DL/bin/nuXmv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nuxmv) NUXMV="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

PYTHON="${PYTHON:-python3}"
cd "${_HERE}"

echo "========================================"
echo "[1/3] DISCOVERY (viability kernel, corridor trace)"
echo "========================================"
"${PYTHON}" acas_inductive_analysis.py

echo ""
echo "========================================"
echo "[2/3] CROWN CERTIFICATE (Q2, the operative contract)"
echo "========================================"
"${PYTHON}" verify_single_state.py \
    --advisory clear --forbidden strong_right \
    --x_mag 7 --y_mag 6 --x_sign 1 --y_sign 1 --heading 10

echo ""
echo "========================================"
echo "[3/3] SMV PATCH + NUXMV"
echo "========================================"
"${PYTHON}" run_acas_compositional_pipeline.py \
    --contracts contracts/crown/corridor_crown_results.json \
    --spec      contracts/crown/corridor_contracts.json \
    --output    results/compositional/discrete_goals/corridor \
    --nuxmv     "${NUXMV}" \
    --skip-tree --skip-smv

echo ""
echo "Done. Report: results/compositional/discrete_goals/corridor/pipeline_report.json"
