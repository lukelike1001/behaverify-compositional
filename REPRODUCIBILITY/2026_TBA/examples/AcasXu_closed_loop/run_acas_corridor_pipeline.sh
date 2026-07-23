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
#   1. [DISCOVERY]      scripts/acas_check_inductive_invariant.py -- viability kernel, corridor,
#                        candidate-injection checks
#   2. [CROWN]           acas_safety_contract_verifier.py --point -- formal certificate for the seed
#                        contract (Q2); re-verifies corridor operative state
#   3. [PATCH+VERIFY]    scripts/run_acas_compositional_pipeline.py -- inject the 1 INVAR line,
#                        run nuXmv
#
# Usage (from AcasXu_closed_loop/):
#   ./run_acas_corridor_pipeline.sh
#   ./run_acas_corridor_pipeline.sh --nuxmv /path/to/nuXmv
#
# Output:
#   results/discrete/safety/corridor/pipeline_report.json
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
"${PYTHON}" scripts/acas_check_inductive_invariant.py

echo ""
echo "========================================"
echo "[2/3] CROWN CERTIFICATE (Q2, the operative contract)"
echo "========================================"
"${PYTHON}" acas_safety_contract_verifier.py \
    --point \
    --advisory clear --forbidden strong_right \
    --x_mag 7 --y_mag 6 --x_sign 1 --y_sign 1 --heading 10

echo ""
echo "========================================"
echo "[3/3] SMV PATCH + NUXMV"
echo "========================================"
"${PYTHON}" scripts/run_acas_compositional_pipeline.py \
    --contracts contracts/crown/discrete/safety/safety_corridor_contract_results.json \
    --spec      contracts/crown/discrete/safety/safety_corridor_contracts.json \
    --output    results/discrete/safety/corridor \
    --nuxmv     "${NUXMV}" \
    --skip-tree --skip-smv

echo ""
echo "Done. Report: results/discrete/safety/corridor/pipeline_report.json"
