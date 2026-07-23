#!/usr/bin/env bash
# run_all_continuous_pipelines.sh
#
# Run the compositional pipeline for every verified-contracts JSON in a given folder.
#
# Usage (from REPRODUCIBILITY/2026_TBA/examples/AcasXu_closed_loop/):
#
#   ./run_all_continuous_pipelines.sh                                               # defaults
#   ./run_all_continuous_pipelines.sh contracts/crown/continuous/enabled_pgd/
#   ./run_all_continuous_pipelines.sh contracts/crown/continuous/disabled_pgd/
#   ./run_all_continuous_pipelines.sh contracts/crown/discrete/
#
# Outputs go to:
#   results/<relative-path-under-crown>/<nn_stem>/pipeline_report.json
#
# The output path mirrors the contracts/crown/ structure under results/:
#   contracts/crown/continuous/enabled_pgd/ -> results/continuous/enabled_pgd/
#   contracts/crown/discrete/safety/        -> results/discrete/safety/

set -euo pipefail

CONTRACTS_DIR="${1:-contracts/crown/continuous/enabled_pgd}"

# Strip leading "contracts/crown/" to get the relative subfolder, then mirror under results/
RELATIVE="${CONTRACTS_DIR#contracts/crown/}"
RELATIVE="${RELATIVE%/}"  # strip trailing slash if present
OUTPUT_BASE="results/${RELATIVE}"

echo "Contracts folder : ${CONTRACTS_DIR}"
echo "Output base      : ${OUTPUT_BASE}/"
echo ""

for CONTRACT_JSON in "${CONTRACTS_DIR}"/*.json; do
    # Strip _crown_results suffix so aprev_clear_crown_results.json -> aprev_clear
    STEM="$(basename "${CONTRACT_JSON}" .json)"
    NN_STEM="${STEM%_crown_results}"
    OUTPUT="${OUTPUT_BASE}/${NN_STEM}"

    echo "========================================"
    echo "Network : ${NN_STEM}"
    echo "Output  : ${OUTPUT}"
    echo "========================================"

    python3 run_acas_compositional_pipeline.py \
        --contracts  "${CONTRACT_JSON}" \
        --output     "${OUTPUT}" \
        --skip-tree \
        --skip-smv
    echo ""
done

echo "All done. Reports in ${OUTPUT_BASE}/"
