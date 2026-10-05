#!/usr/bin/env bash
# KILL CONDITION 1: does the fixed-point encoding escape the table's input-space
# ceiling? If it does, "the table cannot scale" is not a contribution -- a
# reviewer switches encoding and our discrete direction dies tonight.
#
# NeuS Table 3 measured both on the SMALL grid: table 0.07s flat across 100-300
# neurons, fixed-point 53-153s and growing with network size. Nobody has
# measured how either behaves as the INPUT SPACE grows, which is the axis that
# matters for us.
set -u
cd "$(dirname "$0")/../../.." || exit 1
OUT=COLLAB/experiments/encoding_scaling/results.txt
NUXMV=../nuXmv
echo "encoding scaling | $(date -u +%Y-%m-%dT%H:%M:%SZ) | host $(hostname)" > "$OUT"
for t in REPRODUCIBILITY/2024_ANSR/examples/small/table.tree \
         REPRODUCIBILITY/2024_ANSR/examples/small/fixed.tree \
         REPRODUCIBILITY/2024_ANSR/examples/small/float.tree; do
  name=$(basename "$t" .tree)
  echo "=== $name ===" >> "$OUT"
  grep -c "" "$t" >> "$OUT" 2>&1
  /usr/bin/time -v python -m behaverify nuxmv "$t" /tmp/enc_$name \
      --generate --overwrite --recursion_limit 20000 >> "$OUT" 2>&1
  echo "--- smv size ---" >> "$OUT"
  ls -la /tmp/enc_$name/ >> "$OUT" 2>&1
done
echo "DONE" >> "$OUT"
