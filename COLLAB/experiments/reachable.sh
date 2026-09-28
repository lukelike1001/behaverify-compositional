#!/bin/bash
# Reachable-state-space measurement — the evidence for the negative result.
#
# These two models carry the conclusion: behaviour trees DO have large reachable
# state spaces, and monolithic BehaVerify verifies them anyway. That is why
# compositional verification has nothing to contribute here.
#
# The .tree files are already tracked in the repo and are NOT copied or modified.
# Only the measurement lives here.
#
# GOTCHA that cost an hour: drone3 does NOT build with the current behaverify
# CLI. It needs the 2025_FMCAD generator and metamodel. Using the wrong one
# gives a generation failure that looks like a broken model.
#
# Usage:  ./COLLAB/experiments/reachable.sh
set -u
cd "$(git rev-parse --show-toplevel)"
NX=$PWD/REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv
OUT=$(mktemp -d)
CMD_R=$OUT/reach.cmd; CMD_C=$OUT/check.cmd
printf 'go\nprint_reachable_states\nquit\n'      > "$CMD_R"
printf 'go\ncheck_invar\ncheck_ctlspec\nquit\n'  > "$CMD_C"

measure () {          # $1 = label, $2 = smv path
  local label=$1 smv=$2
  local reach diam elapsed answered start
  reach=$("$NX" -source "$CMD_R" "$smv" 2>&1 | grep -i "reachable states" | head -1)
  diam=$("$NX"  -source "$CMD_R" "$smv" 2>&1 | grep -i "system diameter" | head -1)
  start=$(date +%s.%N)
  "$NX" -source "$CMD_C" "$smv" > "$OUT/$label.out" 2>&1
  elapsed=$(echo "$(date +%s.%N) - $start" | bc)
  answered=$(grep -cE "is true|is false" "$OUT/$label.out")
  printf "  %-14s %s\n" "$label" "$reach"
  [ -n "$diam" ] && printf "  %-14s %s\n" "" "$diam"
  printf "  %-14s specs answered: %s in %.2fs\n\n" "" "$answered" "$elapsed"
}

echo "== DrunkenDrone (current behaverify CLI) =="
behaverify nuxmv examples/DrunkenDrone/DrunkenDrone.tree "$OUT/dd" \
  --generate --nuxmv_path "$NX" > "$OUT/dd.gen.log" 2>&1
measure DrunkenDrone "$(find "$OUT/dd" -name '*.smv' | head -1)"

echo "== drone3 (REQUIRES the 2025_FMCAD generator, not the CLI) =="
python3 REPRODUCIBILITY/2025_FMCAD/src/dsl_to_nuxmv.py \
        REPRODUCIBILITY/2025_FMCAD/metamodel/behaverify.tx \
        examples/2025_FMCAD_BT2Fiacre/drone3.tree \
        "$OUT/d3.smv" --no_checks > "$OUT/d3.gen.log" 2>&1
measure drone3 "$OUT/d3.smv"

echo "artefacts in $OUT"
