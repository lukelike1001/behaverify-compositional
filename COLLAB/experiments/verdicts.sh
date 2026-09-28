#!/bin/bash
# Why the generated ring is NOT a faithful generalisation of FiveInRing.
#
# The hand-written original assigns goals irregularly so the robots deadlock:
# most of its CTL specs are FALSE and nuXmv exits on shallow counterexamples.
# gen_ring.py uses goal = (i+3) % npos, which removes the deadlock, making every
# liveness spec TRUE and forcing full nested AG AF fixpoints. That difference -
# not tree size, not state space - is what made the generated ring look hard.
#
# Usage:  ./COLLAB/experiments/verdicts.sh [N ...]     (default 5)
set -u
cd "$(git rev-parse --show-toplevel)"
NX=$PWD/REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv
OUT=$(mktemp -d)
printf 'go\ncheck_invar\ncheck_ctlspec\nquit\n' > "$OUT/c.cmd"

count () {            # $1 = label, $2 = tree path
  local smv
  behaverify nuxmv "$2" "$OUT/$1" --generate --nuxmv_path "$NX" > "$OUT/$1.gen.log" 2>&1
  smv=$(find "$OUT/$1" -name '*.smv' | head -1)
  [ -z "$smv" ] && { printf "  %-22s GENERATION FAILED\n" "$1"; return; }
  "$NX" -source "$OUT/c.cmd" "$smv" > "$OUT/$1.out" 2>&1
  printf "  %-22s %s false / %s true\n" "$1" \
     "$(grep -cE 'is false' "$OUT/$1.out")" "$(grep -cE 'is true' "$OUT/$1.out")"
}

echo "== hand-written original =="
count FiveInRing examples/MultiAgent/FiveInRing.tree

echo "== generated (gen_ring.py) =="
for n in "${@:-5}"; do
  python3 COLLAB/experiments/gen_ring.py "$OUT" "$n" > /dev/null
  count "ring_$n" "$OUT/ring_$n.tree"
done
