#!/bin/bash
# Why the generated ring's apparent wall is an artefact, in three measurements.
#   1. reachable states     - the declared space is irrelevant
#   2. invar vs ctlspec     - the cost is entirely CTL liveness
#   3. -dynamic             - most of it is BDD variable ordering, a tuning flag
# Usage:  ./COLLAB/experiments/ring_diagnosis.sh [N]    (default 7)
set -u
cd "$(git rev-parse --show-toplevel)"
N=${1:-7}
NX=$PWD/REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv
OUT=$(mktemp -d)
python3 COLLAB/experiments/gen_ring.py "$OUT" "$N" > /dev/null
behaverify nuxmv "$OUT/ring_$N.tree" "$OUT/g" --generate --nuxmv_path "$NX" > "$OUT/gen.log" 2>&1
SMV=$(find "$OUT/g" -name '*.smv' | head -1)
echo "ring N=$N  (declared space $(python3 -c "print(f'{($N+1)**$N:,}')"))"

printf 'go\nprint_reachable_states\nquit\n' > "$OUT/r.cmd"
printf "  %-34s %s\n" "reachable:" \
  "$("$NX" -source "$OUT/r.cmd" "$SMV" 2>&1 | grep -i 'reachable states' | head -1)"

for what in check_invar check_ctlspec; do
  printf "go\n$what\nquit\n" > "$OUT/c.cmd"
  s=$(date +%s.%N); "$NX" -source "$OUT/c.cmd" "$SMV" > "$OUT/$what.out" 2>&1
  printf "  %-34s %7.2fs  (%s answered)\n" "$what:" \
    "$(echo "$(date +%s.%N) - $s" | bc)" "$(grep -cE 'is true|is false' "$OUT/$what.out")"
done

printf 'go\ncheck_ctlspec\nquit\n' > "$OUT/d.cmd"
s=$(date +%s.%N); "$NX" -dynamic -source "$OUT/d.cmd" "$SMV" > "$OUT/dyn.out" 2>&1
printf "  %-34s %7.2fs  (%s answered)\n" "check_ctlspec WITH -dynamic:" \
  "$(echo "$(date +%s.%N) - $s" | bc)" "$(grep -cE 'is true|is false' "$OUT/dyn.out")"
