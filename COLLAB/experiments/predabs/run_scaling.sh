#!/bin/bash
# Does SMT-based verification of a symbolically-encoded network scale with width?
# Tests plain BMC against implicit-predicate-abstraction CEGAR.
#
# NOTE: temporarily relaxes the INT-only guard in check_grammar.py and restores
# it on exit. That guard is the only thing preventing REAL network inputs.
set -u
cd "$(git rev-parse --show-toplevel)"
trap 'git checkout -q src/behaverify/check_grammar.py' EXIT
sed -i "s/                        if atom_type != 'INT':/                        if atom_type not in ('INT', 'REAL'):/" src/behaverify/check_grammar.py
D=COLLAB/experiments/predabs
NX=$PWD/REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv
TMO=${TMO:-300}
printf "%-7s %-10s %-12s %-22s %s\n" width smv_lines relus "bmc(een-sorensson)" "bmc_cegar_implabs"
for w in "$@"; do
  rm -rf /tmp/pa_$w
  if ! timeout 300 behaverify nuxmv $D/w$w.tree /tmp/pa_$w --generate --nuxmv_path "$NX" >/tmp/pa_$w.log 2>&1; then
    printf "%-7s GENERATION FAILED: %s\n" "$w" "$(grep -oE 'BTreeException.*|[A-Za-z]*Error.*' /tmp/pa_$w.log | head -1 | cut -c1-60)"; continue
  fi
  S=$(find /tmp/pa_$w -name '*.smv' | head -1)
  relus=$(grep -c "max(0," "$S")
  res=()
  for cmd in "msat_check_invar_bmc -a een-sorensson" "msat_check_invar_bmc_cegar_implabs"; do
    printf 'set on_failure_script_quits\ngo_msat\n%s\nquit\n' "$cmd" > /tmp/pa_$w.cmd
    st=$(date +%s.%N)
    out=$(timeout $TMO "$NX" -source /tmp/pa_$w.cmd "$S" 2>&1); rc=$?
    el=$(echo "$(date +%s.%N) - $st" | bc)
    v=$(grep -oiE "is true|is false" <<<"$out" | head -1)
    [ $rc -eq 124 ] && res+=("TIMEOUT(${TMO}s)") || res+=("$(printf '%s %.1fs' "${v:-no-verdict}" "$el")")
  done
  printf "%-7s %-10s %-12s %-22s %s\n" "$w" "$(wc -l < "$S")" "$relus" "${res[0]}" "${res[1]}"
done
