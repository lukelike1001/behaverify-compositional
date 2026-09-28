#!/bin/bash
# Contract-as-axiom scaling. NAV needed 1,386 contracts.
set -u
cd "$(git rev-parse --show-toplevel)"
T=REPRODUCIBILITY/2026_TBA
U=/tmp/uclid_new/uclid-0.9.5/bin/uclid
Z3=$PWD/$T/uclid5/z3
export PATH=$Z3/bin:$PATH
export LD_LIBRARY_PATH=$Z3/bin:${LD_LIBRARY_PATH:-}
printf "%-10s %-9s %-9s %-10s %s\n" contracts passed failed time verdict
for n in "$@"; do
  f=COLLAB/experiments/uclid5/scale_$n.ucl
  st=$(date +%s.%N)
  out=$(timeout ${TMO:-300} "$U" "$f" 2>&1); rc=$?
  el=$(echo "$(date +%s.%N) - $st" | bc)
  p=$(grep -oE '[0-9]+ assertions passed' <<<"$out" | grep -oE '^[0-9]+')
  fl=$(grep -oE '[0-9]+ assertions failed' <<<"$out" | grep -oE '^[0-9]+')
  if [ $rc -eq 124 ]; then v="TIMEOUT(${TMO:-300}s)"
  elif [ -z "$p" ]; then v="ERROR: $(grep -oiE 'error.*' <<<"$out" | head -1 | cut -c1-50)"
  elif [ "${fl:-0}" = "0" ]; then v="PROVED"
  else v="failed"; fi
  printf "%-10s %-9s %-9s %-10.2f %s\n" "$n" "${p:-–}" "${fl:-–}" "$el" "$v"
done
