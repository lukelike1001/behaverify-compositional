#!/bin/bash
# MANDATORY before trusting any UCLID5 contract proof.
#
# An inconsistent contract set proves EVERYTHING, including 1.0 == 2.0, and it
# looks exactly like a spectacular success. Measured: with contracts asserting
# ctrl(x) >= 1.0 && ctrl(x) <= 0.0, the invariant `1.0 == 2.0` passes 2/0.
#
# The check: replace the real invariant with `false` and run induction.
#   base FAILED  -> contracts are satisfiable, the proof means something
#   base PASSED  -> `false` holds initially, which is only possible if the
#                   assumption set is unsatisfiable. ABORT.
# (The STEP always passes under `false` - assuming false proves anything - so
#  the base case is the only discriminator.)
#
# Usage: ./vacuity_guard.sh model.ucl
set -u
cd "$(git rev-parse --show-toplevel)"
T=REPRODUCIBILITY/2026_TBA
U=${UCLID:-/tmp/uclid_new/uclid-0.9.5/bin/uclid}
Z3=$PWD/$T/uclid5/z3
export PATH=$Z3/bin:$PATH
export LD_LIBRARY_PATH=$Z3/bin:${LD_LIBRARY_PATH:-}

model=$1
tmp=$(mktemp --suffix=.ucl)
sed -E 's|^[[:space:]]*invariant [A-Za-z_0-9]+ :.*|  invariant vacuity_guard : false;|' "$model" > "$tmp"
out=$("$U" "$tmp" 2>&1)
base=$(grep -oE "(PASSED|FAILED) -> induction_base" <<<"$out" | grep -oE "PASSED|FAILED" | head -1)
rm -f "$tmp"

if [ "$base" = "PASSED" ]; then
  echo "VACUOUS: contract set is UNSATISFIABLE - every proof from it is meaningless"; exit 1
elif [ "$base" = "FAILED" ]; then
  echo "OK: contract set is satisfiable"; exit 0
else
  echo "INCONCLUSIVE: could not determine base-case verdict"; echo "$out" | tail -5; exit 2
fi
