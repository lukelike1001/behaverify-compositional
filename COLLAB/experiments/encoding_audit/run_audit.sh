#!/bin/bash
# Support matrix: which (network ops x encoding x input domain) combinations does
# monolithic BehaVerify actually accept? Generation only - we are asking what the
# tool REFUSES, not how fast it is.
set -u
cd "$(git rev-parse --show-toplevel)"
D=COLLAB/experiments/encoding_audit
NX=$PWD/REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv
OUT=$(mktemp -d)
printf "%-16s %-14s %-6s %s\n" network encoding inputs result
for net in net_gemm_relu net_tanh net_sigmoid; do
  for enc in table float fixed fixed_direct real; do
    for dom in int real; do
      t=$D/${net}__${enc}__${dom}.tree
      [ -f "$t" ] || continue
      if timeout 180 behaverify nuxmv "$t" "$OUT/o" --generate --nuxmv_path "$NX" \
           > "$OUT/log" 2>&1; then
        res="OK ($(find "$OUT/o" -name '*.smv' | head -1 | xargs wc -l 2>/dev/null | awk '{print $1}') smv lines)"
      else
        res=$(grep -oE "BTreeException.*|TextXSyntaxError.*|[A-Za-z]*Error: .*" "$OUT/log" \
              | head -1 | cut -c1-92)
        [ -z "$res" ] && res="FAILED (no message)"
      fi
      rm -rf "$OUT/o"
      printf "%-16s %-14s %-6s %s\n" "$net" "$enc" "$dom" "$res"
    done
  done
done
