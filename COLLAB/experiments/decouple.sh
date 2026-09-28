#!/bin/bash
# Which axis costs: tree size, or state space? Tree is FIXED, domain varies.
cd /home/lnam/VSCode/behaverify-compositional
S=/tmp/claude-1000/-home-lnam-VSCode-behaverify-compositional/734b9139-97e7-470a-9610-20dc857cead8/scratchpad/scaling
NX=$PWD/REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv
printf "%-26s %-8s %-9s %-10s %-9s %s\n" model leaves domain smv_lines LTL_s verdict
for f in "$@"; do
  T=$S/$f.tree; O=$S/d_$f; rm -rf $O
  timeout 1200 behaverify nuxmv $T $O --generate --nuxmv_path $NX > $O.gen.log 2>&1
  SMV=$(find $O -name '*.smv' 2>/dev/null | head -1)
  [ -z "$SMV" ] && { printf "%-26s GENERATION FAILED\n" $f; continue; }
  printf 'go\ncheck_ltlspec\nquit\n' > $O.cmd
  st=$(date +%s.%N); timeout 1200 $NX -source $O.cmd $SMV > $O.out 2>&1; rc=$?
  el=$(echo "$(date +%s.%N) - $st" | bc)
  v=$(grep -oE "is true|is false" $O.out | head -1); [ $rc -eq 124 ] && v="TIMEOUT(1200s)"; [ -z "$v" ] && v="no-verdict(rc=$rc)"
  lv=$(echo $f | sed 's/.*_L\([0-9]*\)_.*/\1/'); dm=$(echo $f | sed 's/.*_D//')
  printf "%-26s %-8s %-9s %-10s %-9.2f %s\n" $f $((2**lv)) $dm $(wc -l < $SMV) $el "$v"
done
