#!/bin/bash
# The measurement that actually matters for claim #2: does monolithic BehaVerify
# struggle on the LARGEST REAL behaviour trees in the repo?
cd /home/lnam/VSCode/behaverify-compositional
S=/tmp/claude-1000/-home-lnam-VSCode-behaverify-compositional/734b9139-97e7-470a-9610-20dc857cead8/scratchpad/scaling
NX=$PWD/REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv
printf "%-46s %-6s %-8s %-10s %-9s %s\n" tree nodes gen_s smv_lines check_s result
for T in "$@"; do
  name=$(basename $T .tree); O=$S/real_$name; rm -rf $O
  nodes=$(( $(grep -c 'composite {' $T) + $(grep -c 'action {' $T) + $(grep -c 'check {' $T) ))
  st=$(date +%s.%N)
  timeout 900 behaverify nuxmv $T $O --generate --nuxmv_path $NX > $O.gen.log 2>&1
  gen=$(echo "$(date +%s.%N) - $st" | bc)
  SMV=$(find $O -name '*.smv' 2>/dev/null | head -1)
  [ -z "$SMV" ] && { printf "%-46s %-6s GEN FAILED: %s\n" "${T#examples/}" $nodes "$(grep -oE 'BTreeException.*' $O.gen.log | head -1 | cut -c1-60)"; continue; }
  specs=$(grep -cE "^(LTLSPEC|CTLSPEC|INVARSPEC)" $SMV)
  printf 'go\ncheck_ltlspec\ncheck_ctlspec\ncheck_invar\nquit\n' > $O.cmd
  st=$(date +%s.%N); timeout 900 $NX -source $O.cmd $SMV > $O.out 2>&1; rc=$?
  chk=$(echo "$(date +%s.%N) - $st" | bc)
  n=$(grep -cE "is true|is false" $O.out)
  [ $rc -eq 124 ] && res="TIMEOUT(900s)" || res="${n} specs answered"
  printf "%-46s %-6s %-8.2f %-10s %-9.2f %s\n" "${T#examples/}" $nodes $gen $(wc -l < $SMV) $chk "$res"
done
