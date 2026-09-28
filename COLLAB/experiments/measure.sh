#!/bin/bash
# Claim #2: does monolithic BehaVerify have a scaling wall on these trees?
# Generation and nuXmv are timed separately so we know which one is the bottleneck.
cd /home/lnam/VSCode/behaverify-compositional
S=/tmp/claude-1000/-home-lnam-VSCode-behaverify-compositional/734b9139-97e7-470a-9610-20dc857cead8/scratchpad/scaling
NX=$PWD/REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv
printf "%-7s %-8s %-10s %-10s %-10s %-12s %s\n" levels leaves gen_s smv_lines ltl_s ctl_s verdict
for L in "$@"; do
  T=$S/binary_tree_$L.tree; O=$S/m_$L
  rm -rf $O
  st=$(date +%s.%N)
  timeout 1800 behaverify nuxmv $T $O --generate --nuxmv_path $NX > $O.gen.log 2>&1
  gen=$(echo "$(date +%s.%N) - $st" | bc)
  SMV=$(find $O -name '*.smv' | head -1)
  [ -z "$SMV" ] && { printf "%-7s %-8s GENERATION FAILED\n" $L $((2**L)); continue; }
  lines=$(wc -l < $SMV)
  printf 'go\ncheck_ltlspec\nquit\n' > $O.ltl.cmd
  printf 'go\ncheck_ctlspec\nquit\n' > $O.ctl.cmd
  st=$(date +%s.%N); timeout 1800 $NX -source $O.ltl.cmd $SMV > $O.ltl.out 2>&1; lrc=$?
  ltl=$(echo "$(date +%s.%N) - $st" | bc)
  st=$(date +%s.%N); timeout 1800 $NX -source $O.ctl.cmd $SMV > $O.ctl.out 2>&1
  ctl=$(echo "$(date +%s.%N) - $st" | bc)
  v=$(grep -oE "is true|is false" $O.ltl.out | head -1)
  [ $lrc -eq 124 ] && v="LTL-TIMEOUT(1800s)"
  [ -z "$v" ] && v="no-verdict(rc=$lrc)"
  printf "%-7s %-8s %-10.2f %-10s %-10.2f %-12.2f %s\n" $L $((2**L)) $gen $lines $ltl $ctl "$v"
done
