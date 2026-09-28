#!/bin/bash
cd /home/lnam/VSCode/behaverify-compositional
S=/tmp/claude-1000/-home-lnam-VSCode-behaverify-compositional/734b9139-97e7-470a-9610-20dc857cead8/scratchpad/scaling
NX=$PWD/REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv
printf "%-5s %-7s %-20s %-8s %-10s %-10s %s\n" N nodes state_space gen_s smv_lines check_s result
for n in "$@"; do
  T=$S/ring_$n.tree; O=$S/ring_$n; rm -rf $O
  st=$(date +%s.%N)
  timeout 1800 behaverify nuxmv $T $O --generate --nuxmv_path $NX > $O.gen.log 2>&1
  gen=$(echo "$(date +%s.%N) - $st" | bc)
  SMV=$(find $O -name '*.smv' 2>/dev/null | head -1)
  [ -z "$SMV" ] && { printf "%-5s GEN FAILED: %s\n" $n "$(grep -oE 'BTreeException.*|TextXSyntaxError.*' $O.gen.log | head -1 | cut -c1-70)"; continue; }
  printf 'go\ncheck_invar\ncheck_ctlspec\nquit\n' > $O.cmd
  st=$(date +%s.%N); timeout 1800 $NX -source $O.cmd $SMV > $O.out 2>&1; rc=$?
  chk=$(echo "$(date +%s.%N) - $st" | bc)
  ans=$(grep -cE "is true|is false" $O.out)
  [ $rc -eq 124 ] && res="TIMEOUT(1800s), $ans answered" || res="$ans specs answered"
  ss=$(python3 -c "print(f'{($n+1)**$n:,}')")
  printf "%-5s %-7s %-20s %-8.2f %-10s %-10.2f %s\n" $n $((1+6*n)) "$ss" $gen $(wc -l < $SMV) $chk "$res"
done
