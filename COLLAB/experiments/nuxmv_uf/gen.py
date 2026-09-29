"""Same contracts-as-constraints-on-a-UF experiment as COLLAB/experiments/uclid5/,
but in nuXmv. nuXmv supports FUN (uninterpreted functions), so the architecture
that motivated UCLID5 may not require UCLID5 at all."""
import sys, pathlib
HEAD = """MODULE main
VAR
  px : real;
FUN
  ctrl : real -> real;
INIT px = 0.0;
TRANS next(px) = px + ctrl(px);
{invars}
INVARSPEC {prop};
"""
def gen(n):
    inv = []
    for i in range(n):
        lo, hi = 10.0 + 3.0*i, 13.0 + 3.0*i          # contiguous, like the UCLID5 test
        inv.append(f"INVAR (px >= {lo} & px <= {hi}) -> (ctrl(px) >= -1.0 & ctrl(px) <= 0.0);")
    top = 13.0 + 3.0*(n-1)
    prop = f"(px >= 10.0 & px <= {top}) -> (ctrl(px) >= -1.0 & ctrl(px) <= 0.0)"
    return HEAD.format(invars="\n".join(inv), prop=prop)
out = pathlib.Path(sys.argv[1])
for n in [int(a) for a in sys.argv[2:]]:
    (out / f"uf_{n}.smv").write_text(gen(n)); print(f"  {n:>5} contracts -> uf_{n}.smv")
