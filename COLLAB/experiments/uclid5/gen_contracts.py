"""
Does contract-as-axiom scale? NAV needed 1,386 contracts.

Emits a 2-D continuous plant whose controller is an uninterpreted function
constrained by N box contracts, each of the CROWN form
"over this region the network never outputs <action>". Then k-induction.

The question is whether Z3 stays tractable as N grows, since unbounded
`forall` over reals has no decision procedure in general.
"""
import sys, pathlib

HEAD = """module main {{
  type act_t = enum {{ up, down, stay }};
  function ctrl(x : real, y : real) : act_t;
  var px : real;
  var py : real;
  var mode : act_t;

{axioms}

  init {{ px = 0.0; py = 0.0; mode = stay; }}
  next {{
    mode' = ctrl(px, py);
    px' = if (mode' == up) then px + 1.0 else if (mode' == down) then px - 1.0 else px;
    py' = py;
  }}

  invariant safe : {prop};

  control {{ induction; check; print_results; }}
}}
"""

def gen(n, prop=None):
    ax = []
    for i in range(n):
        lo, hi = 10.0 + 3.0*i, 13.0 + 3.0*i          # CONTIGUOUS: [10,13],[13,16],...
        ax.append(f"  axiom c{i} : (forall (x : real, y : real) :: "
                  f"((x >= {lo}) && (x <= {hi})) ==> (ctrl(x, y) != up));")
    # property spans EVERY box, so the proof needs all n axioms, not just one
    top = 13.0 + 3.0*(n-1)
    prop = prop or f"((px >= 10.0) && (px <= {top})) ==> (ctrl(px, py) != up)"
    return HEAD.format(axioms="\n".join(ax), prop=prop)

out = pathlib.Path(sys.argv[1])
for n in [int(a) for a in sys.argv[2:]]:
    (out / f"scale_{n}.ucl").write_text(gen(n))
    print(f"  {n:>5} contracts -> scale_{n}.ucl")
