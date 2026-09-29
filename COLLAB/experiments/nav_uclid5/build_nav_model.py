"""
Assemble a complete UCLID5 NAV model around the generated CROWN contracts.

The capacity question this answers: nuXmv segfaults between 5,184 and 9,604
INVAR constraints.  Can UCLID5 even LOAD 16,384 assumptions, and if so, does
it decide anything within a usable time budget?

Usage: python3 COLLAB/experiments/nav_uclid5/build_nav_model.py [k]
"""
import pathlib
import sys

ROOT = pathlib.Path("COLLAB/experiments/nav_uclid5")
DT = 0.2

HEAD = """// NAV: abstract control + abstract displacement, both as uninterpreted
// functions constrained by outward-rounded alpha-beta-CROWN contracts.
// Stays in QF_UFLRA: no ReLU phase splitting, no nonlinear plant.
module main {
  function u1(x:real, y:real, v:real, th:real) : real;
  function u2(x:real, y:real, v:real, th:real) : real;
  function dx(v:real, th:real) : real;
  function dy(v:real, th:real) : real;

  var x : real;  var y : real;  var v : real;  var th : real;
  var k : integer;

  // tanh bounds the control everywhere, independent of any cell
  assume tanh_u1 : (u1(x,y,v,th) >= -1.0) && (u1(x,y,v,th) <= 1.0);
  assume tanh_u2 : (u2(x,y,v,th) >= -1.0) && (u2(x,y,v,th) <= 1.0);
  // |v| <= 2 and |cos|,|sin| <= 1, so each displacement is within +-2*dt
  assume disp_x : (dx(v,th) >= -0.4) && (dx(v,th) <= 0.4);
  assume disp_y : (dy(v,th) >= -0.4) && (dy(v,th) <= 0.4);
"""

TAIL = """
  init {{ x = 3.0; y = 3.0; v = 0.0; th = 0.0; k = 0; }}
  next {{
    x'  = x + dx(v, th);
    y'  = y + dy(v, th);
    v'  = v + {dt} * u1(x,y,v,th);
    th' = th + {dt} * u2(x,y,v,th);
    k'  = k + 1;
  }}

  // NAV's obstacle is [1,2]^2
  invariant obstacle : !((x >= 1.0) && (x <= 2.0) && (y >= 1.0) && (y <= 2.0));

  control {{ v = induction({k}); check; print_results; }}
}}
"""


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    contracts = (ROOT / "nav_contracts.ucl.inc").read_text()
    model = HEAD + contracts + TAIL.format(dt=DT, k=k)
    out = ROOT / f"nav_full_k{k}.ucl"
    out.write_text(model)
    print(f"  {out.name}: {len(model.splitlines())} lines, "
          f"{contracts.count('assume c')} cell contracts")


if __name__ == "__main__":
    main()
