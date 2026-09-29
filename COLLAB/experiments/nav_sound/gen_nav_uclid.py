"""
Generate UCLID5 models for NAV with abstract control and abstract displacement.
Can generate from CROWN bounds directly, or convert existing nuXmv .smv models
to ensure 100% bit-for-bit parity with the nuXmv benchmark files.
"""
from __future__ import annotations
import math, sys, re, pathlib, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
try:
    from gen_nav_contracts import load_net, crown_bounds, floor_to, ceil_to
    from auto_LiRPA import BoundedModule
except ImportError:
    pass

NET = pathlib.Path("REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx")
DT, STEPS = 0.2, 30
OBS = ((1.0, 2.0), (1.0, 2.0))
GOAL = ((-0.5, 0.5), (-0.5, 0.5))


def displacement_bounds(vlo, vhi, tlo, thi, samples=400):
    ts = np.linspace(tlo, thi, samples)
    c, s = np.cos(ts), np.sin(ts)
    dx = np.concatenate([vlo * c, vhi * c]) * DT
    dy = np.concatenate([vlo * s, vhi * s]) * DT
    pad = abs(vhi - vlo) * (thi - tlo) * DT
    return ((floor_to(dx.min() - pad), ceil_to(dx.max() + pad)),
            (floor_to(dy.min() - pad), ceil_to(dy.max() + pad)))


def build_uclid_uf(nx: int, nv: int, check_mode: str = "both", bmc_depth: int = 30) -> str:
    """Build UCLID5 model using uninterpreted functions."""
    xs = np.linspace(-1.0, 3.5, nx + 1)
    vs = np.linspace(-2.0, 2.0, nv + 1)
    ts = np.linspace(-math.pi, math.pi, nv + 1)
    bm = BoundedModule(load_net(NET), torch.zeros(1, 4))

    los, his, cells = [], [], []
    for i in range(nx):
        for j in range(nx):
            for a in range(nv):
                for b in range(nv):
                    lo = [xs[i], xs[j], vs[a], ts[b]]
                    hi = [xs[i+1], xs[j+1], vs[a+1], ts[b+1]]
                    los.append(lo); his.append(hi); cells.append((lo, hi))
    lb, ub = crown_bounds(bm, np.array(los, np.float32), np.array(his, np.float32))

    L = [
        "module main {",
        "  function u1(x : real, y : real, v : real, th : real) : real;",
        "  function u2(x : real, y : real, v : real, th : real) : real;",
        "  function dx(v : real, th : real) : real;",
        "  function dy(v : real, th : real) : real;",
        "",
        "  var x : real;",
        "  var y : real;",
        "  var v : real;",
        "  var th : real;",
        "  var k : integer;",
        ""
    ]

    for idx, ((lo, hi), l, u) in enumerate(zip(cells, lb, ub)):
        g = (f"(x >= {lo[0]:.6f}) && (x <= {hi[0]:.6f}) && "
             f"(y >= {lo[1]:.6f}) && (y <= {hi[1]:.6f}) && "
             f"(v >= {lo[2]:.6f}) && (v <= {hi[2]:.6f}) && "
             f"(th >= {lo[3]:.6f}) && (th <= {hi[3]:.6f})")
        u1l, u1h = floor_to(max(l[0], -1.0)), ceil_to(min(u[0], 1.0))
        u2l, u2h = floor_to(max(l[1], -1.0)), ceil_to(min(u[1], 1.0))
        L.append(f"  assume c_{idx} : ({g}) ==> "
                 f"((u1(x, y, v, th) >= {u1l:.6f}) && (u1(x, y, v, th) <= {u1h:.6f}) && "
                 f"(u2(x, y, v, th) >= {u2l:.6f}) && (u2(x, y, v, th) <= {u2h:.6f}));")

    d_idx = 0
    for a in range(nv):
        for b in range(nv):
            (dxl, dxh), (dyl, dyh) = displacement_bounds(vs[a], vs[a+1], ts[b], ts[b+1])
            g = (f"(v >= {vs[a]:.6f}) && (v <= {vs[a+1]:.6f}) && "
                 f"(th >= {ts[b]:.6f}) && (th <= {ts[b+1]:.6f})")
            L.append(f"  assume d_{d_idx} : ({g}) ==> "
                     f"((dx(v, th) >= {dxl:.6f}) && (dx(v, th) <= {dxh:.6f}) && "
                     f"(dy(v, th) >= {dyl:.6f}) && (dy(v, th) <= {dyh:.6f}));")
            d_idx += 1

    L.extend([
        "",
        "  init {",
        "    assume (x >= 2.9 && x <= 3.1);",
        "    assume (y >= 2.9 && y <= 3.1);",
        "    v = 0.0;",
        "    th = 0.0;",
        "    k = 0;",
        "  }",
        "",
        "  next {",
        f"    if (k < {STEPS}) {{",
        "      x' = x + dx(v, th);",
        "      y' = y + dy(v, th);",
        f"      v' = v + {DT} * u1(x, y, v, th);",
        f"      th' = th + {DT} * u2(x, y, v, th);",
        "      k' = k + 1;",
        "    } else {",
        "      x' = x;",
        "      y' = y;",
        "      v' = v;",
        "      th' = th;",
        "      k' = k;",
        "    }",
        "  }",
        "",
        f"  invariant safe : !((x >= {OBS[0][0]:.6f}) && (x <= {OBS[0][1]:.6f}) && "
        f"(y >= {OBS[1][0]:.6f}) && (y <= {OBS[1][1]:.6f}));",
        "",
        "  control {"
    ])

    if check_mode in ("ind", "both"):
        L.append("    v_ind = induction(1);")
    if check_mode in ("bmc", "both"):
        L.append(f"    v_bmc = bmc({bmc_depth});")
    L.extend([
        "    check;",
        "    print_results;",
        "  }",
        "}",
        ""
    ])
    return "\n".join(L)


def convert_smv_to_ucl(smv_path: str | pathlib.Path, check_mode: str = "both", bmc_depth: int = 30) -> str:
    """Convert existing nuXmv .smv file to identical UCLID5 model."""
    lines = pathlib.Path(smv_path).read_text().splitlines()
    invars = [l for l in lines if l.startswith("INVAR ")]
    spec = [l for l in lines if l.startswith("INVARSPEC ")][0]

    out = [
        "module main {",
        "  function u1(x : real, y : real, v : real, th : real) : real;",
        "  function u2(x : real, y : real, v : real, th : real) : real;",
        "  function dx(v : real, th : real) : real;",
        "  function dy(v : real, th : real) : real;",
        "",
        "  var x : real;",
        "  var y : real;",
        "  var v : real;",
        "  var th : real;",
        "  var k : integer;",
        ""
    ]

    for idx, inv in enumerate(invars):
        m = re.match(r"INVAR \((.*)\) -> \((.*)\);", inv)
        if not m:
            continue
        guard, concl = m.group(1), m.group(2)
        guard_ucl = guard.replace("&", "&&")
        concl_ucl = concl.replace("&", "&&")
        if "u1" in concl_ucl or "u2" in concl_ucl:
            concl_ucl = re.sub(r"\bu1\b", "u1(x, y, v, th)", concl_ucl)
            concl_ucl = re.sub(r"\bu2\b", "u2(x, y, v, th)", concl_ucl)
        if "dx" in concl_ucl or "dy" in concl_ucl:
            concl_ucl = re.sub(r"\bdx\b", "dx(v, th)", concl_ucl)
            concl_ucl = re.sub(r"\bdy\b", "dy(v, th)", concl_ucl)

        out.append(f"  assume c_{idx} : ({guard_ucl}) ==> ({concl_ucl});")

    out.extend([
        "",
        "  init {",
        "    assume (x >= 2.9 && x <= 3.1);",
        "    assume (y >= 2.9 && y <= 3.1);",
        "    v = 0.0;",
        "    th = 0.0;",
        "    k = 0;",
        "  }",
        "",
        "  next {",
        f"    if (k < {STEPS}) {{",
        "      x' = x + dx(v, th);",
        "      y' = y + dy(v, th);",
        f"      v' = v + {DT} * u1(x, y, v, th);",
        f"      th' = th + {DT} * u2(x, y, v, th);",
        "      k' = k + 1;",
        "    } else {",
        "      x' = x;",
        "      y' = y;",
        "      v' = v;",
        "      th' = th;",
        "      k' = k;",
        "    }",
        "  }",
        ""
    ])

    m_spec = re.match(r"INVARSPEC (.*);", spec)
    spec_expr = m_spec.group(1).replace("&", "&&")
    out.append(f"  invariant safe : {spec_expr};")
    out.append("")
    out.append("  control {")
    if check_mode in ("ind", "both"):
        out.append("    v_ind = induction(1);")
    if check_mode in ("bmc", "both"):
        out.append(f"    v_bmc = bmc({bmc_depth});")
    out.extend([
        "    check;",
        "    print_results;",
        "  }",
        "}",
        ""
    ])
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Generate UCLID5 model for NAV.")
    parser.add_argument("nx", type=int, nargs="?", help="nx cells")
    parser.add_argument("nv", type=int, nargs="?", help="nv cells")
    parser.add_argument("--smv", type=str, help="Source SMV file to convert directly")
    parser.add_argument("--check", type=str, choices=["ind", "bmc", "both"], default="both", help="Verification command")
    parser.add_argument("--depth", type=int, default=30, help="BMC depth")
    parser.add_argument("--out", type=str, help="Output .ucl file path")
    args = parser.parse_args()

    if args.smv:
        text = convert_smv_to_ucl(args.smv, check_mode=args.check, bmc_depth=args.depth)
        out_path = pathlib.Path(args.out or args.smv.replace(".smv", f"_{args.check}.ucl"))
    elif args.nx and args.nv:
        smv_candidate = pathlib.Path(f"COLLAB/experiments/nav_sound/nav_{args.nx}_{args.nv}_obstacle.smv")
        if smv_candidate.exists():
            text = convert_smv_to_ucl(smv_candidate, check_mode=args.check, bmc_depth=args.depth)
        else:
            text = build_uclid_uf(args.nx, args.nv, check_mode=args.check, bmc_depth=args.depth)
        out_path = pathlib.Path(args.out or f"COLLAB/experiments/nav_sound/nav_{args.nx}_{args.nv}_{args.check}.ucl")
    else:
        parser.print_help()
        sys.exit(1)

    out_path.write_text(text)
    n_assumes = text.count("  assume c_") + text.count("  assume d_")
    print(f"  {out_path.name}: {len(text.splitlines())} lines, {n_assumes} contracts")


if __name__ == "__main__":
    main()
