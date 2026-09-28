"""
Real alpha-beta-CROWN contracts for NAV, emitted as UCLID5 assumptions.

Uses nn-nav-set.onnx (NOT nn-nav-point): identical architecture, but set-based
training folds reachability into the loop, giving Lipschitz 17.09 vs 51.21 and
a 3x larger admissible cell (reports/NAV/2026_09_21_monolithic_lipschitz.md).

SOUNDNESS: CROWN returns float bounds; SMT needs decimal literals. Rounding a
lower bound UP or an upper bound DOWN would silently narrow the certified
interval and make the contracts unsound. Every bound is therefore rounded
OUTWARD (floor for lower, ceil for upper) at a fixed number of decimals.
"""
from __future__ import annotations
import math, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, onnx, torch
from onnx import numpy_helper
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

DECIMALS = 6

def floor_to(v: float, n: int = DECIMALS) -> float:
    """Round DOWN - safe for a lower bound."""
    return math.floor(v * 10**n) / 10**n

def ceil_to(v: float, n: int = DECIMALS) -> float:
    """Round UP - safe for an upper bound."""
    return math.ceil(v * 10**n) / 10**n

def load_net(path: pathlib.Path) -> torch.nn.Sequential:
    """NAV: MatMul/Add/Relu x2 then MatMul/Add/Tanh. 4 -> 64 -> 32 -> 2."""
    g = onnx.load(str(path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    layers: list[torch.nn.Module] = []
    for k, act in ((1, torch.nn.ReLU), (2, torch.nn.ReLU), (3, torch.nn.Tanh)):
        w, b = init[f"fc_{k}_MatMul_W"], init[f"fc_{k}_Add_B"]
        lin = torch.nn.Linear(w.shape[0], w.shape[1])
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(w.T)); lin.bias.copy_(torch.tensor(b))
        layers += [lin, act()]
    return torch.nn.Sequential(*layers)

def crown_bounds(bm: BoundedModule, lo: np.ndarray, hi: np.ndarray):
    """Sound elementwise output bounds over a batch of input boxes."""
    l = torch.tensor(lo, dtype=torch.float32)
    h = torch.tensor(hi, dtype=torch.float32)
    mid = (l + h) / 2
    lb, ub = bm.compute_bounds(
        x=(BoundedTensor(mid, PerturbationLpNorm(x_L=l, x_U=h)),), method="CROWN")
    return lb.detach().numpy(), ub.detach().numpy()

def main() -> None:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    nx = int(sys.argv[2]) if len(sys.argv) > 2 else 4      # cells per position axis
    nv = int(sys.argv[3]) if len(sys.argv) > 3 else 2      # cells per v / theta axis
    onnx_path = pathlib.Path(
        "REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx")
    bm = BoundedModule(load_net(onnx_path), torch.zeros(1, 4))

    # operating box: covers start (3,3), goal (0,0) and the obstacle
    XS = np.linspace(-1.0, 3.5, nx + 1)
    YS = np.linspace(-1.0, 3.5, nx + 1)
    VS = np.linspace(-1.0, 1.0, nv + 1)
    TS = np.linspace(-math.pi, math.pi, nv + 1)

    boxes, los, his = [], [], []
    for i in range(nx):
        for j in range(nx):
            for a in range(nv):
                for b in range(nv):
                    lo = [XS[i], YS[j], VS[a], TS[b]]
                    hi = [XS[i+1], YS[j+1], VS[a+1], TS[b+1]]
                    boxes.append((lo, hi)); los.append(lo); his.append(hi)
    lb, ub = crown_bounds(bm, np.array(los, dtype=np.float32),
                              np.array(his, dtype=np.float32))

    lines = []
    for n, ((lo, hi), l, u) in enumerate(zip(boxes, lb, ub)):
        guard = (f"(x >= {lo[0]:.6f}) && (x <= {hi[0]:.6f}) && "
                 f"(y >= {lo[1]:.6f}) && (y <= {hi[1]:.6f}) && "
                 f"(v >= {lo[2]:.6f}) && (v <= {hi[2]:.6f}) && "
                 f"(th >= {lo[3]:.6f}) && (th <= {hi[3]:.6f})")
        # tanh output is structurally in [-1,1]; intersect, then round OUTWARD
        u1l, u1h = floor_to(max(l[0], -1.0)), ceil_to(min(u[0], 1.0))
        u2l, u2h = floor_to(max(l[1], -1.0)), ceil_to(min(u[1], 1.0))
        lines.append(
            f"  assume c{n} : ({guard}) ==> "
            f"((u1(x,y,v,th) >= {u1l:.6f}) && (u1(x,y,v,th) <= {u1h:.6f}) && "
            f"(u2(x,y,v,th) >= {u2l:.6f}) && (u2(x,y,v,th) <= {u2h:.6f}));")

    out = root / "nav_contracts.ucl.inc"
    out.write_text("\n".join(lines) + "\n")
    widths = np.concatenate([ub[:, 0] - lb[:, 0], ub[:, 1] - lb[:, 1]])
    print(f"  network        nn-nav-set.onnx")
    print(f"  grid           {nx}x{nx}x{nv}x{nv} = {len(boxes)} boxes")
    print(f"  contracts      {len(lines)} -> {out}")
    print(f"  bound width    mean {widths.mean():.4f}  max {widths.max():.4f}  (vacuous = 2.0)")

if __name__ == "__main__":
    main()
