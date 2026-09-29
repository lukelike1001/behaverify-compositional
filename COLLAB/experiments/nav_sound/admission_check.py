"""
PHASE 0: does the abstract model ADMIT every true NAV trajectory?

Over-approximation is sound only if no true behaviour is EXCLUDED. Accidental
exclusion - not accidental inclusion - is what makes a proof wrong, and it is
what an inconsistent contract set, a wrong-direction rounding, or an off-by-one
cell index would cause.

So: simulate the true closed loop with the real ONNX, then check that at every
step the true control lies inside the contract that covers that state. A single
violation means our contracts are unsound and every downstream proof is void.
"""
from __future__ import annotations
import math, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, onnxruntime as ort, torch
from auto_LiRPA import BoundedModule

sys.path.insert(0, "COLLAB/experiments/nav_uclid5")
from gen_nav_contracts import load_net, crown_bounds, floor_to, ceil_to   # noqa: E402

NET = pathlib.Path("REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx")
DT = 0.2
STEPS = 30


class NavSimulator:
    """The true closed loop: ONNX controller plus the benchmark's own dynamics."""

    def __init__(self, net_path: pathlib.Path = NET) -> None:
        self.session = ort.InferenceSession(str(net_path))
        self.input_name = self.session.get_inputs()[0].name

    def control(self, state: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: state.reshape(1, 4).astype(np.float32)})
        return out[0].reshape(-1)

    def step(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """dynamics.m: dx = [v*cos(th), v*sin(th), u1, u2]."""
        x, y, v, th = state
        u = self.control(state)
        nxt = np.array([x + v * math.cos(th) * DT,
                        y + v * math.sin(th) * DT,
                        v + u[0] * DT,
                        th + u[1] * DT], dtype=np.float64)
        return nxt, u

    def trajectory(self, start: np.ndarray, steps: int = STEPS):
        state, out = start.astype(np.float64), []
        for _ in range(steps):
            nxt, u = self.step(state)
            out.append((state.copy(), u.copy()))
            state = nxt
        return out, state


class ContractGrid:
    """Uniform grid over the operating box, with CROWN bounds per visited cell."""

    def __init__(self, nx: int, nv: int,
                 xr=(-1.0, 3.5), yr=(-1.0, 3.5), vr=(-2.0, 2.0),
                 tr=(-math.pi, math.pi)) -> None:
        self.edges = [np.linspace(*xr, nx + 1), np.linspace(*yr, nx + 1),
                      np.linspace(*vr, nv + 1), np.linspace(*tr, nv + 1)]
        self.bm = BoundedModule(load_net(NET), torch.zeros(1, 4))
        self.cache: dict[tuple[int, ...], tuple] = {}

    def cell_of(self, state: np.ndarray) -> tuple[int, ...] | None:
        idx = []
        for value, edge in zip(state, self.edges):
            if value < edge[0] or value > edge[-1]:
                return None                      # outside the modelled box
            k = int(np.searchsorted(edge, value, side="right") - 1)
            idx.append(min(max(k, 0), len(edge) - 2))
        return tuple(idx)

    def contract_for(self, cell: tuple[int, ...]):
        """CROWN bounds for a cell, rounded OUTWARD exactly as emitted to SMV."""
        if cell in self.cache:
            return self.cache[cell]
        lo = np.array([[e[i] for e, i in zip(self.edges, cell)]], dtype=np.float32)
        hi = np.array([[e[i + 1] for e, i in zip(self.edges, cell)]], dtype=np.float32)
        lb, ub = crown_bounds(self.bm, lo, hi)
        bounds = tuple((floor_to(max(lb[0][k], -1.0)), ceil_to(min(ub[0][k], 1.0)))
                       for k in (0, 1))
        self.cache[cell] = bounds
        return bounds


def main() -> None:
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    nv = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    sim, grid = NavSimulator(), ContractGrid(nx, nv)

    starts = [np.array([x, y, 0.0, 0.0]) for x in (2.9, 3.0, 3.1) for y in (2.9, 3.0, 3.1)]
    violations, outside, checked = [], 0, 0
    for start in starts:
        traj, _ = sim.trajectory(start)
        for step, (state, u) in enumerate(traj):
            cell = grid.cell_of(state)
            if cell is None:
                outside += 1
                continue
            (l1, h1), (l2, h2) = grid.contract_for(cell)
            checked += 1
            if not (l1 <= u[0] <= h1 and l2 <= u[1] <= h2):
                violations.append((tuple(start[:2]), step, tuple(np.round(u, 6)),
                                   (l1, h1), (l2, h2)))

    print(f"  grid            {nx}x{nx}x{nv}x{nv}")
    print(f"  trajectories    {len(starts)} starts x {STEPS} steps")
    print(f"  states checked  {checked}   (outside the modelled box: {outside})")
    print(f"  VIOLATIONS      {len(violations)}")
    for v in violations[:5]:
        print(f"    start {v[0]} step {v[1]}: u={v[2]} not in u1{v[3]} u2{v[4]}")
    print("  => contracts ADMIT the true trajectories" if not violations
          else "  => UNSOUND: a true trajectory is EXCLUDED by the contracts")


if __name__ == "__main__":
    main()
