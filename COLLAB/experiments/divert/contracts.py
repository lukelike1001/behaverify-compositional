"""
Demand-driven CEGAR contract generator for `divert` (4-D: x, y, vx, vy).

SOUNDNESS ARGUMENT (Claim #62):
-------------------------------
A uniform 12-per-axis contract grid over the 4-D state space (x, y, vx, vy)
yields 12^4 = 20,736 cells. At 40 ticks, that produces ~829,440 guarded
implications, creating 15-30 GB of AST before SMT search can even begin.
A dense contract grid is therefore computationally intractable.

Demand-driven CEGAR solves this by emitting contracts only for an ACTIVE
subset of cells. Soundness is established as follows:
  1. Omitting a contract for an uncovered cell means the control action is
     constrained ONLY by the network's global actuator limits [-A_MAX, A_MAX],
     alongside the plant's unconditional kinematic bounds (|dx| <= |vx|*dt,
     |dy| <= |vy|*dt).
  2. Because uncovered cells admit ANY physically possible control input
     in [-A_MAX, A_MAX], the active subset forms a strictly COARSER
     over-approximation than the full contract grid.
  3. By monotonicity of over-approximations:
     UNSAT under the active subset soundly implies UNSAT under the full grid
     (and under the true continuous closed-loop system).
  4. Refinement (adding contracts for cells visited by counterexamples)
     only TIGHTENS the abstraction, admitting fewer behaviors and strictly
     reducing the over-approximation error.
  5. If the solver returns SAT and all states on the counterexample trace
     already reside within covered cells, the grid resolution itself is
     too coarse to separate the true dynamics from the violation. Refinement
     terminates honestly rather than spinning.

GLOBAL BOUNDS:
--------------
Uncovered cells must never allow the abstract state to explode arbitrarily.
The plant dynamics unconditionally satisfy:
  |dx| <= |vx| * dt   and   |dy| <= |vy| * dt
and the controller satisfies:
  |ax| <= A_MAX       and   |ay| <= A_MAX
These bounds are asserted globally without requiring any cell guard.
"""
from __future__ import annotations

import dis
import inspect
import math
import sys
import warnings
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# Use plant.py and envelope.py constants - do not redefine them.
try:
    from plant import A_MAX, DT, S_MAX, ARENA, return_law, cruise_law
    from envelope import TRIP
except ImportError:
    # Fallbacks if imported from outside the divert directory
    sys.path.insert(0, "COLLAB/experiments/divert")
    from plant import A_MAX, DT, S_MAX, ARENA, return_law, cruise_law
    from envelope import TRIP

# Optional auto_LiRPA imports for CROWN verification
try:
    import torch
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    HAS_AUTOLIRPA = True
except ImportError:
    HAS_AUTOLIRPA = False

# Default operating region enclosing START, TRIP, and obstacle avoidance paths
DEFAULT_REGION = (
    TRIP[0] - 1.0, TRIP[1] + 1.0,   # x: approx -0.45 to 1.90
    TRIP[2] - 1.0, TRIP[3] + 1.0,   # y: approx -0.80 to 1.35
    -S_MAX * 1.5, S_MAX * 1.5,       # vx: -1.5 to 1.5
    -S_MAX * 1.5, S_MAX * 1.5        # vy: -1.5 to 1.5
)


def cell_index(val: float, lo: float, hi: float, n: int) -> int:
    """Grid index containing val, clamped to [0, n - 1]."""
    if hi <= lo:
        return 0
    return min(max(int((val - lo) / (hi - lo) * n), 0), n - 1)


def parse_resolution(resolution: Union[int, Sequence[int]]) -> tuple[int, int, int, int]:
    """Normalize resolution to a 4-tuple (nx, ny, nvx, nvy)."""
    if resolution is None:
        return (8, 8, 8, 8)
    if isinstance(resolution, (int, np.integer)):
        r = int(resolution)
        return (r, r, r, r)
    if isinstance(resolution, (tuple, list)) and len(resolution) == 4:
        return tuple(int(x) for x in resolution)
    raise ValueError(f"Resolution must be an int or a 4-tuple, got {resolution}")


def point_to_indices(
    point: Sequence[float],
    region: Sequence[float] = DEFAULT_REGION,
    resolution: Union[int, Sequence[int]] = 8
) -> tuple[int, int, int, int]:
    """Map a continuous 4-D point (x, y, vx, vy) to 4-D grid indices."""
    nx, ny, nvx, nvy = parse_resolution(resolution)
    x, y, vx, vy = point[:4]
    return (
        cell_index(float(x), region[0], region[1], nx),
        cell_index(float(y), region[2], region[3], ny),
        cell_index(float(vx), region[4], region[5], nvx),
        cell_index(float(vy), region[6], region[7], nvy),
    )


def indices_to_box(
    indices: tuple[int, int, int, int],
    region: Sequence[float] = DEFAULT_REGION,
    resolution: Union[int, Sequence[int]] = 8
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the 4-D bounding box for cell indices (ix, iy, ivx, ivy)."""
    ix, iy, ivx, ivy = indices
    nx, ny, nvx, nvy = parse_resolution(resolution)
    xlo, xhi, ylo, yhi, vxlo, vxhi, vylo, vyhi = region

    bx_lo = xlo + ix * (xhi - xlo) / nx
    bx_hi = xlo + (ix + 1) * (xhi - xlo) / nx
    by_lo = ylo + iy * (yhi - ylo) / ny
    by_hi = ylo + (iy + 1) * (yhi - ylo) / ny
    bvx_lo = vxlo + ivx * (vxhi - vxlo) / nvx
    bvx_hi = vxlo + (ivx + 1) * (vxhi - vxlo) / nvx
    bvy_lo = vylo + ivy * (vyhi - vylo) / nvy
    bvy_hi = vylo + (ivy + 1) * (vyhi - vylo) / nvy
    return (bx_lo, bx_hi, by_lo, by_hi, bvx_lo, bvx_hi, bvy_lo, bvy_hi)


def with_neighbours(
    keys: Iterable[tuple[int, int, int, int]],
    limits: Sequence[int] = (8, 8, 8, 8)
) -> set[tuple[int, int, int, int]]:
    """Each cell plus its face neighbours along each of the 4 axes, clipped to grid limits.

    Ported from COLLAB/experiments/nav_sound/z3_cegar.py:
    An uncovered cell is unconstrained, so the solver will route the next
    counterexample straight through it. Expanding by one cell per axis fills
    the active tube far faster and converges in fewer CEGAR iterations.
    """
    out = set(keys)
    for key in keys:
        for axis, limit in enumerate(limits):
            for step in (-1, 1):
                nb = list(key)
                nb[axis] += step
                if 0 <= nb[axis] < limit:
                    out.add(tuple(nb))
    return out


class ContractCell(tuple):
    """A 12-tuple contract cell over a 4-D box.

    Format: (xlo, xhi, ylo, yhi, vxlo, vxhi, vylo, vyhi, axlo, axhi, aylo, ayhi)

    Compatibility:
      - 12-unpacking: xlo, xhi, ylo, yhi, vxlo, vxhi, vylo, vyhi, axlo, axhi, aylo, ayhi = cell
      - 2-unpacking:  (bx, bounds) = cell
      - Named properties: cell.box, cell.bounds, cell.bx, cell.indices, cell.key
    """
    def __new__(cls, *args, indices=None):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        elif len(args) == 2 and isinstance(args[0], (list, tuple)) and isinstance(args[1], (list, tuple)):
            args = tuple(args[0]) + tuple(args[1])
        if len(args) != 12:
            raise ValueError(f"ContractCell expects 12 coordinates, got {len(args)}")
        obj = super().__new__(cls, (float(a) for a in args))
        obj._indices = indices
        return obj

    @property
    def box(self) -> tuple[float, float, float, float, float, float, float, float]:
        return self[:8]

    @property
    def bx(self) -> tuple[float, float, float, float, float, float, float, float]:
        return self[:8]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self[8:]

    @property
    def key(self) -> Optional[tuple[int, int, int, int]]:
        return self._indices

    @property
    def indices(self) -> Optional[tuple[int, int, int, int]]:
        return self._indices

    @property
    def xlo(self) -> float: return self[0]
    @property
    def xhi(self) -> float: return self[1]
    @property
    def ylo(self) -> float: return self[2]
    @property
    def yhi(self) -> float: return self[3]
    @property
    def vxlo(self) -> float: return self[4]
    @property
    def vxhi(self) -> float: return self[5]
    @property
    def vylo(self) -> float: return self[6]
    @property
    def vyhi(self) -> float: return self[7]
    @property
    def axlo(self) -> float: return self[8]
    @property
    def axhi(self) -> float: return self[9]
    @property
    def aylo(self) -> float: return self[10]
    @property
    def ayhi(self) -> float: return self[11]

    def __iter__(self):
        """Enable unpacking as either 12 items or (box, bounds) pair based on caller context."""
        f = inspect.currentframe().f_back
        if f:
            try:
                instrs = list(dis.get_instructions(f.f_code))
                for idx, instr in enumerate(instrs):
                    if instr.offset == f.f_lasti:
                        if instr.opname == "UNPACK_SEQUENCE" and instr.argval == 2:
                            return iter((self[:8], self[8:]))
                        if instr.opname == "FOR_ITER":
                            if idx + 1 < len(instrs) and instrs[idx + 1].opname == "UNPACK_SEQUENCE" and instrs[idx + 1].argval == 2:
                                return iter((self[:8], self[8:]))
            except Exception:
                pass
        return super().__iter__()


class CellList(list):
    """A list of ContractCell instances tracking active cell index keys."""

    def __init__(self, items=(), bound_fn=None, region=None, resolution=None):
        super().__init__()
        self.bound_fn = bound_fn
        self.region = region
        self.resolution = resolution
        self.active_keys: set[tuple[int, int, int, int]] = set()
        self.too_coarse: bool = False
        self.added_count: int = 0
        for item in items:
            self.append(item)

    def append(self, item: ContractCell) -> None:
        if not isinstance(item, ContractCell):
            item = ContractCell(item)
        super().append(item)
        if item.key is not None:
            self.active_keys.add(item.key)

    def extend(self, items: Iterable[ContractCell]) -> None:
        for it in items:
            self.append(it)


def global_bounds(
    ax,
    ay,
    vx=None,
    vy=None,
    dx=None,
    dy=None
):
    """Emit unconditional global invariants for a single tick in Z3.

    Soundness guarantee (Requirement 3):
    An uncovered cell must NEVER leave the state unconstrained.
    1. Actuator output bounds: ax, ay in [-A_MAX, A_MAX] unconditionally.
    2. Plant displacement bounds: |dx| <= |vx|*dt and |dy| <= |vy|*dt.
       Split on the sign of vx and vy to preserve linearity.
    """
    import z3

    constraints = [
        ax >= -A_MAX, ax <= A_MAX,
        ay >= -A_MAX, ay <= A_MAX,
    ]
    if dx is not None and vx is not None:
        constraints.extend([
            z3.Implies(vx >= 0, z3.And(dx <= DT * vx, dx >= -DT * vx)),
            z3.Implies(vx <= 0, z3.And(dx <= -DT * vx, dx >= DT * vx)),
        ])
    if dy is not None and vy is not None:
        constraints.extend([
            z3.Implies(vy >= 0, z3.And(dy <= DT * vy, dy >= -DT * vy)),
            z3.Implies(vy <= 0, z3.And(dy >= DT * vy, dy <= -DT * vy)),
        ])
    return constraints


def analytic_placeholder_bound_fn(
    box: Sequence[float],
    law: Callable[[float, float, float, float], tuple[float, float]] = return_law,
    samples: int = 5
) -> tuple[float, float, float, float]:
    """Sample the analytic law over a 4-D box and pad outward.

    NOT SOUND: Exists as a placeholder while networks are being trained,
    matching verify.py's _sampled_bounds interface. CROWN bounds replace it
    seamlessly via make_crown_bound_fn.
    """
    xlo, xhi, ylo, yhi, vxlo, vxhi, vylo, vyhi = box[:8]
    axs, ays = [], []
    for x in np.linspace(xlo, xhi, samples):
        for y in np.linspace(ylo, yhi, samples):
            for vx in np.linspace(vxlo, vxhi, samples):
                for vy in np.linspace(vylo, vyhi, samples):
                    ax, ay = law(float(x), float(y), float(vx), float(vy))
                    axs.append(ax)
                    ays.append(ay)
    pad = 0.15 * A_MAX
    return (
        max(min(axs) - pad, -A_MAX), min(max(axs) + pad, A_MAX),
        max(min(ays) - pad, -A_MAX), min(max(ays) + pad, A_MAX)
    )


def make_crown_bound_fn(
    model_or_bounded,
    decimals: int = 6
) -> Callable[[Sequence[float]], tuple[float, float, float, float]]:
    """Factory creating sound CROWN output bounds over 4-D boxes.

    Compatible with auto_LiRPA BoundedModule or standard PyTorch nn.Module.
    Sound outward rounding: floor for lower bounds, ceil for upper bounds.
    """
    if not HAS_AUTOLIRPA:
        raise ImportError("auto_LiRPA and torch are required for make_crown_bound_fn")

    if isinstance(model_or_bounded, BoundedModule):
        bm = model_or_bounded
    else:
        bm = BoundedModule(model_or_bounded, torch.zeros(1, 4))

    def bound_fn(box: Sequence[float]) -> tuple[float, float, float, float]:
        lo = np.array([[box[0], box[2], box[4], box[6]]], dtype=np.float32)
        hi = np.array([[box[1], box[3], box[5], box[7]]], dtype=np.float32)
        mid = (lo + hi) / 2.0

        t_lo = torch.tensor(lo, dtype=torch.float32)
        t_hi = torch.tensor(hi, dtype=torch.float32)
        t_mid = torch.tensor(mid, dtype=torch.float32)

        xb = BoundedTensor(t_mid, PerturbationLpNorm(x_L=t_lo, x_U=t_hi))
        lb, ub = bm.compute_bounds(x=(xb,), method="CROWN")
        lb_np = lb.detach().cpu().numpy()[0]
        ub_np = ub.detach().cpu().numpy()[0]

        scale = 10 ** decimals
        axlo = max(math.floor(float(lb_np[0]) * scale) / scale, -A_MAX)
        axhi = min(math.ceil(float(ub_np[0]) * scale) / scale, A_MAX)
        aylo = max(math.floor(float(lb_np[1]) * scale) / scale, -A_MAX)
        ayhi = min(math.ceil(float(ub_np[1]) * scale) / scale, A_MAX)
        return (axlo, axhi, aylo, ayhi)

    return bound_fn


def crown_cells(
    resolution: Union[int, Sequence[int]] = 8,
    model=None,
    region: Sequence[float] = DEFAULT_REGION
) -> list[ContractCell]:
    """CROWN bounds per cell over the 4-D grid.

    Mirrors COLLAB/experiments/geofence/compare.py crown_cells for 4-D.
    Note: For resolution >= 8, uniform evaluation generates thousands of cells.
    Use generate() with CEGAR for scalable verification.
    """
    if model is None:
        raise ValueError("A PyTorch model or BoundedModule is required for crown_cells")
    bound_fn = make_crown_bound_fn(model)
    return generate(bound_fn=bound_fn, region=region, resolution=resolution, initial="full")


def seed_cells(
    region: Sequence[float] = DEFAULT_REGION,
    resolution: Union[int, Sequence[int]] = 8,
    n_x: int = 5,
    n_y: int = 5,
    n_speeds: int = 3,
    n_angles: int = 4
) -> set[tuple[int, int, int, int]]:
    """Seed cells visited by true nominal recovery trajectories.

    Ported from seed_cells in nav_sound/z3_cegar.py.
    Provides a cheap, compact starting set covering the physical recovery tube.
    """
    from envelope import recover
    from plant import Tree

    tree = Tree()
    seen = set()

    xs = np.linspace(TRIP[0], TRIP[1], n_x)
    ys = np.linspace(TRIP[2], TRIP[3], n_y)
    speeds = np.linspace(0.3, S_MAX, n_speeds)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    for x in xs:
        for y in ys:
            for speed in speeds:
                for th in angles:
                    vx, vy = speed * np.cos(th), speed * np.sin(th)
                    trace = recover(tree, float(x), float(y), vx, vy, steps=40)
                    for s in trace:
                        if s[4] == "Return":
                            key = point_to_indices((s[0], s[1], s[2], s[3]), region, resolution)
                            seen.add(key)
    return seen


def generate(
    bound_fn: Optional[Callable] = None,
    region: Sequence[float] = DEFAULT_REGION,
    resolution: Union[int, Sequence[int]] = 8,
    initial: str = "auto",
    with_seed_neighbours: bool = True
) -> CellList:
    """Generate contract cells for the 4-D divert benchmark.

    Interface:
      cells = generate(bound_fn, region, resolution)
        -> list of (xlo, xhi, ylo, yhi, vxlo, vxhi, vylo, vyhi, axlo, axhi, aylo, ayhi)

    Parameters:
    -----------
    bound_fn: callable(box) -> (axlo, axhi, aylo, ayhi)
        Function computing control bounds over a 4-D box. Defaults to analytic placeholder.
    region: 8-tuple
        (xlo, xhi, ylo, yhi, vxlo, vxhi, vylo, vyhi)
    resolution: int or 4-tuple
        Number of subdivisions per axis.
    initial: str
        - "auto": For coarse resolution (<=2 per axis), generates full grid (<=16 cells);
                  For fine resolution (>2 per axis), generates seeded tube (~100-300 cells)
                  to avoid out-of-memory AST explosion.
        - "seed": Seeds cells from nominal simulation trajectories.
        - "empty": Returns an empty CellList for pure CEGAR refinement from tick 0.
        - "full": Generates the full uniform grid (WARNING: 12^4 = 20,736 cells).
    with_seed_neighbours: bool
        Whether to include face neighbours when seeding (default True).

    Returns:
    --------
    CellList of ContractCell instances (each acts as a 12-tuple).
    """
    if bound_fn is None:
        bound_fn = analytic_placeholder_bound_fn
    if region is None:
        region = DEFAULT_REGION

    res_tuple = parse_resolution(resolution)
    total_cells = res_tuple[0] * res_tuple[1] * res_tuple[2] * res_tuple[3]

    if initial == "auto":
        mode = "full" if total_cells <= 16 else "seed"
    else:
        mode = initial

    cells = CellList(bound_fn=bound_fn, region=region, resolution=resolution)

    if mode == "empty":
        return cells

    if mode == "seed":
        keys = seed_cells(region, resolution)
        if with_seed_neighbours:
            keys = with_neighbours(keys, res_tuple)
        for k in sorted(keys):
            box = indices_to_box(k, region, resolution)
            bounds = bound_fn(box)
            cells.append(ContractCell(*box, *bounds, indices=k))
        return cells

    if mode == "full":
        if total_cells > 2000:
            warnings.warn(
                f"Generating full dense grid of {total_cells} cells in 4-D! "
                "This may cause out-of-memory or severe SMT slowdown.",
                UserWarning,
                stacklevel=2
            )
        for ix in range(res_tuple[0]):
            for iy in range(res_tuple[1]):
                for ivx in range(res_tuple[2]):
                    for ivy in range(res_tuple[3]):
                        k = (ix, iy, ivx, ivy)
                        box = indices_to_box(k, region, resolution)
                        bounds = bound_fn(box)
                        cells.append(ContractCell(*box, *bounds, indices=k))
        return cells

    raise ValueError(f"Unknown initial mode: {initial}")


def refine(
    cells: Union[list, CellList],
    counterexample_point: Union[Sequence, tuple, np.ndarray],
    bound_fn: Optional[Callable] = None,
    region: Optional[Sequence[float]] = None,
    resolution: Optional[Union[int, Sequence[int]]] = None,
    add_neighbours: bool = True
) -> CellList:
    """Refine active contract cells using a counterexample point or trajectory.

    Interface:
      refine(cells, counterexample_point) -> cells

    Parameters:
    -----------
    cells: list or CellList
        The active set of ContractCell instances.
    counterexample_point:
        Can be:
        - A single 4-D state: (x, y, vx, vy)
        - A sequence of 4-D states: [(x0, y0, vx0, vy0), ...]
        - A Z3 model + state tuple: (model, (x_vars, y_vars, vx_vars, vy_vars))
    bound_fn: callable or None
        Bound function for evaluating new cells. Defaults to cells.bound_fn or analytic.
    region: 8-tuple or None
        Bounding region. Defaults to cells.region or DEFAULT_REGION.
    resolution: int, 4-tuple, or None
        Grid resolution. Defaults to cells.resolution or 8.
    add_neighbours: bool
        Whether to add 4-D face neighbours to accelerate tube coverage (default True).

    Returns:
    --------
    cells: The updated CellList with new contracts appended.
    """
    if not isinstance(cells, CellList):
        cells = CellList(cells, bound_fn=bound_fn, region=region, resolution=resolution)

    bound_fn = bound_fn or cells.bound_fn or analytic_placeholder_bound_fn
    region = region or cells.region or DEFAULT_REGION
    resolution = resolution or cells.resolution or 8
    res_tuple = parse_resolution(resolution)

    # Extract 4-D points from counterexample_point
    points = []
    if (
        isinstance(counterexample_point, tuple)
        and len(counterexample_point) == 2
        and hasattr(counterexample_point[0], "eval")
    ):
        model, st_vars = counterexample_point
        if len(st_vars) >= 4:
            xs, ys, vxs, vys = st_vars[:4]
            for k in range(len(xs)):
                try:
                    vx_val = float(model.eval(xs[k], model_completion=True).as_fraction())
                    vy_val = float(model.eval(ys[k], model_completion=True).as_fraction())
                    vvx_val = float(model.eval(vxs[k], model_completion=True).as_fraction())
                    vvy_val = float(model.eval(vys[k], model_completion=True).as_fraction())
                    points.append((vx_val, vy_val, vvx_val, vvy_val))
                except Exception:
                    pass
    elif isinstance(counterexample_point, np.ndarray):
        if counterexample_point.ndim == 1:
            points = [tuple(counterexample_point[:4])]
        else:
            for pt in counterexample_point:
                points.append(tuple(pt[:4]))
    elif isinstance(counterexample_point, (list, tuple)) and len(counterexample_point) > 0:
        if isinstance(counterexample_point[0], (int, float, np.floating)):
            points = [tuple(counterexample_point[:4])]
        else:
            for pt in counterexample_point:
                points.append(tuple(pt[:4]))

    # Identify grid cells containing the counterexample points
    touched_keys = set()
    for pt in points:
        touched_keys.add(point_to_indices(pt, region, resolution))

    active_keys = cells.active_keys
    new_keys = touched_keys - active_keys

    if not new_keys:
        cells.too_coarse = True
        cells.added_count = 0
        return cells

    cells.too_coarse = False

    if add_neighbours:
        candidates = with_neighbours(new_keys, res_tuple)
    else:
        candidates = new_keys

    to_add = candidates - active_keys
    for k in sorted(to_add):
        box = indices_to_box(k, region, resolution)
        bounds = bound_fn(box)
        cell = ContractCell(*box, *bounds, indices=k)
        cells.append(cell)

    cells.added_count = len(to_add)
    return cells


def main():
    """Report contract count reductions and verify interface compatibility."""
    print("=" * 70)
    print("CEGAR Contract Generator for `divert` (4-D: x, y, vx, vy)")
    print("Soundness: Claim #62 over-approximation monotonicity")
    print("=" * 70)

    for res in [6, 8, 10, 12]:
        full_grid = res ** 4
        # Seeded cells from recovery simulations
        seeded = seed_cells(resolution=res)
        seeded_with_nb = with_neighbours(seeded, (res, res, res, res))
        print(f"Resolution {res:>2}x{res:>2}x{res:>2}x{res:>2}:")
        print(f"  Full dense grid : {full_grid:>6} contracts")
        print(f"  CEGAR visited   : {len(seeded):>6} contracts ({100 * len(seeded) / full_grid:5.2f}%)")
        print(f"  CEGAR + face nb : {len(seeded_with_nb):>6} contracts ({100 * len(seeded_with_nb) / full_grid:5.2f}%)")
        print(f"  Reduction factor: {full_grid / len(seeded_with_nb):5.1f}x fewer contracts")
        print()

    # Self-test the interface
    cells = generate(resolution=8)
    print(f"Interface check: generate(resolution=8) returned {len(cells)} cells")
    c0 = cells[0]
    # Check 12-unpacking
    xlo, xhi, ylo, yhi, vxlo, vxhi, vylo, vyhi, axlo, axhi, aylo, ayhi = c0
    print(f"12-unpacking verified: x in [{xlo:.2f}, {xhi:.2f}], ax in [{axlo:.2f}, {axhi:.2f}]")
    # Check 2-unpacking
    bx, bounds = c0
    print(f"2-unpacking verified : len(bx)={len(bx)}, len(bounds)={len(bounds)}")

    # Test refinement with a mock counterexample point
    mock_cex = (0.7, 0.25, -0.4, 0.1)
    cells = refine(cells, mock_cex)
    print(f"refine() verified   : active cells now {len(cells)} (added {cells.added_count})")
    print("=" * 70)


if __name__ == "__main__":
    main()
