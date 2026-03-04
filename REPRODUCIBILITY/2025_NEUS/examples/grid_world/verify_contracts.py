"""
Verify A/G safety contracts for the grid-world NSBT using alpha-beta-CROWN.

Contract schema (per obstacle o=(ox,oy), per entry direction d):
  Assume : drone at adjacent cell (cx,cy), goal at ANY integer point in [0,6]^2
  Guarantee: NN output != direction d  (d would move drone into obstacle)

Verification mode: INTEGER-ONLY
  For each contract, 49 sub-verifications are run — one per integer goal
  (x_g, y_g) in {0,...,6}^2. Each uses a tiny EPS-ball (0.001) around the
  integer point so CROWN sees a proper interval while staying essentially at
  the integer input. The NN was trained and verified on integer inputs only,
  so this is the correct granularity.

Class index mapping (from draw_network.py CODES, matches DSL declaration order):
  We=0  Ea=1  No=2  So=3  XX=4

Output:
  - Console table (contract, status, marker)
  - contract_results.json in the same directory
"""

import sys
import json
import functools
import datetime
import torch
from abcrown import ABCrownSolver, VerificationSpec, ConfigBuilder, input_vars, output_vars

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_MIN = 0
GRID_MAX = 6

# ε-ball radius for integer grid points. Kept tiny (0.001) so CROWN performs
# essentially a point-check at each integer input. The system only ever
# queries the NN at integer coordinates, so this is the right granularity.
EPS = 0.001

# Per-sub-verification BaB timeout in seconds (5 minutes).
TIMEOUT_SEC = 300

# 18 obstacles from template.tree (flat array index 2k=x, 2k+1=y)
OBSTACLES = [
    (0, 0), (0, 1),
    (1, 0), (1, 1), (1, 2), (1, 4), (1, 6),
    (2, 1), (2, 4), (2, 6),
    (3, 3), (3, 6),
    (4, 1), (4, 2),
    (5, 5),
    (6, 1), (6, 2), (6, 3),
]
OBSTACLE_SET = set(OBSTACLES)

# Direction index → (label, dx, dy)
#   dx/dy is the movement applied when this direction is chosen
DIRECTIONS = {
    0: ("We", -1,  0),   # West:  x decreases
    1: ("Ea", +1,  0),   # East:  x increases
    2: ("No",  0, +1),   # North: y increases
    3: ("So",  0, -1),   # South: y decreases
}

NUM_CLASSES = 5

# Pre-compute all 49 integer goal positions
GOAL_POINTS = [
    (x_g, y_g)
    for x_g in range(GRID_MIN, GRID_MAX + 1)
    for y_g in range(GRID_MIN, GRID_MAX + 1)
]

# ---------------------------------------------------------------------------
# Status normalization
# ---------------------------------------------------------------------------

def normalize_status(raw: str) -> str:
    """Map CROWN result.status to SAT / UNSAT / TIMEOUT."""
    if raw in ("safe", "verified", "safe-incomplete"):
        return "SAT"
    if raw.startswith("unsafe"):
        return "UNSAT"
    return "TIMEOUT"


# ---------------------------------------------------------------------------
# Contract generation
# ---------------------------------------------------------------------------

def generate_contracts():
    """
    For each obstacle and each cardinal direction, produce one contract:
      (cx, cy, forbidden_dir_idx, dir_label, ox, oy, description_string)

    cx, cy  = source cell the drone is standing on
    forbidden_dir_idx = direction index the NN must NOT output
    """
    contracts = []
    for (ox, oy) in OBSTACLES:
        for d_idx, (label, dx, dy) in DIRECTIONS.items():
            # The source cell is one step OPPOSITE to d
            cx, cy = ox - dx, oy - dy
            # Skip if source cell is outside the grid
            if not (GRID_MIN <= cx <= GRID_MAX and GRID_MIN <= cy <= GRID_MAX):
                continue
            # Skip if the source cell is itself an obstacle
            if (cx, cy) in OBSTACLE_SET:
                continue
            desc = f"obstacle ({ox},{oy})  source ({cx},{cy})  forbid {label}"
            contracts.append((cx, cy, d_idx, label, ox, oy, desc))
    return contracts


# ---------------------------------------------------------------------------
# Sub-verification: single integer goal point
# ---------------------------------------------------------------------------

def verify_at_goal(onnx_path, cx, cy, forbidden_d, x_g, y_g, config):
    """
    Verify the contract for one specific integer goal (x_g, y_g).

    Input box: EPS-ball around (cx, cy, x_g, y_g).
    Output constraint (what we PROVE): some other class beats forbidden_d.
    """
    x = input_vars((4,))
    lower = torch.tensor(
        [cx - EPS, cy - EPS, x_g - EPS, y_g - EPS], dtype=torch.float32
    )
    upper = torch.tensor(
        [cx + EPS, cy + EPS, x_g + EPS, y_g + EPS], dtype=torch.float32
    )
    input_constraint = (x >= lower) & (x <= upper)

    y = output_vars(NUM_CLASSES)
    other = [j for j in range(NUM_CLASSES) if j != forbidden_d]
    output_constraint = functools.reduce(
        lambda a, b: a | b,
        [y[j] > y[forbidden_d] for j in other],
    )

    spec = VerificationSpec.build_spec(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )
    solver = ABCrownSolver(spec, onnx_path, config=config)
    return solver.solve()


# ---------------------------------------------------------------------------
# Contract verification: all 49 integer goal points
# ---------------------------------------------------------------------------

def verify_contract(onnx_path, cx, cy, forbidden_d, config):
    """
    Verify the contract over all 49 integer goal positions.

    Returns:
        overall  : "SAT" | "UNSAT" | "TIMEOUT"
        sub      : dict mapping "(x_g,y_g)" -> "SAT"/"UNSAT"/"TIMEOUT"
        ce_goal  : (x_g, y_g) of first counterexample, or None
    """
    sub = {}
    ce_goal = None
    had_timeout = False

    for (x_g, y_g) in GOAL_POINTS:
        result = verify_at_goal(onnx_path, cx, cy, forbidden_d, x_g, y_g, config)
        status = normalize_status(result.status)
        sub[f"({x_g},{y_g})"] = status

        if status == "UNSAT":
            ce_goal = (x_g, y_g)
            # Early exit — contract is already falsified
            return "UNSAT", sub, ce_goal

        if status == "TIMEOUT":
            had_timeout = True

    overall = "TIMEOUT" if had_timeout else "SAT"
    return overall, sub, ce_goal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    onnx_path = "./networks/1000__6_18_0__0200_1.onnx"
    output_path = "./contract_results.json"

    config = (
        ConfigBuilder.from_defaults()
        .set(general__device="cpu")
        .set(attack__pgd_order="skip")   # skip PGD, go straight to BaB
        .set(bab__timeout=TIMEOUT_SEC)
        ()
    )

    contracts = generate_contracts()
    print(f"Generated {len(contracts)} contracts "
          f"(integer-only mode, EPS={EPS}, timeout={TIMEOUT_SEC}s)\n")
    print(f"{'#':<4} {'Description':<45} {'Status':<10} {'Marker'}")
    print("-" * 75)

    n_sat = n_unsat = n_timeout = 0
    json_contracts = []

    for i, (cx, cy, d_idx, label, ox, oy, desc) in enumerate(contracts):
        overall, sub, ce_goal = verify_contract(onnx_path, cx, cy, d_idx, config)

        if overall == "SAT":
            n_sat += 1
            marker = "✓"
        elif overall == "UNSAT":
            n_unsat += 1
            ce_str = f" (counterexample goal={ce_goal})"
            marker = f"✗  ← VIOLATION{ce_str}"
        else:
            n_timeout += 1
            marker = "?  ← TIMEOUT (inconclusive)"

        print(f"{i+1:<4} {desc:<45} {overall:<10} {marker}")
        sys.stdout.flush()

        json_contracts.append({
            "id": i + 1,
            "obstacle": [ox, oy],
            "source": [cx, cy],
            "forbidden_dir": label,
            "forbidden_dir_idx": d_idx,
            "description": desc,
            "status": overall,
            "counterexample_goal": list(ce_goal) if ce_goal else None,
            "sub_results": sub,
        })

    print("-" * 75)
    print(f"\nSummary: {n_sat} SAT, {n_unsat} UNSAT, {n_timeout} TIMEOUT "
          f"out of {len(contracts)} contracts")

    # Save JSON report
    report = {
        "onnx_path": onnx_path,
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": f"integer-only (EPS={EPS})",
        "timeout_sec": TIMEOUT_SEC,
        "summary": {
            "sat": n_sat,
            "unsat": n_unsat,
            "timeout": n_timeout,
            "total": len(contracts),
        },
        "contracts": json_contracts,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
