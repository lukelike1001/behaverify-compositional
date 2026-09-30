"""
The geofence NSBT through the ACTUAL BehaVerify pipeline.

Every NAV experiment this project ran wrote SMV by hand from the ONNX, which
means "compositional BehaVerify on NAV" never went through BehaVerify at all
(claim #71).  This closes that loop for the pilot:

  geofence.tree  --(BehaVerify DSL front end)-->  SMV with REAL state and real
                                                  BT node semantics
                 --(dsl_with_abstract_control_to_nuxmv)-->  control freed and
                                                  CROWN contracts injected
                 --(nuXmv)-->  verdict

The generated model genuinely carries the tree: OutsideFence is a module over
x_stage_0, SafetyBranch is a sequence composite, Envelope is a selector
composite.  Nothing about the structure is hand-encoded.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]
                       / "REPRODUCIBILITY/2026_TBA/src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from compare import crown_cells                                   # noqa: E402
from dsl_with_abstract_control_to_nuxmv import (                  # noqa: E402
    AbstractControlPatcher, Contract, RegionBound)

TREE = pathlib.Path(__file__).resolve().parent / "geofence.tree"
NUXMV = "REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv"
CONTROL = "accel_stage_0"


def generate(out_dir):
    """Run the BehaVerify front end and return the generated SMV path.

    The directory is cleared first: a previous run leaves its PATCHED model
    here too, and globbing would otherwise pick that up and try to free an
    already-freed control.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "behaverify", "nuxmv", str(TREE),
                    str(out_dir), "--generate", "--overwrite"],
                   capture_output=True, text=True, check=True)
    return next(pathlib.Path(out_dir).rglob(f"{TREE.stem}.smv"))


# a cell's upper edge is pulled in by this much so adjacent regions do not
# share a face. RegionBound is closed on both sides, so closed cells make
# corner-touching neighbours overlap at a point; their CROWN guarantees then
# meet at a single value and check_consistency flags the pair on a
# floating-point hair. Non-overlapping regions remove the ambiguity entirely,
# and shrinking rather than growing keeps every contract sound on its own cell.
EDGE = 1e-9


def contracts_from_crown(grid):
    """Turn CROWN cell bounds into Contract objects over disjoint (x, v) regions.

    The first contract has an EMPTY region, so it holds everywhere. Freeing the
    control makes it an unconstrained real, and the per-cell contracts only
    cover the grid - outside it the solver may pick any acceleration at all.
    The network's tanh output bounds it to [-1, 1] everywhere by construction,
    which is a free structural fact and must be stated or the model can escape
    through the gap. nuXmv found exactly that hole on the first run.
    """
    out = [Contract(region=[], target=CONTROL, kind="interval", lo=-1.0, hi=1.0)]
    for xlo, xhi, vlo, vhi, alo, ahi in crown_cells(grid):
        out.append(Contract(
            region=[RegionBound("system.x_stage_0", xlo, xhi - EDGE),
                    RegionBound("system.v_stage_0", vlo, vhi - EDGE)],
            target=CONTROL, kind="interval", lo=alo, hi=ahi))
    return out


def main():
    """Generate, patch, and report what the pipeline produced."""
    grid = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = pathlib.Path("/tmp/geofence_behaverify")
    smv_path = generate(out)
    smv = smv_path.read_text()
    print(f"BehaVerify generated {smv_path.name}: {len(smv.splitlines())} lines")
    reals = sorted({line.split(":")[0].strip()
                    for line in smv.splitlines() if ": real;" in line})
    print(f"  real state variables : {reals}")
    print(f"  BT composites        : "
          f"{sum(1 for l in smv.splitlines() if 'composite_' in l)}")

    contracts = contracts_from_crown(grid)
    patcher = AbstractControlPatcher()
    clashes = patcher.check_consistency(contracts)
    print(f"  contracts            : {len(contracts)} from a {grid}x{grid} grid, "
          f"{len(clashes)} inconsistent pairs")

    patched = patcher.patch(smv, CONTROL, contracts)
    dest = out.parent / "geofence_contracts.smv"
    dest.write_text(patched)
    print(f"  patched model        : {dest} "
          f"({len(patched.splitlines())} lines)")
    print(f"  control freed        : "
          f"{'yes' if f'init({CONTROL})' not in patched else 'NO - still pinned'}")
    print(f"  INVARs injected      : {patched.count('INVAR ')}")


if __name__ == "__main__":
    main()
