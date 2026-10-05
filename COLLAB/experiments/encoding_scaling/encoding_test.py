"""
KILL CONDITION 1: does the fixed-point encoding escape the table's ceiling?

If it does, "monolithic cannot scale to a large input space" is not a
contribution -- a reviewer changes `config { table }` to `config { fixed ... }`
and the benchmark runs. The discrete direction dies and we find out tonight
rather than after a week.

Measured on the PUBLISHED grid world template (2026_TBA), varying the grid
size, which is the axis nobody has measured. NeuS Table 3 varied NETWORK SIZE
on a fixed grid and found the table flat at 0.07s while fixed-point grew
53s -> 153s. The opposite axis is the one that matters to us.

TWO FAILURES ON THE WAY HERE, recorded so nobody repeats them. (1) The 2024
ANSR trees are from an older BehaVerify and do not parse at all against the
current grammar ("Expected 'hypersafety' or 'use_reals'"). (2) BOTH the
2026_TBA template and the PUBLISHED 2025_NEUS grid world trees fail validation
on current BehaVerify with "In Check: need_destination -> Expected a variable
in scope [blackboard] but got destination_x in scope environment". That is a
regression in the tool against its own published reproducibility artifact and
deserves its own report. --no_checks is used here because this measures SMV
GENERATION COST, not model validity.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "REPRODUCIBILITY/2025_NEUS/examples/grid_world/counter_1.tree"
# The 100%-accurate network, so the invariant is TRUE and BOTH encodings must
# PROVE rather than refute. The 0995 network makes it false at larger grids,
# and refuting is the cheap direction - one counterexample against all states -
# so those rows measured the wrong thing.
NETWORK = ROOT / "REPRODUCIBILITY/2026_TBA/examples/grid_world/networks/1000__6_18_0__0200_1.onnx"
OUT = pathlib.Path(__file__).resolve().parent / "generated_true"

ENCODINGS = {
    "table": "table",
    "float32": "float 32",
    "fixed_100_35": "fixed 100 35",
}


def build_tree(encoding_clause, max_val, dest):
    """Write a .tree with one encoding and one grid size."""
    text = TEMPLATE.read_text()
    # The source is resolved RELATIVE to the .tree file's own directory
    # (dsl_to_nuxmv.py: file_prefix = model_file.rsplit('/', 1)[0]), so an
    # absolute path gets concatenated onto it. The network is copied next to
    # the generated trees and referenced by bare filename.
    text = text.replace("'./networks/0995__6_18_0__200_1.onnx'", f"'{NETWORK.name}'")
    text = re.sub(r"config \{\s*table\s*\}", f"config {{ {encoding_clause} }}", text)
    text = re.sub(r"max_val\s*:=\s*\d+", f"max_val := {max_val}", text)
    dest.write_text(text)
    return dest


def generate(tree_path, out_dir, timeout=1800):
    """Run BehaVerify's generator and time it. Returns (status, seconds, size)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "behaverify", "nuxmv", str(tree_path),
             str(out_dir), "--generate", "--overwrite", "--no_checks",
             "--recursion_limit", "20000"],
            capture_output=True, text=True, timeout=timeout, cwd=ROOT, check=False)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start, 0
    secs = time.time() - start
    smv = sorted(out_dir.rglob("*.smv"))
    size = smv[0].stat().st_size if smv else 0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return f"FAILED: {tail[-1][:70] if tail else '?'}", secs, 0
    return "ok", secs, size


def main():
    """Sweep encoding x grid size and print one table with provenance."""
    OUT.mkdir(exist_ok=True)
    print(f"encoding scaling | grid world counter_template | "
          f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"network {NETWORK.name}\n")
    print(f"{'encoding':>14} {'max_val':>8} {'points':>9} {'status':>34} "
          f"{'gen s':>8} {'smv MB':>8}")
    for name, clause in ENCODINGS.items():
        for max_val in (4, 6, 9, 14):
            points = (max_val + 1) ** 4
            tree = build_tree(clause, max_val, OUT / f"{name}_{max_val}.tree")
            status, secs, size = generate(tree, OUT / f"{name}_{max_val}")
            print(f"{name:>14} {max_val:>8} {points:>9} {status:>34} "
                  f"{secs:>8.1f} {size / 1e6:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
