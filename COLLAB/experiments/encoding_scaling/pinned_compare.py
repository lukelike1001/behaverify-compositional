"""
The encoding comparison done the way the artifact intends.

Everything before this in the directory was me reimplementing, badly, a harness
that ships with the paper. REPRODUCIBILITY/2025_NEUS/examples/grid_world has
make_tree.sh, which iterates modes = (table, fixed 140 48, fixed 100 35,
float 140) over every network, filling REPLACE_CONFIG and REPLACE_SOURCE in
template.tree. counter_1.tree is not a benchmark input; it is an artifact of
something else, and feeding it to the compiler is what produced three
successive wrong diagnoses tonight - "a regression", then "the wrong source
tree", then neither.

This uses the artifact's own generated trees and its own PINNED src/, so
validation stays ON and the model is the one the paper verified.

The 1000__ networks are 100 percent accurate, so the invariant is TRUE and
every encoding must PROVE rather than refute; refuting is the cheap direction
and would flatter whichever encoding found a counterexample first.
"""
from __future__ import annotations

import pathlib
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEUS = ROOT / "REPRODUCIBILITY/2025_NEUS"
GW = NEUS / "examples/grid_world"
NUXMV = ROOT / "REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv"
NET = "1000__6_18_0__0200_1"
MODES = ["table", "fixed_100_35", "fixed_140_48", "float_140"]
TIMEOUT = 900


def generate(tree, smv):
    """Compile with the PINNED source, validation on. Returns (ok, seconds)."""
    start = time.time()
    proc = subprocess.run(
        ["python", str(NEUS / "src/dsl_to_nuxmv.py"),
         str(NEUS / "metamodel/behaverify.tx"), str(tree), str(smv),
         "--recursion_limit", "20000"],
        capture_output=True, text=True, timeout=TIMEOUT, check=False)
    return proc.returncode == 0, time.time() - start


def verify(smv):
    """Build and check the invariant; report the verdict honestly."""
    script = HERE / "cmd.nuxmv"
    script.write_text("go\ncheck_invar\nquit\n")
    start = time.time()
    try:
        proc = subprocess.run([str(NUXMV), "-source", str(script), str(smv)],
                              capture_output=True, text=True, timeout=TIMEOUT,
                              check=False)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start
    out = proc.stdout + proc.stderr
    secs = time.time() - start
    if "is false" in out:
        return "false", secs
    if "is true" in out:
        return "true", secs
    return "?", secs


def main():
    """One table: every encoding, same network, same property, pinned source."""
    print(f"grid world 7x7 | network {NET} (100% accurate, invariant TRUE)")
    print(f"pinned source {NEUS.relative_to(ROOT)}/src | "
          f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    print(f"{'encoding':>14} {'gen s':>8} {'smv MB':>8} {'verdict':>9} {'verify s':>9}")
    for mode in MODES:
        tree = GW / "tree" / f"{mode}_{NET}.tree"
        if not tree.exists():
            print(f"{mode:>14} {'-':>8} {'-':>8} {'no tree':>9} {'-':>9}")
            continue
        smv = HERE / "pinned" / f"{mode}_{NET}.smv"
        smv.parent.mkdir(exist_ok=True)
        ok, gsec = generate(tree, smv)
        if not ok or not smv.exists():
            print(f"{mode:>14} {gsec:>8.1f} {'-':>8} {'gen failed':>9} {'-':>9}")
            continue
        verdict, vsec = verify(smv)
        print(f"{mode:>14} {gsec:>8.1f} {smv.stat().st_size / 1e6:>8.2f} "
              f"{verdict:>9} {vsec:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
