"""
The axis that actually decides kill condition 1: nuXmv VERIFICATION time.

Generation cost is already measured and it kills half the argument: the table's
SMV grows 69x from 625 to 50,625 input points while fixed-point and float stay
flat, because they encode the WEIGHTS rather than the input space. So "the
table cannot scale" is not a contribution on model size.

But model size is not verification. NeuS Table 3 measured both on ONE grid and
found the table flat at 0.07s across network sizes while fixed-point ran
53-153s and grew with the NETWORK. Nobody has measured either as the INPUT
SPACE grows, and Gemini's claim #119 asserts fixed-point verification explodes
4.8x from 81 to 625 points and times out at 2,401 - which, if true, points the
opposite way from my generation numbers.

One of those decides the direction, so it gets measured rather than argued.
"""
from __future__ import annotations

import pathlib
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NUXMV = ROOT / "REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv"
CASES = [("table", 4), ("table", 6), ("table", 9), ("table", 14),
         ("fixed_100_35", 4), ("fixed_100_35", 6), ("fixed_100_35", 9),
         ("fixed_100_35", 14)]
TIMEOUT = 900


def verify(smv, timeout=TIMEOUT):
    """Build the model and check its invariant; report wall clock honestly."""
    script = HERE / "cmd.nuxmv"
    script.write_text("go\ncheck_invar\nquit\n")
    start = time.time()
    try:
        proc = subprocess.run([str(NUXMV), "-source", str(script), str(smv)],
                              capture_output=True, text=True, timeout=timeout,
                              check=False)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start
    secs = time.time() - start
    out = proc.stdout + proc.stderr
    if "is false" in out:
        return "false", secs
    if "is true" in out:
        return "true", secs
    return f"? {out.strip().splitlines()[-1][:40] if out.strip() else 'no output'}", secs


def main():
    """One table: encoding x input-space size, generation already done."""
    print(f"nuXmv verification | grid world counter_1 | 1000__ (100% acc, property TRUE) | "
          f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"nuXmv at {NUXMV}, timeout {TIMEOUT}s\n")
    print(f"{'encoding':>14} {'points':>9} {'smv MB':>8} {'verdict':>10} {'sec':>9}")
    for name, max_val in CASES:
        smvs = sorted((HERE / "generated_true" / f"{name}_{max_val}").rglob("*.smv"))
        if not smvs:
            print(f"{name:>14} {(max_val+1)**4:>9} {'-':>8} {'no smv':>10} {'-':>9}")
            continue
        size = smvs[0].stat().st_size / 1e6
        verdict, secs = verify(smvs[0])
        print(f"{name:>14} {(max_val+1)**4:>9} {size:>8.2f} {verdict:>10} "
              f"{secs:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
