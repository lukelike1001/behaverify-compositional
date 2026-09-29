"""
How deep does each contract form get before a SPURIOUS counterexample?

That is the only comparison that matters.  Affine contracts cost more solver
time per constraint but need far fewer constraints for a given precision, so
neither the contract count nor the per-query time decides it on its own - the
depth reached before the abstraction goes spurious does.

Usage: python3 COLLAB/experiments/nav_sound/z3_depth_scan.py <nx> <nv> <nvd> <maxk> <budget_s>
"""
import subprocess
import sys

SCRIPT = "COLLAB/experiments/nav_sound/z3_bmc.py"


def main():
    """Scan increasing BMC depth for both encodings and report the frontier."""
    nx, nv, nvd = sys.argv[1], sys.argv[2], sys.argv[3]
    maxk = int(sys.argv[4])
    budget = int(sys.argv[5]) if len(sys.argv) > 5 else 600
    for kind in ("interval", "affine"):
        print(f"\n=== {kind}, net {nx}x{nv}, disp {nvd} ===")
        for k in range(2, maxk + 1):
            try:
                out = subprocess.run(
                    [sys.executable, SCRIPT, nx, nv, nvd, kind, str(k)],
                    capture_output=True, text=True, timeout=budget, check=False)
            except subprocess.TimeoutExpired:
                print(f"  k={k:<3} TIMEOUT (>{budget}s)")
                break
            line = [ln for ln in out.stdout.splitlines() if "unsat" in ln
                    or "sat" in ln or "unknown" in ln]
            if not line:
                print(f"  k={k:<3} no result\n{out.stdout}{out.stderr[-400:]}")
                break
            print(f"  k={k:<3}{line[0].strip()}")
            if "unsat" not in line[0]:
                print(f"  -> first counterexample at k={k}")
                break


if __name__ == "__main__":
    main()
