"""
Independent audit of the contracts nuXmv actually reads.

Every other soundness check re-derives the bounds by calling affine_expr, the
same code path the generator uses, so a bug inside affine_expr would be
invisible to all of them.  This one PARSES THE EMITTED .smv FILE, evaluates the
guards and the bound expressions as written, and checks them against the true
network and the true displacement.  It shares no code with the generator beyond
the network itself.

Usage: python3 COLLAB/experiments/nav_sound/smv_contract_audit.py <model.smv> [n]
"""
import random
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort

NET = "REPRODUCIBILITY/2026_TBA/examples/NAV/networks/nn-nav-set.onnx"
DT = 0.2
INVAR = re.compile(r"^INVAR \((.+?)\) -> \((.+)\);$")
RANGE = re.compile(r"(\w+) >= (-?[\d.]+) & \1 <= (-?[\d.]+)")
BOUND = re.compile(r"(\w+) (>=|<=) ([^&]+?)(?= & |$)")


def parse(path):
    """Yield (guard_ranges, [(var, op, expr)]) for every INVAR with a guard."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    for line in lines:
        m = INVAR.match(line.strip())
        if not m:
            continue
        guard = {v: (float(a), float(b)) for v, a, b in RANGE.findall(m.group(1))}
        if not guard:
            continue
        yield guard, BOUND.findall(m.group(2))


def sample_in(guard, rng):
    """A random point satisfying the guard; unconstrained axes get a default."""
    full = {"x": (-1.0, 3.5), "y": (-1.0, 3.5), "v": (-2.0, 2.0),
            "th": (-np.pi, np.pi)}
    return {k: rng.uniform(*guard.get(k, lims))
            for k, lims in full.items()}


def main():  # pylint: disable=too-many-locals
    """Check every sampled point against the contracts as written."""
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    sess = ort.InferenceSession(NET)
    inp = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)

    rules = list(parse(path))
    random.Random(0).shuffle(rules)
    viol = checked = 0
    worst = -np.inf
    for guard, bounds in rules[:n]:
        for _ in range(25):
            pt = sample_in(guard, rng)
            state = np.array([[pt["x"], pt["y"], pt["v"], pt["th"]]], np.float32)
            out = sess.run(None, {inp: state})[0].ravel()
            truth = {"u1": float(out[0]), "u2": float(out[1]),
                     "dx": float(pt["v"] * np.cos(pt["th"]) * DT),
                     "dy": float(pt["v"] * np.sin(pt["th"]) * DT)}
            for var, op, expr in bounds:
                if var not in truth:
                    continue
                val = float(eval(expr.strip(), {"__builtins__": {}}, dict(pt)))  # noqa: S307
                slack = (val - truth[var]) if op == ">=" else (truth[var] - val)
                worst = max(worst, slack)
                if slack > 0:
                    viol += 1
                checked += 1
    print(f"{path}")
    print(f"  {len(rules)} guarded INVARs in file; audited {min(n, len(rules))} "
          f"x 25 points = {checked} bound checks")
    print(f"  violations: {viol}")
    print(f"  tightest:   {worst:+.3e} (negative = slack; positive = VIOLATION)")
    print("  CONTRACTS AS WRITTEN ARE SOUND" if viol == 0
          else "  *** EMITTED CONTRACTS ARE UNSOUND ***")


if __name__ == "__main__":
    main()
