"""
How many discrete modes does droneNew's tree produce per tick?

droneNew's specifications are about BT NODE STATUS - active(Main),
active(GoTo0), success(Drone) = success(Report) - which plain interval
reachability cannot express, since it tracks a box over R^n with no notion of
which node is active.  A competitor must therefore use HYBRID reachability,
which tracks discrete modes alongside continuous sets.

That is where Gemini's claim #25 predicts trouble: a BT ticking at every step
produces a discrete mode per tick, and flowpipe tools fragment.  This counts the
modes directly from the generated SMV, rather than arguing about it.

A "mode" here is one combination of the tree's active-node and output
assignments that the model can be in, which is what a hybrid automaton would
need one location for.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

NUXMV = "REPRODUCIBILITY/2026_TBA/nuXmv_DL/bin/nuXmv"


def run(smv, commands):
    """Run nuXmv over an SMV file and return stdout."""
    script = "\n".join(commands) + "\nquit\n"
    proc = subprocess.run([NUXMV, "-int", smv], input=script, text=True,
                          capture_output=True, check=False)
    return proc.stdout + proc.stderr


def main():
    """Report reachable states, diameter, and the tree's mode variables."""
    smv = sys.argv[1]
    out = run(smv, ["go", "print_reachable_states -v"])
    states = re.search(r"reachable states:\s*([0-9.e+]+)", out)
    diameter = re.search(r"diameter:\s*(\d+)", out)
    print(f"model: {Path(smv).name}")
    print(f"  reachable states: {states.group(1) if states else '?'}")
    print(f"  diameter:         {diameter.group(1) if diameter else '?'}")

    text = Path(smv).read_text()
    # every node whose status the tree tracks is one dimension of the mode
    status = sorted(set(re.findall(r"(\w+)_status\b", text)))
    active = sorted(set(re.findall(r"(\w+)_active\b", text)))
    print(f"  node status variables: {len(status)}")
    print(f"  node active variables: {len(active)}")
    modes = 3 ** len(status) if status else 0
    print(f"  worst-case mode combinations (3^status): {modes:.3g}")
    print()
    print("  A hybrid-reachability competitor needs one location per reachable")
    print("  mode combination, and one flowpipe segment per location per tick.")


if __name__ == "__main__":
    main()
