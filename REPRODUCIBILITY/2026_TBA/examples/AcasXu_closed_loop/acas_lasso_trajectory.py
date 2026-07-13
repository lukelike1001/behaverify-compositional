"""
acas_lasso_trajectory.py

Ground-truth closed-loop trajectory: real ONNX inference (no CROWN, no abstraction),
starting from the seed, until the trajectory closes into a cycle (a lasso).
Independently checks two things the inductive-invariant analysis depends on:

  1. The 39-tick stem / 13-tick cycle / 52-state lasso and its exact minimum
     distance, claimed in 2026_07_12_inductive_invariant_stress_test.md Section 7.
  2. Whether the Q2 INVAR's antecedent state (the seed) ever recurs later in the
     trajectory -- if it does, a single-shot INVAR at the seed is not sound; if it
     doesn't, tick 0 is the only place that constraint needs to hold.

Run from examples/AcasXu_closed_loop/:

    python3 acas_lasso_trajectory.py
"""

import sys
from pathlib import Path

import numpy as np

import generate_acas_contracts as g
from acas_reachability import _initial_state

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime is required: pip install onnxruntime")

_HERE = Path(__file__).parent


def _run_onnx(session: "ort.InferenceSession", inputs: np.ndarray) -> np.ndarray:
    """ACAS Xu ONNX files use input shape [1, 1, 1, 5] (legacy conv wrapper);
    matches figures/image_scripts/acas_output_property.py's run_onnx()."""
    x = inputs.astype(np.float32).reshape(1, 1, 1, -1)
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: x})[0][0]


def load_sessions() -> dict[str, "ort.InferenceSession"]:
    return {name: ort.InferenceSession(str(_HERE / onnx_rel)) for name, (_idx, onnx_rel) in g.A_PREV_TO_NN.items()}


def choose_advisory(session, x_mag, y_mag, x_sign, y_sign, heading_own_var) -> str:
    inputs = np.array(g.compute_nn_inputs(x_mag, y_mag, x_sign, y_sign, heading_own_var))
    scores = _run_onnx(session, inputs)
    return g.ADVISORIES[int(np.argmax(scores))]


def main() -> None:
    sessions = load_sessions()
    seed = _initial_state()  # (x_mag, y_mag, x_sign, y_sign, heading_own_var, a_prev)
    state, a_prev = seed[:5], seed[5]

    trajectory = [(state, a_prev)]
    seen = {(state, a_prev): 0}
    min_dist = g.compute_distance(state[0], state[1])

    tick = 0
    while True:
        tick += 1
        advisory = choose_advisory(sessions[a_prev], *state)
        state, a_prev = g.simulate_step(*state, advisory), advisory
        min_dist = min(min_dist, g.compute_distance(state[0], state[1]))

        key = (state, a_prev)
        trajectory.append(key)
        if key in seen:
            cycle_start = seen[key]
            break
        seen[key] = tick

    total_states = len(trajectory) - 1  # exclude the repeated closing state
    stem_len = cycle_start
    cycle_len = total_states - stem_len
    seed_recurs = (cycle_start == 0)

    print(f"[CHECK] stem length = {stem_len} ticks (report claims 39)")
    print(f"[CHECK] cycle length = {cycle_len} ticks (report claims 13)")
    print(f"[CHECK] total distinct augmented states = {total_states} (report claims 52)")
    print(f"[CHECK] minimum distance over trajectory = {min_dist} (report claims exactly 200)")
    print(f"[CHECK] seed state recurs after tick 0: {seed_recurs}")

    tick0_advisory = choose_advisory(sessions[seed[5]], *seed[:5])
    holds = tick0_advisory != 'strong_right'
    print(f"[CHECK] tick-0 advisory chosen by '{seed[5]}' network at the seed: {tick0_advisory} "
          f"(Q2 requires != strong_right: {'HOLDS' if holds else 'VIOLATED'})")


if __name__ == "__main__":
    main()
