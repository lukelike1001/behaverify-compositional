"""
acas_inductive_analysis.py

Inductive-invariant stress test for the simplified ACAS Xu closed loop.
Companion to 2026_07_12_inductive_invariant_stress_test.md.

Computes, using ONLY the physics in generate_acas_contracts.py (no networks):

  1. R      -- reachable augmented states (state, a_prev) under nondeterministic
               advisories, seeded from the closed loop's initial condition.
  2. V      -- the viability kernel: the largest set of safe physical states from
               which SOME advisory sequence remains safe forever (greatest fixpoint).
  3. dV     -- the kernel boundary: states in V where at least one advisory exits V.
  4. Allowed_V(s) = {a : step(s,a) in V} and its histogram.
  5. The spurious counterexample corridor: the unique path in R from the seed to
     the unique unsafe pair, and the (at most two) CROWN contracts that sever it.
  6. Direct model checks: abstract reachability with each candidate contract
     injected, confirming INVARSPEC TRUE.

Run from examples/AcasXu_closed_loop/:

    python3 acas_inductive_analysis.py

Every printed [CHECK] line is an assertion of a claim in the companion report.
"""

from collections import Counter

from acas_reachability import _initial_state, compute_reachable_states
from acas_viability import ADVISORIES, ALL_PHYS, SUCC, is_safe, compute_viability_kernel

_seed = _initial_state()                 # single source of truth: acas_model_params.yaml
SEED = (_seed[:5], _seed[5])             # ((x_mag, y_mag, x_sign, y_sign, h), a_prev)


def _to_pair(flat_state):
    """Convert acas_reachability's flat 6-tuple to this file's (physical_state, a_prev) pair."""
    return (flat_state[:5], flat_state[5])


def _to_flat(pair):
    """Inverse of _to_pair."""
    return (*pair[0], pair[1])


def reachable_pairs(blocked=frozenset()):
    """(physical_state, a_prev) pairs reachable from the seed. Thin wrapper around
    acas_reachability.compute_reachable_states(), converting between its flat 6-tuple
    representation and this file's nested-pair representation."""
    blocked_flat = frozenset((_to_flat(pair), a) for pair, a in blocked)
    return {_to_pair(flat) for flat in compute_reachable_states(blocked=blocked_flat)}


def _walk_corridor(R, unsafe):
    """Walk unique predecessors back from the unsafe pair to the seed."""
    assert len(unsafe) == 1
    corridor = [unsafe[0]]
    while True:
        (s2, p2) = corridor[-1]
        preds = [(s, p) for (s, p) in R if SUCC[s][p2] == s2 and (s, p) != (s2, p2)]
        print(f"[CHECK] predecessors of {corridor[-1]}: {preds}")
        if len(preds) != 1:
            break
        corridor.append(preds[0])
        if corridor[-1][0] == SEED[0]:
            break
    corridor.reverse()
    return corridor


def main() -> None:
    R = reachable_pairs()
    unsafe = [(s, p) for (s, p) in R if not is_safe(s)]
    print(f"[CHECK] |R| = {len(R)} (expect 9428); unsafe pairs = {unsafe}")

    kernel = compute_viability_kernel()
    V = kernel.V
    r_phys = {s for (s, _p) in R}
    hist = Counter(len(v) for v in kernel.allowed.values())
    print(f"[CHECK] |V| = {len(V)}; |boundary| = {len(kernel.boundary)} "
          f"({100 * len(kernel.boundary) / len(V):.2f}% of V)")
    print(f"[CHECK] |Allowed_V| histogram: {dict(sorted(hist.items()))}")
    print(f"[CHECK] boundary ∩ R_phys = {sorted(kernel.boundary & r_phys)}")

    # inductive candidate I0 and its single NN-dependent obligation
    I0 = {(s, p) for (s, p) in R if s in V}
    need_nn = sorted((s, p) for (s, p) in I0
                     if any(SUCC[s][a] not in V for a in ADVISORIES))
    print(f"[CHECK] |I0| = {len(I0)}; seed in I0: {SEED in I0}; "
          f"pairs needing NN info: {need_nn}")

    corridor = _walk_corridor(R, unsafe)
    print(f"[CHECK] corridor (seed -> crash): {corridor}")

    # candidate injections
    mid = corridor[1]      # ((5,3,1,1,8), 'strong_right') expected
    for label, blocked in (
        ("Q1 at mid", {(mid, 'strong_right')}),
        ("Q2 at seed", {(SEED, 'strong_right')}),
    ):
        RR = reachable_pairs(blocked=frozenset(blocked))
        verdict = all(is_safe(s) for (s, _p) in RR)
        print(f"[CHECK] inject {label}: |R| = {len(RR)}, "
              f"INVARSPEC {'TRUE' if verdict else 'FALSE'}")


if __name__ == '__main__':
    main()
