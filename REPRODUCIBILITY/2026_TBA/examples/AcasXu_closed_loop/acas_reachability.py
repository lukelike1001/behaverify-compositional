"""
acas_reachability.py

BFS over the augmented (state, a_prev) space, without querying any network.
Each tick's advisory is treated as nondeterministic (any of the 5), which can
only enlarge the reachable set relative to the real system -- so the result
is a sound over-approximation, safe to use for both classifying and pruning
contracts.

Augmented state: (x_mag, y_mag, x_sign, y_sign, heading_own_var, a_prev).
"""

from __future__ import annotations

from generate_acas_contracts import ADVISORIES, _P, simulate_step

AugmentedState = tuple[int, int, int, int, int, str]


def _initial_state() -> AugmentedState:
    s = _P["initial_state"]
    return (s["x_mag"], s["y_mag"], s["x_sign"], s["y_sign"], s["heading_own_var"], s["a_prev"])


def compute_reachable_states(blocked: frozenset[tuple[AugmentedState, str]] = frozenset()) -> set[AugmentedState]:
    """
    Reachable augmented states from the fixed initial condition.

    `blocked` forbids specific (from_state, next_advisory) transitions -- used to model
    an injected A/G contract when checking whether it closes off a spurious counterexample.
    """
    start = _initial_state()
    reachable = {start}
    frontier = [start]

    while frontier:
        next_frontier = []
        for state in frontier:
            x_mag, y_mag, x_sign, y_sign, heading_own_var, _a_prev = state
            for advisory in ADVISORIES:
                if (state, advisory) in blocked:
                    continue
                next_state = (*simulate_step(x_mag, y_mag, x_sign, y_sign, heading_own_var, advisory), advisory)
                if next_state not in reachable:
                    reachable.add(next_state)
                    next_frontier.append(next_state)
        frontier = next_frontier

    return reachable


def reachable_dangerous_xy(contract: dict, advisory: str, reachable: set[AugmentedState]) -> list[tuple[int, int]]:
    """Contract's dangerous (x_mag, y_mag) states that are actually reachable under this advisory."""
    x_sign, y_sign, heading = contract["x_sign"], contract["y_sign"], contract["heading_own_var"]
    return [
        (x_mag, y_mag) for x_mag, y_mag in contract["dangerous_xy"]
        if (x_mag, y_mag, x_sign, y_sign, heading, advisory) in reachable
    ]


if __name__ == "__main__":
    reachable = compute_reachable_states()
    print(f"{len(reachable)} reachable states out of 96,800 possible")
