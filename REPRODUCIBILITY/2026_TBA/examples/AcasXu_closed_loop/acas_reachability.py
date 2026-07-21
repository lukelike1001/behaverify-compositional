"""
acas_reachability.py

BFS over the augmented (state, a_prev) space, without querying any network.
Each tick's advisory is treated as nondeterministic (any of the 5), which can
only enlarge the reachable set relative to the real system -- sound over-approx
for classifying and pruning contracts.

Uses AcasDomain + AcasAugmentedState. Orthogonal to acas_viability.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from acas_domain import AcasDomain
from acas_state import AcasAugmentedState, AcasState

# Edge key for abstract BFS: (augmented state, next advisory). Same type whether
# the edge is allowed or blocked -- distinction is set membership only.
BlockedTransition = tuple[AcasAugmentedState, str]


@dataclass(frozen=True)
class AcasReachableSet:
    """
    Over-approximate reachable set of augmented states from the closed-loop seed.

    Construct with ``AcasReachableSet.compute(domain, blocked=...)``.
    """

    domain: AcasDomain
    states: frozenset[AcasAugmentedState]
    seed: AcasAugmentedState
    blocked: frozenset[BlockedTransition]

    @staticmethod
    def seed_state(domain: AcasDomain | None = None) -> AcasAugmentedState:
        """Closed-loop seed from acas_model_params.yaml (via AcasDomain)."""
        plant = domain if domain is not None else AcasDomain.from_yaml()
        physical, a_prev = plant.seed_physical_and_aprev()
        return AcasAugmentedState.from_physical(physical, a_prev)

    @classmethod
    def compute(
        cls,
        domain: AcasDomain | None = None,
        blocked: frozenset[BlockedTransition] | None = None,
    ) -> AcasReachableSet:
        """
        BFS from the seed. ``blocked`` forbids (from_state, next_advisory)
        transitions (e.g. an injected A/G contract when stress-testing a corridor).
        """
        plant = domain if domain is not None else AcasDomain.from_yaml()
        blocked_set: frozenset[BlockedTransition] = (
            blocked if blocked is not None else frozenset()
        )
        start = cls.seed_state(plant)
        reachable: set[AcasAugmentedState] = {start}
        frontier = [start]
        advisories = list(plant.advisories)

        while frontier:
            next_frontier: list[AcasAugmentedState] = []
            for state in frontier:
                for advisory in advisories:
                    if (state, advisory) in blocked_set:
                        continue
                    next_physical = plant.simulate_step(
                        state.x_mag,
                        state.y_mag,
                        state.x_sign,
                        state.y_sign,
                        state.heading_own_var,
                        advisory,
                    )
                    next_state = AcasAugmentedState.from_physical(
                        next_physical, advisory,
                    )
                    if next_state not in reachable:
                        reachable.add(next_state)
                        next_frontier.append(next_state)
            frontier = next_frontier

        return cls(
            domain=plant,
            states=frozenset(reachable),
            seed=start,
            blocked=blocked_set,
        )

    def __len__(self) -> int:
        return len(self.states)

    def __contains__(self, state: AcasAugmentedState) -> bool:
        return state in self.states

    def dangerous_xy(
        self,
        contract: dict[str, Any],
        advisory: str,
    ) -> list[tuple[int, int]]:
        """
        (x_mag, y_mag) pairs in the contract that appear in this reachable set
        under a_prev=advisory.
        """
        x_sign = contract["x_sign"]
        y_sign = contract["y_sign"]
        heading = contract["heading_own_var"]
        return [
            (x_mag, y_mag)
            for x_mag, y_mag in contract["dangerous_xy"]
            if AcasAugmentedState(
                x_mag, y_mag, x_sign, y_sign, heading, advisory,
            ) in self.states
        ]

    def physical_by_aprev(self) -> dict[str, frozenset[AcasState]]:
        """Map a_prev -> plant states that appear with that a_prev."""
        by_aprev: dict[str, set[AcasState]] = {
            name: set() for name in self.domain.advisories
        }
        for state in self.states:
            by_aprev[state.a_prev].add(state.physical())
        return {
            name: frozenset(physical_states)
            for name, physical_states in by_aprev.items()
        }


if __name__ == "__main__":
    reachable_set = AcasReachableSet.compute()
    print(f"{len(reachable_set)} reachable states out of 96,800 possible")
