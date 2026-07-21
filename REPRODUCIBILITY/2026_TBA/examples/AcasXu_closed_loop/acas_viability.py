"""
acas_viability.py

Viability kernel V: largest set of plant states from which SOME advisory
sequence stays safe forever (greatest fixpoint). Uses only AcasDomain physics
-- no networks, no a_prev, no contract generation.

Orthogonal to acas_reachability.py: V is geometry (can stay safe); R is the
controller (what the closed loop may visit).

Plant state type: AcasState (see acas_state.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from acas_domain import AcasDomain
from acas_state import AcasState


@dataclass(frozen=True)
class AcasViabilityKernel:
    """
    Safety/viability partition of the plant lattice.

    Classify every AcasState as unsafe, safe-but-doomed, interior(V),
    or boundary(V); expose Allowed_V(s) for s in V.
    """

    domain: AcasDomain
    V: frozenset[AcasState]
    interior: frozenset[AcasState]
    boundary: frozenset[AcasState]
    safe_but_doomed: frozenset[AcasState]
    unsafe: frozenset[AcasState]
    allowed: dict[AcasState, list[str]]
    successors: dict[AcasState, dict[str, AcasState]] = field(
        repr=False, compare=False,
    )

    @classmethod
    def compute(cls, domain: AcasDomain | None = None) -> AcasViabilityKernel:
        """Greatest fixpoint of V = {s safe : exists a with step(s,a) in V}."""
        if domain is None:
            domain = AcasDomain.from_yaml()

        all_physical = [
            AcasState.from_tuple(values)
            for values in domain.all_physical_states()
        ]
        advisories = list(domain.advisories)
        successors: dict[AcasState, dict[str, AcasState]] = {
            state: {
                advisory: AcasState.from_tuple(
                    domain.simulate_step(*state, advisory),
                )
                for advisory in advisories
            }
            for state in all_physical
        }

        safe = frozenset(
            state for state in all_physical
            if domain.is_safe(state.x_mag, state.y_mag)
        )
        unsafe = frozenset(all_physical) - safe

        kernel: set[AcasState] = set(safe)
        while True:
            doomed = {
                state for state in kernel
                if not any(
                    successors[state][advisory] in kernel
                    for advisory in advisories
                )
            }
            if not doomed:
                break
            kernel -= doomed
        viable = frozenset(kernel)

        allowed = {
            state: [
                advisory for advisory in advisories
                if successors[state][advisory] in viable
            ]
            for state in viable
        }
        interior = frozenset(
            state for state, acts in allowed.items()
            if len(acts) == len(advisories)
        )
        boundary = viable - interior
        safe_but_doomed = safe - viable

        return cls(
            domain=domain,
            V=viable,
            interior=interior,
            boundary=boundary,
            safe_but_doomed=safe_but_doomed,
            unsafe=unsafe,
            allowed=allowed,
            successors=successors,
        )

    @property
    def all_physical_states(self) -> frozenset[AcasState]:
        """Full plant lattice (keys of the successor table)."""
        return frozenset(self.successors)

    def is_plant_safe(self, state: AcasState | tuple) -> bool:
        """True iff rho(state) meets the domain safety threshold (not membership in V)."""
        return self.domain.is_safe(state[0], state[1])


if __name__ == "__main__":
    kernel = AcasViabilityKernel.compute()
    num_physical = len(kernel.all_physical_states)
    print(f"|V| = {len(kernel.V)} out of {num_physical} physical states")
    print(
        f"interior = {len(kernel.interior)}, boundary = {len(kernel.boundary)}, "
        f"safe-but-doomed = {len(kernel.safe_but_doomed)}, "
        f"unsafe = {len(kernel.unsafe)}"
    )

