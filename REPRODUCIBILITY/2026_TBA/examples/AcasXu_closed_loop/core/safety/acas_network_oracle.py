"""
acas_network_oracle.py

What the five ONNX networks actually output at a given augmented state.

Real inference, no CROWN and no abstraction. AcasContractMinCut uses it to
label which graph edges the network takes; a cut edge still needs a CROWN
certificate before it may be injected, so this is a selection aid, not a proof.

Usage (from AcasXu_closed_loop/):

    python3 -m core.safety.acas_network_oracle
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from core.acas_domain import AcasDomain
from core.acas_state import AcasAugmentedState
from core.paths import EXAMPLE_ROOT


@dataclass
class AcasNetworkOracle:
    """One ONNX session per a_prev, plus normalized-input construction."""

    domain: AcasDomain
    sessions: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, domain: AcasDomain | None = None) -> AcasNetworkOracle:
        import onnxruntime as ort  # noqa: PLC0415

        if domain is None:
            domain = AcasDomain.from_yaml()
        sessions = {
            # a_prev_to_nn stores paths already relative to the example root
            # (e.g. "networks/aprev_clear.onnx"), so do not prepend "networks".
            a_prev: ort.InferenceSession(
                str(EXAMPLE_ROOT / onnx_name),
                providers=["CPUExecutionProvider"],
            )
            for a_prev, (_idx, onnx_name) in domain.a_prev_to_nn.items()
        }
        return cls(domain=domain, sessions=sessions)

    @cached_property
    def advisory_by_index(self) -> tuple[str, ...]:
        return tuple(self.domain.advisories)

    def advisory_at(self, state: AcasAugmentedState) -> str:
        """The advisory NN_{a_prev} selects at this state (argmax of scores)."""
        return self.advisories_for([state])[state]

    def advisories_for(
        self, states: Iterable[AcasAugmentedState],
    ) -> dict[AcasAugmentedState, str]:
        """
        Argmax for many states, grouped by a_prev so each session is fetched
        once.

        These ONNX models declare input shape [1, 1, 1, 5], so the batch
        dimension is fixed and every state needs its own call. Distinct states
        can share NN inputs (the five normalized features do not separate
        every lattice cell), so scores are memoized on those features.
        """
        import numpy as np  # noqa: PLC0415

        by_a_prev: dict[str, list[AcasAugmentedState]] = {}
        for state in states:
            by_a_prev.setdefault(state.a_prev, []).append(state)

        result: dict[AcasAugmentedState, str] = {}
        for a_prev, group in by_a_prev.items():
            session = self.sessions[a_prev]
            input_name = session.get_inputs()[0].name
            memo: dict[tuple[float, ...], str] = {}
            for state in group:
                features = tuple(self.domain.compute_nn_inputs(
                    state.x_mag, state.y_mag, state.x_sign, state.y_sign,
                    state.heading_own_var,
                ))
                advisory = memo.get(features)
                if advisory is None:
                    scores = session.run(None, {
                        input_name: np.array(
                            features, dtype=np.float32,
                        ).reshape(1, 1, 1, 5),
                    })[0]
                    advisory = self.advisory_by_index[
                        int(np.asarray(scores).reshape(-1).argmax())
                    ]
                    memo[features] = advisory
                result[state] = advisory
        return result


def main() -> None:
    from core.acas_reachability import AcasReachableSet  # noqa: PLC0415

    domain = AcasDomain.from_yaml()
    reachable = AcasReachableSet.compute(domain)
    oracle = AcasNetworkOracle.build(domain)
    advisories = oracle.advisories_for(reachable.states)

    counts: dict[str, int] = {}
    for advisory in advisories.values():
        counts[advisory] = counts.get(advisory, 0) + 1

    print(f"reachable states = {len(reachable.states)}")
    print("network output distribution over R:")
    for advisory in domain.advisories:
        print(f"  {advisory:<14} {counts.get(advisory, 0)}")


if __name__ == "__main__":
    main()
