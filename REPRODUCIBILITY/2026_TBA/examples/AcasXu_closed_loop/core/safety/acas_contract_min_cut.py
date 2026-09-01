"""
acas_contract_min_cut.py

Smallest contract set that establishes the safety invariant, computed as a
min-cut of the abstract reachability graph.

Nodes are augmented states reachable from the seed under a free network; an
edge u -a-> v exists for each advisory a. Edges the network actually takes get
infinite capacity (they cannot be forbidden); every other edge gets capacity 1
(CROWN can certify it). Max-flow then returns the fewest contracts that
separate the seed from the unsafe states.

Correctness argument, corollaries, and instance numbers:
reports/Acas_Xu_closed_loop/2026_09_01_min_cut_contracts.md

Real advisories come in as a dict, so this module stays free of onnxruntime;
AcasNetworkOracle supplies them.

Usage (from AcasXu_closed_loop/):

    python3 -m core.safety.acas_contract_min_cut
    python3 -m core.safety.acas_contract_min_cut --output cut_contracts.json
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.acas_domain import AcasDomain
from core.acas_reachability import AcasReachableSet
from core.acas_state import AcasAugmentedState
from core.acas_viability import AcasViabilityKernel
from core.paths import EXAMPLE_ROOT

# Sentinel node ids for the flow network.
SOURCE = "__source__"
SINK = "__sink__"

INFINITE_CAPACITY = float("inf")
BLOCKABLE_CAPACITY = 1.0


@dataclass(frozen=True)
class AbstractEdge:
    """One (state, advisory) transition of the abstract model."""

    source: AcasAugmentedState
    advisory: str
    target: AcasAugmentedState

    @property
    def is_self_loop(self) -> bool:
        return self.source == self.target


@dataclass
class AcasContractMinCut:
    """
    Build the abstract graph, label real vs blockable edges, and cut it.

    Owns graph construction and max-flow only -- the network's real outputs
    come from AcasNetworkOracle, contract emission stays on the safety
    generator.
    """

    domain: AcasDomain
    reachable: AcasReachableSet
    kernel: AcasViabilityKernel
    real_advisory: dict[AcasAugmentedState, str]
    edges: list[AbstractEdge] = field(default_factory=list, repr=False)
    unsafe: frozenset[AcasAugmentedState] = frozenset()

    @classmethod
    def build(
        cls,
        real_advisory: dict[AcasAugmentedState, str],
        domain: AcasDomain | None = None,
    ) -> AcasContractMinCut:
        if domain is None:
            domain = AcasDomain.from_yaml()
        reachable = AcasReachableSet.compute(domain)
        kernel = AcasViabilityKernel.compute(domain)

        edges: list[AbstractEdge] = []
        for state in reachable.states:
            for advisory in domain.advisories:
                physical = domain.simulate_step(
                    state.x_mag, state.y_mag, state.x_sign, state.y_sign,
                    state.heading_own_var, advisory,
                )
                edges.append(AbstractEdge(
                    source=state,
                    advisory=advisory,
                    target=AcasAugmentedState.from_physical(physical, advisory),
                ))

        unsafe = frozenset(
            state for state in reachable.states
            if not kernel.is_plant_safe(state)
        )
        return cls(
            domain=domain,
            reachable=reachable,
            kernel=kernel,
            real_advisory=real_advisory,
            edges=edges,
            unsafe=unsafe,
        )

    # --- edge classification ------------------------------------------------

    def is_real(self, edge: AbstractEdge) -> bool:
        """True iff the network actually selects this advisory at this state."""
        return self.real_advisory.get(edge.source) == edge.advisory

    def blockable_edges(self) -> list[AbstractEdge]:
        """B: edges the network provably does not take, so CROWN can certify."""
        return [edge for edge in self.edges if not self.is_real(edge)]

    # --- max-flow -----------------------------------------------------------

    def _capacity_graph(self) -> dict[Any, dict[Any, float]]:
        """
        Residual-capacity adjacency for Edmonds-Karp.

        A super-source feeds the seed; every unsafe state feeds a super-sink,
        both with infinite capacity, so the cut can only fall on model edges.
        """
        capacity: dict[Any, dict[Any, float]] = {}

        def add(u: Any, v: Any, cap: float) -> None:
            capacity.setdefault(u, {})
            capacity.setdefault(v, {})
            capacity[u][v] = capacity[u].get(v, 0.0) + cap
            capacity[v].setdefault(u, 0.0)

        add(SOURCE, self.reachable.seed, INFINITE_CAPACITY)
        for state in self.unsafe:
            add(state, SINK, INFINITE_CAPACITY)

        for edge in self.edges:
            if edge.is_self_loop:
                continue
            add(
                edge.source, edge.target,
                INFINITE_CAPACITY if self.is_real(edge) else BLOCKABLE_CAPACITY,
            )
        return capacity

    @staticmethod
    def _augmenting_path(
        capacity: dict[Any, dict[Any, float]],
    ) -> list[Any] | None:
        """BFS for a shortest residual path (Edmonds-Karp)."""
        parent: dict[Any, Any] = {SOURCE: None}
        queue = deque([SOURCE])
        while queue:
            node = queue.popleft()
            for neighbour, residual in capacity.get(node, {}).items():
                if residual > 0 and neighbour not in parent:
                    parent[neighbour] = node
                    if neighbour == SINK:
                        path = [SINK]
                        while parent[path[-1]] is not None:
                            path.append(parent[path[-1]])
                        return list(reversed(path))
                    queue.append(neighbour)
        return None

    def min_cut(self) -> tuple[list[AbstractEdge], float]:
        """
        Minimum blockable edge set separating the seed from the unsafe states.

        Returns (cut_edges, flow_value). An empty cut with zero flow means the
        abstract model already satisfies the invariant with no contracts.
        """
        if not self.unsafe:
            return [], 0.0

        capacity = self._capacity_graph()
        flow = 0.0
        while True:
            path = self._augmenting_path(capacity)
            if path is None:
                break
            bottleneck = min(
                capacity[u][v] for u, v in zip(path, path[1:])
            )
            if bottleneck == INFINITE_CAPACITY:
                # A path of real edges reaches U: the real closed loop is
                # genuinely unsafe, so no contract set can help.
                raise ValueError(
                    "unsafe state reachable through real network edges only; "
                    "no contract set can establish the invariant"
                )
            for u, v in zip(path, path[1:]):
                capacity[u][v] -= bottleneck
                capacity[v][u] += bottleneck
            flow += bottleneck

        # Source side of the residual graph = reachable from SOURCE.
        source_side: set[Any] = set()
        queue = deque([SOURCE])
        source_side.add(SOURCE)
        while queue:
            node = queue.popleft()
            for neighbour, residual in capacity.get(node, {}).items():
                if residual > 0 and neighbour not in source_side:
                    source_side.add(neighbour)
                    queue.append(neighbour)

        cut = [
            edge for edge in self.edges
            if not edge.is_self_loop
            and not self.is_real(edge)
            and edge.source in source_side
            and edge.target not in source_side
        ]
        return cut, flow

    # --- reporting ----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        cut, flow = self.min_cut()
        blockable = self.blockable_edges()
        return {
            "reachable_states": len(self.reachable.states),
            "unsafe_states": len(self.unsafe),
            "abstract_edges": len(self.edges),
            "real_edges": len(self.edges) - len(blockable),
            "blockable_edges": len(blockable),
            "min_cut_size": len(cut),
            "max_flow": flow,
            "cut": [
                {
                    "a_prev": edge.source.a_prev,
                    "x_mag": edge.source.x_mag,
                    "y_mag": edge.source.y_mag,
                    "x_sign": edge.source.x_sign,
                    "y_sign": edge.source.y_sign,
                    "heading_own_var": edge.source.heading_own_var,
                    "forbidden_advisory": edge.advisory,
                    "real_advisory": self.real_advisory.get(edge.source),
                }
                for edge in sorted(
                    cut, key=lambda e: (e.source.as_tuple(), e.advisory),
                )
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=None, help="write the cut summary to this JSON path",
    )
    args = parser.parse_args()

    from core.safety.acas_network_oracle import AcasNetworkOracle  # noqa: PLC0415

    domain = AcasDomain.from_yaml()
    oracle = AcasNetworkOracle.build(domain)
    reachable = AcasReachableSet.compute(domain)
    real_advisory = oracle.advisories_for(reachable.states)

    analysis = AcasContractMinCut.build(real_advisory, domain)
    report = analysis.summary()

    print(f"reachable states   = {report['reachable_states']}")
    print(f"unsafe states in R = {report['unsafe_states']}")
    print(f"abstract edges     = {report['abstract_edges']}")
    print(f"  real             = {report['real_edges']}")
    print(f"  blockable (B)    = {report['blockable_edges']}")
    print(f"MIN CUT            = {report['min_cut_size']} contract(s)")
    for entry in report["cut"]:
        print(
            f"  a_prev={entry['a_prev']:<13} "
            f"({entry['x_mag']},{entry['y_mag']}) "
            f"signs=({entry['x_sign']:+d},{entry['y_sign']:+d}) "
            f"h={entry['heading_own_var']:<3} "
            f"forbid {entry['forbidden_advisory']:<13} "
            f"(network picks {entry['real_advisory']})"
        )

    if args.output:
        path = Path(args.output)
        if not path.is_absolute():
            path = EXAMPLE_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
