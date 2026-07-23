"""
acas_lasso_trajectory.py

Ground-truth closed-loop lasso: real ONNX inference (no CROWN, no abstraction)
from the seed until the walk closes into a cycle.

Backend only — no Gradio/Plotly. Front-end apps load the JSON dump via
``AcasLassoTrajectory.from_json_file`` (no onnxruntime required).

CLI (from examples/AcasXu_closed_loop/):

    python3 acas_lasso_trajectory.py
    python3 acas_lasso_trajectory.py --dump acas_lasso_trajectory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.acas_domain import AcasDomain
from core.acas_reachability import AcasReachableSet
from core.acas_state import AcasAugmentedState
from core.paths import EXAMPLE_ROOT

DEFAULT_LASSO_JSON = (
    EXAMPLE_ROOT / "core" / "liveness" / "acas_lasso_trajectory.json"
)
# Fixed artifact for current trained weights (07-12 inductive analysis).
STEM_LENGTH = 39
EXPECTED_TOTAL = 52  # stem 39 + cycle 13

# Shared plant for methods that still need module-level physics access.
DOMAIN = AcasDomain.from_yaml()


@dataclass
class AcasLassoTrajectory:
    """
    Ordered lasso: stem then one full cycle (no closing return in the dump).

    Construct via ``from_json_file`` (frontend / pins, no ONNX) or
    ``from_onnx`` (regenerate after retraining).
    """

    states: list[AcasAugmentedState]
    cycle_start_index: int

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_json_file(
        cls,
        path: Path,
        cycle_start_index: int,
        expected_total: int,
    ) -> AcasLassoTrajectory:
        """
        Load precomputed dump. Does not import onnxruntime.

        All arguments required — pass them from AcasLivenessContractConfig
        (or explicit CLI values), not module-level defaults.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        states = [AcasAugmentedState.from_json_pair(pair) for pair in raw]
        if len(states) != expected_total:
            raise ValueError(
                f"expected {expected_total} lasso states, got {len(states)}"
            )
        if not (0 <= cycle_start_index < len(states)):
            raise ValueError(f"bad cycle_start_index={cycle_start_index}")
        return cls(states=states, cycle_start_index=cycle_start_index)

    @classmethod
    def from_onnx(cls) -> AcasLassoTrajectory:
        """Walk the real closed loop with ONNX until a (state, a_prev) repeats."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("onnxruntime is required: pip install onnxruntime") from exc

        sessions = cls._load_sessions(ort)
        seed = AcasReachableSet.seed_state(DOMAIN)  # phys + a_prev
        physical = seed[:5]
        a_prev = seed[5]
        first = AcasAugmentedState(*physical, a_prev)

        trajectory = [first]
        seen = {first.pair_key(): 0}
        tick = 0
        while True:
            tick += 1
            advisory = cls._choose_advisory(sessions[a_prev], *physical)
            physical = DOMAIN.simulate_step(*physical, advisory)
            a_prev = advisory
            key = (*physical, a_prev)
            if key in seen:
                return cls(
                    states=trajectory,
                    cycle_start_index=seen[key],
                )
            seen[key] = tick
            trajectory.append(AcasAugmentedState(*physical, a_prev))

    # ------------------------------------------------------------------
    # ONNX helpers (lazy; only used by from_onnx)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_sessions(ort: Any) -> dict[str, Any]:
        return {
            name: ort.InferenceSession(str(EXAMPLE_ROOT / onnx_rel))
            for name, (_idx, onnx_rel) in DOMAIN.a_prev_to_nn.items()
        }

    @staticmethod
    def _run_onnx(session: Any, inputs: np.ndarray) -> np.ndarray:
        """ACAS Xu ONNX files use input shape [1, 1, 1, 5] (legacy conv wrapper)."""
        x = inputs.astype(np.float32).reshape(1, 1, 1, -1)
        input_name = session.get_inputs()[0].name
        return session.run(None, {input_name: x})[0][0]

    @classmethod
    def _choose_advisory(
        cls,
        session: Any,
        x_mag: int,
        y_mag: int,
        x_sign: int,
        y_sign: int,
        heading_own_var: int,
    ) -> str:
        inputs = np.array(
            DOMAIN.compute_nn_inputs(
                x_mag, y_mag, x_sign, y_sign, heading_own_var,
            ),
        )
        scores = cls._run_onnx(session, inputs)
        return DOMAIN.advisories[int(np.argmax(scores))]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.states)

    @property
    def stem_len(self) -> int:
        return self.cycle_start_index

    @property
    def cycle_len(self) -> int:
        return self.n - self.cycle_start_index

    @property
    def cycle_states(self) -> list[AcasAugmentedState]:
        return self.states[self.cycle_start_index:]

    @property
    def stem_states(self) -> list[AcasAugmentedState]:
        return self.states[:self.cycle_start_index]

    def state_at(self, tick: int) -> AcasAugmentedState:
        return self.states[max(0, min(int(tick), self.n - 1))]

    def on_cycle(self, tick: int) -> bool:
        return max(0, min(int(tick), self.n - 1)) >= self.cycle_start_index

    def min_distance(self) -> int:
        return min(
            DOMAIN.compute_distance(state.x_mag, state.y_mag)
            for state in self.states
        )

    def max_distance_on_cycle(self) -> int:
        return max(
            DOMAIN.compute_distance(state.x_mag, state.y_mag)
            for state in self.cycle_states
        )

    def required_advisory_after(self, index: int) -> str:
        """Advisory chosen at states[index] (becomes next a_prev)."""
        if index < self.n - 1:
            return self.states[index + 1].a_prev
        return self.states[self.cycle_start_index].a_prev

    def successor_physical(
        self, state: AcasAugmentedState, advisory: str,
    ) -> tuple[int, int, int, int, int]:
        return DOMAIN.simulate_step(*state.physical_tuple(), advisory)

    # ------------------------------------------------------------------
    # I/O and CLI checks
    # ------------------------------------------------------------------

    def dump(self, path: Path) -> None:
        """Serialize as JSON list of [[x_mag,y_mag,x_sign,y_sign,h], a_prev]."""
        data = [s.to_json_pair() for s in self.states]
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {len(data)} states to {path}")

    def print_checks(self) -> None:
        """Report stem/cycle lengths, min distance, seed recurrence, Q2 tick-0."""
        seed_recurs = self.cycle_start_index == 0
        min_dist = self.min_distance()

        print(f"[CHECK] stem length = {self.stem_len} ticks (report claims 39)")
        print(f"[CHECK] cycle length = {self.cycle_len} ticks (report claims 13)")
        print(f"[CHECK] total distinct augmented states = {self.n} (report claims 52)")
        print(f"[CHECK] minimum distance over trajectory = {min_dist} (report claims exactly 200)")
        print(f"[CHECK] seed state recurs after tick 0: {seed_recurs}")

        seed_a_prev = self.states[0].a_prev
        tick0_advisory = self.states[1].a_prev  # advisory at tick 0 → tick 1 a_prev
        holds = tick0_advisory != "strong_right"
        print(
            f"[CHECK] tick-0 advisory chosen by '{seed_a_prev}' network at the seed: "
            f"{tick0_advisory} (Q2 requires != strong_right: "
            f"{'HOLDS' if holds else 'VIOLATED'})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", type=Path, default=None,
        help="Write the trajectory as JSON to this path",
    )
    parser.add_argument(
        "--from-json", type=Path, default=None,
        help="Skip ONNX; load dump and print checks only",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Liveness YAML for dump shape when using --from-json "
             "(default: acas_liveness_params.yaml)",
    )
    args = parser.parse_args()

    if args.from_json is not None:
        from core.liveness.acas_liveness_contract_config import AcasLivenessContractConfig
        config = AcasLivenessContractConfig.from_yaml(args.config)
        traj = AcasLassoTrajectory.from_json_file(
            args.from_json,
            cycle_start_index=config.expected_stem_length,
            expected_total=config.expected_total_states,
        )
    else:
        traj = AcasLassoTrajectory.from_onnx()

    traj.print_checks()
    if args.dump:
        traj.dump(args.dump)


if __name__ == "__main__":
    main()

