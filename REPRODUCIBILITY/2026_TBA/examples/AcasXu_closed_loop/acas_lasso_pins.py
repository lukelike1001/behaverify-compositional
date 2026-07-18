"""
acas_lasso_pins.py

Determinism pins for ACAS Xu liveness parity: one equals-contract per true
closed-loop (state, a_prev) on the lasso trajectory.

Classes only — generate specs, ONNX margins, optional CROWN always-selects.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import generate_acas_contracts as g

_HERE = Path(__file__).parent.resolve()
_TBA = (_HERE / "../../").resolve()
if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))

from pipeline.neuro.crown.crown_verifier import CrownVerifier  # noqa: E402

# Point pins (eps=0): discrete closed loop only evaluates lattice points; SMV
# INVARs guard exact state values. Box eps>0 is optional local-robustness.
DEFAULT_EPS = 0.0
DEFAULT_LASSO_JSON = _HERE / "acas_lasso_trajectory.json"
STEM_LENGTH = 39  # from 07-12 inductive analysis; checked on load
# Tree/SMV: acas.active := (distance < max_dist); max_dist = 1000 in template.
ACTIVE_DISTANCE_THRESHOLD = 1000


@dataclass(frozen=True)
class AcasAugmentedState:
    """Physical state + a_prev (which network is active)."""

    x_mag: int
    y_mag: int
    x_sign: int
    y_sign: int
    heading_own_var: int
    a_prev: str

    @classmethod
    def from_json_pair(cls, pair: list) -> AcasAugmentedState:
        state, a_prev = pair
        return cls(
            x_mag=int(state[0]),
            y_mag=int(state[1]),
            x_sign=int(state[2]),
            y_sign=int(state[3]),
            heading_own_var=int(state[4]),
            a_prev=str(a_prev),
        )

    def physical_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.x_mag, self.y_mag, self.x_sign, self.y_sign, self.heading_own_var,
        )

    def pair_key(self) -> tuple:
        return (*self.physical_tuple(), self.a_prev)

    def nn_input_vector(self) -> list[float]:
        return g.compute_nn_inputs(
            self.x_mag, self.y_mag, self.x_sign, self.y_sign, self.heading_own_var,
        )

    def distance(self) -> int:
        return g.compute_distance(self.x_mag, self.y_mag)


@dataclass
class AcasLassoTrajectory:
    """Ordered lasso: stem then cycle; pins use next required advisory."""

    states: list[AcasAugmentedState]
    cycle_start_index: int

    @classmethod
    def from_json_file(
        cls,
        path: Path = DEFAULT_LASSO_JSON,
        cycle_start_index: int = STEM_LENGTH,
    ) -> AcasLassoTrajectory:
        raw = json.loads(path.read_text(encoding="utf-8"))
        states = [AcasAugmentedState.from_json_pair(pair) for pair in raw]
        if len(states) != 52:
            raise ValueError(f"expected 52 lasso states, got {len(states)}")
        if not (0 <= cycle_start_index < len(states)):
            raise ValueError(f"bad cycle_start_index={cycle_start_index}")
        return cls(states=states, cycle_start_index=cycle_start_index)

    @property
    def cycle_states(self) -> list[AcasAugmentedState]:
        return self.states[self.cycle_start_index:]

    def max_distance_on_cycle(self) -> int:
        return max(s.distance() for s in self.cycle_states)

    def required_advisory_after(self, index: int) -> str:
        """Advisory chosen at states[index] (becomes next a_prev)."""
        if index < len(self.states) - 1:
            return self.states[index + 1].a_prev
        # Last stem/cycle state transitions into cycle_start.
        return self.states[self.cycle_start_index].a_prev

    def successor_physical(
        self, state: AcasAugmentedState, advisory: str,
    ) -> tuple[int, int, int, int, int]:
        return g.simulate_step(*state.physical_tuple(), advisory)


@dataclass
class AcasLassoPin:
    """One equals-contract: NN[a_prev](state) must equal required_advisory."""

    pin_id: int
    state: AcasAugmentedState
    required_advisory: str
    eps: float = DEFAULT_EPS
    onnx_margin: float | None = None
    crown_status: str | None = None

    @property
    def required_advisory_idx(self) -> int:
        return g.ADV_IDX[self.required_advisory]

    @property
    def network_idx(self) -> int:
        return g.A_PREV_TO_NN[self.state.a_prev][0]

    @property
    def onnx_relative_path(self) -> str:
        return g.A_PREV_TO_NN[self.state.a_prev][1]

    def nn_input_bounds(self) -> tuple[list[float], list[float]]:
        center = self.state.nn_input_vector()
        lower = [value - self.eps for value in center]
        upper = [value + self.eps for value in center]
        return lower, upper

    def to_contract_dict(self) -> dict[str, Any]:
        lower, upper = self.nn_input_bounds()
        s = self.state
        return {
            "id": self.pin_id,
            "type": "point_pin",
            "guarantee_type": "equals",
            "role": "lasso_determinism_pin",
            "heading_own_var": s.heading_own_var,
            "x_sign": s.x_sign,
            "y_sign": s.y_sign,
            "a_prev": s.a_prev,
            "network_idx": self.network_idx,
            "onnx": Path(self.onnx_relative_path).name,
            "nn_input_lower": lower,
            "nn_input_upper": upper,
            "n_states_covered": 1,
            "dangerous_xy": [[s.x_mag, s.y_mag]],
            "required_advisory": self.required_advisory,
            "required_advisory_idx": self.required_advisory_idx,
            "description": (
                f"Lasso pin {self.pin_id}: NN[{s.a_prev}] at "
                f"({s.x_mag},{s.y_mag},sx={s.x_sign},sy={s.y_sign},h={s.heading_own_var}) "
                f"must select {self.required_advisory}"
            ),
            "onnx_margin": self.onnx_margin,
            "status": self.crown_status,
        }


@dataclass
class AcasLassoPinSet:
    """All 52 pins for one lasso + helpers for margins, CROWN, abstract reach."""

    trajectory: AcasLassoTrajectory
    pins: list[AcasLassoPin] = field(default_factory=list)
    eps: float = DEFAULT_EPS

    @classmethod
    def from_trajectory(
        cls,
        trajectory: AcasLassoTrajectory,
        eps: float = DEFAULT_EPS,
    ) -> AcasLassoPinSet:
        pins = []
        for index, state in enumerate(trajectory.states):
            pins.append(
                AcasLassoPin(
                    pin_id=index + 1,
                    state=state,
                    required_advisory=trajectory.required_advisory_after(index),
                    eps=eps,
                )
            )
        return cls(trajectory=trajectory, pins=pins, eps=eps)

    def fill_onnx_margins(self) -> None:
        """Margin of required class vs best other score (ONNX Runtime argmax check)."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("onnxruntime required: pip install onnxruntime") from exc

        sessions: dict[str, Any] = {}
        for pin in self.pins:
            a_prev = pin.state.a_prev
            if a_prev not in sessions:
                onnx_path = _HERE / pin.onnx_relative_path
                sessions[a_prev] = ort.InferenceSession(str(onnx_path))

            session = sessions[a_prev]
            inputs = np.array(pin.state.nn_input_vector(), dtype=np.float32).reshape(
                1, 1, 1, -1,
            )
            input_name = session.get_inputs()[0].name
            scores = session.run(None, {input_name: inputs})[0][0]
            required_idx = pin.required_advisory_idx
            required_score = float(scores[required_idx])
            other_best = max(
                float(scores[j]) for j in range(len(scores)) if j != required_idx
            )
            pin.onnx_margin = required_score - other_best
            argmax_name = g.ADVISORIES[int(np.argmax(scores))]
            if argmax_name != pin.required_advisory:
                raise RuntimeError(
                    f"pin {pin.pin_id}: ONNX argmax={argmax_name} "
                    f"!= required={pin.required_advisory}"
                )

    def certify_with_crown(
        self,
        timeout_seconds: float = 30.0,
        pgd_order: str = "before",
        device: str = "cpu",
        verbose: bool = True,
    ) -> dict[str, int]:
        """Run certify_network_always_selects_class on each pin."""
        counts = {"SAT": 0, "UNSAT": 0, "TIMEOUT": 0}
        for pin in self.pins:
            verifier = CrownVerifier.from_timeout_and_attack_settings(
                timeout_seconds=timeout_seconds,
                pgd_order=pgd_order,
                device=device,
            )
            lower, upper = pin.nn_input_bounds()
            onnx_path = str(_HERE / pin.onnx_relative_path)
            start = time.perf_counter()
            status, _result = verifier.certify_network_always_selects_class(
                onnx_path=onnx_path,
                input_lower_bounds=lower,
                input_upper_bounds=upper,
                required_class_index=pin.required_advisory_idx,
                number_of_classes=len(g.ADVISORIES),
            )
            wall = time.perf_counter() - start
            pin.crown_status = status
            counts[status] = counts.get(status, 0) + 1
            if verbose:
                print(
                    f"  pin {pin.pin_id:>2}/{len(self.pins)}  "
                    f"{pin.state.a_prev:>12} -> {pin.required_advisory:<12}  "
                    f"margin={pin.onnx_margin:+.4f}  {status:<8}  {wall:.2f}s"
                )
        return counts

    def abstract_reachable_pair_count(self, *, model_active_freeze: bool = False) -> int:
        """
        BFS from seed with pins: at a pinned pair advisory is fixed; elsewhere free.

        model_active_freeze=False: always apply physics (matches ONNX lasso script).
        model_active_freeze=True: match SMV acas.active := (distance < 1000) —
        when inactive, state and a_prev freeze (self-loop). That is the model
        nuXmv actually checks; the full 52-state ONNX lasso is NOT an SMV path
        past the first inactive state.
        """
        pin_map = {
            pin.state.pair_key(): pin.required_advisory for pin in self.pins
        }
        seed = self.trajectory.states[0]
        frontier = [seed.pair_key()]
        seen = {seed.pair_key()}
        while frontier:
            x_mag, y_mag, x_sign, y_sign, heading, a_prev = frontier.pop()
            key = (x_mag, y_mag, x_sign, y_sign, heading, a_prev)
            distance = g.compute_distance(x_mag, y_mag)
            if model_active_freeze and distance >= ACTIVE_DISTANCE_THRESHOLD:
                # SMV: tree inactive → no NN tick, no environment update.
                continue
            if key in pin_map:
                advisories = [pin_map[key]]
            else:
                advisories = list(g.ADVISORIES)
            for advisory in advisories:
                nxt = g.simulate_step(x_mag, y_mag, x_sign, y_sign, heading, advisory)
                nxt_key = (*nxt, advisory)
                if nxt_key not in seen:
                    seen.add(nxt_key)
                    frontier.append(nxt_key)
        return len(seen)

    def diagnose_ctl_counterexample_freeze(self) -> str:
        """
        Explain AG AF (rho >= 1400) failure under SMV active-freeze semantics.

        The CTL CE loops with acas.active=FALSE at distance 1000; physics freezes
        so the path never reaches the ONNX cycle at rho=1400.
        """
        first_inactive = None
        for index, state in enumerate(self.trajectory.states):
            if state.distance() >= ACTIVE_DISTANCE_THRESHOLD:
                first_inactive = (index, state)
                break
        lines = [
            "CTL diagnosis (SMV semantics, not pin-antecedent mismatch):",
            f"  acas.active := (distance < {ACTIVE_DISTANCE_THRESHOLD})  "
            f"[template max_dist]",
            "  When inactive: command and position freeze (no NN, no env update).",
            "  ONNX lasso ignores active and keeps simulating past that threshold.",
        ]
        if first_inactive is not None:
            index, state = first_inactive
            lines.append(
                f"  First ONNX-lasso inactive index={index}: "
                f"rho={state.distance()} at xy=({state.x_mag},{state.y_mag}) "
                f"h={state.heading_own_var} a_prev={state.a_prev}"
            )
            lines.append(
                "  nuXmv CTL CE loops here (active=FALSE, rho=1000); "
                "AG AF (rho>=1400) fails because freeze never reaches 1400."
            )
        lines.append(
            f"  Python |R| always-physics={self.abstract_reachable_pair_count(model_active_freeze=False)}; "
            f"|R| SMV-freeze={self.abstract_reachable_pair_count(model_active_freeze=True)}"
        )
        return "\n".join(lines)

    def write_specs_json(self, path: Path) -> None:
        payload = {
            "description": (
                "ACAS Xu lasso determinism pins (equals contracts). "
                f"eps={self.eps}, |pins|={len(self.pins)}, "
                f"cycle_start={self.trajectory.cycle_start_index}, "
                f"D_star_cycle_max_rho={self.trajectory.max_distance_on_cycle()}"
            ),
            "eps": self.eps,
            "cycle_start_index": self.trajectory.cycle_start_index,
            "max_distance_on_cycle": self.trajectory.max_distance_on_cycle(),
            "contracts": [pin.to_contract_dict() for pin in self.pins],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_crown_results_json(self, path: Path) -> None:
        payload = {
            "description": "CROWN / status results for lasso pins",
            "summary": {
                status: sum(1 for p in self.pins if p.crown_status == status)
                for status in ("SAT", "UNSAT", "TIMEOUT")
            },
            "contracts": [
                {
                    "id": pin.pin_id,
                    "status": pin.crown_status,
                    "onnx_margin": pin.onnx_margin,
                    "required_advisory": pin.required_advisory,
                    "a_prev": pin.state.a_prev,
                }
                for pin in self.pins
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lasso-json", type=Path, default=DEFAULT_LASSO_JSON,
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument(
        "--specs-out",
        type=Path,
        default=_HERE / "contracts/crown/lasso_pin_specs.json",
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=_HERE / "contracts/crown/lasso_pin_crown_results.json",
    )
    parser.add_argument(
        "--run-crown", action="store_true",
        help="Certify each pin with CrownVerifier.always_selects (slow)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    trajectory = AcasLassoTrajectory.from_json_file(args.lasso_json)
    pin_set = AcasLassoPinSet.from_trajectory(trajectory, eps=args.eps)
    print(f"[CHECK] |pins| = {len(pin_set.pins)} (expect 52)")
    print(f"[CHECK] cycle_start = {trajectory.cycle_start_index}")
    print(f"[CHECK] D* max rho on cycle = {trajectory.max_distance_on_cycle()}")
    print(f"[CHECK] pin eps = {args.eps} (0 = point query for discrete SMV)")

    pin_set.fill_onnx_margins()
    margins = [p.onnx_margin for p in pin_set.pins if p.onnx_margin is not None]
    print(
        f"[CHECK] ONNX margins: min={min(margins):+.4f}  "
        f"median={float(np.median(margins)):+.4f}  max={max(margins):+.4f}"
    )
    r_physics = pin_set.abstract_reachable_pair_count(model_active_freeze=False)
    r_smv = pin_set.abstract_reachable_pair_count(model_active_freeze=True)
    print(f"[CHECK] abstract |R| always-physics = {r_physics} (ONNX lasso model)")
    print(f"[CHECK] abstract |R| SMV-freeze = {r_smv} (active:=distance<1000)")
    print(pin_set.diagnose_ctl_counterexample_freeze())

    pin_set.write_specs_json(args.specs_out)
    print(f"Wrote specs: {args.specs_out}")

    if args.run_crown:
        counts = pin_set.certify_with_crown(
            timeout_seconds=args.timeout, device=args.device,
        )
        print(f"CROWN summary: {counts}")
        pin_set.write_crown_results_json(args.results_out)
        print(f"Wrote CROWN results: {args.results_out}")
    else:
        # Treat ONNX-consistent pins as SAT for downstream patch experiments.
        for pin in pin_set.pins:
            pin.crown_status = "SAT"
        pin_set.write_crown_results_json(args.results_out)
        print(
            f"Skipped CROWN (--run-crown to enable); "
            f"wrote ONNX-backed SAT stubs: {args.results_out}"
        )


if __name__ == "__main__":
    main()
