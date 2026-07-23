"""
acas_liveness_contract_generator.py

Liveness-side contract generation: one equals contract per true closed-loop
state on the trajectory (stem + cycle).

Produces list[AcasLivenessContract]. Paths and nn_input_eps come from
AcasLivenessContractConfig / acas_liveness_params.yaml. No CROWN
(see AcasLivenessContractVerifier).

Usage (from AcasXu_closed_loop/):

    python3 acas_liveness_contract_generator.py
    python3 acas_liveness_contract_generator.py --check-onnx
    python3 acas_liveness_contract_generator.py --config acas_liveness_params.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.acas_contract import AcasLivenessContract
from core.liveness.acas_liveness_contract_config import AcasLivenessContractConfig
from core.liveness.acas_lasso_trajectory import AcasLassoTrajectory


@dataclass
class AcasLivenessContractGenerator:
    """
    Build liveness (equals) contracts from a closed-loop trajectory.

    At each trajectory index i, required_advisory is the advisory that becomes
    a_prev at step i+1 (cycle wraps via trajectory.required_advisory_after).
    """

    config: AcasLivenessContractConfig
    trajectory: AcasLassoTrajectory

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
    ) -> AcasLivenessContractGenerator:
        config = AcasLivenessContractConfig.from_yaml(path)
        return cls(config=config, trajectory=config.load_trajectory())

    @classmethod
    def from_config(
        cls,
        config: AcasLivenessContractConfig,
        trajectory: AcasLassoTrajectory | None = None,
    ) -> AcasLivenessContractGenerator:
        """Orchestrator path: share one config (and optional preloaded trajectory)."""
        loaded = trajectory if trajectory is not None else config.load_trajectory()
        return cls(config=config, trajectory=loaded)

    @property
    def domain(self):
        return self.config.domain

    @property
    def nn_input_eps(self) -> float:
        return self.config.nn_input_eps

    def generate_all_contracts(self) -> list[AcasLivenessContract]:
        """One AcasLivenessContract per trajectory state."""
        domain = self.domain
        eps = self.nn_input_eps
        contracts: list[AcasLivenessContract] = []
        for index, state in enumerate(self.trajectory.states):
            required = self.trajectory.required_advisory_after(index)
            network_idx, onnx_rel = domain.a_prev_to_nn[state.a_prev]
            center = domain.compute_nn_inputs(
                state.x_mag,
                state.y_mag,
                state.x_sign,
                state.y_sign,
                state.heading_own_var,
            )
            lower = [value - eps for value in center]
            upper = [value + eps for value in center]
            contract_id = index + 1
            contracts.append(AcasLivenessContract(
                contract_id=contract_id,
                a_prev=state.a_prev,
                network_idx=network_idx,
                onnx=Path(onnx_rel).name,
                heading_own_var=state.heading_own_var,
                x_sign=state.x_sign,
                y_sign=state.y_sign,
                nn_input_lower=lower,
                nn_input_upper=upper,
                description=(
                    f"Liveness {contract_id}: NN[{state.a_prev}] at "
                    f"({state.x_mag},{state.y_mag},"
                    f"sx={state.x_sign},sy={state.y_sign},"
                    f"h={state.heading_own_var}) "
                    f"must select {required}"
                ),
                required_advisory=required,
                required_advisory_idx=domain.adv_idx[required],
                x_mag=state.x_mag,
                y_mag=state.y_mag,
                role="liveness",
                contract_type="equals",
            ))
        return contracts

    def fill_onnx_margins(
        self,
        contracts: list[AcasLivenessContract],
    ) -> dict[int, float]:
        """
        ONNX Runtime argmax check: required class margin vs best other score.

        Returns contract_id -> margin. Raises if argmax != required.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("onnxruntime required: pip install onnxruntime") from exc

        sessions: dict[str, Any] = {}
        margins: dict[int, float] = {}
        root = self.config.root
        for contract in contracts:
            a_prev = contract.a_prev
            if a_prev not in sessions:
                _idx, onnx_rel = self.domain.a_prev_to_nn[a_prev]
                sessions[a_prev] = ort.InferenceSession(str(root / onnx_rel))

            session = sessions[a_prev]
            center = [
                (lo + hi) / 2.0
                for lo, hi in zip(contract.nn_input_lower, contract.nn_input_upper)
            ]
            inputs = np.array(center, dtype=np.float32).reshape(1, 1, 1, -1)
            input_name = session.get_inputs()[0].name
            scores = session.run(None, {input_name: inputs})[0][0]
            required_idx = contract.required_advisory_idx
            required_score = float(scores[required_idx])
            other_best = max(
                float(scores[j]) for j in range(len(scores)) if j != required_idx
            )
            margin = required_score - other_best
            margins[contract.contract_id] = margin
            argmax_name = self.domain.advisories[int(np.argmax(scores))]
            if argmax_name != contract.required_advisory:
                raise RuntimeError(
                    f"contract {contract.contract_id}: ONNX argmax={argmax_name} "
                    f"!= required={contract.required_advisory}"
                )
        return margins

    def write_specs(
        self,
        contracts: list[AcasLivenessContract],
        path: Path | None = None,
    ) -> Path:
        """Write liveness specs JSON (no status / margins). Default: config.specs_path."""
        out = path if path is not None else self.config.specs_path
        AcasLivenessContract.dump_json(
            contracts,
            out,
            description=(
                "ACAS Xu liveness equals contracts. "
                f"nn_input_eps={self.nn_input_eps}, "
                f"|contracts|={len(contracts)}, "
                f"cycle_start={self.trajectory.cycle_start_index}, "
                f"D_star_cycle_max_rho={self.trajectory.max_distance_on_cycle()}"
            ),
            extra_meta={
                "nn_input_eps": self.nn_input_eps,
                "cycle_start_index": self.trajectory.cycle_start_index,
                "max_distance_on_cycle": self.trajectory.max_distance_on_cycle(),
            },
        )
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to acas_liveness_params.yaml",
    )
    parser.add_argument(
        "--specs-out",
        type=Path,
        default=None,
        help="Override specs_path from config for this run",
    )
    parser.add_argument(
        "--check-onnx",
        action="store_true",
        help="Run ONNX argmax margin check (requires onnxruntime)",
    )
    args = parser.parse_args()

    generator = AcasLivenessContractGenerator.from_yaml(args.config)
    contracts = generator.generate_all_contracts()
    expected = generator.config.expected_total_states
    print(f"[CHECK] |contracts| = {len(contracts)} (expect {expected})")
    print(f"[CHECK] cycle_start = {generator.trajectory.cycle_start_index}")
    print(
        f"[CHECK] D* max rho on cycle = "
        f"{generator.trajectory.max_distance_on_cycle()}"
    )
    print(f"[CHECK] nn_input_eps = {generator.nn_input_eps}")

    if args.check_onnx:
        margins = generator.fill_onnx_margins(contracts)
        values = list(margins.values())
        print(
            f"[CHECK] ONNX margins: min={min(values):+.4f}  "
            f"median={float(np.median(values)):+.4f}  max={max(values):+.4f}"
        )

    out = generator.write_specs(contracts, args.specs_out)
    print(f"Wrote specs: {out}")


if __name__ == "__main__":
    main()
