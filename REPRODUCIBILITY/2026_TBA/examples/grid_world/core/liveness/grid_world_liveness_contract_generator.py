"""
grid_world_liveness_contract_generator.py

Liveness-side contract generation: one set-membership contract per (source,
goal) progress pair, requiring the NN to pick an action that strictly
decreases dist(., goal).

Produces list[GridWorldLivenessContract] from GridWorldGoalDistance. No CROWN
(see GridWorldLivenessContractVerifier).

Usage (from examples/grid_world/):

    python3 -m core.liveness.grid_world_liveness_contract_generator
    python3 -m core.liveness.grid_world_liveness_contract_generator --check-onnx \\
        networks/1000__6_18_0__0200_1.onnx
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.grid_world_contract import GridWorldLivenessContract
from core.grid_world_domain import ACTIONS, DIR_IDX, GridWorldDomain
from core.liveness.grid_world_goal_distance import GridWorldGoalDistance
from core.liveness.grid_world_liveness_params import GridWorldLivenessParams

_PARAMS = GridWorldLivenessParams.from_yaml()

# XX has no cardinal index; it is the last class in the DSL / NN output order.
STAY_CLASS_INDEX = _PARAMS.stay_class_index

_ACTION_CLASS_INDEX: dict[str, int] = dict(DIR_IDX)
_ACTION_CLASS_INDEX["XX"] = STAY_CLASS_INDEX

DEFAULT_OUTPUT_DIR = _PARAMS.contracts_dir


@dataclass(frozen=True)
class GridWorldLivenessContractGenerator:
    """
    Build liveness (set-membership) contracts from a distance-to-goal table.

    The distance table owns the ranking function; this owns the emission
    policy -- which pairs get a contract and how Dec becomes a CROWN
    obligation.
    """

    goal_distance: GridWorldGoalDistance

    @classmethod
    def from_domain(
        cls, domain: GridWorldDomain | None = None,
    ) -> GridWorldLivenessContractGenerator:
        return cls(goal_distance=GridWorldGoalDistance.compute(domain))

    @property
    def domain(self) -> GridWorldDomain:
        return self.goal_distance.domain

    def generate_all_contracts(self) -> list[GridWorldLivenessContract]:
        """
        One contract per reachable (source, goal) pair with source != goal:

            Assume drone at source AND goal at goal_region
            Guarantee NN in Dec(source, goal)

        Goal cells and unreachable pairs are skipped: the CTL specification
        excuses a walled-off target, and an arrived drone needs no progress.
        """
        contracts: list[GridWorldLivenessContract] = []
        for source, goal in self.goal_distance.progress_pairs():
            allowed = tuple(self.goal_distance.decreasing_actions(source, goal))
            if not allowed:
                raise AssertionError(
                    f"empty Dec at source={source} goal={goal}; the ranking "
                    "function is not total and descent cannot be argued"
                )
            forbidden = tuple(a for a in ACTIONS if a not in allowed)
            contracts.append(GridWorldLivenessContract(
                source=source,
                goal_region=goal,
                allowed_dirs=allowed,
                allowed_dir_idxs=tuple(_ACTION_CLASS_INDEX[a] for a in allowed),
                forbidden_dirs=forbidden,
                forbidden_dir_idxs=tuple(
                    _ACTION_CLASS_INDEX[a] for a in forbidden
                ),
                distance=self.goal_distance.distance(source, goal),
            ))
        return contracts

    def crown_obligation_count(self) -> int:
        """Never-select calls CROWN must discharge if no contracts are merged."""
        return sum(
            len(c.forbidden_dirs) for c in self.generate_all_contracts()
        )

    def check_against_onnx(self, onnx_path: str) -> dict[str, Any]:
        """
        Forward-pass pre-answer: does this network already obey Dec everywhere?

        Not a proof -- one argmax per pair, no bound propagation. It predicts
        the CROWN verdict cheaply, the way the ACAS margins do.
        """
        import numpy as np  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415

        contracts = self.generate_all_contracts()
        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        inputs = np.array(
            [[*c.source, *c.goal_region] for c in contracts], dtype=np.float32,
        )
        predictions = session.run(None, {input_name: inputs})[0]
        predicted = predictions.reshape(len(contracts), -1).argmax(1)

        violations = [
            {
                "id": index + 1,
                "source": list(contract.source),
                "goal": list(contract.goal_region),
                "predicted_idx": int(predicted[index]),
                "allowed_dir_idxs": list(contract.allowed_dir_idxs),
            }
            for index, contract in enumerate(contracts)
            if int(predicted[index]) not in contract.allowed_dir_idxs
        ]
        stalls = sum(
            1 for v in violations if v["predicted_idx"] == STAY_CLASS_INDEX
        )
        return {
            "onnx_path": onnx_path,
            "contracts": len(contracts),
            "obeys_dec": len(contracts) - len(violations),
            "violations": len(violations),
            "stalls": stalls,
            "violation_details": violations[:20],
        }

    def write_specs(
        self,
        output_path: Path,
        contracts: list[GridWorldLivenessContract] | None = None,
    ) -> Path:
        """Write the contract specs as JSON (no status field -- CROWN adds it)."""
        if contracts is None:
            contracts = self.generate_all_contracts()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "description": (
                "Grid-world liveness (progress) contracts. One per reachable "
                "(source, goal) pair: the NN must select an action that "
                "strictly decreases BFS distance to the goal. Membership in "
                "Dec is discharged as never-select obligations over "
                "forbidden_dirs."
            ),
            "kind": "liveness",
            "guarantee_type": "in_set",
            "contracts": [
                c.to_spec_dict(contract_id=i + 1)
                for i, c in enumerate(contracts)
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2))
        return output_path


def generate_contracts(
    config_path: str | None = None,
) -> list[GridWorldLivenessContract]:
    """Public entry point for the compositional pipeline."""
    domain = (
        GridWorldDomain.from_config(config_path=config_path)
        if config_path
        else GridWorldDomain.from_config()
    )
    return GridWorldLivenessContractGenerator.from_domain(
        domain,
    ).generate_all_contracts()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=None,
        help="write contract specs to this JSON path",
    )
    parser.add_argument(
        "--check-onnx", default=None, metavar="ONNX",
        help="forward-pass pre-check of one network against Dec",
    )
    args = parser.parse_args()

    generator = GridWorldLivenessContractGenerator.from_domain()
    contracts = generator.generate_all_contracts()
    widths = [len(c.allowed_dirs) for c in contracts]

    print(f"liveness contracts (progress pairs) = {len(contracts)}")
    print(f"|Dec| min / max                     = {min(widths)} / {max(widths)}")
    print(
        "never-select obligations            = "
        f"{sum(len(c.forbidden_dirs) for c in contracts)}"
    )

    if args.output:
        path = generator.write_specs(Path(args.output), contracts)
        print(f"wrote {path}")

    if args.check_onnx:
        report = generator.check_against_onnx(args.check_onnx)
        print()
        print(f"forward-pass pre-check: {args.check_onnx}")
        print(
            f"  obeys Dec  = {report['obeys_dec']} / {report['contracts']} "
            f"({report['obeys_dec'] / report['contracts']:.1%})"
        )
        print(f"  violations = {report['violations']} "
              f"(of which stalls on XX: {report['stalls']})")


if __name__ == "__main__":
    main()
