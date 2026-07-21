"""
acas_contract_generator.py

Range-based A/G safety contracts from plant physics (AcasDomain).

Owns only:
  - AcasContractGenerator.enumerate_dangerous_pairs
  - AcasContractGenerator.group_range_contracts
  - CLI main (write JSON)

No CROWN; no viability/reachability pruning.

Usage (from AcasXu_closed_loop/):

    python3 acas_contract_generator.py
    python3 acas_contract_generator.py --eps 1e-4 --output path/to/specs.json
    python3 acas_contract_generator.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acas_domain import AcasDomain


@dataclass
class AcasContractGenerator:
    """
    Build range-contract JSON specs from AcasDomain physics.

    One contract per (heading, quadrant, forbidden_advisory, a_prev/network).
    """

    domain: AcasDomain

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> AcasContractGenerator:
        return cls(domain=AcasDomain.from_yaml(path))

    def enumerate_dangerous_pairs(self) -> list[dict[str, Any]]:
        """
        States with distance >= safety_threshold where some advisory makes
        next distance < safety_threshold.
        """
        domain = self.domain
        advisory_index = domain.adv_idx
        pairs: list[dict[str, Any]] = []

        for x_mag, y_mag, x_sign, y_sign, heading_own_var in domain.all_physical_states():
            if not domain.is_safe(x_mag, y_mag):
                continue
            for advisory in domain.advisories:
                next_x_mag, next_y_mag, _, _, _ = domain.simulate_step(
                    x_mag, y_mag, x_sign, y_sign, heading_own_var, advisory,
                )
                if domain.compute_distance(next_x_mag, next_y_mag) < domain.safety_threshold:
                    pairs.append({
                        "state": {
                            "x_mag": x_mag,
                            "y_mag": y_mag,
                            "x_sign": x_sign,
                            "y_sign": y_sign,
                            "heading_own_var": heading_own_var,
                        },
                        "forbidden_advisory": advisory,
                        "forbidden_advisory_idx": advisory_index[advisory],
                        "nn_inputs": domain.compute_nn_inputs(
                            x_mag, y_mag, x_sign, y_sign, heading_own_var,
                        ),
                    })
        return pairs

    def group_range_contracts(
        self,
        pairs: list[dict[str, Any]],
        eps: float = 1e-4,
    ) -> list[dict[str, Any]]:
        """
        Group pairs by (heading, x_sign, y_sign, forbidden_advisory).

        One bounding box over NN inputs per group, then one contract per network.
        """
        domain = self.domain
        advisory_index = domain.adv_idx
        groups: dict[tuple, dict[str, Any]] = {}

        for pair in pairs:
            state = pair["state"]
            group_key = (
                state["heading_own_var"],
                state["x_sign"],
                state["y_sign"],
                pair["forbidden_advisory"],
            )
            if group_key not in groups:
                groups[group_key] = {"inputs": [], "states": []}
            groups[group_key]["inputs"].append(pair["nn_inputs"])
            groups[group_key]["states"].append([state["x_mag"], state["y_mag"]])

        contracts: list[dict[str, Any]] = []
        contract_id = 1

        for group_key in sorted(groups):
            heading_own_var, x_sign, y_sign, forbidden_advisory = group_key
            input_list = groups[group_key]["inputs"]
            dangerous_states = groups[group_key]["states"]
            n = len(input_list[0])

            lower = [min(inputs[i] for inputs in input_list) - eps for i in range(n)]
            upper = [max(inputs[i] for inputs in input_list) + eps for i in range(n)]

            def sign_label(value: int) -> str:
                return "+" if value == 1 else "-"

            for a_prev, (network_idx, onnx_path) in domain.a_prev_to_nn.items():
                contracts.append({
                    "id": contract_id,
                    "type": "range",
                    "heading_own_var": heading_own_var,
                    "x_sign": x_sign,
                    "y_sign": y_sign,
                    "a_prev": a_prev,
                    "network_idx": network_idx,
                    "onnx": onnx_path,
                    "nn_input_lower": lower,
                    "nn_input_upper": upper,
                    "n_states_covered": len(dangerous_states),
                    "dangerous_xy": dangerous_states,
                    "forbidden_advisory": forbidden_advisory,
                    "forbidden_advisory_idx": advisory_index[forbidden_advisory],
                    "description": (
                        f"NN_{network_idx} (a_prev={a_prev}) "
                        f"h={heading_own_var} "
                        f"({sign_label(x_sign)},{sign_label(y_sign)}) "
                        f"covers {len(dangerous_states)} state(s), "
                        f"must not choose {forbidden_advisory}"
                    ),
                })
                contract_id += 1

        return contracts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ACAS Xu A/G contract specs (range-based) for CROWN.",
    )
    parser.add_argument(
        "--output",
        default="contracts/crown/safety_full_contracts.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-4,
        help="Bounding-box margin per NN input dimension (default: 1e-4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print counts only; do not write output file",
    )
    args = parser.parse_args()

    generator = AcasContractGenerator.from_yaml()
    domain = generator.domain

    print("Enumerating dangerous (state, advisory) pairs...")
    pairs = generator.enumerate_dangerous_pairs()
    num_group_keys = len({
        (
            pair["state"]["heading_own_var"],
            pair["state"]["x_sign"],
            pair["state"]["y_sign"],
            pair["forbidden_advisory"],
        )
        for pair in pairs
    })
    print(
        f"  {len(pairs)} dangerous pairs across "
        f"{num_group_keys} (heading, sign, advisory) groups"
    )

    contracts = generator.group_range_contracts(pairs, eps=args.eps)
    num_networks = len(domain.a_prev_to_nn)
    num_groups = len(contracts) // num_networks if num_networks else 0
    print(
        f"  {num_groups} non-empty groups x {num_networks} NNs = "
        f"{len(contracts)} range contracts"
    )
    print(f"  (vs {len(pairs) * num_networks} per-state contracts)")

    if args.dry_run:
        print("Dry run -- no file written.")
        return

    report = {
        "description": "ACAS Xu closed-loop A/G contract specs (range-based)",
        "physics": {
            "degree_multiplier": domain.degree_multiplier,
            "seconds_per_update": domain.seconds_per_update,
            "speed_own": domain.speed_own,
            "speed_int": domain.speed_int,
            "heading_int_degrees": domain.heading_int_degrees,
            "safety_threshold": domain.safety_threshold,
            "heading_update_order": (
                "heading updated first (sequential env_update), "
                "then position computed with new heading"
            ),
        },
        "contract_type": (
            "range-based: bounding box over (x_mag,y_mag) "
            "for fixed (heading,sign,advisory)"
        ),
        "nn_mapping": {
            a_prev: {"network_idx": network_idx, "onnx": onnx_path}
            for a_prev, (network_idx, onnx_path) in domain.a_prev_to_nn.items()
        },
        "total_dangerous_pairs": len(pairs),
        "total_groups": num_groups,
        "total_contracts": len(contracts),
        "contracts": contracts,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved {len(contracts)} contracts to {output_path}")


if __name__ == "__main__":
    main()
