"""
classify_contracts_by_reachability.py

Classify every discrete-mode contract (SAT and UNSAT) by whether it covers a
reachable dangerous state. For UNSAT contracts, "unreachable_explained" means
the UNSAT verdict is not a safety-relevant finding (see acas_contract_verifier.py
--point for resolving exceptions). For SAT contracts, the same check gives the
projected INVAR line count Optimization 1's pruning would inject -- only SAT
contracts are ever turned into INVAR constraints (see run_smv_patch() in
run_acas_compositional_pipeline.py), so this is the number that matters for
sizing that work before writing it.

Input:  contracts/crown/discrete/archive/aprev_*_crown_results.json (5 files)
Output: results/compositional/discrete/contract_reachability_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

from acas_domain import AcasDomain
from acas_reachability import AcasReachableSet

_HERE          = Path(__file__).parent
CONTRACTS_DIR  = _HERE / "contracts/crown/discrete/archive"
REPORT_PATH    = _HERE / "results/compositional/discrete/contract_reachability_report.json"

_DOMAIN = AcasDomain.from_yaml()
_IDX_TO_ADVISORY = {
    network_idx: name
    for name, (network_idx, _onnx) in _DOMAIN.a_prev_to_nn.items()
}


def main() -> None:
    reachable_set = AcasReachableSet.compute(_DOMAIN)

    per_network = {}
    reachable_dangerous_state_contracts = []
    projected_invar_ceiling = 0

    for result_file in sorted(CONTRACTS_DIR.glob("aprev_*_crown_results.json")):
        data = json.loads(result_file.read_text(encoding="utf-8"))
        advisory = _IDX_TO_ADVISORY[data["network_idx"]]

        per_network[advisory] = {}
        for status in ("SAT", "UNSAT"):
            contracts = [c for c in data["contracts"] if c["status"] == status]
            reachable_states_by_id = {
                contract["id"]: reachable_set.dangerous_xy(contract, advisory)
                for contract in contracts
            }

            reachable_dangerous_state_contracts += [
                {"network": advisory, "contract_id": cid, "status": status}
                for cid, states in reachable_states_by_id.items() if states
            ]
            if status == "SAT":
                projected_invar_ceiling += sum(len(states) for states in reachable_states_by_id.values())

            per_network[advisory][status] = {
                "total":                     len(contracts),
                "unreachable_explained":     sum(not states for states in reachable_states_by_id.values()),
                "reachable_dangerous_state": sum(bool(states) for states in reachable_states_by_id.values()),
            }

    report = {
        "per_network": per_network,
        "reachable_dangerous_state_contracts": reachable_dangerous_state_contracts,
        "projected_invar_ceiling": projected_invar_ceiling,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(per_network, indent=2))
    print(f"\nProjected INVAR ceiling after pruning (was 8,982): {projected_invar_ceiling}")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
