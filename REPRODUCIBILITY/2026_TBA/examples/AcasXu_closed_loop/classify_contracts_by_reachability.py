"""
classify_contracts_by_reachability.py

Classify every discrete-mode contract (SAT and UNSAT) by whether it covers a
reachable dangerous state. For UNSAT contracts, "unreachable_explained" means
the UNSAT verdict is not a safety-relevant finding (see verify_single_state.py
for resolving the exceptions). For SAT contracts, the same check gives the
projected INVAR line count Optimization 1's pruning would inject -- only SAT
contracts are ever turned into INVAR constraints (see run_smv_patch() in
run_acas_compositional_pipeline.py), so this is the number that matters for
sizing that work before writing it.

Input:  contracts/crown/discrete_goals/aprev_*_crown_results.json (5 files)
Output: results/compositional/discrete_goals/contract_reachability_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

from generate_acas_contracts import _P
from acas_reachability import compute_reachable_states, reachable_dangerous_xy

_HERE          = Path(__file__).parent
CONTRACTS_DIR  = _HERE / "contracts/crown/discrete_goals"
REPORT_PATH    = _HERE / "results/compositional/discrete_goals/contract_reachability_report.json"

_IDX_TO_ADVISORY = {net["idx"]: name for name, net in _P["networks"].items()}


def main() -> None:
    reachable = compute_reachable_states()

    per_network = {}
    reachable_dangerous_state_contracts = []
    projected_invar_ceiling = 0

    for result_file in sorted(CONTRACTS_DIR.glob("aprev_*_crown_results.json")):
        data = json.loads(result_file.read_text(encoding="utf-8"))
        advisory = _IDX_TO_ADVISORY[data["network_idx"]]

        per_network[advisory] = {}
        for status in ("SAT", "UNSAT"):
            contracts = [c for c in data["contracts"] if c["status"] == status]
            reachable_states_by_id = {c["id"]: reachable_dangerous_xy(c, advisory, reachable) for c in contracts}

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
