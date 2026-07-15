"""
verify_single_state.py

Ad hoc CROWN point-query for one exact state, one network, one forbidden
advisory. Reuses CrownVerifier -- built for resolving the small number of
contracts that classify_contracts_by_reachability.py flags as
"reachable_dangerous_state" (the UNSAT is not explained by unreachability
alone, so the exact reachable state needs direct verification).

Usage:
  python verify_single_state.py --advisory strong_right \\
      --x_mag 2 --y_mag 3 --x_sign 1 --y_sign 1 --heading 6 \\
      --forbidden strong_right
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
_TBA = (_HERE / "../../").resolve()
if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))

from pipeline.neuro.crown.crown_verifier import CrownVerifier
from generate_acas_contracts import A_PREV_TO_NN, ADV_IDX, compute_nn_inputs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--advisory", required=True, choices=A_PREV_TO_NN,
                   help="a_prev -- selects the network")
    p.add_argument("--forbidden", required=True, choices=ADV_IDX,
                   help="forbidden output advisory")
    p.add_argument("--x_mag", type=int, required=True)
    p.add_argument("--y_mag", type=int, required=True)
    p.add_argument("--x_sign", type=int, required=True, choices=[-1, 1])
    p.add_argument("--y_sign", type=int, required=True, choices=[-1, 1])
    p.add_argument("--heading", type=int, required=True, help="heading_own_var")
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()

    _, onnx_rel = A_PREV_TO_NN[args.advisory]
    onnx_path = str(_HERE / onnx_rel)
    exact = compute_nn_inputs(
        args.x_mag, args.y_mag, args.x_sign, args.y_sign, args.heading,
    )

    crown_verifier = CrownVerifier.from_timeout_and_attack_settings(
        timeout_seconds=args.timeout,
        pgd_order="before",
        device="cuda",
    )
    status, result = crown_verifier.certify_network_never_selects_class(
        onnx_path=onnx_path,
        input_lower_bounds=exact,
        input_upper_bounds=exact,
        forbidden_class_index=ADV_IDX[args.forbidden],
        number_of_classes=len(ADV_IDX),
    )

    print(
        f"state=(x_mag={args.x_mag}, y_mag={args.y_mag}, x_sign={args.x_sign}, "
        f"y_sign={args.y_sign}, heading_own_var={args.heading}) "
        f"network={args.advisory} forbidden={args.forbidden}"
    )
    print(f"nn_inputs={exact}")
    print(f"status={status}")
    if status == "UNSAT":
        print(f"raw_status={result.status}")


if __name__ == "__main__":
    main()
