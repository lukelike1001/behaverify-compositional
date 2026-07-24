"""
pipeline.crown_verifier — shared alpha-beta-CROWN adapter.

Owns configuration + one input-box solve. Example scripts build lower/upper
bounds and choose which classification property to certify:

    certify_network_never_selects_class(...)   # safety A/G: NN ≠ forbidden class
    certify_network_always_selects_class(...)  # determinism pins: NN = required class

This module has no example-specific physics or contract JSON schema.

Note on multiprocessing:
    This module imports abcrown at the module level. When used in a spawned
    worker process (verify_acas_contracts_parallel.py), import this module
    lazily *inside* the worker function so that the main process never loads
    abcrown (avoiding multiprocessing / global-state conflicts).

Name note:
    CrownVerifier is our adapter type. It does not clash with abcrown's public
    API (ABCrownSolver, ConfigBuilder, VerificationSpec, ...).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from abcrown import ABCrownSolver, ConfigBuilder, VerificationSpec, input_vars, output_vars


def _normalize_crown_status(raw_status: str) -> str:
    """Map raw α,β-CROWN result.status to SAT / UNSAT / TIMEOUT."""
    if raw_status in ("safe", "verified", "safe-incomplete"):
        return "SAT"
    if raw_status.startswith("unsafe"):
        return "UNSAT"
    return "TIMEOUT"


def _build_crown_config(
    timeout_seconds: float,
    pgd_order: str = "before",
    device: str = "cpu",
    pgd_restarts: int = 50,
    extra_settings: dict[str, Any] | None = None,
) -> Any:
    """Build an alpha-beta-CROWN solver configuration (ConfigBuilder result)."""
    builder = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        .set(attack__pgd_order=pgd_order)
        .set(bab__timeout=timeout_seconds)
    )
    if pgd_order == "before":
        builder = builder.set(attack__pgd_restarts=pgd_restarts)
    if extra_settings:
        for key, value in extra_settings.items():
            builder = builder.set(**{key: value})
    return builder()


@dataclass
class CrownVerifier:
    """
    Thin adapter around ABCrownSolver for classification output properties.

    Holds one CROWN config (timeout, PGD, device) and discharges properties over
    an axis-aligned input box [input_lower_bounds, input_upper_bounds].
    """

    crown_config: Any

    @classmethod
    def from_timeout_and_attack_settings(
        cls,
        timeout_seconds: float,
        pgd_order: str = "before",
        device: str = "cpu",
        pgd_restarts: int = 50,
        extra_settings: dict[str, Any] | None = None,
    ) -> CrownVerifier:
        """Construct a verifier with a freshly built CROWN config."""
        return cls(
            crown_config=_build_crown_config(
                timeout_seconds=timeout_seconds,
                pgd_order=pgd_order,
                device=device,
                pgd_restarts=pgd_restarts,
                extra_settings=extra_settings,
            )
        )

    def certify_network_never_selects_class(
        self,
        onnx_path: str,
        input_lower_bounds: list[float],
        input_upper_bounds: list[float],
        forbidden_class_index: int,
        number_of_classes: int,
    ) -> tuple[str, Any]:
        """
        Certify that the network never selects `forbidden_class_index` on the box.

        Score encoding (unique maximum): there exists another class j with
        y[j] > y[forbidden_class_index]. SAT means the property holds on the
        whole box (safety A/G style).
        """
        other_class_indices = [
            class_index
            for class_index in range(number_of_classes)
            if class_index != forbidden_class_index
        ]
        output_scores = output_vars(number_of_classes)
        output_constraint = functools.reduce(
            lambda left, right: left | right,
            [
                output_scores[other_index] > output_scores[forbidden_class_index]
                for other_index in other_class_indices
            ],
        )
        return self._solve_over_input_box(
            onnx_path=onnx_path,
            input_lower_bounds=input_lower_bounds,
            input_upper_bounds=input_upper_bounds,
            output_score_variables=output_scores,
            output_constraint=output_constraint,
        )

    def certify_network_always_selects_class(
        self,
        onnx_path: str,
        input_lower_bounds: list[float],
        input_upper_bounds: list[float],
        required_class_index: int,
        number_of_classes: int,
    ) -> tuple[str, Any]:
        """
        Certify that the network always selects `required_class_index` on the box.

        Score encoding (unique maximum): for every other class j,
        y[required_class_index] > y[j]. Used for determinism / lasso pin contracts.
        """
        other_class_indices = [
            class_index
            for class_index in range(number_of_classes)
            if class_index != required_class_index
        ]
        output_scores = output_vars(number_of_classes)
        output_constraint = functools.reduce(
            lambda left, right: left & right,
            [
                output_scores[required_class_index] > output_scores[other_index]
                for other_index in other_class_indices
            ],
        )
        return self._solve_over_input_box(
            onnx_path=onnx_path,
            input_lower_bounds=input_lower_bounds,
            input_upper_bounds=input_upper_bounds,
            output_score_variables=output_scores,
            output_constraint=output_constraint,
        )

    def _solve_over_input_box(
        self,
        onnx_path: str,
        input_lower_bounds: list[float],
        input_upper_bounds: list[float],
        output_score_variables: Any,
        output_constraint: Any,
    ) -> tuple[str, Any]:
        """Shared path: input box + caller-built output constraint → CROWN solve."""
        if len(input_lower_bounds) != len(input_upper_bounds):
            raise ValueError(
                "input_lower_bounds and input_upper_bounds must have the same length"
            )

        number_of_inputs = len(input_lower_bounds)
        input_variables = input_vars((number_of_inputs,))
        lower_tensor = torch.tensor(input_lower_bounds, dtype=torch.float32)
        upper_tensor = torch.tensor(input_upper_bounds, dtype=torch.float32)
        input_constraint = (input_variables >= lower_tensor) & (
            input_variables <= upper_tensor
        )

        verification_spec = VerificationSpec.build_spec(
            input_vars=input_variables,
            output_vars=output_score_variables,
            input_constraint=input_constraint,
            output_constraint=output_constraint,
        )
        solver_result = ABCrownSolver(
            verification_spec, onnx_path, config=self.crown_config
        ).solve()
        return _normalize_crown_status(solver_result.status), solver_result
