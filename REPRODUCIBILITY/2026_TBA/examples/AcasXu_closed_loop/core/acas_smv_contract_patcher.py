"""
acas_smv_contract_patcher.py

Symbolic half of ACAS Xu compositional verification: abstract the five NN
lookup tables out of a base SMV and inject SAT A/G contracts as INVARs.

Does not run CROWN or nuXmv. ACAS Xu has one contract kind -- safety
never-select -- so every INVAR emitted here is a not_equals constraint.

Typical use:

    patcher = AcasSmvContractPatcher.from_verifier_yaml()
    sat = patcher.load_sat_contracts(specs_path, results_path)
    text, metrics = patcher.patch(base_smv_text, sat)
    patcher.write_smv(output_path, text)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.paths import EXAMPLE_ROOT

_DEFAULT_VERIFIER_YAML = EXAMPLE_ROOT / "core" / "acas_verifier_params.yaml"

# Advisory domain written into the free NN-output VAR (must match the tree).
_DEFAULT_ADVISORIES = (
    "clear",
    "weak_left",
    "weak_right",
    "strong_left",
    "strong_right",
)

_SMV_VAR_KEYS = (
    "command_prev",
    "command_final",
    "x_mag",
    "y_mag",
    "x_sign",
    "y_sign",
    "heading",
)


@dataclass(frozen=True)
class AcasSmvContractPatcher:
    """
    Patch a BehaVerify-generated ACAS SMV with CROWN-discharged contracts.

    Holds SMV stage variable names and the advisory domain for the free VAR.
    """

    smv_variable_names: dict[str, str]
    advisories: tuple[str, ...] = _DEFAULT_ADVISORIES

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_verifier_yaml(
        cls,
        path: Path | str | None = None,
        *,
        advisories: tuple[str, ...] | None = None,
    ) -> AcasSmvContractPatcher:
        """Load smv_variables from acas_verifier_params.yaml (required keys)."""
        yaml_path = Path(path) if path is not None else _DEFAULT_VERIFIER_YAML
        with open(yaml_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"{yaml_path}: expected a mapping at top level")
        if "smv_variables" not in raw:
            raise ValueError(f"{yaml_path}: missing required key 'smv_variables'")
        smv_section = raw["smv_variables"]
        if not isinstance(smv_section, dict):
            raise ValueError(f"{yaml_path}: 'smv_variables' must be a mapping")

        missing = [key for key in _SMV_VAR_KEYS if key not in smv_section]
        if missing:
            raise ValueError(
                f"{yaml_path}: smv_variables missing key(s): {', '.join(missing)}"
            )

        names = {key: str(smv_section[key]) for key in _SMV_VAR_KEYS}
        return cls(
            smv_variable_names=names,
            advisories=advisories if advisories is not None else _DEFAULT_ADVISORIES,
        )

    # ------------------------------------------------------------------
    # Contract loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_sat_contracts(
        specs_path: Path | str,
        results_path: Path | str,
    ) -> list[dict[str, Any]]:
        """
        Merge specs with CROWN results; return SAT contracts only.

        Spec fields (bounds, advisories, dangerous_xy) win for identity;
        result status (and any extra result keys) overlay on top.
        """
        specs_file = Path(specs_path)
        results_file = Path(results_path)
        with open(specs_file, encoding="utf-8") as handle:
            specs_payload = json.load(handle)
        with open(results_file, encoding="utf-8") as handle:
            results_payload = json.load(handle)

        spec_list = specs_payload["contracts"]
        result_list = results_payload["contracts"]
        spec_by_id = {int(item["id"]): item for item in spec_list}
        status_by_id = {
            int(item["id"]): item for item in result_list
        }

        sat_contracts: list[dict[str, Any]] = []
        for contract_id, result_record in status_by_id.items():
            if result_record.get("status") != "SAT":
                continue
            if contract_id not in spec_by_id:
                raise KeyError(
                    f"results id={contract_id} missing from specs {specs_file}"
                )
            merged = {**spec_by_id[contract_id], **result_record}
            sat_contracts.append(merged)

        print(
            f"  {len(sat_contracts)} SAT contracts loaded "
            f"(of {len(result_list)} total)"
        )
        return sat_contracts

    # ------------------------------------------------------------------
    # SMV string transforms
    # ------------------------------------------------------------------

    def remove_network_define_blocks(self, smv_text: str) -> tuple[str, int]:
        """Remove all five network_k_1_stage_0 DEFINE case blocks."""
        updated = smv_text
        lines_removed = 0
        for network_index in range(1, 6):
            variable_name = f"network_{network_index}_1_stage_0"
            pattern = (
                r" +" + re.escape(variable_name) + r" :=\s+"
                r"case\s+"
                r".*?"
                r"esac;\n"
            )
            line_count_before = updated.count("\n")
            updated, replacements = re.subn(pattern, "", updated, flags=re.DOTALL)
            if replacements == 0:
                raise ValueError(
                    f"DEFINE block for '{variable_name}' not found in SMV"
                )
            lines_removed += line_count_before - updated.count("\n")
        return updated, lines_removed

    def replace_network_tables_with_free_variable(self, smv_text: str) -> str:
        """
        Declare nn_output_free over the advisory domain and point every NN
        TRUE-branch at it (at most one network runs per tick).
        """
        domain = "{" + ", ".join(self.advisories) + "}"
        free_var_declaration = f"        nn_output_free : {domain};\n"
        var_marker = "--START OF BLACKBOARD VARIABLES DECLARATION\n"
        if var_marker not in smv_text:
            raise ValueError(
                "VAR-section marker '--START OF BLACKBOARD VARIABLES "
                "DECLARATION' not found"
            )
        updated = smv_text.replace(var_marker, var_marker + free_var_declaration, 1)

        for network_index in range(1, 6):
            old_branch = (
                f"                TRUE : network_{network_index}_1_stage_0;"
            )
            new_branch = "                TRUE : nn_output_free;"
            if old_branch not in updated:
                raise ValueError(
                    f"Expected staging assignment for "
                    f"network_{network_index}_1_stage_0 not found"
                )
            updated = updated.replace(old_branch, new_branch, 1)
        return updated

    def build_invar_lines(
        self,
        sat_contracts: list[dict[str, Any]],
    ) -> list[str]:
        """
        One INVAR per lattice cell in each SAT contract:

            command_final != forbidden_advisory
        """
        names = self.smv_variable_names
        lines: list[str] = []
        for contract in sat_contracts:
            heading = contract["heading_own_var"]
            x_sign = contract["x_sign"]
            y_sign = contract["y_sign"]
            a_prev = contract["a_prev"]
            for x_mag, y_mag in contract["dangerous_xy"]:
                condition = (
                    f"system.{names['command_prev']} = {a_prev} & "
                    f"system.{names['heading']} = {heading} & "
                    f"system.{names['x_sign']} = {x_sign} & "
                    f"system.{names['y_sign']} = {y_sign} & "
                    f"system.{names['x_mag']} = {x_mag} & "
                    f"system.{names['y_mag']} = {y_mag}"
                )
                forbidden = contract["forbidden_advisory"]
                lines.append(
                    f"INVAR ({condition}) -> "
                    f"system.{names['command_final']} != {forbidden};"
                )
        return lines

    def inject_invar_lines(self, smv_text: str, invar_lines: list[str]) -> str:
        """Insert INVAR block before the SPECIFICATIONS marker."""
        marker = "--------------SPECIFICATIONS\n"
        if marker not in smv_text:
            raise ValueError("SPECIFICATIONS marker not found in SMV")
        block = (
            "-- A/G contract constraints (verified by alpha-beta-CROWN):\n"
            + "\n".join(invar_lines)
            + "\n"
        )
        return smv_text.replace(marker, marker + block, 1)

    # ------------------------------------------------------------------
    # Full patch
    # ------------------------------------------------------------------

    def patch(
        self,
        base_smv_text: str,
        sat_contracts: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """
        Apply NN abstraction + INVAR injection.

        Returns (patched_smv_text, metrics).
        """
        normalized = base_smv_text.replace("\r\n", "\n")
        without_defines, lines_removed = self.remove_network_define_blocks(
            normalized,
        )
        with_free_var = self.replace_network_tables_with_free_variable(
            without_defines,
        )
        invar_lines = self.build_invar_lines(sat_contracts)
        patched = self.inject_invar_lines(with_free_var, invar_lines)
        metrics = {
            "sat_contracts": len(sat_contracts),
            "invar_lines": len(invar_lines),
            "nn_lines_removed": lines_removed,
        }
        return patched, metrics

    def patch_file(
        self,
        base_smv_path: Path | str,
        output_smv_path: Path | str,
        sat_contracts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Read base SMV, patch, write output; return metrics including wall_sec."""
        base_path = Path(base_smv_path)
        output_path = Path(output_smv_path)
        start = time.perf_counter()
        base_text = base_path.read_text(encoding="utf-8")
        patched_text, metrics = self.patch(base_text, sat_contracts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(patched_text, encoding="utf-8")
        metrics["wall_sec"] = round(time.perf_counter() - start, 3)
        metrics["output_smv_path"] = str(output_path)
        return metrics
