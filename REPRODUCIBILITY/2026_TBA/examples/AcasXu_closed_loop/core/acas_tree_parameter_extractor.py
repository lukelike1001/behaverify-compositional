"""
acas_tree_parameter_extractor.py

Refresh tree-sourced fields in acas_model_params.yaml from the closed-loop
BehaVerify template. Does not own the whole YAML: initial_state, advisories,
networks, heading_int_degrees, and safety_threshold stay manual.

    from core.acas_tree_parameter_extractor import AcasTreeParameterExtractor

    AcasTreeParameterExtractor().update_model_params_yaml()

CLI (from AcasXu_closed_loop/):

    python3 -m core.acas_tree_parameter_extractor
    python3 -m core.acas_tree_parameter_extractor \\
        --tree tree/acas_closed_loop_template.tree
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Allow `python3 core/acas_tree_parameter_extractor.py` as well as -m.
_EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_ROOT))

from core.paths import EXAMPLE_ROOT

_DEFAULT_TREE = EXAMPLE_ROOT / "tree" / "acas_closed_loop_template.tree"
_DEFAULT_PARAMS = EXAMPLE_ROOT / "core" / "acas_model_params.yaml"

_PHYSICS_FROM_CONSTANTS = (
    "distance_modifier",
    "max_dist",
    "seconds_per_update",
    "degree_multiplier",
)

_NN_NORMALIZATION_FROM_CONSTANTS = (
    "distance_mean",
    "distance_range",
    "speed_own_mean",
    "speed_own_range",
    "speed_int_mean",
    "speed_int_range",
)


@dataclass(frozen=True)
class AcasTreeParameterExtractor:
    """
    Parse primitive numbers out of the ACAS closed-loop .tree template and
    merge them into the model-params YAML (load → patch tree keys → dump).
    """

    tree_path: Path = _DEFAULT_TREE
    params_path: Path = _DEFAULT_PARAMS

    def read_tree_text(self) -> str:
        return self.tree_path.read_text(encoding="utf-8")

    def parse_constants_block(self, tree_text: str) -> dict[str, int | float]:
        """Extract key := value pairs from the template constants { } block."""
        match = re.search(r"\bconstants\s*\{([^}]*)\}", tree_text, re.DOTALL)
        if not match:
            raise ValueError(f"No 'constants {{ }}' block in {self.tree_path}")
        block = match.group(1)
        result: dict[str, int | float] = {}
        for key_match in re.finditer(
            r"\b(\w+)\s*:=\s*([0-9]+(?:\.[0-9]+)?)",
            block,
        ):
            key, value_str = key_match.group(1), key_match.group(2)
            result[key] = float(value_str) if "." in value_str else int(value_str)
        return result

    def parse_literal_speed(self, tree_text: str, variable_name: str) -> int | None:
        """
        Last literal-integer DEFINE for speed_own / speed_int.

        Template may define a formula-based DEFINE then a literal override;
        only assign{result{<int>}} matches.
        """
        matches = re.findall(
            rf"variable\{{env\s+{re.escape(variable_name)}\s+DEFINE\s+INT\s+"
            rf"assign\{{result\{{(\d+)\}}\}}",
            tree_text,
        )
        return int(matches[-1]) if matches else None

    def tree_sourced_updates(self, tree_text: str) -> dict[str, Any]:
        """
        Build the nested patch applied to model params.

        Returns {"physics": {...}, "nn_normalization": {...}} with only
        keys present in the tree.
        """
        constants = self.parse_constants_block(tree_text)
        physics: dict[str, Any] = {}
        for key in _PHYSICS_FROM_CONSTANTS:
            if key in constants:
                physics[key] = constants[key]
        for speed_name in ("speed_own", "speed_int"):
            speed_value = self.parse_literal_speed(tree_text, speed_name)
            if speed_value is not None:
                physics[speed_name] = speed_value

        nn_normalization: dict[str, Any] = {}
        for key in _NN_NORMALIZATION_FROM_CONSTANTS:
            if key in constants:
                nn_normalization[key] = float(constants[key])

        return {
            "physics": physics,
            "nn_normalization": nn_normalization,
        }

    def merge_into_params(
        self,
        params: dict[str, Any],
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Copy params and overlay tree-sourced sections."""
        merged = dict(params)
        if updates.get("physics"):
            physics = dict(merged.get("physics") or {})
            physics.update(updates["physics"])
            merged["physics"] = physics
        if updates.get("nn_normalization"):
            nn_block = dict(merged.get("nn_normalization") or {})
            nn_block.update(updates["nn_normalization"])
            merged["nn_normalization"] = nn_block
        return merged

    def update_model_params_yaml(self) -> dict[str, Any]:
        """
        Read tree + existing YAML, merge tree-sourced keys, write YAML back.

        Returns the updates that were applied (for logging/tests).
        """
        tree_text = self.read_tree_text()
        updates = self.tree_sourced_updates(tree_text)

        with open(self.params_path, encoding="utf-8") as handle:
            params = yaml.safe_load(handle)
        if not isinstance(params, dict):
            raise ValueError(f"{self.params_path}: expected a mapping")

        merged = self.merge_into_params(params, updates)
        with open(self.params_path, "w", encoding="utf-8") as handle:
            yaml.dump(
                merged,
                handle,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        return updates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh tree-sourced fields in core/acas_model_params.yaml "
            "from the ACAS closed-loop .tree template."
        ),
    )
    parser.add_argument(
        "--tree",
        type=Path,
        default=_DEFAULT_TREE,
        help=f"Template .tree path (default: {_DEFAULT_TREE.name})",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=_DEFAULT_PARAMS,
        help="Model params YAML to update",
    )
    args = parser.parse_args()

    extractor = AcasTreeParameterExtractor(
        tree_path=args.tree,
        params_path=args.params,
    )
    updates = extractor.update_model_params_yaml()
    print(f"Updated {args.params} from {args.tree}")
    print(f"  physics keys: {sorted(updates['physics'])}")
    print(f"  nn_normalization keys: {sorted(updates['nn_normalization'])}")


if __name__ == "__main__":
    main()
