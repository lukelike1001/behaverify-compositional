"""
grid_world_smv_builder.py

Build a contract-injected nuXmv SMV from a grid-world .tree + CROWN JSON.

Pipeline step 2: wraps 2026_TBA/src/dsl_with_contracts_to_nuxmv with grid-specific
variable names and records wall time / RSS / SAT-INVAR count.

Kept in grid_world/ (not 2026_TBA/pipeline/) because the SMV patch parameters
(neural_var, pos_*, action domain) are example-specific.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent.resolve()
_TBA = (_HERE / "../../").resolve()
if str(_TBA) not in sys.path:
    sys.path.insert(0, str(_TBA))

from pipeline.process_memory import ProcessMemory  # noqa: E402


@dataclass
class GridWorldSmvBuilder:
    """
    Paths + SMV symbol names needed to emit a contract-based nuXmv model.

    Prefer from_pipeline_ctx() when calling from run_compositional_pipeline.
    """

    metamodel: Path
    tree_path: Path
    contracts_path: Path
    smv_path: Path
    neural_var: str
    pos_x: str
    pos_y: str
    domain: list[str]
    src_dir: str
    goal_x: str | None = None
    goal_y: str | None = None

    @classmethod
    def from_pipeline_ctx(
        cls,
        ctx: dict[str, Any],
        smv_cfg: dict[str, Any],
    ) -> GridWorldSmvBuilder:
        """
        Assemble from pipeline context dict and smv section of
        pipeline_filepaths_config.yaml (plus src_dir).
        """
        return cls(
            metamodel=Path(ctx["metamodel"]),
            tree_path=Path(ctx["tree_path"]),
            contracts_path=Path(ctx["contracts_path"]),
            smv_path=Path(ctx["smv_path"]),
            neural_var=str(smv_cfg["neural_var"]),
            pos_x=str(smv_cfg["pos_x"]),
            pos_y=str(smv_cfg["pos_y"]),
            domain=list(smv_cfg["domain"]),
            src_dir=str(smv_cfg["src_dir"]),
            goal_x=smv_cfg.get("goal_x"),
            goal_y=smv_cfg.get("goal_y"),
        )

    def count_sat_contracts(self) -> int:
        with open(self.contracts_path, encoding="utf-8") as f:
            data = json.load(f)
        return sum(1 for c in data.get("contracts", []) if c["status"] == "SAT")

    def generate(self) -> dict[str, Any]:
        """
        Run dsl_with_contracts_to_nuxmv and return pipeline metrics:

            wall_sec, peak_rss_kb, peak_traced_bytes, sat_contracts_injected
        """
        print("\n" + "=" * 60)
        print("[2/3] SMV GENERATION")
        print("=" * 60)

        if self.src_dir not in sys.path:
            sys.path.insert(0, self.src_dir)

        import dsl_with_contracts_to_nuxmv as _conv  # noqa: PLC0415

        tracemalloc.start()
        t0 = time.perf_counter()

        _conv.dsl_with_contracts_to_nuxmv(
            metamodel_file=str(self.metamodel),
            tree_file=str(self.tree_path),
            output_file=str(self.smv_path),
            contracts_file=str(self.contracts_path),
            neural_var=self.neural_var,
            pos_x=self.pos_x,
            pos_y=self.pos_y,
            domain=self.domain,
            dir_map=_conv.DEFAULT_DIR_MAP,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            skip_grammar_check=True,
        )

        wall_sec = time.perf_counter() - t0
        rss_after = ProcessMemory.peak_self_rss_kilobytes()
        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        sat_injected = self.count_sat_contracts()
        metrics = {
            "wall_sec": round(wall_sec, 3),
            "peak_rss_kb": rss_after,
            "peak_traced_bytes": peak_traced,
            "sat_contracts_injected": sat_injected,
        }
        print(
            f"\n[smv] {wall_sec:.1f}s  |  peak RSS ≥{rss_after} KB  |  "
            f"{sat_injected} INVAR constraints injected"
        )
        return metrics


def run_smv_generation(ctx: dict[str, Any], smv_cfg: dict[str, Any]) -> dict[str, Any]:
    """Thin pipeline facade (same role as run_verification for the verifier)."""
    return GridWorldSmvBuilder.from_pipeline_ctx(ctx, smv_cfg).generate()
