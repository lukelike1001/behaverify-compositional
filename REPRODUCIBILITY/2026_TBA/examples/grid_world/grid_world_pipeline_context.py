"""
grid_world_pipeline_context.py

Resolve paths and prepare the output directory for the grid-world compositional
pipeline. Grid-specific (counter template → .tree); not shared with ACAS Xu.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GridWorldPipelineContext:
    """
    Path bundle consumed by grid-world pipeline stages.

    Built from CLI args via from_cli_arguments(...). The same information was
    previously a plain dict from pipeline.resolve_pipeline_paths.setup().
    """

    network_name: str
    onnx_path: Path
    tree_path: Path
    contracts_path: Path
    smv_path: Path
    nuxmv_out_path: Path
    report_path: Path
    output_dir: Path
    config_path: Path
    nuxmv_bin: Path
    nuxmv_cmd: Path
    metamodel: Path
    skip_contracts: bool

    def as_dict(self) -> dict[str, Any]:
        """Dict form for stages that still expect the historical ctx mapping."""
        return {
            "network_name": self.network_name,
            "onnx_path": self.onnx_path,
            "tree_path": self.tree_path,
            "contracts_path": self.contracts_path,
            "smv_path": self.smv_path,
            "nuxmv_out_path": self.nuxmv_out_path,
            "report_path": self.report_path,
            "output_dir": self.output_dir,
            "config_path": self.config_path,
            "nuxmv_bin": self.nuxmv_bin,
            "nuxmv_cmd": self.nuxmv_cmd,
            "metamodel": self.metamodel,
            "skip_contracts": self.skip_contracts,
        }

    @classmethod
    def from_cli_arguments(
        cls,
        arguments: argparse.Namespace,
        counter_template_path: Path,
    ) -> GridWorldPipelineContext:
        """
        Resolve all pipeline paths and prepare the output directory.

        If --tree is omitted, auto-generates a .tree by substituting the ONNX
        path into counter_template.tree.
        """
        onnx_path = Path(arguments.onnx).resolve()
        output_dir = Path(arguments.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        network_name = onnx_path.stem

        if arguments.tree:
            tree_path = Path(arguments.tree).resolve()
        else:
            tree_path = output_dir / f"{network_name}.tree"
            template_text = counter_template_path.read_text(encoding="utf-8")
            # dsl_to_nuxmv.py builds the ONNX path as:
            #   file_prefix + '/' + source   (string concat, not os.path.join)
            # where file_prefix = tree_file.rsplit('/', 1)[0].
            # Use a CWD-relative path so the concat resolves correctly.
            onnx_relative_path = os.path.relpath(onnx_path, tree_path.parent)
            tree_text = template_text.replace("REPLACE_SOURCE", onnx_relative_path)
            tree_path.write_text(tree_text, encoding="utf-8")
            print(f"[setup] Auto-generated tree: {tree_path}")

        contracts_path = (
            Path(arguments.contracts).resolve()
            if arguments.contracts
            else output_dir / "contracts.json"
        )

        return cls(
            network_name=network_name,
            onnx_path=onnx_path,
            tree_path=tree_path,
            contracts_path=contracts_path,
            smv_path=output_dir / f"{network_name}_contracts.smv",
            nuxmv_out_path=output_dir / "nuxmv_output.txt",
            report_path=output_dir / "pipeline_report.json",
            output_dir=output_dir,
            config_path=Path(arguments.config).resolve(),
            nuxmv_bin=Path(arguments.nuxmv).resolve(),
            nuxmv_cmd=Path(arguments.nuxmv_cmd).resolve(),
            metamodel=Path(arguments.metamodel).resolve(),
            skip_contracts=bool(arguments.skip_contracts),
        )
