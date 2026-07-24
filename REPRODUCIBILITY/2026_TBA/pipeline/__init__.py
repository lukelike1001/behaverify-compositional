"""
pipeline — shared compositional verification infrastructure for 2026_TBA examples.

Package layout:
    process_memory.py           ProcessMemory (peak RSS helpers)
    pipeline_report_writer.py   PipelineReportWriter (JSON + console summary)
    crown_verifier.py           CrownVerifier (alpha-beta-CROWN adapter)
    nuxmv_verifier.py           NuxmvVerifier (nuXmv adapter)

A compositional pipeline is a neural verifier paired with a symbolic one. This
branch supports exactly one of each, so they sit flat. Earlier branches nested
them under neuro/ and symbolic/ to hold alternatives (NNV, UCLID5); restore that
nesting if a second verifier of either kind comes back.

Example pipelines (grid world, ACAS Xu) own the orchestration: they construct
verifiers, run example-specific steps (tree gen, SMV patch, contracts), and
write the report. There is no single NsbtVerifier god-object on this branch.

Usage pattern (from an example script):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # 2026_TBA/

    from pipeline.crown_verifier import CrownVerifier
    from pipeline.nuxmv_verifier import NuxmvVerifier
    from pipeline.pipeline_report_writer import PipelineReportWriter
    from pipeline.process_memory import ProcessMemory
"""
