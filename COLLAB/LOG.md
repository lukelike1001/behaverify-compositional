# Discussion log

Append-only. Signed and timestamped by `collab.py`.

## [claude] 2026-09-28 01:59:57Z

Scaffolding is up. Seeding the ledger with the claims I am carrying into this branch, all of them mine and therefore all unverified.

I want to be blunt about why the ledger exists rather than dress it up: several of these are things I previously stated with more confidence than the evidence supported, and the human caught them, not me. Claims #1 and #4 below are the ones that decide whether this direction is worth anything, and #1 is the weakest — it rests on a single literature search that came back empty, which is not the same as a gap.

Please attack #4 first. If monolithic BehaVerify already handles the largest available trees comfortably, then compositional verification has no room to show an improvement, and we should find that out today rather than in a week.

## [claude] 2026-09-28 02:00:20Z

**CLAIM #1** (unverified): Compositional (assume-guarantee) verification decomposed over behavior-tree STRUCTURE — subtree contracts — is unexplored in the literature.

_How to check:_ My evidence is one web search that returned a large assume-guarantee literature for dynamical/control systems and nothing for behavior trees. That is weak. Search properly: Colledanchise & Ogren's BT book, BT2Fiacre (arXiv 2502.11904), MoVe4BT, Event-B BTs (arXiv 2209.14045), SETTA 2023 BT synthesis, and arXiv 2012.01648 (compositional verification for modular robotic systems, closest hit). Refute me if anyone has done subtree-level A/G.

## [claude] 2026-09-28 02:00:20Z

**CLAIM #2** (unverified): Monolithic BehaVerify already scales comfortably on the largest behavior trees publicly available, which would leave compositional verification no headroom to demonstrate an improvement.

_How to check:_ This is the direction-killer and it is currently UNCHECKED. Data is already in the repo: REPRODUCIBILITY/2025_FMCAD/ and examples/2025_FMCAD_MoVe4BT/ (LTL-BehaVerify-Concise, CTL-BehaVerify-Concise, create_binary_tree.py generates trees at scale). Find the largest tree monolithic handles and its runtime. If it never struggles, say so and this direction dies.

## [claude] 2026-09-28 02:00:20Z

**CLAIM #3** (unverified): examples/ANSR_ONNX/ANSR_ONNX.tree is a non-trivial neuro-symbolic behavior tree: 10 composites, 5 actions, 6 checks, 2 NEURAL variables.

_How to check:_ I got this from grep -c on the .tree file, so the counts are mechanical but the JUDGEMENT that it is 'non-trivial' is mine. Open the file and say whether the tree does real discrete work or is structurally decorative.

## [claude] 2026-09-28 02:00:20Z

**CLAIM #4** (unverified): ANSR has external provenance (DARPA ANSR programme) rather than being constructed in-house by the BehaVerify author.

_How to check:_ UNCHECKED and it matters: an in-house benchmark means arguing with ourselves. See REPRODUCIBILITY/2024_ANSR/README.md and whether NEUS 2025 (proceedings.mlr.press/v288/serbinowska25a.html) cites an external source for it.

## [claude] 2026-09-28 02:00:20Z

**CLAIM #5** (unverified): Every ARCH-COMP AINNCS benchmark collapses to a single behavior-tree node, making that suite useless for demonstrating BT verification.

_How to check:_ Originally the human's hypothesis, and I confirmed it only for NAV. Generalising from one instance is exactly the error I keep making. Check two or three more AINNCS benchmarks before we treat this as settled.

## [claude] 2026-09-28 02:00:20Z

**CLAIM #6** (unverified): nuXmv has a practical ceiling around 2,000 TRANS constraints, above which it times out or segfaults.

_How to check:_ Observed, not documented: NAV segfaulted at 11,627 constraints, ACAS proved at 1,616 and stalled at 2,595. Both were my own models, so the ceiling may be a property of how I encoded them rather than of nuXmv. Worth an independent probe with a deliberately simple model.

## [claude] 2026-09-28 02:00:20Z

**CLAIM #7** (unverified): Contract count scales as (reachable regions) x (distinct contract values), so a benchmark is only worth attempting when BOTH are small.

_How to check:_ Proposed predictor, fitted to three data points after the fact (grid world lost, NAV lost, ACAS won). Never tested predictively. If it is real it should predict ANSR's contract count BEFORE we generate them.

