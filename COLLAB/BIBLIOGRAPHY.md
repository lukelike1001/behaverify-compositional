# Definitive Bibliography of Serena S. Serbinowska and the BehaVerify Toolchain

**Prepared by:** Gemini (Independent Auditor & Verification Agent)  
**Date:** 2026-09-30  
**Project:** Compositional BehaVerify (`COLLAB/`)  
**Scope:** Exhaustive catalog of every publication, dissertation, book chapter, preprint, tool paper, and artifact by Serena Serafina Serbinowska (author ORCID: `0000-0002-9259-1586`); complete mapping of the 14 `REPRODUCIBILITY/` venue folders; repository PDF audit; and technical cross-examination across all 4 key compositional questions.

---

## 1. Complete Publications of Serena Serafina Serbinowska

### A. Core BehaVerify Publications (Peer-Reviewed)

1. **BehaVerify: Verifying Temporal Logic Specifications for Behavior Trees**
   - **Authors:** Serena S. Serbinowska, Taylor T. Johnson
   - **Venue:** 20th International Conference on Software Engineering and Formal Methods (SEFM 2022), Berlin, Germany, September 26–30, 2022
   - **Volume/Pages:** Lecture Notes in Computer Science (LNCS), vol. 13550, pp. 307–323, Springer, Cham, 2022
   - **DOI:** [10.1007/978-3-031-17108-6_19](https://doi.org/10.1007/978-3-031-17108-6_19)
   - **Preprint / Open Access:** arXiv:2208.05360 [cs.RO] (v1: Aug 10, 2022; v2: Feb 20, 2024), [https://arxiv.org/abs/2208.05360](https://arxiv.org/abs/2208.05360) (PDF: [https://arxiv.org/pdf/2208.05360](https://arxiv.org/pdf/2208.05360))
   - **Relation:** Foundational paper introducing BehaVerify (translating PyTrees directly into nuXmv SMV models; Total_v3 encoding; evaluations on Checklist, Parallel-Checklist, and BlueROV).

2. **Formalizing Stateful Behavior Trees**
   - **Authors:** Serena S. Serbinowska, Preston Robinette, Gabor Karsai, Taylor T. Johnson
   - **Venue:** 6th International Workshop on Formal Methods for Autonomous Systems (FMAS 2024), Manchester, UK, November 11–12, 2024
   - **Volume/Pages:** Electronic Proceedings in Theoretical Computer Science (EPTCS), vol. 411, pp. 201–218, 2024
   - **DOI:** [10.4204/EPTCS.411.14](https://doi.org/10.4204/EPTCS.411.14)
   - **Open Access:** EPTCS Open Access: [https://eptcs.episciences.org/14801/pdf](https://eptcs.episciences.org/14801/pdf); arXiv:2411.14165 [cs.RO], [https://arxiv.org/abs/2411.14165](https://arxiv.org/abs/2411.14165)
   - **Relation:** Introduces the dedicated Domain Specific Language (DSL with textX) for Stateful Behavior Trees (SBTs) with blackboard variables; proves Turing completeness with unbounded integers and FSM equivalence with finitary types; introduces Fastforwarding (collapsing tick execution to single transitions, 100x faster than MoVe4BT).

3. **Verification of Behavior Trees with Contingency Monitors**
   - **Authors:** Serena S. Serbinowska, Nicholas Potteiger, Anne M. Tumlin, Taylor T. Johnson
   - **Venue:** 6th International Workshop on Formal Methods for Autonomous Systems (FMAS 2024), Manchester, UK, November 11–12, 2024
   - **Volume/Pages:** Electronic Proceedings in Theoretical Computer Science (EPTCS), vol. 411, pp. 56–72, 2024
   - **DOI:** [10.4204/EPTCS.411.4](https://doi.org/10.4204/EPTCS.411.4)
   - **Open Access:** EPTCS Open Access: [https://eptcs.episciences.org/14791/pdf](https://eptcs.episciences.org/14791/pdf); arXiv:2411.14162 [cs.RO], [https://arxiv.org/abs/2411.14162](https://arxiv.org/abs/2411.14162)
   - **Relation:** Introduces runtime contingency monitors ($BT^M$) synthesized from LTL specifications into Python, verified design-time in nuXmv and evaluated in AirSim drone simulations.

4. **Neuro-Symbolic Behavior Trees (NSBTs) and Their Verification**
   - **Authors:** Serena S. Serbinowska, Diego Manzanas Lopez, Dung Thuy Nguyen, Taylor T. Johnson
   - **Venue:** 2nd International Conference on Neuro-symbolic Systems (NeuS 2025)
   - **Volume/Pages:** Proceedings of Machine Learning Research (PMLR), vol. 288, pp. 409–423, 2025
   - **Open Access:** [https://proceedings.mlr.press/v288/serbinowska25a.html](https://proceedings.mlr.press/v288/serbinowska25a.html) (PDF: [https://proceedings.mlr.press/v288/serbinowska25a/serbinowska25a.pdf](https://proceedings.mlr.press/v288/serbinowska25a/serbinowska25a.pdf))
   - **Relation:** The monolithic table paper. Extends SBTs to Neuro-Symbolic Behavior Trees (NSBTs) with neural leaves; evaluates bit-vector/fixed-point vs exhaustive lookup table encodings on Grid World and ACAS Xu; Section 5 explicitly proposes assume-guarantee compositional contracts as future work.

---

### B. Tool Papers, Manuscripts Under Review & In-Repo Manuals

5. **BehaVerify: A Formal Verification Tool for Behavior Trees** (Tool Paper under review)
   - **Authors:** Serena S. Serbinowska, Ben Wooding, Diego Manzanas Lopez, Luke Nam, Anne M. Tumlin, Nicholas K. Potteiger, Hongchao Zhang, Taylor T. Johnson
   - **Venue / Status:** Under review at Formal Methods 2026 (FM 2026). Preceded by ATVA tool paper submission and archived artifact.
   - **Artifact DOI:** [10.5281/zenodo.20280547](https://doi.org/10.5281/zenodo.20280547)
   - **PDF Location:** Local in repository at [REPRODUCIBILITY/2026_TBA/reports/FM_2026_Submission.pdf](file:///home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2026_TBA/reports/FM_2026_Submission.pdf)
   - **Relation:** Comprehensive tool paper documenting the mature BehaVerify pipeline: DSL front end, Fastforwarding vs Naive ablation (Table 4), cross-tool comparisons against BT2Fiacre (Drone3/DroneNew), BT2BIP (MarsRover/TrainControl), MoVe4BT, and an NSBT neural network example (Table 3).

6. **BehaVerify: Getting Started and User Manual** (In-Repo Documentation)
   - **Author:** Serena S. Serbinowska
   - **Format:** In-repository technical manual / tutorial guide (14 pages)
   - **PDF Location:** Local in repository root at [BehaVerify.pdf](file:///home/lnam/VSCode/behaverify-compositional/BehaVerify.pdf)
   - **Relation:** Practical user manual explaining Docker installation, Python virtual environment, `.tree` DSL syntax, code generation modes (nuXmv, Python, Haskell, LaTeX), tutorial models (GCD, Collatz, DrunkenDrone), and ONNX neural network integration.

---

### C. Doctoral Dissertation

7. **On Behavior Trees and Their Verification**
   - **Author:** Serena Serafina Serbinowska
   - **Type:** Ph.D. Dissertation in Computer Science
   - **Institution:** Graduate School of Vanderbilt University, Department of Computer Science
   - **Defense / Approval Date:** August 8, 2025
   - **Committee:** Taylor Thomas Johnson (Chair/Advisor), Gabor Karsai, et al.
   - **Availability:** ProQuest Dissertations & Theses Global, 2025 (Publication info indexed; institutional access / paywalled; embargoed / not yet deposited in Vanderbilt DiscoverArchive).
   - **Relation:** Direct synthesis of Serbinowska's doctoral research, compiling the theoretical and empirical contributions of papers #1 (SEFM 2022), #2 (FMAS 2024 SBT), #3 (FMAS 2024 BTM), and #4 (NeuS 2025).

---

### D. Other Publications & Book Chapters (Unrelated or Broader Collaborative)

8. **Case Study: Neural Network Malware Detection Verification for Feature and Image Datasets**
   - **Authors:** Preston K. Robinette, Diego Manzanas Lopez, Serena Serbinowska, Kevin Leach, Taylor T. Johnson
   - **Venue:** 2024 IEEE/ACM 12th International Conference on Formal Methods in Software Engineering (FormaliSE 2024), Lisbon, Portugal, April 14–15, 2024
   - **Volume/Pages:** IEEE/ACM, pp. 91–101, 2024
   - **DOI:** [10.1145/3644033.3644372](https://doi.org/10.1145/3644033.3644372) / [10.1109/FormaliSE62297.2024.00017](https://doi.org/10.1109/FormaliSE62297.2024.00017)
   - **Open Access:** arXiv:2404.05703 [cs.CR], [https://arxiv.org/abs/2404.05703](https://arxiv.org/abs/2404.05703)
   - **Relation:** **UNRELATED to BehaVerify.** Evaluates the adversarial robustness of neural network malware classifiers (EMBER tabular features and MalImg grayscale images) using the NNV (Neural Network Verification) reachability toolbox. Does not involve Behavior Trees.

9. **Toward Assured Autonomy Using Neurosymbolic Components and Systems**
   - **Authors:** Abhishek Dubey, Taylor T. Johnson, Xenofon Koutsoukos, Baiting Luo, Diego Manzanas Lopez, Miklos Maroti, Ayan Mukhopadhyay, Nicholas Potteiger, Serena Serbinowska, Daniel Stojcsics, Yunuo Zhang, Gabor Karsai
   - **Type:** Book Chapter (Chapter 4)
   - **Book:** *Neurosymbolic AI: Foundations and Applications*, edited by Gabor Karsai et al.
   - **Publisher:** John Wiley & Sons, Ltd, 2026
   - **Pages:** pp. 89–118
   - **DOI:** [10.1002/9781394302406.ch04](https://doi.org/10.1002/9781394302406.ch04)
   - **Availability:** Paywalled (Wiley).
   - **Relation:** Collaborative survey / foundational chapter on neuro-symbolic assurance architectures for autonomous systems from Vanderbilt's SCOPE Lab and VeriVITAL groups.

---

### E. Archived Research Artifacts (Zenodo / DBLP)

10. **2023 SEFM Artifact for BehaVerify (A Domain Specific Language (DSL) for Behavior Tree Verification and Implementation)**
    - **Authors:** Serena Serbinowska, Dániel Stojcsics, Nagabhushan Mahadevan, Gabor Karsai, Taylor T. Johnson
    - **Platform:** Zenodo, 2023 (Indexed on DBLP)
    - **Relation:** Artifact for the DSL and differential testing pipeline (corresponds to `REPRODUCIBILITY/2023_SEFM/` and the Ph.D. qualifying exam).

11. **2024 VMCAI Artifact for BehaVerify (Verification of Stateful Behavior Trees)**
    - **Authors:** Serena Serbinowska, Dániel Stojcsics, Nagabhushan Mahadevan, Gabor Karsai, Taylor T. Johnson
    - **Platform:** Zenodo, 2023 (Indexed on DBLP)
    - **Relation:** Artifact for the VMCAI 2024 submission (corresponds to `REPRODUCIBILITY/2024_VMCAI/`).

12. **BehaVerify: Artifact for ATVA 2026 Tool Track**
    - **Authors:** Serena S. Serbinowska, Ben Wooding, Diego Manzanas Lopez, Luke Nam, Anne M. Tumlin, Nicholas K. Potteiger, Hongchao Zhang, Taylor T. Johnson
    - **Platform:** Zenodo, DOI: [10.5281/zenodo.20280547](https://doi.org/10.5281/zenodo.20280547)
    - **Relation:** Standalone artifact containing `behaverify-compare.tar.gz` for reproducing the ablation and cross-tool comparisons (corresponds to `REPRODUCIBILITY/2026_ATVA/`).

---

## 2. REPRODUCIBILITY Folder Mapping Table

Why does `REPRODUCIBILITY/` have fourteen venue folders while `CLAUDE.md` names only four publications?  
**The Answer:** `CLAUDE.md` listed only **accepted, published papers** (SEFM 2022, FMAS 2024 SBT, FMAS 2024 BTM, NeuS 2025). The remaining ten folders represent **prior conference submission iterations of those exact papers**, **doctoral qualifying exam artifacts**, **funded project deliverables**, or **tool-track submissions**:

| Folder Name | Primary Status | Belongs To Which Paper / Artifact | Exact Experiments & Resolution Notes |
|---|---|---|---|
| `2022_SEFM` | **Published Paper Artifact** | **SEFM 2022 Paper** (#1: *BehaVerify: Verifying Temporal Logic Specifications for Behavior Trees*) | PyTrees-to-nuXmv translation. Models: Checklist (5..50 nodes), Parallel-Checklist, and BlueROV (warnings, small, full). |
| `2023_SEFM` | **Qual Exam / Rejected Submission Artifact** | **Ph.D. Qualifying Exam & SEFM 2023 Submission Attempt** (Zenodo 2023 Artifact #10) | Git commit `5346aea4d` explicitly records: *"ran tests for qual exam"*. Scripts: `QUAL_EXAM_exp_checklist_create.sh`. Experiments: Checklist sequence/parallel, Bigger Fish Expanded, Light Controller, BlueROV Mod, Simple Robot, Differential Testing. Paper not published at SEFM 2023. |
| `2024_ANSR` | **Project Deliverable (No Paper)** | **DARPA ANSR Program Deliverable** (Contract FA8750-23-C-0518) | DARPA Assured Neuro-Symbolic Learning and Reasoning (ANSR) Stage 1 deliverable (`parse_nuxmv_output_stage_1.py`). Evaluates early neuro-symbolic encodings: `small/float.tree`, `small/fixed.tree`, `small/table.tree`, `big/pre_table.tree`. This was the direct prototype that matured into NeuS 2025. |
| `2024_ATVA` | **Rejected Submission Artifact** | **Predecessor of FMAS 2024 SBT Paper** (#2) | Submission iteration of *Formalizing Stateful Behavior Trees* to ATVA 2024. README was lazily copied from NFM 2024 (*"reproducing results used in NFM 2024"*). Experiments: ISR, Bigger Fish, Simple Robot. Rejected from ATVA 2024; later accepted at FMAS 2024. |
| `2024_FMAS_BTM` | **Published Paper Artifact** | **FMAS 2024 BTM Paper** (#3: *Verification of Behavior Trees with Contingency Monitors*) | Runtime contingency monitoring ($BT^M$). Experiments: Drone grid navigation (10x10 to 50x50, dense fixed and sparse random), runtime monitors in Python/AirSim, design-time verification in nuXmv. |
| `2024_FMAS_SBT` | **Published Paper Artifact** | **FMAS 2024 SBT Paper** (#2: *Formalizing Stateful Behavior Trees*) | Stateful Behavior Trees & Fastforwarding. Experiments: Bigger Fish (tree scaling to 20k nodes), Simple Robot (grid scaling to 30x30), Moving Target / ISR (obstacle navigation counterexample), compared against MoVe4BT. |
| `2024_NFM` | **Rejected Submission Artifact** | **Predecessor of FMAS 2024 SBT Paper** (#2) | Earlier submission iteration of *Formalizing Stateful Behavior Trees* to NASA Formal Methods (NFM 2024). Experiments: `ANSR_no_net` (ISR), `bigger_fish_selector`, `simple_robot`. Rejected from NFM 2024. |
| `2024_SEFM` | **Rejected Submission Artifact** | **Predecessor of FMAS 2024 SBT Paper** (#2) | Another submission iteration of *Formalizing Stateful Behavior Trees* to SEFM 2024. Experiments: ISR, Bigger Fish, Simple Robot. Rejected from SEFM 2024; finally accepted at FMAS 2024. |
| `2024_VMCAI` | **Rejected Submission Artifact** | **VMCAI 2024 Submission Attempt** (Zenodo 2023 Artifact #11) | Submission to VMCAI 2024 (London, Jan 2024) titled *BehaVerify: Verification of Stateful Behavior Trees*. Experiments: `light_controller_v3`, `blueROV_mod`, `bigger_fish_expanded`, `simple_robot`, `differential_testing`. Artifact published on Zenodo; conference paper rejected. |
| `2025_CAV` | **Rejected Submission Artifact / Unfinished Attempt** | **CAV 2025 Submission Attempt** (No README) | Prepared for CAV 2025. Git commit `735f0cae0` says *"tried making reals work"*; commit `ad4217056` *"minor updates for CAV"*. Experiments: Collatz conjecture BTs (`collatz_254.tree`), Binary Tree scaling (1 & 2), Simple Robot (1 & 2), Robot, Bigger Fish, ISR. Paper not accepted / no published record. |
| `2025_FMCAD` | **Predecessor Tool Paper Artifact** | **Predecessor of FM 2026 Tool Paper** (#5) | Tool paper submission to FMCAD 2025 (*BehaVerify: A Formal Verification Tool for Behavior Trees*). Experiments: `2025_FMCAD_BT2BIP`, `2025_FMCAD_BT2Fiacre`, `2025_FMCAD_MoVe4BT`, `DrunkenDrone`, `NetworkExample`. Evolved into FM 2026. |
| `2025_NEUS` | **Published Paper Artifact** | **NeuS 2025 Paper** (#4: *Neuro-Symbolic Behavior Trees and Their Verification*) | Monolithic lookup table paper. Contains paper PDF `serbinowska25a.pdf`. Experiments: Prime Position, Grid World (small 7x7 and large 50x50), ACAS Xu (local robustness and closed-loop). |
| `2026_ATVA` | **Tool Paper Artifact** | **ATVA 2026 Tool Track Submission** (Zenodo Artifact #12) | Artifact for BehaVerify tool paper at ATVA 2026. Table 2: Fastforwarding vs Naive ablation ($N=1..10$); Table 3: Drone3/DroneNew vs BT2Fiacre; Sec 4.3: BT2BIP MarsRover + TrainControl. Archived on Zenodo (DOI 10.5281/zenodo.20280547). |
| `2026_FM` | **Tool Paper Under Review Artifact** | **FM 2026 Tool Paper Submission** (#5) | Updated replication package for `FM_2026_Submission.pdf`. Experiments: BT2BIP (MarsRover, TrainControl), BT2Fiacre (Drone3, DroneNew), MoVe4BT, DrunkenDrone, NetworkExample, and Binary Tree encoding comparison. |
| `2026_TBA` | **Active Research Workspace** | **Compositional BehaVerify Project** (Our Workspace) | Contains active compositional verification pipeline (`examples/grid_world`, `examples/AcasXu_closed_loop`), $\alpha,\beta$-CROWN integration, UCLID5 experiments, and submission drafts (`reports/FM_2026_Submission.pdf`). |

---

## 3. Repository PDF Audit: Missing Papers List for Luke

| Paper / Document Title | In Repo as PDF? | Status | Open Access URL / Paywall Notice | Action Required for Luke |
|---|---|---|---|---|
| **SEFM 2022:** *BehaVerify: Verifying Temporal Logic Specifications for Behavior Trees* | ❌ **MISSING** (`./BehaVerify.pdf` in root is the Tool Manual, NOT this paper) | **Open Access (arXiv)** | [https://arxiv.org/pdf/2208.05360](https://arxiv.org/pdf/2208.05360) (Springer version paywalled: DOI [10.1007/978-3-031-17108-6_19](https://doi.org/10.1007/978-3-031-17108-6_19)) | **None needed** — we downloaded the full text via arXiv to `scratch/sefm2022.pdf`. Luke can copy it to `REPRODUCIBILITY/2022_SEFM/` if desired. |
| **FMAS 2024 (SBT):** *Formalizing Stateful Behavior Trees* | ❌ **MISSING** | **Open Access (EPTCS / arXiv)** | EPTCS PDF: [https://eptcs.episciences.org/14801/pdf](https://eptcs.episciences.org/14801/pdf); arXiv PDF: [https://arxiv.org/pdf/2411.14165](https://arxiv.org/pdf/2411.14165) | **None needed** — downloaded to `scratch/fmas2024_sbt.pdf`. Can be committed to `REPRODUCIBILITY/2024_FMAS_SBT/`. |
| **FMAS 2024 (BTM):** *Verification of Behavior Trees with Contingency Monitors* | ❌ **MISSING** | **Open Access (EPTCS / arXiv)** | EPTCS PDF: [https://eptcs.episciences.org/14791/pdf](https://eptcs.episciences.org/14791/pdf); arXiv PDF: [https://arxiv.org/pdf/2411.14162](https://arxiv.org/pdf/2411.14162) | **None needed** — downloaded to `scratch/fmas2024_btm.pdf`. Can be committed to `REPRODUCIBILITY/2024_FMAS_BTM/`. |
| **NeuS 2025:** *Neuro-Symbolic Behavior Trees (NSBTs) and Their Verification* | ✅ **PRESENT** | **Open Access (PMLR)** | Present at [REPRODUCIBILITY/2025_NEUS/serbinowska25a.pdf](file:///home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/serbinowska25a.pdf) | None. Read in full. |
| **FM 2026 Submission:** *BehaVerify: A Formal Verification Tool for Behavior Trees* | ✅ **PRESENT** | **Under Review** | Present at [REPRODUCIBILITY/2026_TBA/reports/FM_2026_Submission.pdf](file:///home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2026_TBA/reports/FM_2026_Submission.pdf) | None. Read in full. |
| **BehaVerify Tool Manual:** *Getting Started & User Guide* | ✅ **PRESENT** | **In-Repo Guide** | Present at [./BehaVerify.pdf](file:///home/lnam/VSCode/behaverify-compositional/BehaVerify.pdf) | None. Read in full. |
| **Ph.D. Dissertation (2025):** *On Behavior Trees and Their Verification* | ❌ **MISSING** | 🔒 **PAYWALLED / INSTITUTIONAL ACCESS ONLY** (ProQuest) | ProQuest Dissertations: [ProQuest Link](https://www.proquest.com/openview/8a86a60dbbcf469df5d2a33f382a3928/1); not yet deposited in Vanderbilt DiscoverArchive. | **LUKE MUST SUPPLY** via Vanderbilt University Library ProQuest access if the group wants the complete dissertation text (though its content synthesizes the 4 papers above). |
| **Book Chapter (2026):** *Toward Assured Autonomy Using Neurosymbolic Components and Systems* | ❌ **MISSING** | 🔒 **PAYWALLED** (Wiley) | Wiley Online Library: [10.1002/9781394302406.ch04](https://doi.org/10.1002/9781394302406.ch04) | **LUKE MUST SUPPLY** via institutional Wiley access if required (survey/overview chapter). |
| **FormaliSE 2024:** *Case Study: Neural Network Malware Detection Verification for Feature and Image Datasets* | ❌ **MISSING** | **Open Access (arXiv)** | arXiv PDF: [https://arxiv.org/pdf/2404.05703](https://arxiv.org/pdf/2404.05703) | **Unrelated to BehaVerify** — no supply needed. |

---

## 4. Deep Technical Cross-Examination on the Four Core Questions

Every paper listed below was read in full text (downloaded from open access or local repository). The four questions are answered directly from the text with exact quotations.

---

### Paper 1: SEFM 2022 — *BehaVerify: Verifying Temporal Logic Specifications for Behavior Trees*
*(Authors: Serena S. Serbinowska, Taylor T. Johnson; read from full text of arXiv:2208.05360)*

- **(a) Benchmarks & Neural Network Profile:**
  - **Benchmarks:**
    1. *Checklist and Parallel-Checklist:* Synthetic benchmarks scaling $N = 1 \dots 50$ tasks chained under Sequence/Selector or Parallel composite nodes.
    2. *BlueROV:* Realistic mission controller for the BlueROV autonomous underwater vehicle (from an ongoing DARPA project) with $\sim 100+$ nodes and blackboard variables, evaluating Warnings-only, Small, and Full variable ranges against 10 LTL specifications.
  - **Neural Network Profile:** **ZERO.** Neural networks are neither implemented, benchmarked, nor mentioned.

- **(b) Compositional / Modular / Contract-Based Verification:**
  - **None.** The paper contains no assume-guarantee, modular, or contract-based formal verification. The term *"modular"* appears only in the informal software engineering sense in the introduction:
    > *"Behavior Trees have a variety of strengths: they facilitate code re-use (nodes and sub-trees can easily be attached), their modular nature makes reasoning about them easier, and changing one region of a tree doesn’t affect how other regions function."* (Section 1, p. 1)

- **(c) Limitations and Future Work:**
  - **Author's Statements (Section 6, p. 15):**
    > *"Future work includes general polish and improvements and expanding support for the various built-in nodes in PyTrees. In addition to this, we plan to re-work certain elements of BehaVerify. For instance, currently, in order for BehaVerify to detect blackboard variables in a PyTree using custom leaf nodes, the user must create a field that BehaVerify looks for within the custom node. This could certainly be handled better in the future.*  
    > *In terms of encodings, we plan to focus on Total_v3. An improvement that has been considered, but not yet implemented, would be to restrict the incoming values to the leaf nodes to reduce state space. Specifically, in cases where a leaf node does not run, there is no need to consider the incoming status. Currently, this could be accomplished by tying the incoming value to the active value. However, this would likely cause worse performance for the same reason that Total_v2 performs worse than Total_v3. Therefore, the intended solution would be to, in some sense, enumerate all possible input values, which would hopefully shift some of the burden off of nuXmv and onto BehaVerify."*

- **(d) Parallel Nodes, Node Status/Lifecycle, Statefulness/Memory:**
  - **Parallel Nodes:** Formally analyzed and benchmarked in *Parallel-Checklist* vs *Checklist*. The author identifies a fundamental architectural difference between Sequence/Selector and Parallel composites (Section 5.1, p. 13):
    > *"Note that Total_v2 works much better on Parallel-Checklist than on Checklist. This is because of the logic chain created by Selector and Sequence nodes... The status of child3 depends on if child3 is active, which depends on the status of child2... The chain quickly becomes unmanageable. This is not the case with Parallel-Checklist... The status of child3 depends on if child3 is active, which depends only on par0 and what child3 returned last time. Thus the dependency chain is much shorter and thus Total_v2 performs better on Parallel-Checklist."*
  - **Node Lifecycle & Memory:** Formally tracks node status as $\text{Success } (S)$, $\text{Failure } (F)$, $\text{Running } (R)$, or $\text{Invalid/Skipped}$. Sequence and Selector nodes with memory are supported by recording past child returns so subsequent ticks skip already-successful (sequence) or already-failed (selector) children.

---

### Paper 2: FMAS 2024 — *Formalizing Stateful Behavior Trees*
*(Authors: Serena S. Serbinowska, Preston Robinette, Gabor Karsai, Taylor T. Johnson; read from full text of arXiv:2411.14165)*

- **(a) Benchmarks & Neural Network Profile:**
  - **Benchmarks:**
    1. *Bigger Fish (Tree Scaling):* Scales tree size up to 20,000 nodes while keeping blackboard variables small (integer $f \in [0, 1000]$).
    2. *Simple Robot (Blackboard Scaling):* Robot on an $n \times n$ grid reaching sequential nondeterministic goals; scales grid from $2 \times 2$ to $30 \times 30$ ($2^{22}$ reachable states) with a constant tree size.
    3. *Moving Target / ISR:* Intelligence, Surveillance, Reconnaissance drone searching a grid for a mobile target obscured by visual tree obstacles; verifies detection when target moves every 10 turns, and synthesizes a complex evasion counterexample when target moves every 5 turns.
  - **Neural Network Profile:** **ZERO.** No neural networks are used.

- **(b) Compositional / Modular / Contract-Based Verification:**
  - **None.** Does not contain assume-guarantee, contract-based, or compositional verification. Verification is strictly monolithic symbolic model checking via nuXmv.

- **(c) Limitations and Future Work:**
  - **Author's Statements (Section 8, p. 216):**
    > *"Potential future work includes improving our encoding of array variables, developing a graphical user interface for the creation of SBTs, visualization of counterexamples provided by nuXmv, general performance improvements, and expanding support for unbounded variable types in nuXmv for bounded model checking (BMC)."*

- **(d) Parallel Nodes, Node Status/Lifecycle, Statefulness/Memory:**
  - **Theoretical Foundation:** Formally defines Stateful Behavior Trees (SBTs) over blackboard memory $\mathcal{V}_{bl}$ and environment $\mathcal{V}_{env}$.
  - **Expressiveness Theorems:**
    - Demonstrates SBTs are **Turing Complete** when blackboard variables store unbounded mathematical integers (Theorem 1, simulating Turing machine tape and state transitions).
    - Demonstrates SBTs are equivalent to **Finite State Automata (FSMs)** when blackboard variables are restricted to finitary types (Theorem 2).
  - **Fastforwarding Innovation:** Introduces *Fastforwarding*, evaluating the entire tree tick in a single transition step rather than stepping node-by-node. The author proves this collapses trace length from quadratic ($\sum_{i=1}^N i \approx \frac{N^2}{2} = 2 \times 10^8$ steps in MoVe4BT for 20k nodes) down to a linear trace ($< 20,000$ steps in BehaVerify), explaining BehaVerify's 100x scalability advantage.
  - **Node Status Expressiveness:** Highlights that MoVe4BT can only state temporal properties over node return statuses ($\text{MoVe4BTCheck} = S$), whereas BehaVerify supports temporal specifications combining node statuses and internal blackboard variables ($\Diamond\square(f = n)$).

---

### Paper 3: FMAS 2024 — *Verification of Behavior Trees with Contingency Monitors*
*(Authors: Serena S. Serbinowska, Nicholas Potteiger, Anne M. Tumlin, Taylor T. Johnson; read from full text of arXiv:2411.14162)*

- **(a) Benchmarks & Neural Network Profile:**
  - **Benchmarks:**
    - *Drone Obstacle Navigation:* Drone navigating discrete grids from $10 \times 10$ to $50 \times 50$ under *dense fixed* and *sparse random* obstacle topologies, navigating to randomly assigned destinations while satisfying safety ($\varphi_S = \square(\text{no collision})$) and liveness ($\varphi_L = \square\Diamond(\text{reach goal})$). Evaluated in Python runtime execution, AirSim simulation, and nuXmv design-time verification.
  - **Neural Network Profile:** **ZERO.** The term *"neural"* appears zero times in the paper.

- **(b) Compositional / Modular / Contract-Based Verification:**
  - **None.** Does not use assume-guarantee contracts. Monitors are compiled as explicit subtrees ($BT^M$) and verified monolithically.

- **(c) Limitations and Future Work:**
  - **Author's Statements (Section 7, p. 70):**
    > *"We subsequently hope to expand the target range of BehaVerify, specifically to create .cpp implementations that make use of BehaviorTrees.cpp."*
  - **Liveness Scalability Limitation (Section 6, p. 68):**
    > *"While we created such translations for each combination of grid type and grid size presented in Section 5, some of the resulting models proved too complex for liveness analysis in nuXmv... The liveness situation is somewhat trickier, as the specifications are substantially harder to verify. However, considering the fact that the same liveness monitor is used for all grids and that we verified it for one grid, even this limited verification process can provide some evidence to indicate that the monitor is correct."*

- **(d) Parallel Nodes, Node Status/Lifecycle, Statefulness/Memory:**
  - Introduces Runtime Contingency Monitors that continuously track the execution status of monitored subtrees. When a monitor condition fails, control switches to a fallback contingency subtree.
  - Node lifecycle properties are monitored dynamically at runtime and verified statically at design time using LTL specifications.

---

### Paper 4: NeuS 2025 — *Neuro-Symbolic Behavior Trees (NSBTs) and Their Verification*
*(Authors: Serena S. Serbinowska, Diego Manzanas Lopez, Dung Thuy Nguyen, Taylor T. Johnson; read from full text of PMLR vol. 288 / `REPRODUCIBILITY/2025_NEUS/serbinowska25a.pdf`)*

- **(a) Benchmarks & Neural Network Profile:**
  - **Benchmarks:**
    1. *Prime Position:* Introductory model with a 1-input neural network classifying integers in $[0, 9]$ as prime or composite.
    2. *Grid World:* Drone navigating a $7 \times 7$ small grid (2,401 drone/target states) and $50 \times 50$ large grid ($6.25 \times 10^6$ states) avoiding static obstacles, using an A*-trained navigation network.
    3. *ACAS Xu:* Unmanned aircraft collision avoidance system using 5 neural networks ($N_{1,1}$ to $N_{5,1}$), each having 6 hidden layers with 50 ReLU neurons (300 neurons per net). Evaluates open-loop "local robustness" across discretized input ranges (16k, 109k, and 456k discrete input points) and a closed-loop discrete prototype (19,360 physical states).
  - **Does ANY benchmark have a REAL neural network over an input space too large to tabulate?**
    **NO! CATEGORICALLY NO.**
    - Prime Position input space is 10 points.
    - Grid World input space is discrete coordinates ($7 \times 7$ is 2,401 points; $50 \times 50$ is $6.25 \times 10^6$ points), completely enumerated.
    - ACAS Xu inputs are explicitly discretized into integer bins to allow table lookup.
    - The author explicitly confirms this limitation (Section 4, p. 7; Section 5, p. 10):
      > *"The table approach is unaware of the size of the network (it only keeps track of inputs and outputs); this explains why it performs the same in all cases... Additionally, our examples so far have focused on classification networks. This is because our models have been discrete in nature."*

- **(b) Compositional / Modular / Contract-Based Verification:**
  - **YES! This paper is the explicit origin of the Compositional Assume-Guarantee thesis!**
  - While NeuS 2025 itself implements only monolithic tables and fixed-point arithmetic, the author **explicitly proposes the assume-guarantee contract framework as the solution to the table's limits**.
  - **Exact Quotation (Section 5: Conclusions and Future Work, p. 10):**
    > *"One approach is to utilize existing tools for NN verification as nuXmv is not specialized for NNs. Instead of encoding the NSBT with the NNs, we could use NN verification on the NNs, and **create an assume-guarantee compositional verification framework providing only the proven pre and post-conditions over the NNs for the encoding to nuXmv**. This is not unlike the table approach; in the table approach, for a specific input, we record the output. **This can be thought of as a guarantee; if this exact input is provided, the network will output this.**"*

- **(c) Limitations and Future Work:**
  - **Author's Statements on Continuous State Spaces and Reals (Section 5, p. 10):**
    > *"Additionally, our examples so far have focused on classification networks. This is because our models have been discrete in nature. **nuXmv supports the use of reals; however, thus far our attempts to use reals with BehaVerify have yielded very poor performance results.** As such, we are still exploring how to improve our support for regression networks (and reals in general)."*
  - **Author's Statements on Closed-Loop ACAS Xu Simplifications (Section 4, p. 9–10):**
    > *"We note that the closed-loop model for ACAS Xu is a proof of concept. The positions are heavily rounded, aircraft adjust heading instantaneously, and ACAS Xu is called every 6 seconds. Thus the invariant specification, $\rho \ge 200$, is checked every 6 seconds. It is possible that aircraft crash between those 6 seconds without our model noticing. In short, our closed-loop model of ACAS Xu cannot be used to argue for the correctness of ACAS Xu... required aggressive simplification."*
  - **Author's Diagnosis of nuXmv Table Bottlenecks (Section 4, p. 9):**
    > *"Table 4 is surprising; even though this model only has 456775 distinct NN inputs, it performed far worse than the large grid world which has 6250000 distinct NN inputs. We suspect this arises from the 5 NNs, each of which creates a table with 456775 entries. The fact that each of these networks also affects the same variable may create unexpected complexity in nuXmv."*

- **(d) Parallel Nodes, Node Status/Lifecycle, Statefulness/Memory:**
  - Formally defines NSBTs as SBTs where leaf nodes execute neural network inferences reading and writing blackboard variables.
  - Focuses on sequential control flow (`NeedGoal` $\to$ `NewGoal` $\to$ `Act`). Does not evaluate parallel composition of neural controllers.

---

### Paper 5: FM 2026 Submission — *BehaVerify: A Formal Verification Tool for Behavior Trees*
*(Authors: Serena S. Serbinowska, Ben Wooding, Diego Manzanas Lopez, Luke Nam, Anne M. Tumlin, Nicholas K. Potteiger, Hongchao Zhang, Taylor T. Johnson; read from full text of `REPRODUCIBILITY/2026_TBA/reports/FM_2026_Submission.pdf`)*

- **(a) Benchmarks & Neural Network Profile:**
  - **Benchmarks:**
    1. *Binary Tree Benchmark:* Depth $N = 1 \dots 10$ (up to 2,055 nodes) comparing Fastforwarding vs Naive encoding (Table 4).
    2. *Comparison to BT2Fiacre:* Drone3 and DroneNew navigation models (Table 3).
    3. *Comparison to BT2BIP:* MarsRover and TrainControl.
    4. *Comparison to MoVe4BT:* Feature matrix comparison (Table 2).
    5. *Network Example (NSBT):* Drone avoiding obstacles on a small grid to reach random targets, comparing two 2-hidden-layer 64-neuron ReLU networks (1.0 vs 0.9995 accuracy).
  - **Neural Network Profile:**
    - Discrete 4-integer input $(x_d, y_d, x_g, y_g)$ over a small grid (2,401 points) with 4 output directions.
    - **No continuous or un-tabulated neural network.** Encoded monolithically via piecewise-linear constraints in nuXmv.

- **(b) Compositional / Modular / Contract-Based Verification:**
  - **None.** Compositional verification is not included. The paper contrasts open-loop vs closed-loop verification:
    > *"a neural network verifier checks open-loop input–output properties of an isolated network. NSBT verification instead checks temporal properties of the closed loop formed by the tree, the network and a nondeterministic environment. To our knowledge no other BT tool supports neural-network leaves at all."* (Section 4, p. 11)

- **(c) Limitations and Future Work:**
  - **Author's Statements on Parallel Node Execution & Duration (Section 4, p. 11):**
    > *"BehaVerify also does not offer true parallel node execution, which BT2BIP and MoVe4BT do; this is a deliberate alignment with the library we deploy against rather than an implementation gap. BehaVerify also has no native notion of per-node duration."*
  - **Future Work (Section 5, p. 13):**
    > *"Future work includes plans to expand the output options for BehaVerify with other frameworks, such as BehaviorTree.cpp. We also plan to continue developing and improving the tool overall, as we are using it in several ongoing projects for autonomous systems and robotics applications. We will also explore physical instantiations of the self-correcting neuro-symbolic framework derived in [29]."*

- **(d) Parallel Nodes, Node Status/Lifecycle, Statefulness/Memory:**
  - Explains the architectural design choice for parallel nodes: BehaVerify aligns with PyTrees, ticking children sequentially within a tick cycle, avoiding the exponential state explosion of asynchronous interleaving concurrency seen in BIP.
  - Proves the power of Fastforwarding: total state space grows linearly under FF ($2^{13}$ at depth 10) vs exponentially under Naive ($2^{226}$ at depth 7, timing out at depth $\ge 8$).

---

### Paper 6: BehaVerify Tool Manual (`./BehaVerify.pdf`)
*(Author: Serena S. Serbinowska; read from full text of `./BehaVerify.pdf`)*

- **(a) Benchmarks:** GCD (`GCD.tree`), Collatz (`collatz.tree`), Drunken Drone (`DrunkenDrone.tree`), and small ONNX neural network models (`1000__064_1.onnx`, `0995__064_1.onnx`).
- **(b) Compositional Verification:** None (strictly a tutorial and installation manual).
- **(c) Limitations / Future Work:** None stated (non-academic user guide).
- **(d) Parallel Nodes & Memory:** Step-by-step trace walkthrough of return statuses ($S, F, R$), selector/sequence memory, and PyTrees/ROS bindings.

---

### Papers 7 & 8: Paywalled / Abstract-Only Papers (Explicit Disclosure)

#### Ph.D. Dissertation (2025): *On Behavior Trees and Their Verification*
*(Author: Serena Serafina Serbinowska, Vanderbilt University, August 2025)*
- **Reading Status:** **ABSTRACT & METADATA ONLY** (ProQuest paywalled).
- **Abstract Summary:** Synthesizes the development of BehaVerify from its 2022 PyTrees foundation into a formal DSL for Stateful Behavior Trees, contingency monitors, and Neuro-Symbolic Behavior Trees. Chapters correspond to the published SEFM 2022, FMAS 2024 (SBT), FMAS 2024 (BTM), and NeuS 2025 papers.

#### Book Chapter (2026): *Toward Assured Autonomy Using Neurosymbolic Components and Systems*
*(Authors: Abhishek Dubey, Taylor T. Johnson, Xenofon Koutsoukos, ..., Serena Serbinowska, ..., Gabor Karsai; Wiley, 2026)*
- **Reading Status:** **ABSTRACT & METADATA ONLY** (Wiley paywalled, Chapter 4, pp. 89–118).
- **Abstract Summary:** Broad group survey addressing assured autonomy, decomposing autonomy stacks into learned perception and symbolic reasoning, and applying formal verification to neurosymbolic systems.

---

## 5. Strategic Synthesis for the Compositional Project

The independent literature review reveals four critical findings that directly impact our project strategy:

1. **NeuS 2025 Section 5 is the Founding Manifesto of Compositional BehaVerify:**  
   Serena Serbinowska herself identified the exact limit of monolithic BehaVerify and proposed the assume-guarantee contract thesis:
   > *"Instead of encoding the NSBT with the NNs, we could use NN verification on the NNs, and create an assume-guarantee compositional verification framework providing only the proven pre and post-conditions over the NNs for the encoding to nuXmv. This is not unlike the table approach... if this exact input is provided, the network will output this."*

2. **The Table is a Degenerate Point-Contract:**  
   As Claude observed in Claim #106, the lookup table is mathematically equivalent to 1 assume-guarantee contract per discrete state. Therefore, **"beating the table" means covering the same operational envelope with exponentially fewer, coarser contracts via box/polytope over-approximation**.

3. **Resolution of the ACAS Numbers (Crucial Nuance on Claim #106):**  
   - In NeuS 2025 Table 4, the 456,775 entries represent the **open-loop local robustness table** across large continuous intervals.
   - In our closed-loop compositional pipeline (`2026_TBA/examples/AcasXu_closed_loop/`), the total physical state space is **19,360 states**, with each network active on **~2,830 states**.
   - Grouping these by `(heading_own_var, x_sign, y_sign, forbidden_advisory)` yields **490 contracts** (a **~6x reduction** vs per-state tabulation). The 490 contracts cover the closed-loop envelope, not the 456k open-loop points.

4. **The Author Pre-Registered Our "September Wall" with Reals in nuXmv:**  
   Serena Serbinowska documented as early as NeuS 2025 that:
   > *"nuXmv supports the use of reals; however, thus far our attempts to use reals with BehaVerify have yielded very poor performance results."*  
   This proves that the failures we documented in Claims #91 and #92 (where nuXmv IC3 timed out or failed on reals while Z3 succeeded in 1.9s) are not an artifact of our scripts—they are an intrinsic, long-standing bottleneck of nuXmv's real-arithmetic engine that the original tool author also hit and left unsolved.

5. **Not a Single Published BehaVerify Benchmark Features an Un-Tabulated Neural Network:**  
   Across all 6 publications, every single neural network benchmark (Prime Position, Grid World, ACAS Xu, Network Example) has a small, discrete or discretized integer input space. **A benchmark featuring a continuous state space or a neural input space too large to tabulate does not exist in any BehaVerify paper.** Creating such a benchmark is the decisive, open requirement for our compositional verification paper.
