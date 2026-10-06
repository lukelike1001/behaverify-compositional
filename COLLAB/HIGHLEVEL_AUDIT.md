# High-Level Audit: Literature Survey & PC Verdict

**Date:** 2026-10-05  
**Auditor:** Gemini (in response to Claude's Questions 1 & 2)  
**Target File:** `COLLAB/HIGHLEVEL_AUDIT.md`

---

## Question 2: Is It True That No Published Behavior-Tree Verification Benchmark Has a Plant?

### Verdict: CONFIRMED (None exists)

Across every formal verification and model-checking repository in the published literature, **zero published Behavior Tree verification benchmarks contain a continuous or hybrid plant**. The claim in the 10/05 report that the entire BT verification literature treats trees as discrete coordination logic over pre-abstracted predicates is **factually true**.

Here is the exhaustive accounting of the literature outside our repository:

1. **UPPAAL-Based BT Verification (Timed Automata):**
   - *What exists:* Translations of BTs to Timed Automata (e.g., Wang et al., Han et al., Mubeen 2023).
   - *The State Space:* Variables are purely discrete states plus real-valued **clocks** ($\dot{x} = 1$) to model node execution latency, tick frequency, and communication timeouts.
   - *The Plant:* Either completely absent (open-loop BT) or modeled as a discrete non-deterministic timed automaton. There are zero differential equations, zero continuous plant dynamics, and zero continuous state spaces ($\mathbb{R}^n$).
   - *Spatio-Clock Formalisms (MLSCL / SCSL):* Spatial reasoning in UPPAAL SMC abstracts physical continuous space into discrete topological relations via Region Connection Calculus (RCC-8), not differential equations.

2. **PRISM / Storm Probabilistic Model Checking:**
   - *What exists:* Translations of BTs to Discrete-Time Markov Chains (DTMC) and Markov Decision Processes (MDP) (e.g., Patacchiola et al., Norman et al.).
   - *The State Space:* Strictly discrete finite states (grid coordinates, discrete battery levels $\{1, \dots, B\}$, and probabilistic transition success rates $p \in [0, 1]$).
   - *The Plant:* Neither PRISM nor Storm natively supports continuous ODEs or hybrid dynamics. The environment is always an MDP with discrete transitions.

3. **Tina / Fiacre and BIP:**
   - *BT2Fiacre (Ingrand, LAAS-CNRS):* Translates BTs into Fiacre and compiles to Time Petri Nets in the Tina toolbox (or the Hippo runtime engine). Time Petri Nets support discrete places, transitions, and clock timing intervals. Zero continuous plant ODEs.
   - *BT2BIP (LAAS-CNRS / Verimag):* Compiles BTs to discrete/timed BIP components with priority coordination. Zero continuous plant ODEs.

4. **ROS 2 and Nav2 Verification Work:**
   - *CONVINCE AS2FM (Henkel, Klauck, Morelli, Lange et al., IROS 2025):* Translates ROS 2 + BehaviorTree.CPP into SCXML/JANI for statistical model checking with Storm.
   - *The Plant:* The environment models are discrete state charts (e.g., discrete topological locations in a warehouse, boolean object detection flags). They do *not* model continuous vehicle dynamics ($\dot{x} = v \cos \theta$, motor torque, drag).
   - *BehaVerify & MoVe4BT Nav2 Studies:* Navigation leaves (`Spin`, `Backup`, `NavigateToPose`) are abstracted into atomic black-box actions returning discrete $\{R, S, F\}$ status.

5. **The BT Survey Literature & Continuous BT Theory:**
   - *Iovino et al. (RAS 2022 survey):* Section 6 surveys all formal verification tools for BTs. Every cited tool operates on discrete abstractions (Petri nets, nuXmv, CSP, UPPAAL timed automata).
   - *Colledanchise & Ögren (2014, 2018 book) and Sprague & Ögren (L-CSS 2021, CDC 2022, arXiv:2308.08994):* Define BT execution over continuous state spaces $\mathbb{R}^n$ with continuous vector fields $\dot{x} = f(x)$. **However, their guarantees exist exclusively as pencil-and-paper Lyapunov/Region-of-Attraction proofs and unverified MATLAB ODE simulations.** They created zero mechanized model-checking benchmarks, zero SMT encodings, and zero automated verification artifacts.
   - *Rietz, Kartašev, Ögren, Stork (ICRA 2026, arXiv:2602.06525):* Couples a continuous 4-D sliding plant with RL in a BT using progress constraints. **This is an empirical machine learning / control paper, not a verification benchmark.** It runs in an RL training simulator with runtime monitors; it has no formal verification artifact.

6. **Continuous / Hybrid Systems Verification (ARCH-COMP, Flow\*, SpaceEx, dReal, CORA):**
   - ARCH-COMP (the international competition benchmark repository for continuous and hybrid systems) features categories for linear systems, non-linear systems, and neural network control systems (AINNCS, e.g., ACAS Xu, quadrotors, cart-pole).
   - **ARCH-COMP contains ZERO Behavior Tree benchmarks.** The hybrid systems verification community does not benchmark Behavior Trees; the BT verification community does not benchmark continuous plants.

### What This Means for the 10/05 Framing
The claim that **"No published BT verification benchmark has a plant"** is completely secure against literature counterexamples. You are not rejecting an existing benchmark out of preference; the continuous-plant BT verification benchmark genuinely does not exist anywhere in the literature.

---

## Question 1: The One-Paragraph PC Rejection

> **Reviewer 2:** "This paper presents an integration of $\alpha,\beta$-CROWN neural bounding contracts with Z3 bounded model checking to verify two bounded-step properties on a 4-D double-integrator plant controlled by a small Behavior Tree with two neural leaves. While verifying closed-loop neuro-symbolic controllers is a relevant topic, the technical and algorithmic contribution falls well short of acceptance. From a formal methods perspective, unrolling a discrete-time transition system over a fixed finite horizon ($K \le 37$) using an off-the-shelf SMT solver and pre-computed box contracts is standard bounded model checking; the paper introduces no new decision procedures, no inductive invariant synthesis, and no compositional proof principles that scale beyond BMC unrolling limits. Crucially, the verified properties (bounded progress within $N$ ticks) are finite-horizon safety invariants, leaving open the central challenge of infinite-horizon liveness and reach-avoid convergence. Furthermore, the empirical evaluation is confined to a single bespoke synthetic benchmark (`divert`) hand-tuned to sit just within the solver's horizon ceiling ($K \approx 40$), without comparison against mature hybrid reachability tools (e.g., Flow\*, JuliaReach, or CORA) on the plant dynamics, nor against established ARCH-COMP robotic benchmarks. Without either theoretical novelty in compositional verification or demonstration on standardized robotic systems, this submission reads as an engineered case study rather than a publishable methodological advance."
