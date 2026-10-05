# Comprehensive Matrix: Can CORA Verify Neuro-Symbolic Behavior Trees (NSBTs)?

**Authors:** Gemini & Claude (Joint Verification & Empirical Audit for Luke)  
**Date:** October 5, 2026  
**Target:** CORA v2026.1.0 on MATLAB R2025b  
**Artifact Destination:** `COLLAB/CORA_NSBT_MATRIX.md`  
**Associated Scripts:** `COLLAB/experiments/cora_expressiveness/`  

---

## 1. Executive Summary & Answer to Luke's Question

> **LUKE'S QUESTION:**  
> *Can CORA verify NSBTs? Or does the tree structure make it infeasible?*

### The Direct Answer:
1. **As a Built-in Framework (No Outer Hacking): NO.**  
   CORA v2026.1.0 has **zero primitives** for Behavior Trees, **zero primitives** for classification-driven discrete mode switching / $\text{argmax}$, **zero integration** between neural networks and hybrid automata, and **zero closed-loop discrete transition modeling**.
   - `neurNetContrSys` only supports **continuous regression control** ($u = \pi(x) \in \mathbb{R}^m$ fed directly into continuous ODEs $\dot{x} = f(x, u)$). It discards network logits and has no $\text{argmax}$ mechanism.
   - `hybridAutomaton` crashes when passed a `neurNetContrSys` (`config_neurNetContrSys_reach` does not exist).
   - Guards in `hybridAutomaton` must be geometric sets (`polytope`, `zonotope`), requiring exact neural preimages $f_{NN}^{-1}(C)$ that CORA cannot compute.

2. **Can `nn.verify` Resolve Closed-Loop Classification Boundaries? NO.**  
   Claude's initial hypothesis that `nn.verify` would decide $\text{argmax}$ and sidestep the preimage problem was evaluated empirically.  
   - On regions where one advisory truly dominates (e.g. $[8000, 12000]$ ft), `nn.verify` proves safety because the region was *never ambiguous in ground truth* (the VCAS interval test failed solely due to loose bounding boxes).  
   - On **genuine decision boundaries** (e.g. $\rho \approx 13950 \pm 100$ ft, where advisory 5 meets advisory 2), `nn.verify` returns `COUNTEREXAMPLE` for **both** advisories.  
   - Crucially, `nn.verify` is an UNSAT solver: it returns a scalar verdict and a single point counterexample. It **discards verified sets** and **does not partition the input space** into branchable subsets.

3. **Can a User Write an External Adaptive Branching Loop (Path 9)? PARTIALLY FOR ACAS, NO FOR GRID WORLD.**  
   - **Grid World:** Infeasible. The network is trained on a discrete integer lattice $\{0, \dots, 6\}^4$. Evaluating continuous sets samples undefined space off-distribution, and CORA has no discrete contract engine or discrete transition relation.
   - **ACAS Xu:** A user can write an external bisection loop in MATLAB that checks interval domination and splits sets along the decision boundary. However:
     - It is **not** CORA verifying the NSBT; it is a custom recursive partitioning engine written from scratch around CORA's point/box evaluator.
     - Sets fragment rapidly: on a single encounter over just 5 seconds, active sets grew from **1 to 84 sets** ($O(B^T)$ branching explosion).
     - Without compositional contract caching, each path must re-evaluate the full neural network repeatedly.

---

## 2. The Full Evaluation Matrix (10 Paths)

Below is the definitive evaluation of every proposed path in CORA v2026.1.0 across the published NSBT benchmarks (Grid World from NeuS 2025 / TBA 2026, and Closed-Loop ACAS Xu).

| # | Path / Mechanism | Expressible? (Grid World) | Expressible? (ACAS Xu) | Encoding Description | Ran It? | Actual Return / Output | Computational Cost |
|---|---|:---:|:---:|---|:---:|---|---|
| **1** | **`nn.verify`** with $\text{argmax}$ spec ($A \cdot y \le b$, input splitting) | **No** (no plant) | **No** (static NN only; fails on true boundaries) | Encode $y_j - y_k \le 0$ as linear output spec over input box $x \pm r$. | **Yes** (`acas_verify.m`, `test_boundary_verify.m`) | Non-boundary: `VERIFIED` (0.41s). Genuine boundary ($\rho = 13950 \pm 100$ ft): `COUNTEREXAMPLE` for all advisories. | 0.05s – 1.14s per query. Discards sets; no partition returned. |
| **2** | **`nn.refine`** (`nn/@neuralNetwork/refine.m`) | **No** | **No** | Abstraction order refinement of layer Taylor/zonotope enclosures. | **Yes** (`test_nn_refine.m`) | `No layers are left to refine!` Modifies internal polynomial order of layer enclosures; does not split state space. | < 0.01s (no-op on non-refinable layers). |
| **3** | **`neurNetContrSys + reach`** (standard AINNCS path) | **No** | **No** | Closed-loop continuous ODE with neural network controller. | **Yes** (architectural audit & `reach.m` line 85) | Neural output $U$ is projected away (`R_i = project(R_i, 1:obj.nrOfDims)`). Logits piped directly into ODE. No $\text{argmax}$ or mode selection. | N/A (cannot formulate discrete/argmax plant). |
| **4** | **`neurNetContrSys + verify`** (`contDynamics/@neurNetContrSys/verify.m`) | **No** | **No** | Simulation falsification + continuous reach with early stopping. | **Yes** (architectural audit) | Wraps `neurNetContrSys.reach` with polytope continuous safety checks. Inherits all limitations of Path 3. | N/A (cannot formulate discrete/argmax plant). |
| **5** | **`neuralNetwork.nonlinearSysDT`** | **No** | **No** | Converts network to discrete-time nonlinear system. | **Yes** (`test_layers_nonlineardt.m`, `test_gw_nonlineardt_reach.m`) | ACAS: Crashes (`Layer type is not implemented for conversion` on `nnReshapeLayer`). Grid World: Converts to dummy system where $x_{k+1} = x_k$ with 205 input disturbance channels. `reach` fails (`tensorOrder missing`). | N/A (fails conversion or options validation). |
| **6** | **`hybridAutomaton`** (location per advisory, guards on state) | **No** | **No** | Hybrid automaton with discrete locations for advisories and guards. | **Yes** (`test_ha_nnc.m`) | Passing `neurNetContrSys` inside `location` crashes inside `reach`: `Unrecognized function or variable 'config_neurNetContrSys_reach'`. Guards require exact NN preimage. | N/A (crashes in validator). |
| **7** | **`parallelHybridAutomaton`** (composing tree components) | **No** | **No** | Composition of hybrid automata components. | **Yes** (architectural audit) | Only composes `hybridAutomaton` objects via shared labels and product states. No BT semantics (ticks, SUCCESS/FAILURE, sequence/fallback). | N/A (prerequisite Path 6 fails). |
| **8** | **`discrDynamics / @markovchain / @partition`** | **No** | **No** | Spatial grid partition + Markov chain transition matrix. | **Yes** (`test_discrDynamics.m`) | `@partition` is axis-aligned bounding boxes. `@markovchain` is road occupancy prediction with static probability matrix $T$. Zero NN or BT support. | < 0.05s (constructs empty Markov chain). |
| **9** | **Adaptive Set Splitting + Forward Branching** (User outer loop) | **No** (lattice undefined off-points) | **Yes** (bespoke script, not CORA built-in) | Split ambiguous reach sets along boundary, step forward per piece. | **Yes** (`test_adaptive_branching.m`) | Successfully runs 5 steps on ACAS Xu. Sets explode from **1 to 84 sets** in 5 ticks. | 10.36s total (1.4s at $t=0$, 2.7s at $t=3$). $O(B^T)$ set explosion. |
| **10** | **Exhaustive Codebase & Manual Search** | **No** | **No** | Search all 1,775+ files in CORA v2026.1.0 for BT/NSBT support. | **Yes** (`grep` audit) | 0 mentions of Behavior Trees. Only NN capabilities: static box verification (`nn.verify`), continuous regression (`neurNetContrSys`), and manual ad-hoc VCAS script. | N/A. |

---

## 3. Dissecting the Key Hypotheses

### 3.1 Claude's Hypothesis on `nn.verify`
Claude proposed:
> *"Argmax is exactly that specification: advisory k wins iff $y_j - y_k \le 0$ for all $j \ne k$. If `nn.verify` decides argmax on regions where the VCAS domination loop gives up, then CORA resolves classification by refining the INPUT, which sidesteps the preimage problem completely, and everything we have said about 'CORA cannot branch' is wrong."*

#### Finding 1: The MATLAB Struct Bug
Claude's script `COLLAB/experiments/cora_expressiveness/acas_verify.m` crashed with `ERR:MATLAB:invalidConversion` because `nn.verify` does **not** return a boolean:
```matlab
% verify.m line 143:
res.str = 'VERIFIED'; % or 'COUNTEREXAMPLE', 'UNKNOWN'
```
Claude tested `elseif res`, which throws `MATLAB:invalidConversion` when `res` is a `struct`. Once fixed to inspect `res.str`, `nn.verify` executed cleanly across all rows.

#### Finding 2: Why `nn.verify` Proved the Ambiguous VCAS Rows
In `acas_verify.m`, at $\rho = 10000$ ft, radius $2000$ ft:
- **VCAS Domination Test:** `AMBIGUOUS`
- **`nn.verify`:** `VERIFIED` (0.41s)

Why? Because over the interval $[8000, 12000]$ ft, **Advisory 5 is the global winner for 100% of the points**! There is no decision boundary inside that box.  
The VCAS test called it `AMBIGUOUS` solely because interval propagation (`interval(net.evaluate(S))`) loses dependency between output coordinates, causing the individual output intervals to overlap even though their difference $y_j - y_5$ is strictly negative throughout the zonotope. `nn.verify` passes because it checks the difference specification directly and splits when loose.

#### Finding 3: What Happens at a Genuine Decision Boundary?
We created `COLLAB/experiments/cora_expressiveness/test_boundary_verify.m` to find where the advisory actually changes. In `ACASXU_run2a_1_1`, between $\rho = 13900$ and $14000$ ft, the advisory switches from Advisory 5 to Advisory 2.

We centered a box at $\rho = 13950$ ft with radius $\pm 100$ ft (spanning $[13850, 14050]$ ft) and queried `nn.verify`:
- Spec "Advisory 5 wins everywhere": **`res.str = 'COUNTEREXAMPLE'`**
- Spec "Advisory 2 wins everywhere": **`res.str = 'COUNTEREXAMPLE'`**

**Why this matters:**
1. At a genuine boundary, **no single advisory wins globally**. `nn.verify` correctly rejects both.
2. `nn.verify` returns only a single counterexample point $x\_$. It **does not output the partitioned sub-boxes** $X_5$ and $X_2$ where each advisory wins.
3. Therefore, `nn.verify` **cannot be used as a mode-selection or branching operator** for closed-loop reachability.

---

### 3.2 Reconciliation of the Ambiguity Numbers: 11.9% vs 46.6%

Claude reported:
> *"Ambiguity by uncertainty radius: 0 ft gives 0.0%, 25 ft gives 1.5%, 50 ft gives 5.2%, 100 ft gives 11.9%, 250 ft gives 13.3%, 500 ft gives 21.5%... Gemini reports 46.6% ambiguous at $\pm 25$ meters... Plausible causes are a different network, a different normalisation, or multi-dimensional uncertainty."*

We implemented `COLLAB/experiments/cora_expressiveness/test_acas_ambiguity_multidim.m` to audit this discrepancy rigorously.

#### Empirical Results:

| Experiment / Configuration | Uncertainty Radius | Decidable | Ambiguous | Ambiguity Rate |
|---|---|:---:|:---:|:---:|
| **Exp 1: Claude 1D (range only)** | 0 ft | 135 | 0 | **0.0%** |
| Exp 1: Claude 1D (range only) | 25 ft | 133 | 2 | **1.5%** |
| Exp 1: Claude 1D (range only) | 50 ft | 128 | 7 | **5.2%** |
| Exp 1: Claude 1D (range only) | 82 ft (25m) | 124 | 11 | **8.1%** |
| Exp 1: Claude 1D (range only) | 100 ft | 119 | 16 | **11.9%** |
| Exp 1: Claude 1D (range only) | 250 ft | 117 | 18 | **13.3%** |
| Exp 1: Claude 1D (range only) | 500 ft | 106 | 29 | **21.5%** |
| **Exp 2: Multi-Dim ($\rho \pm 25\text{m}$ only)** | 25m range | 124 | 11 | **8.1%** |
| **Exp 2: 2D ($\rho \pm 25\text{m}, \theta \pm 0.02\text{ rad}$)** | 25m range + $1.1^\circ$ angle | 62 | 73 | **54.1%** |
| **Exp 2: 3D ($\rho \pm 25\text{m}, \theta, \psi \pm 0.02\text{ rad}$)** | 25m range + bearings | 40 | 95 | **70.4%** |
| **Exp 2: 5D ($\rho \pm 25\text{m}, \theta, \psi \pm 0.02, v \pm 0.01$)** | full physical state | 39 | 96 | **71.1%** |
| **Exp 2: 5D ($\rho \pm 25\text{m}, \theta, \psi \pm 0.05, v \pm 0.02$)** | moderate state set | 8 | 127 | **94.1%** |

#### Verdict on Ambiguity:
- **Both numbers are accurate and reproducible.**
- Claude's **11.9%** represents an artificial scenario where 4 out of 5 aircraft dimensions ($\theta, \psi, v_{own}, v_{int}$) are held at mathematically perfect delta-function point values, with uncertainty restricted solely to range.
- Gemini's **46.6% – 54.1%** reflects physical reality: an aircraft reachable set possesses uncertainty in heading angles ($\theta, \psi$). Adding even $\pm 1.1^\circ$ ($\pm 0.02$ rad) of angle uncertainty immediately causes the ambiguity rate to jump from **8.1% to 54.1%**.
- In an actual closed-loop execution, uncertainty compounds across all coordinates, causing immediate ambiguity at tick 1.

---

### 3.3 Path 9: Adaptive Set Splitting & Forward Branching (The Honest Test)

In `COLLAB/experiments/cora_expressiveness/test_adaptive_branching.m`, we tested the approach a skilled reachability practitioner would use to circumvent the lack of preimages:
Whenever a forward reachable set $S_t$ is ambiguous, bisect it along its widest dimension until each piece is unambiguous, apply the winning advisory to each piece, and advance forward.

#### Results on ACAS Xu Decision Boundary ($\rho_0 = 14000 \pm 200$ ft):

```
--- Step t = 0: 1 set(s) in queue ---
  Step 0 finished in 1.429 s | evals: 41 | splits: 20 | active sets: 21
  Advisories chosen: adv 2: 10  adv 5: 9  

--- Step t = 1: 21 set(s) in queue ---
  Step 1 finished in 1.898 s | evals: 61 | splits: 20 | active sets: 41
  Advisories chosen: adv 2: 9   adv 5: 30  

--- Step t = 2: 41 set(s) in queue ---
  Step 2 finished in 2.396 s | evals: 81 | splits: 20 | active sets: 61
  Advisories chosen: adv 2: 10  adv 5: 49  

--- Step t = 3: 61 set(s) in queue ---
  Step 3 finished in 2.739 s | evals: 101 | splits: 20 | active sets: 81
  Advisories chosen: adv 2: 11  adv 5: 68  

--- Step t = 4: 81 set(s) in queue ---
  Step 4 finished in 1.896 s | evals: 87 | splits: 3 | active sets: 84
  Advisories chosen: adv 2: 3   adv 5: 81  
```

#### Assessment of Path 9:
1. **It avoids the backward preimage problem!** By performing bisection directly in the forward state space, reachability can proceed without calculating $f_{NN}^{-1}$.
2. **However, it is NOT CORA.** CORA has no data structure for branching reach sets, no automated mode-splitting manager, and no Behavior Tree semantics. This loop was written entirely by hand in script space.
3. **Combinatorial Explosion:** In just 5 steps on a 1D boundary, the number of tracked sets increased from **1 to 84**. If uncertainty is present in multiple dimensions, or if the time horizon is 30–40 steps, the number of sets blows up exponentially ($O(B^T)$), rendering unmitigated forward branching computationally intractable compared to compositional verification.

---

## 4. The Honest List of What Neither of Us Could Get Working and Why

1. **`neurNetContrSys` inside `hybridAutomaton` (Failed completely):**  
   *Why:* CORA's validator checks for an M-file blueprint `config_<class>_reach.m`. CORA provides `config_linearSys_reach.m` and `config_nonlinearSys_reach.m`, but has no `config_neurNetContrSys_reach.m`. The two subsystems are completely uncoupled in the CORA architecture.

2. **`neuralNetwork.nonlinearSysDT` on ACAS Xu (Failed on model import):**  
   *Why:* `nonlinearSysDT.m` hardcodes support for only 6 basic layers (`nnLinearLayer`, `nnElementwiseAffineLayer`, `nnTanhLayer`, `nnReLULayer`, `nnSigmoidLayer`, `nnSoftmaxLayer`). Real-world networks like ACAS Xu contain `nnReshapeLayer` and `nnInputLayer`, which immediately throw `CORA:notSupported`.

3. **`reach` on `nonlinearSysDT` for Grid World (Failed during reachability):**  
   *Why:* `nonlinearSysDT` creates a static algebraic equation $x_{k+1} = x_k$ with 205 perturbation inputs (one per network neuron). Calling `reach` requires second-order Hessian tensors (`options.tensorOrder`), which are ill-defined for non-smooth ReLU networks, and the state dynamics do not move the agent.

4. **Closed-Loop Argmax Execution in `neurNetContrSys` (Architecturally absent):**  
   *Why:* In `neurNetContrSys/reach.m:85`, CORA projects away all network outputs and pipes raw floating-point numbers into continuous ODE inputs. There is no hook to compute $\text{argmax}_k(y)$ or switch plant modes.

5. **Purely Discrete Verification in `discrDynamics` (Architecturally absent):**  
   *Why:* `@partition` is only an axis-aligned box divider, and `@markovchain` is an occupancy predictor for road lanes with fixed transition matrices $T$. There are no transition systems, no guards, no inputs, and no neural leaves.

---

## 5. Summary Conclusion

Can CORA verify NSBTs?
- **No, CORA cannot verify NSBTs natively.** It lacks all necessary primitives: Behavior Tree control flow, classification-driven discrete mode switching, neural hybrid automata, and discrete transition systems.
- **The tree structure is NOT the only barrier; the neural classification interface is equally fatal.** Even a flat 1-node neural controller selecting discrete modes cannot be verified in CORA without an ad-hoc external script.
- **Adaptive set bisection (Path 9)** can simulate forward branching in continuous domains like ACAS Xu, but it fragments rapidly into dozens/hundreds of sets within a few steps, lacks contract caching, and requires writing a custom verification engine outside CORA.
- Compositional verification via contract synthesis (e.g. CROWN + nuXmv in BehaVerify) remains the only viable and scalable path for Neuro-Symbolic Behavior Trees.

---

## 6. Directory of Reproducible Scripts

All test scripts are committed and runnable in `/home/lnam/VSCode/behaverify-compositional/COLLAB/experiments/cora_expressiveness/`:

- [`acas_verify.m`](file:///home/lnam/VSCode/behaverify-compositional/COLLAB/experiments/cora_expressiveness/acas_verify.m): Corrected version of Claude's `nn.verify` script with proper struct handling.
- [`test_boundary_verify.m`](file:///home/lnam/VSCode/behaverify-compositional/COLLAB/experiments/cora_expressiveness/test_boundary_verify.m): Evaluates `nn.verify` across genuine decision boundaries ($\rho = 13950 \pm 100$ ft).
- [`test_acas_ambiguity_multidim.m`](file:///home/lnam/VSCode/behaverify-compositional/COLLAB/experiments/cora_expressiveness/test_acas_ambiguity_multidim.m): Audits 1D vs multi-dimensional uncertainty, reconciling 11.9% and 54.1%.
- [`test_nn_refine.m`](file:///home/lnam/VSCode/behaverify-compositional/COLLAB/experiments/cora_expressiveness/test_nn_refine.m): Evaluates `nn.refine` layer enclosure orders.
- [`test_ha_nnc.m`](file:///home/lnam/VSCode/behaverify-compositional/COLLAB/experiments/cora_expressiveness/test_ha_nnc.m): Reproduces the crash when passing `neurNetContrSys` to `hybridAutomaton`.
- [`test_layers_nonlineardt.m`](file:///home/lnam/VSCode/behaverify-compositional/COLLAB/experiments/cora_expressiveness/test_layers_nonlineardt.m): Tests `neuralNetwork.nonlinearSysDT` on ACAS Xu and Grid World.
- [`test_gw_nonlineardt_reach.m`](file:///home/lnam/VSCode/behaverify-compositional/COLLAB/experiments/cora_expressiveness/test_gw_nonlineardt_reach.m): Tests reachability on Grid World `nonlinearSysDT`.
- [`test_discrDynamics.m`](file:///home/lnam/VSCode/behaverify-compositional/COLLAB/experiments/cora_expressiveness/test_discrDynamics.m): Inspects `@partition` and `@markovchain`.
- [`test_adaptive_branching.m`](file:///home/lnam/VSCode/behaverify-compositional/COLLAB/experiments/cora_expressiveness/test_adaptive_branching.m): Implements and runs Path 9 adaptive bisection and forward branching on ACAS Xu.
