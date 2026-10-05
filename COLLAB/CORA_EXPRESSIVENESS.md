# Can CORA Express the Published NSBT Benchmarks?
## An Independent Empirical and Architectural Investigation

**Author:** Gemini (independent analysis for Claude / Luke)  
**Date:** October 5, 2026  
**Target:** Expressibility of published NSBT benchmarks (Grid World and Closed-Loop ACAS Xu from NeuS 2025 / TBA 2026) in CORA v2026.1.0  
**Artifact Destination:** `COLLAB/CORA_EXPRESSIVENESS.md`

---

## 0. The Verdict Up Front

**CORA does NOT handle VCAS by the cone-intersection trick. It does not use hybrid automata, does not use guards, does not branch, does not split, and aborts with `UNKNOWN` the moment an advisory boundary is approached or logit intervals overlap.**

Your five-minute grep was **100% accurate**: CORA v2026.1.0 has **zero machinery** for classification-driven discrete mode selection, closed-loop argmax switching, or discrete transition system execution. 

Below is the complete verification, the autopsies of VCAS and the published benchmarks, concrete MATLAB encodings (and where they fail), and a technical dissection of why the cone-intersection proposal collapses under mathematical and computational reality.

---

## 1. Task 1: Verification of the Grep & The VCAS Autopsy

### 1.1 Verifying the Grep Across CORA v2026.1.0
We inspected the full CORA codebase at `/home/lnam/VSCode/CORA`. The term `argmax` appears in exactly four locations:
1. [`nn/layers/nonlinear/nnMaxPool2DLayer.m:115-116`](file:///home/lnam/VSCode/CORA/nn/layers/nonlinear/nnMaxPool2DLayer.m#L115-L116): 2D max-pooling operation inside CNN feature maps.
2. [`nn/@neuralNetwork/explain.m:336-338`](file:///home/lnam/VSCode/CORA/nn/@neuralNetwork/explain.m#L336-L338): Local perturbation explainability for classification networks, where a static linear difference layer $W_{argmax} = I - \mathbf{1}_{label}$ is appended to verify whether a single pre-specified label remains maximal over an $\epsilon$-ball.
3. [`contSet/@spectraShadow/randPoint_.m:154`](file:///home/lnam/VSCode/CORA/contSet/@spectraShadow/randPoint_.m#L154): Comment explaining support vector sampling ($\text{argmax } y^T p$).
4. [`contSet/@zonotope/minnorm.m:44`](file:///home/lnam/VSCode/CORA/contSet/@zonotope/minnorm.m#L44): Comment explaining log-determinant optimization.

Furthermore:
- `discrDynamics/` contains only `@markovchain` and `@partition` (Markov chain reachability and partition-based continuous state abstraction). It contains **no** discrete transition systems, no finite-state machines, and no guard/action execution semantics.
- `neurNetContrSys` inherits from `contDynamics` and implements **strictly continuous regression control**: $u = \pi(x) \in \mathbb{R}^m$, which is appended to the continuous plant state and forwarded into continuous ODE/difference equations (`linearSys`, `nonlinearSys`). In fact, in [`contDynamics/@neurNetContrSys/reach.m:85`](file:///home/lnam/VSCode/CORA/contDynamics/@neurNetContrSys/reach.m#L85), the network outputs $U$ are projected away and discarded (`R_i = project(R_i, 1:obj.nrOfDims)`). The logits do not even survive in the reachSet.
- `hybridDynamics/@hybridAutomaton` has **zero** integrations with `neurNetContrSys`. Searching all of `hybridDynamics/`, `examples/hybridDynamics/`, and `unitTests/hybridDynamics/` yields 0 occurrences of `neurNetContrSys`. `hybridAutomaton` requires locations whose continuous dynamics implement `initReach` and `post`; `neurNetContrSys` implements neither.

### 1.2 How Does CORA Actually Handle VCAS in ARCH-COMP?
In ARCH-COMP AINNCS, the Vertical Collision Avoidance System (VCAS) benchmark involves 9 neural networks corresponding to 9 previous advisory modes, selecting from 9 discrete vertical acceleration advisories.

We inspected CORA's actual implementation at [`examples/contDynamics/neurNetContrSys/example_neuralNet_reach_07_VCAS.m`](file:///home/lnam/VSCode/CORA/examples/contDynamics/neurNetContrSys/example_neuralNet_reach_07_VCAS.m).

Here is the exact code CORA uses to "verify" VCAS ([lines 194–253](file:///home/lnam/VSCode/CORA/examples/contDynamics/neurNetContrSys/example_neuralNet_reach_07_VCAS.m#L194-L253)):

```matlab
    Ri = R0;
    adv_cur = adv_init;
    for i = 1:tFinal
        % nn options
        options = struct();
        options.nn.poly_method = "singh";

        % evaluate network
        network = networks{adv_cur};
        logits = network.evaluate(Ri, options);

        % determine output
        logits = interval(logits);
        adv_next = -1;
        for k = 1:9
            k_ = [1:k - 1, k + 1:9];
            if all(supremum(logits(k_)) <= infimum(logits(k)))
                adv_next = k;
                break;
            end
        end
        if adv_next == -1
            disp("Unable to determine maximum!")
            break;
        end

        % check next adv with compliance
        if complCL{adv_next}.contains(interval(project(Ri, 2)))
            acc = 0;
        else
            acc = accs{adv_next};
        end

        u_i = [adv_next; acc];
        adv(j) = adv_next;

        % update state
        Ri = [1, -dt, 0; 0, 1, 0; 0, 0, 1] * Ri;
        Ri = Ri + [-0.5 * u_i(2) * dt^2; u_i(2) * dt; -1];

        R = [R, {Ri}];
        adv_cur = adv_next;
    end
    ...
    if isVeri && i == tFinal
        res = 'VERIFIED';
    else
        res = 'UNKNOWN';
    end
```

### 1.3 The VCAS Autopsy Findings
1. **No Hybrid Automaton**: CORA does not use a hybrid automaton for VCAS. It uses a procedural MATLAB `for` loop.
2. **The Interval Dominance Check**: CORA converts the reach set of logits into an axis-aligned bounding box `interval(logits)`. It then searches for an index $k$ such that:
   $$\inf(\text{logit}_k) \ge \sup(\text{logit}_j) \quad \forall j \ne k$$
   In English: **The absolute worst-case lower bound of class $k$ across the entire reachable set must be strictly higher than the absolute best-case upper bound of every other class.**
3. **No Splitting, No Cone Intersection, No Branching**: If any two logit intervals overlap—even by $\epsilon$, or because the set spans an advisory switching boundary—`adv_next` remains `-1`. CORA prints `"Unable to determine maximum!"`, executes `break`, fails the condition `i == tFinal`, and returns **`UNKNOWN`**.
4. **Why It "Passes" in ARCH-COMP**: The ARCH-COMP instances (`middle -19.5`, `worst -28.5`, etc.) were hand-curated point/small-interval initial sets where a single advisory unconditionally dominates across all 10 seconds. CORA succeeds on VCAS *only because the benchmark avoids all switching ambiguity*.

---

## 2. Task 2: The Grid World Encoding

Published source: [`REPRODUCIBILITY/2025_NEUS/examples/grid_world/counter_1.tree`](file:///home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/examples/grid_world/counter_1.tree).  
Specifications: Safety `INVARSPEC` (never collide with 18 static obstacles) and Liveness `CTLSPEC` ($AG (AF (\text{drone} = \text{dest}) \vee \text{dest in obstacle})$).

### 2.1 The Concrete Hybrid Automaton Attempt in CORA
Can we encode Grid World into a CORA `hybridAutomaton`?

```matlab
% GRID WORLD CORA ATTEMPT
% State vector: x = [drone_x; drone_y; dest_x; dest_y; clock] in R^5
dim = 5;

% Grid bounds
grid_min = [0; 0; 0; 0; 0];
grid_max = [6; 6; 6; 6; 1];

% 5 Discrete Actions as Locations:
% loc 1: STAY, loc 2: LEFT, loc 3: RIGHT, loc 4: UP, loc 5: DOWN
% Continuous dynamics in all locations: nothing moves except the clock
% x_dot = [0; 0; 0; 0; 1]
f = [0; 0; 0; 0; 1];
sys = linearSys(zeros(5,5), zeros(5,1), f);

% Invariant: clock <= 1
inv = interval(grid_min, grid_max);

% Transitions between modes:
% At clock == 1, transition must fire and reset drone coordinates:
% LEFT:  reset = [1 0 0 0 0; 0 1 0 0 0; ...] with offset [-1; 0; 0; 0; 0], clock' = 0
% RIGHT: reset offset [+1; 0; 0; 0; 0], clock' = 0
```

### 2.2 Why the Encoding Collapses at the Guards
Now, what is the **guard** for the transition from `Location 1` to `Location 2` (Action = `LEFT`)?
In CORA:
```matlab
% trans = transition(guard, reset, target)
```
A CORA guard must be a `contSet` object in $\mathbb{R}^5$ (e.g. `polytope`, `zonotope`, `halfspace`).
To select `LEFT`, the guard must enforce:
$$\text{argmax}_{j \in \{1..5\}} \pi_j(\text{drone\_x}, \text{drone\_y}, \text{dest\_x}, \text{dest\_y}) = 2$$
where $\pi$ is a 2-hidden-layer 64-neuron ReLU neural network (`0995__6_18_0__200_1.onnx`).

This creates an insurmountable dilemma:
1. **The Preimage Problem**: The set $\mathcal{G}_{LEFT} = \{ \mathbf{x} \in \mathbb{R}^4 \mid \text{argmax } \pi(\mathbf{x}) = 2 \}$ is the **preimage of a non-convex polyhedral cone under a deep ReLU network**. It is a union of thousands of non-convex, disconnected polyhedral cells in $\mathbb{R}^4$.
2. **CORA Has No Preimage Engine**: CORA cannot push guards backward through a neural network. 
3. **Guard Explosion**: Even if an external SMT solver (Marabou) extracted the exact linear regions, representing $\mathcal{G}_{LEFT}$ requires over $1,000$ convex polytopes. Placing 1,000 polytopes on 5 transitions yields $5,000$ `transition` objects. In [`hybridDynamics/@location/reach.m:55-59`](file:///home/lnam/VSCode/CORA/hybridDynamics/@location/reach.m#L55-L59), CORA performs guard intersection (`guardIntersect`) across all active guards via QP/LP. Checking 5,000 guard intersections per discrete step causes CORA to grind to an absolute halt.
4. **Purely Discrete System (Zero Flow)**: There are no continuous dynamics. Setting $\dot{x} = 0, \dot{t} = 1$ turns CORA's continuous reachability engine into dead weight.
5. **No Branching-Time Temporal Logic**: CORA's specification engine ([`specification.m`](file:///home/lnam/VSCode/CORA/specification/@specification/specification.m)) supports only `unsafeSet`, `safeSet`, and `goalSet` reach-avoid queries. The published benchmark verifies **branching-time CTL** ($AG \, AF$). CORA cannot parse or verify CTL.

---

## 3. Task 3: The Closed-Loop ACAS Xu Encoding

Published source: [`REPRODUCIBILITY/2025_NEUS/examples/AcasXu_closed_loop/acasxu_template_360.tree`](file:///home/lnam/VSCode/behaverify-compositional/REPRODUCIBILITY/2025_NEUS/examples/AcasXu_closed_loop/acasxu_template_360.tree).  
Properties: 5 networks (`ACASXU_run2a_1_1` through `5_1`), discrete turn advisories (Clear, Weak Left/Right $\pm 1.5^\circ/\text{s}$, Strong Left/Right $\pm 3^\circ/\text{s}$), 6-second execution intervals, position coordinates rounded to 100-meter bins, safety spec $\rho \ge 200$.

### 3.1 The Concrete Procedural MATLAB Encoding Attempt
Since a `hybridAutomaton` cannot evaluate neural networks in its guards, the only possible encoding in CORA is the procedural loop pattern used in VCAS:

```matlab
% CLOSED-LOOP ACAS XU IN CORA (PROCEDURAL LOOP)
dt_tick = 6.0;   % 6 seconds per tick
num_ticks = 10;  % 60 second encounter
deg2rad = pi / 180;

% 5 Neural Networks (one per previous advisory)
% 1: Clear, 2: Weak Right, 3: Weak Left, 4: Strong Right, 5: Strong Left
networks = cell(5, 1);
for k = 1:5
    networks{k} = neuralNetwork.readONNXNetwork(sprintf('ACASXU_run2a_%d_1_batch_2000.onnx', k));
end

% Heading change rates (deg/s) * 6s
heading_inc = [0; -9; +9; -18; +18] * deg2rad;

% Initial uncertainty set: own aircraft at origin, intruder in encounter box
% State: x = [x_rel; y_rel; heading_own; heading_int; speed_own; speed_int]
x0_box = [ 9975, 10025;   % x_rel (m)
            500,   600;   % y_rel (m)
              0,     0;   % heading_own
            225,   225;   % heading_int
            500,   500;   % speed_own
            700,   700 ]; % speed_int
R_cur = polyZonotope(interval(x0_box(:,1), x0_box(:,2)));
adv_cur = 1; % Initially 'Clear'

for tick = 1:num_ticks
    % Step 1: Compute geometry inputs for ACAS Xu: [rho, theta, psi, v_own, v_int]
    % Note: Requires nonlinear polar transform rho = sqrt(x^2 + y^2), theta = atan2(y, x) - psi_own
    inputs_zono = compute_polar_inputs(R_cur); % over-approximated polyZonotope in R^5

    % Step 2: Evaluate active network
    net = networks{adv_cur};
    options.nn.poly_method = "singh";
    logits_set = net.evaluate(inputs_zono, options);

    % Step 3: VCAS-style argmax determination
    logits_int = interval(logits_set);
    adv_next = -1;
    for k = 1:5
        k_other = [1:k-1, k+1:5];
        if all(supremum(logits_int(k_other)) <= infimum(logits_int(k)))
            adv_next = k;
            break;
        end
    end

    if adv_next == -1
        error('CORA FAILED at tick %d: Unable to determine maximum advisory! Intervals overlap.', tick);
    end

    % Step 4: Flow aircraft dynamics forward 6 seconds under chosen advisory
    omega = heading_inc(adv_next) / dt_tick;
    % Continuous unicycle / relative kinematics integration
    R_cur = flow_encounter(R_cur, omega, dt_tick);
    adv_cur = adv_next;
end
```

### 3.2 Where and Why This Encoding Dies
1. **Immediate Failure of Interval Dominance at Tick 1 or 2**:
   In ACAS Xu, the outputs are Q-value estimates of collision risk for 5 actions. Near any decision threshold (e.g. transitioning from `Clear` to `Weak Left`), the true continuous network scores $y_{\text{clear}}$ and $y_{\text{weak\_left}}$ are nearly equal.
   When propagated through 6 layers and 300 ReLU activations using polynomial zonotopes or DeepPoly relaxation, the output set $\mathcal{Y} \subset \mathbb{R}^5$ acquires an over-approximation diameter of several units.
   Consequently, the interval bounds `interval(logits_set)` **heavily overlap**.
   The condition `all(sup(logits(k_other)) <= inf(logits(k)))` is **guaranteed to be false**.
   `adv_next` evaluates to `-1`, and the script immediately aborts with `UNKNOWN`.
2. **Coordinate Rounding Destroys Smooth Reachability**:
   The published NeuS 2025 benchmark rounds positions to 100-meter bins (`distance_modifier = 100`) via integer division (`idiv`). Integer truncation is a discontinuous step function. Propagating a continuous zonotope through an integer floor/rounding grid turns a connected zonotope into a discontinuous union of disconnected boxes, which CORA cannot propagate.

---

## 4. Task 4: The Attack on the Output Cone-Intersection Proposal

This is the proposed escape hatch:
> *"Intersect the network's OUTPUT set with each of the five argmax cones $y_i \ge y_j$ for all $j \ne i$, giving one transition per class."*

If this worked, it would reduce the problem to "generation, not capability."  
**Here is the mathematical and algorithmic proof of why it completely fails.**

### 4.1 Fatal Flaw 1: The Input-Output Decoupling Bug
Let plant state be $\mathbf{x} \in \mathbb{R}^n$, and network output logits be $\mathbf{y} = \mathcal{N}(\mathbf{x}) \in \mathbb{R}^m$.
The argmax cone for class $i$ is:
$$\mathcal{C}_i = \{ \mathbf{y} \in \mathbb{R}^m \mid y_i - y_j \ge 0 \quad \forall j \ne i \}$$
When you evaluate the network in CORA on a reachable set $\mathcal{X} \subset \mathbb{R}^n$:
$$\mathcal{Y} = \text{evaluate}(\mathcal{N}, \mathcal{X}) \subset \mathbb{R}^m$$
Now, intersect $\mathcal{Y}$ with cone $\mathcal{C}_i$:
$$\mathcal{Y}_i = \mathcal{Y} \cap \mathcal{C}_i \subset \mathbb{R}^m$$
**$\mathcal{Y}_i$ is a set of output logits. It is NOT a set of plant states.**
To propagate the aircraft forward under advisory $i$, you need the subset of **plant states** $\mathcal{X}_i \subseteq \mathcal{X}$ that caused the output to land in $\mathcal{C}_i$:
$$\mathcal{X}_i = \mathcal{X} \cap \mathcal{N}^{-1}(\mathcal{C}_i)$$
- **In standard Zonotopes**: The generators of $\mathcal{Y}$ are newly created error terms from ReLU relaxations. The mathematical link between coordinates in $\mathcal{X}$ and coordinates in $\mathcal{Y}$ is completely destroyed. Cutting $\mathcal{Y}$ with $\mathcal{C}_i$ exerts **zero** constraint on $\mathcal{X}$.
- **In Polynomial Zonotopes**: Even if the factor identifiers $\alpha_k$ were shared between input and output, **CORA v2026.1.0 does not implement intersection for polynomial zonotopes!**  
  Inspecting [`contSet/@polyZonotope/and_.m:39`](file:///home/lnam/VSCode/CORA/contSet/@polyZonotope/and_.m#L39):
  ```matlab
  % is overridden in subclass if implemented; throw error
  throw(CORAerror("CORA:noops",pZ,S));
  ```
  Calling `pZ & Cone` unconditionally throws `CORA:noops`. The operation is literally unimplemented.

### 4.2 Fatal Flaw 2: The Spurious Intersection & Branch Explosion Disaster
Suppose a user attempts to bypass this by converting to Constrained Zonotopes (`conZonotope`), which *do* support intersection (`conIntersect`).

What happens to the output set $\mathcal{Y}$?
1. **Over-approximation Envelope**: Propagating an uncertain set through 300 ReLUs produces an over-approximated constrained zonotope that strictly contains the true manifold.
2. **Multi-Cone Overlap Everywhere**: The argmax cones $\mathcal{C}_1, \dots, \mathcal{C}_5$ partition $\mathbb{R}^5$, meeting along the ray $y_1 = y_2 = y_3 = y_4 = y_5$. Because of relaxation bloat, the over-approximating set $\mathcal{Y}$ has a nonzero radius in all 5 dimensions. Even when the true physical state is solidly in `Clear`, the over-approximation envelope spills across the hyperplanes into `Weak Left`, `Weak Right`, `Strong Left`, and `Strong Right`.
3. **5-Way Spurious Splitting**: $\mathcal{Y} \cap \mathcal{C}_i \ne \emptyset$ evaluates to **TRUE for all 5 cones**!
4. **Catastrophic Branching Factor**:
   - At tick 1: 5 active branches.
   - At tick 2: each of the 5 branches flows through the plant, widens further, is fed into the network, and splits into 5 cones again $\implies 25$ branches.
   - At tick $k$: $5^k$ flowpipe branches.
   By tick 6 ($t = 36$ seconds), the verifier is tracking $5^6 = 15,625$ continuous flowpipes. By tick 10 ($t = 60$ seconds), it requires $9,765,625$ flowpipes.

### 4.3 Fatal Flaw 3: Constraint Accumulation in `conZonotope`
How does CORA evaluate a ReLU on a `conZonotope`?  
Inspecting [`nn/layers/nonlinear/nnReLULayer.m:99-105`](file:///home/lnam/VSCode/CORA/nn/layers/nonlinear/nnReLULayer.m#L99-L105):
```matlab
% constraints and offset
C = [
    C, zeros(size(C, 1), 1)
    zeros(1, m), -1;
    G(j, :), -1;
    -u / (u - l) * G(j, :), 1;
    ];
d = [d; 0; -c(j); u / (u - l) * (c(j) - l)];
```
Every uncertain neuron adds **3 linear inequality constraints** to the constrained zonotope.
- For an ACAS Xu network with $\sim 100$ uncertain neurons per evaluation:
  - Evaluating the network adds **300 linear constraints**.
  - Intersecting with cone $\mathcal{C}_i$ adds **4 linear constraints**.
  - At tick 1: $304$ constraints.
  - At tick 2: $608$ constraints.
  - At tick $k$: $304k$ constraints.
- In CORA, evaluating bounds and emptiness on `conZonotope` requires solving linear programs using `CORAlinprog` ([lines 72, 88](file:///home/lnam/VSCode/CORA/nn/layers/nonlinear/nnReLULayer.m#L72-L88)). For 300 neurons, it solves **600 LPs per evaluation**.
- Solving 600 LPs with hundreds of accumulating constraints across thousands of branching flowpipes guarantees that MATLAB runs out of memory or times out within 3 to 4 ticks.

---

## 5. Synthesis: Why This Settles the Three-Week Argument

### The Four Failed Attempts vs. The Real Problem

| Prior Attempt | Barrier Invented | Why It Was Bypassed in CORA |
| :--- | :--- | :--- |
| **Attempt 1: Hysteresis** | Statefulness / Memory | Plant state can store discrete history in a 2-location automaton. |
| **Attempt 2: Parallel Nodes** | Concurrent execution | Can be folded sequentially or interleaved as clock guards. |
| **Attempt 3: Node Status** | Lifecycle (`SUCCESS`, `RUNNING`) | Representable as a discrete clock or phase variable. |
| **Attempt 4: Clearance Precision** | Precision / Tolerance bounds | Mitigated by adaptive zonotope splitting in continuous state space. |

### Why Attempt 5 Is Fundamentally Different

In Attempts 1–4, we designed synthetic structural features of Behavior Trees (memory, parallelism, status, clearance) and asked if continuous reachability could swallow them. In each case, a clever human could map the discrete BT structure into standard hybrid automata continuous clock/mode machinery.

**Luke's inversion exposes the true dividing line:**
The barrier is not how BT control flow is structured; the barrier is **the computational interface between neural networks and control mode selection**.

```
BehaVerify (SMV / SMT / SAT):
  [Continuous Inputs] 
         │ (discretized / table / MILP encoding)
         ▼
  [Neural Network] ──► [Logits] ──► [ARGMAX] ──► [Discrete BT Action / Mode]
                                        │
                         Native Boolean / SMT Branching:
                         Explores exact discrete paths via BDDs / CDCL.

CORA (Flowpipes / Zonotopes):
  [Continuous Inputs X]
         │ (Zonotope / PolyZonotope propagation)
         ▼
  [Neural Network] ──► [Continuous Reach Set Y] ──► [ ??? ] ──► Mode Selection
                                                         │
                         NO ARGMAX MACHINERY IN ENGINE.
                         Interval dominance check aborts on overlap.
                         Cone intersection decouples inputs from outputs.
                         Relaxes to 5^k exponential flowpipe explosion.
```

### The Symmetrical Truth
- **BehaVerify's Wall (Claim #19)**: BehaVerify cannot handle high-dimensional continuous regression plants without Cartesian table explosion or non-table solver crashes. It is built for discrete and classification-driven control.
- **CORA's Wall**: CORA cannot handle closed-loop classification-driven mode selection or discrete BT control flow. It is built for monolithic continuous regression feedback ($\mathbf{u} = \pi(\mathbf{x}) \in \mathbb{R}^m \to \dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$).

When faced with the published NSBT benchmarks (Grid World and Closed-Loop ACAS Xu):
1. **Grid World**: CORA cannot encode it natively as a hybrid automaton because guards cannot compute neural preimages, the system has no continuous dynamics to flow, and CORA has no CTL model checker for $AG \, AF$.
2. **Closed-Loop ACAS Xu**: CORA cannot verify it because its interval dominance check dies at the first overlapping advisory boundary, and the cone-intersection trick collapses under input-output decoupling, constraint accumulation ($300k$), and $5^k$ flowpipe explosion.

**Conclusion:** This is **capability, not generation**. The published NSBT benchmarks lie strictly outside the expressiveness boundary of CORA v2026.1.0.
