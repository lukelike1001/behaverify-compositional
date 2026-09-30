# Compositional BehaVerify: Why Should We Care?

**Date:** 2026-09-30

**Scope:** Is compositional BehaVerify, despite all these dead ends, still
valuable? Even reporting a negative result we didn't want is acceptable. It's
research, after all. However, surely there must be something enticing about
chasing a positive result, even as hard as it may be to find it.

**Audience:** a first-year CS undergraduate. Formal-methods terms are used, but
each is explained where it first appears. If a paragraph needs a PhD to parse,
it is a bug in the writing, not a feature.

---

## 1. The problem, in one picture

A delivery drone is flying to a waypoint. Two different pieces of software are
in charge of it at the same time:

- A **behavior tree** decides *what to do*: "if the battery is critical, land;
  otherwise fly to the next waypoint." A human wrote this. You can read it,
  point at a box, and say "that's the rule that fired."
- A **neural network** decides *how to do it*: given the drone's position and
  velocity, how hard to push the throttle. Nobody wrote these rules. The network
  learned them from data, and its "logic" is a few thousand numbers.

This is not a hypothetical architecture. It is how a large fraction of modern
robots are actually built, because each half is good at something the other is
bad at. Trees are readable and modular; networks are good at continuous control
that is painful to write by hand.

**The question this project asks: can we prove such a system is safe?**

---

## 2. Why "just test it" isn't enough

You could fly the drone a thousand times and see if it crashes. That is testing,
and it is genuinely useful. But testing samples; it cannot cover.

The drone's position is a *real number*. Between x = 3.0 and x = 3.1 there are
infinitely many positions. A thousand flights check a thousand points out of
infinitely many. Formal **verification** is the alternative: instead of sampling
behaviours, you prove a statement about *all* of them at once, using a
mathematical model of the system.

The rest of this report is about why that is hard when a neural network is in
the loop, and what we tried.

---

## 3. Why α,β-CROWN alone is not enough

α,β-CROWN is a **neural network verifier**, and a very good one. Give it a
network and a box of inputs, and it proves a statement about the outputs:

> "For every input in this box, output 1 is greater than output 2."

That is an **open-loop** property — a fact about the network *considered by
itself*, frozen, evaluated once.

But our drone does not evaluate the network once. It evaluates it, moves,
evaluates it again at the new position, moves again — hundreds of times. The
property we care about is **closed-loop**:

> "Starting anywhere in the takeoff zone, the drone never enters the obstacle,
> no matter how the 300 decisions interleave."

α,β-CROWN cannot state that, because it has no notion of "and then the drone
moved, and then we asked again." It does not know there *is* a drone. It also
has no notion of the behavior tree — it cannot express "when the battery check
fails, the landing branch takes over."

**So the NN verifier is necessary and not sufficient.** It answers one-step
questions perfectly and many-step questions not at all.

---

## 4. Why not just use Z3 (or any SMT solver)?

Reasonable idea. Z3 is an **SMT solver** — think of it as a very powerful
equation solver that can also handle logic: "does there exist an x such that
these hundred constraints all hold?" You could write down the drone, the tree,
*and* the network as one enormous constraint system and ask Z3 whether a crash
is possible.

People do this. It works on small problems. It falls over on real ones, for a
specific reason: **ReLU**.

A neural network is mostly made of ReLU units, each computing `max(0, x)`. That
is a *case split*: either x ≤ 0 and the output is 0, or x > 0 and the output is
x. Each ReLU doubles the number of cases the solver may have to consider. A
network with 100 ReLUs has up to 2¹⁰⁰ cases — more than atoms in the observable
universe — and you get a fresh copy of that for every time step.

Solvers are cleverer than brute force, but the blowup is real, and we measured
it on this project: symbolic neural verification inside nuXmv **failed at three
hidden neurons** with generic weights (claim #37). Three.

The useful mental model: the NN verifier is good at networks and blind to time;
the SMT solver is good at time and logic and drowns in networks.

---

## 5. Why not just discretize everything?

This is the strongest objection, and for a long time the right answer.

**Discretizing** means replacing the continuous world with a grid. Instead of
"the drone is at x = 3.047," you say "the drone is in square 3." Now the state
space is *finite*, and finite means you can check **every** case exhaustively.
That is not a hack — it is what makes the problem decidable at all.

It also works. BehaVerify's discrete pipeline verifies a 7×7 grid world with a
neural controller in under a second, including a liveness property ("the drone
eventually reaches its goal"). Monolithic BehaVerify handles a 4,095-node tree
in 203 seconds, while real trees are 11–41 nodes. For grid-like domains —
warehouse robots on aisles, board-game agents — discretization is simply the
right model.

So why care about continuous state at all?

**Because of a result from this very project, on 2026-03-31.** The grid world
networks were **100% accurate on every integer point**. Perfect on the grid.
Then we ran PGD — an adversarial attack that searches the *continuous* space
between grid points — and it found counterexamples: positions where the network
picks an unsafe action. The networks were **safe on the lattice and unsafe
between the squares**.

A real drone does not teleport from square 3 to square 4. It passes through the
space in between. The grid model could not see the bug, because the bug lived
where the model had no squares.

> **Discretization proves things about a robot that does not exist.** Usually
> that robot is close enough to the real one. Sometimes — exactly when the
> network was trained on grid points and deployed on real ones — it is not.

That is the honest case for continuous state. Not that discretization is wrong,
but that it has a specific, demonstrated blind spot.

---

## 6. What "compositional" means, and why we tried it

The monolithic approach compiles the whole network into the model checker: one
giant lookup table listing what the network outputs for every possible input.
Exhaustive and exact — and it only terminates because the input space is finite.
On a continuous domain there is no table to build.

The **compositional** approach splits the job in two, and this is the whole idea
of the project:

1. **α,β-CROWN proves small local facts** about the network. "Whenever the drone
   is in this box of positions, the throttle command is between −0.99 and −0.98."
   These are **assume-guarantee contracts**: a promise about outputs, given an
   assumption about inputs.
2. **The model checker reasons about time**, treating the network as a black box
   that obeys those contracts. It never sees a ReLU.

Each tool does what it is good at. The contracts are the interface.

**Why this is sound:** the contracts *over-approximate* the network — they allow
at least everything the real network does, and possibly more. So if the model
checker proves the abstract system is safe, the real one is too. If it finds a
crash, that crash might be *spurious* (an artefact of the extra freedom), which
is why the counterexamples have to be checked against the real system.

---

## 7. What we found

Honest summary, positive and negative.

**Where it works — the discrete benchmarks:**

| benchmark | result |
|---|---|
| ACAS Xu closed loop | **8.6 s vs 49.3 s** monolithic, ~11× less memory, **one contract** instead of 8,982 |
| Grid world | safety **and** liveness proved in **0.33 s** |

The ACAS number deserves a sentence of explanation, because it is the best
result the project has. Injecting all 1,659 verified contracts expanded to 8,982
`INVAR` lines and **crashed** nuXmv. The fix was not a bigger machine; it was
realising that safety only depends on contracts at the *boundary* of the set of
states from which a crash is still avoidable. That boundary met the reachable
states at exactly **one** point. One contract, correctly chosen, did the work of
8,982.

**Where it fails — continuous state (NAV):**

We ported the architecture to NAV, a continuous benchmark, and it stalls at
**8 of 30 time steps**. Worse, a much simpler method beat it: plain per-step
interval reachability reaches **step 10 using 10 contracts and no solver at
all**, against our 8 steps with ~1,300 contracts and Z3.

**And we found out why**, which is the part worth keeping. Contracts are written
per grid cell. Once the abstraction's uncertainty grows past one cell, the
control it must allow is the **union** of every spanned cell's contract. Making
the grid finer gives proportionally *more, smaller* cells covering the same
uncertainty — so the union is just as wide. We measured it: **4× refinement
(64× the cells) buys 10%.** Closing the gap would need roughly a 40,000× finer
grid.

That is a feedback loop, not a budget problem, and it is why every refinement
strategy we tried — finer grids, anisotropic grids, non-uniform grids,
demand-driven refinement, CEGAR splitting — failed the same way.

---

## 8. So why still care?

Three reasons, in increasing order of how much I believe them.

**(a) The discrete wins are real.** ACAS and grid world are published NeuS 2025
benchmarks, and compositional beats monolithic on both. That is a result today,
not a promise.

**(b) The negative result is publishable and precise.** "It doesn't scale" is
not a contribution. "It fails because the admitted control becomes the union
over spanned contract cells, refinement gains 10% per 4×, and here is the
measured baseline that beats it" — that is a contribution, because the next
person does not have to spend two weeks rediscovering it.

**(c) The expressiveness argument, which survived every negative result.**

This is the one that matters. Per-step reachability beat us on NAV — but look at
what the drone benchmark actually asks:

```
(implies, (active, Main), (implies, (eq, battery, 'critical'),
          (or, (eq, move, 'land'), (eq, z_a, 0))))
```

In English: *"whenever the main mission branch is running and the battery is
critical, the drone must either be landing or already on the ground."*

A reachability tool computes the set of positions the drone can reach. It has
**no notion of which behavior-tree node is active**, or what status a node
returned. It cannot state this property, let alone prove it. Neither can
α,β-CROWN.

These are not reachability properties. They are **structural** properties about
the controller's decision-making — and they are exactly what a safety engineer
writes, because "the emergency branch must take over within 0.2 s of the
geofence check failing" is the requirement, not "the reachable set avoids the
obstacle."

So the honest pitch is not *"compositional is faster."* It is:

> **These are the properties people actually need, and nothing that handles
> continuous state can express them.**

---

## 9. Where this leaves the project

There is a real gap, and it sits between two mature literatures that do not
touch:

- **BT verification** (BehaVerify, BT2Fiacre, BT2BIP, MoVe4BT, BTCompiler) — all
  discrete. Only BehaVerify handles neural leaves at all.
- **Neural control verification** (Verisig, ReachNN, CORA, JuliaReach, Flow*,
  NNV, the ARCH-COMP AINNCS entrants) — all continuous. None handle behavior
  trees.

And the semantics already exist: Sprague & Ögren, *"Continuous-Time Behavior
Trees as Discontinuous Dynamical Systems"* (IEEE L-CSS, 2021) formalised
continuous-time BTs as hybrid dynamical systems and proved convergence
conditions. Somebody already argued the formalism matters. Nobody built the
verifier.

**The risk, stated plainly:** the gap may be unfilled because it is hard, not
because nobody noticed. We have measured exactly how hard. That cuts both ways —
it is evidence the problem is real, and evidence it may not yield.

---

## 10. One-paragraph version

Modern robots are part hand-written decision logic (behavior trees) and part
learned control (neural networks). Neural network verifiers prove one-step facts
and cannot reason about time; model checkers reason about time and drown in
ReLUs; discretizing the world makes the problem finite but proves things about a
robot that does not exist — and we have a counterexample where that gap hid a
real bug. Compositional verification splits the work: the NN verifier proves
local contracts, the model checker reasons about time. It wins clearly on
discrete benchmarks and stalls on continuous ones for a reason we can now state
precisely. What keeps it interesting is that the properties engineers actually
write — "the emergency branch must take over when the battery check fails" — are
about the controller's structure, and no continuous-state tool can even express
them.
