# `divert`: contracts over a trained network, and where the horizon wall is

2026-10-06. Overnight session. Gemini (Antigravity) reviewed the plan before it
was built and the conclusions after.

## a) High-level goal

Get contracts working on a neuro-symbolic behavior tree in **continuous** state
space - the thing monolithic BehaVerify cannot do at all, because
`check_grammar.py:643` requires neural network inputs to be `INT`.

The property is the **finite-escape hypothesis** of Ögren's behavior-tree
convergence theorem, as stated in Rietz et al. (arXiv:2602.06525v2, Thm 1):
`x_t in Omega_i` implies `x_{t+h} not in Omega_i`. Rietz et al. enforce the
theorem's *other* hypothesis (invariance) by masking actions at runtime, and
leave this one to reward shaping; their Table I reports 7% outright failure.
The property is bounded, so by Alpern and Schneider it is a safety property,
not liveness - but it is the missing precondition of a published liveness
theorem, which is a better thing to be than a liveness claim we cannot support.

## b) Why a reviewer should care

**The headline is a proved bound over a trained network.** `net_divert` is a
4-64-64-64-2 ReLU policy; its contracts come from `auto_LiRPA`'s
`compute_bounds(..., return_A=True)` and are the real linear relaxations, with
`lA != uA` and cross-axis coefficients. Nothing here is a placeholder.

**The negative half is quantified, and that is what makes it useful.** Unrolled
SMT - the method this project has used throughout - cannot verify this
benchmark, and two independent measurements say why. They do not meet by about
three orders of magnitude.

**The night also overturned four of our own results.** Three were soundness
defects that all pushed the same way: they let the model finish *earlier* than
the plant, so a proof under them would not have been a proof. The fourth made
every earlier refinement experiment vacuous.

## c) How this relates to compositional BehaVerify

Directly: the proof is decomposed **along the tree's own sequence node**, which
is the clearest instance so far of the tree structure doing verification work.
Brake and Return are proved separately, with Brake's handover box as the
contract interface between them. Brake is analytic and carries no network
contracts at all; every contract in the proof sits inside Return, over a much
smaller initial set than the trip region.

Gemini's correction, adopted: **do not claim latching modes are beyond a hybrid
automaton.** A latching tree node is an acyclic finite-state graph with linear
guards, which SpaceEx, CORA, JuliaReach and NNV have handled for a decade. The
differentiator is the assume-guarantee *decomposition* the tree syntax dictates,
not the modes themselves.
