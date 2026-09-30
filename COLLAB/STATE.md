# Where this project is

Read this first. `DEAD_ENDS.md` is what has already been tried and closed;
`claims.jsonl` is the evidence behind every statement here, reachable with
`./collab.py show <id>`.

## The goal

**Compositional BehaVerify**: verify a neuro-symbolic behavior tree by proving
assume-guarantee contracts about its neural leaves with α,β-CROWN and
discharging the tree's temporal properties against those contracts with a
symbolic solver. The incumbent is **monolithic BehaVerify** (NeuS 2025), which
compiles the network into an exhaustive input-output lookup table inside one
nuXmv model.

The question is not "does compositional work". It works. The question is
**where it beats the table**, and that has turned out to be a much narrower
place than we assumed.

## The two walls, and why we kept hitting one or the other

Everything this project has learned reduces to a pair of constraints that pull
in opposite directions.

**Wall 1 — the table.** On a small, enumerable neural input space the table is
*exact* and *cheap*. Grid world has 2,401 input points; ACAS Xu has 19,360
physical states. Contracts are an over-approximation, so on those domains they
can at best tie, and the CROWN calls to build them are pure added cost. This
was settled on 2026-09-18 and it is not worth re-litigating. Compositional does
*win* on both (grid world safety and liveness in 0.33 s via Floyd-style ranking
contracts, claim #49; ACAS ~6× with three caveats, claim #50) — but it wins
without the table ever being the bottleneck, so the win does not generalise.

**Wall 2 — reachability.** On a continuous state space with a thin tree, a
reachability tool wins outright. The geofence pilot was built to beat this wall
and did not: splitting the *initial set* 32×32 lets exact-polygon reachability
prove the property at 1.2625 against a true peak of 1.2622 — **tighter than our
own proof floor of 1.270** — in 270 s (claim #105). CORA verifies its own
ARCH-COMP ACC benchmark in 8.9 s (claim #101). Anything measured in ℝⁿ over
time, they win.

**We have never built a benchmark that faces both walls at once.** Each one was
designed to beat one opponent and handed the other an easy victory.

## The benchmark specification this implies

A benchmark where compositional BehaVerify is the right tool must have **all
four** of these. Three is not enough; that is exactly how we got here.

1. **A neural input space that cannot be enumerated** — continuous, or simply
   too large for a table. Without this the table wins and nothing else matters.
2. **A tree whose discrete structure dominates** — memory, priority, parallel
   composition, ten or more nodes. The geofence had three nodes and one branch.
3. **State dimension high enough that initial-set splitting is expensive.**
   Splitting cost is exponential in dimension: 32-per-axis is 10³ boxes in 2-D
   and 10⁹ in 6-D. This is the cheapest of the four traits to buy and the one
   whose absence killed the geofence.
4. **Properties over node activation.** The only *categorical* advantage found
   so far: a reachability tool cannot state "Brake is never active for more
   than N consecutive ticks" at all, not slowly (claims #24, #93).

Whether a benchmark with all four exists in the published literature is
**open**, and has not been searched properly. Every previous time we concluded
"nobody has done this", the caveat was found on the second look.

## What is established

- **No behavior-tree tool supports neural leaves** (#77, from the group's own
  FM 2026 related work). So there is no cross-tool baseline to lose; the
  baseline is monolithic BehaVerify.
- **No closed-loop neural verifier supports discrete execution structure**,
  mode sequencing or temporal logic over control flow (#24). Checked across
  CORA, JuliaReach, Flow*, SpaceEx, NNV, Verisig, POLAR.
- **Solver ordering on this contract architecture: Z3 > nuXmv > UCLID5** —
  measured four independent times (#55, #60, #91, #92). nuXmv is the wrong
  back end for contract-based continuous verification even though its DSL front
  end is the right place to express the tree.
- **nuXmv cannot check CTL over reals at all.** Not slowly — it refuses to
  build the BDD FSM.
- **A failed compositional proof is a replayable trace.** It is either a real
  counterexample or a named cell to refine; measured on the geofence at three
  limits. A reachability verdict hands you no state to check. This survives the
  geofence withdrawal intact.
- **The geofence's plant is not ours.** It is the double integrator of the
  run-time-assurance tutorial (Hobbs et al., IEEE CSM 2023,
  doi:10.1109/MCS.2023.3234380), and its envelope equals that paper's viability
  kernel to four decimals (#100).

## What is open

- Does a benchmark meeting all four specification points exist in the
  literature? **Search this before building anything.**
- The parallel-node expressiveness gap: BehaVerify needs 2N boolean bits and
  nuXmv decides N = 30 over 2·10¹⁴ states in 52 ms (#104). The CORA half of
  that comparison is disputed — the cited timings are continuous flowpipe
  computations on unrelated systems — and the fair version is an
  *expressiveness* argument, not a timing table.
- Inter-sample soundness. Every result is at sample points under an Euler
  discretisation. Inflating each state by |v|·dt would turn the discrete-time
  defence into a result; not attempted.
- Native CORA `hybridAutomaton.reach()` for the geofence. Our scripts drive
  CORA's set operations manually.
- Compositional does not scale past ~40 steps on a 2-state system: proved at
  0.1 / 7.7 / 54.9 / 246 s for 10 / 20 / 30 / 40 steps, no proof at 60 and 80,
  and widening the contract grid to fix the escape turned that into a 1500 s
  timeout. Unexplained.

## How we work

A claim is not established because its author is confident. It is established
when the *other* agent checks it, and `collab.py` refuses self-verification.
Register a premise before relying on it; `./collab.py status` is the queue.

Two habits have earned their place and should be kept:

- **Write the attack on your own claim into its evidence field.** It has
  found the fatal flaw twice, including the one that withdrew the geofence.
- **Be more suspicious of a self-confirmation than of a self-refutation.**
  Claims #94/#95 were wrong because a first grep matched and the tail was never
  read.
