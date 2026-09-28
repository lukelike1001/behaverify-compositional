# COLLAB — shared scratchpad for Claude + Gemini

Two agents, one branch. Everything here is **append-only** so git merges instead
of conflicting, and neither agent can quietly overwrite the other.

## Why the claim ledger exists

The failure mode this is built to prevent: an agent asserts something plausible,
the other agent builds on it, and three days later it turns out nobody checked.
That happened repeatedly in the week before this branch — a contract set that
verified vacuously, a "validated" input function that had never been compared
properly, a literature search that found benchmarks and got reported as finding
a research gap. Each cost real time, and each was caught by the human rather
than by us.

So: **a claim is not established because its author is confident. It is
established when the other agent independently checks it.** `collab.py` refuses
to let you verify your own claim.

## Rules

1. **Append only.** Use `collab.py`. Never hand-edit `LOG.md` or `claims.jsonl`.
2. **Register before relying.** Any factual premise — a benchmark's size, a tool's
   limit, what a paper says — goes in the ledger before it's used to justify work.
3. **You cannot check your own claim.** The other agent does it, and says *how*.
4. **Check `status` before building.** If a premise is still `unverified`, either
   verify it first or state plainly that you're proceeding on an assumption.
5. **Refuting is a contribution.** Killing a bad direction early is the best
   outcome available, better than a week spent confirming it.
6. **Claim your files.** Put what you're editing in `TASKS.md` before you start,
   so we don't collide in the same module.
7. **Negative results get written down.** "We tried X, it failed, here is why" is
   the most reusable thing either of us produces.

## Commands

```bash
./COLLAB/collab.py status                      # what still needs checking
./COLLAB/collab.py post   claude "message"
./COLLAB/collab.py claim  claude "statement" --evidence "how to check it"
./COLLAB/collab.py verify gemini 3 "ran X, got Y, matches"
./COLLAB/collab.py refute gemini 3 "ran X, got Z, contradicts"
./COLLAB/collab.py log -n 60
```

## Context in one paragraph

BehaVerify verifies behavior trees by compiling them to nuXmv. The *monolithic*
approach (published, NEUS 2025) puts the whole system including any neural
network into one model. This project is trying to find a *compositional*
approach that beats it — verify pieces separately, compose the results. The
benchmark hunt so far: grid world and ACAS Xu (compositional never beat the
table), and NAV (a week; the tree is one node, so there was nothing for a model
checker to do). The live question is whether decomposing along **tree structure**
rather than state space is the version that works. Note that behavior-tree
verification is dominated by this lab's own prior work, so "novel" almost always
means "novel relative to monolithic BehaVerify," not relative to an outsider.
