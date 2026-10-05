# COLLAB — shared scratchpad for Claude + Gemini

Two agents, one branch. The claim ledger is append-only so git merges instead
of conflicting, and neither agent can quietly overwrite the other.

## Start here

| file | what it is |
|---|---|
| `STATE.md` | where the project is: the goal, the two walls, what is established, what is open. **Read this first.** |
| `DEAD_ENDS.md` | what has already been tried and closed, and what would revive each one |
| `claims.jsonl` | the ledger: every claim, verification and refutation, verbatim |
| `LOG.md` | free-form discussion between the two agents |
| `AVENUES.md` | the running list of avenues attempted, one sentence each |
| `TASKS.md` | who is editing which files right now |
| `LOG_archive.md` | discussion from before 2026-09-30 |
| `experiments/` | the code behind the claims |

## Why the claim ledger exists

The failure mode this is built to prevent: an agent asserts something
plausible, the other builds on it, and three days later nobody checked. That
happened repeatedly before this branch — a contract set that verified
vacuously, a "validated" input function never properly compared, a literature
search that found benchmarks and got reported as finding a research gap. Each
cost real time, and each was caught by the human rather than by us.

So: **a claim is not established because its author is confident. It is
established when the other agent independently checks it.** `collab.py` refuses
to let you verify your own claim.

## Rules

1. **Append only.** Use `collab.py`. Never hand-edit `LOG.md` or `claims.jsonl`.
2. **Register before relying.** Any factual premise — a benchmark's size, a
   tool's limit, what a paper says — goes in the ledger before it justifies work.
3. **You cannot check your own claim.** The other agent does, and says *how*.
4. **Write the attack on your own claim into its evidence field.** This has
   found the fatal flaw twice, including the one that withdrew the geofence
   headline. It is the single highest-yield habit on this project.
5. **Call the other agent BEFORE stating a number**, and before any
   architectural or implementation decision — not after the claim is formed.
   Send the raw output and ask what it shows. Token cost is not a reason to
   skip this.
6. **Re-running someone's script is not verification.** It reproduces their
   method including its flaws; that is how a shared-cache timing artifact
   passed review. Take a different path, or record it as "reproduced, method
   unchecked".
7. **Baselines come from the pinned source, never from the repo's current
   `src/`.** Every `REPRODUCIBILITY/` folder ships its own source, metamodel
   and Dockerfile. Monolithic numbers use `2025_NEUS/src/` because that is the
   version the paper reports; compositional uses `2026_TBA/src/`. Comparing
   against a post-overhaul dev build invites "you benchmarked an unreleased
   version", and the repo's current `src/` should appear in no measurement.
8. **Run the artifact's own harness before writing one.** `make_tree.sh` and
   `time_make_smv.sh` in `2025_NEUS/examples/grid_world` already sweep every
   encoding across every network. A night was spent reimplementing them badly,
   and three wrong diagnoses came from compiling `template.tree` -- a
   generator INPUT -- as though it were a benchmark.
9. **Check `status` before building.** If a premise is still `unverified`,
   verify it or say plainly that you are proceeding on an assumption.
10. **Claim your files** in `TASKS.md` before editing, so we do not collide.
11. **Negative results get written down** — in `DEAD_ENDS.md`, with the
   condition that would revive them. "We tried X, it failed, here is why" is the
   most reusable thing either of us produces.

## Commands

```bash
./COLLAB/collab.py status                      # what still needs checking
./COLLAB/collab.py show   105                  # one claim in full
./COLLAB/collab.py post   claude "message"
./COLLAB/collab.py claim  claude "statement" --evidence "how to check it"
./COLLAB/collab.py verify gemini 3 "ran X, got Y, matches"
./COLLAB/collab.py refute gemini 3 "ran X, got Z, contradicts"
./COLLAB/collab.py log -n 60
```

Claims are no longer copied into `LOG.md` in full — that duplicated the ledger
word for word and grew the file to 204 KB. The log now carries the discussion
and a one-line pointer per claim; `show` reads the claim itself.
