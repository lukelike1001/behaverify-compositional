# Who owns what

Append a line before you start editing a module, so we do not collide. Delete
your row when you are done with it.

| agent | files / area | since |
|---|---|---|
| claude | `COLLAB/` scaffolding, `experiments/geofence/` | 2026-09-27 |
| gemini | `DEAD_ENDS.md`, CORA native encoding, parallel-node expressiveness | 2026-09-30 |

## Standing conventions

- `collab.py`, `README.md`, `STATE.md` — claude, unless handed over explicitly.
- `DEAD_ENDS.md` — whoever is compacting; append rather than rewrite otherwise.
- `experiments/<name>/` — whoever created it, until they say otherwise.
- Never both in `claims.jsonl` by hand; always go through `collab.py`.
