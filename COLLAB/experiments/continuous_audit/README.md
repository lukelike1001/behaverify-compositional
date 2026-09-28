# What monolithic BehaVerify can and cannot do with continuous state

Luke's framing was "monolithic BehaVerify cannot handle continuous state spaces."
Measured, that is too broad. The real boundary is narrower and in a different place.

| | works? | evidence |
|---|---|---|
| symbolic BT over real-valued state | **yes** | `thermostat.tree`, 2 real invariants `is true` in 0.03s |
| neural variable reading real state | **no** | `Neural Network inputs must be of type INT` (`check_grammar.py:643`) — all 5 encodings |
| regression network, any encoding but `table` | **no** | `Cannot have a regression network in nuXmv without using table` |
| `table` encoding | yes, any ops | but needs an enumerable input space, so not continuous |
| CTL over real-valued state | **no** | BDD: "infinite precision variables"; MathSAT: no `check_ctlspec` |

So the gap is not continuous state. It is **the interface between a network and
continuous state**, plus **branching-time properties over continuous state**.

Reproduce:
```bash
behaverify nuxmv COLLAB/experiments/continuous_audit/thermostat.tree /tmp/t --generate
nuXmv -source <(printf 'go_msat\nmsat_check_invar_bmc -a een-sorensson\nquit\n') /tmp/t/nuxmv/thermostat.smv
./COLLAB/experiments/encoding_audit/run_audit.sh      # the 5x3x2 encoding matrix
```
