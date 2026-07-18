# ACAS Xu: Lasso Determinism Pins (Liveness Dual)

**Date:** 2026-07-15

**Scope:** Dual of safety-only contracts: pin the abstract NN on the true closed-loop
lasso so temporal properties can hold on the SMV model. Feature classes only.

**Reproduce:**

```bash
cd REPRODUCIBILITY/2026_TBA/examples/AcasXu_closed_loop
python3 acas_lasso_pins.py --eps 0 --run-crown
python3 run_acas_liveness_pipeline.py --skip-generate \
  --nuxmv-cmd ../../commands/nuxmv_commands/command_combo_invar_ctl
```

---

## 1. Idea

Safety contracts **never select** a forbidden advisory; the free NN elsewhere breaks CTL
(grid-world §6).

Liveness wants a **small** faithful controller on the real trajectory. The ONNX closed
loop is a **52**-state lasso (stem 39 + cycle 13). At each pair pin

$$
\mathrm{NN}_{a_{\mathrm{prev}}}(s) = a^{*}(s)
$$

with equals-INVARs. Discrete SMV guards exact states ⇒ certify pins at **$\varepsilon = 0$**
(point queries), not a robustness box.

---

## 2. Code

| Type | Role |
|---|---|
| `AcasAugmentedState` | $(x, y, \mathrm{signs}, h, a_{\mathrm{prev}})$ |
| `AcasLassoTrajectory` | load JSON, cycle start, cycle max $\rho$ |
| `AcasLassoPin` / `AcasLassoPinSet` | equals contracts, margins, CROWN, BFS |
| `run_acas_liveness_pipeline.py` | patch SMV + `NuxmvVerifier` |
| `_build_invar_lines` | `guarantee_type: equals` → `command = a^{*}$ |

Default pin $\varepsilon = 0$. Optional $\varepsilon > 0$ for local-robustness histograms only.

Artifacts: `contracts/crown/lasso_pin_specs.json`, `lasso_pin_crown_results.json`.

---

## 3. CTL false at $\rho \geq 1400$: diagnosis (not pin antecedents)

nuXmv CTL counterexample loops with `acas.active = FALSE` and $\rho = 1000$.

Tree / SMV:

$$
\texttt{acas.active} := (\texttt{distance} < \texttt{max\_dist}), \quad \texttt{max\_dist} = 1000.
$$

When inactive, **command and position freeze** (no NN, no env update). The ONNX lasso
script **ignores** `active` and keeps simulating; first inactive lasso index is **7**
($(x,y)=(5,9)$, $\rho=1000$, $a_{\mathrm{prev}}=\texttt{weak\_right}$).

| Model of physics | Abstract reachable size under 52 pins |
|---|---|
| Always update (ONNX lasso) | **52** |
| SMV freeze when $\rho \geq 1000$ | **8** (stem until freeze, then self-loop) |

So:

- **Python collapse to 52** certified the *physics-always* pin map, **not** the SMV encoding.
- **CTL $\mathrm{AG}\,\mathrm{AF}(\rho \geq 1400)$ is false** because freeze never reaches the ONNX cycle at 1400 — not because stage-0 guards failed to fire on the active stem.
- Lemma 3 should split: **containment / collapse under SMV semantics** (freeze), separate from ONNX-lasso bookkeeping.

Correct SMV-facing liveness (active cutoff):

$$
\mathrm{AG}\,\mathrm{AF}(\rho \geq 1000)
$$

(“always eventually reach the far / inactive region”).

---

## 4. Results

### ONNX / abstract

| Check | Result |
|---|---|
| Pins | **52** |
| $\varepsilon$ for certificates | **0** (point) |
| ONNX margins (lattice) | min $+0.0005$, median $+0.0100$, max $+0.0239$ |
| ONNX argmax = required | yes (same argmax that built the lasso; tautological unless second runtime) |
| $\|R\|$ always-physics | **52** |
| $\|R\|$ SMV-freeze | **8** |

### CROWN always-selects at $\varepsilon = 0$

| | Count |
|---|---|
| SAT | **52 / 52** |
| UNSAT | 0 |
| TIMEOUT | 0 |

Pin 43 box-UNSAT at $\varepsilon=10^{-4}$ (earlier session) was **local robustness failure**, not a bad lattice pin. Point certificates close the discrete SMV story.

Box robustness ($\varepsilon \in \{10^{-4}, 10^{-3}, \ldots\}$) remains a **fragility histogram** next to $\rho=200$ grazing — not required for SMV soundness.

### nuXmv (52 equals-INVARs, free NN off pins)

| Spec | Verdict |
|---|---|
| INVARSPEC $\rho \geq 200$ | **true** |
| CTLSPEC $\mathrm{AG}\,\mathrm{AF}(\rho \geq 1000)$ | **true** |
| CTLSPEC $\mathrm{AG}\,\mathrm{AF}(\rho \geq 1400)$ (old) | **false** (freeze diagnosis) |

### Three-way table

| Model | INVAR $\rho \geq 200$ | CTL $\mathrm{AG}\,\mathrm{AF}(\rho \geq 1000)$ |
|---|---|---|
| Monolithic (table) | true (prior baseline) | **not re-run this session** |
| Pinned compositional (52 equals, $\varepsilon=0$ certs) | **true** | **true** |
| Safety-only compositional (corridor Q2) | true (07-12) | **expected false** (free NN; not re-run) |

---

## 5. Duality one-liner

Safety wants a **large** inductive set and **never-selects** contracts on $\partial V$;
liveness wants a **small** controller abstraction and **always-selects** pins on
$\mathrm{Reach}_{\mathrm{true}}$. ACAS is where both ends are small enough to try
($|\partial V \cap R| = 1$ corridor; 52 lattice pins). Under SMV, “true trajectory”
must respect `active` freeze — not the pure-ONNX lasso past $\rho = 1000$.

---

## 6. Follow-ups

1. Fill monolithic + safety-only CTL rows (single reruns).  
2. Optional margin histogram vs $\varepsilon$ (fragility figure).  
3. Optional second ONNX runtime for non-tautological argmax parity.  
4. Optional `AF AG` on freeze attractor once desired for the paper.
