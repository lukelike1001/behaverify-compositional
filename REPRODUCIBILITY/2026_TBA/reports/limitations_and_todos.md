# Compositional BehaVerify Limitations and To-dos

**Last Updated:** 2026-07-14

2026-08-25 Note: This is very outdated, so I wouldn't rely on using this as the
most up-to-date stuff.

Living document (unlike dated snapshot reports). Append, edit, and reprioritize freely.

---

## To-dos

**1. ACAS Xu 52 Lassos:** Run the liveness implementation (run the 52 lasso pins through CROWN, patch, and check the CTL spec. This completes the temporal-parity claim with real tool output rather than my sandbox check).

**2. ACAS Xu Perturbations:** Zero-slack fragility experiment from the perturbation discussion.

**3. ACAS Xu Continuous Contracts:** The real question remains the under-approximated kernel representation.

---

## Limitations

**1. Soundness is scattered, not paper-ready.**
Pieces exist (free-NN over-approx, nondeterministic $R \supseteq \mathrm{Reach}_{\mathrm{true}}$, corridor termination relative to mono + shared physics), but there is no single crisp paragraph stating “abstract safety ⇒ real closed-loop safety” with explicit assumptions (SAT-only injection, dropped UNSATs, when loose $R$ can spuriously fail vs when `INVARSPEC=true` is conclusive).

**2. Proved vs computed is easy to blur.**
Hover ⇒ $V=\mathrm{Safe}$ is theorem-shaped; 38-contract set equality, 790/791 unreachable UNSATs, and corridor end-to-end runs are computations/tool artifacts. Corridor “≤2 CROWN queries” is conditional on mono true + shared physics. Reviewers will poke any claim that mixes these levels.

**3. Framework is safety-inductive; CTLSPEC stays open compositionally.**
Kernel / $\partial V$ contracts only constrain forbidden actions. The patched NN is still free within $Allowed$. On grid world, stay is always allowed, so eventual-goal CTL remains false under the abstraction even when monolithic is true (same structural gap as April 2026). Progress/ranking contracts or another abstraction are required—not more crash edges on $\partial V$.

**4. Continuous mode is the strategic hole.**
The main differentiator vs BehaVerify’s discrete table approach is continuous domains. Continuous results today are weak or embarrassing (off-lattice UNSATs; ACAS continuous unfinished). Under-approximated kernel representation for continuous $V$ is open. For SAIV/NeuS, continuous must be an experiment, a partial result, or an explicit out-of-scope with a technical reason—not silent.

**5. “1 contract / ~6× faster” is instance-dependent, not a law.**
ACAS corridor $|\partial V \cap R| = 1$ and the ~6× wall-time / lower-memory story are properties of this seed, physics, and weights. Not established for $|\partial V \cap R| \gg 1$, other maps, or warm vs cold CROWN. Paper must not claim general compositional speedup.

**6. Classical viability / controlled-invariance citation gap.**
Finite-state $V$ here instantiates classical viability ideas; the lit bridge (and precise “what is new: NSBT + CROWN + BehaVerify contracts on $\partial V$”) is not yet written. Risk of looking like renamed controlled invariance without citation.

**7. Tool-vs-theory product is underspecified.**
Current strength is a verification *method* for NSBT safety (artifact + two examples + unification), not a deep complexity/theory result. Venue pitch (method paper vs theory paper) and how much more benchmark breadth is required are still open.

**8. Solo + multi-agent provenance.**
Key claims need re-runs pinned to commits by the author; report provenance notes are good hygiene but are not a substitute for reproducible scripts and human-owned theorems in the camera-ready story.

**9. Second non-degenerate discrete instance missing.**
Besides ACAS corridor, there is no controlled example where $|\partial V \cap R|$ is moderate (tens–hundreds) to show the method degrades gracefully. Grid world is the degenerate floor; ACAS is currently an extreme ceiling of 1.
