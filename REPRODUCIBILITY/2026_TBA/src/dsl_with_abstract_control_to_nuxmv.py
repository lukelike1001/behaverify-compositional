"""
dsl_with_abstract_control_to_nuxmv.py

Inject verified A/G contracts into a BehaVerify SMV **without ever generating a
neural lookup table**.

WHY THIS EXISTS
---------------
`dsl_with_contracts_to_nuxmv.py` generates the full monolithic SMV with the
network encoded as a `table`, then regexes the table back out. That inherits
every limitation of the table path, and one of them is fatal for continuous
domains: a `NEURAL` variable's inputs must be type `INT`
(`check_grammar.py:643`), so the state can never be real-valued.

The restriction is attached to the `NEURAL` declaration, not to nuXmv. If the
tree declares the control as an **ordinary blackboard variable**, no table is
produced, the check never fires, and the state may be `real`. Measured on
`COLLAB/experiments/no_table/`: continuous-state model, `is true` with contracts
and `is false` without.

So this module does not remove anything. It **frees** an already-ordinary
control variable and constrains it with contracts.

SOUNDNESS
---------
Interval bounds come from CROWN as floats and are emitted as SMV decimals.
Rounding a lower bound up, or an upper bound down, narrows the certified
interval and makes the contracts unsound. Every bound is rounded **outward**.

An inconsistent contract set proves everything. `check_consistency` must be run
before any verdict is trusted; see `COLLAB/experiments/uclid5/vacuity_guard.sh`
for the measured failure (an unsatisfiable set proves `1.0 == 2.0`).

The original module is left untouched so the April 2026 results stay reproducible.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import subprocess
import tempfile
from dataclasses import dataclass, field

SPEC_MARKER = "--------------SPECIFICATIONS\n"
VAR_MARKER = "--START OF BLACKBOARD VARIABLES DECLARATION\n"
DECIMALS = 6


class VacuityError(ValueError):
    """Raised when a model or contract set is vacuous (e.g. empty initial states)."""


class DeadlockError(VacuityError):
    """Raised when a model fails to make progress to the declared horizon."""



def floor_to(value: float, decimals: int = DECIMALS) -> float:
    """Round DOWN. Safe for a lower bound: never narrows the interval."""
    scale = 10 ** decimals
    return math.floor(value * scale) / scale


def ceil_to(value: float, decimals: int = DECIMALS) -> float:
    """Round UP. Safe for an upper bound: never narrows the interval."""
    scale = 10 ** decimals
    return math.ceil(value * scale) / scale


@dataclass(frozen=True)
class RegionBound:
    """One dimension of a contract's region of applicability."""

    var: str
    lo: float | int
    hi: float | int

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"empty region on {self.var}: [{self.lo}, {self.hi}]")

    @property
    def bare(self) -> str:
        """Name without a 'system.' prefix, so callers may pass either form."""
        return self.var[len("system."):] if self.var.startswith("system.") else self.var

    def to_smv(self) -> str:
        """Render this dimension as an SMV conjunct."""
        return (f"system.{self.bare} >= {_literal(self.lo)} & "
                f"system.{self.bare} <= {_literal(self.hi)}")


@dataclass(frozen=True)
class Contract:
    """
    One verified assume-guarantee contract.

    ``interval``      the control lies in [lo, hi] throughout the region
    ``never_select``  the control never takes ``forbidden`` in the region
    ``affine``        the control lies between two LINEAR functions of the
                      state, ``sum(coeffs) + lo`` and ``sum(coeffs) + hi``

    The affine kind exists because concretizing CROWN's linear relaxation to a
    box discards most of its strength: on NAV, affine contracts are 2.5-8x
    tighter than intervals at an identical contract count, which is a ~32x
    saving in contracts needed for a given precision.  It costs nothing in
    expressiveness - the constraints stay linear, so the same solvers apply.
    """

    region: list[RegionBound]
    target: str
    kind: str
    lo: float | None = None
    hi: float | None = None
    forbidden: str | None = None
    coeffs: tuple[tuple[str, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.kind == "interval":
            if self.lo is None or self.hi is None:
                raise ValueError("interval contract needs both lo and hi")
            if self.lo > self.hi:
                raise ValueError(f"empty interval [{self.lo}, {self.hi}]")
        elif self.kind == "never_select":
            if self.forbidden is None:
                raise ValueError("never_select contract needs 'forbidden'")
        elif self.kind == "affine":
            if self.lo is None or self.hi is None:
                raise ValueError("affine contract needs both lo and hi")
            if self.lo > self.hi:
                raise ValueError(f"empty interval [{self.lo}, {self.hi}]")
            if not self.coeffs:
                raise ValueError("affine contract needs 'coeffs'")
            known = {b.bare for b in self.region}
            unknown = [v for v, _ in self.coeffs if v not in known]
            if unknown:
                raise ValueError(
                    f"affine coefficients reference {unknown} but the region "
                    "does not bound them; the rounding pad would be unsound")
        else:
            raise ValueError(f"unknown contract kind {self.kind!r}")

    def _rounding_pad(self) -> float:
        """How far the constants must move to absorb coefficient rounding.

        Rounding a COEFFICIENT can break an affine bound at some point of the
        region in a way rounding an interval endpoint cannot: the error is
        multiplied by the state value.  Over the region, |error_i * s_i| is at
        most |error_i| * max(|lo_i|, |hi_i|), so summing that bounds the total.
        """
        extent = {b.bare: max(abs(b.lo), abs(b.hi)) for b in self.region}
        return sum(abs(c - round(c, DECIMALS)) * extent[v]
                   for v, c in self.coeffs)

    def _affine_side(self, free_var: str, const: float, lower: bool) -> str:
        """One side of an affine guarantee, with the pad applied outward."""
        pad = self._rounding_pad()
        bound = floor_to(const - pad) if lower else ceil_to(const + pad)
        terms = [f"{_literal(round(c, DECIMALS))} * system.{v}"
                 for v, c in self.coeffs if round(c, DECIMALS) != 0.0]
        terms.append(_literal(bound))
        return (f"system.{free_var} {'>=' if lower else '<='} "
                f"{' + '.join(terms)}")

    def guarantee_smv(self, free_var: str) -> str:
        """The consequent. Every bound is rounded OUTWARD."""
        if self.kind == "interval":
            return (f"(system.{free_var} >= {_literal(floor_to(self.lo))} & "
                    f"system.{free_var} <= {_literal(ceil_to(self.hi))})")
        if self.kind == "affine":
            return (f"({self._affine_side(free_var, self.lo, True)} & "
                    f"{self._affine_side(free_var, self.hi, False)})")
        return f"system.{free_var} != {self.forbidden}"

    def to_invar(self, free_var: str) -> str:
        """Render the whole contract as one INVAR line."""
        guarantee = self.guarantee_smv(free_var)
        if not self.region:
            return f"INVAR {guarantee};"
        antecedent = " & ".join(b.to_smv() for b in self.region)
        return f"INVAR ({antecedent}) -> {guarantee};"


def _literal(value: float | int) -> str:
    """SMV literal. Reals keep a decimal point; ints stay bare."""
    if isinstance(value, int):
        return str(value)
    text = repr(float(value))
    if "e" in text or "E" in text:
        text = f"{value:.{DECIMALS}f}"
    return text if "." in text else text + ".0"


@dataclass
class AbstractControlPatcher:
    """Frees an ordinary control variable and constrains it with contracts."""

    free_suffix: str = "_free"
    notes: list[str] = field(default_factory=list)

    def free_name(self, var: str) -> str:
        """
        Name of the unconstrained stand-in for ``var``.

        The stage suffix is dropped so both ``u1`` and ``u1_stage_0`` yield
        ``u1_free``, matching the proof-of-concept patch.
        """
        base = re.sub(r"_stage_\d+$", "", var)
        return f"{base}{self.free_suffix}"

    def _var_type(self, smv: str, var: str) -> str:
        match = re.search(rf"^\s*{re.escape(var)} : ([^;]+);", smv, re.MULTILINE)
        if match is None:
            raise ValueError(
                f"control variable {var!r} not declared in the SMV. "
                "It must be an ordinary blackboard variable, not a NEURAL one.")
        return match.group(1).strip()

    def free_control(self, smv: str, var: str) -> str:
        """
        Make ``var`` unconstrained so contracts alone determine it.

        Removes its ``init`` and ``next`` assignments and redirects the staged
        definition at a fresh free variable. A leftover assignment would pin the
        control and silently make every contract vacuous, so this verifies its
        own work before returning.
        """
        free = self.free_name(var)
        if free in smv:
            raise ValueError(f"{var!r} already freed; patching twice is unsafe")
        var_type = self._var_type(smv, var)

        updated = re.sub(rf"\n[ \t]*init\({re.escape(var)}\) := [^;]*;", "", smv)
        updated = re.sub(rf"\n[ \t]*next\({re.escape(var)}\) := [^;]*;", "", updated)

        stage = _next_stage(var)
        block = re.compile(
            rf"\n[ \t]*{re.escape(stage)} :=\n[ \t]*case\n.*?[ \t]*esac;",
            re.DOTALL)
        updated, replaced = block.subn(f"\n        {stage} := {free};", updated)
        if replaced == 0:
            updated = re.sub(rf"\n[ \t]*{re.escape(stage)} := [^;]*;",
                             f"\n        {stage} := {free};", updated)

        if VAR_MARKER not in updated:
            raise ValueError(f"VAR-section marker not found: {VAR_MARKER!r}")
        updated = updated.replace(
            VAR_MARKER, f"{VAR_MARKER}        {free} : {var_type};\n", 1)

        self._assert_unpinned(updated, var)
        self.notes.append(f"freed {var} as {free} : {var_type}")
        return updated

    @staticmethod
    def _assert_unpinned(smv: str, var: str) -> None:
        """A surviving init/next would make every contract vacuous."""
        for kind in ("init", "next"):
            if re.search(rf"{kind}\({re.escape(var)}\) :=", smv):
                raise ValueError(
                    f"{kind}({var}) survived freeing; contracts would be vacuous")

    def inject(self, smv: str, contracts: list[Contract]) -> str:
        """Emit contracts as INVAR lines at MODULE main scope."""
        if not contracts:
            return smv
        conflicts = self.check_consistency(contracts)
        if conflicts:
            raise ValueError(
                "inconsistent contracts would make every proof vacuous; "
                f"conflicting pairs (by index): {conflicts}")
        targets = {c.target for c in contracts}
        if len(targets) != 1:
            raise ValueError(f"contracts target multiple variables: {sorted(targets)}")
        free = self.free_name(next(iter(targets)))
        if SPEC_MARKER not in smv:
            raise ValueError(f"marker not found: {SPEC_MARKER!r}")
        lines = [c.to_invar(free) for c in contracts]
        block = ("\n-- A/G contracts (verified by alpha-beta-CROWN). "
                 "No lookup table was generated.\n" + "\n".join(lines) + "\n")
        self.notes.append(f"injected {len(lines)} contract(s) on {free}")
        return smv.replace(SPEC_MARKER, SPEC_MARKER + block, 1)

    def patch(self, smv: str, var: str, contracts: list[Contract]) -> str:
        """Free the control and inject its contracts, in that order."""
        return self.inject(self.free_control(smv, var), contracts)

    @staticmethod
    def check_consistency(contracts: list[Contract]) -> list[tuple[int, int]]:
        """
        Find contract pairs whose regions overlap but whose guarantees cannot
        both hold. An inconsistent set proves everything, so callers must treat
        a non-empty result as fatal rather than advisory.
        """
        conflicts = []
        for i, a in enumerate(contracts):
            for j, b in enumerate(contracts[i + 1:], start=i + 1):
                if a.kind != "interval" or b.kind != "interval":
                    continue
                if not _regions_overlap(a.region, b.region):
                    continue
                if max(a.lo, b.lo) > min(a.hi, b.hi):
                    conflicts.append((i, j))
        return conflicts

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def check_progress(
        self,
        smv: str,
        horizon_var: str,
        horizon_val: int | float,
        nuxmv_bin: str | Path,
        command: str = "check_invar_ic3",
        timeout: int = 30,
    ) -> tuple[bool, str]:
        """
        Verify that initial states exist and the horizon is reachable without deadlock.
        """
        return check_nuxmv_progress(
            smv, horizon_var, horizon_val,
            nuxmv_bin=nuxmv_bin, command=command, timeout=timeout
        )


def _regions_overlap(left: list[RegionBound], right: list[RegionBound]) -> bool:
    """True when the two boxes share at least one point."""
    by_var = {b.var: b for b in right}
    for bound in left:
        other = by_var.get(bound.var)
        if other is None:
            continue
        if bound.hi < other.lo or other.hi < bound.lo:
            return False
    return True


def _next_stage(var: str) -> str:
    """`u1_stage_0` is read; `u1_stage_1` is what the tree writes."""
    match = re.match(r"^(.*)_stage_(\d+)$", var)
    if match is None:
        return var
    return f"{match.group(1)}_stage_{int(match.group(2)) + 1}"


def set_invarspec(smv: str, spec: str) -> str:
    """
    Replace any existing INVARSPEC line(s) in smv with a single INVARSPEC {spec}; line.
    Preserves all INVAR statements (contracts and state invariants).
    """
    pattern = r"\bINVARSPEC\b[\s\S]*?;"
    if re.search(pattern, smv):
        first = True
        def _repl(_match):
            nonlocal first
            if first:
                first = False
                return f"INVARSPEC {spec};"
            return ""
        smv = re.sub(pattern, _repl, smv)
    else:
        end_marker = "--------------END OF SPECIFICATIONS"
        if end_marker in smv:
            smv = smv.replace(end_marker, f"INVARSPEC {spec};\n\n{end_marker}", 1)
        elif SPEC_MARKER in smv:
            smv = smv.replace(SPEC_MARKER, f"{SPEC_MARKER}\nINVARSPEC {spec};\n", 1)
        else:
            smv += f"\nINVARSPEC {spec};\n"
    return smv


def build_progress_guard_smv(smv: str, horizon_var: str, horizon_val: int | float) -> str:
    """
    Emits an SMV model asserting INVARSPEC !({horizon_var} = {horizon_val});.
    In nuXmv, if the horizon IS reachable, nuXmv finds a counterexample and reports 'is false'.
    If the model deadlocks before reaching horizon_val, nuXmv reports 'is true'.
    """
    val_str = _literal(horizon_val)
    return set_invarspec(smv, f"!({horizon_var} = {val_str})")


def build_initial_vacuity_guard_smv(smv: str) -> str:
    """
    Emits an SMV model asserting INVARSPEC FALSE;.
    In nuXmv, if initial states exist and satisfy INVAR, nuXmv reports 'is false' at bound 0.
    If initial states are empty or violate INVAR, nuXmv reports 'is true'.
    """
    return set_invarspec(smv, "FALSE")


def run_nuxmv_interactive(
    smv_text: str,
    nuxmv_bin: str | Path,
    command: str = "check_invar_ic3",
    timeout: int = 30,
) -> tuple[int, str, str]:
    """Execute nuXmv MathSAT interactively on the provided SMV content."""
    bin_path = Path(nuxmv_bin)
    if not (bin_path.exists() and subprocess.os.access(bin_path, subprocess.os.X_OK)):
        raise FileNotFoundError(f"nuXmv not executable at {bin_path}")

    with tempfile.NamedTemporaryFile("w", suffix=".smv", delete=False) as tmp:
        tmp.write(smv_text)
        tmp_path = tmp.name

    script = f"go_msat\n{command}\nquit\n"
    try:
        proc = subprocess.run(
            [str(bin_path), "-int", tmp_path],
            input=script,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


# pylint: disable-next=too-many-arguments,too-many-positional-arguments


# pylint: disable-next=too-many-arguments,too-many-positional-arguments


# pylint: disable=too-many-arguments,too-many-positional-arguments
def check_nuxmv_progress(
    smv: str,
    horizon_var: str,
    horizon_val: int | float,
    nuxmv_bin: str | Path,
    command: str = "check_invar_ic3",
    timeout: int = 30,
) -> tuple[bool, str]:
    """
    nuXmv analogue of UCLID5's vacuity guard.

    Checks:
    1. Initial state is non-empty (INVARSPEC FALSE is false).
    2. Horizon is reachable (INVARSPEC !(horizon_var = horizon_val) is false).
    Raises VacuityError if initial state is empty.
    Raises DeadlockError if horizon is unreachable.
    """
    # 1. Initial vacuity check
    init_smv = build_initial_vacuity_guard_smv(smv)
    ret, out, err = run_nuxmv_interactive(init_smv, nuxmv_bin, command=command, timeout=timeout)
    if ret != 0:
        raise RuntimeError(
            f"nuXmv exited with {ret} during initial vacuity check:\n{err}")
    if "is true" in out:
        raise VacuityError(
            "Initial state is empty or violates INVAR (INVARSPEC FALSE is "
            "true); proof is vacuous.")
    if "is false" not in out:
        raise RuntimeError(
            "nuXmv produced unexpected output on initial vacuity check:\n"
            f"{out}\n{err}")

    # 2. Horizon reachability check
    reach_smv = build_progress_guard_smv(smv, horizon_var, horizon_val)
    ret, out, err = run_nuxmv_interactive(
        reach_smv, nuxmv_bin, command=command, timeout=timeout)
    if ret != 0:
        raise RuntimeError(
            f"nuXmv exited with {ret} during horizon progress check:\n{err}")
    if "is true" in out:
        raise DeadlockError(
            f"Horizon {horizon_var} = {horizon_val} is unreachable "
            f"(INVARSPEC !({horizon_var} = {horizon_val}) is true). "
            "Transition relation deadlocked or truncated; proof is vacuous."
        )
    if "is false" not in out:
        raise RuntimeError(
            "nuXmv produced unexpected output on horizon progress check:\n"
            f"{out}\n{err}")

    return True, ("Progress verified: initial state non-empty and horizon "
                  f"{horizon_var} = {horizon_val} reachable.")



def _contains(region: list[RegionBound], point: dict) -> bool:
    """Whether a trajectory point lies inside every bound of a region."""
    for bound in region:
        value = point.get(bound.bare)
        if value is not None and not bound.lo <= value <= bound.hi:
            return False
    return True


def _check_trajectory_admission(patcher, contracts, trajectory) -> None:
    """Every contract whose region contains a point must admit that point.

    Both ``interval`` and ``affine`` contracts are checked.  Skipping affine
    ones would let an unsound linear bound through the battery unnoticed.
    """
    for step_num, point in enumerate(trajectory):
        for contract in contracts:
            if contract.kind not in ("interval", "affine"):
                continue
            if not _contains(contract.region, point):
                continue
            name = patcher.free_name(contract.target)
            value = point.get(name)
            if value is None:
                value = point.get(contract.target)
            if value is None:
                continue
            offset = 0.0
            if contract.kind == "affine":
                offset = sum(coef * point[var] for var, coef in contract.coeffs
                             if var in point)
            low, high = contract.lo + offset, contract.hi + offset
            if not low <= value <= high:
                raise ValueError(
                    f"Trajectory admission failure at step {step_num}: "
                    f"control {value} not in contract bound [{low}, {high}]")


# pylint: disable-next=too-many-arguments,too-many-positional-arguments


# pylint: disable-next=too-many-arguments,too-many-positional-arguments


# pylint: disable-next=too-many-locals
def verify_with_battery(
    unpatched_smv: str,
    control_var: str,
    contracts: list[Contract],
    horizon_var: str,
    horizon_val: int | float,
    nuxmv_bin: str | Path,
    command: str = "check_invar_ic3",
    reference_trajectory: list[dict[str, float]] | None = None,
    timeout: int = 30,
) -> dict[str, bool]:
    """
    Phase 5 validation battery for abstract control proofs.

    Enforces 5 mandatory guards before a verification verdict is trusted:
    1. consistency_guard: check_consistency() finds overlapping
       contradictions.
    2. progress_guard: check_nuxmv_progress() ensures initial states exist
       and the horizon is reachable.
    3. strip_contracts_fails: unconstrained control MUST FAIL the
       specification (contracts are load-bearing).
    4. drop_one_contract_fails: dropping any contract flips the verdict.
    5. trajectory_admission: the reference trajectory is admitted.

    Returns a dict of check statuses. Raises an exception if any guard fails.
    """
    patcher = AbstractControlPatcher()

    # 1. Consistency guard
    conflicts = patcher.check_consistency(contracts)
    if conflicts:
        raise VacuityError(f"Inconsistent contracts detected: {conflicts}")

    # 2. Patch model
    patched_smv = patcher.patch(unpatched_smv, control_var, contracts)

    # 3. Progress / Non-deadlock guard
    check_nuxmv_progress(
        patched_smv, horizon_var, horizon_val,
        nuxmv_bin=nuxmv_bin, command=command, timeout=timeout
    )

    # 4. Strip contracts must fail
    freed_smv = patcher.free_control(unpatched_smv, control_var)
    ret_freed, out_freed, _ = run_nuxmv_interactive(
        freed_smv, nuxmv_bin, command=command, timeout=timeout
    )
    if ret_freed != 0 or "is false" not in out_freed:
        raise ValueError(
            "Non-vacuity failure: stripped contracts model did not fail "
            f"specification!\n{out_freed}")

    # 5. Patched model must pass
    ret_patched, out_patched, err_patched = run_nuxmv_interactive(
        patched_smv, nuxmv_bin, command=command, timeout=timeout
    )
    if ret_patched != 0 or "is true" not in out_patched:
        raise ValueError(
            f"Patched model failed specification!\n{out_patched}\n{err_patched}")

    # 6. Drop-one-contract must fail (tested on each contract if len(contracts) > 1)
    if len(contracts) > 1:
        for idx in range(len(contracts)):
            subset = [c for i, c in enumerate(contracts) if i != idx]
            subset_smv = patcher.patch(unpatched_smv, control_var, subset)
            _, out_sub, _ = run_nuxmv_interactive(
                subset_smv, nuxmv_bin, command=command, timeout=timeout
            )
            if "is false" not in out_sub:
                raise ValueError(
                    f"Sensitivity failure: dropping contract {idx} did not "
                    "fail specification (contract is redundant or not "
                    "load-bearing)."
                )

    # 7. Trajectory admission (if supplied)
    if reference_trajectory is not None:
        _check_trajectory_admission(patcher, contracts, reference_trajectory)

    return {
        "consistency_guard": True,
        "progress_guard": True,
        "strip_contracts_fails": True,
        "drop_one_contract_fails": True,
        "specification_verified": True,
        "trajectory_admission": reference_trajectory is not None,
    }
