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
import re
from dataclasses import dataclass, field

SPEC_MARKER = "--------------SPECIFICATIONS\n"
VAR_MARKER = "--START OF BLACKBOARD VARIABLES DECLARATION\n"
DECIMALS = 6


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
    """

    region: list[RegionBound]
    target: str
    kind: str
    lo: float | None = None
    hi: float | None = None
    forbidden: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "interval":
            if self.lo is None or self.hi is None:
                raise ValueError("interval contract needs both lo and hi")
            if self.lo > self.hi:
                raise ValueError(f"empty interval [{self.lo}, {self.hi}]")
        elif self.kind == "never_select":
            if self.forbidden is None:
                raise ValueError("never_select contract needs 'forbidden'")
        else:
            raise ValueError(f"unknown contract kind {self.kind!r}")

    def guarantee_smv(self, free_var: str) -> str:
        """The consequent. Interval bounds are rounded OUTWARD."""
        if self.kind == "interval":
            return (f"(system.{free_var} >= {_literal(floor_to(self.lo))} & "
                    f"system.{free_var} <= {_literal(ceil_to(self.hi))})")
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
