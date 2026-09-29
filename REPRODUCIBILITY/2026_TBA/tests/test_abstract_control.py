"""
REPRODUCIBILITY/2026_TBA/tests/test_abstract_control.py

Validation harness and adversarial test suite for dsl_with_abstract_control_to_nuxmv.py.

ENFORCING:
1. Soundness of rounding: CROWN float bounds must round outward (floor_to for lo,
   ceil_to for hi). Naive formatting/rounding narrows certified intervals and fails tests.
2. Vacuity trap: check_consistency detects contradictory contracts, and the pipeline
   refuses to emit a vacuous model with inconsistent contracts.
3. Non-vacuity end-to-end: verified on abstract_nav.tree with nuXmv MathSAT.
   With contracts -> is true; without contracts -> is false.
4. Complete freeing: no surviving init/next assignments for the freed variable.
5. Idempotence & error handling: double patching, nonexistent variables, invalid contracts,
   and multi-target injections raise ValueError.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure REPRODUCIBILITY/2026_TBA/src is on sys.path
_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dsl_with_abstract_control_to_nuxmv import (  # noqa: E402
    AbstractControlPatcher,
    Contract,
    DeadlockError,
    RegionBound,
    VacuityError,
    build_initial_vacuity_guard_smv,
    build_progress_guard_smv,
    DECIMALS,
    ceil_to,
    check_nuxmv_progress,
    floor_to,
    set_invarspec,
    verify_with_battery,
)
import dsl_to_nuxmv  # noqa: E402

_REPO_ROOT = _TESTS_DIR.parents[2]
_METAMODEL_PATH = _TESTS_DIR.parent / "metamodel" / "behaverify.tx"
_ABSTRACT_NAV_TREE = _REPO_ROOT / "COLLAB" / "experiments" / "no_table" / "abstract_nav.tree"
_NUXMV_BIN = _TESTS_DIR.parent / "nuXmv_DL" / "bin" / "nuXmv"


@pytest.fixture(scope="module")
def abstract_nav_smv() -> str:
    """Generate the baseline unpatched SMV from abstract_nav.tree."""
    assert _ABSTRACT_NAV_TREE.exists(), f"Missing fixture {_ABSTRACT_NAV_TREE}"
    assert _METAMODEL_PATH.exists(), f"Missing metamodel {_METAMODEL_PATH}"

    with tempfile.NamedTemporaryFile(suffix=".smv", mode="w+", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        dsl_to_nuxmv.dsl_to_nuxmv(
            str(_METAMODEL_PATH),
            str(_ABSTRACT_NAV_TREE),
            tmp_path,
            True,   # keep_stage_0
            False,  # keep_last_stage
            False,  # do_not_trim
            False,  # behave_only
            0,      # recursion_limit
            False,  # return_values
            False,  # skip_grammar_check
            None,   # record_times
        )
        with open(tmp_path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_nuxmv_interactive(smv_text: str, command: str = "check_invar_ic3") -> tuple[int, str, str]:
    """Execute nuXmv MathSAT interactively on the provided SMV content."""
    assert _NUXMV_BIN.exists() and os.access(_NUXMV_BIN, os.X_OK), f"nuXmv not runnable at {_NUXMV_BIN}"

    with tempfile.NamedTemporaryFile(suffix=".smv", mode="w+", delete=False) as tmp:
        tmp.write(smv_text)
        tmp_path = tmp.name

    script = f"go_msat\n{command}\nquit\n"
    try:
        proc = subprocess.run(
            [str(_NUXMV_BIN), "-int", tmp_path],
            input=script,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.returncode, proc.stdout, proc.stderr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================================================
# 1. SOUNDNESS OF ROUNDING
# =============================================================================

class TestRoundingSoundness:
    """
    Lower bounds must round DOWN (floor_to) and upper bounds UP (ceil_to).
    Any naive formatting (round, str truncation, f"{x:.6f}") rounds in the
    wrong direction for half of all floats, silently under-approximating the
    certified neural output and producing unsound contracts.
    """

    @pytest.mark.parametrize("val", [
        0.1234569,          # naive round(val, 6) -> 0.123457 > val (UNSOUND)
        -0.1234561,         # naive round(val, 6) -> -0.123456 > val (UNSOUND)
        1.0000001,
        -1.0000009,
        0.3333333333333333,
        -0.6666666666666666,
        1e-5,
        -1e-5,
        0.0,
        -0.0,
    ])
    def test_floor_to_lower_bound_soundness(self, val: float) -> None:
        """floor_to must guarantee result <= original_value (never tightens lower bound)."""
        rounded = floor_to(val, decimals=6)
        assert rounded <= val, f"Soundness violation: floor_to({val}) = {rounded} > {val}"

    @pytest.mark.parametrize("val", [
        0.1234561,          # naive round(val, 6) -> 0.123456 < val (UNSOUND)
        -0.8765439,         # naive round(val, 6) -> -0.876544 < val (UNSOUND)
        1.0000009,
        -1.0000001,
        0.3333333333333333,
        -0.6666666666666666,
        1e-5,
        -1e-5,
        0.0,
        -0.0,
    ])
    def test_ceil_to_upper_bound_soundness(self, val: float) -> None:
        """ceil_to must guarantee result >= original_value (never tightens upper bound)."""
        rounded = ceil_to(val, decimals=6)
        assert rounded >= val, f"Soundness violation: ceil_to({val}) = {rounded} < {val}"

    def test_contract_emitted_bounds_are_strictly_sound(self) -> None:
        """
        Adversarial test: verify that Contract.to_invar() emits lower bounds <= original
        and upper bounds >= original on inputs where naive rounding rounds inwards.
        """
        # Test case 1: Positive floats where naive round rounds inwards
        lo_pos = 0.1234569   # naive round -> 0.123457 (> lo, unsound)
        hi_pos = 0.8765431   # naive round -> 0.876543 (< hi, unsound)
        c_pos = Contract(
            region=[RegionBound("v_stage_0", -1.0, 1.0)],
            target="u1_stage_0",
            kind="interval",
            lo=lo_pos,
            hi=hi_pos,
        )
        invar_pos = c_pos.to_invar("u1_stage_0_free")

        # Parse emitted lower and upper bounds
        lo_match = re.search(r"system\.u1_stage_0_free >= ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", invar_pos)
        hi_match = re.search(r"system\.u1_stage_0_free <= ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", invar_pos)
        assert lo_match is not None, f"Failed to parse lower bound from: {invar_pos}"
        assert hi_match is not None, f"Failed to parse upper bound from: {invar_pos}"

        emitted_lo = float(lo_match.group(1))
        emitted_hi = float(hi_match.group(1))

        assert emitted_lo <= lo_pos, f"Lower bound was unsoundly rounded up: {emitted_lo} > {lo_pos}"
        assert emitted_hi >= hi_pos, f"Upper bound was unsoundly rounded down: {emitted_hi} < {hi_pos}"

        # Test case 2: Negative floats where naive round rounds inwards
        lo_neg = -0.5555551  # naive round -> -0.555555 (> lo, unsound)
        hi_neg = -0.2222229  # naive round -> -0.222223 (< hi, unsound)
        c_neg = Contract(
            region=[RegionBound("v_stage_0", -1.0, 1.0)],
            target="u1_stage_0",
            kind="interval",
            lo=lo_neg,
            hi=hi_neg,
        )
        invar_neg = c_neg.to_invar("u1_stage_0_free")

        lo_match_neg = re.search(r"system\.u1_stage_0_free >= ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", invar_neg)
        hi_match_neg = re.search(r"system\.u1_stage_0_free <= ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", invar_neg)
        assert lo_match_neg is not None
        assert hi_match_neg is not None

        emitted_lo_neg = float(lo_match_neg.group(1))
        emitted_hi_neg = float(hi_match_neg.group(1))

        assert emitted_lo_neg <= lo_neg, f"Negative lo was unsoundly rounded up: {emitted_lo_neg} > {lo_neg}"
        assert emitted_hi_neg >= hi_neg, f"Negative hi was unsoundly rounded down: {emitted_hi_neg} < {hi_neg}"

    def test_mechanically_demonstrate_naive_formatting_fails(self) -> None:
        """
        Verify that this test suite would catch and reject naive f-string formatting.
        This proves the test is truly adversarial and not passing vacuously.
        """
        test_lo = 0.1234569
        test_hi = 0.8765431

        # What naive formatting would produce
        naive_lo = float(f"{test_lo:.6f}")
        naive_hi = float(f"{test_hi:.6f}")

        # Assert that naive formatting FAILS the soundness condition
        assert naive_lo > test_lo, "Sanity check: naive formatting should round lo up"
        assert naive_hi < test_hi, "Sanity check: naive formatting should round hi down"

        # Assert that our implementation SUCCEEDS where naive fails
        sound_lo = floor_to(test_lo, 6)
        sound_hi = ceil_to(test_hi, 6)
        assert sound_lo <= test_lo
        assert sound_hi >= test_hi


# =============================================================================
# 2. THE VACUITY TRAP & CONSISTENCY CHECKING
# =============================================================================

class TestVacuityTrapAndConsistency:
    """
    An inconsistent contract set constrains the free variable to empty,
    making the antecedent false or the system deadlocked, which vacuously proves
    ANY specification. The patcher must detect this and refuse to emit such models.
    """

    def test_check_consistency_hook_exists_and_detects_contradictions(self) -> None:
        """Demand that check_consistency exists and detects overlapping conflicting intervals."""
        patcher = AbstractControlPatcher()
        assert hasattr(patcher, "check_consistency"), "AbstractControlPatcher must expose check_consistency"

        # Region overlaps: v in [-1.0, 1.0] and v in [0.0, 2.0] overlap on [0.0, 1.0]
        # Guarantees conflict: u in [0.0, 0.5] vs u in [0.6, 1.0] (disjoint)
        c1 = Contract(
            region=[RegionBound("v_stage_0", -1.0, 1.0)],
            target="u1_stage_0",
            kind="interval",
            lo=0.0,
            hi=0.5,
        )
        c2 = Contract(
            region=[RegionBound("v_stage_0", 0.0, 2.0)],
            target="u1_stage_0",
            kind="interval",
            lo=0.6,
            hi=1.0,
        )

        conflicts = patcher.check_consistency([c1, c2])
        assert len(conflicts) > 0, "check_consistency must detect contradictory contracts on overlapping regions"
        assert conflicts == [(0, 1)], f"Expected conflict between contract 0 and 1, got {conflicts}"

    def test_check_consistency_detects_contradiction_in_universal_contracts(self) -> None:
        """Universal contracts (empty region list = everywhere) with disjoint intervals must conflict."""
        patcher = AbstractControlPatcher()
        c1 = Contract(region=[], target="u1_stage_0", kind="interval", lo=-1.0, hi=0.0)
        c2 = Contract(region=[], target="u1_stage_0", kind="interval", lo=0.5, hi=1.0)

        conflicts = patcher.check_consistency([c1, c2])
        assert len(conflicts) > 0, "Universal contracts with disjoint intervals must be detected as conflicting"

    def test_check_consistency_allows_compatible_overlapping_regions(self) -> None:
        """Overlapping regions with overlapping intervals share a solution and are valid."""
        patcher = AbstractControlPatcher()
        # Overlapping regions, overlapping guarantees: [0.0, 1.0] and [0.5, 1.5] share [0.5, 1.0]
        c1 = Contract(
            region=[RegionBound("v_stage_0", -1.0, 1.0)],
            target="u1_stage_0",
            kind="interval",
            lo=0.0,
            hi=1.0,
        )
        c2 = Contract(
            region=[RegionBound("v_stage_0", 0.0, 2.0)],
            target="u1_stage_0",
            kind="interval",
            lo=0.5,
            hi=1.5,
        )

        conflicts = patcher.check_consistency([c1, c2])
        assert conflicts == [], f"Compatible contracts should not produce conflicts, got {conflicts}"

    def test_check_consistency_allows_disjoint_regions(self) -> None:
        """Disjoint regions can have completely disjoint guarantees without conflict."""
        patcher = AbstractControlPatcher()
        # Region 1: v in [-10.0, -1.0], Region 2: v in [1.0, 10.0] -> Disjoint regions!
        c1 = Contract(
            region=[RegionBound("v_stage_0", -10.0, -1.0)],
            target="u1_stage_0",
            kind="interval",
            lo=0.0,
            hi=1.0,
        )
        c2 = Contract(
            region=[RegionBound("v_stage_0", 1.0, 10.0)],
            target="u1_stage_0",
            kind="interval",
            lo=-1.0,
            hi=0.0,
        )

        conflicts = patcher.check_consistency([c1, c2])
        assert conflicts == [], f"Contracts with disjoint regions must not conflict, got {conflicts}"

    def test_check_consistency_multi_dimensional_overlap(self) -> None:
        """Contracts in 2D: overlapping in dimension X but disjoint in dimension Y are disjoint overall."""
        patcher = AbstractControlPatcher()
        c1 = Contract(
            region=[RegionBound("x_stage_0", 0.0, 1.0), RegionBound("y_stage_0", 0.0, 1.0)],
            target="u1_stage_0",
            kind="interval",
            lo=0.0,
            hi=1.0,
        )
        c2 = Contract(
            region=[RegionBound("x_stage_0", 0.0, 1.0), RegionBound("y_stage_0", 2.0, 3.0)],
            target="u1_stage_0",
            kind="interval",
            lo=2.0,
            hi=3.0,
        )

        conflicts = patcher.check_consistency([c1, c2])
        assert conflicts == [], "2D regions disjoint in y must not report a conflict"

    def test_pipeline_refuses_inconsistent_contracts(self, abstract_nav_smv: str) -> None:
        """
        Adversarial test: the pipeline MUST refuse to patch a model with inconsistent contracts.
        Emitting an inconsistent SMV creates a vacuous proof, which must be blocked.
        """
        patcher = AbstractControlPatcher()
        c1 = Contract(
            region=[RegionBound("v_stage_0", -1.0, 1.0)],
            target="u1_stage_0",
            kind="interval",
            lo=0.0,
            hi=0.5,
        )
        c2 = Contract(
            region=[RegionBound("v_stage_0", 0.0, 2.0)],
            target="u1_stage_0",
            kind="interval",
            lo=0.6,
            hi=1.0,
        )

        with pytest.raises((ValueError, RuntimeError), match="(?i)inconsistent|conflict|contradict"):
            patcher.patch(abstract_nav_smv, "u1_stage_0", [c1, c2])


# =============================================================================
# 3. THE FREEING IS COMPLETE
# =============================================================================

class TestFreeingCompleteness:
    """
    After free_control, the SMV must contain NO remaining init or next assignment
    for that variable. Any leftover assignment pins the control, silently making
    every contract vacuous.
    """

    def test_no_remaining_init_or_next_assignments(self, abstract_nav_smv: str) -> None:
        """Assert init(u1_stage_0) and next(u1_stage_0) are completely stripped."""
        patcher = AbstractControlPatcher()
        freed = patcher.free_control(abstract_nav_smv, "u1_stage_0")

        # Must not contain init(u1_stage_0)
        assert not re.search(r"\binit\s*\(\s*u1_stage_0\s*\)", freed), (
            "free_control left an init(u1_stage_0) assignment in the SMV!"
        )
        # Must not contain next(u1_stage_0)
        assert not re.search(r"\bnext\s*\(\s*u1_stage_0\s*\)", freed), (
            "free_control left a next(u1_stage_0) assignment in the SMV!"
        )

    def test_stage_1_definition_redirected_to_free_variable(self, abstract_nav_smv: str) -> None:
        """The staged definition u1_stage_1 := ... must point directly to the free variable."""
        patcher = AbstractControlPatcher()
        free_name = patcher.free_name("u1_stage_0")
        freed = patcher.free_control(abstract_nav_smv, "u1_stage_0")

        # The stage definition must be redirected to free_name
        assert re.search(rf"\bu1_stage_1\s*:=\s*{re.escape(free_name)}\s*;", freed), (
            f"u1_stage_1 was not redirected to {free_name}"
        )

        # The old case statement defining u1_stage_1 must be gone
        assert not re.search(r"u1_stage_1\s*:=\s*case", freed), (
            "The old case statement for u1_stage_1 is still present!"
        )

    def test_free_variable_declared_with_matching_type(self, abstract_nav_smv: str) -> None:
        """The newly introduced free variable must be declared with the same type (real)."""
        patcher = AbstractControlPatcher()
        free_name = patcher.free_name("u1_stage_0")
        freed = patcher.free_control(abstract_nav_smv, "u1_stage_0")

        assert re.search(rf"\b{re.escape(free_name)}\s*:\s*real\s*;", freed), (
            f"Declaration for {free_name} : real; not found in VAR section"
        )

    def test_free_control_preserves_other_variables(self, abstract_nav_smv: str) -> None:
        """Freeing u1_stage_0 must leave x, v, and ticks declarations, init, and next intact."""
        patcher = AbstractControlPatcher()
        freed = patcher.free_control(abstract_nav_smv, "u1_stage_0")

        for other_var in ("x_stage_0", "v_stage_0", "ticks_stage_0"):
            assert re.search(rf"\binit\s*\(\s*{other_var}\s*\)", freed), (
                f"init({other_var}) was accidentally corrupted or removed!"
            )
            assert re.search(rf"\bnext\s*\(\s*{other_var}\s*\)", freed), (
                f"next({other_var}) was accidentally corrupted or removed!"
            )


# =============================================================================
# 4. IDEMPOTENCE & ERROR HANDLING
# =============================================================================

class TestIdempotenceAndErrorHandling:
    """
    Patching twice, patching a variable that does not exist, or invalid inputs
    must raise rather than silently produce an invalid or vacuous SMV.
    """

    def test_patching_twice_raises_error(self, abstract_nav_smv: str) -> None:
        """Calling free_control or patch twice on the same model must raise ValueError."""
        patcher = AbstractControlPatcher()
        freed = patcher.free_control(abstract_nav_smv, "u1_stage_0")

        with pytest.raises(ValueError, match="(?i)already freed"):
            patcher.free_control(freed, "u1_stage_0")

    def test_patching_nonexistent_variable_raises_error(self, abstract_nav_smv: str) -> None:
        """Attempting to free a variable not declared in the SMV must raise ValueError."""
        patcher = AbstractControlPatcher()
        with pytest.raises(ValueError, match="(?i)not declared"):
            patcher.free_control(abstract_nav_smv, "nonexistent_var_stage_0")

    def test_contract_constructor_validation(self) -> None:
        """Contract dataclass must validate its fields on construction."""
        # Empty interval (lo > hi)
        with pytest.raises(ValueError, match="empty interval"):
            Contract(region=[], target="u", kind="interval", lo=1.0, hi=0.0)

        # Missing bounds for interval
        with pytest.raises(ValueError, match="needs both lo and hi"):
            Contract(region=[], target="u", kind="interval", lo=1.0, hi=None)

        # Missing forbidden for never_select
        with pytest.raises(ValueError, match="needs 'forbidden'"):
            Contract(region=[], target="u", kind="never_select", forbidden=None)

        # Unknown kind
        with pytest.raises(ValueError, match="unknown contract kind"):
            Contract(region=[], target="u", kind="unsupported_kind")

    def test_inject_contracts_targeting_multiple_variables_raises(self, abstract_nav_smv: str) -> None:
        """inject() expects contracts targeting a single freed control variable."""
        patcher = AbstractControlPatcher()
        c1 = Contract(region=[], target="u1_stage_0", kind="interval", lo=-1.0, hi=1.0)
        c2 = Contract(region=[], target="u2_stage_0", kind="interval", lo=-1.0, hi=1.0)

        with pytest.raises(ValueError, match="multiple variables"):
            patcher.inject(abstract_nav_smv, [c1, c2])

    def test_inject_missing_specifications_marker_raises(self) -> None:
        """inject() must raise if SPEC_MARKER is missing."""
        patcher = AbstractControlPatcher()
        c = Contract(region=[], target="u", kind="interval", lo=0.0, hi=1.0)
        corrupted_smv = "MODULE main\nVAR x : boolean;\n"
        with pytest.raises(ValueError, match="marker not found"):
            patcher.inject(corrupted_smv, [c])


# =============================================================================
# 5. EMITTED SMV FORMS & SYNTAX
# =============================================================================

class TestEmittedForms:
    """Verify exact nuXmv syntax of emitted INVAR lines."""

    def test_interval_with_region(self) -> None:
        c = Contract(
            region=[RegionBound("v_stage_0", -1.0, 1.0)],
            target="u1",
            kind="interval",
            lo=-1.0,
            hi=1.0,
        )
        line = c.to_invar("u1_free")
        expected = "INVAR (system.v_stage_0 >= -1.0 & system.v_stage_0 <= 1.0) -> (system.u1_free >= -1.0 & system.u1_free <= 1.0);"
        assert line == expected

    def test_interval_everywhere(self) -> None:
        c = Contract(
            region=[],
            target="u1",
            kind="interval",
            lo=0.0,
            hi=2.0,
        )
        line = c.to_invar("u1_free")
        expected = "INVAR (system.u1_free >= 0.0 & system.u1_free <= 2.0);"
        assert line == expected

    def test_never_select_with_region(self) -> None:
        c = Contract(
            region=[RegionBound("v_stage_0", 1.0, 2.0)],
            target="u1",
            kind="never_select",
            forbidden="stop",
        )
        line = c.to_invar("u1_free")
        expected = "INVAR (system.v_stage_0 >= 1.0 & system.v_stage_0 <= 2.0) -> system.u1_free != stop;"
        assert line == expected


# =============================================================================
# 6. NON-VACUITY END TO END (nuXmv)
# =============================================================================

class TestEndToEndNonVacuity:
    """
    End-to-end nuXmv execution:
    1. Verify with contracts -> expect TRUE.
    2. Strip contracts (control unconstrained) -> expect FALSE with counterexample.
    3. If stripping does not flip verdict, contracts were not load-bearing and test fails.
    """

    def test_abstract_nav_contracts_are_load_bearing(self, abstract_nav_smv: str) -> None:
        """
        Execute nuXmv on abstract_nav:
        With contracts: INVARSPEC (v >= -10 & v <= 10) is true
        Without contracts: INVARSPEC (v >= -10 & v <= 10) is false
        """
        patcher = AbstractControlPatcher()

        # The 3 contracts from COLLAB/experiments/no_table/README.md
        # When |v| <= 1.0: u in [-1.0, 1.0]
        # When v > 1.0: u in [-1.0, 0.0] (brake / decelerate)
        # When v < -1.0: u in [0.0, 1.0] (accelerate forward)
        contracts = [
            Contract(
                region=[RegionBound("v_stage_0", -1.0, 1.0)],
                target="u1_stage_0",
                kind="interval",
                lo=-1.0,
                hi=1.0,
            ),
            Contract(
                region=[RegionBound("v_stage_0", 1.0, 100.0)],
                target="u1_stage_0",
                kind="interval",
                lo=-1.0,
                hi=0.0,
            ),
            Contract(
                region=[RegionBound("v_stage_0", -100.0, -1.0)],
                target="u1_stage_0",
                kind="interval",
                lo=0.0,
                hi=1.0,
            ),
        ]

        # 1. Generate model WITH contracts
        patched_smv = patcher.patch(abstract_nav_smv, "u1_stage_0", contracts)

        # 2. Run nuXmv on model WITH contracts
        ret_with, out_with, err_with = run_nuxmv_interactive(patched_smv)
        assert ret_with == 0, f"nuXmv exited with {ret_with}:\n{err_with}"
        assert "-- invariant (system.v_stage_1 >= -10.0 & system.v_stage_1 <= 10.0)  is true" in out_with, (
            f"Expected invariant to be TRUE with contracts, but got:\n{out_with}"
        )

        # 3. Generate model WITHOUT contracts (only freeing the control)
        freed_only_smv = patcher.free_control(abstract_nav_smv, "u1_stage_0")

        # 4. Run nuXmv on model WITHOUT contracts
        ret_without, out_without, err_without = run_nuxmv_interactive(freed_only_smv)
        assert ret_without == 0, f"nuXmv exited with {ret_without}:\n{err_without}"
        assert "-- invariant (system.v_stage_1 >= -10.0 & system.v_stage_1 <= 10.0)  is false" in out_without, (
            f"Expected invariant to be FALSE without contracts, but got:\n{out_without}"
        )
        assert "Trace Description: IC3 smt counterexample" in out_without or "Trace Type: Counterexample" in out_without, (
            "nuXmv reported false but did not provide counterexample trace"
        )

        # 5. Confirm verdict flipped: Non-vacuity confirmed!
        assert ("is true" in out_with) and ("is false" in out_without), (
            "Non-vacuity failure: contracts did not flip the verification verdict!"
        )


# =============================================================================
# 7. PROGRESS & NON-DEADLOCK GUARDS (nuXmv)
# =============================================================================

class TestProgressAndNonDeadlockGuard:
    """
    An unsatisfiable INVAR or deadlocked transition relation truncates paths.
    In nuXmv, truncated paths vacuously satisfy implications like `ticks = 30 -> Goal`
    because the antecedent `ticks = 30` is never reached.
    The progress/non-deadlock guard ensures:
    1. Initial state is non-empty (`INVARSPEC FALSE` returns `is false`).
    2. Horizon is reachable without deadlock (`INVARSPEC !(ticks = 30)` returns `is false`).
    """

    def test_set_invarspec_preserves_contracts(self) -> None:
        smv = (
            "MODULE main\n"
            "--------------SPECIFICATIONS\n"
            "INVAR system.v >= 0.0;\n"
            "INVARSPEC system.v <= 10.0;\n"
            "--------------END OF SPECIFICATIONS\n"
        )
        res = set_invarspec(smv, "system.ticks != 30")
        assert "INVAR system.v >= 0.0;" in res
        assert "INVARSPEC system.ticks != 30;" in res
        assert "system.v <= 10.0" not in res

    def test_progress_guard_passes_on_valid_abstract_nav(self, abstract_nav_smv: str) -> None:
        patcher = AbstractControlPatcher()
        contracts = [
            Contract(
                region=[RegionBound("v_stage_0", -1.0, 1.0)],
                target="u1_stage_0",
                kind="interval",
                lo=-1.0,
                hi=1.0,
            ),
            Contract(
                region=[RegionBound("v_stage_0", 1.0, 100.0)],
                target="u1_stage_0",
                kind="interval",
                lo=-1.0,
                hi=0.0,
            ),
            Contract(
                region=[RegionBound("v_stage_0", -100.0, -1.0)],
                target="u1_stage_0",
                kind="interval",
                lo=0.0,
                hi=1.0,
            ),
        ]
        patched_smv = patcher.patch(abstract_nav_smv, "u1_stage_0", contracts)
        success, msg = check_nuxmv_progress(
            patched_smv, "system.ticks_stage_0", 30, nuxmv_bin=_NUXMV_BIN
        )
        assert success
        assert "reachable" in msg

    def test_progress_guard_detects_empty_initial_state(self, abstract_nav_smv: str) -> None:
        patcher = AbstractControlPatcher()
        freed = patcher.free_control(abstract_nav_smv, "u1_stage_0")
        # Make init unsatisfiable
        bad_smv = freed.replace("MODULE main", "MODULE main\nINVAR system.v_stage_0 > 100.0;\n")

        with pytest.raises(VacuityError, match="(?i)initial state is empty"):
            check_nuxmv_progress(bad_smv, "system.ticks_stage_0", 30, nuxmv_bin=_NUXMV_BIN)

    def test_progress_guard_detects_deadlock_before_horizon(self, abstract_nav_smv: str) -> None:
        patcher = AbstractControlPatcher()
        contracts = [
            Contract(
                region=[RegionBound("v_stage_0", -1.0, 1.0)],
                target="u1_stage_0",
                kind="interval",
                lo=-1.0,
                hi=1.0,
            ),
        ]
        patched = patcher.patch(abstract_nav_smv, "u1_stage_0", contracts)
        # Artificially deadlock after step 1: require v > 10.0 which cannot be reached in 1 step
        deadlocked_smv = patched.replace(
            "MODULE main",
            "MODULE main\nINVAR (system.ticks_stage_0 >= 1) -> (system.v_stage_0 > 10.0);\n"
        )

        with pytest.raises(DeadlockError, match="(?i)unreachable|deadlock"):
            check_nuxmv_progress(deadlocked_smv, "system.ticks_stage_0", 30, nuxmv_bin=_NUXMV_BIN)

    def test_mechanically_demonstrate_deadlock_vacuity_trap(self, abstract_nav_smv: str) -> None:
        """
        Adversarial proof:
        Show that without the progress guard, a deadlocked model proves a completely
        FALSE claim (at ticks=30, x is 9999.0) as TRUE in nuXmv, but check_nuxmv_progress catches it.
        """
        patcher = AbstractControlPatcher()
        contracts = [
            Contract(
                region=[RegionBound("v_stage_0", -1.0, 1.0)],
                target="u1_stage_0",
                kind="interval",
                lo=-1.0,
                hi=1.0,
            ),
        ]
        patched = patcher.patch(abstract_nav_smv, "u1_stage_0", contracts)

        # Add artificial deadlock at step 1: require v > 10.0 which cannot be reached in 1 step from v=0 with |u1|<=1
        deadlocked_smv = patched.replace(
            "MODULE main",
            "MODULE main\nINVAR (system.ticks_stage_0 >= 1) -> (system.v_stage_0 > 10.0);\n"
        )

        # Put a completely false claim about the horizon state
        false_spec_smv = set_invarspec(deadlocked_smv, "(system.ticks_stage_0 = 30) -> (system.x_stage_0 = 9999.0)")

        ret, out, err = run_nuxmv_interactive(false_spec_smv)
        assert ret == 0
        # Demonstrating the trap: nuXmv says it is TRUE!
        assert "is true" in out, "Vacuity trap demonstration failed: nuXmv did not report is true on deadlocked implication!"

        # Now verify that our progress guard blocks this model!
        with pytest.raises(DeadlockError):
            check_nuxmv_progress(false_spec_smv, "system.ticks_stage_0", 30, nuxmv_bin=_NUXMV_BIN)


# =============================================================================
# 8. VALIDATION BATTERY (Phase 5 automated test suite)
# =============================================================================

class TestValidationBattery:
    """
    Phase 5 validation battery: every trusted verdict must pass:
    1. consistency guard
    2. progress/non-deadlock guard
    3. strip-contracts-must-fail
    4. drop-one-contract-must-fail
    5. trajectory-admission check
    """

    @pytest.fixture
    def valid_contracts(self) -> list[Contract]:
        return [
            Contract(
                region=[RegionBound("v_stage_0", -1.0, 1.0)],
                target="u1_stage_0",
                kind="interval",
                lo=-1.0,
                hi=1.0,
            ),
            Contract(
                region=[RegionBound("v_stage_0", 1.0, 100.0)],
                target="u1_stage_0",
                kind="interval",
                lo=-1.0,
                hi=0.0,
            ),
            Contract(
                region=[RegionBound("v_stage_0", -100.0, -1.0)],
                target="u1_stage_0",
                kind="interval",
                lo=0.0,
                hi=1.0,
            ),
        ]

    def test_battery_passes_on_sound_model(self, abstract_nav_smv: str, valid_contracts: list[Contract]) -> None:
        results = verify_with_battery(
            unpatched_smv=abstract_nav_smv,
            control_var="u1_stage_0",
            contracts=valid_contracts,
            horizon_var="system.ticks_stage_0",
            horizon_val=30,
            nuxmv_bin=_NUXMV_BIN,
        )
        assert results["consistency_guard"] is True
        assert results["progress_guard"] is True
        assert results["strip_contracts_fails"] is True
        assert results["drop_one_contract_fails"] is True
        assert results["specification_verified"] is True

    def test_battery_trajectory_admission_check(self, abstract_nav_smv: str, valid_contracts: list[Contract]) -> None:
        # Trajectory that stays within contract bounds
        valid_traj = [
            {"v_stage_0": 0.0, "u1_stage_0": 0.0},
            {"v_stage_0": 1.5, "u1_stage_0": -0.5},
            {"v_stage_0": -1.5, "u1_stage_0": 0.5},
        ]
        res = verify_with_battery(
            unpatched_smv=abstract_nav_smv,
            control_var="u1_stage_0",
            contracts=valid_contracts,
            horizon_var="system.ticks_stage_0",
            horizon_val=30,
            nuxmv_bin=_NUXMV_BIN,
            reference_trajectory=valid_traj,
        )
        assert res["trajectory_admission"] is True

        # Trajectory violating contract bounds: at v=1.5 (region 2, requires u <= 0.0), control is +0.5
        invalid_traj = [
            {"v_stage_0": 1.5, "u1_stage_0": 0.5},
        ]
        with pytest.raises(ValueError, match="(?i)trajectory admission failure"):
            verify_with_battery(
                unpatched_smv=abstract_nav_smv,
                control_var="u1_stage_0",
                contracts=valid_contracts,
                horizon_var="system.ticks_stage_0",
                horizon_val=30,
                nuxmv_bin=_NUXMV_BIN,
                reference_trajectory=invalid_traj,
            )

    def test_battery_blocks_inconsistent_contracts(self, abstract_nav_smv: str) -> None:
        contradictory = [
            Contract(region=[RegionBound("v_stage_0", -1.0, 1.0)], target="u1_stage_0", kind="interval", lo=0.0, hi=0.5),
            Contract(region=[RegionBound("v_stage_0", 0.0, 2.0)], target="u1_stage_0", kind="interval", lo=0.6, hi=1.0),
        ]
        with pytest.raises(VacuityError, match="(?i)inconsistent contracts"):
            verify_with_battery(
                unpatched_smv=abstract_nav_smv,
                control_var="u1_stage_0",
                contracts=contradictory,
                horizon_var="system.ticks_stage_0",
                horizon_val=30,
                nuxmv_bin=_NUXMV_BIN,
            )

    def test_battery_blocks_deadlocked_model(self, abstract_nav_smv: str, valid_contracts: list[Contract]) -> None:
        deadlocked = abstract_nav_smv.replace(
            "MODULE main",
            "MODULE main\nINVAR (system.ticks_stage_0 >= 1) -> (system.v_stage_0 > 10.0);\n"
        )
        with pytest.raises(DeadlockError):
            verify_with_battery(
                unpatched_smv=deadlocked,
                control_var="u1_stage_0",
                contracts=valid_contracts,
                horizon_var="system.ticks_stage_0",
                horizon_val=30,
                nuxmv_bin=_NUXMV_BIN,
            )




class TestAffineContracts:
    """
    The affine kind keeps CROWN's linear relaxation instead of concretizing it.

    The soundness-critical part is the rounding pad: rounding a COEFFICIENT
    can break the bound at some point of the region, because the error is
    multiplied by the state value rather than standing alone.
    """

    @staticmethod
    def _region():
        return [RegionBound("system.v", -2.0, -1.5),
                RegionBound("system.th", 0.0, 0.5)]

    def test_affine_renders_linear_terms(self):
        contract = Contract(region=self._region(), target="u1", kind="affine",
                            lo=-0.9, hi=-0.6,
                            coeffs=(("v", 0.2), ("th", -0.04)))
        invar = contract.to_invar("u1")
        assert "0.2 * system.v" in invar
        assert "-0.04 * system.th" in invar
        assert invar.count("system.u1") == 2

    def test_affine_requires_coeffs(self):
        with pytest.raises(ValueError, match="needs 'coeffs'"):
            Contract(region=self._region(), target="u1", kind="affine",
                     lo=0.0, hi=1.0)

    def test_affine_rejects_empty_interval(self):
        with pytest.raises(ValueError, match="empty interval"):
            Contract(region=self._region(), target="u1", kind="affine",
                     lo=1.0, hi=0.0, coeffs=(("v", 0.2),))

    def test_affine_rejects_unbounded_coefficient(self):
        """A coefficient on a variable the region does not bound has no
        finite extent, so no sound pad can be computed for it."""
        with pytest.raises(ValueError, match="does not bound them"):
            Contract(region=self._region(), target="u1", kind="affine",
                     lo=0.0, hi=1.0, coeffs=(("x", 1.0),))

    def test_rounding_pad_widens_the_bound(self):
        """A coefficient needing rounding must push the constants OUTWARD."""
        exact = Contract(region=self._region(), target="u1", kind="affine",
                         lo=-0.5, hi=0.5, coeffs=(("v", 0.25),))
        rounded = Contract(region=self._region(), target="u1", kind="affine",
                           lo=-0.5, hi=0.5,
                           coeffs=(("v", 0.2500000123456789),))
        assert rounded._rounding_pad() > exact._rounding_pad() == 0.0

    def test_mechanically_demonstrate_naive_affine_rounding_is_unsound(self):
        """
        Without the pad, a rounded coefficient can EXCLUDE a real behaviour.

        Take a true lower bound f(v) >= c*v with c and v negative, so that
        c*v is positive and large.  Rounding c toward zero raises the claimed
        lower bound above the true value: the contract now forbids something
        the system actually does.
        """
        coef = 0.2500004999
        region = [RegionBound("system.v", -2.0, -2.0)]
        contract = Contract(region=region, target="u1", kind="affine",
                            lo=0.0, hi=1.0, coeffs=(("v", coef),))
        pad = contract._rounding_pad()
        naive = round(coef, DECIMALS) * -2.0          # what the SMV computes
        exact = coef * -2.0                           # the certified bound
        assert naive > exact, "this coefficient must round the wrong way"
        assert pad >= naive - exact, "the pad must cover the whole error"

    def test_affine_and_interval_agree_when_coefficients_vanish(self):
        """With zero coefficients an affine contract degenerates to interval."""
        region = self._region()
        affine = Contract(region=region, target="u1", kind="affine",
                          lo=-0.9, hi=-0.6, coeffs=(("v", 0.0),))
        interval = Contract(region=region, target="u1", kind="interval",
                            lo=-0.9, hi=-0.6)
        assert affine.guarantee_smv("u1") == interval.guarantee_smv("u1")
