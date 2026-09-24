"""Tests for ``core.plausibility`` — bounds the design declared, checked in code.

The gap this fills: ``_is_degenerate_result`` fires only when every metric is
~0, so it catches a run that produced nothing and misses a run that produced
the wrong thing convincingly. A unit error, a factor of two or a sign flip
exits 0, parses fine, and passes every downstream gate because all of them are
a language model reading text.

Two properties matter, and the second as much as the first: a declared bound
must be enforced, and an *undeclared* one must never be invented. Silence has
to mean "nothing was claimed".
"""
from __future__ import annotations

import pytest

from core import plausibility as pl
from core.engine import _assertion_violations


def _design(*assertions: dict) -> dict:
    return {"hypothesis": "h", "result_assertions": list(assertions)}


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def test_value_below_min_is_a_violation() -> None:
    d = _design({"path": "cd_nm", "min": 0, "reason": "a CD is positive"})
    v = pl.check_design({"cd_nm": -3.2}, d)
    assert len(v) == 1
    assert v[0].value == -3.2
    assert "positive" in v[0].describe()


def test_value_above_max_is_a_violation() -> None:
    d = _design({"path": "contrast", "min": 0, "max": 1,
                 "reason": "normalised contrast cannot exceed 1"})
    v = pl.check_design({"contrast": 1.8}, d)
    assert len(v) == 1
    assert "contrast" in v[0].describe()


def test_value_inside_the_band_passes() -> None:
    d = _design({"path": "k1", "min": 0.25, "max": 0.5})
    assert pl.check_design({"k1": 0.31}, d) == []


def test_bounds_are_inclusive() -> None:
    d = _design({"path": "contrast", "min": 0, "max": 1})
    assert pl.check_design({"contrast": 1.0}, d) == []
    assert pl.check_design({"contrast": 0.0}, d) == []


def test_a_value_equal_to_its_bound_to_the_last_bit_is_on_the_bound() -> None:
    """A bound is a number with a precision. A design that worked its maximum
    out as 2/3 declares 0.6666666666666666; a script that computes 1 - 1/3
    produces 0.6666666666666667. An exact ``>`` called that correct answer a
    violation of a bound the message printed as the very same number. A value
    that is really larger still fails."""
    d = _design({"path": "p", "min": 0.0, "max": 0.6666666666666666})
    assert pl.check_design({"p": 0.6666666666666667}, d) == []
    assert [v.kind for v in pl.check_design({"p": 0.67}, d)] == ["out_of_range"]


def test_a_hair_below_a_minimum_is_on_the_minimum_not_under_it() -> None:
    """The same equality on the other side: a drift of -1e-13 against a min of
    0 is where a subtraction of two close numbers lands, and a model told that
    breaks the bound writes a clamp. A real sign error still goes back."""
    d = _design({"path": "energy_drift", "min": 0.0, "max": 1.0})
    assert pl.check_design({"energy_drift": -1e-13}, d) == []
    assert [v.kind for v in pl.check_design({"energy_drift": -1e-6}, d)] == [
        "out_of_range"]


def test_assertion_matches_at_any_nesting_depth() -> None:
    """A design names the metric; it cannot predict how the script nests it."""
    d = _design({"path": "cd_nm", "min": 0})
    v = pl.check_design({"sweep": [{"cd_nm": -1.0}, {"cd_nm": 37.0}]}, d)
    assert len(v) == 1
    assert v[0].path == "sweep[0].cd_nm"


def test_exact_path_also_works() -> None:
    d = _design({"path": "metrics.contrast", "max": 1})
    v = pl.check_design({"metrics": {"contrast": 2.0}}, d)
    assert len(v) == 1 and v[0].path == "metrics.contrast"


# ---------------------------------------------------------------------------
# Silence when nothing was claimed
# ---------------------------------------------------------------------------


def test_no_assertions_means_no_violations() -> None:
    assert pl.check_design({"anything": -999.0}, {"hypothesis": "h"}) == []


def test_unmatched_path_is_not_a_violation() -> None:
    """An assertion about a metric the run never emitted is not a failure —
    it is an assertion with nothing to say."""
    d = _design({"path": "nils", "min": 1.0})
    assert pl.check_design({"contrast": 0.4}, d) == []


@pytest.mark.parametrize("bad", [
    {"path": "", "min": 0},                       # no path
    {"path": "x"},                                # asserts nothing
    {"path": "x", "min": 5, "max": 1},            # inverted bounds
    {"path": "x", "min": "not a number"},         # unparseable
    {"path": "x", "min": float("nan")},           # non-finite
    "not even a dict",
])
def test_malformed_assertions_are_dropped_not_raised(bad) -> None:
    """This is LLM output. A malformed entry must be ignored, never fatal."""
    assert pl.parse_assertions({"result_assertions": [bad]}) == []


def test_non_dict_design_is_handled() -> None:
    assert pl.parse_assertions(None) == []
    assert pl.parse_assertions("design") == []
    assert pl.parse_assertions({"result_assertions": "nope"}) == []


def test_booleans_are_not_measurements() -> None:
    d = _design({"path": "converged", "min": 2})
    assert pl.check_design({"converged": True}, d) == []


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def test_engine_helper_surfaces_violations() -> None:
    state = {
        "result_json": {"cd_nm": -3.2},
        "design": _design({"path": "cd_nm", "min": 0}),
    }
    assert len(_assertion_violations(state)) == 1


def test_engine_helper_is_silent_without_assertions() -> None:
    state = {"result_json": {"cd_nm": -3.2}, "design": {"hypothesis": "h"}}
    assert _assertion_violations(state) == []


def test_engine_helper_never_raises(monkeypatch) -> None:
    """A defect in the checker must not stall a quest."""
    from core import plausibility

    monkeypatch.setattr(
        plausibility, "check_design",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert _assertion_violations({"result_json": {}, "design": {}}) == []


def test_missing_state_keys_are_safe() -> None:
    assert _assertion_violations({}) == []


# ---------------------------------------------------------------------------
# The design prompt must actually ask for this
# ---------------------------------------------------------------------------


def test_design_prompt_requests_assertions_and_warns_against_hypothesis_bounds() -> None:
    """The field is worthless if the design bounds results to what it *expects* —
    the run could then never disagree with the hypothesis."""
    from pathlib import Path

    md = (Path(__file__).resolve().parent.parent / "agents" / "design.md").read_text(
        encoding="utf-8"
    )
    # Collapse whitespace: the guidance is prose and wraps, so asserting on a
    # literal substring would break on a reflow rather than on a real change.
    flat = " ".join(md.split())
    assert "result_assertions" in flat
    assert "never what you *expect* to happen" in flat
    assert "bounding a result to your hypothesis" in flat
    # A range that still admits the trivial answer cannot catch it.
    assert "as tight as that guarantee allows" in flat


# ---------------------------------------------------------------------------
# One quantity exactly on a bound in several settings
# ---------------------------------------------------------------------------


def _sweep(**values: float) -> dict:
    return {"by_R0": {name: {"final_size": v} for name, v in values.items()}}


_UNIT = _design({"path": "final_size", "min": 0, "max": 1})


def test_one_quantity_on_its_bound_in_two_settings_is_a_violation() -> None:
    """From a real quest: a deterministic final size of 0.0 at two reproduction
    numbers, inside the declared [0, 1] and wrong."""
    v = pl.check_design(_sweep(r15=0.0, r30=0.0, r09=0.12), _UNIT)
    assert [x.kind for x in v] == ["at_bound"]
    assert v[0].value == 0.0 and v[0].paths == ("by_R0.r15.final_size", "by_R0.r30.final_size")
    assert "2 settings" in v[0].describe()


def test_tiny_p_values_are_computed_answers_not_a_zero() -> None:
    """From a real quest: one-sided Wilcoxon p-values of 8.5e-28 and smaller (z about 10.9) were reported as
    "equals its bound 0 in 11 settings", and two repairs were spent on a correct script. Only an exact 0 is the
    trivial answer; a tiny value is inside the range and says nothing about the computation."""
    p = _design({"path": "wilcoxon_p_one_sided", "min": 0, "max": 1})
    tiny = {"by_set": {k: {"wilcoxon_p_one_sided": v} for k, v in {"a": 8.5e-28, "b": 3.1e-40, "c": 1.2e-19}.items()}}
    assert pl.check_design(tiny, p) == []
    zeros = {"by_set": {k: {"wilcoxon_p_one_sided": v} for k, v in {"a": 0.0, "b": 0.0, "c": 0.3}.items()}}
    assert [x.kind for x in pl.check_design(zeros, p)] == ["at_bound"], "an exact 0 in several settings is still caught"
    below = {"by_set": {"a": {"wilcoxon_p_one_sided": -1e-13}}}
    assert pl.check_design(below, p) == [], "a hair below 0 is still on the minimum, not under it"


def test_one_setting_on_a_bound_is_ordinary() -> None:
    assert pl.check_design(_sweep(r15=0.0, r30=0.94), _UNIT) == []


def test_the_upper_bound_counts_too() -> None:
    v = pl.check_design(_sweep(a=1.0, b=1.0, c=0.3), _UNIT)
    assert [(x.kind, x.value) for x in v] == [("at_bound", 1.0)]


# A real quest's design declared a residual in [0, 1e-06]; the script solved the
# final-size equation and got exactly 0.0 at six settings. That is a perfect solve.
_TOLERANCE = _design({"path": "final_size_relation_residual", "min": 0, "max": 1e-06})


def _residuals(**values: float) -> dict:
    return {"by_R0": {n: {"final_size_relation_residual": v} for n, v in values.items()}}


def test_zeros_on_a_tolerance_range_are_the_best_answer_not_a_trivial_one() -> None:
    assert pl.check_design(_residuals(a=0.0, b=0.0, c=0.0, d=0.0, e=0.0, f=0.0), _TOLERANCE) == []


def test_a_tolerance_range_still_catches_a_value_outside_it() -> None:
    v = pl.check_design(_residuals(a=0.0, b=0.0, c=0.02), _TOLERANCE)
    assert [(x.kind, x.value) for x in v] == [("out_of_range", 0.02)]


def test_the_same_zeros_on_a_unit_range_are_still_flagged() -> None:
    """The control: the tolerance rule is about the range's scale, not about zero."""
    v = pl.check_design(_sweep(a=0.0, b=0.0, c=0.5), _UNIT)
    assert [x.kind for x in v] == ["at_bound"]
    wide = _design({"path": "final_size", "min": 0, "max": 0.01})
    assert [x.kind for x in pl.check_design(_sweep(a=0.0, b=0.0, c=0.005), wide)] == ["at_bound"]


# A real quest's design declared the branching-process probability in [0, 2/3]
# (max(0, 1 - 1/R0) over R0 in {0.9, 1.5, 3}). The result reached 2/3 at R0 = 3 for
# three population sizes, as designed. The script had copied the bound in as an assert.
_BRANCHING = _design({"path": "branching_process_major_probability", "min": 0, "max": 0.6666666667})
_STRATA = {"strata": [
    {"R0": r0, "N": n, "branching_process_major_probability": max(0.0, 1 - 1 / r0)}
    for r0 in (0.9, 1.5, 3.0) for n in (100, 1000, 5000)
]}
_STRATA["strata"][6:9] = [
    {**s, "branching_process_major_probability": 0.6666666667} for s in _STRATA["strata"][6:9]
]
_CHECKED = "def run():\n    branching = 1 - 1 / 3.0\n    assert 0.0 <= branching <= 0.6666666667\n"
_CAPPED = "def run():\n    return min(1 - 1 / 3.0, 0.6666666667)\n"


def test_a_bound_the_script_only_asserts_is_still_a_bound_the_design_derived() -> None:
    """The implement step copies the declared bounds into the script as asserts. That
    made the derived maximum 'a number the code writes down', and reaching it in three
    settings was sent back for repair."""
    assert pl.check_design(_STRATA, _BRANCHING, code=_CHECKED) == []
    assert pl.check_design(_STRATA, _BRANCHING) == []


def test_a_bound_the_script_uses_as_a_value_is_still_a_number_it_writes_down() -> None:
    """The control: the same number as a cap the code applies is one it can return
    without computing anything, so it is still reported."""
    v = pl.check_design(_STRATA, _BRANCHING, code=_CAPPED)
    assert v and {x.kind for x in v} <= {"at_bound", "clamped"}


def test_numeric_literals_skip_a_number_that_only_an_assert_names() -> None:
    assert 0.6666666667 not in pl.numeric_literals(_CHECKED)
    assert 0.6666666667 in pl.numeric_literals(_CAPPED)
    both = "cap = 0.6666666667\nassert x <= 0.6666666667\n"
    assert 0.6666666667 in pl.numeric_literals(both)


def test_settings_on_different_bounds_do_not_add_up() -> None:
    """One 0 and one 1 are two ordinary values, not a quantity stuck on a bound."""
    assert pl.check_design(_sweep(a=0.0, b=1.0), _UNIT) == []


def test_an_exact_zero_below_a_positive_min_is_out_of_range() -> None:
    """Zeros were dropped before the check, so a design that raised the min
    above 0 to catch a trivial answer could not catch it."""
    d = _design({"path": "final_size", "min": 1e-6, "max": 1})
    v = pl.check_design(_sweep(a=0.0, b=0.5), d)
    assert [(x.kind, x.value) for x in v] == [("out_of_range", 0.0)]


def test_at_bound_reaches_the_repair_gate() -> None:
    state = {"design": _UNIT, "result_json": _sweep(a=0.0, b=0.0)}
    assert [v.kind for v in _assertion_violations(state)] == ["at_bound"]


# --- a bounded quantity may not disappear --------------------------------------------------------------------------

_P = _design({"path": "wilcoxon_p_one_sided", "min": 0, "max": 1})


def test_a_bounded_quantity_a_repair_renamed_is_reported() -> None:
    """A repair that renames (or deletes) a bounded key leaves the gate nothing to check and says nothing about why."""
    first = {"by_set": {"a": {"wilcoxon_p_one_sided": 0.2}}}
    seen = pl.bounded_paths(first, _P)
    assert seen == ["wilcoxon_p_one_sided"]
    renamed = {"by_set": {"a": {"wilcoxon_p_asymptotic_normal": 0.2}}}
    v = pl.check_design(renamed, _P, seen=seen)
    assert [x.kind for x in v] == ["missing"]
    assert "wilcoxon_p_one_sided is no longer in the results" in v[0].describe()
    assert "report it again under the name `wilcoxon_p_one_sided`" in v[0].describe()


def test_a_bounded_quantity_reported_as_null_is_not_missing() -> None:
    """The repair prompt's own answer for a value the method cannot give is null with a flag saying why. Both live
    cases looked like this: a diverging Euler error set to null beside a "diverged: ..." flag, and a p-value set to
    null and reported as its log10."""
    seen = ["wilcoxon_p_one_sided"]
    for honest in ({"by_set": {"a": {"wilcoxon_p_one_sided": None, "diverged": True}}},
                   {"by_set": {"a": {"wilcoxon_p_one_sided": None, "wilcoxon_log10_p_one_sided": -27.07}}},
                   {"by_set": {"a": {"wilcoxon_p_one_sided": float("nan")}}}):
        assert pl.check_design(honest, _P, seen=seen) == []


def test_a_name_that_never_matched_is_not_reported() -> None:
    """A design and a script that named a quantity differently from the first run is common (63 of 295 assertions in
    a sweep of 83 quests); nothing says which key was meant, so it is left alone."""
    assert pl.check_design({"prob_outbreak": 0.4}, _design({"path": "outbreak_probability", "min": 0, "max": 1})) == []
    assert pl.check_design({"prob_outbreak": 0.4}, _design({"path": "outbreak_probability", "min": 0, "max": 1}),
                           seen=[]) == []


def test_the_engine_remembers_what_a_run_reported_and_holds_the_next_to_it() -> None:
    from core.engine import _assertion_violations, _bounded_seen_after

    state = {"design": _P, "result_json": {"wilcoxon_p_one_sided": 0.2}}
    state["bounded_seen"] = _bounded_seen_after(state, state["result_json"])
    assert state["bounded_seen"] == ["wilcoxon_p_one_sided"] and _assertion_violations(state) == []
    later = {**state, "result_json": {"wilcoxon_p_asymptotic_normal": 0.2}}
    later["bounded_seen"] = _bounded_seen_after(later, later["result_json"])
    assert later["bounded_seen"] == ["wilcoxon_p_one_sided"], "what was seen is kept"
    assert [v.kind for v in _assertion_violations(later)] == ["missing"]


def test_a_bound_the_design_itself_dropped_is_not_asked_for() -> None:
    """A redesign that removes an assertion also removes the demand: only the design's current bounds are held to."""
    assert pl.check_design({"other_key": 0.3}, _design({"path": "other_key", "min": 0, "max": 1}),
                           seen=["wilcoxon_p_one_sided"]) == []
