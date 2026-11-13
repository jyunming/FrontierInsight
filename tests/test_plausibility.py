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
    d = _design({"path": "gain", "min": 1.0})
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


def test_an_undeclared_bound_hit_is_disclosed_and_does_not_reach_the_repair_gate() -> None:
    """Old code: a quantity on its bound in several settings, with no declared role, was sent back for repair."""
    from core.engine import _assertion_warnings

    state = {"design": _UNIT, "result_json": _sweep(a=0.0, b=0.0)}
    assert _assertion_violations(state) == []
    said = _assertion_warnings(state)
    assert [v.kind for v in said] == ["at_bound"]
    sentence = said[0].disclosure()
    assert "final_size equals its bound 0 in 2 settings" in sentence and "did not say" in sentence


def test_a_declared_bound_still_stops_when_a_value_leaves_it() -> None:
    """The known positive: a quantity the plan declares as a physical bound is still caught outside its range."""
    from core.engine import _assertion_violations as violations

    design = {"result_assertions": [{"path": "final_size", "min": 0, "max": 1}],
              "protocol": {"metrics": [{"id": "final_size", "role": "physical_bound"}]}}
    state = {"design": design, "result_json": _sweep(a=0.0, b=0.0, c=1.5), "code": ""}
    assert [v.kind for v in violations(state)] == ["out_of_range"]


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


_GAIN = _design({"path": "gain", "min": 0.5, "max": 5})


def test_a_bounded_quantity_reported_as_a_list_is_there_and_each_item_is_checked() -> None:
    """A real run first reported a bounded quantity as one number, then as a list of one number per setting: the gate
    called it no longer in the results and sent the run to a repair, and never checked the listed values."""
    seen = pl.bounded_paths({"gain": 1.2}, _GAIN)
    assert seen == ["gain"]
    assert pl.check_design({"gain": [0.8, 3.3, 3.6]}, _GAIN, seen=seen) == []
    assert pl.check_design({"by": {"a": {"gain": [[1.0, 2.0], [4.0]]}}}, _GAIN, seen=seen) == []
    v = pl.check_design({"gain": [0.8, 7.0, 3.6]}, _GAIN, seen=seen)
    assert [(x.kind, x.path, x.value) for x in v] == [("out_of_range", "gain[1]", 7.0)]
    assert pl.bounded_paths({"gain": [1.0, 2.0]}, _GAIN) == ["gain"]


def test_a_list_of_records_still_names_each_record_s_own_keys() -> None:
    assert pl.check_design({"sweep": [{"gain": 1.0}, {"gain": 9.0}]}, _GAIN) != []
    assert pl.check_design({"gain_by_setting": [9.0, 9.0]}, _GAIN) == []
    assert pl.check_design([9.0, 9.0], _GAIN) == []


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


def test_a_failure_count_the_protocol_names_may_be_zero_everywhere() -> None:
    """A live quest's failed_run_count was 0 in all nine settings and its manifest listed no failed trial; the
    at-bound rule sent the correct script back twice."""
    from core.engine import _assertion_warnings, _zero_expected_paths

    design = {
        "result_assertions": [{"path": "failed_run_count", "min": 0, "max": 300},
                              {"path": "final_size", "min": 0, "max": 1}],
        "protocol": {"failure_policy": "Any solver failure is recorded in the 'failed_run_count' metric."},
    }
    result = {"by": {k: {"failed_run_count": 0, "final_size": 0.0} for k in ("a", "b", "c")}}
    state = {"design": design, "result_json": result, "manifest_failed_trials": 0}
    assert _zero_expected_paths(state) == {"failed_run_count"}
    kinds = sorted((v.kind, v.path) for v in _assertion_warnings(state))
    assert kinds == [("at_bound", "final_size")], "only the failure count is exempt; a final size of 0 is disclosed"
    # A manifest that lists failures contradicts the zeros; no failure policy means nothing names the count.
    assert _zero_expected_paths({**state, "manifest_failed_trials": 5}) == set()
    assert _zero_expected_paths({**state, "design": {**design, "protocol": {}}}) == set()
    # A name that is only a part of the one the policy names is not named by it.
    partial = {**design, "result_assertions": [{"path": "run_count", "min": 0, "max": 300}]}
    assert _zero_expected_paths({**state, "design": partial}) == set()
    # No manifest read means nothing corroborates the zero.
    assert _zero_expected_paths({**state, "manifest_failed_trials": None}) == set()
    # A bare word in the policy's prose names nothing; the same word quoted as a name does.
    rate = {"result_assertions": [{"path": "rate", "min": 0, "max": 1}]}
    prose = {**rate, "protocol": {"failure_policy": "Our failure rate is expected to be 0."}}
    named = {**rate, "protocol": {"failure_policy": "Failures are counted in `rate`."}}
    assert _zero_expected_paths({**state, "design": prose}) == set()
    assert _zero_expected_paths({**state, "design": named}) == {"rate"}


def test_the_path_index_picks_the_leaves_matches_picks_in_result_order() -> None:
    from core.plausibility import _matches, _PathIndex

    paths = ["cd_nm", "sweep[0].cd_nm", "sweep[1].cd_nm", "a.b.cd_nm", "cd_nm_x", "sweep[2]", "x.cd", "LER_values[3]",
             "by.cd_nm.y", "<root>", "a.b"]
    index = _PathIndex(paths)
    for a in ["cd_nm", "cd_nm.*", "sweep[0].cd_nm", "a.b", "a.b.*", "cd", "y", "LER_values", "missing", "sweep[2]"]:
        assert index.matching(a) == [i for i, p in enumerate(paths) if _matches(a, p)], a
        assert index.any_match(a) == any(_matches(a, p) for p in paths), a


def test_a_result_with_a_million_values_is_checked_in_seconds() -> None:
    import time

    from core.plausibility import Assertion, violations

    result = {"mean_ler": 2.0, "ler_values": [1.0] * 500_000, "cells": [{"ler": 1.0, "p": 0.1} for _ in range(100_000)]}
    assertions = [Assertion(path=f"q{i}", min=0.0, max=1.0, unit="", reason="") for i in range(30)]
    assertions.append(Assertion(path="ler", min=0.0, max=0.5, unit="", reason=""))
    started = time.monotonic()
    found = violations(result, assertions)
    assert time.monotonic() - started < 30
    assert len([v for v in found if v.assertion.path == "ler"]) == 100_000


# --- one walk per result ---------------------------------------------------------------------------------------------

def _reference_dotted(tokens):
    out = ""
    for kind, val in tokens:
        out = f"{out}[{val}]" if kind == "i" else (f"{out}.{val}" if out else str(val))
    return out or "<root>"


def _reference_leaves(obj, tokens=()):
    """The recursive walk the iterative one replaced, kept here to pin that it finds the same leaves."""
    import math

    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            yield tokens, _reference_dotted(tokens), float(obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _reference_leaves(v, tokens + (("k", str(k)),))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _reference_leaves(v, tokens + (("i", i),))


def _reference_present(obj, tokens=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _reference_present(v, tokens + (("k", str(k)),))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _reference_present(v, tokens + (("i", i),))
    else:
        yield _reference_dotted(tokens)


_SHAPES = [
    4.2, "text", None, [], {}, [1.0, 2.0], (1, (2, 3)),
    {"a": {"b": [1.5, {"c": 2.5}, 0.0]}, "d": 3.5, "e": {"0.9": {"f": -1.0}}, "g": True, "h": None, "s": "x"},
    {"": 1.0, "k": {"": 2.0, "x": [{"": 3.0}]}, "a.b": 4.0, "c[0]": 5.0, "\n": 6.0},
    {"by_R0": {"0.9": {"by_N": {"100": [0.1, float("nan"), float("inf"), 2]}}}, "flag": [True, False]},
    {"cells": [{"ler": 0.1, "p": 0.1}, {"ler": 0.2, "p": "n/a"}, [[1.0], [2.0, [3.0]]]], "ler": 9.0},
    {"a": {"b": {"c": {"d": {"e": 1.0}}}}, "z": [[], {}, [None]]},
]


@pytest.mark.parametrize("shape", _SHAPES, ids=range(len(_SHAPES)))
def test_the_iterative_walk_finds_the_leaves_the_recursive_walk_found(shape) -> None:
    from core.plausibility import LeafWalk, _PathIndex, _walk_leaves

    assert list(_walk_leaves(shape)) == list(_reference_leaves(shape))
    walk = LeafWalk(shape)
    assert walk.count == len(list(_reference_leaves(shape)))
    assert walk.present_paths() == list(_reference_present(shape))
    # The indexes built from the walk (names read from the keys) pick what an index built path by path picks.
    ref_paths = [p for _t, p, _v in _reference_leaves(shape)]
    ref_index = _PathIndex(ref_paths)
    for a in ["a", "d", "c", "ler", "p", "by_R0", "", "b", "x", "k", "<root>", "a.b", "c[0]", "e.*", "\n", "h", "g"]:
        assert walk.numeric_index().matching(a) == ref_index.matching(a), a
        assert walk.numeric_index().any_match(a) == ref_index.any_match(a), a
    present = _PathIndex(list(_reference_present(shape)))
    for a in ["a", "h", "s", "g", "z", "flag", "p"]:
        assert walk.present_index().matching(a) == present.matching(a), a


def test_violations_and_check_design_are_the_same_with_a_shared_walk() -> None:
    from core.plausibility import Assertion, LeafWalk, bounded_paths, check_design, violations

    result = {"cells": [{"ler": 0.1 * i, "p": 0.0} for i in range(30)], "ler": 9.0, "gone": None, "t": {"p": 0.0}}
    design = {"result_assertions": [
        {"path": "ler", "min": 0, "max": 1}, {"path": "p", "min": 0, "max": 1}, {"path": "gone", "min": 0, "max": 1},
        {"path": "nowhere", "min": 0, "max": 1}]}
    fresh = check_design(result, design, seen=["nowhere", "gone"])
    walk = LeafWalk(result)
    shared = check_design(result, design, seen=["nowhere", "gone"], walk=walk)
    assert [(v.path, v.kind, v.value) for v in shared if v.value == v.value] == [
        (v.path, v.kind, v.value) for v in fresh if v.value == v.value]
    assert [v.kind for v in shared] == [v.kind for v in fresh]
    assert bounded_paths(result, design, walk=walk) == bounded_paths(result, design)
    assertions = [Assertion(path="ler", min=0.0, max=0.5, unit="", reason="")]
    assert violations(result, assertions, walk=walk) == violations(result, assertions)
    # A walk made for another result is not used for this one.
    other = {"ler": 5.0}
    assert [v.value for v in violations(other, assertions, walk=walk)] == [5.0]


def test_a_result_read_by_every_check_is_walked_once(monkeypatch) -> None:
    from core import plausibility as pl

    built = []
    original = pl.LeafWalk._build

    def counting(self):
        if not self._built:
            built.append(id(self.root))
        original(self)

    monkeypatch.setattr(pl.LeafWalk, "_build", counting)
    result = {"ler_values": [0.3] * 1000, "mean": 0.5, "note": "x"}
    design = {"result_assertions": [{"path": "mean", "min": 0, "max": 1}, {"path": "ler_values", "min": 0, "max": 1}]}
    memo = pl.WalkMemo()
    pl.check_design(result, design, seen=["mean"], walk=memo.get(result))
    pl.check_design(result, design, seen=["mean"], walk=memo.get(result))
    pl.bounded_paths(result, design, walk=memo.get(result))
    assert len(built) == 1
    # Growing the result, or handing over another one, walks again.
    result["extra"] = [1.0]
    pl.bounded_paths(result, design, walk=memo.get(result))
    pl.bounded_paths({"mean": 1.0}, design, walk=memo.get({"mean": 1.0}))
    assert len(built) == 3
    # Without a walk each call still works (and walks for itself).
    pl.bounded_paths(result, design)
    assert len(built) == 4


def test_two_million_values_are_walked_once_and_quickly(monkeypatch) -> None:
    import time

    from core import plausibility as pl

    walked = []  # the size of each result a walk is built for (not the walk: that would keep every result alive)
    original = pl.LeafWalk._build

    def counting(self):
        if not self._built:
            walked.append(len(self.root["trials"]))
        original(self)

    monkeypatch.setattr(pl.LeafWalk, "_build", counting)
    design = {"result_assertions": [{"path": "mean", "min": 0, "max": 1}, {"path": "ler", "min": 0, "max": 0.5},
                                    {"path": "gone", "min": 0, "max": 1}]}

    def first_check(n: int) -> tuple[float, dict, pl.LeafWalk]:
        result = {"mean": 0.5, "trials": [{"ler": 0.1, "vals": [0.2] * 4} for _ in range(n // 5)]}
        walk = pl.LeafWalk(result)
        started = time.perf_counter()
        assert pl.check_design(result, design, seen=["gone"], walk=walk)[0].kind == "missing"
        return time.perf_counter() - started, result, walk

    before = first_check(200_000)[0]
    first, result, walk = first_check(2_000_000)
    after = first_check(200_000)[0]
    for _ in range(3):
        pl.check_design(result, design, seen=["gone"], walk=walk)
    assert walked.count(len(result["trials"])) == 1  # a repeat reads the kept walk; nothing walks that result again
    # Quickly, as a machine of any speed can be held to it: ten times the values take about ten times as long (13 times
    # where this was written), against the slower of two checks of a tenth of the values, one made just before and one
    # just after. The larger result checked ten times over, which is what a cost growing with the square of the count
    # comes to at these two sizes, took about a hundred times. What this does not hold is the number of seconds: a limit
    # of 30 s failed on a shared machine whose time for the first check moved between runs of unchanged code (34 and 36
    # s in the two runs that failed).
    small = max(before, after)
    assert first < 40 * small, f"ten times the values took {first / small:.0f} times as long"


# ---- a result the plan declares as physical_bound may sit on its range edge --------------------------------------------

def _edge_state(role: str | None) -> dict:
    metric = {"id": "share", **({"role": role} if role else {})}
    result = {"by_setting": {f"s{i}": {"share": 0.0 if i < 4 else 0.1 * i} for i in range(1, 7)}}
    design = {"result_assertions": [{"path": "share", "min": 0, "max": 1}], "protocol": {"metrics": [metric]}}
    return {"design": design, "result_json": result, "code": ""}


def test_a_physical_bound_result_on_its_edge_in_several_settings_is_kept() -> None:
    from core.engine import _assertion_violations, _physical_bound_paths

    state = _edge_state("physical_bound")
    assert _physical_bound_paths(state) == {"share"}
    assert _assertion_violations(state) == []


def test_the_same_edge_without_the_role_is_disclosed_and_a_value_outside_the_range_never_passes() -> None:
    from core.engine import _assertion_violations, _assertion_warnings, _physical_bound_paths

    state = _edge_state(None)
    assert _physical_bound_paths(state) == set()
    assert _assertion_violations(state) == []
    assert [v.kind for v in _assertion_warnings(state)] == ["at_bound"]
    outside = _edge_state("physical_bound")
    outside["result_json"]["by_setting"]["s6"]["share"] = 1.5
    assert [v.kind for v in _assertion_violations(outside)] == ["out_of_range"]


def test_a_physical_bound_result_on_its_edge_in_every_setting_is_said_not_stopped() -> None:
    from core.engine import _assertion_violations, _assertion_warnings

    state = _edge_state("physical_bound")
    state["result_json"] = {"by_setting": {f"s{i}": {"share": 0.0} for i in range(1, 7)}}
    assert _assertion_violations(state) == []
    said = _assertion_warnings(state)
    assert [v.kind for v in said] == ["edge_everywhere"]
    assert said[0].describe() == "share is on its bound in every setting; the result may carry no information"
    assert _assertion_warnings(_edge_state("physical_bound")) == []  # on the edge in some settings only: silent


def test_the_prose_of_the_failure_policy_is_read_only_when_no_result_declares_a_role() -> None:
    from core.engine import _zero_expected_paths

    design = {"result_assertions": [{"path": "failed_run_count", "min": 0, "max": 100}],
              "protocol": {"failure_policy": "A trial that fails is counted in `failed_run_count`.",
                           "metrics": [{"id": "failed_run_count"}]}}
    state = {"design": design, "manifest_failed_trials": 0}
    assert _zero_expected_paths(state) == {"failed_run_count"}
    with_role = {"design": {**design, "protocol": {**design["protocol"], "metrics": [
        {"id": "failed_run_count", "role": "result"}]}}, "manifest_failed_trials": 0}
    assert _zero_expected_paths(with_role) == set()


def test_a_physical_bound_result_exempts_the_assertion_paths_that_end_in_its_whole_name() -> None:
    from core.engine import _physical_bound_paths

    design = {"result_assertions": [{"path": "share", "min": 0, "max": 1}, {"path": "other_group.share", "min": 0, "max": 1},
                                    {"path": "vote_share", "min": 0, "max": 1}],
              "protocol": {"metrics": [{"id": "share", "role": "physical_bound"}]}}
    assert _physical_bound_paths({"design": design}) == {"share", "other_group.share"}


def test_a_dotted_assertion_path_that_ends_in_the_physical_bound_result_is_exempt() -> None:
    from core.engine import _assertion_violations, _physical_bound_paths

    design = {"result_assertions": [{"path": "by_setting.share", "min": 0, "max": 1},
                                    {"path": "vote_share", "min": 0, "max": 1}],
              "protocol": {"metrics": [{"id": "share", "role": "physical_bound"}]}}
    assert _physical_bound_paths({"design": design}) == {"by_setting.share"}
    state = _edge_state("physical_bound")
    state["design"]["result_assertions"] = [{"path": "by_setting.share", "min": 0, "max": 1}]
    state["result_json"] = {"by_setting": {"share": 0.0, "s2": {"share": 0.0}, "s3": {"share": 0.0}, "s4": {"share": 0.5}}}
    assert _assertion_violations(state) == []


def test_a_role_on_one_result_keeps_the_failure_policy_exemption_of_the_others() -> None:
    from core.engine import _zero_expected_paths

    design = {"result_assertions": [{"path": "failed_run_count", "min": 0, "max": 100}, {"path": "wall", "min": 0, "max": 9e9}],
              "protocol": {"failure_policy": "A trial that fails is counted in `failed_run_count` and in `wall`.",
                           "metrics": [{"id": "wall", "role": "timing"}, {"id": "failed_run_count"}]}}
    assert _zero_expected_paths({"design": design, "manifest_failed_trials": 0}) == {"failed_run_count"}


def test_every_result_on_its_bound_is_named_in_the_limitations_however_many_there_are(tmp_path) -> None:
    """Old code listed three and then 'And N more.', so a result that may be a failed computation could be hidden."""
    import json

    from core import disclosure

    rows = [{"text": f"r{i} equals its bound 0 in 3 settings (a, b, c); r{i} must be [0, 1]. The plan did not say more",
             "short": f"r{i} equals its bound 0 in 3 settings"} for i in range(1, 4)]
    rows.append("share is on its bound in every setting; the result may carry no information")
    rows.append({"text": "r9 equals its bound 1 in 2 settings (x, y); r9 must be [0, 1]. Long detail", "short": "r9 equals its bound 1 in 2 settings"})
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "edge_everywhere.json").write_text(json.dumps({"results": rows}), encoding="utf-8")
    text = disclosure._edge_sentence(json.loads((tmp_path / ".fi" / "edge_everywhere.json").read_text()))
    for name in ("r1 ", "r2 ", "r3 ", "share ", "r9 "):
        assert name in text
    assert "Long detail" not in text and "r9 equals its bound 1 in 2 settings." in text
