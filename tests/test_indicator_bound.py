"""The mean of a 0/1 indicator reaches its bound honestly; a capped continuous quantity is still a clamp.

A live quest's coverage proportion was 1 in settings where every trial's indicator was 1 (the script wrote
``1.0 if inside else 0.0``), and the clamp check read that conditional as a cap at 1.
"""
from __future__ import annotations

from core import plausibility as pl
from core.engine import _assertion_violations, _assertion_warnings, _indicator_paths

# the interval helper elsewhere in the study caps at 1; the indicator itself is a label
INDICATOR_CODE = "out['hit_a'] = 1.0 if lo <= truth <= hi else 0.0\nhi_w = min(1.0, centre + half)\n"
CAPPED_CODE = "out['hit_a'] = min(1.0, raw)\n"
DESIGN = {"result_assertions": [{"path": "hit_a", "min": 0.0, "max": 1.0, "unit": "x"}]}


def _result(values: list[float | None], mean: float) -> dict:
    return {"cell": {"hit_a": mean, "hit_a_values": values}}


def _state(code: str, result: dict, kind: str | None) -> dict:
    metric = {"id": "hit_a", **({"kind": kind} if kind else {})}
    return {"design": {**DESIGN, "protocol": {"metrics": [metric]}}, "result_json": result, "code": code}


def test_a_proportion_declared_by_the_plan_is_not_a_clamp_at_its_bound() -> None:
    state = _state(INDICATOR_CODE, {"cell": {"hit_a": 1.0}}, "proportion")
    assert _indicator_paths(state) == {"hit_a"}
    assert _assertion_violations(state) == []
    assert _assertion_warnings(state) == []


def test_the_same_result_without_a_kind_or_trial_values_is_still_a_clamp() -> None:
    state = _state(INDICATOR_CODE, {"cell": {"hit_a": 1.0}}, None)
    assert _indicator_paths(state) == set()
    assert [v.kind for v in _assertion_violations(state)] == ["clamped"]


def test_trial_values_that_are_all_zero_or_one_show_an_indicator() -> None:
    result = _result([1.0, 1.0, None, 1.0], 1.0)
    assert pl.check_design(result, DESIGN, code=INDICATOR_CODE) == []


def test_trial_values_that_are_not_all_zero_or_one_leave_the_cap_flagged() -> None:
    result = _result([1.0, 0.7, 1.0], 1.0)
    found = pl.check_design(result, DESIGN, code=CAPPED_CODE)
    assert [v.kind for v in found] == ["clamped"]


def test_a_continuous_result_capped_in_code_is_flagged_even_when_declared_as_a_mean() -> None:
    state = _state(CAPPED_CODE, {"cell": {"hit_a": 1.0}}, "mean")
    assert [v.kind for v in _assertion_violations(state)] == ["clamped"]


def test_a_declared_proportion_whose_trial_values_are_not_zero_or_one_is_still_flagged() -> None:
    state = _state(CAPPED_CODE, _result([1.0, 0.4], 1.0), "proportion")
    assert [v.kind for v in _assertion_violations(state)] == ["clamped"]


def test_an_indicator_outside_its_range_is_still_out_of_range() -> None:
    result = _result([1.0, 1.0], 1.5)
    assert [v.kind for v in pl.check_design(result, DESIGN, indicator_ok={"hit_a"})] == ["out_of_range"]
