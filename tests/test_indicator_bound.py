"""The mean of a 0/1 indicator reaches its bound honestly; a value the script really capped is still a clamp.

A live quest's coverage proportion was 1 in settings where every trial's indicator was 1. The script wrote the indicator
as ``1.0 if inside else 0.0`` (a choice between two constants, not a cap) and, elsewhere, ``if failed: out["failed"] =
1.0`` (a constant set under another result key). The clamp rule read both as caps at 1. FI now reads what the code says:
neither pins the coverage value. Neither the plan's metric kind nor the result's own trial list is trusted to excuse a
value: a ``min(raw, 1.0)`` cap is flagged whatever they say.
"""
from __future__ import annotations

from core import plausibility as pl
from core.engine import _assertion_violations

INDICATOR_CODE = (
    "out['hit_a'] = 1.0 if lo <= truth <= hi else 0.0\n"
    "if failed:\n"
    "    out['failed'] = 1.0\n"
    "    out['n_valid'] = 0.0\n"
)
CAPPED_CODE = "out['hit_a'] = min(raw, 1.0)\n"
DESIGN = {"result_assertions": [{"path": "hit_a", "min": 0.0, "max": 1.0, "unit": "x"}]}


def _result(values: object, mean: float = 1.0) -> dict:
    return {"cell": {"hit_a": mean, "hit_a_values": values}}


def _state(code: str, result: dict, kind: str | None) -> dict:
    metric = {"id": "hit_a", **({"kind": kind} if kind else {})}
    return {"design": {**DESIGN, "protocol": {"metrics": [metric]}}, "result_json": result, "code": code}


def _kinds(result: dict, code: str) -> list[str]:
    return [v.kind for v in pl.check_design(result, DESIGN, code=code)]


def test_a_choice_between_two_constants_is_not_a_cap_site() -> None:
    assert pl.clamp_constants("x = 1.0 if inside else 0.0\n") == set()


def test_a_conditional_that_keeps_the_value_in_one_branch_is_still_a_cap_site() -> None:
    assert pl.clamp_constants("rmse = 10.0 if diverged else rmse\n") == {10.0}


def test_the_real_shape_indicator_mean_on_its_bound_is_not_flagged_without_any_plan_or_trial_record() -> None:
    assert _kinds({"cell": {"hit_a": 1.0}}, INDICATOR_CODE) == []
    assert _assertion_violations(_state(INDICATOR_CODE, {"cell": {"hit_a": 1.0}}, None)) == []


def test_a_constant_set_under_another_key_does_not_pin_this_value() -> None:
    code = "if failed:\n    out['failed'] = 1.0\n"
    assert pl.clamp_keys(code) == {1.0: {"failed"}}
    assert _kinds({"cell": {"hit_a": 1.0}}, code) == []


def test_the_same_key_set_to_the_bound_inside_an_if_is_still_flagged() -> None:
    code = "if raw > 1.0:\n    out['hit_a'] = 1.0\n"
    assert _kinds({"cell": {"hit_a": 1.0}}, code) == ["clamped"]


def test_a_constant_that_is_also_a_cap_elsewhere_still_pins_every_value() -> None:
    code = "if failed:\n    out['failed'] = 1.0\nout['hit_a'] = min(raw, 1.0)\n"
    assert pl.clamp_keys(code) == {}
    assert _kinds({"cell": {"hit_a": 1.0}}, code) == ["clamped"]


def test_a_real_cap_is_flagged_even_when_the_result_lists_ones() -> None:
    assert _kinds(_result([1, 1]), CAPPED_CODE) == ["clamped"]


def test_a_real_cap_is_flagged_when_the_trial_list_is_zeros() -> None:
    assert _kinds(_result([0, 0]), CAPPED_CODE) == ["clamped"]


def test_a_real_cap_is_flagged_when_the_trial_list_is_not_a_list_or_has_junk() -> None:
    assert _kinds(_result(1), CAPPED_CODE) == ["clamped"]
    assert _kinds(_result([1, "bad", None]), CAPPED_CODE) == ["clamped"]


def test_a_real_cap_is_flagged_whatever_kind_the_plan_gives_the_result() -> None:
    for kind in ("proportion", "indicator", "mean", None):
        state = _state(CAPPED_CODE, {"cell": {"hit_a": 1.0}}, kind)
        assert [v.kind for v in _assertion_violations(state)] == ["clamped"], kind


def test_an_indicator_outside_its_range_is_still_out_of_range() -> None:
    assert _kinds(_result([1.0, 1.0], 1.5), INDICATOR_CODE) == ["out_of_range"]
