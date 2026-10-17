"""An interval printed beside a value is read as that value's interval (replay of suite r9k first_order_kinetics: the
paper printed "0.918 (367/400; 0.886–0.941)", every number the run's own Wilson interval, and the provenance check
called the four endpoints untraceable because the sentence did not repeat the word "coverage"; the one real error in
the paper, an OLS SD of 1,920 where the run computed 1,849.0, was flagged among them)."""

from __future__ import annotations

import json
from pathlib import Path

from core import number_provenance as npv

FIXTURE = Path(__file__).parent / "fixtures" / "provenance_r9k"


def _flagged(text: str, result_json: object) -> set[str]:
    return {f.token for f in npv.check(text, result_json=result_json).findings}


def _r9k() -> tuple[str, object]:
    return ((FIXTURE / "results_section.md").read_text(encoding="utf-8"),
            json.loads((FIXTURE / "result_json.json").read_text(encoding="utf-8")))


def test_r9k_only_the_wrong_number_is_flagged() -> None:
    text, results = _r9k()
    assert _flagged(text, results) == {"1,920"}


def test_a_wrong_endpoint_beside_a_right_value_is_still_flagged() -> None:
    text, results = _r9k()
    assert "0.886–0.941" in text
    # Both ends are then flagged: the pair no longer sits under the result that holds 0.918.
    assert {"0.886", "0.951"} <= _flagged(text.replace("0.886–0.941", "0.886–0.951"), results)


def test_the_value_and_both_ends_must_sit_under_one_result() -> None:
    results = {"a": {"estimate": 0.5, "ci": [0.41, 0.59]}, "b": {"estimate": 0.7, "ci": [0.63, 0.77]},
               "noise": [round(0.001 * i, 3) for i in range(1000)]}
    npv_text = "The rate was 0.5 (0.41 to 0.59) and the other 0.5 (0.63 to 0.77)."
    beside = npv._intervals_beside(npv_text)
    assert ("0.5", "0.41", "0.59") in beside["0.41"]
    traceable, _ = npv.build_traceable(result_json=results)
    assert npv._interval_of_a_result("0.41", beside, traceable)
    # 0.63/0.77 are the interval of b's 0.7, not of a 0.5
    assert not npv._interval_of_a_result("0.63", beside, traceable)


def test_only_a_pair_that_holds_the_value_is_its_interval() -> None:
    assert npv._intervals_beside("It was 0.9 (0.41 to 0.59).") == {}
    assert npv._intervals_beside("It was 0.5 [0.41, 0.59] here.")["0.59"] == [("0.5", "0.41", "0.59")]
    assert npv._intervals_beside("It was 0.5 (95% CI 0.41-0.59).")["0.41"] == [("0.5", "0.41", "0.59")]
    assert npv._intervals_beside("0.813 for OLS (325/400; 0.771–0.848)")["0.848"] == [("0.813", "0.771", "0.848")]
    assert npv._intervals_beside("0.813 for the OLS fit at (0.771–0.848)") == {}


def test_a_large_group_is_not_one_result() -> None:
    """A value and two ends anywhere in a wide dictionary or a list are not one result's interval."""
    wide = {"summary": {f"k{i}": round(0.3 + 0.001 * i, 3) for i in range(300)}}
    text = "The rate was 0.35 (0.31 to 0.39)."
    beside = npv._intervals_beside(text)
    traceable, _ = npv.build_traceable(result_json=wide)
    assert not npv._interval_of_a_result("0.31", beside, traceable)
    listed = {"x": {"vals": [0.31, 0.35, 0.39] + [round(0.5 + 0.001 * i, 3) for i in range(20)]}}
    traceable, _ = npv.build_traceable(result_json=listed)
    assert not npv._interval_of_a_result("0.31", beside, traceable)
    one = {"x": {"rate": {"estimate": 0.35, "ci": [0.31, 0.39], "n": 400}}}
    traceable, _ = npv.build_traceable(result_json=one)
    assert npv._interval_of_a_result("0.31", beside, traceable)
