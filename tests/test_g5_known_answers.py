"""A known-answer expectation that is wrong must not condemn correct code: an example that is not its equation as written
is refused, never met by widening the bar (core/equation_tests.py::example_rows), and a value the plan's own model
derived is reported as the plan's (core/evidence.py::_plan_derived_failure_gaps).
Neutral made-up models only."""

from __future__ import annotations

from typing import Any

import pytest

from core import equation_tests as eqt
from core import evidence


def _one(formula: str, example: dict[str, Any]) -> dict[str, Any]:
    return eqt.example_rows({"model": {"equations": [{"id": "E1", "formula": formula, "role": "generates",
                                                      "example": example}]}})[0]


def test_an_example_that_is_not_its_equation_as_written_is_refused_never_met_by_widening_the_bar() -> None:
    """Old code: an example within the rounding the written constant could explain was accepted and the test's bar grown
    to the gap: 0.6137 against an equation written 0.61 (0.6%) passed, so code using 0.6137 passed a test of 0.61. The
    digits a plan writes are not evidence that two numbers are one; the example is refused and the plan writes one."""
    row = _one("w = 0.61 * a / b", {"inputs": {"a": 3.0, "b": 2.0}, "expected_formula": "0.6137 * a / b"})
    assert row["state"] == "unusable" and row["equation_agrees"] is False
    assert "the equation as written gives" in row["why"] and "rounded_form" not in row
    full = _one("w = 0.61 * a / b", {"inputs": {"a": 3.0, "b": 2.0}, "expected_formula": "0.6098 * a / b"})
    assert full["state"] == "unusable" and full["equation_agrees"] is False
    same = _one("w = 0.61 * a / b", {"inputs": {"a": 3.0, "b": 2.0}, "expected_formula": "0.61 * a / b"})
    assert same["state"] == "ok" and same["equation_agrees"] is True and same["rel"] == 1e-6


def test_a_gap_larger_than_the_equation_s_rounding_is_still_a_contradiction() -> None:
    """Known positive: a 5% difference is not rounding of 0.61 (0.8%), so the example is still refused."""
    row = _one("w = 0.61 * a / b", {"inputs": {"a": 3.0, "b": 2.0}, "expected_formula": "0.64 * a / b"})
    assert row["state"] == "unusable" and row["equation_agrees"] is False


def test_an_equation_without_rounded_constants_keeps_the_strict_bar() -> None:
    row = _one("w = 2 * a / b", {"inputs": {"a": 3.0, "b": 2.0}, "expected_formula": "2.0003 * a / b"})
    assert row["state"] == "unusable" and row["equation_agrees"] is False


PROTOCOL = {"oracles": [
    {"name": "null_rate", "kind": "special_case", "expected": 0.0, "tolerance": 0.01,
     "reference": "derivation: the null rate is 0 when there is no effect"},
    {"name": "cited_value", "kind": "published_value", "expected": 1.0, "tolerance": 0.01, "reference": "[1]"},
]}
SOURCES = [{"label": "1", "title": "A source", "doi": "10.1000/x"}]


def _record(*names: str) -> dict[str, Any]:
    return {"status": "went_on_failing",
            "went_on": [{"name": n, "measured": 0.05, "by": "automatic"} for n in names]}


def test_a_failed_plan_derived_check_is_reported_as_the_plan_s_claim_not_a_code_error() -> None:
    """Old code: the went-on gap said only that the check failed; nothing said its expected value was the plan's own
    derivation, so a wrong typed 0 for a 0.05 rate read as a code error."""
    gaps = evidence._plan_derived_failure_gaps(PROTOCOL, {"sources": SOURCES}, _record("null_rate"))
    assert len(gaps) == 1 and "null_rate" in gaps[0] and "plan's own model worked out" in gaps[0]


def test_a_failed_check_with_a_retrieved_source_is_not_called_plan_derived() -> None:
    """Known positive for the other side: a value cited from a retrieved source stays a known answer."""
    assert evidence._plan_derived_failure_gaps(PROTOCOL, {"sources": SOURCES}, _record("cited_value")) == []


def test_a_check_with_a_stated_disagreement_or_no_measurement_is_left_to_the_went_on_sentence() -> None:
    rec = {"status": "went_on_failing", "went_on": [
        {"name": "null_rate", "measured": 0.05, "disagreement": "x"}, {"name": "null_rate", "measured": None}]}
    assert evidence._plan_derived_failure_gaps(PROTOCOL, {"sources": SOURCES}, rec) == []
    assert evidence._plan_derived_failure_gaps(PROTOCOL, None, "not a record") == []


def test_rounding_of_a_constant_in_a_small_term_does_not_excuse_a_gap_in_the_whole_answer() -> None:
    """Old code: the allowance was the constant's relative rounding times the WHOLE answer (about 8.7 of 1060), so a gap of 1,
    which 0.61 -> 0.605/0.615 on x = 100 (at most 0.5) cannot explain, was taken as rounding, and the bar was widened."""
    row = _one("y = 0.61*x + 1000", {"inputs": {"x": 100.0}, "expected_formula": "1060"})
    assert row["state"] == "unusable" and row["equation_agrees"] is False


def test_a_gap_in_a_sum_is_refused_too_and_an_example_that_is_the_equation_passes() -> None:
    """Old code: 0.6098 vs 0.61 on x = 100 (a gap of 0.02) was accepted as rounding; it is two numbers, so refused."""
    row = _one("y = 0.61*x + 1000", {"inputs": {"x": 100.0}, "expected_formula": "0.6098*x + 1000"})
    assert row["state"] == "unusable" and row["equation_agrees"] is False
    same = _one("y = 0.61*x + 1000", {"inputs": {"x": 100.0}, "expected_formula": "0.61*x + 1000"})
    assert same["state"] == "ok" and same["equation_agrees"] is True
