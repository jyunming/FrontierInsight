"""A known-answer expectation that is wrong (a rounded form compared too tightly, a value the plan's own model derived)
must not condemn correct code (core/equation_tests.py::rounding_of, core/evidence.py::_plan_derived_failure_gaps).
Neutral made-up models only."""

from __future__ import annotations

from typing import Any

import pytest

from core import equation_tests as eqt
from core import evidence


def _one(formula: str, example: dict[str, Any]) -> dict[str, Any]:
    return eqt.example_rows({"model": {"equations": [{"id": "E1", "formula": formula, "role": "generates",
                                                      "example": example}]}})[0]


def test_rounding_of_counts_only_decimal_constants_with_two_or_more_digits() -> None:
    assert eqt.rounding_of("w = 0.61 * a / b") == pytest.approx(0.005 / 0.61)
    assert eqt.rounding_of("w = 2.0 * a + 0.5 * b ** 2 + 3 * c") == 0.0
    assert eqt.rounding_of("w = a / b") == 0.0
    assert eqt.rounding_of("w = 9.81 * t") < eqt.rounding_of("w = 0.61 * t")


def test_an_example_at_full_precision_of_a_rounded_equation_is_accepted_and_the_bar_covers_both() -> None:
    """Old code: the example (full constant 0.6098...) differed from the equation (0.61) by 0.03%, beyond the 1e-6 bar, so
    the example was called contradicting its equation (``unusable``) and correct code was repaired against it."""
    row = _one("w = 0.61 * a / b", {"inputs": {"a": 3.0, "b": 2.0}, "expected_formula": "0.6098 * a / b"})
    assert row["state"] == "ok" and row["equation_agrees"] is True and row["rounded_form"] is True
    full, rounded = row["expected"], row["equation_value"]
    assert abs(rounded - full) <= row["rel"] * abs(full), "code following the equation as written passes the test"
    assert row["rel"] < 0.01, "the bar grows to the gap only, not to the constant's whole rounding"


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
