"""An ``expected_formula`` is read the way a plan writes it: LaTeX spellings are the calculator's own arithmetic, a name
may be a setting of the plan (a fixed one) as well as of the check's case, and a name nothing sets is asked about and
never guessed (core/oracle_forms.py::_formula_result, ``fixed_settings``).

Fake models only: nothing here calls a real model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from core import oracle_forms as of
from core.engine import Engine
from tests.test_oracle_review import MODEL, _Model, _config

SINK = {"name": "sink_limit", "kind": "special_case", "check": "R_th", "expected": 9.0, "tolerance": 0.01,
        "tolerance_mode": "absolute", "reference": "derivation: R = L / (k * A), so R = 0.5 / (200 * 0.0004) = 6.25",
        "case": {"L": 0.5}, "measure": "R_th"}
NO_REVIEW = "I think these checks look fine."


def _oracle(formula: Any, **case: Any) -> dict[str, Any]:
    return {"expected_formula": formula, "case": case}


def test_a_formula_with_latex_spellings_is_computed() -> None:
    value, why = of.formula_value(_oracle(r"\frac{L}{k \cdot A}", L=0.5, k=200, A=0.0004))
    assert why == "" and value == pytest.approx(6.25)
    value, why = of.formula_value(_oracle(r"2 \times \pi \times \sqrt{x}^2", x=4))
    assert why == "" and value == pytest.approx(2 * math.pi * 4)
    value, why = of.formula_value(_oracle(r"x^2 + 1", x=3))
    assert why == "" and value == 10


def test_a_double_backslash_and_a_greek_name_are_plain_names_never_a_value() -> None:
    # `\\lambda` as a YAML string holds two backslashes; `lambda` is a Python word, and still only a name.
    value, why = of.formula_value(_oracle(r"(\\lambda / w) * (2 * pi / p)", **{"lambda": 3, "w": 1.5, "p": 2}))
    assert why == "" and value == pytest.approx(2 * math.pi)
    value, why = of.formula_value(_oracle(r"\mu * 2"))
    assert value is None and "mu" in why, "a Greek letter is never given a value of FI's choosing"
    _v, _w, names, _u = of._formula_result(_oracle(r"\lambda * 2"), {})
    assert names == ["lambda"]


def test_a_name_the_plan_fixes_is_used_and_the_check_case_wins() -> None:
    plan_fixed = of.fixed_settings({"thresholds": {"k": 200.0}, "grid": {"A": [0.0004], "n": [1, 2]},
                                    "optimisation": {"baseline": {"values": {"h": 5}}}})
    assert plan_fixed == {"k": 200.0, "A": 0.0004, "h": 5.0}, "a swept parameter is not a fixed setting"
    value, why = of.formula_value(_oracle("L / (k * A)", L=0.5), plan_fixed)
    assert why == "" and value == pytest.approx(6.25)
    value, _ = of.formula_value(_oracle("L / (k * A)", L=0.5, k=100), plan_fixed)
    assert value == pytest.approx(12.5)


def test_a_name_given_two_numbers_in_the_plan_is_not_a_fixed_setting() -> None:
    assert of.fixed_settings({"thresholds": {"k": 1.0}, "grid": {"k": [2.0]}}) == {}


def test_a_formula_that_already_worked_gives_the_same_number() -> None:
    value, why = of.formula_value(_oracle("2*pi*sqrt(1/9.81)"))
    assert why == "" and value == pytest.approx(2 * math.pi * math.sqrt(1 / 9.81), rel=1e-15)
    value, why = of.formula_value(_oracle("a * b", a=2, b=3), {"a": 99.0})
    assert why == "" and value == 6


def test_the_request_says_where_a_name_may_come_from_and_asks_for_the_values_of_the_rest() -> None:
    findings = of.formula_findings({"oracles": [{**SINK, "expected_formula": "L / (k * A)"}],
                                    "thresholds": {"k": 200.0}})
    assert findings[0]["state"] == "unusable" and findings[0]["names"] == ["A"]
    text = of.formula_request(findings, fixed={"k": 200.0})
    assert "setting of that check's `case` or one of the plan's fixed settings (k = 200)" in text
    assert "uses A" in text and "write the formula with numbers" in text


@pytest.mark.asyncio
async def test_a_latex_formula_with_a_fixed_setting_is_applied_by_the_engine(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    protocol = {"grid": {"dt": [0.01]}, "thresholds": {"k": 200.0, "A": 0.0004}, "model": MODEL,
                "oracles": [{**SINK, "expected_formula": r"\frac{L}{k \cdot A}"}]}
    model = _Model(protocol, NO_REVIEW)
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert engine._planned_oracles()["sink_limit"]["expected"] == pytest.approx(6.25)
    assert len(model.revisions) == 1 and "FI computed your `expected_formula`" in model.revisions[0]


@pytest.mark.asyncio
async def test_an_unknown_name_is_asked_about_once_and_never_guessed(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    protocol = {"grid": {"dt": [0.01]}, "model": MODEL,
                "oracles": [{**SINK, "expected_formula": r"\frac{L}{k \cdot A}"}]}
    model = _Model(protocol, NO_REVIEW)
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len(model.revisions) == 1
    assert "uses k, A" in model.revisions[0] and "(dt = 0.01)" in model.revisions[0]
    assert engine._planned_oracles()["sink_limit"]["expected"] == 9.0, "the check stays as the plan wrote it"
