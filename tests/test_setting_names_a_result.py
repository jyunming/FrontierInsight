"""A setting of the plan (``protocol.grid`` or ``protocol.in_each_run``) lists levels of an input or alternative methods, never
the plan's own outputs. A value that is text equal to an id the plan itself declares as a result is refused, and the plan's model
is asked once to write the part again (the existing part request). Fake model only; neutral made-up names."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import plan
from tests.test_plan_parts import RawModel, _design, _text
from tests.test_plan_optimisation import _engine
from tests.test_protocol_part_asked import GOOD_GRID, GOOD_METRICS, TOPIC, _draft, _record

OUTPUTS = {"metrics": [{"id": "base_temperature", "kind": "mean", "estimand": "mean base temperature", "unit": "one sink"},
                       {"id": "Spread-Index", "kind": "mean", "estimand": "spread", "unit": "one sink"}]}


def test_values_that_are_the_plans_own_output_ids_are_refused() -> None:
    fixed, why = plan.normalize_protocol({**OUTPUTS, "in_each_run": {"readings": ["base_temperature", "spread index"]}})
    assert fixed is None
    assert "`protocol.in_each_run.readings`" in (why or "") and "base_temperature" in (why or "")
    # a check or a criterion name and a grid value are held to the same rule
    fixed, why = plan.normalize_protocol({"oracles": [{"name": "Nesting", "kind": "invariant", "check": "x", "expected": 0}],
                                          "grid": {"what": ["nesting", "other"]}})
    assert fixed is None and "`protocol.grid.what`" in (why or "")


def test_real_levels_and_method_names_are_not_refused() -> None:
    fixed, why = plan.normalize_protocol({**OUTPUTS, "in_each_run": {"solver": ["direct", "iterative"], "load": [1, 2, 3]},
                                          "grid": {"fin_spacing": [2.0, 3.0], "scheme": ["upwind", "central"]}})
    assert why is None and fixed is not None
    assert fixed["in_each_run"] == {"solver": ["direct", "iterative"], "load": [1, 2, 3]}


def test_a_repaired_plan_names_the_refusal_and_asks_for_the_part() -> None:
    fixed, notes = plan.repair_protocol({**OUTPUTS, "in_each_run": {"readings": ["base_temperature", "other"]}})
    assert fixed is not None and "in_each_run" not in fixed
    assert any(plan.ASK_MARK in n and "declares as a result" in n for n in notes)


@pytest.mark.asyncio
async def test_the_part_is_asked_once_and_the_plan_stays_valid(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    draft = _draft(grid=GOOD_GRID, metrics=GOOD_METRICS, in_each_run={"readings": ["base_temperature", "other_reading"]})
    model = RawModel(eng, [json.dumps({"protocol": {"in_each_run": {"solver": ["direct", "iterative"]}}})], draft)
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    assert len(model.asked) == 1, "one request, not a loop"
    assert "in_each_run" in model.asked[0] and "never a value here" in model.asked[0]
    after = _design(eng)
    assert after["protocol"]["in_each_run"] == {"solver": ["direct", "iterative"]}
    assert plan.normalize_design(after)[0] is not None
    assert after["protocol"]["grid"] == GOOD_GRID, "nothing else of the plan moved"
    assert _record(eng)["count"] == 1
    assert "readings" not in _text(eng).split("## The design")[-1]
