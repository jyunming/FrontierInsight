"""The answers part (core/contract_parts.py) asks the plan's model, for each quantity, the unit the topic asks it in and the unit
of the result it names (and of the result the plan maps it to); a result in another unit is asked again, and a quantity still in
two units keeps the plan's result with both units (``unit_conflict``): the paper answers by that result and states both units. The fake model is a function. Neutral made-up study: a pond."""

from __future__ import annotations

import json
from typing import Any

from core import contract_parts as cp
from core import paper_facts as pf
from tests.test_answers_map_part import DESIGN, PROTOCOL, REQ, _ask, _quest

GAP = {"asked": "m", "result": "1", "declared": False}
ASKED = {"Q1": {"name": "peak algae", "stated": "report the peak", "by": "metric:algae_peak"}}


def _map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {**DESIGN, "protocol": {**PROTOCOL, "topic_map": rows}}


def test_a_plan_map_in_another_unit_than_the_question_is_asked_again_with_both_units() -> None:
    design = _map([{"item": "Q1", "by": "metric:algae_peak"}])
    bad = {"answers": {"Q1": {"unit": "m", "by": "metric:algae_peak", "by_unit": "1"}}}
    good = {"answers": {"Q1": {"unit": "m", "by": "none", "plan_unit": "1"}}}
    out, seen = _ask(design, [bad, good], ASKED)
    assert out.parts["answers"]["valid"] and out.parts["answers"]["asks"] == 2
    assert "question asks for a quantity in `m`, but `metric:algae_peak` is in `1`" in seen[1]
    row = next(r for r in out.design["protocol"]["topic_map"] if r["item"] == "Q1")
    assert row["unit_conflict"] == GAP


def test_a_plan_map_still_in_another_unit_after_the_last_ask_is_kept_as_not_answered() -> None:
    design = _map([{"item": "Q1", "by": "metric:algae_peak"}])
    bad = {"answers": {"Q1": {"unit": "m", "by": "metric:algae_peak", "by_unit": "1"}}}
    out, _seen = _ask(design, [bad] * cp.MAX_ASKS, ASKED)
    assert out.parts["answers"]["valid"] and out.parts["answers"]["asks"] == cp.MAX_ASKS
    row = next(r for r in out.design["protocol"]["topic_map"] if r["item"] == "Q1")
    assert row["unit_conflict"] == GAP


def test_a_result_in_the_questions_unit_replaces_a_plan_map_in_another_unit() -> None:
    design = _map([{"item": "Q1", "by": "metric:algae_peak"}])
    reply = {"answers": {"Q1": {"unit": "m", "by": "metric:bloom", "by_unit": "m", "plan_unit": "1"}}}
    out, seen = _ask(design, [reply], ASKED)
    assert len(seen) == 1
    row = next(r for r in out.design["protocol"]["topic_map"] if r["item"] == "Q1")
    assert row["by"] == "metric:bloom" and "unit_conflict" not in row


def test_none_or_another_result_leaves_a_plan_map_in_the_same_unit_as_it_was() -> None:
    """The known positive: units agree, so the plan's own map stands whatever else the model prefers."""
    design = _map([{"item": "Q1", "by": "metric:algae_peak"}])
    for by in ("none", "metric:bloom"):
        reply = {"answers": {"Q1": {"unit": "1", "by": by, "by_unit": "1", "plan_unit": "1"}}}
        out, _seen = _ask(design, [reply], ASKED)
        row = next(r for r in out.design["protocol"]["topic_map"] if r["item"] == "Q1")
        assert row["by"] == "metric:algae_peak" and "unit_conflict" not in row


def test_a_plan_map_kept_in_the_same_unit_is_not_flagged_and_spelling_of_case_and_spaces_is_ignored() -> None:
    design = _map([{"item": "Q1", "by": "metric:algae_peak"}])
    reply = {"answers": {"Q1": {"unit": "Mg / L", "by": "metric:algae_peak", "by_unit": "mg/l"}}}
    out, seen = _ask(design, [reply], ASKED)
    assert len(seen) == 1 and out.parts["answers"]["valid"]
    assert "unit_conflict" not in next(r for r in out.design["protocol"]["topic_map"] if r["item"] == "Q1")


def test_an_answer_without_units_is_taken_as_before() -> None:
    design = _map([{"item": "Q1", "by": "metric:algae_peak"}])
    out, seen = _ask(design, [{"answers": {"Q1": {"by": "none"}}}], ASKED)
    assert len(seen) == 1
    assert next(r for r in out.design["protocol"]["topic_map"] if r["item"] == "Q1") == {"item": "Q1", "by": "metric:algae_peak", "at": {}}


def test_a_new_map_in_another_unit_is_asked_again_and_the_plans_declared_unit_wins_over_the_stated_one() -> None:
    protocol = {**PROTOCOL, "metrics": [{**PROTOCOL["metrics"][0], "physical_unit": "count"}, PROTOCOL["metrics"][1]]}
    design = {**DESIGN, "protocol": protocol}
    asked = {"Q1": {"name": "peak algae", "stated": "report the peak", "by": ""}}
    bad = {"answers": {"Q1": {"unit": "m", "by": "metric:algae_peak", "by_unit": "m"}}}
    good = {"answers": {"Q1": {"unit": "m", "by": "none"}}}
    out, seen = _ask(design, [bad, good], asked)
    assert out.parts["answers"]["asks"] == 2 and "is in `count`" in seen[1]
    assert "- algae_peak: peak algae count [count]" in seen[0]


def test_the_paper_answers_by_the_result_and_states_both_units(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch, [])
    rows = [{"item": "Q1", "by": "metric:algae_peak", "at": {}, "unit_conflict": GAP},
            {"item": "Q2", "by": "metric:bloom", "at": {"nutrient": 1}}]
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: {**PROTOCOL, "topic_map": rows})
    (root / ".fi" / "topic_requirements.json").write_text(json.dumps(REQ), encoding="utf-8")
    text = pf.answers_text(root)
    first = next(line for line in text.splitlines() if line.startswith("- peak algae"))
    assert "not answered" not in first and "{{v:algae_peak" in first
    assert "is in the plan states" not in first and "the plan states that the result used" in first
    assert "is in a pure number (no unit); the question asks in m, and FI did not convert it)" in first
    assert "{{v:bloom:mean_ci @ nutrient=1, temp=10}}" in text  # the known positive is unchanged


def test_a_replacement_whose_unit_is_not_stated_does_not_replace_the_plans_map() -> None:
    p = {**PROTOCOL, "metrics": [*PROTOCOL["metrics"], {"id": "depth", "kind": "mean", "estimand": "pond depth", "unit": "one pond run"}],
         "topic_map": [{"item": "peak algae", "by": "metric:bloom"}]}
    design = {**DESIGN, "protocol": p}
    asked = {"Q1": {"name": "peak algae", "stated": "report the peak", "by": "metric:bloom"}}
    answer = {"Q1": {"unit": "m", "plan_unit": "1", "by": "metric:depth"}}
    row = cp._apply_answers(p, design, answer, asked)["topic_map"][0]
    assert row["by"] == "metric:bloom" and row["unit_conflict"]["asked"] == "m"
    answer["Q1"]["by_unit"] = "m"
    row = cp._apply_answers(p, design, answer, asked)["topic_map"][0]
    assert row["by"] == "metric:depth" and "unit_conflict" not in row

