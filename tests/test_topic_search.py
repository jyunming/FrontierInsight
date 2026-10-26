"""A design the topic asks the study itself to search for: FI's reading of the topic lists it (``searches``), and the plan
covers it only as a design variable of its search, never as a value of a setting worded as already optimised."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core import paper_facts, plan, plan_ledger
from core import topic_requirements as tr
from tests.test_plan_optimisation import EXTRA, HEAT_SINK, _NO_OBJECTIONS, _engine

TOPIC = ("How does the base temperature of a heat sink change with the fin count (8, 12, 16)? Also find the fin spacing "
         "and fin thickness that give the lowest base temperature at 20 W.")
SEARCH = {"what": "the fin spacing and fin thickness", "objective": "the lowest base temperature at 20 W",
          "stated": "find the fin spacing and fin thickness that give the lowest base temperature at 20 W", "main": True}
READING = {"factors": [{"name": "fin count", "levels": [8, 12, 16], "range": [], "stated": "the fin count (8, 12, 16)",
                        "compared": True, "main": True}],
           "quantities": [{"name": "base temperature", "stated": "the base temperature", "main": True}],
           "formulas": [], "comparisons": [], "constraints": [], "stated_results": [], "searches": [SEARCH]}


def _req() -> dict:
    out = tr.parse(copy.deepcopy(READING))
    assert out is not None
    return out


def _sweep(**grid: list) -> dict:
    """The draft that hides the search: a measurement whose setting has a value worded as already optimised."""
    return {**{k: v for k, v in HEAT_SINK.items() if k not in ("protocol", "study_type")}, "study_type": "measure",
            "protocol": {"grid": {"fin_count": [8, 12, 16], **grid}, "oracles": HEAT_SINK["protocol"]["oracles"]}}


def test_the_reading_keeps_the_searches_and_an_older_reading_has_none() -> None:
    req = _req()
    assert req["searches"] == [{"name": "the fin spacing and fin thickness", "objective": SEARCH["objective"],
                                "stated": SEARCH["stated"], "main": True}]
    assert tr.searched(req)[0]["name"] == "the fin spacing and fin thickness"
    older = {k: v for k, v in READING.items() if k != "searches"}
    assert tr.searched(tr.parse(older)) == [] and tr.searched({"error": "no reading"}) == []
    assert tr.parse({"searches": [{"what": "a pixel map", "main": True}]}) is not None  # a reading of a search alone
    assert not tr.is_empty({"searches": [{"name": "a pixel map"}]})


def test_a_value_of_a_setting_worded_as_optimised_is_not_a_search() -> None:
    req = _req()
    view = tr.view(_sweep(geometry=["10 fins of 1 mm", "spacing and thickness optimised for the lowest temperature"]))
    gaps, unresolved = tr.match(req, view)
    found = [g for g in gaps if g["kind"] == "search"]
    assert len(found) == 1 and "S1" not in unresolved
    assert "does not search for it" in found[0]["sentence"] and "design variable" in found[0]["sentence"]
    assert tr.main_gaps(gaps) == found
    text = "\n".join(tr.plan_lines(req, gaps, reviewer="", same_model=False, topic=TOPIC))
    assert "- To search for (FI runs the search): the fin spacing and fin thickness" in text
    assert "NOT searched for in the plan" in text
    asked = tr.request(TOPIC, found, req=req)
    assert "`protocol.optimisation.design_variables`" in asked and "S1: the fin spacing and fin thickness" in asked
    # The paper's limitations and the plan's ledger name it the same way.
    assert paper_facts._topic_sentence(found[0])[1].startswith("The topic asks to search for the fin spacing")
    sweep = _sweep(geometry=["10 fins of 1 mm", "spacing and thickness optimised for the lowest temperature"])
    assert [u["id"] for u in plan_ledger.uncovered(req, sweep["protocol"], design=sweep) if u["kind"] == "search"] == ["S1"]


def test_a_design_variable_of_the_search_covers_it_and_another_name_goes_to_a_reader() -> None:
    req = _req()
    named = tr.view(copy.deepcopy(HEAT_SINK))  # design variables fin_spacing, fin_thickness, fin_count
    gaps, unresolved = tr.match(req, named)
    assert not [g for g in gaps if g["kind"] == "search"] and "S1" not in unresolved
    other = copy.deepcopy(HEAT_SINK)
    for v in other["protocol"]["optimisation"]["design_variables"]:
        v["name"] = {"fin_spacing": "pitch_mm", "fin_thickness": "web_mm", "fin_count": "n"}[v["name"]]
        v.pop("meaning", None)
    gaps, unresolved = tr.match(req, tr.view(other))
    assert "S1" in unresolved and not [g for g in gaps if g["kind"] == "search"]
    items, shown = tr.judge_parts(req, tr.view(other), ["S1"])
    assert json.loads(items)[0]["kind"] == "search" and json.loads(shown)["searched_for"][0]["name"] == "pitch_mm"
    # The judge's "replaced" is not a cover for a search; "covered" is.
    assert [g["kind"] for g in tr.gaps_from(req, tr.view(other), {"S1": "replaced"})[0]] == ["search"]
    assert not tr.gaps_from(req, tr.view(other), {"S1": "covered"})[0]


def test_the_plans_own_mapping_counts_only_for_a_design_variable() -> None:
    req = _req()
    other = copy.deepcopy(HEAT_SINK)
    other["protocol"]["optimisation"]["design_variables"][0]["name"] = "pitch_mm"
    other["protocol"]["optimisation"]["design_variables"][1]["name"] = "web_mm"
    for v in other["protocol"]["optimisation"]["design_variables"]:
        v.pop("meaning", None)
    other["protocol"]["topic_map"] = [{"item": "S1", "by": "setting:pitch_mm"}]
    gaps, unresolved = tr.match(req, tr.view(other))
    assert "S1" not in unresolved and not [g for g in gaps if g["kind"] == "search"]
    sweep = _sweep(geometry=["optimised"])
    sweep["protocol"]["topic_map"] = [{"item": "S1", "by": "setting:geometry"}]
    gaps, _unresolved = tr.match(req, tr.view(sweep))
    assert [g["kind"] for g in gaps if g["kind"] == "search"] == ["search"]


@pytest.mark.asyncio
async def test_a_topic_that_asks_for_a_search_makes_the_plan_a_search_whatever_the_draft_calls_itself(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps(READING), json.dumps({**_sweep(), "plan": EXTRA}), _NO_OBJECTIONS])
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert plan.parse(text).design["study_type"] == "find_best_design"
    assert "The topic asks the study to search for the fin spacing and fin thickness" in text
    assert "A measurement, or a search for the best design?" in eng._client.chat.await_args_list[1].args[0][0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [None, "1"])
async def test_no_search_in_the_reading_or_a_person_who_answered_measure_keeps_the_measurement(
        tmp_path: Path, answer: str | None) -> None:
    reading = {**READING, "searches": [] if answer is None else [SEARCH]}
    eng = _engine(tmp_path, [json.dumps(reading), json.dumps({**_sweep(), "plan": EXTRA}), _NO_OBJECTIONS])
    state = {"topic": TOPIC, "iteration": 0, **({"clarify_answers": {"study_type": answer}} if answer else {})}
    await eng._node_plan(state)
    assert plan.parse(plan.plan_path(eng.quest_root).read_text(encoding="utf-8")).design["study_type"] == "measure"


def test_a_level_that_is_the_searched_design_is_covered_by_the_search_not_by_a_setting() -> None:
    """The topic compares kinds of heat sink, one of them the shape the study must search for: a plan that sweeps the
    other kinds as a setting and searches for that shape covers the factor; the shape is not a value the setting must
    hold (the search item is checked on its own), and the reader is told which level the search covers."""
    reading = copy.deepcopy(READING)
    reading["factors"].append({"name": "heat sink kind", "levels": ["straight fins", "pin fins", "searched fin shape"],
                               "range": [], "stated": "straight fins, pin fins and a searched fin shape",
                               "compared": True, "main": True})
    reading["searches"] = [{"what": "a searched fin shape", "objective": "the lowest base temperature",
                            "stated": "a searched fin shape", "main": True}]
    req = tr.parse(reading)
    searching = {"axes": {"fin_count": [8, 12, 16], "heat_sink_kind": ["straight fins", "pin fins"]}, "metrics": [],
                 "equations": [], "assumptions": "", "topic_map": [{"item": "F2", "by": "setting:heat_sink_kind"}], "search": True,
                 "design_variables": {"fin_profile": "the fin shape the search changes"}}
    factor = req["factors"][1]
    assert tr.searched_levels(req, searching, factor) == ["searched fin shape"]
    gaps, unresolved = tr.match(req, searching)
    assert not [g for g in gaps if g.get("kind") == "level"] and "F2" not in unresolved
    row = next(r for r in json.loads(tr.judge_parts(req, searching, ["F2"])[0]) if r["id"] == "F2")
    assert row["levels"] == ["straight fins", "pin fins"] and row["searched_by_fi"] == ["searched fin shape"]
    # With no search in the plan the shape is a level like any other: the setting leaves it out.
    plain = {**searching, "search": False, "design_variables": {}}
    assert tr.searched_levels(req, plain, factor) == []
    gaps, _ = tr.match(req, plain)
    assert any(g.get("kind") == "level" for g in gaps)
