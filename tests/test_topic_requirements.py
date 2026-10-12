"""What the topic asks for: read once from the topic text, held against the plan and the design that will run, asked of the
plan's model when the plan lacks it, a stop before anything runs when it is the question's own, a line in the paper's
limitations when it is not; and the code-writing requests are shown the topic and its stated formulas. Fake models only,
neutral topics (a cup of coffee cooling under different lids)."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import plan
from core import topic_requirements as tr
from tests.test_plan_optimisation import EXTRA, HEAT_SINK, Paused, _engine, _NO_OBJECTIONS

TOPIC = (
    "Compare how a cup of coffee cools under three lid materials (paper, foam, steel) at room temperatures from 10 to 30 "
    "degrees. Report the time to reach 60 degrees and the heat lost per minute. The cup follows Newton's law of cooling, "
    "dT/dt = -k*(T - Tr), with Tr the room temperature. The expected value of the cooling constant k is 0.05 per minute. "
    "Run on a CPU."
)

EXTRACTION: dict[str, Any] = {
    "factors": [
        {"name": "lid material", "levels": ["paper", "foam", "steel"], "range": [],
         "stated": "three lid materials (paper, foam, steel)", "main": True},
        {"name": "room temperature", "levels": [], "range": [10, 30],
         "stated": "room temperatures from 10 to 30 degrees", "main": False},
    ],
    "quantities": [
        {"name": "time to reach 60 degrees", "stated": "the time to reach 60 degrees", "main": True},
        {"name": "heat lost per minute", "stated": "the heat lost per minute", "main": False},
    ],
    "formulas": [{"name": "Newton's law of cooling", "text": "dT/dt = -k*(T - Tr)", "main": False}],
    "comparisons": [{"what": "how the cooling differs between the lid materials", "between": ["paper", "foam", "steel"]}],
    "constraints": ["Run on a CPU."],
    "stated_results": ["0.05 per minute"],
}

MODEL = {
    "summary": "A lumped cup cooling in a room.",
    "assumptions": ["uniform temperature in the cup"],
    "equations": [{"id": "E1", "formula": "T(t) = Tr + (T0 - Tr)*exp(-k*t)", "role": "generates",
                   "source": "derivation: the solution of the cooling law",
                   "example": {"inputs": {"Tr": 20, "T0": 90, "k": 0.05, "t": 10},
                               "expected_formula": "20 + (90 - 20)*exp(-0.05*10)"}}],
}
METRIC_TIME = {"id": "time_to_60_degrees", "kind": "mean", "estimand": "mean time to reach 60 degrees", "unit": "minute"}
METRIC_HEAT = {"id": "heat_lost_per_minute", "kind": "mean", "estimand": "mean heat lost per minute", "unit": "joule"}
LID_AXIS = {"lid_material": ["paper", "foam", "steel"]}
NEWTON = {"id": "E2", "formula": "dT/dt = -k*(T - Tr)", "role": "analyses",
          "source": "derivation: Newton's law of cooling, as the topic states it",
          "example": {"inputs": {"T": 80, "Tr": 20, "k": 0.05}, "expected_formula": "-0.05*(80 - 20)"}}


def _draft(**protocol: Any) -> dict[str, Any]:
    draft = copy.deepcopy(HEAT_SINK)
    draft["study_type"] = "measure"
    draft["protocol"] = {"oracles": copy.deepcopy(HEAT_SINK["protocol"]["oracles"]), "grid": {"room_temperature_C": [10, 20, 30]},
                         "metrics": [METRIC_TIME], "model": copy.deepcopy(MODEL), **protocol}
    return draft


def _items_and_plan(prompt: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blocks = re.findall(r"```json\n(.*?)\n```", prompt, re.DOTALL)
    return json.loads(blocks[0]), json.loads(blocks[1])


class Reader:
    """The model, in every part it plays: it drafts and audits the plan, lists what the topic asks for, judges what a name
    match could not find, and rewrites the plan when asked (only what ``adds`` names)."""

    def __init__(self, eng: Any, draft: dict[str, Any], *, adds: set[str] = frozenset(), extraction: dict[str, Any] | None = EXTRACTION,
                 judge: Any = None, judge_fails: bool = False) -> None:
        self.eng, self.draft, self.adds = eng, draft, set(adds)
        self.extraction = extraction
        self.judge_override = judge
        self.judge_fails = judge_fails
        self.confirmed: list[str] = []
        self.prompts: list[str] = []
        self.asked: list[str] = []
        self.judged = 0
        self.extracted = 0
        self.drafted = False
        eng.config.topic = TOPIC
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    def _judge(self, prompt: str) -> str:
        self.judged += 1
        if self.judge_fails:
            raise RuntimeError("no answer")
        items, shown = _items_and_plan(prompt)
        settings = " ".join(shown["settings_swept"]).lower()
        metrics = " ".join(m["id"] + " " + m["text"] for m in shown["metrics"]).lower()
        equations = " ".join(e["formula"] for e in shown["equations"])
        out = []
        for item in items:
            name = str(item.get("name") or item.get("text") or "").lower()
            if self.judge_override is not None:
                verdict = self.judge_override(item, shown)
            elif "plan_says_it_is_covered_by" in item:
                self.confirmed.append(item["plan_says_it_is_covered_by"])
                verdict = "covered"
            elif item["kind"] == "factor":
                verdict = "covered" if name.split()[0] in settings else "missing"
            elif item["kind"] == "quantity":
                verdict = "covered" if name.split()[0] in metrics else "missing"
            else:
                verdict = "covered" if item["text"].replace(" ", "") in equations.replace(" ", "") else "missing"
            out.append({"id": item["id"], "verdict": verdict, "where": "", "why": "x"})
        return json.dumps({"items": out})

    def _rewrite(self, text: str) -> str:
        def change(design: dict[str, Any]) -> dict[str, Any]:
            protocol = dict(design["protocol"])
            if "lid" in self.adds:
                protocol["grid"] = {**protocol.get("grid", {}), **LID_AXIS}
            if "heat" in self.adds:
                protocol["metrics"] = [*protocol.get("metrics", []), METRIC_HEAT]
            if "newton" in self.adds:
                model = dict(protocol.get("model") or {})
                model["equations"] = [*model.get("equations", []), NEWTON]
                protocol["model"] = model
            if "map_cover" in self.adds:  # the same factor under another name, with the plan's own mapping
                protocol["grid"] = {**protocol.get("grid", {}), "cover_kind": ["paper", "foam", "steel"]}
                protocol["topic_map"] = [{"item": "lid material", "by": "setting:cover_kind"}]
            if "map_bad" in self.adds:  # a mapping to something the plan does not have
                protocol["topic_map"] = [{"item": "lid material", "by": "setting:nothing_like_it"}]
            return {**design, "protocol": protocol, "tolerance_changed": True}

        out = plan.edit_design_block(text, change)
        assert out is not None
        return out

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if prompt.lstrip().startswith("# What a Research Topic Asks For"):
            self.extracted += 1
            if self.extraction is None:
                raise RuntimeError("no answer")
            return json.dumps(self.extraction)
        if prompt.lstrip().startswith("# Is It In The Plan?"):
            return self._judge(prompt)
        if "The topic asks for things this plan does not yet include" in prompt:
            self.asked.append(prompt)
            return self._rewrite(plan.plan_path(self.eng.quest_root).read_text(encoding="utf-8"))
        if not self.drafted:
            self.drafted = True
            return json.dumps({**self.draft, "plan": EXTRA})
        return _NO_OBJECTIONS


def _honest_stop(eng: Any) -> dict[str, Any]:
    """The plain stop a plan that leaves out what the topic needs ends in (no paper, nobody asked to pick another model)."""
    stop = eng._topic_unmet_stop()
    assert stop is not None, "the quest records the stop for the code step"
    return stop["exec_result"]


def _stop(eng: Any) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    return seen


def _stored(eng: Any) -> dict[str, Any]:
    return json.loads((eng.fi_dir / "topic_requirements.json").read_text(encoding="utf-8"))


def _plan_text(eng: Any) -> str:
    return plan.plan_path(eng.quest_root).read_text(encoding="utf-8")


# --- the matching that needs no model --------------------------------------------------------------------------------


def test_a_setting_that_does_not_cover_the_stated_levels_or_range_is_a_gap_and_a_name_match_settles_the_rest() -> None:
    req = tr.parse(EXTRACTION)
    assert req is not None
    plan_view = tr.view(_draft(grid={"room_temperature_C": [10, 20], "lid_material": ["paper", "foam"]}))
    gaps, unresolved = tr.match(req, plan_view)
    by_kind = {(g["kind"], g["name"]) for g in gaps}
    assert ("level", "room temperature") in by_kind and ("level", "lid material") in by_kind
    assert "its settings run from 10 to 20, not 10 to 30" in [g["sentence"] for g in gaps if g["name"] == "room temperature"][0]
    assert "steel" in [g["sentence"] for g in gaps if g["name"] == "lid material"][0]
    assert unresolved == ["Q2", "M1"], "the heat quantity and the formula need a reader; the rest a name settles"
    full = tr.view(_draft(grid={"room_temperature_C": [10, 20, 30], **LID_AXIS}, model={**MODEL, "equations": [*MODEL["equations"], NEWTON]}))
    assert tr.match(req, full)[0] == [] and tr.match(req, full)[1] == ["Q2"]


def test_a_reply_that_names_nothing_is_no_reading_and_a_search_variable_counts_as_a_setting() -> None:
    assert tr.parse({"factors": [], "quantities": [], "formulas": []}) is None and tr.parse("x") is None
    req = tr.parse({"factors": [{"name": "fin spacing", "range": [1, 6], "main": True}]})
    block = {"objective": {"quantity": "base_temperature"}, "design_variables": [{"name": "fin_spacing", "low": 1.0, "high": 6.0}]}
    assert tr.match(req, tr.view({"protocol": {"optimisation": block}})) == ([], [])


def test_named_levels_held_as_numbers_are_left_to_a_reader_not_passed() -> None:
    req = tr.parse({"factors": [{"name": "lid material", "levels": ["paper", "foam", "steel"], "main": True}]})
    coded = tr.view({"protocol": {"grid": {"lid_material": [1, 2, 3]}}})
    assert tr.match(req, coded) == ([], ["F1"])
    gaps, unjudged = tr.gaps_from(req, coded, {})
    assert gaps == [] and unjudged == ["lid material"], "no judgement is not a pass: it is named as unchecked"
    assert tr.gaps_from(req, coded, {"F1": "missing"})[0][0]["kind"] == "factor"
    assert tr.gaps_from(req, coded, {"F1": "covered"}) == ([], [])


def test_the_topic_reading_is_a_reading_of_the_plan_not_of_the_checks_in_the_quality_report() -> None:
    from core import quality_report as qr

    assert qr._stage_of("oracle_review.topic") == "plan"
    assert qr._reads("plan", "oracle_review.topic") and not qr._reads("checks", "oracle_review.topic")
    assert qr._reads("checks", "oracle_review") and not qr._reads("plan", "oracle_review")


# --- Part 1: what the topic asks for, read once and shown ----------------------------------------------------------


@pytest.mark.asyncio
async def test_what_the_topic_asks_for_is_read_once_stored_and_shown_in_plain_words(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(grid={**LID_AXIS, "room_temperature_C": [10, 20, 30]}, metrics=[METRIC_TIME, METRIC_HEAT],
                               model={**MODEL, "equations": [*MODEL["equations"], NEWTON]}))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    kept = _stored(eng)
    assert [f["name"] for f in kept["factors"]] == ["lid material", "room temperature"]
    assert kept["factors"][0]["levels"] == ["paper", "foam", "steel"] and kept["factors"][1]["range"] == [10.0, 30.0]
    assert kept["formulas"][0]["text"] == "dT/dt = -k*(T - Tr)" and "same_model" in kept and "error" not in kept
    text = _plan_text(eng)
    assert "## What the topic asks for" in text
    assert "To vary or compare: lid material (paper, foam, steel)." in text
    assert "To report: heat lost per minute." in text and "NOT" not in text.split("## What the topic asks for")[1].split("##")[0]
    assert model.extracted == 1 and model.asked == [], "a plan that has everything is not asked for anything"
    await eng._cover_the_topic_in_the_plan({})  # a resume does not read the topic again
    assert model.extracted == 1


@pytest.mark.asyncio
async def test_a_reading_that_could_not_be_had_is_said_and_never_stops_the_quest(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Reader(eng, _draft(), extraction=None)
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    # (The section also holds what the plan left out of its simplifications and definitions: only the topic's own lines are tested.)
    assert "Read from the topic by" not in _plan_text(eng), "a plan a person reads is not changed by a reading that failed"
    assert not any("what the topic asks for" in str(h.get("note") or "") for h in plan.history(eng.quest_root))
    seen = _stop(eng)
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [] and "design" in patch
    note = tr.write_note(eng.quest_root)
    assert "could not read the topic for what it explicitly asks for" in note and "not checked" in note


# --- Part 2: every requirement maps to something runnable ----------------------------------------------------------


@pytest.mark.asyncio
async def test_what_could_not_be_judged_is_never_passed_in_silence(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(), judge_fails=True)
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    shown = _plan_text(eng).split("## What the topic asks for")[1].split("##")[0]
    assert "To vary or compare: lid material (paper, foam, steel) - could not be checked against the plan." in shown
    assert "To report: heat lost per minute - could not be checked against the plan." in shown
    assert model.asked == [], "nothing is asked of the plan for what nobody found missing"
    seen = _stop(eng)
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [] and "design" in patch, "an unchecked item is not a stop"
    note = tr.write_note(eng.quest_root)
    assert "could not check whether the plan includes what the topic asks for about" in note and "lid material" in note
    assert "Say plainly in the limitations that this was not checked" in note


@pytest.mark.asyncio
async def test_a_topic_factor_missing_from_the_grid_is_asked_of_the_plan_added_and_runs(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(), adds={"lid", "heat", "newton"})
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    protocol = patch["design"]["protocol"]
    assert seen == [], "the plan was completed, so nothing stops"
    assert protocol["grid"]["lid_material"] == ["paper", "foam", "steel"] and protocol["grid"]["room_temperature_C"] == [10, 20, 30]
    assert [m["id"] for m in protocol["metrics"]] == ["time_to_60_degrees", "heat_lost_per_minute"]
    assert [e["id"] for e in protocol["model"]["equations"]] == ["E1", "E2"]
    assert len(model.asked) == 1 and "lid material" in model.asked[0] and "heat lost per minute" in model.asked[0]
    assert "tolerance_changed" not in _plan_text(eng), "only what was asked is kept from the rewrite"
    assert "FI asked the plan's model to add what was missing; it is in the plan now" in _plan_text(eng)
    assert json.loads((eng.fi_dir / "topic_asked.json").read_text(encoding="utf-8"))["count"] == 1


@pytest.mark.asyncio
async def test_a_stated_formula_is_carried_into_the_equations(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(grid={**LID_AXIS, "room_temperature_C": [10, 20, 30]}, metrics=[METRIC_TIME, METRIC_HEAT]),
                   adds={"newton"})
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    equations = [e["formula"] for e in plan.parse(_plan_text(eng)).design["protocol"]["model"]["equations"]]
    assert "dT/dt = -k*(T - Tr)" in equations
    assert len(model.asked) == 1 and "states the formula `dT/dt = -k*(T - Tr)`" in model.asked[0]


@pytest.mark.asyncio
async def test_a_formula_the_plan_replaces_and_says_why_is_not_a_gap(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(grid={**LID_AXIS, "room_temperature_C": [10, 20, 30]}, metrics=[METRIC_TIME, METRIC_HEAT]),
                   judge=lambda item, shown: "replaced" if item["kind"] == "formula" else "covered")
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    assert model.asked == [] and "NOT" not in _plan_text(eng).split("## What the topic asks for")[1].split("##")[0]


@pytest.mark.asyncio
async def test_a_main_factor_still_missing_after_the_bound_stops_before_anything_runs(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(), adds=set())  # the plan's model never adds anything
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [], "nobody is asked to pick another model: the plan's rules, not the model, left it out"
    assert 1 <= len(model.asked) <= 4
    result = _honest_stop(eng)
    shown = " ".join([result["topic_unmet"], *result["topic_unmet_tried"]])
    assert "lid material" in shown and "asked the plan's model" in shown
    assert "heat lost per minute" not in shown, "a secondary item does not stop the quest"
    assert not (eng.quest_root / "paper.md").exists()
    asked_before = len(model.asked)
    await eng._node_design({"topic": TOPIC, "iteration": 0})  # a resume stops again and asks no more
    assert len(model.asked) == asked_before and seen == []
    assert "the quest stops before anything runs" in _plan_text(eng)
    assert "the quest stops before anything runs" in _plan_text(eng)


@pytest.mark.asyncio
async def test_a_secondary_item_still_missing_is_said_in_the_limitations_and_the_quest_goes_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Reader(eng, _draft(), adds={"lid"})  # the main factor is added; the heat and the formula are not
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [] and "lid_material" in patch["design"]["protocol"]["grid"]
    note = tr.write_note(eng.quest_root)
    assert "The topic asked for more than this study did" in note and "heat lost per minute" in note
    assert "Say plainly in the limitations that these were not studied" in note
    assert tr.write_note(tmp_path / "nowhere") == ""


@pytest.mark.asyncio
async def test_a_redesign_that_drops_an_axis_is_caught_and_asked_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(grid={**LID_AXIS, "room_temperature_C": [10, 20, 30]}, metrics=[METRIC_TIME, METRIC_HEAT],
                               model={**MODEL, "equations": [*MODEL["equations"], NEWTON]}))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    redesign = copy.deepcopy(plan.parse(_plan_text(eng)).design)
    del redesign["protocol"]["grid"]["lid_material"]
    state = {"topic": TOPIC, "iteration": 1, "design": redesign}
    # The design step's model puts the axis back when asked.
    original = model.chat

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if "The topic asks for things this plan does not yet include" in prompt:
            model.asked.append(prompt)
            return json.dumps({**redesign, "protocol": {**redesign["protocol"], "grid": {**redesign["protocol"]["grid"], **LID_AXIS}}})
        return await original(messages, **kw)

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    design, _sha = await eng._hold_the_design_to_the_topic(state, redesign, "")
    assert design["protocol"]["grid"]["lid_material"] == ["paper", "foam", "steel"]
    assert len(model.asked) == 1 and "lid material" in model.asked[0]


@pytest.mark.asyncio
async def test_a_redesign_that_stays_without_the_axis_stops_before_the_run_and_is_asked_within_its_bound(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(grid={**LID_AXIS, "room_temperature_C": [10, 20, 30]}, metrics=[METRIC_TIME, METRIC_HEAT],
                               model={**MODEL, "equations": [*MODEL["equations"], NEWTON]}))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    redesign = copy.deepcopy(plan.parse(_plan_text(eng)).design)
    del redesign["protocol"]["grid"]["lid_material"]
    seen = _stop(eng)
    state = {"topic": TOPIC, "iteration": 1, "design": redesign}
    for _ in range(2):
        await eng._hold_the_design_to_the_topic(state, redesign, "")
    assert seen == []
    asked = [p for p in model.prompts if "The topic asks for things this plan does not yet include" in p]
    assert len(asked) == 4 and "Your last answer was not used" not in asked[0] and "Your last answer was not used" in asked[1]
    assert "lid material" in _honest_stop(eng)["topic_unmet"]


@pytest.mark.asyncio
async def test_a_mapping_to_an_axis_with_another_name_counts_when_the_reader_confirms_it(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(), adds={"map_cover", "heat", "newton"})
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [], "the plan named a setting that exists and covers the levels: the reader's word does not stop the quest"
    assert patch["design"]["protocol"]["grid"]["cover_kind"] == ["paper", "foam", "steel"]
    assert "protocol.topic_map" in model.asked[0]
    assert model.confirmed == ["cover_kind"], "an axis with an unrelated name is a cover only because the reader confirmed it"
    covered = json.loads((eng.fi_dir / "topic_covered.json").read_text(encoding="utf-8"))
    assert covered["left"] == [], "nothing is left open"


@pytest.mark.asyncio
async def test_a_mapping_to_something_the_plan_does_not_have_does_not_save_a_missing_main_factor(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Reader(eng, _draft(), adds={"map_bad"})
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [] and "lid material" in _honest_stop(eng)["topic_unmet"]


def test_a_mapping_is_checked_for_the_levels_it_must_cover() -> None:
    req = tr.parse(EXTRACTION)
    short = tr.view(_draft(grid={"cover_kind": ["paper", "foam"]}, topic_map=[{"item": "lid material", "by": "setting:cover_kind"}]))
    assert tr.mapping_of(req, short, "factors", req["factors"][0])[0] == "bad"
    full = tr.view(_draft(grid={"cover_kind": ["paper", "foam", "steel"]},
                          topic_map=[{"item": "lid material", "by": "setting:cover_kind"},
                                     {"item": "dT/dt = -k*(T - Tr)", "by": "equation:E1"}]))
    assert tr.mapping_of(req, full, "factors", req["factors"][0]) == ("judge", "cover_kind")
    assert tr.mapping_of(req, full, "formulas", req["formulas"][0]) == ("judge", "E1")
    named = tr.view(_draft(grid={"lid_material": ["paper", "foam", "steel"]}, topic_map=[{"item": "lid material", "by": "setting:lid_material"}]))
    assert tr.mapping_of(req, named, "factors", req["factors"][0]) == ("ok", "lid_material"), "a related name needs no reader"
    assert tr.mapping_of(req, full, "quantities", req["quantities"][1]) is None


def test_an_axis_whose_values_merely_fit_is_not_a_cover_and_a_formula_needs_the_reader_too() -> None:
    req = tr.parse({"factors": [{"name": "room temperature", "range": [10, 20], "main": True}],
                    "formulas": [{"name": "cooling law", "text": "dT/dt = -k*(T - Tr)", "main": False}]})
    view = tr.view(_draft(grid={"batch_size": [10, 15, 20]}, topic_map=[
        {"item": "room temperature", "by": "setting:batch_size"}, {"item": "dT/dt = -k*(T - Tr)", "by": "equation:E1"}]))
    assert tr.mapping_of(req, view, "factors", req["factors"][0]) == ("judge", "batch_size")
    assert tr.mapping_of(req, view, "formulas", req["formulas"][0]) == ("judge", "E1")
    gaps, unresolved = tr.match(req, view)
    assert gaps == [] and unresolved == ["F1", "M1"], "neither is covered until a reader says so"
    items = json.loads(tr.judge_parts(req, view, unresolved)[0])
    assert items[0]["plan_says_it_is_covered_by"] == "batch_size" and items[1]["plan_says_it_is_covered_by"] == "E1"
    rejected, _ = tr.gaps_from(req, view, {"F1": "missing", "M1": "missing"})
    assert [(g["kind"], g["main"]) for g in rejected] == [("factor", True), ("formula", False)]
    assert tr.main_gaps(rejected)[0]["name"] == "room temperature", "the main factor stops the quest"
    assert tr.gaps_from(req, view, {"F1": "covered", "M1": "covered"}) == ([], [])
    unread, unjudged = tr.gaps_from(req, view, {})
    assert unread == [] and unjudged == ["room temperature", "cooling law"], "no answer is named as unchecked, never a pass"


@pytest.mark.asyncio
async def test_a_formula_mapped_to_an_unrelated_equation_that_the_reader_rejects_is_a_limitation_not_a_stop(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Reader(eng, _draft(grid={**LID_AXIS, "room_temperature_C": [10, 20, 30]}, metrics=[METRIC_TIME, METRIC_HEAT],
                       topic_map=[{"item": "dT/dt = -k*(T - Tr)", "by": "equation:E1"}]),
           judge=lambda item, shown: "missing" if item["kind"] == "formula" else "covered")
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [] and "design" in patch
    assert "states the formula `dT/dt = -k*(T - Tr)`" in tr.write_note(eng.quest_root)


# --- constraints: the cap and the writer's note -----------------------------------------------------------------------


def test_parse_keeps_thirty_constraints_when_given_thirty_five() -> None:
    obj = {"factors": [{"name": "alpha"}], "constraints": [f"limit {n}." for n in range(35)]}
    parsed = tr.parse(obj)
    assert parsed is not None and len(parsed["constraints"]) == 30
    assert parsed["constraints"][-1] == "limit 29."


def test_write_note_names_each_constraint_when_no_covered_record_exists(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    fi.mkdir()
    (fi / tr.FILE).write_text(json.dumps({"constraints": ["Alpha bound one.", "Beta bound two.", "Gamma bound three."]}),
                              encoding="utf-8")
    note = tr.write_note(tmp_path)
    assert "Alpha bound one; Beta bound two; Gamma bound three." in note
    assert "say plainly in the limitations" in note


def test_write_note_is_empty_when_requirements_have_no_constraints(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    fi.mkdir()
    (fi / tr.FILE).write_text(json.dumps({"factors": [{"name": "alpha"}]}), encoding="utf-8")
    assert tr.write_note(tmp_path) == ""


def test_the_cover_reader_counts_a_metric_of_an_estimate_for_a_recovery_topic() -> None:
    """r5 kinetics: "how accurately can k and Ea be recovered" with k/Ea error, bias, spread and coverage metrics was read
    as "no metric reports k itself" and the quest stopped. A metric of the estimate reports the quantity it estimates."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "agents" / "topic_cover.md").read_text(encoding="utf-8")
    assert "how well that quantity is estimated, fitted or recovered" in text
    assert "a metric of its estimate" in text


@pytest.mark.asyncio
async def test_a_rejected_answer_for_the_topic_is_asked_again_with_the_reason_and_the_second_one_is_used(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(grid={**LID_AXIS, "room_temperature_C": [10, 20, 30]}, metrics=[METRIC_TIME, METRIC_HEAT],
                               model={**MODEL, "equations": [*MODEL["equations"], NEWTON]}))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    redesign = copy.deepcopy(plan.parse(_plan_text(eng)).design)
    del redesign["protocol"]["grid"]["lid_material"]
    state = {"topic": TOPIC, "iteration": 1, "design": redesign}
    original = model.chat
    prompts: list[str] = []
    bad = {**redesign, "protocol": {**redesign["protocol"], "grid": {**redesign["protocol"]["grid"], "lid_material": []}}}
    good = {**redesign, "protocol": {**redesign["protocol"], "grid": {**redesign["protocol"]["grid"], **LID_AXIS}}}

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if "The topic asks for things this plan does not yet include" in prompt:
            prompts.append(prompt)
            return json.dumps(bad if len(prompts) == 1 else good)
        return await original(messages, **kw)

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    design, _sha = await eng._hold_the_design_to_the_topic(state, redesign, "")
    assert design["protocol"]["grid"]["lid_material"] == ["paper", "foam", "steel"]
    assert len(prompts) == 2 and "Your last answer was not used" not in prompts[0]
    assert "protocol.grid.lid_material" in prompts[1] and "non-empty list" in prompts[1]
    record = json.loads((eng.fi_dir / "topic_design_asked.json").read_text(encoding="utf-8"))
    assert record["asks"] == 2
    await eng._hold_the_design_to_the_topic(state, redesign, "")  # a resume never asks past the bound
    assert len(prompts) == 2


async def _redesign_setup(tmp_path: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    eng = _engine(tmp_path, [])
    model = Reader(eng, _draft(grid={**LID_AXIS, "room_temperature_C": [10, 20, 30]}, metrics=[METRIC_TIME, METRIC_HEAT],
                               model={**MODEL, "equations": [*MODEL["equations"], NEWTON]}))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    redesign = copy.deepcopy(plan.parse(_plan_text(eng)).design)
    del redesign["protocol"]["grid"]["lid_material"]
    return eng, model, redesign, {"topic": TOPIC, "iteration": 1, "design": redesign}


@pytest.mark.asyncio
async def test_the_stop_names_the_design_steps_real_ask_count_and_why_its_last_answer_was_not_used(tmp_path: Path) -> None:
    eng, _model, redesign, state = await _redesign_setup(tmp_path)  # the model never adds the axis
    await eng._hold_the_design_to_the_topic(state, redesign, "")
    record = json.loads((eng.fi_dir / "topic_design_asked.json").read_text(encoding="utf-8"))
    assert record["asks"] == 4 and record["problem"]  # written after the last failed attempt too
    stop = json.loads((eng.fi_dir / "topic_unmet.json").read_text(encoding="utf-8"))
    assert stop["asked"] == 4
    assert record["problem"] in eng._topic_last_problem(1)
    assert any(record["problem"][:40] in t for t in stop["tried"])


@pytest.mark.asyncio
async def test_a_failed_call_for_the_topic_is_noted_and_the_next_try_is_still_made(tmp_path: Path) -> None:
    eng, model, redesign, state = await _redesign_setup(tmp_path)
    original = model.chat
    prompts: list[str] = []
    good = {**redesign, "protocol": {**redesign["protocol"], "grid": {**redesign["protocol"]["grid"], **LID_AXIS}}}

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if "The topic asks for things this plan does not yet include" in prompt:
            prompts.append(prompt)
            if len(prompts) == 1:
                raise RuntimeError("connection reset")
            return json.dumps(good)
        return await original(messages, **kw)

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    design, _sha = await eng._hold_the_design_to_the_topic(state, redesign, "")
    assert design["protocol"]["grid"]["lid_material"] == ["paper", "foam", "steel"]
    assert len(prompts) == 2 and "connection reset" in prompts[1]
