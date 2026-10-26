"""The plan side of the correct-at-source design, as the engine wires it: what the topic asks for is read BEFORE the plan is
asked for and shown to it as a list; the plan directive asks for simplifications, sourced definitions and step cards; the
parts the plan wrote badly are asked of its model once (bounded, kept across a resume); a quote that is not in the sources
is one more thing asked, never a stop; a gap that stays is said in plan.md and to the writer; `.fi/ledger.json` is written.
Fake models only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import engine as engine_mod
from core import plan, plan_ledger, plan_steps
from core import topic_requirements as tr
from tests.plan_parts_helpers import as_parts
from tests.test_plan_optimisation import EXTRA, HEAT_SINK, _engine, _NO_OBJECTIONS

TOPIC = "Measure how fin spacing changes the base temperature of a heat sink, with fins from 2 to 4 mm apart"
ASK = "are missing or incomplete"
GOOD_GRID = {"fin_spacing": [2.0, 3.0, 4.0]}
METRIC = {"id": "base_temperature", "kind": "mean", "estimand": "mean base temperature", "unit": "one simulated heat sink"}
READING = {"factors": [{"name": "fin spacing", "levels": [], "range": [2, 4], "stated": "fins from 2 to 4 mm apart",
                        "main": True}],
           "quantities": [{"name": "base temperature", "stated": "the base temperature", "main": True}],
           "formulas": [], "comparisons": [], "constraints": [], "stated_results": []}
SIMPLIFICATION = {"id": "X1", "replaces": "radiation from the fins", "because": "it is small at these temperatures",
                  "effect": "the base temperature is a little too high", "probe": "switch radiation on in one run and compare"}
SOURCE_TEXT = "The base temperature is defined as the mean temperature of the plate that touches the chip, in kelvin."
DEFINITION = {"metric": "base_temperature", "defined_as": "mean plate temperature", "source": "[1]",
              "quote": "the mean temperature of the plate that touches the chip", "differs": ""}
LITERATURE = [{"content": SOURCE_TEXT, "metadata": {"title": "Heat sinks", "doi": "10.1/hs", "source": "openalex"}}]


def _draft(**protocol: Any) -> dict[str, Any]:
    draft = copy.deepcopy(HEAT_SINK)
    draft["study_type"] = "measure"
    draft["protocol"] = {"oracles": copy.deepcopy(HEAT_SINK["protocol"]["oracles"]), "grid": GOOD_GRID,
                         "topic_map": [{"item": "F1", "by": "setting:fin_spacing"}], **protocol}
    return draft


def _set(**parts: Any) -> Any:
    def answer(text: str) -> str:
        out = plan.edit_design_block(text, lambda d: {**d, "protocol": {**d["protocol"], **parts}})
        assert out is not None
        return out
    return answer


class Model:
    """Reads the topic, drafts the plan, answers each request for missing parts from ``answers`` (plan.md edited by a
    function, or text), and finds no objection."""

    def __init__(self, eng: Any, answers: list[Any], draft: dict[str, Any], reading: dict[str, Any] | None = READING) -> None:
        self.eng, self.answers, self.draft, self.reading = eng, list(answers), draft, reading
        self.asked: list[str] = []
        self.prompts: list[tuple[str, str]] = []  # (kind, prompt)
        self.drafted = False
        eng.config.topic = TOPIC
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if prompt.lstrip().startswith("# What a Research Topic Asks For"):
            self.prompts.append(("reading", prompt))
            if self.reading is None:
                raise RuntimeError("no answer")
            return json.dumps(self.reading)
        if ASK in prompt:
            self.asked.append(prompt)
            self.prompts.append(("parts", prompt))
            answer = self.answers.pop(0) if self.answers else None
            if answer is None:
                raise RuntimeError("no answer")
            return as_parts(answer(plan.plan_path(self.eng.quest_root).read_text(encoding="utf-8")) if callable(answer) else answer)
        if prompt.lstrip().startswith("# Is It In The Plan?"):
            return json.dumps({"items": []})
        if not self.drafted:
            self.drafted = True
            self.prompts.append(("plan", prompt))
            return json.dumps({**self.draft, "plan": EXTRA})
        return _NO_OBJECTIONS


def _record(eng: Any) -> dict[str, Any]:
    return json.loads((eng.fi_dir / "protocol_part_asked.json").read_text(encoding="utf-8"))


def _plan_text(eng: Any) -> str:
    return plan.plan_path(eng.quest_root).read_text(encoding="utf-8")


def _state(**more: Any) -> dict[str, Any]:
    return {"topic": TOPIC, "iteration": 0, **more}


# -- B0 and the directive -------------------------------------------------------------------------------------------


def test_the_new_prompts_are_registered() -> None:
    prompts = engine_mod._load_prompts()
    for name in ("second_impl_write", "second_impl_adjudicate"):
        assert name in prompts


def test_the_plan_directive_asks_for_simplifications_definitions_and_step_cards() -> None:
    text = engine_mod._PLAN_DIRECTIVE
    assert plan_ledger.DIRECTIVE_TEXT in text and plan_steps.DIRECTIVE_TEXT in text
    assert "@@" not in text
    assert "`simplifications`" in text and "`definitions`" in text and "protocol.model.steps" in text
    assert "word for word" in text and "differs" in text
    assert "Mistakes that stop a run" in text and "no entry in `simplifications`" in text


# -- the list of what the topic asks for, read before the plan ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_plan_is_shown_what_the_topic_asks_for_and_the_reading_is_made_before_it(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _draft(metrics=[METRIC]))
    await eng._node_plan(_state())
    kinds = [k for k, _ in model.prompts]
    assert kinds.index("reading") < kinds.index("plan"), "the topic is read before the plan is asked for"
    plan_prompt = dict(model.prompts)["plan"]
    assert "What the topic asks for (each item must appear as a setting, a metric or an equation" in plan_prompt
    assert "- F1: vary or compare fin spacing: 2 to 4" in plan_prompt and "- Q1: report base temperature" in plan_prompt
    assert plan_prompt.index("What the topic asks for (") < plan_prompt.index("Also write the plan")
    assert kinds.count("reading") == 1, "read once; the cover step reuses the kept reading"


@pytest.mark.asyncio
async def test_a_reading_that_failed_or_is_empty_gives_the_prompt_as_before(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _draft(), reading=None)
    await eng._node_plan(_state())
    assert "What the topic asks for (" not in dict(model.prompts)["plan"]
    assert eng._topic_checklist(_state()) == ""
    empty = _engine(tmp_path / "e", [])
    Model(empty, [], _draft(), reading={k: [] for k in READING})
    await empty._read_topic_before_the_prompt(_state())
    assert empty._topic_checklist(_state()) == ""


@pytest.mark.asyncio
async def test_the_design_prompt_has_the_list_only_when_the_quest_runs_code(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], _draft())
    await eng._read_topic_before_the_prompt(_state())
    assert "- F1: vary or compare fin spacing" in eng._design_prompt(_state())
    assert "What the topic asks for (" not in eng._design_prompt(_state(no_simulation_resolved=True))


# -- the parts request: once, kept, never a stop ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_metric_with_no_definition_is_asked_once_and_the_answer_is_kept(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_set(simplifications=[SIMPLIFICATION], definitions=[DEFINITION])], _draft(metrics=[METRIC]))
    await eng._node_plan(_state(literature=LITERATURE))
    assert len(model.asked) == 1 and "`protocol.definitions`" in model.asked[0]
    assert "the metric base_temperature has no definition" in model.asked[0]
    assert "Parts of this plan's protocol are missing or incomplete" in model.asked[0]
    assert "definitions" in _record(eng)["parts"]
    text = _plan_text(eng)
    assert "FI asked the plan's model to write `protocol.definitions` again, and it is complete now" in text
    design, _ = plan.load_design(eng.quest_root)
    assert design["protocol"]["definitions"][0]["quote"] == DEFINITION["quote"]
    await eng._node_design(_state(literature=LITERATURE))  # the design step asks nothing more
    assert len(model.asked) == 1


@pytest.mark.asyncio
async def test_a_gap_that_stays_is_said_in_the_plan_and_to_the_writer_and_never_stops_a_resume_asks_no_more(
        tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _draft(metrics=[METRIC]))  # the model gives no answer
    await eng._node_plan(_state())
    count = _record(eng)["count"]
    assert len(model.asked) >= 1 and count >= 1
    text = _plan_text(eng)
    assert "## What the topic asks for" in text and "`protocol.definitions` is incomplete" in text
    assert "it still has gaps (the experiment goes on; the paper says so)" in text
    note = tr.write_note(eng.quest_root)
    assert "simplifications or definitions of this study are not fully stated" in note
    asked_before = len(model.asked)
    patch = await eng._node_design(_state())  # nothing stops, nothing is asked again
    assert "design" in patch and len(model.asked) == asked_before
    await eng._node_design(_state())
    assert len(model.asked) == asked_before
    assert text.count("`protocol.definitions` is incomplete") == _plan_text(eng).count("`protocol.definitions` is incomplete")


@pytest.mark.asyncio
async def test_a_quote_that_is_not_in_the_sources_is_asked_for_and_never_a_stop(tmp_path: Path) -> None:
    invented = {**DEFINITION, "quote": "the temperature at the hottest point of the lid"}
    eng = _engine(tmp_path, [])
    model = Model(eng, [_set(definitions=[DEFINITION])],
                  _draft(metrics=[METRIC], simplifications=[SIMPLIFICATION], definitions=[invented]))
    await eng._node_plan(_state(literature=LITERATURE))
    assert len(model.asked) == 1
    assert "the quote for base_temperature is not in the sources FI found" in model.asked[0]
    design, _ = plan.load_design(eng.quest_root)
    assert design["protocol"]["definitions"][0]["quote"] == DEFINITION["quote"]
    assert eng._ledger_unread(design, SOURCE_TEXT) == {}


@pytest.mark.asyncio
async def test_a_quote_that_is_in_the_sources_asks_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _draft(metrics=[METRIC], simplifications=[SIMPLIFICATION], definitions=[DEFINITION]))
    await eng._node_plan(_state(literature=LITERATURE))
    assert model.asked == [] and not (eng.fi_dir / "protocol_part_asked.json").exists()


def test_with_no_sources_found_a_quote_is_not_judged_and_a_definition_marked_as_the_plans_own_needs_no_source() -> None:
    eng = engine_mod.Engine.__new__(engine_mod.Engine)
    eng._log = engine_mod.logging.getLogger("test")
    own = {"metric": "base_temperature", "defined_as": "my own", "source": "", "quote": "", "differs": "X1"}
    protocol = {"metrics": [METRIC], "simplifications": [SIMPLIFICATION], "definitions": [own]}
    assert eng._ledger_unread({"protocol": protocol}, "some text") == {}
    cited = {**DEFINITION, "quote": "words that are nowhere"}
    quoted = {"metrics": [METRIC], "simplifications": [SIMPLIFICATION], "definitions": [cited]}
    assert eng._ledger_unread({"protocol": quoted}, "") == {}, "no passages: nothing to judge against"
    assert "definitions" in eng._ledger_unread({"protocol": quoted}, "other text entirely")


def test_a_step_card_that_cannot_be_read_is_one_more_part_to_ask() -> None:
    eng = engine_mod.Engine.__new__(engine_mod.Engine)
    eng._log = engine_mod.logging.getLogger("test")
    protocol = {"model": {"summary": "s", "equations": [], "steps": [{"id": "S1", "name": "n"}]}}
    assert list(eng._ledger_unread({"protocol": protocol})) == ["steps"]


def test_an_old_plan_without_the_new_keys_behaves_as_before() -> None:
    eng = engine_mod.Engine.__new__(engine_mod.Engine)
    eng._log = engine_mod.logging.getLogger("test")
    assert eng._ledger_unread({"protocol": {"grid": GOOD_GRID}}) == {}
    assert eng._ledger_unread({"hypothesis": "h"}) == {}
    assert eng._ledger_unread(None) == {}


@pytest.mark.asyncio
async def test_an_old_plan_with_no_metrics_asks_nothing_and_still_runs(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _draft())
    await eng._node_plan(_state())
    patch = await eng._node_design(_state())
    assert model.asked == [] and "design" in patch
    assert "FI asked the plan's model to write `protocol.definitions`" not in _plan_text(eng)


@pytest.mark.asyncio
async def test_step_cards_are_taken_from_the_answer_without_touching_the_rest_of_the_model(tmp_path: Path) -> None:
    model_block = {"summary": "A fin array", "assumptions": ["steady"],
                   "equations": [{"id": "E1", "formula": "q = h*A*dT", "role": "generates", "source": "derivation",
                                  "derivation": "Newton cooling"}]}
    bad = [{"id": "S1", "name": "n"}]
    good = [{"id": "S1", "name": "mesh", "does": "builds the mesh", "inputs": {"n": "count"}, "output": "cells"}]
    answer = _set(model={**model_block, "steps": good, "summary": "CHANGED"})
    eng = _engine(tmp_path, [])
    m = Model(eng, [answer], _draft(model={**model_block, "steps": bad}))
    await eng._node_plan(_state())
    assert len(m.asked) == 1 and "`protocol.model.steps`" in m.asked[0]
    design, _ = plan.load_design(eng.quest_root)
    assert design["protocol"]["model"]["steps"][0]["does"] == "builds the mesh"
    assert design["protocol"]["model"]["summary"] == "A fin array", "only the steps are taken from the answer"


# -- the ledger file --------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_ledger_is_written_at_the_end_of_the_plan_and_keeps_what_later_steps_added(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], _draft(metrics=[METRIC], simplifications=[SIMPLIFICATION], definitions=[DEFINITION]))
    await eng._node_plan(_state(literature=LITERATURE))
    record = plan_ledger.read_ledger(eng.fi_dir / "ledger.json")
    assert record is not None and record["plan_sha"] == plan.sha256(_plan_text(eng))
    assert [i["id"] for i in record["items"]] == ["X1"] and record["items"][0]["status"] == "not_tested"
    assert plan_ledger.untested(record)[0]["replaces"] == "radiation from the fins"
    # a later step attaches a relation; a rewrite of the ledger for a changed plan keeps it
    record["items"][0]["relation_ids"] = ["R1"]
    plan_ledger.write_ledger(eng.fi_dir / "ledger.json", {**record, "plan_sha": "old"})
    eng._write_the_ledger(_state())
    assert plan_ledger.read_ledger(eng.fi_dir / "ledger.json")["items"][0]["relation_ids"] == ["R1"]


@pytest.mark.asyncio
async def test_a_plan_with_no_simplifications_still_gets_a_ledger_file(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], _draft())
    await eng._node_plan(_state())
    record = plan_ledger.read_ledger(eng.fi_dir / "ledger.json")
    assert record is not None and record["items"] == []


# -- the reader prompts ------------------------------------------------------------------------------------------------


def test_the_two_reader_prompts_ask_about_what_is_approximated_without_an_entry() -> None:
    critique = (Path(engine_mod.__file__).parent.parent / "agents" / "design_self_critique.md").read_text(encoding="utf-8")
    review = (Path(engine_mod.__file__).parent.parent / "agents" / "oracle_review.md").read_text(encoding="utf-8")
    assert "protocol.simplifications" in critique and "protocol.definitions" in critique
    assert "`simplifications`" in review and "definition matches the source it quotes" in review


# -- review fixes ---------------------------------------------------------------------------------------------------------


def _texts(eng: Any, **protocol: Any) -> tuple[str, dict[str, Any]]:
    """plan.md as it stands, and the design block with ``protocol`` keys set (what the model's JSON answer carries)."""
    before = _plan_text(eng)
    after = plan.edit_design_block(before, lambda d: {**d, "protocol": {**d["protocol"], **protocol}})
    assert after is not None
    block = plan.raw_design_block(after)
    assert isinstance(block, dict)
    return before, block


@pytest.mark.asyncio
async def test_an_answer_whose_ledger_parts_are_malformed_is_not_taken_and_the_exact_reason_goes_back(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], _draft(metrics=[METRIC], simplifications=[SIMPLIFICATION], definitions=[DEFINITION]))
    await eng._node_plan(_state(literature=LITERATURE))
    # a simplification with no `because`
    before, after = _texts(eng, simplifications=[{"id": "X9", "replaces": "r", "effect": "e", "probe": "p"}])
    with pytest.raises(ValueError, match="simplification X9 has no `because`"):
        eng._keep_only_the_parts(before, after, ["simplifications"])
    # a definition with no quote
    before, after = _texts(eng, definitions=[{"metric": "base_temperature", "defined_as": "x", "source": "[1]"}])
    with pytest.raises(ValueError, match="not taken from a source"):
        eng._keep_only_the_parts(before, after, ["definitions"])
    # a quote that is not in the sources
    before, after = _texts(eng, definitions=[{**DEFINITION, "quote": "words that are nowhere"}])
    with pytest.raises(ValueError, match="is not in the sources FI found"):
        eng._keep_only_the_parts(before, after, ["definitions"], "other text entirely")
    assert eng._keep_only_the_parts(before, after, ["definitions"], "") is not None, "no sources: not judged"
    # a step card with no name
    before, after = _texts(eng, model={"summary": "s", "equations": [], "steps": [{"id": "S1"}]})
    with pytest.raises(ValueError, match="step S1 has no `name`"):
        eng._keep_only_the_parts(before, after, ["steps"])


@pytest.mark.asyncio
async def test_steps_are_taken_from_the_protocol_top_level_too_and_a_missing_old_model_is_empty(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], _draft())
    await eng._node_plan(_state())
    card = {"id": "S1", "name": "mesh", "does": "builds the mesh", "inputs": {"n": "count"}, "output": "cells"}
    before, after = _texts(eng, steps=[card])
    kept = eng._keep_only_the_parts(before, after, ["steps"])
    design = plan.parse(kept).design
    assert design["protocol"]["model"]["steps"][0]["name"] == "mesh"
    before, after = _texts(eng, note="nothing")
    with pytest.raises(ValueError, match="step cards are missing"):
        eng._keep_only_the_parts(before, after, ["steps"])


@pytest.mark.asyncio
async def test_ledger_parts_are_asked_afresh_for_a_new_plan_or_round_and_not_for_fi_own_notes(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _draft(metrics=[METRIC]))
    await eng._node_plan(_state())
    assert len(model.asked) == 1, "one request: the ledger parts are asked once"
    assert _record(eng)["count"] == 1
    design, sha = eng._design_from_plan()
    await eng._ask_plan_for_the_parts(_state(), design, sha)
    assert len(model.asked) == 1, "FI's notes changed plan.md; the same problem is not asked again"
    other = {**METRIC, "id": "wall_flux"}
    design["protocol"]["metrics"] = [METRIC, other]
    await eng._ask_plan_for_the_parts(_state(), design, sha)
    assert len(model.asked) == 2, "a plan with another problem (a revised plan, --from plan) is asked afresh"
    await eng._ask_plan_for_the_parts(_state(iteration=1), design, sha)
    assert len(model.asked) == 3, "another design round is asked afresh"


def test_a_ledger_record_whose_iteration_is_not_a_number_is_unreadable_and_does_not_crash(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    record = {"iteration": "x", "model": eng._plan_model_name(), "seen": {"base_temperature": "abc"}}
    (eng.fi_dir / engine_mod._LEDGER_ASKED).write_text(json.dumps(record), encoding="utf-8")
    assert eng._ledger_seen(0) == {}, "a record that cannot be read is treated as no record"
    record["iteration"] = 0
    (eng.fi_dir / engine_mod._LEDGER_ASKED).write_text(json.dumps(record), encoding="utf-8")
    assert eng._ledger_seen(0) == {"base_temperature": "abc"}, "a valid record is still read as before"


def test_the_plan_step_rerun_clears_the_ledger_and_the_parts_records() -> None:
    from core import rerun_from
    for name in (".fi/protocol_part_asked.json", ".fi/ledger.json", ".fi/ledger_asked.json"):
        assert name in rerun_from._OWN["plan"] and name in rerun_from._OWN["literature"]


@pytest.mark.asyncio
async def test_metrics_the_plan_defines_itself_are_recorded_and_the_writer_is_told(tmp_path: Path) -> None:
    own = {"metric": "base_temperature", "defined_as": "the plate temperature", "source": "", "quote": "", "differs": "X1"}
    eng = _engine(tmp_path, [])
    Model(eng, [], _draft(metrics=[METRIC], simplifications=[SIMPLIFICATION], definitions=[own]))
    await eng._node_plan(_state(literature=LITERATURE))
    assert eng._topic_covered_read()["unsourced_definitions"] == ["base_temperature"]
    assert plan_ledger.read_ledger(eng.fi_dir / "ledger.json")["unsourced_definitions"] == ["base_temperature"]
    note = tr.write_note(eng.quest_root)
    assert "the study's own definitions" in note and "base_temperature" in note and "in the limitations" in note
    sourced = _engine(tmp_path / "s", [])
    Model(sourced, [], _draft(metrics=[METRIC], simplifications=[SIMPLIFICATION], definitions=[DEFINITION]))
    await sourced._node_plan(_state(literature=LITERATURE))
    assert "own definitions" not in tr.write_note(sourced.quest_root)


@pytest.mark.asyncio
async def test_a_topic_item_the_plan_drops_without_a_simplification_is_a_problem_of_the_simplifications(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    reading = {**READING, "factors": [{"name": "fin material", "levels": ["copper", "steel"], "range": [], "stated": "x",
                                       "main": False}]}
    Model(eng, [], _draft(metrics=[METRIC], definitions=[DEFINITION]), reading=reading)
    await eng._read_topic_before_the_prompt(_state())
    design = _draft(metrics=[METRIC], definitions=[DEFINITION], simplifications=[SIMPLIFICATION])
    problem = eng._ledger_unread(design, "")["simplifications"]
    assert "the topic asks for fin material" in problem and "`simplifications`" in problem
    named = {"id": "X2", "replaces": "fin material", "because": "one material only", "effect": "none known",
             "probe": "run steel once"}
    design = _draft(metrics=[METRIC], definitions=[DEFINITION], simplifications=[named])
    assert "simplifications" not in eng._ledger_unread(design, "")


@pytest.mark.asyncio
async def test_a_redesign_gets_its_ledger_checked_and_written(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _draft(metrics=[METRIC], simplifications=[SIMPLIFICATION], definitions=[DEFINITION]))
    await eng._node_plan(_state(literature=LITERATURE))
    asked = len(model.asked)
    redraft = _draft(metrics=[METRIC], simplifications=[{**SIMPLIFICATION, "id": "X7"}],
                     model={"summary": "s", "equations": [], "steps": [{"id": "S1"}]})  # a step card with no name
    model.draft, model.drafted = redraft, False
    patch = await eng._node_design(_state(iteration=1, design={"hypothesis": "old"}, literature=LITERATURE))
    assert "design" in patch and len(model.asked) == asked, "nothing is asked at a redesign; nothing stops"
    record = plan_ledger.read_ledger(eng.fi_dir / "ledger.json")
    assert [i["id"] for i in record["items"]] == ["X7"]
    assert "`protocol.steps` is incomplete" in " ".join(eng._topic_covered_read()["ledger_left"])


def test_the_writer_is_told_when_the_method_steps_are_incomplete(tmp_path: Path) -> None:
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "topic_covered.json").write_text(json.dumps(
        {"ledger_left": ["`protocol.steps` is incomplete: step S1 has no `name`"]}), encoding="utf-8")
    assert "steps of the method are not all described in full" in tr.write_note(tmp_path)


# -- the real wiring, with the topic read before the plan (scripted replies, as the older tests have them) -----------


@pytest.mark.asyncio
async def test_scripted_replies_with_the_topic_read_first_and_a_resume_that_reads_it_no_more(tmp_path: Path) -> None:
    from tests.test_plan import _engine as scripted_engine
    from tests.test_plan import _plan_reply
    audit = json.dumps({"objections_addressed": []})
    eng = scripted_engine(tmp_path, [json.dumps(READING), _plan_reply(), audit])
    eng.config.topic = TOPIC
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    prompts = [c.args[0][-1]["content"] for c in eng._client.chat.await_args_list]
    assert prompts[0].lstrip().startswith("# What a Research Topic Asks For")
    assert "What the topic asks for (" in prompts[1] and "- F1: vary or compare fin spacing" in prompts[1]
    assert (eng.fi_dir / "topic_requirements.json").is_file()
    calls = eng._client.chat.await_count
    await eng._node_plan({"topic": TOPIC, "iteration": 0})  # a resume: plan.md is read, nothing is asked again
    assert eng._client.chat.await_count == calls


@pytest.mark.asyncio
async def test_design_critique_style_replies_with_the_topic_read_first(tmp_path: Path) -> None:
    from tests.test_design_self_critique import _AMENDED_DESIGN, _DRAFT_DESIGN, _mk_cfg
    eng = engine_mod.Engine(_mk_cfg(tmp_path))
    replies = [json.dumps(READING), json.dumps(_DRAFT_DESIGN),
               json.dumps({"objections_addressed": [{"check": "other", "where": "hypothesis and method", "objection": "o", "fix": "f"}],
                           "amended_design": _AMENDED_DESIGN})]
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=replies)})()
    patch = await eng._node_design({"topic": "OPC", "iteration": 0})
    assert patch["design"] == _AMENDED_DESIGN
    assert eng._client.chat.await_args_list[0].args[0][-1]["content"].lstrip().startswith("# What a Research Topic Asks For")
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=[json.dumps(_DRAFT_DESIGN), json.dumps({})])})()
    await eng._node_design({"topic": "OPC", "iteration": 1})  # the reading is kept: the next design round asks no reading
    assert not any("A Research Topic Asks For" in c.args[0][-1]["content"] for c in eng._client.chat.await_args_list)
