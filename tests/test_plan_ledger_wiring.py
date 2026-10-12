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
    draft["protocol"] = {"oracles": copy.deepcopy(HEAT_SINK["protocol"]["oracles"]), "grid": GOOD_GRID, **protocol}
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
            return answer(plan.plan_path(self.eng.quest_root).read_text(encoding="utf-8")) if callable(answer) else answer
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
    for name in ("probe_relations", "second_impl_write", "second_impl_adjudicate"):
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
    assert "- vary or compare fin spacing: 2 to 4" in plan_prompt and "- report base temperature" in plan_prompt
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
    assert "- vary or compare fin spacing" in eng._design_prompt(_state())
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
