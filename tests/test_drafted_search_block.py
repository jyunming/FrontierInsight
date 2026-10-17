"""A design drafted at the design step (a redesign: it did not come from plan.md) that is a search for the best design
with a missing or unusable block is completed the same bounded way the plan's is, whatever the design's source. Fake model."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import plan
from tests.test_plan_optimisation import BLOCK, EXTRA, HEAT_SINK, Paused, _engine, _NO_OBJECTIONS

TOPIC = "Find the fin spacing and thickness that minimise the base temperature of a heat sink"
ASK = "Answer with the `optimisation` block alone"


class Model:
    """The plan draft first; an answer to each request for the block of a drafted design; no objections otherwise."""

    def __init__(self, eng: Any, answers: list[Any]) -> None:
        self.eng, self.answers = eng, list(answers)
        self.asked: list[str] = []
        self.drafted = False
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if ASK in prompt:
            self.asked.append(prompt)
            answer = self.answers.pop(0) if self.answers else None
            if answer is None:
                raise RuntimeError("no answer")
            return answer if isinstance(answer, str) else json.dumps(answer)
        if not self.drafted:
            self.drafted = True
            return json.dumps({**HEAT_SINK, "plan": EXTRA})
        return _NO_OBJECTIONS


def _no_budget() -> dict[str, Any]:
    design = copy.deepcopy(HEAT_SINK)
    del design["protocol"]["optimisation"]["evaluation_budget"]
    return design


async def _setup(tmp_path: Path, answers: list[Any]) -> tuple[Any, Model]:
    eng = _engine(tmp_path, [])
    model = Model(eng, answers)
    await eng._node_plan({"topic": TOPIC, "iteration": 0})  # plan.md holds a complete search block
    return eng, model


def _record(eng: Any) -> dict[str, Any]:
    return json.loads((eng.fi_dir / "drafted_search_asked.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_redesign_without_a_budget_takes_the_plans_block_when_it_is_the_same_search(tmp_path: Path) -> None:
    eng, model = await _setup(tmp_path, [])
    out = await eng._complete_a_drafted_search({"topic": TOPIC}, _no_budget(), iteration=1)
    assert out["protocol"]["optimisation"]["evaluation_budget"] == BLOCK["evaluation_budget"]
    assert out["study_type"] == "find_best_design"
    assert model.asked == [] and not (eng.fi_dir / "drafted_search_asked.json").exists()


@pytest.mark.asyncio
async def test_a_redesign_with_other_settings_is_asked_and_goes_on(tmp_path: Path) -> None:
    eng, model = await _setup(tmp_path, [None])
    drafted = _no_budget()
    drafted["protocol"]["optimisation"]["design_variables"] = [
        {"name": "fin_height", "low": 5.0, "high": 20.0, "kind": "continuous"}]
    drafted["protocol"]["optimisation"]["baseline"] = {"values": {"fin_height": 10.0}, "source": "the topic"}
    answer = copy.deepcopy(drafted["protocol"]["optimisation"])
    answer["evaluation_budget"] = {"starts": 3, "per_start": 40}
    model.answers = [{"optimisation": answer}]
    out = await eng._complete_a_drafted_search({"topic": TOPIC}, drafted, iteration=1)
    block = out["protocol"]["optimisation"]
    assert block["evaluation_budget"] == {"starts": 3, "per_start": 40}
    assert [v["name"] for v in block["design_variables"]] == ["fin_height"], "the plan's other search was not swapped in"
    assert len(model.asked) == 1 and "evaluation budget" in model.asked[0] and TOPIC in model.asked[0]
    assert _record(eng)["count"] == 1 and _record(eng)["iteration"] == 1


@pytest.mark.asyncio
async def test_two_unusable_answers_end_in_the_stop_and_a_resume_asks_no_more(tmp_path: Path) -> None:
    eng, model = await _setup(tmp_path, [])
    drafted = _no_budget()
    drafted["protocol"]["optimisation"]["design_variables"] = [{"name": "fin_height", "low": 5.0, "high": 20.0}]
    drafted["protocol"]["optimisation"]["baseline"] = {"values": {"fin_height": 10.0}, "source": "the topic"}
    model.answers = ["not json at all", {"objective": {"quantity": "x"}}, {"objective": {"quantity": "x"}}]
    out = await eng._complete_a_drafted_search({"topic": TOPIC}, drafted, iteration=1)
    assert "evaluation_budget" not in out["protocol"]["optimisation"] and len(model.asked) == 2
    again = await eng._complete_a_drafted_search({"topic": TOPIC}, drafted, iteration=1)  # a resume
    assert len(model.asked) == 2, "never asked beyond the bound"
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    eng._stop_if_the_search_cannot_start({"topic": TOPIC}, again)  # no pause: nobody is asked to pick another model
    assert seen == []
    stop = eng._search_cannot_start_stop({"topic": TOPIC}, again)
    assert stop is not None
    steps = " ".join(stop["exec_result"]["no_search_tried"])
    assert "asked the design step's model 2 time(s)" in steps and "never ran as a sweep" not in steps


@pytest.mark.asyncio
async def test_a_new_design_round_or_model_is_asked_afresh_and_a_measurement_is_never_asked(tmp_path: Path) -> None:
    eng, model = await _setup(tmp_path, [])
    drafted = _no_budget()
    drafted["protocol"]["optimisation"]["design_variables"] = [{"name": "fin_height", "low": 5.0, "high": 20.0}]
    drafted["protocol"]["optimisation"]["baseline"] = {"values": {"fin_height": 10.0}, "source": "the topic"}
    model.answers = [None, None, None, None]
    await eng._complete_a_drafted_search({"topic": TOPIC}, drafted, iteration=1)
    assert len(model.asked) == 2
    await eng._complete_a_drafted_search({"topic": TOPIC}, drafted, iteration=2)
    assert len(model.asked) == 4
    measurement = {"hypothesis": "h", "study_type": "measure", "protocol": {"grid": {"x": [1, 2]}}}
    out = await eng._complete_a_drafted_search({"topic": TOPIC}, measurement, iteration=3)
    assert out == measurement and len(model.asked) == 4


@pytest.mark.asyncio
async def test_the_design_step_asks_for_the_block_of_a_redesign(tmp_path: Path) -> None:
    eng, model = await _setup(tmp_path, [])
    model.drafted = False  # the next unrelated call is the redesign's draft
    redraft = _no_budget()
    redraft["protocol"]["optimisation"]["design_variables"] = [{"name": "fin_height", "low": 5.0, "high": 20.0}]
    redraft["protocol"]["optimisation"]["baseline"] = {"values": {"fin_height": 10.0}, "source": "the topic"}
    model.draft = redraft
    answer = copy.deepcopy(redraft["protocol"]["optimisation"])
    answer["evaluation_budget"] = {"starts": 2, "per_start": 30}
    model.answers = [answer]
    original = model.chat

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        if ASK not in messages[-1]["content"] and not model.drafted:
            model.drafted = True
            return json.dumps({**redraft, "plan": EXTRA})
        return await original(messages, **kw)

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    patch = await eng._node_design({"topic": TOPIC, "iteration": 1, "design": {"hypothesis": "old"}})
    assert seen == []
    assert patch["design"]["protocol"]["optimisation"]["evaluation_budget"] == {"starts": 2, "per_start": 30}
    assert plan.plan_path(eng.quest_root).is_file()


@pytest.mark.asyncio
async def test_after_the_freeze_the_frozen_block_is_used_and_no_model_is_asked(tmp_path: Path) -> None:
    from core import frozen_protocol as fp

    eng, model = await _setup(tmp_path, [])
    frozen = {"grid": {"x": [1, 2]}, "runs_per_setting": 3, "optimisation": copy.deepcopy(BLOCK)}
    fp.freeze(eng.quest_root, frozen, approved_by="human: test", source="plan.md")
    drafted = _no_budget()
    drafted["protocol"]["optimisation"]["design_variables"] = [{"name": "fin_height", "low": 5.0, "high": 20.0}]
    out = await eng._complete_a_drafted_search({"topic": TOPIC}, drafted, iteration=1)
    assert out["protocol"]["optimisation"]["evaluation_budget"] == BLOCK["evaluation_budget"]
    assert model.asked == []


@pytest.mark.asyncio
async def test_a_redesign_that_leaves_its_objective_out_is_not_the_same_search_as_the_plans(tmp_path: Path) -> None:
    eng, model = await _setup(tmp_path, [])
    drafted = _no_budget()  # the plan's own variables, but no objective stated
    del drafted["protocol"]["optimisation"]["objective"]
    answer = copy.deepcopy(BLOCK)
    answer["evaluation_budget"] = {"starts": 2, "per_start": 20}
    model.answers = [answer]
    out = await eng._complete_a_drafted_search({"topic": TOPIC}, drafted, iteration=1)
    assert len(model.asked) == 1, "the model is asked; the plan's block is not copied"
    assert out["protocol"]["optimisation"]["evaluation_budget"] == {"starts": 2, "per_start": 20}
