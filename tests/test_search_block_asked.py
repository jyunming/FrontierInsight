"""A search for the best design whose plan lacks what a search needs: FI asks the plan's model to write it (twice at most)
before it stops, and the stop never asks a person to write FI's own block. Fake model only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import optimisation_plan as op
from core import plan
from tests.test_plan_optimisation import BLOCK, EXTRA, HEAT_SINK, Paused, _engine, _NO_OBJECTIONS

TOPIC = "Find the fin spacing and thickness that minimise the base temperature of a heat sink"
MARK = "the plan has no `protocol.optimisation` block FI can search with yet"


def _broken_draft() -> dict[str, Any]:
    """The real quest's draft: the objective names neither a quantity nor a direction, so the block is left out."""
    draft = copy.deepcopy(HEAT_SINK)
    draft["protocol"]["optimisation"]["objective"] = {"meaning": "the common depth of focus"}
    return draft


def _with_block(text: str, block: dict[str, Any]) -> str:
    def change(design: dict[str, Any]) -> dict[str, Any]:
        protocol = {**(design.get("protocol") or {}), "optimisation": block}
        return {**design, "study_type": "find_best_design", "protocol": protocol}

    out = plan.edit_design_block(text, change)
    assert out is not None
    return out


class Model:
    """A fake model: the plan draft first, an answer to each request for the search block, no objections otherwise."""

    def __init__(self, eng: Any, answers: list[Any], draft: dict[str, Any]) -> None:
        self.eng, self.answers, self.draft = eng, list(answers), draft
        self.asked: list[str] = []
        self.drafted = False
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if MARK in prompt:
            self.asked.append(prompt)
            answer = self.answers.pop(0) if self.answers else None
            if answer is None:
                raise RuntimeError("no answer")
            text = plan.plan_path(self.eng.quest_root).read_text(encoding="utf-8")
            return answer(text) if callable(answer) else answer
        if not self.drafted:
            self.drafted = True
            return json.dumps({**self.draft, "plan": EXTRA})
        return _NO_OBJECTIONS

    @property
    def requests(self) -> int:
        """Requests, not tries: a reply that cannot be used is asked once more with the reason, inside one request."""
        return len([p for p in self.asked if "Your last reply could not be used" not in p])


def _complete(text: str) -> str:
    return _with_block(text, copy.deepcopy(BLOCK))


def _no_budget(text: str) -> str:
    block = copy.deepcopy(BLOCK)
    del block["evaluation_budget"]
    return _with_block(text, block)


def _record(eng: Any) -> dict[str, Any]:
    return json.loads((eng.fi_dir / "search_block_asked.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_block_that_could_not_be_read_is_written_by_the_plans_model_and_the_search_goes_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_complete], _broken_draft())
    await eng._node_plan({"topic": TOPIC, "iteration": 0})  # the plan a person reads is already complete
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [], "no stop: the search can start"
    assert patch["design"]["protocol"]["optimisation"]["objective"]["quantity"] == "max_base_temperature"
    assert model.requests == 1 and _record(eng)["count"] == 1
    # The request quotes what was missing and says what each part means, and takes the goal from the topic.
    request = model.asked[0]
    assert "no `optimisation` block" in request and "`direction`" in request and "`evaluation_budget`" in request
    assert "left out because" in request and TOPIC in request
    # plan.md shows the block it now has (not the old "no block" note), and the change is a recorded version by the engine.
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "max_base_temperature" in text and "does not yet say" not in text
    assert plan.history(eng.quest_root)[-1]["by"] == "engine"


@pytest.mark.asyncio
async def test_a_block_still_incomplete_after_two_requests_stops_with_plain_choices_and_a_resume_asks_no_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_no_budget, _no_budget, _no_budget], _broken_draft())
    await eng._node_plan({"topic": TOPIC, "iteration": 0})

    def interrupted(payload: Any) -> Any:
        raise Paused("plan")

    monkeypatch.setattr("core.engine.interrupt", interrupted)
    with pytest.raises(Paused):
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert model.requests == 2 and _record(eng)["count"] == 2
    shown = (eng.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    for word in ("optimisation block", "plan.md", "YAML", "evaluation_budget", "`optimisation`"):
        assert word not in shown, word
    assert "FI asked the plan's model 2 time(s)" in shown and "how many designs the search may try" in shown
    assert "another model" in shown and op.MEASURE_INSTEAD in shown and "Nothing was run" in shown
    # A resume checks again and stops again, and does not ask a third time.
    with pytest.raises(Paused):
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert model.requests == 2
    # Another model for the plan step is asked afresh.
    eng.config.provider.node_models = {"plan_revise": "another-model"}
    model.answers = [_complete]
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert model.requests == 3 and patch["design"]["protocol"]["optimisation"]["evaluation_budget"]["per_start"] == 60


@pytest.mark.asyncio
async def test_a_reply_that_cannot_be_used_counts_as_a_request_and_leaves_the_plan_as_it_was(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])

    def unusable(text: str) -> str:
        return _with_block(text, {"objective": {"quantity": "x"}})  # no direction, no variables

    model = Model(eng, [unusable, unusable, unusable, unusable], _broken_draft())
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    before = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    seen = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    with pytest.raises(Paused):
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert model.requests == 2 and _record(eng)["count"] == 2
    assert plan.plan_path(eng.quest_root).read_text(encoding="utf-8") == before
    steps = " ".join(seen[0]["steps"])
    assert "what the search should make as low or as high as possible" in steps and "asked the plan's model 2 time(s)" in steps


@pytest.mark.asyncio
async def test_a_model_that_gives_no_answer_is_not_counted_and_a_resume_asks_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _broken_draft())  # every request fails
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    with pytest.raises(Paused):
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert not (eng.fi_dir / "search_block_asked.json").is_file()
    assert "could not get the plan's model" in " ".join(seen[0]["steps"])
    model.answers = [_complete]
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert patch["design"]["protocol"]["optimisation"]["objective"]["direction"] == "minimise"


@pytest.mark.asyncio
async def test_only_the_search_part_of_a_rewrite_is_kept(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])

    def moves_a_check(text: str) -> str:
        done = _complete(text).replace("tolerance: 0.01", "tolerance: 0.5")
        return done.replace("method: search the fin geometry", "method: something else entirely")

    model = Model(eng, [moves_a_check], _broken_draft())
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    design = plan.parse(text).design
    assert model.requests == 1 and _record(eng)["outcome"] == "kept"
    assert design["protocol"]["optimisation"]["objective"]["quantity"] == "max_base_temperature"
    assert "tolerance: 0.01" in text and "tolerance: 0.5" not in text and "something else" not in text


@pytest.mark.asyncio
async def test_a_usable_block_and_a_measurement_are_not_asked_about(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_complete], copy.deepcopy(HEAT_SINK))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert model.asked == [] and patch["design"]["protocol"]["optimisation"]["evaluation_budget"]["per_start"] == 60

    measure = {**{k: v for k, v in HEAT_SINK.items() if k not in ("protocol", "study_type")},
               "study_type": "measure", "protocol": {"grid": {"fin_spacing": [2.0, 3.0]}}}
    other = _engine(tmp_path / "m", [])
    again = Model(other, [_complete], measure)
    await other._node_plan({"topic": "How does fin spacing change the base temperature?", "iteration": 0})
    patch = await other._node_design({"topic": "How does fin spacing change the base temperature?", "iteration": 0})
    assert again.asked == [] and patch["design"]["study_type"] == "measure"


@pytest.mark.asyncio
async def test_the_plain_choice_to_measure_instead_lets_the_quest_go_on(tmp_path: Path) -> None:
    """The sentence the stop offers is a plain ``--revise-plan`` request: the plan's model turns the plan into a
    measurement over the settings it lists, and the quest then runs it (it is not stopped for a search)."""
    eng = _engine(tmp_path, [])
    model = Model(eng, [None, None], _broken_draft())
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    with pytest.raises(Paused):
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert any(op.MEASURE_INSTEAD in step for step in seen[0]["steps"])

    async def measure_reply(messages: list[dict[str, str]], **kw: Any) -> str:
        assert op.MEASURE_INSTEAD in messages[-1]["content"]
        text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
        out = plan.edit_design_block(text, lambda d: {
            **{k: v for k, v in d.items() if k != "study_type"}, "study_type": "measure",
            "protocol": {**{k: v for k, v in d["protocol"].items() if k != "optimisation"},
                         "grid": {"fin_spacing": [2.0, 3.0, 4.0]}}})
        assert out is not None
        return out

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=measure_reply)})()
    await eng.revise_plan(op.MEASURE_INSTEAD)
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=[_NO_OBJECTIONS, _NO_OBJECTIONS])})()
    eng._pause_for_human = lambda **kw: (_ for _ in ()).throw(AssertionError("stopped"))  # type: ignore[method-assign]
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert patch["design"]["study_type"] == "measure" and model.requests == 2


@pytest.mark.asyncio
async def test_it_is_not_specific_to_one_field(tmp_path: Path) -> None:
    """A second, unrelated search problem (a storage tank's radius): the same request completes the plan."""
    tank = {
        "objective": {"quantity": "heat_loss_W", "direction": "minimise", "unit": "W"},
        "design_variables": [{"name": "radius_m", "low": 0.5, "high": 3.0, "unit": "m", "kind": "continuous"}],
        "baseline": {"values": {"radius_m": 1.0}, "source": "the current tank"},
        "evaluation_budget": {"starts": 3, "per_start": 40},
    }
    topic = "Find the tank radius that minimises heat loss for a fixed volume"
    eng = _engine(tmp_path, [])
    model = Model(eng, [lambda text: _with_block(text, tank)], {**_broken_draft(), "study_type": "find_best_design"})
    await eng._node_plan({"topic": topic, "iteration": 0})
    design = plan.parse(plan.plan_path(eng.quest_root).read_text(encoding="utf-8")).design
    assert model.requests == 1 and design["protocol"]["optimisation"]["objective"]["quantity"] == "heat_loss_W"
    assert topic in model.asked[0]
