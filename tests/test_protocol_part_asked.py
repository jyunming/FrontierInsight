"""A protocol part that changes what is measured (the settings to sweep, the metrics) and could not be read: FI asks the
plan's model to write it again (the bounded request the search block uses), keeps only that part of the answer, and never
asks a person to write it. A sweep that is still unreadable stops the quest plainly (no one-setting run called a sweep); a
study that never listed a sweep goes on. Fake model only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import plan
from tests.test_plan_optimisation import EXTRA, HEAT_SINK, Paused, _engine, _NO_OBJECTIONS

TOPIC = "Measure how fin spacing changes the base temperature of a heat sink"
MARK = "Parts of this plan's protocol could not be read"
GOOD_GRID = {"fin_spacing": [2.0, 3.0, 4.0]}
GOOD_METRICS = [{"id": "base_temperature", "kind": "mean", "estimand": "mean base temperature", "unit": "one simulated heat sink"}]


def _draft(**protocol: Any) -> dict[str, Any]:
    draft = copy.deepcopy(HEAT_SINK)
    draft["study_type"] = "measure"
    draft["protocol"] = {"oracles": copy.deepcopy(HEAT_SINK["protocol"]["oracles"]), **protocol}
    return draft


def _set(**parts: Any) -> Any:
    def answer(text: str) -> str:
        out = plan.edit_design_block(text, lambda d: {**d, "protocol": {**d["protocol"], **parts}})
        assert out is not None
        return out
    return answer


class Model:
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
            return answer(plan.plan_path(self.eng.quest_root).read_text(encoding="utf-8")) if callable(answer) else answer
        if not self.drafted:
            self.drafted = True
            return json.dumps({**self.draft, "plan": EXTRA})
        return _NO_OBJECTIONS


def _record(eng: Any) -> dict[str, Any]:
    return json.loads((eng.fi_dir / "protocol_part_asked.json").read_text(encoding="utf-8"))


def _stop(eng: Any) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    return seen


@pytest.mark.asyncio
async def test_an_unreadable_sweep_is_asked_of_the_plans_model_and_the_sweep_runs(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    mixed = {"fin_spacing": [2.0, "wide"]}  # numbers and names in one list: cannot be read
    model = Model(eng, [_set(grid=GOOD_GRID, tolerance_changed=True)], _draft(grid=mixed))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [] and patch["design"]["protocol"]["grid"] == GOOD_GRID
    assert _record(eng)["count"] == 1
    assert "must be a non-empty list of numbers" in model.asked[0] and "protocol.grid" in model.asked[0]
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "Write each parameter" not in text and "put it right here" not in text and "NO settings" not in text
    assert "FI asked the plan's model to write `protocol.grid` again, and it is now in the plan" in text
    assert "tolerance_changed" not in text, "only the asked part is kept from the rewrite"
    assert "tolerance: 0.01" in text
    assert plan.history(eng.quest_root)[-1]["by"] == "engine"


@pytest.mark.asyncio
async def test_a_sweep_still_unreadable_after_the_bounded_requests_stops_plainly_and_a_resume_asks_no_more(
        tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_set(grid={"fin_spacing": [2.0, "wide"]})] * 4, _draft(grid={"fin_spacing": [2.0, "wide"]}))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    with pytest.raises(Paused):
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert _record(eng)["count"] == 2 and len(model.asked) <= 6
    shown = " ".join(seen[0]["steps"])
    assert "Nothing was run" in shown and "asked the plan's model 2 time(s)" in shown
    assert "You do not need to write them yourself" in shown and "plan.md" not in shown
    assert not (eng.quest_root / "paper.md").exists()
    with pytest.raises(Paused):  # a resume stops again without asking a third time
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert _record(eng)["count"] == 2 and len(model.asked) <= 6
    assert "the quest stops" in plan.plan_path(eng.quest_root).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_readable_sweep_is_untouched_and_nothing_is_asked(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_set(grid={"x": [1]})], _draft(grid=GOOD_GRID))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert model.asked == [] and seen == [] and patch["design"]["protocol"]["grid"] == GOOD_GRID
    assert not (eng.fi_dir / "protocol_part_asked.json").exists()


@pytest.mark.asyncio
async def test_a_study_that_never_listed_a_sweep_has_one_setting_and_goes_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _draft())
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert model.asked == [] and seen == [] and "grid" not in patch["design"]["protocol"]


@pytest.mark.asyncio
async def test_unreadable_metrics_are_asked_once_and_when_still_unreadable_the_study_goes_on_saying_so(
        tmp_path: Path) -> None:
    bad = [{"id": "base_temperature", "kind": "median", "estimand": "x", "unit": "y"}]
    eng = _engine(tmp_path, [])
    model = Model(eng, [_set(metrics=GOOD_METRICS)], _draft(grid=GOOD_GRID, metrics=bad))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    patch = await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [] and patch["design"]["protocol"]["metrics"][0]["id"] == "base_temperature"
    assert "`kind` must be `proportion` or `mean`" in model.asked[0]
    assert "put it right here" not in plan.plan_path(eng.quest_root).read_text(encoding="utf-8")

    other = _engine(tmp_path / "again", [])
    still = Model(other, [_set(metrics=bad)] * 3, _draft(grid=GOOD_GRID, metrics=bad))
    await other._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(other)
    patch = await other._node_design({"topic": TOPIC, "iteration": 0})
    assert seen == [] and _record(other)["count"] == 2 and "metrics" not in patch["design"]["protocol"]
    assert "goes on without it" in plan.plan_path(other.quest_root).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_part_asked_about_at_the_plan_step_is_not_asked_again_at_the_design_step(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _draft(grid={"fin_spacing": [2.0, "wide"]}))  # every request fails
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    after_plan = len(model.asked)
    assert after_plan >= 1 and _record(eng)["parts"] == ["grid"]
    seen = _stop(eng)
    with pytest.raises(Paused):
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert len(model.asked) == after_plan, "one normal run asks at most once per part"
    assert "asked the plan's model" in " ".join(seen[0]["steps"])
