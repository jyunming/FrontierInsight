"""The request that asks the plan's model to complete a search for the best design is bounded (also when a call fails),
never overwrites a person's edit, never changes what the plan already said, and the "measure instead" rewrite keeps the
plan's question and settings. Fake model only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import optimisation_plan as op
from core import plan
from tests.test_plan_optimisation import BLOCK, HEAT_SINK, Paused, _engine, _NO_OBJECTIONS
from tests.test_search_block_asked import (
    TOPIC, Model, _broken_draft, _complete, _no_budget, _record, _with_block,
)


def _stop(eng: Any) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    return seen


@pytest.mark.asyncio
async def test_failed_calls_count_across_resumes_and_total_chat_calls_stay_within_six(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [], _broken_draft())  # every request raises
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    _stop(eng)
    for _ in range(5):  # a resume each time
        with pytest.raises(Paused):
            await eng._node_design({"topic": TOPIC, "iteration": 0})
    record = _record(eng)
    assert record["count"] == 2 and record["calls"] <= 6
    assert len(model.asked) <= 6


@pytest.mark.asyncio
async def test_unusable_answers_never_make_more_than_six_chat_calls(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])

    def junk(text: str) -> str:
        return "no plan file here"

    model = Model(eng, [junk] * 20, _broken_draft())
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    _stop(eng)
    for _ in range(4):
        with pytest.raises(Paused):
            await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert len(model.asked) <= 6 and _record(eng)["calls"] <= 6


@pytest.mark.asyncio
async def test_a_plan_a_person_saved_during_the_request_is_kept(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    path = plan.plan_path(eng.quest_root)
    mine = {"text": ""}

    def edited_meanwhile(text: str) -> str:
        mine["text"] = text + "\n<!-- my note -->\n"
        path.write_text(mine["text"], encoding="utf-8")  # the person saves while the model is answering
        return _complete(text)

    Model(eng, [edited_meanwhile], _broken_draft())
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    assert path.read_text(encoding="utf-8") == mine["text"], "the person's version is on disk, not the model's"
    assert "my note" in path.read_text(encoding="utf-8")
    assert "evaluation_budget" not in path.read_text(encoding="utf-8"), "the model's block was not written"


@pytest.mark.asyncio
async def test_what_the_plan_already_said_is_kept_when_the_model_changes_it(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    draft = copy.deepcopy(HEAT_SINK)
    block = draft["protocol"]["optimisation"]
    del block["evaluation_budget"]
    block["improvement_tolerance"] = 0.5
    block["target"] = {"value": 300.0}

    def changes_more(text: str) -> str:
        full = copy.deepcopy(BLOCK)
        full["improvement_tolerance"] = 5.0  # the model also moves a value the plan already had
        full["constraints"] = []
        full["target"] = {"value": 999.0}
        return _with_block(text, full)

    model = Model(eng, [changes_more], draft)
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    got = plan.parse(plan.plan_path(eng.quest_root).read_text(encoding="utf-8")).design["protocol"]["optimisation"]
    assert model.requests == 1
    assert got["evaluation_budget"] == BLOCK["evaluation_budget"], "the missing part came from the model"
    assert got["improvement_tolerance"] == 0.5 and got["target"] == {"value": 300.0}
    assert got["constraints"] == BLOCK["constraints"]


async def _revise(eng: Any, new_design: Any) -> Any:
    async def reply(messages: list[dict[str, str]], **kw: Any) -> str:
        text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
        out = plan.edit_design_block(text, new_design)
        assert out is not None
        return out

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=reply)})()
    return await eng.revise_plan(op.MEASURE_INSTEAD)


def _measure(grid: dict[str, Any], **top: Any):
    def change(design: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in design.items() if k != "study_type"}
        out.update(top)
        out["study_type"] = "measure"
        out["protocol"] = {**{k: v for k, v in design["protocol"].items() if k != "optimisation"}, "grid": grid}
        return out
    return change


@pytest.mark.asyncio
async def test_measure_instead_keeps_the_question_and_the_listed_settings(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], copy.deepcopy(HEAT_SINK))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    await _revise(eng, _measure({"fin_spacing": [2.0, 3.0, 4.0]}))
    design = plan.parse(plan.plan_path(eng.quest_root).read_text(encoding="utf-8")).design
    assert design["study_type"] == "measure" and design["protocol"]["grid"]["fin_spacing"] == [2.0, 3.0, 4.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("grid, top, said", [
    ({"fin_spacing": [2.0, 50.0]}, {}, "outside what the plan listed"),
    ({"coolant_flow": [1.0, 2.0]}, {}, "did not list as a setting"),
    ({"fin_spacing": [2.0]}, {"hypothesis": "a different question altogether"}, "changed the plan's hypothesis"),
])
async def test_measure_instead_that_changes_the_question_or_the_settings_is_put_back(
    tmp_path: Path, grid: dict[str, Any], top: dict[str, Any], said: str,
) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], copy.deepcopy(HEAT_SINK))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    before = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=said):
        await _revise(eng, _measure(grid, **top))
    assert plan.plan_path(eng.quest_root).read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_no_request_is_made_when_it_cannot_be_written_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, [])
    model = Model(eng, [_complete, _complete], _broken_draft())
    real = Path.write_text

    def failing(self: Path, *a: Any, **kw: Any) -> Any:
        if self.name == "search_block_asked.json":
            raise OSError("disk full")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", failing)
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    seen = _stop(eng)
    with pytest.raises(Paused):
        await eng._node_design({"topic": TOPIC, "iteration": 0})
    assert model.asked == [], "no marker, no request"
    assert seen, "the quest went on to the plain stop"


@pytest.mark.asyncio
async def test_measure_instead_with_no_grid_is_put_back(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], copy.deepcopy(HEAT_SINK))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    before = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")

    def no_grid(design: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in design.items() if k != "study_type"}
        out["study_type"] = "measure"
        out["protocol"] = {k: v for k, v in design["protocol"].items() if k != "optimisation"}
        return out

    with pytest.raises(ValueError, match="does not say which of the plan's own settings"):
        await _revise(eng, no_grid)
    assert plan.plan_path(eng.quest_root).read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_measure_instead_of_a_plan_that_listed_no_settings_is_put_back(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], copy.deepcopy(HEAT_SINK))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    path = plan.plan_path(eng.quest_root)
    path.write_text(plan.edit_design_block(path.read_text(encoding="utf-8"), lambda d: {
        **d, "protocol": {k: v for k, v in d["protocol"].items() if k != "optimisation"}, "study_type": "measure",
        "variables": {**d["variables"], "independent": []}}),
        encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="does not say which of the plan's own settings"):
        await _revise(eng, _measure({"fin_spacing": [2.0, 3.0]}))
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_measure_instead_of_a_setting_the_plan_gave_no_values_for_is_refused(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], _broken_draft())  # the search block cannot be read, so the plan only names its settings
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    path = plan.plan_path(eng.quest_root)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="write the values in the plan first"):
        await _revise(eng, _measure({"fin_spacing": [999.0]}))
    assert path.read_text(encoding="utf-8") == before
