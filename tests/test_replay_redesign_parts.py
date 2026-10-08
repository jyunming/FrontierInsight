"""Replay of a real stop: a redesign (drafted at the design step, not read from plan.md) came back with settings to vary
that could not be read, and the quest paused with "FI did not ask the plan's model" although the plan's model had been
asked at the plan step. The evidence is the real plan.md of that quest (tests/fixtures/replay/redesign_sweep_unread/).
No model is called: a stub answers."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import plan
from tests.test_plan_optimisation import Paused, _engine

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "redesign_sweep_unread"
GOOD_GRID = {"axis_a": [1.0, 2.0, 3.0]}


def _setup(tmp_path: Path, answers: list[Any]) -> tuple[Any, dict[str, Any], list[str]]:
    eng = _engine(tmp_path, [])
    shutil.copy(FIXTURE / "plan.md", plan.plan_path(eng.quest_root))
    planned, sha = eng._design_from_plan()
    assert planned is not None and sha, "the real plan must be readable"
    redesign = copy.deepcopy(planned)
    redesign["protocol"].pop("grid", None)  # what the redesign came back with: no readable settings to vary
    asked: list[str] = []
    pending = list(answers)

    async def chat(messages: list[dict[str, str]], **kw: Any) -> str:
        asked.append(messages[-1]["content"])
        answer = pending.pop(0) if pending else None
        if answer is None:
            raise RuntimeError("no answer")
        return answer

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    return eng, redesign, asked


def test_the_evidence_is_what_the_run_log_says() -> None:
    assert "asked (0 time(s))" in (FIXTURE / "run_log_excerpt.txt").read_text(encoding="utf-8")
    assert json.loads((FIXTURE / "protocol_part_asked.json").read_text(encoding="utf-8"))["parts"][0] == "grid"


@pytest.mark.asyncio
async def test_before_the_fix_the_redesign_stopped_without_an_ask(tmp_path: Path) -> None:
    eng, redesign, asked = _setup(tmp_path, [])
    assert "grid" in eng._unread_parts(redesign)  # plan.md still carries the note; the redesign has no readable grid
    with pytest.raises(Paused):
        eng._stop_if_no_settings_to_vary({"topic": "t"}, redesign)  # the old path went straight here
    assert asked == []


@pytest.mark.asyncio
async def test_the_plans_model_is_asked_for_the_redesigns_settings_and_the_quest_goes_on(tmp_path: Path) -> None:
    eng, redesign, asked = _setup(tmp_path, [json.dumps({"grid": GOOD_GRID})])
    state = {"topic": "t"}
    out = await eng._complete_drafted_parts(state, redesign, iteration=1)
    assert out["protocol"]["grid"] == GOOD_GRID and len(asked) == 1 and "protocol.grid" in asked[0]
    eng._stop_if_no_settings_to_vary(state, out)  # no stop
    assert json.loads((eng.fi_dir / "drafted_parts_asked.json").read_text(encoding="utf-8"))["count"] == 1


@pytest.mark.asyncio
async def test_a_part_the_plans_model_cannot_write_is_taken_from_the_plan_after_the_bounded_asks(tmp_path: Path) -> None:
    mixed = json.dumps({"grid": {"axis_a": [1.0, "wide"]}})
    eng, redesign, asked = _setup(tmp_path, [mixed, mixed, mixed])
    planned, _ = eng._design_from_plan()
    out = await eng._complete_drafted_parts({"topic": "t"}, redesign, iteration=1)
    assert len(asked) == 2, "bounded: two asks, no more"
    assert out["protocol"]["grid"] == planned["protocol"]["grid"], "the plan's own settled settings"
    eng._stop_if_no_settings_to_vary({"topic": "t"}, out)  # no stop
    # a resume asks no more
    again = await eng._complete_drafted_parts({"topic": "t"}, redesign, iteration=1)
    assert len(asked) == 2 and again["protocol"]["grid"] == planned["protocol"]["grid"]


@pytest.mark.asyncio
async def test_the_stop_only_comes_after_the_bounded_asks_fail_and_says_so(tmp_path: Path) -> None:
    eng, redesign, asked = _setup(tmp_path, [])
    eng._design_from_plan = lambda: (None, "")  # type: ignore[method-assign]  # no settled plan to fall back on
    out = await eng._complete_drafted_parts({"topic": "t"}, redesign, iteration=1)
    assert len(asked) == 2 and "grid" not in out["protocol"]
    seen: list[dict[str, Any]] = []
    eng._pause_for_human = lambda **kw: seen.append(kw) or (_ for _ in ()).throw(Paused("plan"))  # type: ignore[method-assign]
    with pytest.raises(Paused):
        eng._stop_if_no_settings_to_vary({"topic": "t"}, out)
    shown = " ".join(seen[0]["steps"])
    assert "asked the plan's model 2 time(s)" in shown and "did not ask" not in shown


@pytest.mark.asyncio
async def test_using_the_plans_settings_for_a_redesign_is_recorded_and_reaches_the_report_and_the_writer(tmp_path: Path) -> None:
    from core import quality_report

    mixed = json.dumps({"grid": {"axis_a": [1.0, "wide"]}})
    eng, redesign, _asked = _setup(tmp_path, [mixed, mixed])
    out = await eng._complete_drafted_parts({"topic": "t"}, redesign, iteration=1)
    record = json.loads((eng.fi_dir / "drafted_parts_asked.json").read_text(encoding="utf-8"))
    assert record["substituted"] == ["grid"] and record["count"] == 2 and "grid" in out["protocol"]
    # the writer is told that what ran is the plan's, not the redesign's
    note = eng._drafted_parts_note()
    assert "settings to vary" in note and "plan's own" in note and "do NOT describe" in note
    # the quality report lists it as a problem of the design that was not carried out
    items = [p for st in quality_report.build(eng.quest_root, running=False)["stages"] if st["stage"] == "design"
             for p in st["problems"]["items"]]
    assert any("settings to vary" in p["text"] and p["status"] == "left out" for p in items), items
    # a resume asks no more and keeps the record
    await eng._complete_drafted_parts({"topic": "t"}, redesign, iteration=1)
    assert json.loads((eng.fi_dir / "drafted_parts_asked.json").read_text(encoding="utf-8"))["substituted"] == ["grid"]


@pytest.mark.asyncio
async def test_nothing_is_recorded_when_the_models_answer_is_used_and_an_old_record_is_dropped(tmp_path: Path) -> None:
    eng, redesign, _asked = _setup(tmp_path, [json.dumps({"grid": GOOD_GRID})])
    await eng._complete_drafted_parts({"topic": "t"}, redesign, iteration=1)
    assert eng._drafted_parts_note() == ""
    eng._record_drafted_parts_substituted(["grid"], 1)
    assert eng._drafted_parts_note() != ""
    complete = copy.deepcopy(redesign)
    complete["protocol"]["grid"] = GOOD_GRID
    await eng._complete_drafted_parts({"topic": "t"}, complete, iteration=2)  # a later round reads in full
    assert eng._drafted_parts_note() == ""


@pytest.mark.asyncio
async def test_a_substitution_that_cannot_be_written_down_is_not_run_the_quest_stops(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mixed = json.dumps({"grid": {"axis_a": [1.0, "wide"]}})
    eng, redesign, _asked = _setup(tmp_path, [mixed, mixed])
    real_write = Path.write_text

    def failing_write(self: Path, data: str, *a: Any, **kw: Any) -> int:
        if self.name == "drafted_parts_asked.json" and "substituted" in data:
            raise PermissionError("disk")
        return real_write(self, data, *a, **kw)

    monkeypatch.setattr(Path, "write_text", failing_write)
    out = await eng._complete_drafted_parts({"topic": "t"}, redesign, iteration=1)
    assert "grid" not in out["protocol"], "the plan's settings were not used: nobody would have been told"
    record = json.loads((eng.fi_dir / "drafted_parts_asked.json").read_text(encoding="utf-8"))
    assert "substituted" not in record and eng._drafted_parts_note() == ""
    with pytest.raises(Paused):
        eng._stop_if_no_settings_to_vary({"topic": "t"}, out)
