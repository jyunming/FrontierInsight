"""When FI rewrote the plan itself and nothing has run, the methodology audit's objections are met before implement."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core import plan, receipts
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine

DRAFT = {
    "hypothesis": "the integrator converges at fourth order",
    "variables": {"independent": ["dt"], "dependent": ["period_error"], "controls": ["seed"]},
    "method": "measure the period from zero-crossings",
    "expected_outcome": "error falls as dt^4",
    "figures_planned": ["c.png"],
    "dependencies": ["numpy"],
    "protocol": {"runs_per_setting": 5},
}
OBJECTION = {"check": "numerical_convergence", "objection": "timing error is O(dt) and masks O(dt^4)",
             "fix": "interpolate the zero-crossings"}
METHOD = "measure the period from interpolated zero-crossings"


def _engine(tmp_path: Path, replies: list[str]) -> Engine:
    cfg = Config(
        topic="pendulum period", title="pend", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=replies)})()
    return eng


def _write_plan(eng: Engine, by: str) -> dict:
    design, why = plan.normalize_design(DRAFT)
    assert design is not None, why
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render("pendulum period", {}, design, []), encoding="utf-8")
    plan.record_version(eng.quest_root, path.read_text(encoding="utf-8"), by=by, note="test")
    parsed = plan.parse(path.read_text(encoding="utf-8")).design
    assert parsed is not None
    return parsed


def _receipt(eng: Engine) -> dict:
    return json.loads((eng.quest_root / "needs" / "receipts" / "design_audit.json").read_text(encoding="utf-8"))


def _calls(eng: Engine) -> int:
    return eng._client.chat.await_count


def _amending(design: dict) -> str:
    """The audit's reply: it sees the design as plan.md holds it (normalised) and returns the whole amended design."""
    return json.dumps({"objections_addressed": [OBJECTION], "amended_design": {**design, "method": METHOD}})


def _objects_then_calm(design: dict) -> list[str]:
    return [_amending(design), json.dumps({"objections_addressed": [], "amended_design": {**design, "method": METHOD}})]


@pytest.mark.asyncio
async def test_an_engine_rewrite_before_the_freeze_is_changed_to_meet_the_audit(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    design = _write_plan(eng, "engine")
    eng._client.chat.side_effect = _objects_then_calm(design)
    old_sha = plan.sha256(plan.plan_path(eng.quest_root).read_text(encoding="utf-8"))
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert "interpolated" in runs["method"]
    assert _calls(eng) == 2  # one audit that may change the design, one on the design that runs
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert plan.parse(text).design["method"] == runs["method"]  # plan.md and the design that runs agree
    assert plan.sha256(text) != old_sha
    last = plan.history(eng.quest_root)[-1]
    assert last["by"] == "engine" and last["sha256"] == plan.sha256(text)
    receipt = _receipt(eng)
    assert receipt["status"] == "pass"
    assert receipt["output_hash"] == receipts.sha256(receipts.design_core(runs))
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "FI changed the design to meet it: timing error is O(dt)" in log
    # The design that runs now matches its receipt, so asking again takes no call.
    again = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, runs)
    assert again == runs and _calls(eng) == 2


@pytest.mark.asyncio
async def test_a_persons_edit_is_only_recorded_never_changed(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    design = _write_plan(eng, "user")
    eng._client.chat.side_effect = _objects_then_calm(design)
    before = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert runs == design and _calls(eng) == 1
    assert plan.plan_path(eng.quest_root).read_text(encoding="utf-8") == before
    assert _receipt(eng)["status"] == "fail"  # the objection is recorded, as before


@pytest.mark.asyncio
async def test_a_persons_request_is_only_recorded_never_changed(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    design = _write_plan(eng, "request")
    eng._client.chat.side_effect = _objects_then_calm(design)
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert runs == design and _calls(eng) == 1


@pytest.mark.asyncio
async def test_a_plan_edited_on_disk_after_the_last_version_is_a_persons(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    _write_plan(eng, "engine")
    path = plan.plan_path(eng.quest_root)
    path.write_text(path.read_text(encoding="utf-8") + "\nA note I added.\n", encoding="utf-8")
    design = plan.parse(path.read_text(encoding="utf-8")).design
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert runs == design and _calls(eng) == 1


@pytest.mark.asyncio
async def test_a_frozen_protocol_is_only_recorded_never_changed(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, [])
    design = _write_plan(eng, "engine")
    eng._client.chat.side_effect = _objects_then_calm(design)
    monkeypatch.setattr("core.engine._frozen.load", lambda _root: {"frozen": True})
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert runs == design and _calls(eng) == 1


@pytest.mark.asyncio
async def test_a_design_that_is_not_the_plans_is_only_recorded(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    _write_plan(eng, "engine")
    elsewhere = {**DRAFT, "method": "an amendment settled outside plan.md"}
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, elsewhere)
    assert runs == elsewhere and _calls(eng) == 1


@pytest.mark.asyncio
async def test_a_resume_does_not_change_the_same_design_twice(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    design = _write_plan(eng, "engine")
    eng._client.chat.side_effect = [_amending(design)] * 4
    key = receipts.sha256(receipts.design_core(design))
    eng._mark_design_audit_acted(key)  # the first run acted on this design, then the process died
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert runs == design and _calls(eng) == 1  # recorded only, no second change


@pytest.mark.asyncio
async def test_an_audit_that_objects_again_is_recorded_and_the_quest_goes_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    design = _write_plan(eng, "engine")
    eng._client.chat.side_effect = [_amending(design)] * 4
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert "interpolated" in runs["method"] and _calls(eng) == 2  # no loop
    assert _receipt(eng)["status"] == "fail"
    assert _receipt(eng)["output_hash"] == receipts.sha256(receipts.design_core(runs))


@pytest.mark.asyncio
async def test_an_audit_with_nothing_to_change_still_leaves_an_honest_receipt(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    design = _write_plan(eng, "engine")
    eng._client.chat.side_effect = [json.dumps({"objections_addressed": [], "amended_design": design}), json.dumps({"objections_addressed": [OBJECTION], "amended_design": design})]
    runs = await eng._audit_the_design_that_runs({"topic": "t", "iteration": 0}, design)
    assert runs == design and _calls(eng) == 2
    assert _receipt(eng)["status"] == "fail"
