"""The methodology audit of a design leaves a record (needs/DESIGN_CRITIQUE.json) and asks the questions the audit found missing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core import plan
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine

AGENTS = Path(__file__).resolve().parent.parent / "agents"

DRAFT = {
    "hypothesis": "model-based OPC reduces EPE more than rule-based",
    "variables": {"independent": ["strategy"], "dependent": ["epe"], "controls": ["seed"]},
    "method": "compare three strategies on synthetic clips",
    "expected_outcome": "model-based wins",
    "figures_planned": ["c.png"],
    "dependencies": ["numpy"],
    "protocol": {"runs_per_setting": 300},
}
OBJECTION = {"check": "precision", "objection": "300 runs give about +-0.057", "fix": "raised the runs to 1000"}


def _engine(tmp_path: Path, replies: list[str]) -> Engine:
    cfg = Config(
        topic="OPC under low k1", title="opc", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=replies)})()
    return eng


def _record(eng: Engine) -> list[dict]:
    return json.loads((eng.quest_root / "needs" / "DESIGN_CRITIQUE.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_what_the_audit_objected_to_and_changed_is_kept_in_full(tmp_path: Path) -> None:
    amended = {**DRAFT, "protocol": {"runs_per_setting": 1000}, "method": "compare three strategies with 1000 runs each"}
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [OBJECTION], "amended_design": amended})])
    design, objections = await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    assert design == amended and objections == [OBJECTION]
    (entry,) = _record(eng)
    assert entry["status"] == "ok" and entry["failure"] == "" and entry["iteration"] == 0
    assert entry["objections"] == [OBJECTION]
    assert entry["changed_keys"] == ["method", "protocol"]
    assert entry["before"]["protocol"] == {"runs_per_setting": 300} and entry["after"]["protocol"] == {"runs_per_setting": 1000}


@pytest.mark.asyncio
async def test_an_audit_that_found_nothing_is_recorded_as_such(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [], "amended_design": DRAFT})])
    await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    (entry,) = _record(eng)
    assert entry["status"] == "ok" and entry["objections"] == [] and entry["changed_keys"] == []


@pytest.mark.asyncio
async def test_an_audit_that_failed_says_so_instead_of_looking_like_one_that_found_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=RuntimeError("provider down"))})()
    design, objections = await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    assert design == DRAFT and objections is None
    (entry,) = _record(eng)
    assert entry["status"] == "failed" and "provider down" in entry["failure"] and entry["changed_keys"] == []


@pytest.mark.asyncio
async def test_an_amended_design_that_dropped_a_key_is_recorded_as_failed_and_the_draft_is_kept(tmp_path: Path) -> None:
    dropped = {k: v for k, v in DRAFT.items() if k != "protocol"}
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [OBJECTION], "amended_design": dropped})])
    design, _ = await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    assert design == DRAFT
    (entry,) = _record(eng)
    assert entry["status"] == "failed" and "protocol" in entry["failure"]


@pytest.mark.asyncio
async def test_every_pass_adds_an_entry(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [], "amended_design": DRAFT})] * 2)
    await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    await eng._audit_design({"topic": "OPC", "iteration": 1}, dict(DRAFT))
    assert [e["iteration"] for e in _record(eng)] == [0, 1]


@pytest.mark.asyncio
async def test_the_plan_says_what_the_audit_changed_or_that_it_did_not_finish(tmp_path: Path) -> None:
    amended = {**DRAFT, "method": "a different method"}
    eng = _engine(tmp_path, [json.dumps(DRAFT), json.dumps({"objections_addressed": [OBJECTION], "amended_design": amended})])
    await eng._node_plan({"topic": "OPC", "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "The methodology audit amended the design: method" in text and "needs/DESIGN_CRITIQUE.json" in text

    failing = _engine(tmp_path / "f", [json.dumps(DRAFT)])
    failing._client = type("Stub", (), {"chat": AsyncMock(side_effect=[json.dumps(DRAFT), RuntimeError("down")])})()
    await failing._node_plan({"topic": "OPC", "iteration": 0})
    assert "The methodology audit did not complete" in plan.plan_path(failing.quest_root).read_text(encoding="utf-8")


def test_the_audit_asks_about_precision_the_estimand_thresholds_streams_convergence_oracles_and_failed_runs() -> None:
    text = (AGENTS / "design_self_critique.md").read_text(encoding="utf-8")
    for check in ("precision", "estimand", "threshold_sensitivity", "rng_independence", "numerical_convergence", "oracle", "failed_runs"):
        assert check in text, check
    for words in ("0.98/sqrt(n)", "not a sample size", "common-random-numbers", "protocol.oracles", "Never drop a key the draft has"):
        assert words in text, words


@pytest.mark.asyncio
async def test_each_audit_leaves_a_receipt_and_a_research_quest_stops_once_when_it_could_not_judge(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps({"objections_addressed": [], "amended_design": DRAFT})])
    await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    receipt = json.loads((eng.quest_root / "needs" / "receipts" / "design_audit.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "pass" and receipt["input_hashes"]["design"]

    eng.config = eng.config.model_copy(update={"rigor_profile": "research"})
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=RuntimeError("provider down"))})()
    stops: list[dict] = []

    def pause(**kwargs):  # noqa: ANN003
        stops.append(kwargs)
        raise RuntimeError("stopped")

    eng._pause_for_human = pause  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="stopped"):
        await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))
    assert stops[0]["kind"] == "design_audit_unknown"
    receipt = json.loads((eng.quest_root / "needs" / "receipts" / "design_audit.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "unknown" and "provider down" in receipt["error"]
    await eng._audit_design({"topic": "OPC", "iteration": 0}, dict(DRAFT))  # the retry fails too: no second stop
    assert len(stops) == 1
