"""A design is changed one part at a time: the methodology audit changes only the parts an objection names, and a redesign
asks only for the parts its reason concerns while the rest is carried over unchanged. Fake model only; the studies are
made up."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import design_parts as dp
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine

DESIGN: dict[str, Any] = {
    "hypothesis": "the filter lowers the error",
    "variables": {"independent": ["strength"], "dependent": ["error"], "controls": ["seed"]},
    "method": "compare the filter with no filter on synthetic signals",
    "expected_outcome": "the error falls",
    "figures_planned": ["error.png"],
    "dependencies": ["numpy"],
    "protocol": {"grid": {"strength": [1, 2, 3]}, "runs_per_setting": 100, "seed_policy": "one stream per setting"},
}


def _engine(tmp_path: Path, replies: list[str]) -> Engine:
    eng = Engine(Config(
        topic="a filter and its error", title="f", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=replies)})()
    return eng


def _sent(eng: Engine) -> list[str]:
    out = []
    for c in eng._client.chat.call_args_list:
        arg = c.args[0]
        out.append(arg[-1]["content"] if isinstance(arg, list) else str(arg))
    return out


# ---- the pure part --------------------------------------------------------------------------------------------------

def test_a_reply_of_parts_changes_only_the_asked_parts() -> None:
    reply = {"protocol": {"runs_per_setting": 900, "seed_policy": "x"}, "hypothesis": "something else"}
    got, applied, refused = dp.merge_parts(DESIGN, reply, ["protocol.runs_per_setting"])
    assert got["protocol"]["runs_per_setting"] == 900
    assert got["protocol"]["seed_policy"] == DESIGN["protocol"]["seed_policy"]
    assert got["hypothesis"] == DESIGN["hypothesis"]
    assert applied == ["protocol.runs_per_setting"]
    assert {r["key"] for r in refused} == {"protocol.seed_policy", "hypothesis"}


def test_a_part_that_is_not_a_part_of_a_design_is_refused() -> None:
    got, applied, refused = dp.merge_parts(DESIGN, {"banana": 1, "protocol": {"two words": 2}}, None)
    assert got == DESIGN and applied == []
    assert {r["key"] for r in refused} == {"banana", "protocol.two words"}


def test_the_notes_name_parts_by_the_designs_own_part_names() -> None:
    assert dp.allowed_for_notes(["The grid has a single point and the seeds are shared"]) == [
        "protocol.grid", "protocol.seed_policy"]
    assert dp.allowed_for_notes(["it is not convincing"]) == []
    assert dp.allowed_for_notes([], ["circular_evaluation"]) == ["method"]


# ---- the audit ------------------------------------------------------------------------------------------------------

async def _audit(tmp_path: Path, reply: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    eng = _engine(tmp_path, [json.dumps(reply)])
    return await eng._audit_design({"topic": "t", "iteration": 0}, copy.deepcopy(DESIGN))


@pytest.mark.asyncio
async def test_an_objection_on_one_key_changes_only_that_key(tmp_path: Path) -> None:
    obj = [{"check": "weak_baseline", "where": "method", "objection": "no tuning of the comparator", "why": "w",
            "fix": "tune it", "severity": "high",
            "changes": [{"path": "method", "value": "compare the filter with a tuned comparator"}]}]
    design, objections = await _audit(tmp_path, {"objections_addressed": obj, "checked_no_problem": []})
    assert design["method"] == "compare the filter with a tuned comparator"
    assert {k: v for k, v in design.items() if k != "method"} == {k: v for k, v in DESIGN.items() if k != "method"}
    assert objections == obj


@pytest.mark.asyncio
async def test_a_change_under_a_protocol_part_keeps_the_rest_of_the_protocol(tmp_path: Path) -> None:
    obj = [{"check": "precision", "where": "protocol.runs_per_setting", "objection": "too few", "why": "w", "fix": "more",
            "severity": "medium", "changes": [{"path": "protocol.runs_per_setting", "value": 400}]}]
    design, _ = await _audit(tmp_path, {"objections_addressed": obj, "checked_no_problem": []})
    assert design["protocol"] == {**DESIGN["protocol"], "runs_per_setting": 400}


@pytest.mark.asyncio
async def test_a_part_changed_in_a_whole_reply_that_no_objection_named_is_refused(tmp_path: Path) -> None:
    whole = {**DESIGN, "method": "a tuned comparator", "hypothesis": "a different claim",
             "protocol": {**DESIGN["protocol"], "runs_per_setting": 5}}
    obj = [{"check": "weak_baseline", "where": "method", "objection": "no tuning", "why": "w", "fix": "tune",
            "severity": "high"}]
    design, _ = await _audit(tmp_path, {"objections_addressed": obj, "amended_design": whole})
    assert design["method"] == "a tuned comparator"
    assert design["hypothesis"] == DESIGN["hypothesis"]
    assert design["protocol"]["runs_per_setting"] == 100


@pytest.mark.asyncio
async def test_a_change_to_a_path_that_is_not_a_part_of_a_design_is_refused(tmp_path: Path) -> None:
    obj = [{"check": "other", "where": "method", "objection": "o", "why": "w", "fix": "f", "severity": "low",
            "changes": [{"path": "secret_part", "value": 1}, {"path": "method", "value": "better"}]}]
    design, _ = await _audit(tmp_path, {"objections_addressed": obj, "checked_no_problem": []})
    assert "secret_part" not in design and design["method"] == "better"


def test_the_audit_prompt_no_longer_asks_for_a_whole_design() -> None:
    text = (Path(__file__).resolve().parent.parent / "agents" / "design_self_critique.md").read_text(encoding="utf-8")
    assert "amended_design" not in text and '"changes"' in text


# ---- the redesign ---------------------------------------------------------------------------------------------------

PRECISION_STATE: dict[str, Any] = {
    "topic": "a filter and its error", "iteration": 1, "design": DESIGN, "cross_check_route": "redesign",
}


@pytest.mark.asyncio
async def test_a_precision_redesign_changes_only_the_sizing_parts_and_shows_the_previous_design(tmp_path: Path) -> None:
    reply = {"protocol": {"runs_per_setting": 900, "seed_policy": "reworded"}, "hypothesis": "a new claim",
             "method": "a different method"}
    eng = _engine(tmp_path, [json.dumps(reply)])
    got = await eng._redesign_by_parts(PRECISION_STATE)
    assert got is not None
    assert got["protocol"]["runs_per_setting"] == 900
    assert {k: v for k, v in got.items() if k != "protocol"} == {k: v for k, v in DESIGN.items() if k != "protocol"}
    assert {k: v for k, v in got["protocol"].items() if k != "runs_per_setting"} == \
        {k: v for k, v in DESIGN["protocol"].items() if k != "runs_per_setting"}
    prompt = _sent(eng)[0]
    assert DESIGN["method"] in prompt and DESIGN["hypothesis"] in prompt  # the previous design is shown
    assert "`protocol.runs_per_setting`" in prompt and "- `method`" not in prompt


@pytest.mark.asyncio
async def test_a_review_that_names_a_part_allows_that_part_only(tmp_path: Path) -> None:
    state = {"topic": "t", "iteration": 1, "design": DESIGN,
             "review": {"verdict": "revise", "weaknesses": ["the comparator is not tuned"], "suggestions": [], "blocking": ""}}
    eng = _engine(tmp_path, [json.dumps({"method": "tuned", "expected_outcome": "changed"})])
    got = await eng._redesign_by_parts(state)
    assert got is not None and got["method"] == "tuned"
    assert got["expected_outcome"] == DESIGN["expected_outcome"]


@pytest.mark.asyncio
async def test_a_redesign_whose_notes_name_no_part_asks_for_the_whole_design(tmp_path: Path) -> None:
    state = {"topic": "t", "iteration": 1, "design": DESIGN, "review": {"verdict": "revise", "weaknesses": ["unconvincing"]}}
    eng = _engine(tmp_path, [])
    assert await eng._redesign_by_parts(state) is None
    assert eng._client.chat.call_count == 0


@pytest.mark.asyncio
async def test_the_first_design_is_not_a_redesign(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    assert await eng._redesign_by_parts({**PRECISION_STATE, "iteration": 0}) is None
