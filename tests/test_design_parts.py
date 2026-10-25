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


def test_cited_paths_are_checked_against_the_real_design_in_full() -> None:
    got = dp.cited_parts(["protocol.grid.strength", "method", "protocol", "protocol.orracles", "nothing", 3], DESIGN)
    assert got == ["protocol.grid", "method"]  # `protocol` alone and unknown names are dropped; a deeper path is its listed part


def test_a_protocol_pointer_names_only_that_sub_part() -> None:
    assert dp.names("protocol.grid", "protocol.grid") and dp.names("protocol.grid.strength", "protocol.grid")
    assert not dp.names("protocol", "protocol.runs_per_setting")
    assert not dp.names("protocol.grid", "protocol.runs_per_setting")
    assert dp.names("method", "method")


def test_a_misspelt_protocol_part_is_refused_with_the_reason() -> None:
    got, applied, refused = dp.merge_parts(DESIGN, {"protocol": {"orracles": [1]}}, None)
    assert got == DESIGN and applied == []
    assert refused[0]["key"] == "protocol.orracles" and "misspelt" in refused[0]["reason"]
    assert dp.invalid_reason("protocol.grid.x", DESIGN) == ""  # a setting under a part that exists is fine
    assert dp.invalid_reason("method.x", DESIGN)


def test_a_part_that_is_needed_beyond_the_asked_ones_is_accepted_with_its_reason() -> None:
    reply = {"protocol": {"runs_per_setting": 900, "seed_policy": "x"},
             "needs": [{"path": "protocol.seed_policy", "why": "more runs need more streams"},
                       {"path": "method", "why": ""}]}
    got, applied, refused = dp.merge_parts(DESIGN, reply, ["protocol.runs_per_setting"])
    assert got["protocol"]["seed_policy"] == "x" and got["method"] == DESIGN["method"]
    assert applied == ["protocol.runs_per_setting", "protocol.seed_policy"] and refused == []


def test_a_need_that_is_a_whole_group_of_parts_is_refused() -> None:
    reply = {"protocol": {"runs_per_setting": 900, "seed_policy": "x", "grid": {"strength": [9]}},
             "needs": [{"path": "protocol", "why": "everything"}, {"path": "protocol.grid", "why": ""}]}
    got, applied, refused = dp.merge_parts(DESIGN, reply, ["protocol.runs_per_setting"])
    assert applied == ["protocol.runs_per_setting"] and got["protocol"]["seed_policy"] == DESIGN["protocol"]["seed_policy"]
    _ok, bad = dp.accepted_needs(DESIGN, reply, ["protocol.runs_per_setting"])
    assert {b["key"] for b in bad} == {"protocol", "protocol.grid"}
    # an ancestor of an allowed part would widen it to everything under it
    _ok, bad = dp.accepted_needs(DESIGN, {"needs": [{"path": "protocol.grid", "why": "w"}]}, ["protocol.grid.strength"])
    assert bad and "whole group" in bad[0]["reason"]


def test_a_panel_review_carries_and_merges_the_cited_parts() -> None:
    from core.engine import _aggregate_panel_reviews as agg
    panel = [{"persona": "a", "verdict": "revise", "score": 3, "status": "ok", "design_parts": ["method"]},
             {"persona": "b", "verdict": "revise", "score": 3, "status": "ok", "design_parts": ["method", "protocol.grid"]}]
    assert agg(panel)["design_parts"] == ["method", "protocol.grid"]


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


@pytest.mark.asyncio
async def test_an_objection_pointing_at_protocol_alone_names_no_protocol_part(tmp_path: Path) -> None:
    whole = {**DESIGN, "protocol": {**DESIGN["protocol"], "runs_per_setting": 5}}
    obj = [{"check": "precision", "where": "protocol", "objection": "o", "why": "w", "fix": "f", "severity": "low"}]
    design, _ = await _audit(tmp_path, {"objections_addressed": obj, "amended_design": whole})
    assert design["protocol"]["runs_per_setting"] == 100


@pytest.mark.asyncio
async def test_a_change_to_a_misspelt_protocol_part_is_refused(tmp_path: Path) -> None:
    obj = [{"check": "oracle", "where": "protocol.oracles", "objection": "o", "why": "w", "fix": "f", "severity": "low",
            "changes": [{"path": "protocol.orracles", "value": [1]}]}]
    design, _ = await _audit(tmp_path, {"objections_addressed": obj, "checked_no_problem": []})
    assert design == DESIGN


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
async def test_a_precision_redesign_is_not_widened_by_what_a_person_said(tmp_path: Path) -> None:
    state = {**PRECISION_STATE, "feedback_history": [{"iteration": 0, "text": "change the method"}],
             "review": {"verdict": "revise", "design_parts": ["method"]}}
    eng = _engine(tmp_path, [])
    assert await eng._redesign_scope(state) == dp.allowed_for_precision()
    assert eng._client.chat.call_count == 0


@pytest.mark.asyncio
async def test_a_review_that_cites_a_part_allows_that_part_only_and_asks_no_mapping(tmp_path: Path) -> None:
    state = {"topic": "t", "iteration": 1, "design": DESIGN,
             "review": {"verdict": "revise", "weaknesses": ["w"], "design_parts": ["method", "protocol.nonsense"]}}
    eng = _engine(tmp_path, [json.dumps({"method": "tuned", "expected_outcome": "changed"})])
    got = await eng._redesign_by_parts(state)
    assert got is not None and got["method"] == "tuned"
    assert got["expected_outcome"] == DESIGN["expected_outcome"]
    assert eng._client.chat.call_count == 1  # the parts request only


@pytest.mark.asyncio
async def test_a_persons_note_with_no_path_is_mapped_by_the_model_once(tmp_path: Path) -> None:
    state = {"topic": "t", "iteration": 1, "design": DESIGN,
             "feedback_history": [{"iteration": 0, "text": "use more repeats and a wider spread of strengths"}]}
    eng = _engine(tmp_path, [json.dumps({"parts": ["protocol.runs_per_setting", "protocol.grid", "not.a.part"]}),
                             json.dumps({"protocol": {"runs_per_setting": 400}, "method": "other"})])
    got = await eng._redesign_by_parts(state)
    assert got is not None and got["protocol"]["runs_per_setting"] == 400 and got["method"] == DESIGN["method"]
    sent = _sent(eng)
    assert len(sent) == 2 and "- `protocol.runs_per_setting`" in sent[0] and "more repeats" in sent[0]
    assert "`protocol.grid`" in sent[1] and "- `method`" not in sent[1]


@pytest.mark.asyncio
async def test_a_note_that_maps_to_no_part_asks_for_the_whole_design(tmp_path: Path) -> None:
    state = {"topic": "t", "iteration": 1, "design": DESIGN, "review": {"verdict": "revise", "weaknesses": ["unconvincing"]}}
    eng = _engine(tmp_path, [json.dumps({"parts": []})])
    assert await eng._redesign_by_parts(state) is None
    assert eng._client.chat.call_count == 1  # the mapping only


def test_the_review_prompt_asks_for_the_parts_it_is_about() -> None:
    from core.engine import _design_parts_listing
    assert "`protocol.runs_per_setting`" in _design_parts_listing(DESIGN) and "`method`" in _design_parts_listing(DESIGN)
    text = (Path(__file__).resolve().parent.parent / "agents" / "review.md").read_text(encoding="utf-8")
    assert '"design_parts"' in text


@pytest.mark.asyncio
async def test_the_first_design_is_not_a_redesign(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    assert await eng._redesign_by_parts({**PRECISION_STATE, "iteration": 0}) is None
