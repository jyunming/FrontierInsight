"""A redesign (the design step run again after a run) goes through the same checks as the first design: its grid is made
small enough by the plan's model also when what the topic asks for was added to it after the first look at its size; its
code outline belongs to the design it was written for; its hypothesis is not changed after a run that produced nothing.
Fake model only; the studies are made up."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_grid_size_at_plan import BIG, SMALL, TOPIC
from tests.test_plan_optimisation import HEAT_SINK, _engine


def _redesign(grid: dict[str, list[Any]]) -> dict[str, Any]:
    draft = copy.deepcopy(HEAT_SINK)
    draft["study_type"] = "measure"
    draft["protocol"] = {"oracles": copy.deepcopy(HEAT_SINK["protocol"]["oracles"]), "grid": grid}
    return draft


@pytest.mark.asyncio
async def test_a_redesign_made_larger_by_the_topic_is_asked_to_make_its_grid_small_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    sent: list[str] = []

    async def chat(prompt: str, **kw: Any) -> str:
        sent.append(prompt)
        return json.dumps({"grid": SMALL, "sub_studies": [{"name": "all", "vary": list(SMALL), "hold": {}}]})

    async def draft(prompt: str, node: str = "") -> dict[str, Any]:
        return _redesign(SMALL)

    async def audit(state: Any, design: Any) -> tuple[Any, Any]:
        return design, None

    async def topic(state: Any, design: Any, plan_sha: str) -> tuple[Any, str]:
        # what the topic asks for is added here, and the study becomes too large to run
        return {**design, "protocol": {**design["protocol"], "grid": BIG}}, plan_sha

    async def nothing(state: Any, design: Any, **kw: Any) -> Any:
        return design

    eng._chat = chat  # type: ignore[method-assign]
    eng._client = object()
    eng._design_reply = draft  # type: ignore[method-assign]
    eng._audit_design = audit  # type: ignore[method-assign]
    eng._hold_the_design_to_the_topic = topic  # type: ignore[method-assign]
    eng._read_topic_before_the_prompt = lambda state: _none()  # type: ignore[method-assign]
    eng._design_prompt = lambda state: ""  # type: ignore[method-assign]
    eng._complete_a_drafted_search = nothing  # type: ignore[method-assign]
    eng._complete_drafted_parts = nothing  # type: ignore[method-assign]
    eng._audit_the_design_that_runs = lambda state, design: _same(design)  # type: ignore[method-assign]
    eng._check_the_ledger_of_a_redesign = lambda state, design: None  # type: ignore[method-assign]
    state = {"topic": TOPIC, "iteration": 1, "design": _redesign(SMALL)}
    patch = await eng._node_design(state)
    assert len(sent) == 1 and "settings in `protocol.grid` are" in sent[0]
    assert patch["design"]["protocol"]["grid"] == SMALL and patch["design"]["protocol"]["sub_studies"]
    assert eng._grid_too_large_stop({**state, "design": patch["design"]}) is None
    assert (eng.fi_dir / "drafted_grid_asked_again.json").is_file()


async def _none() -> None:
    return None


async def _same(design: Any) -> Any:
    return design


def _outline_engine(tmp_path: Path, calls: list[str]) -> Any:
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
    from core.engine import Engine
    eng = Engine(Config(
        topic="made-up study", title="outline-key", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False), execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(kinds=["paper_md"], output_dir=tmp_path / "out")))
    eng.quest_root.mkdir(parents=True, exist_ok=True)

    async def chat(prompt: str, **kw: Any) -> str:
        calls.append(prompt)
        return json.dumps({"scaffold": f"# outline {len(calls)}\n", "functions": [], "constants": []})

    eng._chat = chat  # type: ignore[method-assign]
    return eng


@pytest.mark.asyncio
async def test_the_outline_is_kept_for_the_same_design_and_written_again_for_a_new_one(tmp_path: Path) -> None:
    calls: list[str] = []
    eng = _outline_engine(tmp_path, calls)
    state: dict[str, Any] = {"topic": "t", "title": "x", "design": {"hypothesis": "h", "protocol": {"grid": {"a": [1, 2]}}}}
    first = await eng._node_implement_outline(state)
    assert len(calls) == 1 and first["implement_outline"]["scaffold"] == "# outline 1\n"
    state = {**state, **first}
    assert await eng._node_implement_outline(state) == {} and len(calls) == 1  # a resume of the same design: no call
    redesign = {**state, "design": {"hypothesis": "h", "protocol": {"grid": {"a": [1, 2, 3]}}}}
    again = await eng._node_implement_outline(redesign)
    assert len(calls) == 2 and again["implement_outline"]["scaffold"] == "# outline 2\n"
    assert again["implement_outline_for"] != first["implement_outline_for"]


@pytest.mark.asyncio
async def test_an_outline_from_a_checkpoint_made_before_the_key_existed_is_still_reused(tmp_path: Path) -> None:
    calls: list[str] = []
    eng = _outline_engine(tmp_path, calls)
    state = {"topic": "t", "design": {"hypothesis": "h"}, "implement_outline": {"scaffold": "# old\n"}}
    assert await eng._node_implement_outline(state) == {} and calls == []
