"""A design drafted from the topic at the design step (a redesign: it did not come from plan.md) is asked with the same
study-type rules as the plan step, chosen by the same rule, so a topic that asks for the best design can be written as
a search and not as a plain sweep. Fake model; the prompt is captured."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core import engine as eng_mod
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine

SEEK = "Find the wall thickness and height that give the lightest beam whose tip deflection stays under 20 mm"
MEASURE = "How does the settling time of a damped spring change with the damping ratio?"


class Captured(Exception):
    pass


def _engine(tmp_path: Path, topic: str) -> Engine:
    return Engine(Config(
        topic=topic, title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))


async def _redesign_prompt(eng: Engine, topic: str, monkeypatch: pytest.MonkeyPatch) -> str:
    seen: list[str] = []

    async def fake_reply(prompt: str, *, node: str) -> dict[str, Any]:
        seen.append(prompt)
        raise Captured

    monkeypatch.setattr(eng, "_design_reply", fake_reply)
    with pytest.raises(Captured):
        await eng._node_design({"topic": topic, "iteration": 1, "review": {}})
    return seen[0]


@pytest.mark.asyncio
async def test_redesign_of_a_best_design_topic_carries_the_study_type_rules(tmp_path, monkeypatch):
    eng = _engine(tmp_path, SEEK)
    prompt = await _redesign_prompt(eng, SEEK, monkeypatch)
    assert eng_mod._STUDY_TYPE_DIRECTIVE in prompt


@pytest.mark.asyncio
async def test_redesign_of_a_measurement_topic_is_unchanged(tmp_path, monkeypatch):
    eng = _engine(tmp_path, MEASURE)
    prompt = await _redesign_prompt(eng, MEASURE, monkeypatch)
    assert "find_best_design" not in prompt
    assert prompt == eng._design_prompt({"topic": MEASURE, "iteration": 1, "review": {}})


def test_plan_and_redesign_use_one_rule(tmp_path):
    eng = _engine(tmp_path, SEEK)
    state = {"topic": SEEK}
    assert eng._seeks_best_design(state) is True
    assert eng._seeks_best_design({"topic": MEASURE}) is False
    assert eng._study_type_part(state, seek=True).startswith(eng_mod._STUDY_TYPE_DIRECTIVE)
    assert eng._study_type_part(state, seek=False) == ""
