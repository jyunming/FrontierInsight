"""What the design prompt carries: skill summaries and short source excerpts.

Design decides what experiment to run. It needs to know what each source
found and which skill fits, not the passages carrying a source's numbers or
a skill's full API — those are for analyze, write and implement, which keep
them. On the SIR validation quest the full skill text and 4,000-character
excerpts made design the largest prompt of the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine

_SENTENCE = "Sentence {i} reports the final outbreak size."
_MARK = "reports the final outbreak size."


def _cfg(tmp_path: Path, **knowledge) -> Config:
    return Config(
        topic="SIR final outbreak size",
        title="sir",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False, passage_ranking="lexical", **knowledge),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


def _state() -> dict:
    content = " ".join(_SENTENCE.format(i=i) for i in range(400))
    return {
        "topic": "SIR final outbreak size",
        "iteration": 0,
        "exec_result": {},
        "result_json": {},
        "literature": [{
            "content": content,
            "metadata": {"title": "Final size of SIR epidemics", "year": 2001,
                         "doi": "10.1000/sir"},
        }],
    }


def _capture(eng: Engine, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    prompts: dict[str, str] = {}

    async def fake_chat(prompt, node=None, **kw):  # noqa: ANN001
        prompts.setdefault(node, prompt)
        if node == "design":
            return json.dumps({"hypothesis": "h", "method": "m", "dependencies": [],
                               "figures_planned": ["f.png"]})
        if node == "analyze":
            return '{"summary": "ok", "key_findings": []}'
        return "{}"

    monkeypatch.setattr(eng, "_chat", fake_chat)
    return prompts


@pytest.mark.asyncio
async def test_design_reads_short_excerpts_while_analyze_keeps_the_full_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = Engine(_cfg(tmp_path))
    prompts = _capture(eng, monkeypatch)
    await eng._node_design(_state())
    await eng._node_analyze(_state())

    # Passages are chosen whole, so an excerpt stays under its budget
    # rather than filling it; bound the count by the shortest sentence.
    shortest = len(_SENTENCE.format(i=0))
    in_design = prompts["design"].count(_MARK)
    in_analyze = prompts["analyze"].count(_MARK)
    assert 0 < in_design <= 800 // shortest
    assert in_analyze > 3000 // len(_SENTENCE.format(i=100))


@pytest.mark.asyncio
async def test_the_design_excerpt_budget_is_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {}
    for budget in (800, 2000):
        eng = Engine(_cfg(tmp_path, design_literature_excerpt_chars=budget))
        prompts = _capture(eng, monkeypatch)
        await eng._node_design(_state())
        counts[budget] = prompts["design"].count(_MARK)
    assert counts[800] < counts[2000] <= 2000 // len(_SENTENCE.format(i=0))


@pytest.mark.asyncio
async def test_design_gets_the_skill_summary_not_the_full_skill_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = Engine(_cfg(tmp_path))
    prompts = _capture(eng, monkeypatch)
    monkeypatch.setattr(eng, "_skills_summary_block", lambda state=None: "SUMMARY_SENTINEL")
    monkeypatch.setattr(eng, "_skills_block", lambda state=None: "FULL_SENTINEL")
    await eng._node_design(_state())
    assert "SUMMARY_SENTINEL" in prompts["design"]
    assert "FULL_SENTINEL" not in prompts["design"]
