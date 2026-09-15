"""What the design prompt carries: skill summaries, and the full source excerpts.

Design decides which skill a method rests on; the implement stages write the
calls. So design gets each selected skill's summary rather than its full
instructions. On the SIR validation quest, the full text of five skills was
98,147 of design's 146,904 prompt characters.

Design keeps the same literature excerpts as analyze. A design-only budget of
800 characters a source was tried and measured. Replaying that quest's design
step, it chose a 1% major-outbreak threshold in 6 of 10 runs, which at N=100
counts a single extra infection as a major outbreak. With the full excerpts it
did so in 0 of 5 runs, whether design got the skill summary or the full skill
text.
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


def _cfg(tmp_path: Path) -> Config:
    return Config(
        topic="SIR final outbreak size",
        title="sir",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False, passage_ranking="lexical"),
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
async def test_design_reads_the_same_excerpts_as_analyze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = Engine(_cfg(tmp_path))
    prompts = _capture(eng, monkeypatch)
    await eng._node_design(_state())
    await eng._node_analyze(_state())

    in_design = prompts["design"].count(_MARK)
    assert in_design == prompts["analyze"].count(_MARK)
    assert in_design > 3000 // len(_SENTENCE.format(i=100))


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
