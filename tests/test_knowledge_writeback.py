"""Phase F validation: cross-quest memory write-back.

Replaces the in-engine Knowledge with a stub that records every
add_quest_artifacts call, then runs a fake-LLM quest and confirms the
paper was written back with the expected tag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import Engine


_FAKE_EXPERIMENT_CODE = """\
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1]); plt.savefig('figures/r.png', dpi=72)
print('RESULT_JSON: {"v": 1}')
"""


_FAKE = {
    "Ideation": json.dumps({
        "ideas": [{"title": "f", "summary": "x", "feasibility": "high", "novelty": "low"}],
        "chosen": {"title": "f", "rationale": "only"},
    }),
    "Experiment Design": json.dumps({
        "hypothesis": "h",
        "variables": {"independent": [], "dependent": [], "controls": []},
        "method": "m",
        "expected_outcome": "e",
        "figures_planned": ["r.png"],
        "dependencies": ["matplotlib"],
    }),
    "Implementation": json.dumps({"code": _FAKE_EXPERIMENT_CODE, "deps": ["matplotlib"]}),
    "Analysis": json.dumps({
        "summary": "Two points plotted.",
        "key_findings": ["linear"],
        "claims_supported": [],
        "claims_unsupported": [],
        "limitations": [],
    }),
    "Writing": "# f\n\nResults.\n",
    "Review": json.dumps({
        "verdict": "accept", "score": 4,
        "strengths": [], "weaknesses": [], "suggestions": [], "blocking": "",
    }),
}


def _fake_response(prompt: str) -> str:
    head = prompt.lstrip().splitlines()[0]
    for tag, resp in _FAKE.items():
        if tag in head:
            return resp
    return "{}"


@pytest.mark.asyncio
async def test_quest_writeback_invokes_axon_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config(
        topic="memory writeback test",
        title="memory-test",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=True, write_back_quests=True),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    # Stand-in for Axon: capture add_quest_artifacts calls.
    captured: list[dict] = []

    def fake_add(self, *, quest_id: str, paper_md_path: Path, summary: str) -> bool:  # noqa: ANN001
        captured.append({
            "quest_id": quest_id,
            "paper_md_exists": paper_md_path.exists(),
            "summary": summary,
        })
        return True

    def fake_search(self, query: str, *, top_k=None):  # noqa: ANN001
        return []

    monkeypatch.setattr("core.knowledge.Knowledge.add_quest_artifacts", fake_add)
    monkeypatch.setattr("core.knowledge.Knowledge.search", fake_search)
    # Force enabled even without axon installed.
    monkeypatch.setattr(
        "core.knowledge.Knowledge.__init__",
        lambda self, c: setattr(self, "cfg", c) or setattr(self, "enabled", True)
        or setattr(self, "_brain", object()) or setattr(self, "_retriever", None),
    )

    engine = Engine(cfg)
    art = await engine.run()

    assert art.paper_md is not None and art.paper_md.exists()
    assert len(captured) == 1
    assert captured[0]["quest_id"] == engine.quest_id
    assert captured[0]["paper_md_exists"] is True
    assert "Two points plotted." in captured[0]["summary"]
    assert "- linear" in captured[0]["summary"]
