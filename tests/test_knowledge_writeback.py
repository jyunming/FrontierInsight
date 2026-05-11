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

    def fake_add(self, *, quest_id, paper_md_path, summary, metadata=None) -> bool:  # noqa: ANN001
        captured.append({
            "quest_id": quest_id,
            "paper_md_exists": paper_md_path.exists(),
            "summary": summary,
            "metadata": metadata or {},
        })
        return True

    async def fake_asearch(self, query, *, top_k=None, chosen_idea=None, chat_fn=None):  # noqa: ANN001
        return []

    monkeypatch.setattr("core.knowledge.Knowledge.add_quest_artifacts", fake_add)
    monkeypatch.setattr("core.knowledge.Knowledge.asearch", fake_asearch)
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
    # Rich metadata threaded through: verdict + structured findings.
    meta = captured[0]["metadata"]
    assert meta["verdict"] == "accept"
    assert meta["key_findings"] == ["linear"]
    assert meta["provider"] == "openai"
    assert "result_json" in meta


# Build a "revise"-verdict variant of the fake responses so the engine
# terminates with verdict != accept and the accept-gate fires.
_FAKE_REVISE = dict(_FAKE)
_FAKE_REVISE["Review"] = json.dumps({
    "verdict": "revise", "score": 2,
    "strengths": [], "weaknesses": ["thin"], "suggestions": ["redo"], "blocking": "n",
})


def _fake_response_revise(prompt: str) -> str:
    head = prompt.lstrip().splitlines()[0]
    for tag, resp in _FAKE_REVISE.items():
        if tag in head:
            return resp
    return "{}"


@pytest.mark.asyncio
async def test_quest_writeback_skipped_when_verdict_revise_and_accept_gate_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `write_back_only_on_accept=True` (default), a quest whose
    final review verdict is `revise` must NOT be ingested."""
    cfg = Config(
        topic="revise-verdict test",
        title="revise-test",
        provider=ProviderConfig(name="openai"),
        # max_iterations=1 + review_loop=False means a single pass; the
        # review-node verdict goes straight to "done" regardless.
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(
            enabled=True, write_back_quests=True, write_back_only_on_accept=True,
        ),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_revise(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    captured: list[dict] = []
    def fake_add(self, *, quest_id, paper_md_path, summary, metadata=None) -> bool:  # noqa: ANN001
        captured.append({"quest_id": quest_id})
        return True
    async def fake_asearch(self, q, *, top_k=None, chosen_idea=None, chat_fn=None):  # noqa: ANN001
        return []
    monkeypatch.setattr("core.knowledge.Knowledge.add_quest_artifacts", fake_add)
    monkeypatch.setattr("core.knowledge.Knowledge.asearch", fake_asearch)
    monkeypatch.setattr(
        "core.knowledge.Knowledge.__init__",
        lambda self, c: setattr(self, "cfg", c) or setattr(self, "enabled", True)
        or setattr(self, "_brain", object()) or setattr(self, "_retriever", None),
    )

    engine = Engine(cfg)
    await engine.run()

    # Accept-gate fired: no write-back happened despite write_back_quests=True.
    assert captured == []


@pytest.mark.asyncio
async def test_quest_writeback_runs_on_revise_when_accept_gate_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `write_back_only_on_accept=False`, every finished quest
    lands (used for bootstrapping an empty corpus)."""
    cfg = Config(
        topic="ungated writeback test",
        title="ungated-test",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(
            enabled=True, write_back_quests=True, write_back_only_on_accept=False,
        ),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_revise(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    captured: list[dict] = []
    def fake_add(self, *, quest_id, paper_md_path, summary, metadata=None) -> bool:  # noqa: ANN001
        captured.append({"quest_id": quest_id, "verdict": (metadata or {}).get("verdict")})
        return True
    async def fake_asearch(self, q, *, top_k=None, chosen_idea=None, chat_fn=None):  # noqa: ANN001
        return []
    monkeypatch.setattr("core.knowledge.Knowledge.add_quest_artifacts", fake_add)
    monkeypatch.setattr("core.knowledge.Knowledge.asearch", fake_asearch)
    monkeypatch.setattr(
        "core.knowledge.Knowledge.__init__",
        lambda self, c: setattr(self, "cfg", c) or setattr(self, "enabled", True)
        or setattr(self, "_brain", object()) or setattr(self, "_retriever", None),
    )

    engine = Engine(cfg)
    await engine.run()

    assert len(captured) == 1
    assert captured[0]["verdict"] == "revise"
