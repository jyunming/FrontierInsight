"""Resume capability: Engine(resume_quest_id=...) reuses the prior
quest_id (and therefore quest_root and state.sqlite), so a quest that
died mid-pipeline on a transient upstream outage can be re-entered
without losing checkpointed work.

The first test pass shipped only id/path checks because the assumption
was that LangGraph would auto-resume from a thread that already had
checkpoints. In production that assumption broke: ainvoke(initial,
config={thread_id}) on a thread with prior state STARTED OVER instead
of resuming, because the non-empty `initial` payload re-seeded the
START node. The fix in `Engine.run` is to call
`graph.aget_state(...)` first and pass `None` as the input to
`ainvoke` when prior state exists; this test pins that decision.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


def _cfg(tmp_path: Path) -> Config:
    return Config(
        topic="resume unit",
        title="resume-unit",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(),
        execution=ExecutionConfig(),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def test_engine_without_resume_generates_fresh_quest_id(tmp_path: Path) -> None:
    e1 = Engine(_cfg(tmp_path))
    e2 = Engine(_cfg(tmp_path))
    assert e1.quest_id != e2.quest_id


def test_engine_with_resume_quest_id_reuses_id_and_root(tmp_path: Path) -> None:
    prior_id = "1700000000-resume-unit-deadbe"
    e = Engine(_cfg(tmp_path), resume_quest_id=prior_id)
    assert e.quest_id == prior_id
    assert e.quest_root.name == prior_id
    # state.sqlite path is what LangGraph keys checkpoints by; reuse
    # is the whole point.
    assert (e.fi_dir / "state.sqlite").parent == e.quest_root / ".fi"


def test_engine_resume_two_engines_same_id_share_checkpoint_path(
    tmp_path: Path,
) -> None:
    """The contract: two Engine instances built with the same
    resume_quest_id resolve to the same on-disk checkpoint. That is
    sufficient for LangGraph's AsyncSqliteSaver to load the prior
    state when ainvoke is called with thread_id=quest_id."""
    qid = "1700000000-resume-unit-cafe11"
    a = Engine(_cfg(tmp_path), resume_quest_id=qid)
    b = Engine(_cfg(tmp_path), resume_quest_id=qid)
    assert a.fi_dir / "state.sqlite" == b.fi_dir / "state.sqlite"


@pytest.mark.asyncio
async def test_engine_run_passes_none_to_ainvoke_when_checkpoint_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual resume contract: when a thread already has saved state,
    ``Engine.run`` must call ``graph.ainvoke(None, ...)`` so LangGraph
    continues from the last completed node. Passing the full initial
    payload (the prior behavior) caused real quests to restart from
    `ideate` with whatever topic the YAML had — silently wiping the
    prior run's design/literature work."""
    qid = "1700000000-resume-good-deadbe"
    engine = Engine(_cfg(tmp_path), resume_quest_id=qid)

    # Capture what `ainvoke` is called with.
    seen_payloads: list[object] = []

    class _FakeGraph:
        def __init__(self, has_state: bool) -> None:
            self._has_state = has_state

        async def aget_state(self, _cfg):
            snap = MagicMock()
            snap.values = (
                {"topic": "prior topic", "design": {"hypothesis": "X"}}
                if self._has_state else {}
            )
            snap.next = ("implement",) if self._has_state else ()
            return snap

        async def ainvoke(self, payload, config=None):
            seen_payloads.append(payload)
            return {"topic": "prior topic", "verdict": "accept"}

    # Stub out _build_graph().compile(checkpointer=...) to return our fake.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_saver(_path):
        yield MagicMock()

    monkeypatch.setattr(
        "core.engine.AsyncSqliteSaver.from_conn_string", fake_saver,
    )
    monkeypatch.setattr(
        "core.engine.resolve_endpoint_async", AsyncMock(return_value=MagicMock(
            base_url="x", model="m",
        )),
    )
    # No-op executor.setup.
    engine.executor.setup = AsyncMock()
    engine._client_cls = MagicMock()

    fake_built = MagicMock()
    fake_built.compile = lambda checkpointer=None: _FakeGraph(has_state=True)
    monkeypatch.setattr(engine, "_build_graph", lambda: fake_built)

    await engine.run()

    # Resume contract: input to ainvoke must be None when a prior state
    # snapshot exists. Anything else (especially the initial payload)
    # restarts the graph from the entry node.
    assert seen_payloads == [None], (
        f"expected ainvoke(None, ...) on resume; got {seen_payloads!r}"
    )


@pytest.mark.asyncio
async def test_engine_run_passes_initial_when_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh quest (no prior state): pass the full ``initial`` payload
    so the graph starts at the entry node with the topic from YAML."""
    engine = Engine(_cfg(tmp_path))  # no resume_quest_id

    seen_payloads: list[object] = []

    class _FakeGraph:
        async def aget_state(self, _cfg):
            snap = MagicMock()
            snap.values = {}   # empty → fresh thread
            snap.next = ()
            return snap

        async def ainvoke(self, payload, config=None):
            seen_payloads.append(payload)
            return {"topic": "resume unit", "verdict": "accept"}

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_saver(_path):
        yield MagicMock()

    monkeypatch.setattr(
        "core.engine.AsyncSqliteSaver.from_conn_string", fake_saver,
    )
    monkeypatch.setattr(
        "core.engine.resolve_endpoint_async", AsyncMock(return_value=MagicMock(
            base_url="x", model="m",
        )),
    )
    engine.executor.setup = AsyncMock()

    fake_built = MagicMock()
    fake_built.compile = lambda checkpointer=None: _FakeGraph()
    monkeypatch.setattr(engine, "_build_graph", lambda: fake_built)

    await engine.run()

    assert len(seen_payloads) == 1
    assert isinstance(seen_payloads[0], dict)
    assert seen_payloads[0]["topic"] == "resume unit"


# ---- copilot_cli agentic-CLI warning -------------------------------------


def test_warn_when_copilot_cli_selected(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """`copilot_cli` is an agent loop, not a chat API. Real symptom seen
    in production: paper.md filled with conversational "Are you trying
    to debug X?" replies, experiment.py reduced to the empty stub. The
    engine must warn loudly so users don't lose hours chasing a phantom
    bug that's actually 'wrong provider'."""
    from core.engine import _PROXY_WARN_SHOWN, _warn_if_unsanctioned_provider

    # Reset the dedup set so this test is order-independent.
    _PROXY_WARN_SHOWN.clear()
    monkeypatch.delenv("FI_SUPPRESS_PROXY_WARN", raising=False)

    with caplog.at_level("WARNING", logger="frontier_insight"):
        _warn_if_unsanctioned_provider("copilot_cli")

    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "agentic" in msg.lower() or "AGENTIC" in msg
    assert "vscode_extension" in msg
    # Sanity: known-good providers do NOT trigger any warning.
    caplog.clear()
    _PROXY_WARN_SHOWN.clear()
    with caplog.at_level("WARNING", logger="frontier_insight"):
        for ok in ("openai", "claude_cli", "codex_cli", "gemini_cli",
                   "vscode_extension"):
            _warn_if_unsanctioned_provider(ok)
    assert not caplog.records, (
        f"clean providers should not warn; got {caplog.records!r}"
    )
