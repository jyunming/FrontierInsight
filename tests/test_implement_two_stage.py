"""Tests for the two-stage implement node introduced in Phase 2.

The implement flow is now: ``design → implement_outline → implement
→ execute``. The outline node produces a JSON scaffold (function
signatures + constants + RESULT_JSON template + deps); the body
node consumes that scaffold and fills in the function bodies via
the legacy two-section fenced-block format.

The split exists to address the "one-shot must be perfect" prompt
design: ~9 minute extended-thinking spans on Sonnet 4.6 for complex
topics (lithography sims, multi-axis sweeps) become two shorter
calls with feedback in between.

Covered here:
  - ``implement_outline`` parses scaffold JSON and populates state.
  - When the outline call fails (transient error, unparseable
    response), the body node falls back to the legacy single-shot
    prompt — preserves resumability AND the pre-Phase-2 contract.
  - A pre-Phase-2 checkpoint (no ``implement_outline`` field)
    resumes cleanly: the outline node skips, the body uses legacy.
  - The graph edges: design routes to ``implement_outline`` which
    chains to ``implement``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

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


def _make_config(tmp_path: Path) -> Config:
    return Config(
        topic="phase 2 outline+body smoke",
        title="phase2-smoke",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(kinds=["paper_md"], output_dir=tmp_path / "outputs"),
    )


# ---------------------------------------------------------------------------
# Outline-stage parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_implement_outline_populates_state_on_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: the outline LLM returns a scaffold + signatures
    payload; ``_node_implement_outline`` parses and stores it."""
    cfg = _make_config(tmp_path)
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    # Lazy-init the LLM client via a mock; the node calls ``self._chat``
    # which routes through it.
    mock_client = MagicMock()
    # Real attrs that ``_log_chat_cost`` reads after each chat call;
    # MagicMock auto-attrs blow up json.dumps inside the cost logger.
    mock_client.last_usage = None
    mock_client.last_model = "test-model"
    mock_client.endpoint = MagicMock()
    mock_client.endpoint.provider_name = "openai"
    expected = {
        "scaffold": "import json\n\n\ndef foo():\n    raise NotImplementedError\n",
        "functions": [
            {"name": "foo", "signature": "def foo() -> None:",
             "one_line_purpose": "demo"}
        ],
        "data_flow": "trivial",
        "constants": [
            {"name": "C", "value": "299792458", "source": "first-principles"}
        ],
        "result_json_template": "{}",
        "deps": [],
    }

    async def fake_chat(*args, **kwargs):
        return json.dumps(expected)

    mock_client.chat = fake_chat
    eng._client = mock_client
    state = {
        "topic": "t", "title": "x",
        "design": {"hypothesis": "h"},
        "clarify_answers": {},
    }
    result = await eng._node_implement_outline(state)
    assert "implement_outline" in result
    outline = result["implement_outline"]
    assert outline["scaffold"].startswith("import json")
    assert outline["functions"][0]["name"] == "foo"
    assert outline["constants"][0]["source"] == "first-principles"


@pytest.mark.asyncio
async def test_implement_outline_falls_back_on_unparseable_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the LLM returns JSON missing the required ``scaffold`` key,
    the outline node logs a warning and returns an empty outline —
    triggering the body node's legacy single-shot fallback."""
    cfg = _make_config(tmp_path)
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    mock_client = MagicMock()
    # Real attrs that ``_log_chat_cost`` reads after each chat call;
    # MagicMock auto-attrs blow up json.dumps inside the cost logger.
    mock_client.last_usage = None
    mock_client.last_model = "test-model"
    mock_client.endpoint = MagicMock()
    mock_client.endpoint.provider_name = "openai"

    async def fake_chat(*args, **kwargs):
        return json.dumps({"code": "print('hi')", "deps": []})  # wrong shape

    mock_client.chat = fake_chat
    eng._client = mock_client
    state = {"topic": "t", "title": "x", "design": {}, "clarify_answers": {}}
    result = await eng._node_implement_outline(state)
    # Empty outline → body's fallback path.
    assert result == {"implement_outline": {}}


@pytest.mark.asyncio
async def test_implement_outline_failure_isolated_on_chat_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient provider error during the outline call must NOT
    propagate — it returns an empty outline so the body can fall back
    to the legacy single-shot path. Matches the ``design_self_critique``
    contract from PR #131."""
    cfg = _make_config(tmp_path)
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    mock_client = MagicMock()
    # Real attrs that ``_log_chat_cost`` reads after each chat call;
    # MagicMock auto-attrs blow up json.dumps inside the cost logger.
    mock_client.last_usage = None
    mock_client.last_model = "test-model"
    mock_client.endpoint = MagicMock()
    mock_client.endpoint.provider_name = "openai"

    async def fake_chat(*args, **kwargs):
        raise RuntimeError("simulated transient failure")

    mock_client.chat = fake_chat
    eng._client = mock_client
    state = {"topic": "t", "title": "x", "design": {}, "clarify_answers": {}}
    result = await eng._node_implement_outline(state)
    assert result == {"implement_outline": {}}


@pytest.mark.asyncio
async def test_implement_outline_skips_when_already_populated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume idempotency: if the checkpoint already has a populated
    outline (mid-run re-entry after a body failure), the outline node
    must NOT re-bill an LLM call. Same contract as ``_node_clarify``
    when ``clarify_done`` is True."""
    cfg = _make_config(tmp_path)
    eng = Engine(cfg)
    mock_client = MagicMock()
    # Real attrs that ``_log_chat_cost`` reads after each chat call;
    # MagicMock auto-attrs blow up json.dumps inside the cost logger.
    mock_client.last_usage = None
    mock_client.last_model = "test-model"
    mock_client.endpoint = MagicMock()
    mock_client.endpoint.provider_name = "openai"
    chat_invocations = {"n": 0}

    async def fake_chat(*args, **kwargs):
        chat_invocations["n"] += 1
        return "{}"

    mock_client.chat = fake_chat
    eng._client = mock_client
    state = {
        "topic": "t", "title": "x",
        "design": {}, "clarify_answers": {},
        "implement_outline": {
            "scaffold": "import json\n",
            "functions": [],
            "constants": [],
            "deps": [],
        },
    }
    result = await eng._node_implement_outline(state)
    assert result == {}
    assert chat_invocations["n"] == 0


# ---------------------------------------------------------------------------
# Body stage routing — uses outline if present, falls back if absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_implement_body_uses_outline_block_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The body node's prompt must interpolate the outline's scaffold +
    function list. We snoop the prompt text to confirm."""
    cfg = _make_config(tmp_path)
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir()
    captured_prompts: list[str] = []
    mock_client = MagicMock()
    # Real attrs that ``_log_chat_cost`` reads after each chat call;
    # MagicMock auto-attrs blow up json.dumps inside the cost logger.
    mock_client.last_usage = None
    mock_client.last_model = "test-model"
    mock_client.endpoint = MagicMock()
    mock_client.endpoint.provider_name = "openai"

    async def fake_chat(messages, **kw):
        captured_prompts.append(messages[-1]["content"] if isinstance(messages, list) else "")
        return "```python\nimport json\nprint('RESULT_JSON: {}')\n```\nDEPS: numpy"

    mock_client.chat = fake_chat
    eng._client = mock_client
    state = {
        "topic": "t", "title": "x",
        "design": {"hypothesis": "h", "dependencies": ["numpy"]},
        "clarify_answers": {},
        "implement_outline": {
            "scaffold": "import json\n\ndef foo(): raise NotImplementedError\n",
            "functions": [{"name": "foo", "signature": "def foo() -> None:", "one_line_purpose": "demo"}],
            "data_flow": "trivial",
            "constants": [],
            "result_json_template": "{}",
            "deps": ["numpy"],
        },
    }
    result = await eng._node_implement(state)
    assert result["code"]
    assert "numpy" in result["deps"]
    # The outline scaffold should have been substituted into the body
    # prompt — search for the function-signature literal.
    assert captured_prompts, "no prompt was captured"
    body_prompt = captured_prompts[-1]
    assert "def foo() -> None:" in body_prompt
    assert "Implementation Body" in body_prompt


@pytest.mark.asyncio
async def test_implement_body_falls_back_to_legacy_when_outline_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the outline is empty (pre-Phase-2 resume OR outline call
    failed), the body uses the legacy ``agents/implement.md`` prompt
    — preserves the single-shot contract for resume."""
    cfg = _make_config(tmp_path)
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir()
    captured_prompts: list[str] = []
    mock_client = MagicMock()
    # Real attrs that ``_log_chat_cost`` reads after each chat call;
    # MagicMock auto-attrs blow up json.dumps inside the cost logger.
    mock_client.last_usage = None
    mock_client.last_model = "test-model"
    mock_client.endpoint = MagicMock()
    mock_client.endpoint.provider_name = "openai"

    async def fake_chat(messages, **kw):
        captured_prompts.append(messages[-1]["content"] if isinstance(messages, list) else "")
        return "```python\nprint('RESULT_JSON: {}')\n```\nDEPS: "

    mock_client.chat = fake_chat
    eng._client = mock_client
    # No ``implement_outline`` field — simulates a pre-Phase-2
    # checkpoint OR an outline call that returned empty.
    state = {
        "topic": "t", "title": "x",
        "design": {"hypothesis": "h"},
        "clarify_answers": {},
    }
    result = await eng._node_implement(state)
    assert result["code"]
    body_prompt = captured_prompts[-1]
    # Legacy prompt headline: "Implementation" (no "Body" qualifier).
    # The "Implementation Body" string MUST NOT appear in the fallback.
    assert "Implementation Body" not in body_prompt
    assert "Implementation" in body_prompt
