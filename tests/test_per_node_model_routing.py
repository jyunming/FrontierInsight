"""Per-node model routing tests.

Verifies that ``provider.node_models`` actually routes per-call model
selection through both transports:

  * CLI exec: `_run_cli` receives the resolved model in argv after
    `_CliSpec.model_flag` for ``copilot_cli`` / ``claude_cli`` / etc.
  * HTTP proxy and direct: the OpenAI-compatible POST body's
    ``"model"`` field is the resolved model.

Plus the Engine-side resolution rules (`_model_for_node`):
  - exact node-name match wins,
  - dotted-key fallback (e.g. ``review_panel.methodologist`` →
    ``review_panel``),
  - falls through to None (= endpoint default) on miss.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine
from core.provider import LLMClient, resolve_endpoint


def _mk_engine(tmp_path: Path, *, node_models=None, provider_model="gpt-5") -> Engine:
    cfg = Config(
        topic="phase O routing tests",
        title="po",
        provider=ProviderConfig(
            name="openai", model=provider_model,
            node_models=node_models,
        ),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="off",
            ideate_reflect=False,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    return Engine(cfg)


# --- Engine._model_for_node resolution rules -------------------------------


def test_model_for_node_returns_none_when_no_node_models(tmp_path: Path) -> None:
    eng = _mk_engine(tmp_path, node_models=None)
    assert eng._model_for_node("review") is None
    assert eng._model_for_node(None) is None


def test_model_for_node_exact_match_wins(tmp_path: Path) -> None:
    eng = _mk_engine(tmp_path, node_models={
        "ideate": "claude-opus-4-7",
        "review": "gpt-5",
    })
    assert eng._model_for_node("ideate") == "claude-opus-4-7"
    assert eng._model_for_node("review") == "gpt-5"


def test_model_for_node_dotted_key_falls_back_to_base(tmp_path: Path) -> None:
    """Persona-keyed lookups (`review_panel.methodologist`) fall back
    to the un-qualified `review_panel` entry when there's no
    persona-specific override."""
    eng = _mk_engine(tmp_path, node_models={
        "review_panel": "claude-opus-4-7",
    })
    assert eng._model_for_node("review_panel.methodologist") == "claude-opus-4-7"
    assert eng._model_for_node("review_panel.devil_advocate") == "claude-opus-4-7"


def test_model_for_node_specific_persona_overrides_base(tmp_path: Path) -> None:
    eng = _mk_engine(tmp_path, node_models={
        "review_panel": "claude-opus-4-7",
        "review_panel.statistician": "gpt-5",
    })
    assert eng._model_for_node("review_panel.statistician") == "gpt-5"
    assert eng._model_for_node("review_panel.methodologist") == "claude-opus-4-7"


def test_model_for_node_miss_returns_none(tmp_path: Path) -> None:
    """A node name absent from `node_models` resolves to None so the
    LLMClient falls through to `endpoint.model`."""
    eng = _mk_engine(tmp_path, node_models={"ideate": "x"})
    assert eng._model_for_node("review") is None


# --- LLMClient.chat — HTTP path per-call model override --------------------


@pytest.mark.asyncio
async def test_http_chat_uses_per_call_model_override() -> None:
    """When `model` is passed to `LLMClient.chat`, the OpenAI-compat
    POST body sends that model, NOT the endpoint default. This is the
    plumbing that makes `provider.node_models` work for the HTTP-proxy
    and HTTP-direct transports (claude_code, github_copilot_*, openai, …)."""
    ep = resolve_endpoint(ProviderConfig(name="openai", model="default-model"))
    captured: dict = {}

    async def fake_post(url, json=None, headers=None):  # noqa: ANN001
        captured["url"] = url
        captured["body"] = json
        resp = MagicMock()
        resp.status_code = 200
        resp.json = lambda: {"choices": [{"message": {"content": "ok"}}]}
        resp.raise_for_status = lambda: None
        return resp

    client = LLMClient(ep)
    client._http = MagicMock()
    client._http.post = fake_post
    client._http.aclose = AsyncMock()
    try:
        result = await client.chat(
            [{"role": "user", "content": "hi"}], model="overridden-model",
        )
    finally:
        await client.aclose()

    assert result == "ok"
    assert captured["body"]["model"] == "overridden-model"


@pytest.mark.asyncio
async def test_http_chat_falls_back_to_endpoint_model_when_no_override() -> None:
    ep = resolve_endpoint(ProviderConfig(name="openai", model="default-model"))
    captured: dict = {}

    async def fake_post(url, json=None, headers=None):  # noqa: ANN001
        captured["body"] = json
        resp = MagicMock()
        resp.status_code = 200
        resp.json = lambda: {"choices": [{"message": {"content": "ok"}}]}
        resp.raise_for_status = lambda: None
        return resp

    client = LLMClient(ep)
    client._http = MagicMock()
    client._http.post = fake_post
    client._http.aclose = AsyncMock()
    try:
        await client.chat([{"role": "user", "content": "hi"}])  # no model kwarg
    finally:
        await client.aclose()

    assert captured["body"]["model"] == "default-model"


# --- LLMClient.chat — CLI exec path per-call model override ----------------


@pytest.mark.asyncio
async def test_cli_chat_uses_per_call_model_override(tmp_path: Path) -> None:
    """For CLI providers, the per-call `model` arg flows down to
    `_run_cli` and gets injected after `_CliSpec.model_flag` — so the
    Copilot CLI binary actually receives `--model <X>` for that one
    call (different per call within the same quest)."""
    from core.provider import resolve_endpoint
    ep = resolve_endpoint(ProviderConfig(name="copilot_cli"))
    client = LLMClient(ep, cli_timeout_s=5.0)
    try:
        captured_argv: list = []

        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"hello", b""))
        proc.returncode = 0

        async def fake_spawn(*args, **kw):
            captured_argv.append(list(args))
            return proc

        with patch("core.provider.shutil.which", return_value="/usr/bin/copilot"), \
             patch("core.provider.asyncio.create_subprocess_exec", new=fake_spawn):
            await client.chat(
                [{"role": "user", "content": "hi"}],
                model="claude-opus-4-7",
            )
    finally:
        await client.aclose()

    args = captured_argv[0]
    # The spawn argv looks like: [resolved_copilot, "--model",
    # "claude-opus-4-7", ...spec.argv[1:], ...].
    assert args[0] == "/usr/bin/copilot"
    assert "--model" in args
    mi = args.index("--model")
    assert args[mi + 1] == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_cli_chat_per_call_override_takes_precedence_over_endpoint(
    tmp_path: Path,
) -> None:
    """Even when the endpoint already has a `cli_model_override` set
    (from `provider.model`), a non-empty per-call `model` kwarg wins."""
    ep = resolve_endpoint(ProviderConfig(name="claude_cli", model="opus"))
    assert ep.cli_model_override == "opus"  # endpoint default

    client = LLMClient(ep, cli_timeout_s=5.0)
    try:
        captured: list = []
        # claude_cli now uses ``output_via="stream_json"`` so the
        # path through ``_run_cli`` reads stdout line-by-line. The
        # mock supplies an empty stream (immediate EOF) — this test
        # only cares about argv injection, not the assembled answer.
        proc = AsyncMock()
        stdout = AsyncMock()
        stdout.readline = AsyncMock(return_value=b"")
        proc.stdout = stdout
        proc.stdin = AsyncMock()
        proc.stdin.write = lambda b: None
        proc.stdin.drain = AsyncMock(return_value=None)
        proc.stdin.close = lambda: None
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b"")
        proc.wait = AsyncMock(return_value=0)
        proc.returncode = 0
        proc.kill = lambda: None

        async def fake_spawn(*args, **kw):
            captured.append(list(args))
            return proc

        with patch("core.provider.shutil.which", return_value="/usr/bin/claude"), \
             patch("core.provider.asyncio.create_subprocess_exec", new=fake_spawn):
            # Per-call override.
            await client.chat([{"role": "user", "content": "x"}], model="sonnet")
    finally:
        await client.aclose()

    args = captured[0]
    mi = args.index("--model")
    assert args[mi + 1] == "sonnet"  # NOT "opus"


# --- end-to-end: per-node model routing through Engine -------------------


@pytest.mark.asyncio
async def test_engine_node_calls_route_per_node_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises the full path: `Engine._chat(prompt, node=X)` →
    `self._model_for_node(X)` → `LLMClient.chat(model=...)` → request
    body. We monkeypatch `LLMClient.chat` and capture (node, model)
    tuples to assert routing."""
    eng = _mk_engine(tmp_path, node_models={
        "clarify": "gpt-5-mini",
        "review": "claude-opus-4-7",
        "review_panel.devil_advocate": "gemini-2.5-pro",
        "review_panel": "gpt-5",  # default for un-specified personas
    })

    seen: list[tuple[str, str | None]] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        seen.append((messages[-1]["content"][:30], kw.get("model")))
        return "x"
    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    # `_client` must be an actual LLMClient instance so the monkeypatch
    # on the class actually applies. Build one against a dummy endpoint.
    ep = resolve_endpoint(ProviderConfig(name="openai", model="default-model"))
    eng._client = LLMClient(ep)

    await eng._chat("clarify-prompt", node="clarify")
    await eng._chat("design-prompt", node="design")        # unmapped → None
    await eng._chat("review-prompt", node="review")
    await eng._chat("methodologist", node="review_panel.methodologist")
    await eng._chat("devil_advocate", node="review_panel.devil_advocate")

    by_prompt = {p: m for p, m in seen}
    assert by_prompt["clarify-prompt"] == "gpt-5-mini"
    assert by_prompt["design-prompt"] is None  # falls through to endpoint default
    assert by_prompt["review-prompt"] == "claude-opus-4-7"
    assert by_prompt["methodologist"] == "gpt-5"            # dotted-key base match
    assert by_prompt["devil_advocate"] == "gemini-2.5-pro"  # exact persona match
