"""Tests for runtime provider-model discovery (``core.provider_models_discover``).

Each provider's discovery function hits a different upstream (OpenAI
HTTP, Ollama HTTP, the VSCode persistent bridge), so the suite mocks
those at the urllib / probe layer rather than exercising any network.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from core.provider_models_discover import (
    discover,
    discover_ollama,
    discover_openai,
    discover_vscode_extension,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def test_discover_openai_returns_none_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No API key → no discovery. Caller falls back to the curated
    static list; we don't try the request and don't surface a bogus
    error in the UI."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _run(discover_openai()) is None


def test_discover_openai_filters_to_chat_shaped_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI /v1/models returns embeddings / image / deprecated
    models alongside chat ones; the picker only renders chat-shaped
    IDs (gpt-, o1, o3, o4, chatgpt-)."""
    captured: dict[str, Any] = {}

    async def fake_get(url: str, *, headers=None, timeout_s=5.0):
        captured["url"] = url
        captured["auth"] = (headers or {}).get("Authorization")
        return {
            "data": [
                {"id": "gpt-5"},
                {"id": "gpt-4o"},
                {"id": "text-embedding-3-large"},
                {"id": "dall-e-3"},
                {"id": "o1-mini"},
                {"id": "whisper-1"},
            ],
        }
    monkeypatch.setattr(
        "core.provider_models_discover._http_get_json", fake_get,
    )
    models = _run(discover_openai(api_key="sk-test"))
    assert models is not None
    values = {m["value"] for m in models}
    assert values == {"gpt-5", "gpt-4o", "o1-mini"}
    assert captured["url"] == "https://api.openai.com/v1/models"
    assert captured["auth"] == "Bearer sk-test"


def test_discover_openai_returns_none_on_http_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get(*_args, **_kwargs):
        return None  # _http_get_json swallows errors and returns None
    monkeypatch.setattr(
        "core.provider_models_discover._http_get_json", fake_get,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert _run(discover_openai()) is None


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def test_discover_ollama_translates_tags_to_picker_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama /api/tags reports installed models with size in bytes;
    the picker shows a human-readable GB label as the description."""
    async def fake_get(url: str, *, headers=None, timeout_s=5.0):
        assert url.endswith("/api/tags")
        return {
            "models": [
                {"name": "llama3.1:70b", "size": 40_000_000_000},
                {"name": "qwen2.5:32b", "size": 18_000_000_000},
            ],
        }
    monkeypatch.setattr(
        "core.provider_models_discover._http_get_json", fake_get,
    )
    models = _run(discover_ollama())
    assert models is not None
    assert len(models) == 2
    names = {m["value"] for m in models}
    assert names == {"llama3.1:70b", "qwen2.5:32b"}
    # GB label appears in description
    by_name = {m["value"]: m for m in models}
    assert "GB" in by_name["llama3.1:70b"]["description"]


def test_discover_ollama_returns_none_when_server_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get(*_args, **_kwargs):
        return None
    monkeypatch.setattr(
        "core.provider_models_discover._http_get_json", fake_get,
    )
    assert _run(discover_ollama()) is None


# ---------------------------------------------------------------------------
# vscode_extension
# ---------------------------------------------------------------------------


def test_discover_vscode_extension_returns_none_without_bridge() -> None:
    """No socket and no port → nothing to ask. Caller falls back to
    the static {gpt-4o, claude-3-5-sonnet, gemini-2.0-flash} list."""
    assert _run(discover_vscode_extension()) is None


def test_discover_vscode_extension_returns_none_when_socket_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Socket configured but probe says no listener → return None
    (don't try to open a doomed bridge)."""
    async def _no(*_args, **_kwargs):
        return False
    # Patch the source module — discover_vscode_extension imports the
    # name lazily inside the function body, so the module-local symbol
    # in core.provider_models_discover doesn't exist until first call.
    monkeypatch.setattr("web._bridge_probe.is_socket_listening", _no)
    assert _run(discover_vscode_extension(socket_path="/tmp/fake.sock")) is None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_discover_dispatch_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_openai(*_args, **_kwargs):
        return [{"value": "gpt-5", "label": "gpt-5", "description": ""}]
    monkeypatch.setattr(
        "core.provider_models_discover.discover_openai", fake_openai,
    )
    out = _run(discover("openai"))
    assert out == [{"value": "gpt-5", "label": "gpt-5", "description": ""}]


def test_discover_unknown_provider_returns_none() -> None:
    """CLI providers (claude_cli, codex_cli, gemini_cli, copilot_cli)
    don't have a stable list endpoint — None means "use static list."""
    for p in ("claude_cli", "codex_cli", "gemini_cli", "copilot_cli", "made-up"):
        assert _run(discover(p)) is None, p
