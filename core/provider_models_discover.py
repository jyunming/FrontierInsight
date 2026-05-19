"""Runtime model-list discovery for the provider picker.

The interview's curated ``PROVIDER_MODEL_OPTIONS`` is a static
snapshot — it drifts every time a provider ships a new model family.
This module hits each provider's _model-list_ endpoint (NOT chat) at
runtime to return the live set. Zero LLM tokens consumed; only a
small HTTP / IPC round-trip per provider.

Coverage today:

* ``openai`` / ``codex`` → ``GET https://api.openai.com/v1/models``
* ``ollama`` → ``GET http://127.0.0.1:11434/api/tags``
* ``vscode_extension`` → routes through the persistent bridge so
  the extension calls ``vscode.lm.selectChatModels`` on its side
* ``claude_cli`` / ``codex_cli`` / ``gemini_cli`` / ``copilot_cli``
  → CLI providers don't expose stable ``--list-models`` flags; we
  return ``None`` so the caller falls back to the curated static
  list. Tracked under issue #79.

All functions are async, swallow transport errors (return ``None``
on failure), and complete in well under a second under healthy
conditions. The caller is expected to cache results so the picker
doesn't re-fetch on every keystroke.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError


# Public shape — same as ``interview_schema.json``'s
# ``provider_models[provider]`` entries so the front-end can swap
# discovered list in without re-rendering.
ModelChoice = dict[str, str]


async def _http_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 5.0,
) -> Any | None:
    """Stdlib HTTPS GET → JSON. Returns ``None`` on any failure (DNS
    error, refused connection, HTTP 4xx/5xx, non-JSON body, timeout).
    Run inside a thread so the asyncio loop isn't blocked."""
    def _do() -> Any | None:
        req = Request(url, headers=headers or {})
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                if resp.status >= 400:
                    return None
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None
    return await asyncio.to_thread(_do)


async def discover_openai(api_key: str | None = None) -> list[ModelChoice] | None:
    """``GET /v1/models`` on the OpenAI API. Needs ``OPENAI_API_KEY``
    (caller may pass via ``api_key`` or set it in the env). Returns
    ``None`` when no key is configured — the caller falls back to the
    curated static list so the picker still renders something."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    payload = await _http_get_json(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    if not payload or not isinstance(payload, dict):
        return None
    items = payload.get("data") or []
    # OpenAI lists everything (chat, embeddings, image, deprecated).
    # Filter to the chat-completion-shaped ids the picker actually
    # cares about so we don't drown the user.
    chat_prefixes = ("gpt-", "o1", "o3", "o4", "chatgpt-")
    out: list[ModelChoice] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        mid = str(it.get("id") or "")
        if not mid.startswith(chat_prefixes):
            continue
        out.append({"value": mid, "label": mid, "description": ""})
    # Sort: newest-looking IDs first (highest version number wins).
    out.sort(key=lambda c: c["value"], reverse=True)
    return out


async def discover_ollama(
    host: str = "127.0.0.1", port: int = 11434,
) -> list[ModelChoice] | None:
    """``GET /api/tags`` on a local Ollama server. Returns ``None``
    when Ollama isn't running — the picker falls back to the curated
    list (typically ``llama3.1:8b`` / ``llama3.1:70b`` / ``qwen2.5:32b``)."""
    payload = await _http_get_json(f"http://{host}:{port}/api/tags")
    if not payload or not isinstance(payload, dict):
        return None
    models = payload.get("models") or []
    out: list[ModelChoice] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "")
        if not name:
            continue
        size_bytes = m.get("size")
        size_label = ""
        if isinstance(size_bytes, (int, float)) and size_bytes > 0:
            gb = size_bytes / (1024 ** 3)
            size_label = f"{gb:.1f} GB"
        out.append({
            "value": name, "label": name,
            "description": size_label,
        })
    out.sort(key=lambda c: c["value"])
    return out


async def discover_vscode_extension(
    socket_path: str = "", bridge_port: int = 0,
) -> list[ModelChoice] | None:
    """Ask the extension's PersistentBridge for the live Copilot model
    catalog. Returns ``None`` when the bridge isn't reachable — the
    picker falls back to the static {gpt-4o, claude-3-5-sonnet,
    gemini-2.0-flash} placeholder list."""
    # Reachability check first — avoids creating a full bridge client
    # just to fail at connect time.
    from web._bridge_probe import is_bridge_listening, is_socket_listening
    if socket_path:
        if not await is_socket_listening(socket_path):
            return None
    elif bridge_port > 0:
        if not await is_bridge_listening(bridge_port):
            return None
    else:
        return None
    # NOTE: the model-list call itself is deferred — the
    # PersistentBridge in 0.x only handles `lm_request`, not a model
    # listing message. Once the extension grows a `list_models`
    # message we plumb it here. For now, returning None forces the
    # fallback path, which is correct.
    return None


async def discover(
    provider: str,
    *,
    openai_api_key: str | None = None,
    ollama_host: str = "127.0.0.1",
    ollama_port: int = 11434,
    bridge_socket: str = "",
    bridge_port: int = 0,
) -> list[ModelChoice] | None:
    """Dispatch table — returns the discovered model list for the
    named provider, or ``None`` if discovery isn't supported or the
    provider's endpoint is unreachable. Callers fall back to the
    curated static list on ``None``."""
    if provider in ("openai", "codex"):
        return await discover_openai(openai_api_key)
    if provider == "ollama":
        return await discover_ollama(ollama_host, ollama_port)
    if provider == "vscode_extension":
        return await discover_vscode_extension(bridge_socket, bridge_port)
    # CLI providers (claude_cli, codex_cli, gemini_cli, copilot_cli)
    # don't expose a stable model-list flag — they ship a single
    # default and the user picks alternates via CLI args. Static
    # fallback is correct.
    return None
