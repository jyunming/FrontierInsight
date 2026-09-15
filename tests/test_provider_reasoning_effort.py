"""``provider.reasoning_effort``: config validation, the OpenAI-compatible
request body, the argv each CLI provider gets, and the one-time warning where
a provider or a level cannot be applied.

No real model or CLI is called: HTTP goes through ``httpx.MockTransport`` and
the CLI spawn is mocked, so these tests pin what FI *sends*.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml
from pydantic import ValidationError

from core import provider as provider_mod
from core.config import REASONING_EFFORT_LEVELS, Config, ProviderConfig
from core.provider import (
    _CLI_SPECS,
    LLMClient,
    ResolvedEndpoint,
    _run_cli,
    resolve_endpoint,
    resolve_endpoint_async,
)


@pytest.fixture(autouse=True)
def _fresh_warning_state():
    """The warn-once set is process-wide; each test starts from a clean one."""
    provider_mod._REASONING_EFFORT_WARNED.clear()
    yield
    provider_mod._REASONING_EFFORT_WARNED.clear()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_reasoning_effort_defaults_to_unset() -> None:
    assert ProviderConfig().reasoning_effort is None
    assert REASONING_EFFORT_LEVELS == ("minimal", "low", "medium", "high", "xhigh", "max")


@pytest.mark.parametrize("level", REASONING_EFFORT_LEVELS)
def test_every_level_is_accepted(level: str) -> None:
    assert ProviderConfig(reasoning_effort=level).reasoning_effort == level


def test_level_is_normalised_and_blank_means_unset() -> None:
    assert ProviderConfig(reasoning_effort="  High ").reasoning_effort == "high"
    assert ProviderConfig(reasoning_effort="").reasoning_effort is None
    assert ProviderConfig(reasoning_effort="   ").reasoning_effort is None


@pytest.mark.parametrize("bad", ["extreme", "none", "ultra", "hi", True, 3])
def test_an_unknown_level_is_rejected_with_the_list_of_levels(bad: object) -> None:
    with pytest.raises(ValidationError) as exc:
        ProviderConfig(reasoning_effort=bad)
    msg = str(exc.value)
    assert "provider.reasoning_effort must be one of minimal, low, medium, high, xhigh, max" in msg


def test_yaml_key_loads_and_a_typo_fails(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump({
        "topic": "t", "provider": {"name": "ollama", "reasoning_effort": "high"},
    }), encoding="utf-8")
    assert Config.from_yaml(good).provider.reasoning_effort == "high"

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "topic": "t", "provider": {"name": "ollama", "reasoning_effort": "hihg"},
    }), encoding="utf-8")
    with pytest.raises(ValidationError):
        Config.from_yaml(bad)


# ---------------------------------------------------------------------------
# Endpoint resolution carries the level to every transport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,extra", [
    ("ollama", {}), ("openai", {}), ("vllm", {}),
    ("codex_cli", {}), ("claude_cli", {}), ("antigravity_cli", {}),
    ("vscode_extension", {"bridge_port": 1}),
])
def test_resolve_endpoint_carries_the_level(name: str, extra: dict) -> None:
    assert resolve_endpoint(
        ProviderConfig(name=name, extra=extra, reasoning_effort="medium"),
    ).reasoning_effort == "medium"
    assert resolve_endpoint(ProviderConfig(name=name, extra=extra)).reasoning_effort == ""


@pytest.mark.asyncio
async def test_proxy_endpoint_carries_the_level() -> None:
    sup = MagicMock()
    sup.acquire = AsyncMock(return_value=MagicMock(port=4321))
    ep = await resolve_endpoint_async(
        ProviderConfig(name="github_copilot_cli", reasoning_effort="high"), sup,
    )
    assert ep.reasoning_effort == "high"


# ---------------------------------------------------------------------------
# HTTP transport: the request body
# ---------------------------------------------------------------------------


async def _sent_body(ep: ResolvedEndpoint, **chat_kwargs) -> dict:  # noqa: ANN003
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        })

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(ep, http=http)
    try:
        assert await client.chat([{"role": "user", "content": "hi"}], **chat_kwargs) == "ok"
    finally:
        await client.aclose()
        await http.aclose()
    return seen["body"]


@pytest.mark.asyncio
async def test_ollama_high_is_sent_in_the_body() -> None:
    ep = resolve_endpoint(ProviderConfig(name="ollama", model="gemma4:31b-cloud", reasoning_effort="high"))
    body = await _sent_body(ep)
    assert body["reasoning_effort"] == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["ollama", "openai", "vllm"])
async def test_unset_sends_no_reasoning_field(name: str) -> None:
    body = await _sent_body(resolve_endpoint(ProviderConfig(name=name)))
    assert "reasoning_effort" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("level", REASONING_EFFORT_LEVELS)
async def test_openai_gets_every_level_as_is(level: str) -> None:
    body = await _sent_body(resolve_endpoint(ProviderConfig(name="openai", reasoning_effort=level)))
    assert body["reasoning_effort"] == level


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["minimal", "xhigh", "max"])
async def test_ollama_skips_a_level_its_server_rejects_and_warns_once(level: str) -> None:
    ep = resolve_endpoint(ProviderConfig(name="ollama", reasoning_effort=level))
    with patch.object(provider_mod, "_log") as log:
        first = await _sent_body(ep)
        second = await _sent_body(ep)
    assert "reasoning_effort" not in first and "reasoning_effort" not in second
    assert log.warning.call_count == 1
    args = log.warning.call_args[0]
    assert args[1] == level and args[2] == "ollama"
    assert "low, medium, high" in args[3]


@pytest.mark.asyncio
async def test_proxy_provider_does_not_send_it_and_warns_once() -> None:
    ep = ResolvedEndpoint(
        base_url="http://127.0.0.1:4321/v1", model="m", api_key="x",
        provider_name="github_copilot_cli", reasoning_effort="high",
    )
    with patch.object(provider_mod, "_log") as log:
        body = await _sent_body(ep)
        await _sent_body(ep)
    assert "reasoning_effort" not in body
    assert log.warning.call_count == 1
    assert log.warning.call_args[0][2] == "github_copilot_cli"


@pytest.mark.asyncio
async def test_a_per_call_extra_still_wins() -> None:
    ep = resolve_endpoint(ProviderConfig(name="ollama", reasoning_effort="high"))
    body = await _sent_body(ep, extra={"reasoning_effort": "low"})
    assert body["reasoning_effort"] == "low"


# ---------------------------------------------------------------------------
# CLI transports: the argv
# ---------------------------------------------------------------------------


async def _argv(provider: str, level: str) -> list[str]:
    spec = _CLI_SPECS[provider]
    spawn = AsyncMock(return_value=MagicMock())
    with patch("core.provider.shutil.which", return_value=f"/bin/{spec.argv[0]}"), \
         patch("core.provider.asyncio.create_subprocess_exec", new=spawn), \
         patch("core.provider._collect_via_communicate", new=AsyncMock(return_value="ok")), \
         patch("core.provider._collect_via_streaming", new=AsyncMock(return_value="ok")):
        assert await _run_cli(spec, "the prompt", reasoning_effort=level) == "ok"
    return list(spawn.call_args[0])


def _has_pair(argv: list[str], flag: str, value: str) -> bool:
    return any(argv[i] == flag and argv[i + 1] == value for i in range(len(argv) - 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("level", REASONING_EFFORT_LEVELS)
async def test_codex_cli_gets_the_config_override(level: str) -> None:
    argv = await _argv("codex_cli", level)
    assert _has_pair(argv, "-c", f'model_reasoning_effort="{level}"')


@pytest.mark.asyncio
async def test_codex_cli_unset_adds_no_override() -> None:
    argv = await _argv("codex_cli", "")
    assert "-c" not in argv
    assert not any("reasoning_effort" in a for a in argv)


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["low", "medium", "high"])
async def test_antigravity_cli_gets_effort_flag(level: str) -> None:
    argv = await _argv("antigravity_cli", level)
    assert _has_pair(argv, "--effort", level)


@pytest.mark.asyncio
async def test_antigravity_cli_unset_adds_no_flag() -> None:
    assert "--effort" not in await _argv("antigravity_cli", "")


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["minimal", "xhigh", "max"])
async def test_antigravity_cli_skips_a_level_agy_does_not_accept(level: str) -> None:
    with patch.object(provider_mod, "_log") as log:
        first = await _argv("antigravity_cli", level)
        second = await _argv("antigravity_cli", level)
    assert "--effort" not in first and "--effort" not in second
    assert first == await _argv("antigravity_cli", "")
    assert log.warning.call_count == 1
    args = log.warning.call_args[0]
    assert args[1] == level and args[2] == "antigravity_cli"
    assert "low, medium, high" in args[3]


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
async def test_claude_cli_gets_effort_flag(level: str) -> None:
    assert _has_pair(await _argv("claude_cli", level), "--effort", level)


@pytest.mark.asyncio
async def test_claude_cli_skips_minimal_and_unset_adds_nothing() -> None:
    with patch.object(provider_mod, "_log") as log:
        argv = await _argv("claude_cli", "minimal")
    assert "--effort" not in argv
    assert log.warning.call_count == 1
    assert "--effort" not in await _argv("claude_cli", "")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["copilot_cli", "gemini_cli"])
async def test_cli_without_a_setting_passes_nothing_and_warns_once(provider: str) -> None:
    baseline = await _argv(provider, "")
    with patch.object(provider_mod, "_log") as log:
        first = await _argv(provider, "high")
        second = await _argv(provider, "high")
    assert first == baseline and second == baseline
    assert log.warning.call_count == 1
    assert log.warning.call_args[0][2] == provider
    if _CLI_SPECS[provider].pass_prompt_via == "arg":
        assert first[-1] == "the prompt"


@pytest.mark.asyncio
async def test_llm_client_threads_the_level_from_config_to_the_cli() -> None:
    ep = resolve_endpoint(ProviderConfig(name="codex_cli", reasoning_effort="medium"))
    client = LLMClient(ep)
    run = AsyncMock(return_value="ok")
    try:
        with patch("core.provider._run_cli", new=run):
            assert await client.chat([{"role": "user", "content": "hi"}]) == "ok"
    finally:
        await client.aclose()
    assert run.await_args.kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_vscode_extension_warns_once_and_sends_nothing() -> None:
    ep = resolve_endpoint(ProviderConfig(
        name="vscode_extension", extra={"bridge_port": 1}, reasoning_effort="high",
    ))
    client = LLMClient(ep)
    bridge = AsyncMock(return_value="ok")
    try:
        with patch.object(provider_mod, "_log") as log, \
             patch.object(LLMClient, "_chat_vscode_bridge", new=bridge):
            await client.chat([{"role": "user", "content": "hi"}])
            await client.chat([{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()
    assert log.warning.call_count == 1
    assert log.warning.call_args[0][2] == "vscode_extension"
    assert "reasoning_effort" not in bridge.await_args.kwargs
