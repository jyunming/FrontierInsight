"""Tests for the `codex_cli` / `claude_cli` CLI-exec providers.

These exercise the spawn argv, prompt-passing transport (stdin vs argv),
output-collection mode (stdout vs --output-last-message file), retry on
transient non-zero exit, and clean error when the binary is missing.

All tests mock `asyncio.create_subprocess_exec` so no real CLI is invoked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.config import ProviderConfig
from core.provider import (
    LLMClient,
    ProxySupervisor,
    _CLI_PROVIDERS,
    _CLI_SPECS,
    _messages_to_text,
    resolve_endpoint,
    resolve_endpoint_async,
)


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def test_cli_provider_set_matches_known_providers() -> None:
    assert _CLI_PROVIDERS == {"codex_cli", "claude_cli"}


def test_cli_specs_have_required_fields() -> None:
    for name, spec in _CLI_SPECS.items():
        assert spec.argv, f"{name}: empty argv"
        assert spec.pass_prompt_via in ("stdin", "arg"), name
        assert spec.output_via in ("stdout", "last_message_file"), name


def test_resolve_endpoint_claude_cli_sets_transport_to_cli() -> None:
    ep = resolve_endpoint(ProviderConfig(name="claude_cli"))
    assert ep.transport == "cli"
    assert ep.cli_spec is _CLI_SPECS["claude_cli"]
    assert ep.base_url == ""
    assert ep.api_key == "not-needed"


def test_resolve_endpoint_codex_cli_sets_transport_to_cli() -> None:
    ep = resolve_endpoint(ProviderConfig(name="codex_cli"))
    assert ep.transport == "cli"
    assert ep.cli_spec is _CLI_SPECS["codex_cli"]


@pytest.mark.asyncio
async def test_resolve_endpoint_async_cli_is_pass_through() -> None:
    """CLI providers don't need the supervisor — they short-circuit through
    the same synchronous path as direct providers."""
    sup = ProxySupervisor()
    ep = await resolve_endpoint_async(ProviderConfig(name="claude_cli"), sup)
    assert ep.transport == "cli"
    # Supervisor's handle table must remain empty (no spawn happened).
    assert sup._handles == {}


# ---------------------------------------------------------------------------
# _messages_to_text — flattening contract
# ---------------------------------------------------------------------------


def test_messages_to_text_single_user_message() -> None:
    assert _messages_to_text([{"role": "user", "content": "hello"}]) == "hello"


def test_messages_to_text_no_role_treated_as_user() -> None:
    assert _messages_to_text([{"content": "bare"}]) == "bare"


def test_messages_to_text_system_prefix() -> None:
    out = _messages_to_text([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ])
    assert "[system]\nS" in out
    assert "U" in out
    assert out.index("[system]") < out.index("U")


def test_messages_to_text_assistant_prior_turn() -> None:
    out = _messages_to_text([
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ])
    assert "Q1" in out and "[assistant prior turn]\nA1" in out and "Q2" in out


def test_messages_to_text_drops_empty_content_parts() -> None:
    out = _messages_to_text([
        {"role": "user", "content": ""},
        {"role": "user", "content": "real"},
    ])
    assert out == "real"


# ---------------------------------------------------------------------------
# LLMClient._chat_cli — spawn + output collection
# ---------------------------------------------------------------------------


def _fake_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_claude_cli_passes_prompt_via_stdin_and_returns_stdout() -> None:
    ep = resolve_endpoint(ProviderConfig(name="claude_cli"))
    client = LLMClient(ep)
    try:
        proc = _fake_proc(stdout=b"42\n")
        with patch(
            "core.provider.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ) as spawn:
            result = await client.chat(
                [{"role": "user", "content": "what is 6 * 7?"}]
            )
    finally:
        await client.aclose()

    assert result == "42"
    spawn.assert_awaited_once()
    args, kwargs = spawn.call_args
    # claude CLI argv: no prompt appended (stdin transport).
    assert args[0] == "claude"
    assert "--print" in args and "--output-format" in args and "text" in args
    assert all(a != "what is 6 * 7?" for a in args), "prompt should NOT be in argv"
    proc.communicate.assert_awaited_once()
    stdin_arg = proc.communicate.await_args[0][0]
    assert stdin_arg == b"what is 6 * 7?"


@pytest.mark.asyncio
async def test_codex_cli_passes_prompt_as_arg_and_reads_last_message_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force tempfile to land inside the test tmp_path so we can assert on
    # the path passed to --output-last-message.
    seen_paths: list[str] = []
    real_NamedTemporaryFile = __import__("tempfile").NamedTemporaryFile

    def stub_named_tempfile(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        tf = real_NamedTemporaryFile(*args, **kwargs)
        seen_paths.append(tf.name)
        # Pre-fill what the CLI would write so _run_cli can read it back.
        Path(tf.name).write_text("RESULT: nine", encoding="utf-8")
        return tf

    monkeypatch.setattr("core.provider.tempfile.NamedTemporaryFile", stub_named_tempfile)

    ep = resolve_endpoint(ProviderConfig(name="codex_cli"))
    client = LLMClient(ep)
    try:
        proc = _fake_proc(stdout=b"tokens used\n9,000\n")
        with patch(
            "core.provider.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ) as spawn:
            result = await client.chat([{"role": "user", "content": "What is 3*3?"}])
    finally:
        await client.aclose()

    assert result == "RESULT: nine"
    # argv = ["codex", "exec", "--output-last-message", <tmpfile>, "<prompt>"].
    args = spawn.call_args[0]
    assert args[0] == "codex" and args[1] == "exec"
    assert args[2] == "--output-last-message"
    assert args[3] == seen_paths[0]
    assert args[4] == "What is 3*3?"
    # The tmpfile is cleaned up after reading.
    assert not Path(seen_paths[0]).exists()


@pytest.mark.asyncio
async def test_cli_missing_binary_raises_clear_runtime_error() -> None:
    ep = resolve_endpoint(ProviderConfig(name="claude_cli"))
    client = LLMClient(ep)
    try:
        with patch(
            "core.provider.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=FileNotFoundError("claude not on PATH")),
        ):
            with pytest.raises(RuntimeError, match="not found on PATH"):
                await client.chat([{"role": "user", "content": "x"}])
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_cli_nonzero_exit_retries_then_raises() -> None:
    """A persistently non-zero CLI exit must surface after retries are
    exhausted — wrapped as `_CliTransientError` and reraised by tenacity."""
    from core.provider import _CliTransientError

    ep = resolve_endpoint(ProviderConfig(name="claude_cli"))
    client = LLMClient(ep)
    try:
        proc = _fake_proc(stderr=b"auth required", returncode=1)
        spawner = AsyncMock(return_value=proc)
        with patch(
            "core.provider.asyncio.create_subprocess_exec",
            new=spawner,
        ):
            # Patch tenacity wait so the test isn't slow.
            with patch("core.provider.wait_exponential", return_value=lambda *a, **kw: 0):
                with pytest.raises(_CliTransientError, match="auth required"):
                    await client.chat([{"role": "user", "content": "x"}])
        # 4 retries per the stop_after_attempt(4) policy.
        assert spawner.await_count == 4
    finally:
        await client.aclose()
