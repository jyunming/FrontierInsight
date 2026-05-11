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


@pytest.fixture(autouse=True)
def _stub_shutil_which():
    """`_run_cli` now resolves argv[0] via `shutil.which()` before spawning,
    so on a CI host where claude/codex/copilot aren't installed the tests
    would all error out at "binary not on PATH" before reaching their real
    assertions. Pin `shutil.which` to a fake-but-truthy path by default;
    tests that need to exercise the missing-binary branch override this
    via their own `patch` context."""
    with patch("core.provider.shutil.which", return_value="/fake/path/bin") as p:
        yield p


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


def test_cli_provider_set_matches_known_providers() -> None:
    assert _CLI_PROVIDERS == {"codex_cli", "claude_cli", "copilot_cli", "gemini_cli"}


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
        # Pin shutil.which so the assertion can compare to a known sentinel
        # instead of whatever the test host's PATH happens to resolve to.
        with patch("core.provider.shutil.which", return_value="/usr/bin/claude"), \
             patch(
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
    # argv[0] is the shutil.which-resolved path, not the bare name.
    assert args[0] == "/usr/bin/claude"
    assert "--print" in args and "--output-format" in args and "text" in args
    assert all(a != "what is 6 * 7?" for a in args), "prompt should NOT be in argv"
    proc.communicate.assert_awaited_once()
    stdin_arg = proc.communicate.await_args[0][0]
    assert stdin_arg == b"what is 6 * 7?"


@pytest.mark.asyncio
async def test_codex_cli_passes_prompt_via_stdin_and_reads_last_message_file(
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
        # On Windows, NamedTemporaryFile can't always be reopened by name
        # while the original handle is open — write through the original
        # handle and flush rather than `Path(...).write_text(...)`.
        tf.write(b"RESULT: nine")
        tf.flush()
        return tf

    monkeypatch.setattr("core.provider.tempfile.NamedTemporaryFile", stub_named_tempfile)

    ep = resolve_endpoint(ProviderConfig(name="codex_cli"))
    client = LLMClient(ep)
    try:
        proc = _fake_proc(stdout=b"tokens used\n9,000\n")
        with patch("core.provider.shutil.which", return_value="/usr/bin/codex"), \
             patch(
                 "core.provider.asyncio.create_subprocess_exec",
                 new=AsyncMock(return_value=proc),
             ) as spawn:
            result = await client.chat([{"role": "user", "content": "What is 3*3?"}])
    finally:
        await client.aclose()

    assert result == "RESULT: nine"
    # argv = [<resolved-codex-path>, "exec", "--output-last-message", <tmpfile>]
    # — prompt is NOT in argv (security: avoid leaking via local process listings).
    args = spawn.call_args[0]
    assert args[0] == "/usr/bin/codex" and args[1] == "exec"
    assert args[2] == "--output-last-message"
    assert args[3] == seen_paths[0]
    assert all(a != "What is 3*3?" for a in args), "prompt must NOT be in argv"
    # Prompt was passed on stdin.
    proc.communicate.assert_awaited_once()
    stdin_arg = proc.communicate.await_args[0][0]
    assert stdin_arg == b"What is 3*3?"
    # The tmpfile is cleaned up after reading.
    assert not Path(seen_paths[0]).exists()


@pytest.mark.asyncio
async def test_codex_cli_sends_stdout_to_devnull_when_using_last_message_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stdout for codex_cli is just the agent log — we should not pipe it
    into memory. Verify the spawn was called with stdout=DEVNULL."""
    import asyncio as _asyncio

    real_NamedTemporaryFile = __import__("tempfile").NamedTemporaryFile

    def stub_named_tempfile(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        tf = real_NamedTemporaryFile(*args, **kwargs)
        tf.write(b"ok")
        tf.flush()
        return tf

    monkeypatch.setattr("core.provider.tempfile.NamedTemporaryFile", stub_named_tempfile)

    ep = resolve_endpoint(ProviderConfig(name="codex_cli"))
    client = LLMClient(ep)
    try:
        proc = _fake_proc()
        with patch(
            "core.provider.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ) as spawn:
            await client.chat([{"role": "user", "content": "x"}])
    finally:
        await client.aclose()

    kwargs = spawn.call_args[1]
    assert kwargs["stdout"] == _asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_cli_tmpfile_cleaned_up_on_spawn_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `create_subprocess_exec` raises FileNotFoundError for a CLI
    using `last_message_file` mode, the temp file must still be removed
    so we don't accumulate `fi_cli_out_*.txt` litter over time."""
    real_NamedTemporaryFile = __import__("tempfile").NamedTemporaryFile
    seen: list[str] = []

    def stub_named_tempfile(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        tf = real_NamedTemporaryFile(*args, **kwargs)
        seen.append(tf.name)
        return tf

    monkeypatch.setattr("core.provider.tempfile.NamedTemporaryFile", stub_named_tempfile)

    ep = resolve_endpoint(ProviderConfig(name="codex_cli"))
    client = LLMClient(ep)
    try:
        with patch(
            "core.provider.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=FileNotFoundError("codex not on PATH")),
        ):
            with pytest.raises(RuntimeError, match="not found on PATH"):
                await client.chat([{"role": "user", "content": "x"}])
    finally:
        await client.aclose()

    assert seen, "the codex_cli path must allocate a tmp file"
    assert not Path(seen[0]).exists(), "tmp file leaked after spawn failure"


@pytest.mark.asyncio
async def test_copilot_cli_passes_prompt_via_arg_and_returns_stdout() -> None:
    """`copilot_cli` argv contract: `copilot -s --allow-all-tools -p <prompt>`,
    stdout returns the agent response (stats stripped by --silent)."""
    ep = resolve_endpoint(ProviderConfig(name="copilot_cli"))
    client = LLMClient(ep)
    try:
        proc = _fake_proc(stdout=b"42\n")
        with patch("core.provider.shutil.which", return_value="/usr/bin/copilot"), \
             patch(
                 "core.provider.asyncio.create_subprocess_exec",
                 new=AsyncMock(return_value=proc),
             ) as spawn:
            result = await client.chat([{"role": "user", "content": "what is 6 * 7?"}])
    finally:
        await client.aclose()

    assert result == "42"
    args = spawn.call_args[0]
    # argv = [<resolved-copilot-path>, "-s", "--allow-all-tools", "-p", "<prompt>"]
    assert args[0] == "/usr/bin/copilot"
    assert "-s" in args and "--allow-all-tools" in args and "-p" in args
    assert args[-1] == "what is 6 * 7?", "prompt must be the last argv element after -p"
    # No stdin for copilot_cli — communicate() is called with no input.
    proc.communicate.assert_awaited_once()
    assert proc.communicate.await_args[0] == () or proc.communicate.await_args[0][0] is None


@pytest.mark.asyncio
async def test_gemini_cli_pipes_prompt_on_stdin_and_extracts_json_response() -> None:
    """`gemini_cli` passes prompt on stdin (no argv leakage) and parses
    the `response` field out of the `-o json` envelope, stripping the
    leading CLI warnings."""
    fake_stdout = (
        b"Warning: True color not detected.\n"
        b"YOLO mode is enabled.\n"
        b"Ripgrep is not available. Falling back to GrepTool.\n"
        b"MCP issues detected.\n"
        b'{\n  "session_id": "abc",\n  "response": "the real answer",\n'
        b'  "stats": {"tokens": {"total": 42}}\n}\n'
    )
    ep = resolve_endpoint(ProviderConfig(name="gemini_cli"))
    client = LLMClient(ep)
    try:
        proc = _fake_proc(stdout=fake_stdout)
        with patch("core.provider.shutil.which", return_value="/usr/bin/gemini"), \
             patch(
                 "core.provider.asyncio.create_subprocess_exec",
                 new=AsyncMock(return_value=proc),
             ) as spawn:
            result = await client.chat([{"role": "user", "content": "say hi"}])
    finally:
        await client.aclose()

    assert result == "the real answer"
    args = spawn.call_args[0]
    assert args[0] == "/usr/bin/gemini"
    assert "--yolo" in args and "-o" in args and "json" in args
    # Prompt is on stdin (`-p ""` triggers non-interactive mode that reads stdin).
    assert "say hi" not in args
    proc.communicate.assert_awaited_once()
    assert proc.communicate.await_args[0][0] == b"say hi"


def test_gemini_response_extractor_falls_back_on_unparseable_envelope() -> None:
    """If the JSON envelope can't be parsed, the extractor returns the
    raw text rather than raising — keeps the engine's downstream
    `_parse_json_lenient` available as the last line of defense."""
    from core.provider import _extract_gemini_response

    assert _extract_gemini_response("just plain text, no braces") == \
        "just plain text, no braces"
    assert _extract_gemini_response("warnings...\n{not valid json}\n") == \
        "warnings...\n{not valid json}\n"
    assert _extract_gemini_response(
        '{"session_id": "abc", "stats": {}}'
    ) == '{"session_id": "abc", "stats": {}}'


def test_gemini_response_extractor_handles_trailing_output_after_envelope() -> None:
    """`gemini -o json` sometimes emits a closing log line after the JSON
    envelope. A naive `find('{')` + `rfind('}')` slice would include that
    trailing chunk, breaking `json.loads`. The incremental decoder used
    by `_extract_gemini_response` scans `raw_decode` from each `{` and
    succeeds on the first valid object, ignoring whatever follows."""
    from core.provider import _extract_gemini_response

    raw = (
        'Warning: True color not detected.\n'
        '{"session_id": "abc", "response": "the real answer", "stats": {}}\n'
        'Done in 1.3s.\n'
    )
    assert _extract_gemini_response(raw) == "the real answer"


def test_gemini_response_extractor_skips_unrelated_brace_before_envelope() -> None:
    """A CLI warning line containing `{...}` (e.g. a JSON-formatted error
    message printed before the real envelope) must NOT shift the start
    position past the real envelope's opening brace. The incremental
    decoder retries from each `{` until one yields a valid JSON object
    with the `response` field."""
    from core.provider import _extract_gemini_response

    raw = (
        'Warning: invalid config {something=1}\n'
        'MCP issues: {server="x"}\n'
        '{"session_id": "abc", "response": "the real answer"}\n'
    )
    assert _extract_gemini_response(raw) == "the real answer"


@pytest.mark.asyncio
async def test_cli_missing_binary_raises_clear_runtime_error() -> None:
    """When `shutil.which` returns None (binary not on PATH), `_run_cli`
    raises a clean RuntimeError BEFORE attempting to spawn — this is the
    up-front check that fixed the Windows PATHEXT mis-diagnosis."""
    ep = resolve_endpoint(ProviderConfig(name="claude_cli"))
    client = LLMClient(ep)
    try:
        # Override the autouse stub.
        with patch("core.provider.shutil.which", return_value=None):
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
