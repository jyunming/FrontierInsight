"""Tests for the `codex_cli` / `claude_cli` CLI-exec providers.

These exercise the spawn argv, prompt-passing transport (stdin vs argv),
output-collection mode (stdout vs --output-last-message file), retry on
transient non-zero exit, and clean error when the binary is missing.

All tests mock `asyncio.create_subprocess_exec` so no real CLI is invoked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import ProviderConfig
from core.provider import (
    LLMClient,
    ProxySupervisor,
    _CLI_PROVIDERS,
    _CLI_SPECS,
    _messages_to_text,
    _truncate_prompt_to_fit,
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
        assert spec.output_via in ("stdout", "last_message_file", "stream_json"), name


def test_cli_input_caps() -> None:
    """codex_cli rejects >1 MB turns; copilot_cli passes the prompt as a CLI
    arg and hits the Windows cmd.exe ~8191-char command-line limit — both
    carry a max_input_chars cap. The stdin-based CLIs have no such limit."""
    assert _CLI_SPECS["codex_cli"].max_input_chars == 900_000
    assert _CLI_SPECS["codex_cli"].max_input_chars < 1_048_576  # under codex's hard limit
    # copilot: capped under the ~8191 Windows command-line limit.
    assert _CLI_SPECS["copilot_cli"].max_input_chars == 7000
    assert _CLI_SPECS["copilot_cli"].max_input_chars < 8191
    assert _CLI_SPECS["copilot_cli"].pass_prompt_via == "arg"  # why it needs the cap
    # stdin-based CLIs: no arg-length limit.
    for name in ("claude_cli", "gemini_cli"):
        assert _CLI_SPECS[name].max_input_chars is None, name


def test_truncate_prompt_to_fit_keeps_head_and_tail() -> None:
    """A prompt over the cap is trimmed to fit, dropping the MIDDLE while
    preserving the head (task) and tail (output-format instructions)."""
    head = "TASK: do the thing.\n"
    tail = "\nOUTPUT FORMAT: return JSON only."
    middle = "X" * 5000
    prompt = head + middle + tail
    out = _truncate_prompt_to_fit(prompt, 1200)
    assert len(out) <= 1200
    assert out.startswith("TASK: do the thing.")
    assert out.endswith("OUTPUT FORMAT: return JSON only.")
    assert "trimmed here to fit" in out
    # The bulky middle is mostly gone.
    assert out.count("X") < 5000


def test_truncate_prompt_to_fit_passes_through_under_cap() -> None:
    prompt = "short prompt"
    assert _truncate_prompt_to_fit(prompt, 1000) == prompt


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
    """Build a fake ``Process`` that satisfies BOTH legacy and
    streaming code paths inside ``_run_cli``.

    The legacy path (``output_via="stdout"`` / ``"last_message_file"``)
    consumes ``proc.communicate(stdin)`` once. The streaming path
    (``output_via="stream_json"`` — claude_cli) consumes
    ``proc.stdout.readline()`` until EOF, ``proc.stdin.write/drain/
    close``, ``proc.wait()``, ``proc.stderr.read()``.

    Many tests only assert on argv, not output. For those, the
    streaming surfaces below produce immediate EOF so the call
    completes with an empty answer — fine for argv assertions.
    ``stdout=b"..."`` tests that DO inspect the returned text still
    work via the legacy path (those tests target gemini_cli /
    copilot_cli / codex_cli, not claude_cli)."""
    proc = AsyncMock()
    # Legacy path:
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    # Streaming path:
    stdout_obj = AsyncMock()
    # If a stdout payload was provided, emit it as ONE line then EOF
    # so streaming tests can still read it (rare path; argv tests on
    # claude_cli supply b"" so EOF is immediate).
    lines = [stdout] if stdout else []
    lines_iter = iter(lines + [b""])
    async def fake_readline():
        return next(lines_iter, b"")
    stdout_obj.readline = fake_readline
    proc.stdout = stdout_obj
    proc.stdin = AsyncMock()
    proc.stdin.write = lambda b: None
    proc.stdin.drain = AsyncMock(return_value=None)
    proc.stdin.close = lambda: None
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=stderr)
    proc.wait = AsyncMock(return_value=returncode)
    proc.kill = lambda: None
    return proc


def _fake_streaming_proc(stream_lines: list[bytes], returncode: int = 0):
    """Build a fake ``Process`` for the stream-json claude_cli path.

    ``proc.stdout.readline()`` returns each line in ``stream_lines`` in
    order, then ``b""`` for EOF. ``proc.stdin.write`` / ``.drain`` /
    ``.close`` are no-ops. ``proc.wait()`` returns ``returncode``.
    Mirrors what ``_collect_via_streaming`` consumes."""
    proc = AsyncMock()
    stdout = AsyncMock()
    lines_iter = iter(list(stream_lines) + [b""])
    async def fake_readline():
        return next(lines_iter, b"")
    stdout.readline = fake_readline
    proc.stdout = stdout
    proc.stdin = AsyncMock()
    proc.stdin.write = lambda b: None
    proc.stdin.drain = AsyncMock(return_value=None)
    proc.stdin.close = lambda: None
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.wait = AsyncMock(return_value=returncode)
    proc.returncode = returncode
    proc.kill = lambda: None
    return proc


@pytest.mark.asyncio
async def test_claude_cli_passes_prompt_via_stdin_and_returns_stream_json_text() -> None:
    """claude_cli now uses ``--output-format stream-json
    --include-partial-messages``. The aggregator concatenates
    ``text_delta`` events into the answer; ``thinking_delta`` events
    count as activity but their bodies are discarded. Mock a typical
    Sonnet response (two text deltas + EOF) and assert the result."""
    ep = resolve_endpoint(ProviderConfig(name="claude_cli"))
    client = LLMClient(ep)
    try:
        stream = [
            (b'{"type":"stream_event","event":{"type":"content_block_delta",'
             b'"delta":{"type":"thinking_delta","thinking":"computing..."}}}\n'),
            (b'{"type":"stream_event","event":{"type":"content_block_delta",'
             b'"delta":{"type":"text_delta","text":"4"}}}\n'),
            (b'{"type":"stream_event","event":{"type":"content_block_delta",'
             b'"delta":{"type":"text_delta","text":"2"}}}\n'),
        ]
        proc = _fake_streaming_proc(stream)
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
    assert args[0] == "/usr/bin/claude"
    # New argv: stream-json, not plain text.
    assert "--print" in args
    assert "--output-format" in args and "stream-json" in args
    assert "--include-partial-messages" in args
    assert all(a != "what is 6 * 7?" for a in args), "prompt should NOT be in argv"


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
async def test_cli_model_flag_injected_when_provider_model_set() -> None:
    """When the YAML config sets `provider.model`, `_run_cli` inserts
    `[model_flag, model]` right after argv[0]. Empty model passes the
    CLI's own default through (covered by the explicit-default test)."""
    ep = resolve_endpoint(ProviderConfig(name="claude_cli", model="opus"))
    client = LLMClient(ep)
    try:
        proc = _fake_proc(stdout=b"ok")
        with patch("core.provider.shutil.which", return_value="/usr/bin/claude"), \
             patch(
                 "core.provider.asyncio.create_subprocess_exec",
                 new=AsyncMock(return_value=proc),
             ) as spawn:
            await client.chat([{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()

    args = spawn.call_args[0]
    # ["/usr/bin/claude", "--model", "opus", "--print", "--output-format", "text"]
    assert args[0] == "/usr/bin/claude"
    assert args[1] == "--model"
    assert args[2] == "opus"
    assert "--print" in args[3:]


@pytest.mark.asyncio
async def test_cli_model_flag_omitted_when_provider_model_blank() -> None:
    """No `provider.model` => the CLI's own default is preserved (the
    user can set e.g. `~/.codex/config.toml` or claude `/model` and have
    it honored)."""
    ep = resolve_endpoint(ProviderConfig(name="claude_cli"))  # no model
    client = LLMClient(ep)
    try:
        proc = _fake_proc(stdout=b"ok")
        with patch("core.provider.shutil.which", return_value="/usr/bin/claude"), \
             patch(
                 "core.provider.asyncio.create_subprocess_exec",
                 new=AsyncMock(return_value=proc),
             ) as spawn:
            await client.chat([{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()

    args = spawn.call_args[0]
    # No --model flag injected.
    assert "--model" not in args
    assert args[1] == "--print"


@pytest.mark.asyncio
async def test_cli_call_killed_after_timeout_raises_transient() -> None:
    """A CLI that hangs longer than `cli_timeout_s` is killed and the
    raised `_CliTransientError` is retryable (so tenacity will try
    again up to 4 attempts before giving up).

    Uses ``codex_cli`` (legacy ``communicate()`` path) for the hang
    simulation. claude_cli's stream-json path has its own dedicated
    timeout tests in ``tests/test_provider_streaming.py``."""
    from core.provider import _CliTransientError

    ep = resolve_endpoint(ProviderConfig(name="codex_cli"))
    # Very short cli_timeout_s so the test doesn't actually wait.
    client = LLMClient(ep, cli_timeout_s=0.05)
    try:
        # communicate() that never returns within the timeout window.
        proc = AsyncMock()
        async def hang_forever(_=None):
            await asyncio.sleep(10)
        proc.communicate = hang_forever
        # Process.kill() is sync; Process.wait() is async.
        proc.kill = MagicMock()
        async def _wait_done():
            return 0
        proc.wait = _wait_done
        proc.returncode = -1
        with patch("core.provider.shutil.which", return_value="/usr/bin/codex"), \
             patch(
                 "core.provider.asyncio.create_subprocess_exec",
                 new=AsyncMock(return_value=proc),
             ), \
             patch("core.provider.wait_random_exponential", return_value=lambda *a, **kw: 0):
            with pytest.raises(_CliTransientError, match="exceeded.*wall-clock"):
                await client.chat([{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()


def test_cli_resolve_endpoint_preserves_display_model_when_unset() -> None:
    """When the user does not pin `provider.model`, the resolved
    endpoint still carries a human-readable display string so Engine's
    startup log `provider claude_cli -> (...)` is not blank. The
    invisible `cli_model_override` field stays empty so `_run_cli` does
    NOT inject a model flag (= CLI default is honored)."""
    ep = resolve_endpoint(ProviderConfig(name="claude_cli"))
    assert ep.model == "claude_cli (CLI default)"
    assert ep.cli_model_override == ""

    ep2 = resolve_endpoint(ProviderConfig(name="codex_cli", model="gpt-5"))
    assert ep2.model == "gpt-5"
    assert ep2.cli_model_override == "gpt-5"


@pytest.mark.asyncio
async def test_cli_timeout_subsecond_value_keeps_precision_in_error() -> None:
    """When `cli_timeout_s` is below 1 second (as test timeouts often
    are), the raised `_CliTransientError` message must NOT round to
    `0s`. Sub-second values format as milliseconds with `:g` precision.

    Targets the legacy ``communicate()`` timeout-formatter path via
    ``codex_cli`` since claude_cli now uses stream-json which has a
    different formatter inside ``_collect_via_streaming``."""
    from core.provider import _CliTransientError

    ep = resolve_endpoint(ProviderConfig(name="codex_cli"))
    client = LLMClient(ep, cli_timeout_s=0.05)
    try:
        proc = AsyncMock()
        async def hang_forever(_=None):
            await asyncio.sleep(10)
        proc.communicate = hang_forever
        proc.kill = MagicMock()
        async def _wait_done():
            return 0
        proc.wait = _wait_done
        proc.returncode = -1
        with patch("core.provider.shutil.which", return_value="/usr/bin/codex"), \
             patch(
                 "core.provider.asyncio.create_subprocess_exec",
                 new=AsyncMock(return_value=proc),
             ), \
             patch("core.provider.wait_random_exponential", return_value=lambda *a, **kw: 0):
            with pytest.raises(_CliTransientError) as ei:
                await client.chat([{"role": "user", "content": "hi"}])
        msg = str(ei.value)
        assert "50ms" in msg  # 0.05s = 50ms with :g precision
        assert "0s" not in msg.replace("50ms", "")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_cli_timeout_handles_already_exited_child() -> None:
    """If the child exits between the communicate() timeout firing and
    our `proc.kill()`, Python raises `ProcessLookupError`. The cleanup
    must swallow it (not propagate) so the user still sees the original
    _CliTransientError describing the wall-clock cause.

    Uses ``codex_cli`` (legacy ``communicate()`` path) since the race
    being tested is specific to that code path."""
    from core.provider import _CliTransientError

    ep = resolve_endpoint(ProviderConfig(name="codex_cli"))
    client = LLMClient(ep, cli_timeout_s=0.05)
    try:
        proc = AsyncMock()
        async def hang_forever(_=None):
            await asyncio.sleep(10)
        proc.communicate = hang_forever
        # Simulate the race: by the time we kill, child is already gone.
        proc.kill = MagicMock(side_effect=ProcessLookupError("[WinError 87]"))
        async def _wait_done():
            return -1
        proc.wait = _wait_done
        proc.returncode = -1
        with patch("core.provider.shutil.which", return_value="/usr/bin/codex"), \
             patch(
                 "core.provider.asyncio.create_subprocess_exec",
                 new=AsyncMock(return_value=proc),
             ), \
             patch("core.provider.wait_random_exponential", return_value=lambda *a, **kw: 0):
            with pytest.raises(_CliTransientError, match="exceeded.*wall-clock"):
                await client.chat([{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()


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
            with patch("core.provider.wait_random_exponential", return_value=lambda *a, **kw: 0):
                with pytest.raises(_CliTransientError, match="auth required"):
                    await client.chat([{"role": "user", "content": "x"}])
        # 4 retries per the stop_after_attempt(4) policy.
        assert spawner.await_count == 4
    finally:
        await client.aclose()


# ---- claude_cli wedge: distinct type + fail-fast after 2 attempts --------


def test_cli_wedge_error_is_transient_subclass() -> None:
    from core.provider import _CliTransientError, _CliWedgeError
    # Subclass so the normal retry predicate still catches it, but code can
    # target the wedge specifically.
    assert issubclass(_CliWedgeError, _CliTransientError)


def test_stop_on_repeated_wedge() -> None:
    import types
    from core.provider import (
        _CliTransientError, _CliWedgeError, _stop_on_repeated_wedge,
    )

    def rs(exc, n):
        return types.SimpleNamespace(
            attempt_number=n,
            outcome=types.SimpleNamespace(exception=lambda: exc),
        )

    # A wedge on attempt 1 keeps going (give it one retry).
    assert _stop_on_repeated_wedge(rs(_CliWedgeError("x"), 1)) is False
    # A 2nd wedge bails — don't burn attempts 3 & 4 on a hang.
    assert _stop_on_repeated_wedge(rs(_CliWedgeError("x"), 2)) is True
    # A normal transient still gets the full budget (not wedge-capped).
    assert _stop_on_repeated_wedge(rs(_CliTransientError("x"), 2)) is False
    # No exception captured → don't stop on this predicate.
    assert _stop_on_repeated_wedge(rs(None, 3)) is False


def test_wedge_error_message_carries_switch_provider_guidance() -> None:
    from core.provider import _CliWedgeError
    msg = str(_CliWedgeError(
        "claude stdout closed but child didn't exit within 60s (no output "
        "collected) — an extended-thinking CLI hang, not a transient blip. "
        "Retrying wedges the same way; the fix is a different provider for "
        "this node (e.g. resume with `--config <codex_or_openai>.yaml`)."
    ))
    assert "different provider" in msg and "--config" in msg


@pytest.mark.asyncio
async def test_cli_retry_loop_stops_after_second_wedge() -> None:
    """Integration: drive a repeated ``_CliWedgeError`` through the ACTUAL
    ``AsyncRetrying`` loop (not just ``_stop_on_repeated_wedge`` in isolation)
    to prove the ``stop_after_attempt(4) | _stop_on_repeated_wedge``
    composition is wired correctly. A wedge must bail after 2 attempts — not
    burn the full 4-attempt budget on an unrecoverable hang — and reraise with
    its switch-provider guidance."""
    from core.provider import _CliWedgeError

    ep = resolve_endpoint(ProviderConfig(name="claude_cli"))
    client = LLMClient(ep)
    try:
        runner = AsyncMock(side_effect=_CliWedgeError(
            "stdout closed but child won't exit — the fix is a different "
            "provider for this node (e.g. resume with `--config <codex>.yaml`)."
        ))
        with patch("core.provider._run_cli", new=runner):
            # Instant backoff so the single inter-attempt wait doesn't sleep.
            with patch(
                "core.provider.wait_random_exponential",
                return_value=lambda *a, **kw: 0,
            ):
                with pytest.raises(_CliWedgeError, match="different provider"):
                    await client.chat([{"role": "user", "content": "x"}])
        # stop_after_attempt(4) alone would call 4×; the wedge predicate caps
        # it at 2. This is the wiring the unit test on the predicate can't see.
        assert runner.await_count == 2
    finally:
        await client.aclose()


# ---- heartbeat on the non-streaming communicate() path -------------------


@pytest.mark.asyncio
async def test_communicate_path_emits_heartbeats(monkeypatch: pytest.MonkeyPatch) -> None:
    """codex_cli/copilot_cli/gemini_cli use the bare communicate() path and
    emit no stream events, so without a beat they're silent in run.log and the
    dashboard reads them as 'pending'. The path now ticks heartbeat_cb while
    the child runs."""
    import core.provider as provider

    monkeypatch.setattr(provider, "_COMMUNICATE_HEARTBEAT_INTERVAL_S", 0.03)
    beats: list[dict] = []
    proc = MagicMock()
    proc.returncode = 0

    async def slow_comm(*_a, **_k):
        await asyncio.sleep(0.2)
        return (b"the model output text here", b"")

    proc.communicate = slow_comm
    spec = _CLI_SPECS["copilot_cli"]  # output_via="stdout", no extractor

    result = await provider._collect_via_communicate(
        proc, ["copilot"], spec, None, None, timeout_s=5.0,
        heartbeat_cb=lambda p: beats.append(p), node="implement",
    )

    assert result == "the model output text here"
    assert len(beats) >= 2, beats            # ~6 beats at 0.03 s over 0.2 s
    assert beats[0]["kind"] == "cli_progress"
    assert beats[0]["node"] == "implement"


@pytest.mark.asyncio
async def test_communicate_path_without_heartbeat_cb_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No heartbeat_cb → no beat task spun up; the output still comes back."""
    import core.provider as provider

    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(return_value=(b"plain output", b""))
    result = await provider._collect_via_communicate(
        proc, ["copilot"], _CLI_SPECS["copilot_cli"], None, None, timeout_s=5.0,
    )
    assert result == "plain output"
