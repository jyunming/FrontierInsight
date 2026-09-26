"""FI's CLI calls are answer-only.

FI asks ``codex_cli`` and ``claude_cli`` for text, but both CLIs run as agents
unless told otherwise: one real codex call inside an FI node read the user's
personal skills, ran shell commands and did 8 web searches. These tests pin
what turns that off -- the argv each CLI gets and the empty working directory
every CLI call runs in -- and three fixes that came with it: claude's answer is
its ``result`` envelope, a failed codex call says why, and codex's "at capacity"
waits like any other capacity error.

The collectors are mocked where only argv or cwd matter; the working-directory,
stream and failure tests also run a real Python script as the CLI.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import ProviderConfig
from core.provider import (
    _CLI_SPECS,
    LLMClient,
    _CliCapacityError,
    _CliSpec,
    _CliTransientError,
    _cli_retry_wait,
    _cli_stdout_errors,
    _run_cli,
    resolve_endpoint,
)

FIXTURES = Path(__file__).parent / "fixtures"
# A real claude haiku stream (sanitized): asked to run `echo hello` with every
# tool turned off, the model narrated, invented bash / shell / powershell tool
# calls, got "No such tool available" for each, and answered CANNOT RUN.
CLAUDE_STREAM = FIXTURES / "claude_stream_invented_tool_calls.jsonl"

CODEX_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "skill_search", "apps", "plugins", "browser_use",
    "computer_use", "memories", "multi_agent", "image_generation", "code_mode_host",
    "view_image", "tool_suggest",
)

# The events a real `codex exec --json` printed when gpt-5.6-luna was at
# capacity (thread id replaced), code mode disabled: the "Code Mode is
# unavailable" notice arrives as an item, the failure as top-level events.
CODEX_CAPACITY_STDOUT = "\n".join([
    '{"type":"thread.started","thread_id":"00000000-0000-0000-0000-000000000000"}',
    '{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Code Mode is '
    'unavailable because code-mode host is disabled. Code mode will fail closed; enable '
    '`features.code_mode_host` and install `codex-code-mode-host`."}}',
    '{"type":"turn.started"}',
    '{"type":"error","message":"Selected model is at capacity. Please try a different model."}',
    '{"type":"turn.failed","error":{"message":"Selected model is at capacity. Please try a '
    'different model."}}',
]) + "\n"


def _values(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i in range(len(argv) - 1) if argv[i] == flag]


async def _spawn_record(provider: str, **run_kwargs) -> tuple[list[str], dict, dict]:  # noqa: ANN003
    """Run ``_run_cli`` for a provider with the spawn and collectors mocked.

    Returns the argv, the spawn kwargs, and what the working directory looked
    like at spawn time.
    """
    spec = _CLI_SPECS[provider]
    seen: dict = {}

    async def spawn(*argv, **kwargs):  # noqa: ANN002, ANN003
        seen["argv"] = list(argv)
        seen["kwargs"] = kwargs
        cwd = kwargs.get("cwd")
        seen["cwd_is_dir"] = bool(cwd) and os.path.isdir(cwd)
        seen["cwd_listing"] = sorted(os.listdir(cwd)) if seen["cwd_is_dir"] else None
        return MagicMock()

    with patch("core.provider.shutil.which", return_value=f"/bin/{spec.argv[0]}"), \
         patch("core.provider.asyncio.create_subprocess_exec", new=spawn), \
         patch("core.provider._collect_via_communicate", new=AsyncMock(return_value="ok")), \
         patch("core.provider._collect_via_streaming", new=AsyncMock(return_value="ok")):
        assert await _run_cli(spec, "the prompt", **run_kwargs) == "ok"
    return seen["argv"], seen["kwargs"], seen


# ---------------------------------------------------------------------------
# codex_cli argv
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", CODEX_DISABLED_FEATURES)
async def test_codex_disables_each_tool_feature(feature: str) -> None:
    argv, _, _ = await _spawn_record("codex_cli")
    assert feature in _values(argv, "--disable")


@pytest.mark.asyncio
async def test_codex_disables_web_search() -> None:
    argv, _, _ = await _spawn_record("codex_cli")
    assert "web_search=disabled" in _values(argv, "-c")


@pytest.mark.asyncio
async def test_codex_ignores_user_config_saves_no_session_and_is_read_only() -> None:
    argv, _, _ = await _spawn_record("codex_cli")
    assert "--ignore-user-config" in argv
    assert "--ephemeral" in argv
    assert "--skip-git-repo-check" in argv
    assert _values(argv, "-s") == ["read-only"]


@pytest.mark.asyncio
async def test_codex_is_pointed_at_the_calls_own_directory() -> None:
    argv, kwargs, seen = await _spawn_record("codex_cli")
    assert _values(argv, "-C") == [kwargs["cwd"]]
    assert seen["cwd_listing"] == []


@pytest.mark.asyncio
async def test_codex_model_and_effort_still_reach_the_cli() -> None:
    """With config.toml not read, the model and effort FI passes are the only
    ones codex gets, so both must still be there."""
    argv, _, _ = await _spawn_record("codex_cli", model="gpt-5.6-luna", reasoning_effort="low")
    assert argv[1:4] == ["-m", "gpt-5.6-luna", "exec"]
    assert _values(argv, "-c") == ["web_search=disabled", 'model_reasoning_effort="low"']
    assert "the prompt" not in argv


# ---------------------------------------------------------------------------
# claude_cli argv
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claude_gets_no_tools_as_an_empty_argument() -> None:
    argv, _, _ = await _spawn_record("claude_cli")
    i = argv.index("--tools")
    assert argv[i + 1] == ""
    # `--tools` takes a list, so the next argument must be a flag, not a value
    # it could swallow.
    assert argv[i + 2].startswith("--")


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [
    "--strict-mcp-config", "--disable-slash-commands", "--safe-mode", "--no-session-persistence",
])
async def test_claude_turns_off_customisations(flag: str) -> None:
    argv, _, _ = await _spawn_record("claude_cli")
    assert flag in argv


@pytest.mark.asyncio
async def test_claude_images_and_effort_keep_the_answer_only_flags() -> None:
    argv, _, _ = await _spawn_record(
        "claude_cli", images=[("image/png", b"\x89PNG-fake")], reasoning_effort="high",
    )
    assert argv[argv.index("--tools") + 1] == ""
    assert _values(argv, "--effort") == ["high"]
    assert _values(argv, "--input-format") == ["stream-json"]
    assert "--safe-mode" in argv


def test_unverified_clis_keep_their_argv() -> None:
    """copilot_cli and gemini_cli could not be checked against the real CLI (quota, account tier), so their argv is
    unchanged; only the empty working directory applies to them. antigravity_cli has no flag that turns its tools off:
    its answer-only settings go through a home of the call's own (see the agy tests below)."""
    assert _CLI_SPECS["copilot_cli"].argv == ("copilot", "-s", "--allow-all-tools", "-p")
    assert _CLI_SPECS["gemini_cli"].argv == ("gemini", "--yolo", "-o", "json", "-p", "")
    assert _CLI_SPECS["antigravity_cli"].argv == (
        "agy", "--print", "", "--input-format", "stream-json",
        "--output-format", "stream-json", "--print-timeout", "30m",
    )
    for name in ("copilot_cli", "gemini_cli", "antigravity_cli", "claude_cli"):
        assert _CLI_SPECS[name].cwd_flag is None, name


# ---------------------------------------------------------------------------
# The empty working directory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(_CLI_SPECS))
async def test_every_cli_starts_in_an_empty_directory_that_is_removed(provider: str) -> None:
    argv, kwargs, seen = await _spawn_record(provider)
    cwd = kwargs["cwd"]
    assert seen["cwd_is_dir"] and seen["cwd_listing"] == []
    assert Path(cwd).name.startswith("fi_cli_call_")
    assert os.path.normcase(os.path.abspath(cwd)) != os.path.normcase(os.getcwd())
    assert not os.path.exists(cwd)
    if _CLI_SPECS[provider].pass_prompt_via == "arg":
        assert argv[-1] == "the prompt"


@pytest.mark.asyncio
async def test_each_call_gets_a_new_directory() -> None:
    _, first, _ = await _spawn_record("codex_cli")
    _, second, _ = await _spawn_record("codex_cli")
    assert first["cwd"] != second["cwd"]


@pytest.mark.asyncio
async def test_the_directory_is_removed_when_the_spawn_fails() -> None:
    seen: dict = {}

    async def spawn(*argv, **kwargs):  # noqa: ANN002, ANN003
        seen["cwd"] = kwargs["cwd"]
        raise FileNotFoundError("gone")

    with patch("core.provider.shutil.which", return_value="/bin/claude"), \
         patch("core.provider.asyncio.create_subprocess_exec", new=spawn):
        with pytest.raises(RuntimeError, match="not found on PATH"):
            await _run_cli(_CLI_SPECS["claude_cli"], "hi")
    assert seen["cwd"] and not os.path.exists(seen["cwd"])


@pytest.mark.asyncio
async def test_the_directory_is_removed_after_a_timeout_kill() -> None:
    seen: dict = {}
    proc = AsyncMock()

    async def hang(_=None):  # noqa: ANN001
        await asyncio.sleep(10)

    proc.communicate = hang
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=-9)
    proc.returncode = -9

    async def spawn(*argv, **kwargs):  # noqa: ANN002, ANN003
        seen["cwd"] = kwargs["cwd"]
        return proc

    with patch("core.provider.shutil.which", return_value="/bin/codex"), \
         patch("core.provider.asyncio.create_subprocess_exec", new=spawn):
        with pytest.raises(_CliTransientError, match="wall-clock"):
            await _run_cli(_CLI_SPECS["codex_cli"], "hi", timeout_s=0.05)
    proc.kill.assert_called()
    assert seen["cwd"] and not os.path.exists(seen["cwd"])


def _script(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_a_real_process_runs_in_the_empty_directory(tmp_path: Path) -> None:
    """A real child reports its own working directory and what is in it, then
    leaves a file there; the directory is still removed."""
    script = _script(tmp_path, "fake_cli.py", (
        "import json, os, sys\n"
        "flag = sys.argv[sys.argv.index('-C') + 1] if '-C' in sys.argv else None\n"
        "report = {'cwd': os.getcwd(), 'listing': sorted(os.listdir('.')), 'flag_dir': flag,\n"
        "          'prompt': sys.stdin.read()}\n"
        # Only inside a call directory: if the cwd were ever not passed, the
        # child would be in the test run's own directory.
        "if os.path.basename(os.getcwd()).startswith('fi_cli_call_'):\n"
        "    open('left_behind.txt', 'w').write('x')\n"
        "print(json.dumps(report))\n"
    ))
    spec = _CliSpec(
        argv=(sys.executable, str(script)), pass_prompt_via="stdin", output_via="stdout",
        cwd_flag="-C",
    )
    report = json.loads(await _run_cli(spec, "hello from FI", timeout_s=60))

    def same(a: str) -> str:
        return os.path.normcase(os.path.normpath(a))

    assert report["listing"] == []
    assert report["prompt"] == "hello from FI"
    assert same(report["flag_dir"]) == same(report["cwd"])
    assert Path(report["cwd"]).name.startswith("fi_cli_call_")
    assert same(report["cwd"]) != same(os.getcwd())
    assert not os.path.exists(report["cwd"])


@pytest.mark.asyncio
async def test_a_real_codex_style_call_keeps_its_answer_file_outside_the_directory(
    tmp_path: Path,
) -> None:
    """The answer file codex writes is passed by absolute path outside the
    call's directory, so the directory is empty at start and the answer is
    still read back."""
    script = _script(tmp_path, "fake_codex.py", (
        "import json, os, sys\n"
        "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
        "listing = sorted(os.listdir('.'))\n"
        "open(out, 'w', encoding='utf-8').write(json.dumps({'cwd': os.getcwd(), 'listing': listing,\n"
        "    'out_inside_cwd': os.path.dirname(os.path.abspath(out)) == os.getcwd()}))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 5, 'output_tokens': 1}}))\n"
    ))
    codex = _CLI_SPECS["codex_cli"]
    spec = dataclasses.replace(codex, argv=(sys.executable, str(script), *codex.argv[1:]))
    usage: dict = {}
    report = json.loads(await _run_cli(spec, "hi", timeout_s=60, usage_out=usage))
    assert report["listing"] == [] and report["out_inside_cwd"] is False
    assert usage["prompt_tokens"] == 5
    assert not os.path.exists(report["cwd"])


# ---------------------------------------------------------------------------
# claude: the answer is the result envelope
# ---------------------------------------------------------------------------


def _replay_spec(tmp_path: Path, stream: Path) -> _CliSpec:
    script = _script(tmp_path, "replay.py", (
        "import sys\n"
        "sys.stdout.write(open(sys.argv[1], encoding='utf-8').read())\n"
    ))
    return dataclasses.replace(
        _CLI_SPECS["claude_cli"], argv=(sys.executable, str(script), str(stream)),
    )


def test_the_fixture_is_the_stream_that_used_to_leak_narration() -> None:
    events = [json.loads(line) for line in CLAUDE_STREAM.read_text(encoding="utf-8").splitlines() if line]
    deltas = "".join(
        e["event"]["delta"]["text"] for e in events
        if e["type"] == "stream_event" and e["event"]["delta"]["type"] == "text_delta"
    )
    assert deltas == (
        "I'll run the echo command using the bash tool since it's available in this "
        "environment.Let me try other tool names:CANNOT RUN"
    )
    tool_uses = [
        b["name"] for e in events if e["type"] == "assistant"
        for b in e["message"]["content"] if b["type"] == "tool_use"
    ]
    refusals = [
        b["content"] for e in events if e["type"] == "user"
        for b in e["message"]["content"] if b["type"] == "tool_result"
    ]
    assert tool_uses == ["bash", "shell", "powershell"]
    assert all("No such tool available" in r for r in refusals) and len(refusals) == 3
    assert [e["result"] for e in events if e["type"] == "result"] == ["CANNOT RUN"]


@pytest.mark.asyncio
async def test_claude_answer_is_the_result_text_for_the_real_stream(tmp_path: Path) -> None:
    usage: dict = {}
    answer = await _run_cli(
        _replay_spec(tmp_path, CLAUDE_STREAM), "run echo", timeout_s=60, usage_out=usage,
    )
    assert answer == "CANNOT RUN"
    assert usage["prompt_tokens"] == 33 + 7456 + 19202
    assert usage["completion_tokens"] == 2131


@pytest.mark.asyncio
async def test_llm_client_returns_only_the_result_text(tmp_path: Path) -> None:
    lines = [line.encode("utf-8") + b"\n" for line in CLAUDE_STREAM.read_text(encoding="utf-8").splitlines()]
    it = iter(lines + [b""])
    proc = AsyncMock()
    proc.stdout = AsyncMock()

    async def readline() -> bytes:
        return next(it, b"")

    proc.stdout.readline = readline
    proc.stdin = AsyncMock()
    proc.stdin.write = lambda b: None
    proc.stdin.drain = AsyncMock(return_value=None)
    proc.stdin.close = lambda: None
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.wait = AsyncMock(return_value=0)
    proc.returncode = 0
    proc.kill = lambda: None
    client = LLMClient(resolve_endpoint(ProviderConfig(name="claude_cli", model="claude-haiku-4-5")))
    try:
        with patch("core.provider.shutil.which", return_value="/bin/claude"), \
             patch("core.provider.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            answer = await client.chat([{"role": "user", "content": "run echo"}])
    finally:
        await client.aclose()
    assert answer == "CANNOT RUN"
    assert client.last_usage["prompt_tokens"] == 26691


def _stream_file(tmp_path: Path, events: list[dict]) -> Path:
    path = tmp_path / "stream.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return path


def _delta(text: str) -> dict:
    return {"type": "stream_event", "event": {"type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text}}}


@pytest.mark.asyncio
async def test_without_an_envelope_the_deltas_are_the_answer(tmp_path: Path) -> None:
    stream = _stream_file(tmp_path, [_delta("Hello "), _delta("world")])
    assert await _run_cli(_replay_spec(tmp_path, stream), "x", timeout_s=60) == "Hello world"


@pytest.mark.asyncio
async def test_an_envelope_with_no_text_keeps_the_deltas(tmp_path: Path) -> None:
    stream = _stream_file(tmp_path, [_delta("Hello "), _delta("world"), {"type": "result", "result": ""}])
    assert await _run_cli(_replay_spec(tmp_path, stream), "x", timeout_s=60) == "Hello world"


@pytest.mark.asyncio
async def test_an_envelope_alone_is_the_answer(tmp_path: Path) -> None:
    stream = _stream_file(tmp_path, [{"type": "result", "result": "only the envelope"}])
    assert await _run_cli(_replay_spec(tmp_path, stream), "x", timeout_s=60) == "only the envelope"


# ---------------------------------------------------------------------------
# codex: a failed call says why, and "at capacity" waits like capacity
# ---------------------------------------------------------------------------


class _Outcome:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def exception(self) -> BaseException:
        return self._exc


class _State:
    def __init__(self, exc: BaseException) -> None:
        self.outcome = _Outcome(exc)
        self.attempt_number = 1
        self.idle_for = 0.0
        self.upcoming_sleep = 0.0


def _failing_codex(tmp_path: Path, stdout: str, stderr: str = "Reading prompt from stdin...\n") -> _CliSpec:
    stdout_file = tmp_path / "stdout.jsonl"
    stdout_file.write_text(stdout, encoding="utf-8")
    script = _script(tmp_path, "failing_codex.py", (
        "import sys\n"
        "sys.stdin.read()\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.stdout.write(open({str(stdout_file)!r}, encoding='utf-8').read())\n"
        "sys.exit(1)\n"
    ))
    codex = _CLI_SPECS["codex_cli"]
    return dataclasses.replace(codex, argv=(sys.executable, str(script), *codex.argv[1:]))


@pytest.mark.asyncio
async def test_codex_at_capacity_says_so_and_waits_30_to_90_seconds(tmp_path: Path) -> None:
    with pytest.raises(_CliCapacityError) as info:
        await _run_cli(_failing_codex(tmp_path, CODEX_CAPACITY_STDOUT), "hi", timeout_s=60, usage_out={})
    message = str(info.value)
    assert "Selected model is at capacity" in message
    assert "exited rc=1" in message
    assert "Code Mode is unavailable" not in message
    for _ in range(20):
        assert 30.0 <= _cli_retry_wait(_State(info.value)) <= 90.0


@pytest.mark.asyncio
async def test_codex_other_failure_is_reported_and_keeps_the_short_wait(tmp_path: Path) -> None:
    stdout = (
        '{"type":"turn.started"}\n'
        '{"type":"error","message":"stream disconnected before completion: error sending request"}\n'
        '{"type":"turn.failed","error":{"message":"stream disconnected before completion: error sending request"}}\n'
    )
    with pytest.raises(_CliTransientError) as info:
        await _run_cli(_failing_codex(tmp_path, stdout), "hi", timeout_s=60, usage_out={})
    assert not isinstance(info.value, _CliCapacityError)
    assert str(info.value).count("stream disconnected before completion") == 1
    for _ in range(20):
        assert _cli_retry_wait(_State(info.value)) <= 20.0


@pytest.mark.asyncio
async def test_codex_failure_without_json_keeps_the_stderr_message(tmp_path: Path) -> None:
    spec = _failing_codex(
        tmp_path, "", stderr="Reading prompt from stdin...\nNot inside a trusted directory\n",
    )
    with pytest.raises(_CliTransientError) as info:
        await _run_cli(spec, "hi", timeout_s=60, usage_out={})
    assert not isinstance(info.value, _CliCapacityError)
    message = str(info.value).replace("\r\n", "\n")  # a Windows child writes CRLF
    assert "exited rc=1: Reading prompt from stdin...\nNot inside a trusted directory" in message


@pytest.mark.asyncio
async def test_codex_non_capacity_failure_still_retries_four_times() -> None:
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(
        b'{"type":"error","message":"stream disconnected before completion"}\n',
        b"Reading prompt from stdin...\n",
    ))
    proc.returncode = 1
    spawner = AsyncMock(return_value=proc)
    client = LLMClient(resolve_endpoint(ProviderConfig(name="codex_cli")))
    try:
        with patch("core.provider.shutil.which", return_value="/bin/codex"), \
             patch("core.provider.asyncio.create_subprocess_exec", new=spawner), \
             patch("core.provider.wait_random_exponential", return_value=lambda *a, **kw: 0):
            with pytest.raises(_CliTransientError, match="stream disconnected before completion"):
                await client.chat([{"role": "user", "content": "x"}])
    finally:
        await client.aclose()
    assert spawner.await_count == 4


def test_stdout_errors_reads_only_top_level_failure_events() -> None:
    assert _cli_stdout_errors(CODEX_CAPACITY_STDOUT.encode("utf-8")) == [
        "Selected model is at capacity. Please try a different model.",
    ]
    assert _cli_stdout_errors(None) == []
    assert _cli_stdout_errors(b"not json\n{broken\n[1, 2]\n") == []
    assert _cli_stdout_errors(b'{"type":"turn.failed","error":"plain string"}\n') == ["plain string"]



# ---------------------------------------------------------------------------
# antigravity_cli: answer-only through settings in a home of the call's own
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agy_runs_with_a_home_of_its_own_whose_settings_refuse_its_tools() -> None:
    """agy follows ~/.gemini/antigravity-cli/settings.json, often "always-proceed": in real quests it ran shell
    commands, fetched papers and read FI's source mid-answer. Every call gets a fresh home holding FI's settings."""
    seen: dict = {}

    async def spawn(*argv, **kwargs):  # noqa: ANN002, ANN003
        env = kwargs["env"]
        seen["home"], seen["profile"] = env["HOME"], env["USERPROFILE"]
        settings = Path(env["HOME"]) / ".gemini" / "antigravity-cli" / "settings.json"
        seen["settings"] = json.loads(settings.read_text(encoding="utf-8"))
        seen["path"] = env.get("PATH")
        return MagicMock()

    with patch("core.provider.shutil.which", return_value="/bin/agy"),          patch("core.provider.asyncio.create_subprocess_exec", new=spawn),          patch("core.provider._collect_via_communicate", new=AsyncMock(return_value="ok")),          patch("core.provider._collect_via_streaming", new=AsyncMock(return_value="ok")):
        assert await _run_cli(_CLI_SPECS["antigravity_cli"], "the prompt") == "ok"
    assert seen["home"] == seen["profile"] and Path(seen["home"]).name.startswith("fi_cli_home_")
    assert os.path.normcase(seen["home"]) != os.path.normcase(os.path.expanduser("~"))
    settings = seen["settings"]
    assert settings["toolPermission"] == "request-review" and settings["allowNonWorkspaceAccess"] is False
    assert set(settings["permissions"]["deny"]) == {"command(*)", "write_file(*)", "read_url(*)", "mcp(*)"}
    assert seen["path"] == os.environ.get("PATH"), "the rest of the environment is kept (the CLI must still start)"
    assert not os.path.exists(seen["home"]), "the home goes with the call"


@pytest.mark.asyncio
async def test_other_clis_get_no_home_of_their_own() -> None:
    _, kwargs, _ = await _spawn_record("claude_cli")
    assert kwargs.get("env") is None or kwargs["env"].get("HOME") == os.environ.get("HOME")


def test_agy_is_told_its_tools_are_off() -> None:
    from core.provider import _encode_antigravity_stdin

    content = json.loads(_encode_antigravity_stdin("What is 2+2?"))["message"]["content"]
    assert content.startswith("Answer this request directly") and content.endswith("What is 2+2?")
    assert "search the web" in content and "image files the request names" in content
