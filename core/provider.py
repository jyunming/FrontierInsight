"""Unified async LLM client + proxy supervisor.

Every provider presents the same `LLMClient.chat(messages) -> str` surface
but resolves to one of three transports:

* **HTTP** (default) — direct OpenAI-compatible endpoint. Used by
  `codex`, `openai`, `gemini`, `ollama`, `vllm`.
* **HTTP via local proxy** — `ProxySupervisor` spawns a child process
  that exposes an OpenAI-compatible REST API on a free localhost port,
  then the http path targets that port. Used by `claude_code` (wrapper
  via `RichardAtCT/claude-code-openai-wrapper`) and `github_copilot_*`
  (via `npx copilot-api@latest start`). Ref-counted across quests.
* **CLI exec** — `LLMClient` spawns a local CLI binary per chat call,
  pipes the prompt in, and reads the response back. No proxy process,
  no HTTP. Used by:
  - `codex_cli` — `codex exec --output-last-message <tmp>` reusing the
    user's `codex login` ChatGPT Plus/Pro OAuth. Prompt is piped on
    stdin (not argv) so it does not appear in local process listings.
  - `claude_cli` — `claude --print --output-format text` reusing the
    user's `claude login` Claude Pro/Max OAuth (no `ANTHROPIC_API_KEY`
    needed; OAuth from the CLI's keychain is honored). Prompt on stdin.
  - `copilot_cli` — `copilot -s --allow-all-tools -p <prompt>` reusing
    the user's `gh auth login` Copilot Pro/Business credentials.
    `-s/--silent` strips the trailing stats block; `--allow-all-tools`
    is required for non-interactive mode. **Prompt is on argv** because
    the CLI doesn't document a stdin path — prefer `claude_cli`,
    `codex_cli`, or `gemini_cli` for sensitive prompts.
  - `gemini_cli` — `gemini --yolo -o json -p ""` (from
    `@google/gemini-cli`) with the prompt on stdin. `--yolo` auto-
    approves tool calls (else stdin deadlocks on confirmation prompts).
    `-o json` emits a structured envelope after some CLI warnings; the
    `output_extractor` pulls out the `response` field.

Proxy spawn details:
* `claude_code` — `poetry run python main.py <port>` from
  `FI_CLAUDE_CODE_WRAPPER_DIR` (defaults to `~/claude-code-openai-wrapper`).
* `github_copilot_cli` / `github_copilot_vscode` —
  `npx copilot-api@latest start --port <N> --rate-limit 60 --wait`.
  Both names route to the same proxy.

`ProxySupervisor` is reference-counted so a fleet of N quests using the
same proxy provider shares one proxy process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import ProviderConfig

_log = logging.getLogger("frontier_insight.provider")

# Known direct providers and their default endpoints. Users override via
# `provider.base_url` / `provider.model` / `provider.api_key_env`.
_DIRECT_DEFAULTS: dict[str, dict[str, str]] = {
    "codex": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-pro",
        "api_key_env": "GEMINI_API_KEY",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5-coder:32b",
        "api_key_env": "",  # Ollama ignores auth; pass empty.
    },
    "vllm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "api_key_env": "",
    },
}

PROXY_PROVIDERS: frozenset[str] = frozenset(
    {"claude_code", "github_copilot_cli", "github_copilot_vscode"}
)
# Back-compat alias for code that already imports the underscore-prefixed name.
# New callers should prefer `PROXY_PROVIDERS`.
_PROXY_PROVIDERS = PROXY_PROVIDERS

# `github_copilot_cli` and `github_copilot_vscode` are config aliases
# that BOTH spawn the same `npx copilot-api@latest start` proxy
# (see ``ProxySupervisor._spawn`` at L313). Without canonicalization
# the supervisor's ``_handles`` dict would key by raw provider name,
# so a fleet with one quest on each alias would spawn TWO redundant
# proxies (and only release one back). Normalize at the supervisor
# boundary so both aliases share a single handle with a single
# refcount.
_PROXY_ALIASES: dict[str, str] = {
    "github_copilot_vscode": "github_copilot_cli",
}


def _canonical_proxy_name(provider_name: str) -> str:
    """Return the canonical name for proxy-provider aliases that
    spawn the same upstream proxy. Used by ``ProxySupervisor`` to
    key its handle dict so two aliases share one proxy process."""
    return _PROXY_ALIASES.get(provider_name, provider_name)


@dataclass(frozen=True)
class _CliSpec:
    """How to invoke a local CLI as a chat endpoint."""

    argv: tuple[str, ...]            # base command; prompt may be appended
    pass_prompt_via: str             # "stdin" | "arg"
    output_via: str                  # "stdout" | "last_message_file"
    # Optional post-process step on the raw collected content. Used when
    # the CLI emits warnings/info before the real response or wraps the
    # response in a JSON envelope (see gemini_cli). `None` means no
    # extraction — `_run_cli` returns the raw collected content as-is.
    output_extractor: Callable[[str], str] | None = None
    # If set, and the user passed `provider.model` in their YAML config,
    # FI inserts `[model_flag, <provider.model>]` after argv[0] so the
    # CLI uses the user-specified model instead of its default. Leave
    # `provider.model` empty in YAML to keep the CLI's own default
    # (set by e.g. `~/.codex/config.toml` for codex, or the most-recent
    # `/model` selection for claude).
    model_flag: str | None = None


_CLI_SPECS: dict[str, _CliSpec] = {
    "claude_cli": _CliSpec(
        # `claude --print` prints the response to stdout and exits. Output
        # format "text" keeps it raw; "json" wraps in a JSON envelope. We
        # use "text" and let the engine's `_parse_json_lenient` cope with
        # whatever the model produces.
        argv=("claude", "--print", "--output-format", "text"),
        pass_prompt_via="stdin",
        output_via="stdout",
        model_flag="--model",   # provider.model = "opus" / "sonnet" / "claude-opus-4-7"
    ),
    "codex_cli": _CliSpec(
        # `codex exec` runs Codex non-interactively. We pipe the prompt
        # on stdin (codex reads "instructions from stdin" when no
        # positional PROMPT is given) rather than placing it on the
        # command line — argv would otherwise be visible in `ps`/Task
        # Manager and any other local process listing. stdout is the
        # agent log (token counts, tool calls); the final assistant
        # message is written to the file passed via --output-last-message.
        argv=("codex", "exec"),
        pass_prompt_via="stdin",
        output_via="last_message_file",
        model_flag="-m",        # provider.model = "gpt-5.5"; default reads ~/.codex/config.toml
    ),
    "copilot_cli": _CliSpec(
        # GitHub Copilot CLI (`copilot --prompt`). WARNING — this is an
        # AGENTIC CLI: it interprets prompts as user coding tasks and
        # may reply conversationally instead of running stateless LLM
        # inference. Empirically broken as a chat backend for FI's
        # pipeline (paper.md fills with "Are you trying to X?", code
        # node returns the empty stub). engine._warn_if_unsanctioned_provider
        # prints a loud warning when this provider is selected. Kept
        # in _CLI_SPECS so the configuration shape remains stable for
        # users who set it via the interview before reading docs.
        #
        # `-s/--silent` strips the trailing stats block. `--allow-all-tools`
        # is needed to avoid interactive permission prompts; removing it
        # would make the CLI hang on confirmation. The fundamental issue
        # is the agent loop, not this flag — switch to vscode_extension /
        # claude_cli / codex_cli / gemini_cli / openai for FI use.
        argv=("copilot", "-s", "--allow-all-tools", "-p"),
        pass_prompt_via="arg",
        output_via="stdout",
        model_flag="--model",   # provider.model = "gpt-5.2"
    ),
    "gemini_cli": _CliSpec(
        # `@google/gemini-cli` non-interactive. `--yolo` auto-approves
        # tool calls (otherwise stdin would deadlock waiting for user
        # confirmation). `-o json` emits a structured envelope whose
        # `response` field holds the agent answer; the envelope is
        # preceded by a few lines of CLI-level warnings (true-color,
        # MCP issues, etc.) that we strip via the output extractor.
        # Prompt is piped on stdin (the CLI documents stdin support and
        # appends -p text after it; we pass an empty -p so stdin alone
        # is the prompt content). Avoids argv leakage.
        argv=("gemini", "--yolo", "-o", "json", "-p", ""),
        pass_prompt_via="stdin",
        output_via="stdout",
        output_extractor=lambda raw: _extract_gemini_response(raw),
        model_flag="-m",        # provider.model = "gemini-3-pro"
    ),
}
CLI_PROVIDERS: frozenset[str] = frozenset(_CLI_SPECS)
_CLI_PROVIDERS = CLI_PROVIDERS  # back-compat alias

# Sentinel returned by `resolve_endpoint` when no API key env var was set
# (or the provider is configured as keyless, e.g. ollama/vllm). The OpenAI
# SDK requires a non-empty key string; downstream callers special-case this
# value to skip the `Authorization` header entirely.
_NO_KEY_SENTINEL = "not-needed"


@dataclass
class ResolvedEndpoint:
    base_url: str
    model: str
    api_key: str
    transport: str = "http"          # "http" | "cli" | "vscode_bridge"
    # The YAML `provider.name` (e.g. "openai", "claude_cli",
    # "vscode_extension") that this endpoint was resolved from. Carried
    # through to ``LLMClient._error_note`` so a failed LLM call's
    # traceback says ``provider=openai`` rather than just
    # ``transport=http`` — same transport class can come from openai,
    # gemini, ollama, vllm, … and the user wants to know which one.
    # Empty when an endpoint was hand-constructed in a test.
    provider_name: str = ""
    cli_spec: _CliSpec | None = None  # set when transport == "cli"
    # Only set (non-empty) when the user explicitly chose a CLI model via
    # YAML `provider.model`. `_run_cli` injects `[spec.model_flag, value]`
    # into argv only when this is non-empty. Keeps `model` free to carry
    # a human-readable display string for the Engine's startup log line
    # (e.g. "claude_cli (CLI default)") rather than going blank.
    cli_model_override: str = ""
    # VSCode-extension bridge. When transport == "vscode_bridge",
    # this is the localhost TCP port the FI extension is listening on;
    # the bridge client connects there for every chat call.
    vscode_bridge_port: int = 0
    # As with `cli_model_override`: the user's explicit YAML
    # `provider.model` (empty when unset). Sent as the wire-level
    # `model_hint` to the extension; an empty string is the documented
    # signal for "use the model selected in the Chat picker." We keep
    # this separate from `model` because the latter carries a
    # human-readable display string (e.g. "(VSCode chat default)")
    # for the Engine's startup log, and that string is NOT a valid
    # selectChatModels family filter.
    vscode_model_override: str = ""


@dataclass
class _ProxyHandle:
    name: str
    port: int
    proc: subprocess.Popen[bytes]
    refcount: int = 0


@dataclass
class ProxySupervisor:
    """Reference-counted lifecycle for proxy subprocesses.

    The spawn paths call out to `claude-code-openai-wrapper` /
    `copilot-api` to start a localhost proxy and reuse it across
    quests that share the same provider name.
    """

    _handles: dict[str, _ProxyHandle] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self, provider_name: str) -> _ProxyHandle:
        if provider_name not in _PROXY_PROVIDERS:
            raise ValueError(f"{provider_name!r} is not a proxy provider")
        # Canonicalize aliases (e.g. github_copilot_vscode → github_copilot_cli)
        # so two aliases that spawn the same proxy share a single handle.
        key = _canonical_proxy_name(provider_name)
        async with self._lock:
            handle = self._handles.get(key)
            if handle is None:
                # `_spawn` ends with a blocking poll of `/v1/models` (up to
                # 60s). Run it in a worker thread so concurrent quests doing
                # other work don't stall the event loop while one of them
                # waits for the proxy to come up.
                handle = await asyncio.to_thread(self._spawn, key)
                self._handles[key] = handle
            handle.refcount += 1
            return handle

    async def release(self, provider_name: str) -> None:
        key = _canonical_proxy_name(provider_name)
        async with self._lock:
            handle = self._handles.get(key)
            if handle is None:
                return
            handle.refcount -= 1
            if handle.refcount <= 0:
                handle.proc.terminate()
                try:
                    handle.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    handle.proc.kill()
                # Use the canonical key so the entry the terminated
                # handle is actually stored under gets removed.
                # Without this, releasing via a non-canonical alias
                # leaves a dead handle in _handles and the next
                # acquire returns it without respawning.
                self._handles.pop(key, None)

    async def shutdown(self) -> None:
        async with self._lock:
            for h in list(self._handles.values()):
                h.proc.terminate()
            for h in list(self._handles.values()):
                try:
                    h.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    h.proc.kill()
            self._handles.clear()

    def _spawn(self, provider_name: str) -> _ProxyHandle:
        port = _free_port()
        env = os.environ.copy()
        if provider_name == "claude_code":
            # Repo path is configurable via FI_CLAUDE_CODE_WRAPPER_DIR; the
            # wrapper has no PyPI release. Default assumes a sibling clone.
            wrapper_dir = env.get(
                "FI_CLAUDE_CODE_WRAPPER_DIR",
                str(Path.home() / "claude-code-openai-wrapper"),
            )
            cmd = ["poetry", "run", "python", "main.py", str(port)]
            cwd: str | None = wrapper_dir
            env["PORT"] = str(port)
        elif provider_name in ("github_copilot_cli", "github_copilot_vscode"):
            # `npx` will resolve and run copilot-api; `--rate-limit` and
            # `--wait` are recommended defaults given the abuse-detection
            # caveat in the upstream README.
            cmd = [
                "npx", "copilot-api@latest", "start",
                "--port", str(port),
                "--rate-limit", "60",
                "--wait",
            ]
            cwd = None
        else:
            raise NotImplementedError(provider_name)

        # cwd validation gives a clearer error than the generic
        # "proxy CLI not found" when the wrapper checkout is missing.
        if cwd is not None and not Path(cwd).is_dir():
            raise RuntimeError(
                f"proxy {provider_name!r}: working directory {cwd!r} does not exist. "
                f"Set FI_CLAUDE_CODE_WRAPPER_DIR to a clone of "
                f"RichardAtCT/claude-code-openai-wrapper with `poetry install` run."
            )
        try:
            # stdout/stderr -> DEVNULL: the proxies are long-lived and
            # write enough log volume to fill an OS pipe buffer if we
            # left them as PIPE without draining. Drop them entirely.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=cwd,
                env=env,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"proxy CLI {cmd[0]!r} not found on PATH. "
                f"For claude_code: clone RichardAtCT/claude-code-openai-wrapper, "
                f"`poetry install`, set FI_CLAUDE_CODE_WRAPPER_DIR. "
                f"For github_copilot_*: ensure Node and `npx` are on PATH; "
                f"run `npx copilot-api@latest auth` once."
            ) from e
        # If readiness times out, kill the orphan to avoid leaking proxies.
        try:
            _wait_for_openai_endpoint(port, timeout_s=60)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        return _ProxyHandle(name=provider_name, port=port, proc=proc)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_TRANSIENT_BRIDGE_MARKERS = (
    "net::err_http2",
    "net::err_connection",
    "net::err_network",
    "err_http2_protocol_error",
    "econnreset",
    "etimedout",
    "socket hang up",
    "503",
    "504",
    "502",
    "network connection",
    "firewall rules and network",
    "temporarily unavailable",
    "rate limit",
    "request failed",
    "bridge connection dropped",
    "bridge write failed",
    # The TS-side bridge fires this when it sees no streaming chunks
    # for 180 s; treat as transient so Python's 6-attempt budget
    # retries the request. Wall-time math: each Python attempt invokes
    # the TS bridge (which itself does up to 4 retries with 2/4/8 s
    # backoff = ~14 s) + the 180 s stall budget = up to ~3 min per
    # attempt. Python then waits its own exponential backoff (4/8/16/
    # 32/60 s, capped at 60 s) between attempts. Worst-case wall time
    # before a clean "upstream Copilot unavailable" error: ~18-20 min,
    # not the indefinite hang the old code allowed. Most quests
    # converge on a successful retry well before that.
    "bridge stalled",
)


def _is_bridge_error_transient(msg: str) -> bool:
    """Classify whether a BridgeError message looks worth retrying.

    Pattern-matches against well-known transient markers Copilot's
    backend emits via `vscode.lm.sendRequest` failures (HTTP/2
    protocol errors, connection resets, 5xx, rate limits) AND against
    bridge-side connection failures. Auth errors and "no model
    available for hint" are NOT considered transient.
    """
    m = (msg or "").lower()
    return any(marker in m for marker in _TRANSIENT_BRIDGE_MARKERS)


def _extract_gemini_response(raw: str) -> str:
    """`gemini -o json` emits a structured envelope after a few lines of
    CLI-level warnings (true-color hint, MCP issues, etc.). Scan stdout
    for the first valid JSON object (incrementally decoding from each
    `{`) and return its `response` field. Falls back to the raw text if
    no envelope is parseable — most failures still yield usable content
    for `_parse_json_lenient` downstream.

    Incremental decode avoids two failure modes of a naive
    `find('{')` + `rfind('}')` slice: (a) trailing non-JSON output after
    the envelope confuses `json.loads`, and (b) an earlier `{` inside a
    warning line shifts the start past the real envelope's opening
    brace, again breaking the parse."""
    decoder = json.JSONDecoder()
    i = 0
    n = len(raw)
    while i < n:
        if raw[i] != "{":
            i += 1
            continue
        try:
            envelope, _ = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(envelope, dict):
            response = envelope.get("response")
            if isinstance(response, str):
                return response
        # Parsed an unrelated object; keep scanning past its opening brace.
        i += 1
    return raw


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    """Flatten OpenAI Chat-Completions messages into one text block.

    FI's engine usually sends a single `user` message; we still handle
    system/assistant prior-turn messages defensively for callers that
    build multi-message conversations.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[system]\n{content}")
        elif role == "assistant":
            parts.append(f"[assistant prior turn]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(p for p in parts if p)


class _CliTransientError(RuntimeError):
    """Raised when a CLI invocation fails in a way worth retrying
    (non-zero exit with no parseable output). Distinct from `RuntimeError`
    so the retry predicate can target it precisely."""


async def _run_cli(
    spec: _CliSpec,
    prompt: str,
    *,
    model: str = "",
    timeout_s: float = 300.0,
) -> str:
    # Resolve the binary up front. On Windows, `asyncio.create_subprocess_exec`
    # does NOT honor PATHEXT, so an unqualified name like "codex" raises
    # FileNotFoundError even when `codex.CMD` is sitting in a PATH directory.
    # `shutil.which` does honor PATHEXT and returns the qualified path, so
    # we substitute argv[0] with whatever it resolves to.
    binary_name = spec.argv[0]
    resolved = shutil.which(binary_name)
    if resolved is None:
        raise RuntimeError(
            f"CLI provider binary {binary_name!r} not found on PATH. "
            f"Install and log in (`claude login`, `codex login`, or "
            f"`copilot` via GitHub Copilot CLI) before using this provider."
        )
    # Inject explicit model selection right after the resolved binary,
    # if both the spec supports it and the caller supplied a model.
    # Empty model => CLI keeps its own default.
    argv = [resolved]
    if spec.model_flag and model:
        argv.extend([spec.model_flag, model])
    argv.extend(spec.argv[1:])
    tmp_out_path: Path | None = None
    if spec.output_via == "last_message_file":
        tmp = tempfile.NamedTemporaryFile(
            prefix="fi_cli_out_", suffix=".txt", delete=False
        )
        tmp.close()
        tmp_out_path = Path(tmp.name)
        argv.extend(["--output-last-message", str(tmp_out_path)])

    if spec.pass_prompt_via == "arg":
        argv.append(prompt)
        stdin_bytes: bytes | None = None
    else:  # stdin
        stdin_bytes = prompt.encode("utf-8")

    # When the real answer lands in `tmp_out_path`, the CLI's stdout is
    # just an agent log; capturing it into a PIPE for a long prompt
    # wastes memory. Drop it. stderr stays piped so we can include its
    # tail in error messages.
    stdout_target = (
        asyncio.subprocess.DEVNULL
        if spec.output_via == "last_message_file"
        else asyncio.subprocess.PIPE
    )

    # Single try/finally so the tmpfile is unlinked on every exit path —
    # spawn failure, transient error, exception during communicate(), or
    # success. Previously a `FileNotFoundError` from spawn leaked the file.
    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin_bytes is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=stdout_target,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"CLI provider binary {argv[0]!r} not found on PATH. "
                f"Install and log in (`claude login` or `codex login`) "
                f"before using this provider."
            ) from e

        # Bounded wait: a CLI that hangs (e.g. concurrent-fleet
        # contention on a CLI that does heavy startup like gemini's MCP
        # bootup) gets killed here so tenacity retries the call. Without
        # this, a single stuck child blocks the whole quest indefinitely.
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(stdin_bytes), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            # Format the timeout for the error message at full precision
            # so sub-second values (test timeouts, fast retries) don't
            # round to "0s" and lose debuggability.
            elapsed = f"{timeout_s:g}s" if timeout_s >= 1 else f"{timeout_s * 1000:g}ms"
            try:
                proc.kill()
            except ProcessLookupError:
                pass  # already exited between communicate-timeout and our kill
            kill_clean = True
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                # Child did not reap within 5s after SIGKILL. Surface
                # this — a leaked child on POSIX becomes a zombie until
                # the parent dies, and on Windows the handle stays open.
                kill_clean = False
                _log.warning(
                    "CLI %s did not reap within 5s after SIGKILL; "
                    "process may be wedged in uninterruptible state",
                    spec.argv[0],
                )
            raise _CliTransientError(
                f"{spec.argv[0]} exceeded {elapsed} wall-clock and was killed"
                + ("" if kill_clean else " (post-kill wait timed out)")
            )
        if proc.returncode != 0:
            # Retryable: covers transient backend hiccups. Auth/quota
            # failures also land here but in practice clear after the
            # CLI refreshes OAuth; if they persist across 4 attempts the
            # error surfaces.
            raise _CliTransientError(
                f"{argv[0]} exited rc={proc.returncode}: "
                f"{stderr_b.decode('utf-8', 'replace')[-500:]}"
            )

        if spec.output_via == "last_message_file":
            assert tmp_out_path is not None
            content = tmp_out_path.read_text(encoding="utf-8", errors="replace")
        else:
            content = (stdout_b or b"").decode("utf-8", errors="replace")
        if spec.output_extractor is not None:
            content = spec.output_extractor(content)
        return content.strip()
    finally:
        if tmp_out_path is not None:
            tmp_out_path.unlink(missing_ok=True)


def _wait_for_openai_endpoint(port: int, *, timeout_s: int) -> None:
    """Both proxies expose `/v1/models`. Poll it for actual readiness
    rather than just a TCP bind — the FastAPI/Bun startup window between
    bind and serve has burned us before."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=2.0) as c:
                r = c.get(url)
                if r.status_code < 500:
                    return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise TimeoutError(
        f"proxy /v1/models on 127.0.0.1:{port} did not respond within {timeout_s}s "
        f"(last error: {last_err!r})"
    )


def resolve_endpoint(
    provider: ProviderConfig,
    supervisor: ProxySupervisor | None = None,
) -> ResolvedEndpoint:
    """Synchronous resolution for direct providers; raises for proxy ones.

    Proxy providers must go through `resolve_endpoint_async` so the
    supervisor can spawn the child process and assign a port.
    """
    name = provider.name
    if name in _PROXY_PROVIDERS:
        raise RuntimeError(
            f"provider {name!r} requires async resolution via resolve_endpoint_async"
        )
    if name == "vscode_extension":
        # The FI VSCode extension is the parent process;
        # it spawned us with `--vscode-bridge-port N` and the port
        # lives in provider.extra["bridge_port"] (the launch.py flag
        # writes it there at config-load time).
        port = int(provider.extra.get("bridge_port", 0))
        if port <= 0:
            raise RuntimeError(
                "vscode_extension provider requires extra['bridge_port'] "
                "to be set (the FI VSCode extension passes this via "
                "--vscode-bridge-port). Are you launching FI from outside "
                "the extension? Use copilot_cli for headless Copilot runs."
            )
        return ResolvedEndpoint(
            base_url="",
            model=provider.model or "(VSCode chat default)",
            api_key=_NO_KEY_SENTINEL,
            transport="vscode_bridge",
            provider_name=name,
            vscode_bridge_port=port,
            # The display string above is for logs only — it would be
            # an invalid family filter for selectChatModels. The real
            # override is empty unless the YAML pinned a model.
            vscode_model_override=provider.model or "",
        )
    if name in _CLI_PROVIDERS:
        # CLI providers exec a local binary per chat call. No URL, no key
        # (the CLI uses its own OAuth keychain). `cli_model_override` is
        # the user's explicit YAML choice (empty → CLI keeps its own
        # default). `model` carries a human-readable display string for
        # the Engine's startup log so the line never goes blank.
        return ResolvedEndpoint(
            base_url="",
            model=provider.model or f"{name} (CLI default)",
            api_key=_NO_KEY_SENTINEL,
            transport="cli",
            provider_name=name,
            cli_spec=_CLI_SPECS[name],
            cli_model_override=provider.model or "",
        )
    defaults = _DIRECT_DEFAULTS.get(name)
    if defaults is None:
        raise ValueError(f"unknown provider: {name!r}")
    api_key_env = provider.api_key_env or defaults["api_key_env"]
    api_key = os.environ.get(api_key_env, "") if api_key_env else _NO_KEY_SENTINEL
    return ResolvedEndpoint(
        base_url=provider.base_url or defaults["base_url"],
        model=provider.model or defaults["model"],
        api_key=api_key or _NO_KEY_SENTINEL,
        provider_name=name,
    )


async def resolve_endpoint_async(
    provider: ProviderConfig,
    supervisor: ProxySupervisor,
) -> ResolvedEndpoint:
    if provider.name not in _PROXY_PROVIDERS:
        return resolve_endpoint(provider)
    handle = await supervisor.acquire(provider.name)
    api_key = os.environ.get(provider.api_key_env or "", "") or _NO_KEY_SENTINEL
    return ResolvedEndpoint(
        base_url=f"http://127.0.0.1:{handle.port}/v1",
        model=provider.model or "default",
        api_key=api_key,
        provider_name=provider.name,
    )


# Per-1k-token USD pricing for the model variants FI talks to. Used by
# the cost-instrumentation hook in `Engine` to convert each chat
# response's ``usage`` block into a $ figure for ``.fi/cost.jsonl``.
# Numbers come from each provider's published rate card; update
# opportunistically when rates change. Missing entries fall through
# to cost=None (we don't fabricate). All values are USD per 1000
# tokens; downstream multiplies by token count and divides by 1000.
#
# Entries are matched by SUBSTRING on the model name (case-insensitive)
# — handles versioned variants like "claude-opus-4-7-20251201" matching
# "claude-opus-4-7". Order matters: more-specific (longer) keys are
# checked first.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI — gpt-5 family
    "gpt-5-mini": {"prompt_per_1k": 0.00025, "completion_per_1k": 0.002},
    "gpt-5": {"prompt_per_1k": 0.005, "completion_per_1k": 0.015},
    # OpenAI — gpt-4o family
    "gpt-4o-mini": {"prompt_per_1k": 0.00015, "completion_per_1k": 0.0006},
    "gpt-4o": {"prompt_per_1k": 0.0025, "completion_per_1k": 0.01},
    "gpt-4-turbo": {"prompt_per_1k": 0.01, "completion_per_1k": 0.03},
    # OpenAI — o1 reasoning family
    "o1-preview": {"prompt_per_1k": 0.015, "completion_per_1k": 0.06},
    "o1-mini": {"prompt_per_1k": 0.003, "completion_per_1k": 0.012},
    # Anthropic — Claude 4.x
    "claude-opus-4-7": {"prompt_per_1k": 0.015, "completion_per_1k": 0.075},
    "claude-sonnet-4-6": {"prompt_per_1k": 0.003, "completion_per_1k": 0.015},
    "claude-haiku-4-5": {"prompt_per_1k": 0.0008, "completion_per_1k": 0.004},
    # Anthropic — Claude 3.x (still in active use via copilot_cli etc.)
    "claude-3-5-sonnet": {"prompt_per_1k": 0.003, "completion_per_1k": 0.015},
    # Gemini
    "gemini-2.5-pro": {"prompt_per_1k": 0.00125, "completion_per_1k": 0.005},
    "gemini-2.5-flash": {"prompt_per_1k": 0.00015, "completion_per_1k": 0.0006},
    # Local — free
    "llama3.1:70b": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
    "llama3.1:8b": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
    "qwen2.5:32b": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
}


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Multiply token counts by per-model rates from
    :data:`MODEL_PRICING`. Returns ``None`` when no pricing row matches
    — callers log usage but skip the cost field so partial data is
    obvious in the chart."""
    if not model:
        return None
    needle = model.lower()
    # Longest key first so "gpt-4o-mini" matches before "gpt-4o".
    for key in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
        if key.lower() in needle:
            rates = MODEL_PRICING[key]
            return (
                prompt_tokens * rates["prompt_per_1k"] / 1000.0
                + completion_tokens * rates["completion_per_1k"] / 1000.0
            )
    return None


class LLMClient:
    """Thin async wrapper that speaks OpenAI Chat Completions.

    Built on `httpx.AsyncClient` rather than the openai SDK directly so
    multiple Engines in one process can share a single connection pool
    cleanly. The request shape is OpenAI-standard.

    Each call updates ``self.last_usage`` (or sets it to ``None``) with
    the response's token-usage info when the upstream returned one.
    The Engine reads this attribute after each chat call to write a
    line to ``<quest_root>/.fi/cost.jsonl``. CLI transports
    (claude_cli / codex_cli / copilot_cli / gemini_cli) and the
    vscode_bridge don't return structured usage today; their calls
    leave ``last_usage = None`` and the chart shows runtime-only.
    """

    def __init__(
        self,
        endpoint: ResolvedEndpoint,
        *,
        http: httpx.AsyncClient | None = None,
        timeout_s: float = 120.0,
        cli_timeout_s: float = 300.0,
    ) -> None:
        self.endpoint = endpoint
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=timeout_s)
        # CLI providers (claude_cli/codex_cli/copilot_cli/gemini_cli)
        # use this wall-clock cap per chat call; a stuck child gets
        # killed and tenacity retries. Defaults to 5 minutes — longer
        # than the typical 10–90 s per call but bounded so concurrent
        # fleet contention can't hang the whole quest indefinitely.
        self._cli_timeout_s = cli_timeout_s
        # Lazily-built VSCode-extension bridge client. The bridge
        # connection is shared across every chat call from this
        # LLMClient instance.
        self._bridge: Any | None = None
        # Last chat call's usage info (prompt_tokens /
        # completion_tokens / total_tokens) parsed from the response
        # body, or ``None`` when the transport doesn't return one.
        # Engine reads this after each chat() call to write a
        # cost-tracking row. Reset to None at the top of every
        # chat() so a previous call's usage can't leak into a
        # subsequent one if the new transport doesn't populate it.
        self.last_usage: dict[str, int] | None = None
        # Same idea for the model that ACTUALLY ran — for proxy
        # transports (copilot_cli etc.) the requested model and the
        # routed model can differ. Engine uses last_model to look up
        # the right pricing row when writing cost.jsonl.
        self.last_model: str | None = None

    async def aclose(self) -> None:
        if self._bridge is not None:
            try:
                await self._bridge.aclose()
            except Exception:
                pass
            self._bridge = None
        if self._owns_http:
            await self._http.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
        model: str | None = None,
        node: str = "",
    ) -> str:
        """Run one chat completion. ``model`` is an optional per-call
        override for per-node model routing. When provided and
        non-empty, it replaces the endpoint's default model for THIS
        call only — useful for sending different nodes through different
        models on the same provider (most relevant on Copilot, where
        all the model variants share one CLI and one premium-request
        budget). Falls back to ``self.endpoint.model`` when omitted.

        ``node`` is the FI engine node name (``"ideate"``, ``"implement"``,
        …); only used by the vscode_bridge transport so the extension
        can tag chat-panel progress messages with the right node name.
        Other transports ignore it for routing but DO attach it to any
        exception via ``Exception.add_note`` so error tracebacks show
        which node + which provider failed instead of just a raw
        httpx / CLI stderr stack.
        """
        try:
            return await self._chat_impl(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=extra,
                model=model,
                node=node,
            )
        except Exception as e:
            # ``except Exception`` excludes ``asyncio.CancelledError``
            # (a BaseException subclass since Python 3.8) — see the
            # cancellation-contract comment in _chat_impl below for why
            # that matters. CancelledError must propagate clean and
            # un-noted; everything else gets a "[FI] provider=…, node=…,
            # model=…" note so the user knows what was running when it
            # blew up. ``add_note`` is non-invasive (no exception
            # wrapping, no type change), keeps the original traceback
            # intact, and appears in standard `traceback.print_exc`
            # output on Python >=3.11 (our minimum).
            try:
                e.add_note(self._error_note(node=node, model=model))
            except AttributeError:  # pragma: no cover — needs Py<3.11
                pass
            raise

    def _error_note(self, *, node: str, model: str | None) -> str:
        """Format the FI-context note attached to LLM call exceptions.

        Includes the provider name, transport class, effective model,
        and (when set) the engine node name. Single line, prefixed
        ``[FI]`` so it's grep-able in error reports."""
        provider = self.endpoint.provider_name or self.endpoint.transport
        effective_model = model or self.endpoint.model
        parts = [
            f"provider={provider}",
            f"transport={self.endpoint.transport}",
            f"model={effective_model}",
        ]
        if node:
            parts.append(f"node={node}")
        return "[FI] " + ", ".join(parts)

    async def _chat_impl(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int | None,
        extra: dict[str, Any] | None,
        model: str | None,
        node: str,
    ) -> str:
        """The actual chat dispatch, split out from ``chat`` so the
        error-context note in ``chat`` wraps a single call site."""
        # Reset cost-tracking state at the top of every
        # call so a transport that DOESN'T return usage (CLI / bridge)
        # leaves last_usage = None until the post-call estimator fills
        # it in below. The Engine cost-logger reads last_usage after
        # the call returns; today every call carries at least an
        # estimated row so the cost chart shows real numbers instead
        # of empty bars for CLI / bridge transports.
        self.last_usage = None
        self.last_model = (model or self.endpoint.model)
        if self.endpoint.transport == "cli":
            text = await self._chat_cli(messages, model_override=model)
            self._fill_usage_estimate_if_missing(messages, text)
            return text
        if self.endpoint.transport == "vscode_bridge":
            text = await self._chat_vscode_bridge(
                messages, model_override=model, temperature=temperature,
                node=node,
            )
            self._fill_usage_estimate_if_missing(messages, text)
            return text
        body: dict[str, Any] = {
            "model": (model or self.endpoint.model),
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra:
            body.update(extra)
        url = self.endpoint.base_url.rstrip("/") + "/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        # Ollama (and vLLM with auth disabled) treat the OpenAI-compat
        # endpoint as keyless. Sending `Authorization: Bearer not-needed`
        # is harmless against most servers but Ollama's strict-mode reverse
        # proxies have rejected it. Only attach when we actually have a key.
        if self.endpoint.api_key and self.endpoint.api_key != _NO_KEY_SENTINEL:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        # Retry on transient upstream failures (5xx from cloud-routed Ollama
        # models, brief network blips). 4xx errors are not retried.
        #
        # Cancellation contract: ``asyncio.CancelledError`` is a
        # BaseException subclass (Python 3.8+), NOT Exception, so
        # tenacity's ``retry_if_exception_type(...)`` predicate cannot
        # match it — cancellation propagates straight through the retry
        # wrapper. httpx >= 0.27 then propagates that cancellation
        # through its anyio-based transport so the in-flight POST
        # unwinds promptly. ``test_chat_propagates_cancellation_promptly``
        # in tests/ verifies the asyncio await-chain unwinds in <1 s on
        # cancel (the test uses ``httpx.MockTransport``, which is
        # in-memory and never opens a real socket — socket-level
        # cancellation is a downstream httpx/anyio contract we trust
        # the upstream test suites to enforce).
        #
        # The two dangerous patterns that would silently break this:
        #   1. Wrapping the retry block in ``except BaseException``
        #      (``except Exception`` is fine — that's the rule above;
        #      it's also what the outer ``chat`` wrapper above uses
        #      for its error-context note attach).
        #   2. Adding ``asyncio.CancelledError`` to the
        #      ``retry_if_exception_type`` tuple.
        # Either would let tenacity catch the cancel and retry the
        # POST instead of letting it propagate. Re-run the two
        # cancellation tests in tests/test_provider.py if you touch
        # this retry config.
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=2, max=20),
            retry=retry_if_exception_type(
                (httpx.HTTPStatusError, httpx.TransportError, httpx.ReadTimeout)
            ),
            reraise=True,
        ):
            with attempt:
                r = await self._http.post(url, json=body, headers=headers)
                if r.status_code >= 500:
                    r.raise_for_status()
                # 4xx surfaces immediately — no retry on auth/quota errors.
                r.raise_for_status()
        data = r.json()
        # Capture token usage when the upstream returned
        # one. OpenAI-compatible servers (openai / codex / gemini /
        # ollama recent versions) include ``usage`` at the response
        # top level; older Ollama omits it. Missing → leave
        # last_usage = None so the cost-logger writes "unknown".
        usage = data.get("usage") or {}
        if usage and isinstance(usage, dict):
            self.last_usage = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(
                    usage.get("total_tokens", 0)
                    or (usage.get("prompt_tokens", 0) or 0)
                    + (usage.get("completion_tokens", 0) or 0)
                ),
            }
        # Some providers (notably some Copilot proxies) echo the
        # actual-routed model in the response — record it so the
        # cost chart attributes the call to the model that ran,
        # not the one we asked for.
        if isinstance(data.get("model"), str):
            self.last_model = data["model"]
        text = data["choices"][0]["message"]["content"]
        # Older Ollama versions omit ``usage`` from the response. Fall
        # through to char-based estimation so the cost.jsonl row still
        # carries real numbers instead of nulls.
        self._fill_usage_estimate_if_missing(messages, text)
        return text

    # ~4 chars per token holds reasonably well across English-text
    # tokenizers (BPE / tiktoken / SentencePiece). It's not exact —
    # code-heavy prompts run ~3 chars/token, math-heavy ~5 — but for
    # a cost chart that previously showed empty bars on CLI/bridge
    # transports, "roughly correct" beats "null". Estimated rows are
    # flagged with ``estimated: True`` so the chart can render them
    # differently if it ever wants to.
    _CHARS_PER_TOKEN = 4

    def _fill_usage_estimate_if_missing(
        self, messages: list[dict[str, str]], completion: str,
    ) -> None:
        """Populate ``last_usage`` with a char-based token estimate when
        the transport didn't return structured usage. No-op when usage
        was already captured upstream (HTTP responses with ``usage``)."""
        if self.last_usage is not None:
            return
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        completion_chars = len(completion or "")
        prompt_tokens = max(1, prompt_chars // self._CHARS_PER_TOKEN)
        completion_tokens = max(0, completion_chars // self._CHARS_PER_TOKEN)
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated": True,
        }

    async def _chat_vscode_bridge(
        self,
        messages: list[dict[str, str]],
        *,
        model_override: str | None = None,
        temperature: float = 0.2,
        node: str = "",
    ) -> str:
        """Route the chat call through the FI VSCode extension
        via the localhost TCP bridge. The extension makes the actual
        ``vscode.lm`` call on the authenticated user's behalf so we
        never touch the Copilot HTTP API directly — that's the whole
        point of the sanctioned path. ``model_override`` becomes the
        ``model_hint`` the extension passes to ``selectChatModels``."""
        # Lazy-import so non-VSCode runs don't pay for the module load.
        from .vscode_bridge import VSCodeBridgeClient
        if self._bridge is None:
            self._bridge = VSCodeBridgeClient(
                host="127.0.0.1", port=self.endpoint.vscode_bridge_port,
            )
            await self._bridge.connect()
        # Per-call override wins; otherwise use the user's
        # YAML-pinned model (`vscode_model_override`). DO NOT fall
        # through to `self.endpoint.model` — that field carries a
        # human-readable display string ("(VSCode chat default)")
        # for logging, and selectChatModels would reject it as an
        # invalid family filter. Empty hint = "use whatever the user
        # picked in the Chat model picker", which the extension
        # handles by calling selectChatModels({vendor: "copilot"}).
        if model_override is not None:
            hint = model_override
        else:
            hint = self.endpoint.vscode_model_override
        # Retry transient bridge errors with exponential backoff. The
        # extension's own sendRequest retries inside the TS bridge,
        # but a user on an older .vsix won't have that — and even on
        # the latest, the bridge surfaces `lm_error` after its own
        # retry exhausts. This is the second-chance layer.
        #
        # The retry predicate is a callable, NOT `retry_if_exception_type(BridgeError)`
        # — the latter would retry every BridgeError (including auth /
        # no-model-available / user-cancelled), which is wrong. We
        # specifically only want to retry transient-looking ones.
        from .vscode_bridge import BridgeError

        def _retry_transient_bridge(exc: BaseException) -> bool:
            return (
                isinstance(exc, BridgeError)
                and _is_bridge_error_transient(str(exc))
            )

        # Budget: 6 attempts with 5 inter-attempt waits of
        # 4/8/16/32/60s = ~2 minutes of cumulative backoff. Sustained
        # Copilot HTTP/2 outages have been observed lasting 30-90s;
        # the prior 3-attempt / ~14s budget was too tight and crashed
        # quests on transient upstream issues. The TS-side bridge
        # also retries 4x, so total wall time before a real failure
        # exceeds 2 min.
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(6),
                wait=wait_exponential(multiplier=2, min=4, max=60),
                retry=retry_if_exception(_retry_transient_bridge),
                reraise=True,
            ):
                with attempt:
                    return await self._bridge.chat(
                        messages, model_hint=hint or "", temperature=temperature,
                        node=node,
                    )
        except BridgeError as exc:
            if _is_bridge_error_transient(str(exc)):
                raise BridgeError(
                    "Copilot backend was unavailable across 6 retry "
                    "attempts (~2 min of cumulative backoff). This is "
                    "an upstream Copilot/HTTP issue, not a problem "
                    "with your config or network — please retry the "
                    f"quest in a few minutes. Last error: {exc}"
                ) from exc
            raise
        # Unreachable — tenacity reraise=True always raises on exhaustion.
        raise RuntimeError("vscode-bridge retry exhausted without raising")

    async def _chat_cli(
        self,
        messages: list[dict[str, str]],
        *,
        model_override: str | None = None,
    ) -> str:
        """Exec a local CLI binary with the prompt and return its output.

        Flattens the OpenAI-style `messages` list into a single text prompt
        (most LLM CLIs don't have a separate system/user channel — they
        accept one block of text). Retries with exponential backoff on
        `OSError` (transport-level OS errors during spawn) and
        `_CliTransientError` (any non-zero CLI exit). A missing CLI binary
        on PATH raises `RuntimeError` from `_run_cli` and is NOT retried —
        the user must install the CLI before this provider can succeed.
        Persistent auth/quota failures also surface here when the CLI's
        OAuth refresh has run out of options, since they are reported as
        non-zero exits and so will be retried up to 4 times before raising.
        """
        spec = self.endpoint.cli_spec
        if spec is None:  # pragma: no cover — guarded by transport check
            raise RuntimeError("transport=cli but no cli_spec set")
        prompt = _messages_to_text(messages)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=2, max=20),
            retry=retry_if_exception_type((OSError, _CliTransientError)),
            reraise=True,
        ):
            with attempt:
                # Per-call override takes precedence over the
                # endpoint-level override set at resolve time. Empty
                # string means "use the CLI's own default" — which is
                # also what cli_model_override="" means, so consistent.
                effective_model = (
                    model_override if model_override is not None
                    else self.endpoint.cli_model_override
                )
                return await _run_cli(
                    spec, prompt,
                    model=effective_model,
                    timeout_s=self._cli_timeout_s,
                )
        raise RuntimeError("unreachable: tenacity reraise=True must raise on exhaustion")
