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
import contextlib
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
    wait_random_exponential,
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
    output_via: str                  # "stdout" | "last_message_file" | "stream_json"
    # Optional post-process step on the raw collected content. Used when
    # the CLI emits warnings/info before the real response or wraps the
    # response in a JSON envelope (see gemini_cli). `None` means no
    # extraction — `_run_cli` returns the raw collected content as-is.
    # Ignored for ``output_via="stream_json"`` (the stream parser is
    # already format-aware).
    output_extractor: Callable[[str], str] | None = None
    # If set, and the user passed `provider.model` in their YAML config,
    # FI inserts `[model_flag, <provider.model>]` after argv[0] so the
    # CLI uses the user-specified model instead of its default. Leave
    # `provider.model` empty in YAML to keep the CLI's own default
    # (set by e.g. `~/.codex/config.toml` for codex, or the most-recent
    # `/model` selection for claude).
    model_flag: str | None = None
    # Hard ceiling (characters) on the prompt this CLI will accept on a
    # single turn. ``None`` = no known limit. When set and a prompt exceeds
    # it, ``_run_cli`` trims the MIDDLE of the prompt (keeping the task head
    # + the output-format tail) so the call still goes through instead of the
    # CLI rejecting it with a hard "input_too_large" error. codex_cli caps a
    # turn at 1,048,576 chars; we sit under it to leave room for codex's own
    # system prompt + tool schema wrapped around our instructions.
    max_input_chars: int | None = None


_CLI_SPECS: dict[str, _CliSpec] = {
    "claude_cli": _CliSpec(
        # `claude --print` prints the response to stdout and exits.
        # We use ``stream-json`` + ``--include-partial-messages`` so the
        # CLI emits SSE-style events (one JSON-per-line) instead of one
        # silent flush at the end. This is load-bearing on Sonnet 4.6:
        # complex prompts go into extended-thinking mode (tens of
        # thousands of ``thinking_delta`` tokens before any answer text)
        # which under ``--output-format text`` looks identical to a
        # hung process. With stream-json the reader sees a steady
        # stream of events and the inactivity-timer watchdog correctly
        # distinguishes "model is thinking" from "process is stuck".
        argv=(
            "claude", "--print",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",  # required by claude_cli for stream-json + --print
        ),
        pass_prompt_via="stdin",
        output_via="stream_json",
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
        # codex `turn/start` rejects input over 1,048,576 chars with
        # ``input_too_large``. Sit ~150K under to leave room for codex's
        # own system prompt + tools schema. Hit by analyze/write on quests
        # whose literature + result_json grow large (e.g. after a broaden).
        max_input_chars=900_000,
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
        # Prompt is passed as a command-line ARG, so on Windows the whole
        # command line is subject to the cmd.exe limit (~8191 chars) — copilot
        # ships as `copilot.BAT` and a long design/write prompt (with
        # accumulated feedback_history) fails with `rc=1: The command line is
        # too long`. Cap under that so the shared prompt-trimmer (see
        # `_truncate_prompt_to_fit`) kicks in first. NOTE: 7 KB is small for a
        # rich node prompt, so copilot_cli trims heavily on long-context nodes
        # — a stdin/temp-file transport (like codex/gemini) is the real fix;
        # prefer codex_cli / openai for long prompts on Windows.
        max_input_chars=7000,
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
    # the client picks ONE of two transports based on what the caller
    # populated:
    #   * ``vscode_bridge_socket`` (preferred for --serve / --tools) →
    #     Unix-domain socket (POSIX) or named pipe (Windows) at the
    #     extension's session-long PersistentBridge address.
    #   * ``vscode_bridge_port`` (per-chat-command spawns) → loopback
    #     TCP port the extension's per-command Bridge picked.
    vscode_bridge_port: int = 0
    vscode_bridge_socket: str = ""
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


class _CliWedgeError(_CliTransientError):
    """The specific failure where a CLI streams (often minutes of
    extended-thinking) then closes stdout WITHOUT producing an answer and
    won't exit even to SIGKILL — an unrecoverable hang, not a transient blip.
    Retrying it just wedges again the same way (observed 4×/4 on a heavy
    claude_cli code-gen prompt). A subclass so the retry loop can cap it
    faster than a normal transient and the message can carry switch-provider
    guidance."""


def _stop_on_repeated_wedge(retry_state: "Any") -> bool:
    """Tenacity stop: bail after the 2nd wedge instead of burning the full
    4-attempt budget on an unrecoverable hang. A wedge means the CLI streamed
    then died without an answer and won't reap — retrying reproduces it, so
    the remaining ~2 attempts (each a multi-minute call) are pure waste. The
    `_CliWedgeError` then reraises with switch-provider guidance."""
    outcome = getattr(retry_state, "outcome", None)
    exc = outcome.exception() if outcome is not None else None
    return isinstance(exc, _CliWedgeError) and retry_state.attempt_number >= 2


# Output gates. CLI vendors sometimes deliver upstream-state messages
# as plain ``text_delta`` events instead of structured error envelopes
# — the message then gets aggregated into the response and the engine
# happily writes it to ``paper.md`` / ``experiment.py`` / etc. The
# OPC quest produced a 64-byte paper.md containing literally
# ``"You've hit your session limit · resets 2:30am (Europe/Brussels)"``.
# Each gate names itself, defines a short list of patterns it watches
# for, and a length cap so a legitimate long paper that happens to
# mention "rate limit" in body text doesn't false-positive. The
# registry is iterated in declaration order; the first gate that
# fires names itself in the raised ``_CliTransientError`` so retry
# logs show *which* gate triggered (and a future maintainer can tune
# the right pattern list without grepping the whole file).
#
# All gates produce ``_CliTransientError`` → tenacity retries (and
# may escalate the model via the per-node fallback table). We do NOT
# silently swallow these into the artifact — that's the bug they exist
# to catch.

_CLI_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "you've hit your session limit",
    "you have hit your session limit",
    "rate limit exceeded · resets",
    "rate limit exceeded - resets",
    "out of credits - upgrade your plan",
    "claude usage limit reached",
)

# Long-duration / non-recoverable CLI failures: an hours-away session or usage
# limit, exhausted credits, or a dead / mis-scoped auth token. These CANNOT
# clear within the ~4-attempt retry window, so retrying just burns 3-4
# multi-minute calls before crashing anyway (the failure mode that truncated an
# earlier run here). Matched case-insensitively against the CLI error message
# (which carries the CLI's stderr). Deliberately separate from the momentary
# "rate limit exceeded · resets <soon>" case, which IS worth a retry.
_CLI_FATAL_MARKERS: tuple[str, ...] = (
    "session limit",
    "usage limit reached",
    "out of credits",
    "token may be invalid",
    "copilot requests",          # gh token lacking the Copilot scope
    "run '/login'",
    "re-authenticate",
)


def _retry_http_error(exc: BaseException) -> bool:
    """HTTP retry predicate: retry genuine transients only. A 4xx (bad/expired
    key, content policy, malformed or oversized body) is deterministic —
    retrying wastes the backoff budget and crashes anyway — so only 5xx and 429
    are retried alongside transport / read-timeout errors. (The prior
    ``retry_if_exception_type(HTTPStatusError)`` retried every 4xx despite the
    inline comment claiming it didn't.)"""
    if isinstance(exc, (httpx.TransportError, httpx.ReadTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        resp = getattr(exc, "response", None)
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc >= 500 or sc == 429
        # A real HTTP error always carries an int status; a malformed/absent
        # one is surfaced rather than looped.
        return False
    return False


def _retry_cli_error(exc: BaseException) -> bool:
    """CLI retry predicate: retry OSErrors and transient CLI failures, EXCEPT a
    long-duration session/usage/credit limit or an auth failure (see
    ``_CLI_FATAL_MARKERS``) — those can't clear within the retry window, so
    abort fast with the original message rather than burning the budget."""
    if isinstance(exc, OSError):
        return True
    if isinstance(exc, _CliTransientError):
        return not any(m in str(exc).lower() for m in _CLI_FATAL_MARKERS)
    return False


# --- Process-wide LLM-call backpressure -------------------------------------
#
# A single quest fans out to a handful of concurrent LLM calls (a review panel,
# an ensemble). A ``--fleet`` run multiplies that by the number of live quests,
# so an unbounded process can burst into dozens/hundreds of simultaneous
# provider calls and trip provider rate limits (audit: "no global backpressure").
# This semaphore caps concurrent in-flight ``chat`` dispatches across EVERY
# LLMClient in the process. It is created lazily inside the running event loop
# (an ``asyncio.Semaphore`` binds to the loop it is first awaited on), sized from
# ``FI_MAX_CONCURRENT_LLM_CALLS``. Default 0 == unlimited (a no-op nullcontext,
# zero behaviour change) so single quests are never throttled unless an operator
# opts in; set e.g. ``FI_MAX_CONCURRENT_LLM_CALLS=8`` for a large fleet.
#
# Deadlock note: the slot is held only around a single ``_chat_impl`` dispatch,
# which never re-enters ``chat``. Fallback/retry loops acquire a fresh slot per
# attempt (release between), so a saturated cap queues — it cannot deadlock.
_LLM_CALL_SEM: "asyncio.Semaphore | None" = None
_LLM_CALL_SEM_LIMIT: int = -1  # -1 == "not yet resolved from the environment"
_LLM_CALL_SEM_LOOP: "asyncio.AbstractEventLoop | None" = None


def _llm_call_limit() -> int:
    """Resolve the concurrency cap from the environment (0/negative == off)."""
    try:
        return int(os.environ.get("FI_MAX_CONCURRENT_LLM_CALLS", "0") or "0")
    except ValueError:
        return 0


def _llm_call_slot():
    """Return an async context manager gating one LLM dispatch on the
    process-wide cap. ``nullcontext`` (no-op) when the cap is disabled.

    A single shared semaphore enforces the cap across all concurrent callers on
    one event loop; it is recreated when the limit changes (env tweaked) or the
    running loop changes (an ``asyncio.Semaphore`` is bound to the loop it was
    created on — reusing one across loops raises, which pytest-asyncio would
    hit since each test gets a fresh loop)."""
    global _LLM_CALL_SEM, _LLM_CALL_SEM_LIMIT, _LLM_CALL_SEM_LOOP
    limit = _llm_call_limit()
    if limit <= 0:
        return contextlib.nullcontext()
    loop = asyncio.get_running_loop()
    if (
        _LLM_CALL_SEM is None
        or _LLM_CALL_SEM_LIMIT != limit
        or _LLM_CALL_SEM_LOOP is not loop
    ):
        _LLM_CALL_SEM = asyncio.Semaphore(limit)
        _LLM_CALL_SEM_LIMIT = limit
        _LLM_CALL_SEM_LOOP = loop
    return _LLM_CALL_SEM


# Placeholder / "I gave up" patterns that some CLIs emit instead of a
# real response. Distinct from refusal — these signal the LLM lost the
# plot mid-call rather than declining to help.
_CLI_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "[placeholder]",
    "please provide more details",
    "i need more information to continue",
    "i'm unable to generate a response right now",
)

# Refusal patterns. False-positive risk is higher here — a real paper
# might quote a refusal. So we keep the list tight to phrases that
# typically come from policy-classifier interception (not from the
# topic itself) and only flag when the whole output is short.
_CLI_REFUSAL_MARKERS: tuple[str, ...] = (
    "i cannot assist with that",
    "i can't help with that request",
    "i'm sorry, but i cannot",
    "as an ai language model, i cannot",
    "i'm not able to provide",
)


_CLI_OUTPUT_GATE_MAX_LEN = 500


class _ContentGate:
    """One row in the output-gate registry. ``name`` shows up in retry
    logs / exception messages so you can see which gate fired without
    grepping the marker tables. ``check`` returns the matched marker
    when the gate triggers (otherwise None).
    """

    __slots__ = ("name", "description", "_check")

    def __init__(
        self,
        name: str,
        description: str,
        check: Callable[[str], str | None],
    ) -> None:
        self.name = name
        self.description = description
        self._check = check

    def check(self, text: str) -> str | None:
        return self._check(text)


def _short_marker_check(
    markers: tuple[str, ...], max_len: int = _CLI_OUTPUT_GATE_MAX_LEN,
) -> Callable[[str], str | None]:
    """Build a gate-check that fires when ``text`` is short AND contains
    one of the given case-insensitive markers. The length guard is
    what keeps a real paper-length output from tripping these gates
    just by mentioning the marker substring in body text."""

    def _check(text: str) -> str | None:
        if not text or len(text) > max_len:
            return None
        lowered = text.lower()
        for marker in markers:
            if marker in lowered:
                return marker
        return None

    return _check


# Note: there is intentionally no ``empty_response`` gate here even
# though an empty CLI response is degenerate. Some legitimate paths
# return empty (mock-CLI unit tests assert argv shape against an
# empty-stdout fake; the engine itself handles empty-result-json via
# the execute_reflect loop). Adding a transient retry on empty would
# burn 4× retry budget on those paths for no upstream benefit.
_OUTPUT_GATES: tuple[_ContentGate, ...] = (
    _ContentGate(
        name="rate_limit_message",
        description="upstream rate-limit / session-limit text delivered as content",
        check=_short_marker_check(_CLI_RATE_LIMIT_MARKERS),
    ),
    _ContentGate(
        name="placeholder_response",
        description="model emitted a placeholder / give-up pattern",
        check=_short_marker_check(_CLI_PLACEHOLDER_MARKERS),
    ),
    _ContentGate(
        name="refusal",
        description="policy-classifier refusal short-circuited the response",
        check=_short_marker_check(_CLI_REFUSAL_MARKERS),
    ),
)


def _check_output_gates(text: str) -> tuple[str, str] | None:
    """Iterate the gate registry in declaration order. Return
    ``(gate_name, matched_marker)`` for the first gate that fires, or
    None if every gate cleared the output."""
    for gate in _OUTPUT_GATES:
        marker = gate.check(text)
        if marker is not None:
            return gate.name, marker
    return None


def _looks_like_rate_limit_message(text: str) -> str | None:
    """Back-compat shim for callers that only care about the
    rate-limit gate. New callers should use ``_check_output_gates``
    so every gate gets its chance."""
    return _short_marker_check(_CLI_RATE_LIMIT_MARKERS)(text)


def _truncate_prompt_to_fit(prompt: str, max_chars: int) -> str:
    """Trim ``prompt`` to at most ``max_chars`` by cutting the MIDDLE, so the
    task framing (head) and the output-format instructions (tail) — both of
    which FI puts at the ends of every node prompt — survive. The bulky
    context (literature excerpts, large result_json) lives in the middle and
    is what gets dropped. Lossy, but it lets a capped CLI finish instead of
    hard-failing with ``input_too_large``."""
    if len(prompt) <= max_chars:
        return prompt
    dropped = len(prompt) - max_chars
    marker = (
        f"\n\n[... {dropped} characters of context were trimmed here to fit "
        f"this provider's input limit; the task above and the output-format "
        f"instructions below are intact ...]\n\n"
    )
    budget = max_chars - len(marker)
    if budget <= 0:  # absurdly small cap — just hard-cut the head
        return prompt[:max_chars]
    head = int(budget * 0.78)   # keep more of the head (task + early data)
    tail = budget - head        # keep the output-format instructions
    return prompt[:head] + marker + prompt[-tail:]


async def _run_cli(
    spec: _CliSpec,
    prompt: str,
    *,
    model: str = "",
    timeout_s: float = 300.0,
    inactivity_timeout_s: float = 180.0,
    post_eof_reap_timeout_s: float = 60.0,
    heartbeat_cb: Callable[[dict[str, Any]], None] | None = None,
    node: str = "",
) -> str:
    """Spawn the CLI and collect its response. Three output modes:

    * ``output_via="stdout"`` — collect raw stdout until EOF.
    * ``output_via="last_message_file"`` — final answer lands in a temp
      file; stdout is treated as an opaque agent log.
    * ``output_via="stream_json"`` — line-buffered JSON events (Claude
      Code CLI's ``--output-format stream-json``); text deltas are
      aggregated into the return value, thinking deltas are counted but
      discarded.

    Two timeouts protect against stuck children:

    * ``timeout_s`` — hard ceiling on TOTAL wall-clock. Default 300 s
      preserves historical behaviour; per-node overrides in
      ``ProviderConfig.node_cli_timeout_s`` can raise this for
      reasoning-heavy nodes (``implement``, ``execute_reflect``).
    * ``inactivity_timeout_s`` — soft watchdog reset on every stdout
      line or stream event. Default 180 s. This is what tells "model is
      thinking and emitting thinking_delta events" apart from "process
      hung silently". The hard ceiling still applies on top — even a
      well-streaming child must finish within ``timeout_s``.

    ``last_message_file`` mode can't watch stdout for activity (it's
    DEVNULL'd to save memory on long-running CLI logs), so the
    inactivity timer is silently disabled there and only the hard
    ceiling applies. Same for any future spec that points stdout
    elsewhere.

    ``heartbeat_cb`` is called periodically (best-effort) with a dict
    describing progress: ``{kind, elapsed_s, idle_s, text_chars,
    thinking_tokens}``. The Engine wires this into its run.log so the
    user sees ``[implement] still thinking — 2400 thinking deltas,
    elapsed 4m22s`` instead of silent dead air. Errors raised by the
    callback are swallowed; never block the LLM call.
    """
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

    # Cap the prompt for CLIs with a hard per-turn input limit (codex_cli),
    # trimming the middle context so the call goes through instead of the
    # CLI rejecting it with ``input_too_large`` and failing the node.
    if spec.max_input_chars is not None and len(prompt) > spec.max_input_chars:
        orig_len = len(prompt)
        prompt = _truncate_prompt_to_fit(prompt, spec.max_input_chars)
        _log.warning(
            "[provider] %s%s prompt was %d chars, over the %d-char input cap; "
            "trimmed middle context to fit. The model may miss some "
            "literature/result detail — consider a leaner knowledge footprint "
            "(top_k / literature_excerpt_chars) or evidence_gate_max_broaden: 0.",
            spec.argv[0], f" [{node}]" if node else "", orig_len,
            spec.max_input_chars,
        )

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

        if spec.output_via == "stream_json":
            # Streaming path: only used by claude_cli today. Reads
            # stdout line-by-line, parses ``--output-format stream-
            # json`` events, resets the inactivity watchdog on each
            # event, surfaces text_deltas as the aggregated answer.
            # Required for Sonnet 4.6 extended-thinking spans.
            return await _collect_via_streaming(
                proc, argv, spec, stdin_bytes,
                timeout_s=timeout_s,
                inactivity_timeout_s=inactivity_timeout_s,
                post_eof_reap_timeout_s=post_eof_reap_timeout_s,
                heartbeat_cb=heartbeat_cb,
                node=node,
            )
        # Legacy ``communicate()`` path for everything else
        # (codex_cli's ``last_message_file``, plus gemini_cli /
        # copilot_cli plain ``stdout`` mode). Only the total
        # wall-clock budget applies — these CLIs don't emit
        # stream-style progress, so an inactivity timer would either
        # fire false-positives (silent agent log) or do nothing useful
        # (single stdout flush at end). Preserved here so existing
        # tests that mock ``proc.communicate()`` keep working.
        return await _collect_via_communicate(
            proc, argv, spec, stdin_bytes, tmp_out_path, timeout_s,
            heartbeat_cb=heartbeat_cb, node=node,
        )
    finally:
        if tmp_out_path is not None:
            tmp_out_path.unlink(missing_ok=True)


# Cadence of the run.log heartbeat for the non-streaming CLIs (codex/copilot/
# gemini). Kept well under the Engine's 30 s heartbeat throttle so a fresh
# "still waiting" line reliably lands every ~30 s. Module-level so tests can
# shrink it.
_COMMUNICATE_HEARTBEAT_INTERVAL_S = 10.0


async def _collect_via_communicate(
    proc: asyncio.subprocess.Process,
    argv: list[str],
    spec: _CliSpec,
    stdin_bytes: bytes | None,
    tmp_out_path: Path | None,
    timeout_s: float,
    *,
    heartbeat_cb: Callable[[dict[str, Any]], None] | None = None,
    node: str = "",
) -> str:
    """Legacy ``communicate()`` path. Two sub-cases:

    * ``output_via="last_message_file"`` (codex_cli) — stdout is
      DEVNULL because the real answer lives in a temp file. The temp
      file gets read after the child exits.
    * ``output_via="stdout"`` (gemini_cli, copilot_cli) — stdout is
      captured as the raw response; ``output_extractor`` may then
      strip per-CLI envelope/banner lines.

    Only the total wall-clock budget applies. The stream-json
    inactivity-timer path lives in ``_collect_via_streaming`` and is
    used for ``output_via="stream_json"`` (claude_cli) — that one
    distinguishes "model is thinking" from "process is hung" by
    resetting on every stream event.

    These CLIs emit no stream events, so without help they'd be totally
    silent in run.log for the whole call — and the dashboard, which infers
    a quest's status from log recency, shows them as "pending"/idle. A
    background heartbeat ticks ``heartbeat_cb`` every ~10 s with zero
    progress counters (there's genuinely no token-level signal here), which
    the Engine renders as ``[node] still waiting on LLM — no events yet,
    elapsed=Ns`` at its usual 30 s throttle. Purely cosmetic — it can't
    affect the result and is torn down in the finally."""
    _beat_stop = asyncio.Event()
    _beat_task: asyncio.Task[None] | None = None
    if heartbeat_cb is not None:
        _beat_start = time.monotonic()

        async def _beat() -> None:
            while not _beat_stop.is_set():
                try:
                    await asyncio.wait_for(
                        _beat_stop.wait(), timeout=_COMMUNICATE_HEARTBEAT_INTERVAL_S,
                    )
                except asyncio.TimeoutError:
                    el = time.monotonic() - _beat_start
                    try:
                        heartbeat_cb({
                            "kind": "cli_progress", "node": node,
                            "elapsed_s": el, "idle_s": el,
                            "text_chars": 0, "thinking_tokens": 0,
                        })
                    except Exception:
                        pass

        _beat_task = asyncio.ensure_future(_beat())
    try:
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(stdin_bytes), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            elapsed = f"{timeout_s:g}s" if timeout_s >= 1 else f"{timeout_s * 1000:g}ms"
            kill_clean = await _kill_and_reap(proc, spec.argv[0])
            raise _CliTransientError(
                f"{spec.argv[0]} exceeded {elapsed} wall-clock and was killed"
                + ("" if kill_clean else " (post-kill wait timed out)")
            )
        if proc.returncode != 0:
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
        final = content.strip()
        gate_hit = _check_output_gates(final)
        if gate_hit is not None:
            gate_name, marker = gate_hit
            raise _CliTransientError(
                f"{spec.argv[0]} tripped output gate {gate_name!r} (matched: "
                f"{marker!r}). This would have been written to disk as the "
                f"artifact; treating as transient so tenacity retries (which "
                f"may also escalate the model)."
            )
        return final
    finally:
        _beat_stop.set()
        if _beat_task is not None:
            try:
                await _beat_task
            except asyncio.CancelledError:
                pass


async def _collect_via_streaming(
    proc: asyncio.subprocess.Process,
    argv: list[str],
    spec: _CliSpec,
    stdin_bytes: bytes | None,
    *,
    timeout_s: float,
    inactivity_timeout_s: float,
    post_eof_reap_timeout_s: float,
    heartbeat_cb: Callable[[dict[str, Any]], None] | None,
    node: str,
) -> str:
    """Read the child's stdout line-by-line with two independent
    timeouts (total + inactivity) and emit periodic heartbeats.

    Distinguishes "model is thinking — events flowing, just no text
    yet" from "process is hung — no events at all" by resetting the
    inactivity timer on every line, regardless of whether it parsed
    as a useful event.
    """
    assert proc.stdout is not None
    start = time.monotonic()
    last_activity = start
    stop = asyncio.Event()
    aggregated: list[str] = []
    # Running counter so the heartbeat doesn't recompute
    # ``sum(len(s) for s in aggregated)`` on every 1-s tick (that
    # would be O(N²) in stream length). Always updated in lock-step
    # with ``aggregated.append`` so the two never drift.
    text_chars_total = 0
    thinking_token_count = 0
    error_message: str | None = None  # populated from stream_json error events
    result_envelope_seen = False  # see _parse_stream_json_line caller below

    async def write_stdin() -> None:
        if stdin_bytes is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(stdin_bytes)
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            # Child exited before consuming stdin — the stdout reader
            # will surface the real error (rc != 0 with stderr tail).
            pass

    async def read_stdout() -> None:
        nonlocal last_activity, thinking_token_count, error_message
        nonlocal text_chars_total, result_envelope_seen
        while True:
            try:
                line = await proc.stdout.readline()
            except Exception:
                break
            if not line:
                return  # EOF
            last_activity = time.monotonic()
            if spec.output_via == "stream_json":
                text_delta, thinking_inc, err, is_result = (
                    _parse_stream_json_line(line)
                )
                if text_delta:
                    # The CLI sometimes emits a final ``result`` envelope
                    # carrying the full assembled text AFTER streaming
                    # all the ``text_delta`` chunks. If we already
                    # collected the chunks, the result envelope would
                    # duplicate them. Skip the envelope's body when we
                    # already have streamed text.
                    if is_result and aggregated:
                        result_envelope_seen = True
                    else:
                        aggregated.append(text_delta)
                        text_chars_total += len(text_delta)
                        if is_result:
                            result_envelope_seen = True
                thinking_token_count += thinking_inc
                if err is not None and error_message is None:
                    error_message = err
            else:
                decoded = line.decode("utf-8", errors="replace")
                aggregated.append(decoded)
                text_chars_total += len(decoded)

    async def watchdog() -> None:
        while not stop.is_set():
            await asyncio.sleep(1.0)
            now = time.monotonic()
            elapsed = now - start
            idle = now - last_activity
            if elapsed > timeout_s:
                raise _CliTransientError(
                    f"{spec.argv[0]} exceeded {timeout_s:g}s total wall-clock "
                    f"and was killed (last activity {idle:.0f}s ago, "
                    f"thinking_tokens={thinking_token_count})"
                )
            if idle > inactivity_timeout_s:
                raise _CliTransientError(
                    f"{spec.argv[0]} silent for {idle:.0f}s "
                    f"(inactivity threshold {inactivity_timeout_s:g}s); "
                    f"thinking_tokens={thinking_token_count}, "
                    f"text_chars={text_chars_total}"
                )
            # Best-effort heartbeat. Cadence: only when the caller asked
            # for one. Errors swallowed; never block the LLM call.
            if heartbeat_cb is not None:
                try:
                    heartbeat_cb({
                        "kind": "cli_progress",
                        "node": node,
                        "elapsed_s": elapsed,
                        "idle_s": idle,
                        "text_chars": text_chars_total,
                        "thinking_tokens": thinking_token_count,
                    })
                except Exception:
                    pass

    writer_task = asyncio.create_task(write_stdin())
    reader_task = asyncio.create_task(read_stdout())
    watchdog_task = asyncio.create_task(watchdog())
    watchdog_raised: BaseException | None = None
    try:
        # Race: either the reader finishes (EOF) or the watchdog
        # raises. ``return_when=FIRST_COMPLETED`` lets us catch either.
        done, _pending = await asyncio.wait(
            {reader_task, watchdog_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in done:
            if t.exception() is not None:
                # Save the watchdog's _CliTransientError so we can
                # raise it AFTER we've reaped the child — otherwise
                # the watchdog timeout escapes ``_collect_via_streaming``
                # with the child still running.
                watchdog_raised = t.exception()
    finally:
        stop.set()
        for t in (writer_task, reader_task, watchdog_task):
            if not t.done():
                t.cancel()
        # Reap any cancellations cleanly.
        for t in (writer_task, reader_task, watchdog_task):
            try:
                await t
            except (asyncio.CancelledError, _CliTransientError, Exception):
                pass
        # If the watchdog tripped a timeout, the child is still alive
        # (the watchdog's raise pre-empted any reap path). Kill it
        # before we propagate the error to the caller so it doesn't
        # keep burning CPU + LLM tokens in the background. Done
        # inside ``finally`` so even if the caller cancels us mid-
        # watchdog the kill still runs.
        if watchdog_raised is not None:
            await _kill_and_reap(proc, spec.argv[0])
    if watchdog_raised is not None:
        raise watchdog_raised
    # Reader finished (EOF). Wait for the child to exit and check rc.
    # Important: if we already collected text_deltas before EOF, the
    # LLM call SUCCEEDED — claude.exe streamed its response and then
    # closed stdout. The child handle lingering past a few seconds on
    # Windows (proactor proc cleanup latency) is NOT a reason to throw
    # away ~8 minutes of generation. Two-stage policy:
    #
    #   1. Wait up to 60 s for the OS to fully reap (generous because
    #      this is post-EOF, the heavy work is done; we're just paying
    #      cleanup latency).
    #   2. On reap-timeout: if we got text, RETURN IT (with a warning).
    #      Only kill+raise when there's literally nothing to return.
    #
    # This fixes a regression where a successful 8-minute body call
    # got discarded because Windows took 11 s to mark claude.exe
    # exited — tenacity then re-ran the whole call from scratch.
    have_output = text_chars_total > 0 or bool(aggregated)
    try:
        rc = await asyncio.wait_for(proc.wait(), timeout=post_eof_reap_timeout_s)
    except asyncio.TimeoutError:
        if have_output:
            # Don't waste the LLM result. Kill the lingering process
            # (best-effort) and continue with what we collected.
            _log.warning(
                "%s stdout closed and reap timed out, but the call "
                "produced %d text chars — returning result anyway "
                "(killing lingering child best-effort)",
                spec.argv[0], text_chars_total,
            )
            await _kill_and_reap(proc, spec.argv[0])
            # Skip the rc-based error check below — we never got rc.
            # An empty error_message and have_output=True means good.
            if error_message is not None:
                raise _CliTransientError(
                    f"{spec.argv[0]} stream error: {error_message[:500]}"
                )
            return _finalise_stream_content(aggregated, spec)
        # No output AND no exit — genuinely stuck.
        await _kill_and_reap(proc, spec.argv[0])
        stderr_b = b""
        if proc.stderr is not None:
            try:
                stderr_b = await asyncio.wait_for(proc.stderr.read(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        raise _CliWedgeError(
            f"{spec.argv[0]} stdout closed but child didn't exit within "
            f"{post_eof_reap_timeout_s:g}s (no output collected) — an "
            f"extended-thinking CLI hang, not a transient blip. Retrying wedges "
            f"the same way; the fix is a different provider for this node (e.g. "
            f"resume with `--config <codex_or_openai>.yaml`). stderr tail: "
            f"{stderr_b.decode('utf-8', 'replace')[-500:]}"
        )
    if rc != 0:
        # Even on non-zero exit, prefer the collected text over a
        # retry IF we got substantial output. Some CLIs emit a clean
        # stream then exit non-zero on a benign cleanup error
        # (atexit hooks, signal handler, etc.) — throwing the
        # response away there is strictly worse than logging the rc
        # and returning.
        if have_output:
            stderr_b = b""
            if proc.stderr is not None:
                try:
                    stderr_b = await asyncio.wait_for(proc.stderr.read(), timeout=2)
                except asyncio.TimeoutError:
                    pass
            _log.warning(
                "%s exited rc=%d but emitted %d text chars — returning "
                "result. stderr tail: %s",
                spec.argv[0], rc, text_chars_total,
                stderr_b.decode("utf-8", "replace")[-300:],
            )
            if error_message is not None:
                raise _CliTransientError(
                    f"{spec.argv[0]} stream error: {error_message[:500]}"
                )
            return _finalise_stream_content(aggregated, spec)
        stderr_b = b""
        if proc.stderr is not None:
            try:
                stderr_b = await asyncio.wait_for(proc.stderr.read(), timeout=2)
            except asyncio.TimeoutError:
                pass
        raise _CliTransientError(
            f"{argv[0]} exited rc={rc}: "
            f"{stderr_b.decode('utf-8', 'replace')[-500:]}"
        )
    if error_message is not None:
        # stream-json carried a fatal error event but the process exited
        # 0 anyway (claude_cli does this for some error classes). Treat
        # as transient so tenacity retries.
        raise _CliTransientError(
            f"{spec.argv[0]} stream error: {error_message[:500]}"
        )
    return _finalise_stream_content(aggregated, spec)


def _finalise_stream_content(aggregated: list[str], spec: _CliSpec) -> str:
    """Concatenate streamed text, apply per-spec extractor, and
    apply the rate-limit-pattern guard. Centralised so all three
    streaming return paths get the same protection — without this,
    the OPC quest's ``paper.md`` became literally the string
    ``"You've hit your session limit · resets 2:30am ..."`` because
    claude_cli delivered the rate-limit message as ``text_delta``
    events instead of an error envelope, and the engine wrote it to
    disk as if it were the assistant's answer."""
    content = "".join(aggregated)
    if spec.output_extractor is not None and spec.output_via != "stream_json":
        content = spec.output_extractor(content)
    final = content.strip()
    gate_hit = _check_output_gates(final)
    if gate_hit is not None:
        gate_name, marker = gate_hit
        raise _CliTransientError(
            f"{spec.argv[0]} tripped output gate {gate_name!r} (matched: "
            f"{marker!r}). This would have been written to disk as the "
            f"artifact; treating as transient so tenacity retries (and "
            f"may escalate the model)."
        )
    return final


async def _kill_and_reap(proc: asyncio.subprocess.Process, name: str) -> bool:
    """Kill the child and wait up to 5 s for the OS to reap it. Returns
    True iff the wait completed cleanly. Logs a warning otherwise — a
    leaked child becomes a zombie on POSIX or holds an OS handle on
    Windows until the parent exits."""
    try:
        proc.kill()
    except ProcessLookupError:
        return True  # already exited
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
        return True
    except asyncio.TimeoutError:
        _log.warning(
            "CLI %s did not reap within 5s after SIGKILL; "
            "process may be wedged in uninterruptible state",
            name,
        )
        return False


def _parse_stream_json_line(raw: bytes) -> tuple[str, int, str | None, bool]:
    """Parse one line of Claude CLI's ``--output-format stream-json``
    output. Returns
    ``(text_delta, thinking_delta_token_count, error, is_result_envelope)``.

    - ``text_delta`` is the new assistant-visible text (empty for
      thinking-only events).
    - ``thinking_delta_token_count`` is a count, NOT the thinking text
      itself; we don't keep the thinking-text body because it's both
      large and not useful to downstream parsers.
    - ``error`` is a fatal-error message extracted from
      ``{"type":"error",...}`` events, else None.
    - ``is_result_envelope`` is True when this line came from a
      ``{"type":"result", "result":"<full text>"}`` envelope. Callers
      use it to deduplicate: the CLI sometimes emits BOTH a stream of
      ``text_delta`` events AND a final result envelope carrying the
      same content; appending both would double the answer.

    Tolerates non-JSON lines (the CLI sometimes emits status lines
    before stream-json events fully start) by returning all-empties.
    """
    try:
        msg = json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return "", 0, None, False
    if not isinstance(msg, dict):
        return "", 0, None, False
    mtype = msg.get("type")
    # Stream event wrapper — actual content is nested under .event.
    if mtype == "stream_event":
        event = msg.get("event") or {}
        if not isinstance(event, dict):
            return "", 0, None, False
        if event.get("type") == "content_block_delta":
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                return str(delta.get("text", "")), 0, None, False
            if dtype == "thinking_delta":
                thinking = str(delta.get("thinking", ""))
                # Rough token estimate: 4 chars/token. Used only for
                # heartbeat progress, not billing.
                return "", max(1, len(thinking) // 4), None, False
        return "", 0, None, False
    # Some events carry the final assembled text in a different shape
    # (e.g. ``"type":"result"`` with ``"result":"<text>"``). Capture
    # it as a fallback so we don't return empty on quests that emit
    # the result event instead of streaming deltas. The ``is_result``
    # flag lets the caller skip appending when streamed deltas already
    # supplied the same content.
    if mtype == "result":
        result = msg.get("result")
        if isinstance(result, str):
            return result, 0, None, True
    if mtype == "error":
        err = msg.get("error") or msg.get("message") or "unknown stream error"
        return "", 0, str(err), False
    return "", 0, None, False


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
        # The FI VSCode extension is the parent process for chat-spawned
        # quests; it passes ``--vscode-bridge-port N`` and the port
        # lives in ``provider.extra["bridge_port"]``.
        #
        # For --serve / --tools subprocesses we don't have the
        # extension as a parent — instead they connect to the
        # extension's session-long PersistentBridge over a per-user
        # OS-managed IPC channel (Unix socket on POSIX, named pipe on
        # Windows). The path arrives via ``provider.extra["bridge_socket"]``
        # (``launch.py`` populates it from ``--vscode-bridge-socket``
        # or the canonical per-user default).
        socket_path = provider.extra.get("bridge_socket") or ""
        port = int(provider.extra.get("bridge_port", 0))
        if not socket_path and port <= 0:
            raise RuntimeError(
                "vscode_extension provider requires either "
                "extra['bridge_socket'] (--vscode-bridge-socket; what "
                "--serve / --tools use) or extra['bridge_port'] "
                "(--vscode-bridge-port; what the VSCode chat-spawn "
                "path uses). Are you launching FI from outside the "
                "extension's reach? Use copilot_cli for headless "
                "Copilot runs."
            )
        return ResolvedEndpoint(
            base_url="",
            model=provider.model or "(VSCode chat default)",
            api_key=_NO_KEY_SENTINEL,
            transport="vscode_bridge",
            provider_name=name,
            vscode_bridge_port=port,
            vscode_bridge_socket=socket_path,
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
        cli_inactivity_timeout_s: float | None = 180.0,
        node_cli_timeout_s: dict[str, float] | None = None,
        node_http_timeout_s: dict[str, float] | None = None,
        node_model_fallbacks: dict[str, str] | None = None,
        max_prompt_chars: int = 0,
        heartbeat_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=timeout_s)
        # Base HTTP read-timeout (seconds) and per-node overrides for the
        # OpenAI-compatible transport. Reasoning-heavy nodes (implement/write)
        # routinely outrun a flat 120 s on slow/local models; the per-node map
        # gives them headroom while cheap nodes keep the tight base that
        # catches a hung server fast. Lookup misses fall back to the base.
        self._http_timeout_s = timeout_s
        self._node_http_timeout_s = node_http_timeout_s or {}
        # Last-resort prompt-size guard (total characters across messages).
        # 0 disables. Prevents a runaway prompt from blowing the context
        # window into a hard 400 / stall. See ``_trim_messages``.
        self._max_prompt_chars = max(0, int(max_prompt_chars))
        # CLI providers (claude_cli/codex_cli/copilot_cli/gemini_cli)
        # use this wall-clock cap per chat call; a stuck child gets
        # killed and tenacity retries. Defaults to 5 minutes — longer
        # than the typical 10–90 s per call but bounded so concurrent
        # fleet contention can't hang the whole quest indefinitely.
        self._cli_timeout_s = cli_timeout_s
        # Inactivity-watchdog soft timeout. None disables and lets only
        # the total ceiling apply (compat shim for callers that haven't
        # opted into streaming yet). See ProviderConfig docstring.
        self._cli_inactivity_timeout_s = cli_inactivity_timeout_s
        # Per-node ``cli_timeout_s`` overrides; map node-name → seconds.
        # Lookup misses fall back to ``self._cli_timeout_s``. Used by
        # ``_chat_cli`` when ``node`` is passed in.
        self._node_cli_timeout_s = node_cli_timeout_s or {}
        # Per-node model escalation map. Used by ``_chat_cli`` on
        # retry 2+ when the primary model failed transiently. The
        # primary motivation is the empirically-observed Sonnet 4.6
        # paralysis-thinking on long code-gen prompts: extended
        # thinking spins forever without producing any text. Escaping
        # to Opus 4.7 on retry consistently lands. ``node_models``
        # (above) sets the FIRST-ATTEMPT model; this dict sets the
        # SECOND-ATTEMPT-AND-LATER model.
        self._node_model_fallbacks = node_model_fallbacks or {}
        # Optional periodic progress callback (Engine wires this to its
        # run.log heartbeat logger). Receives a dict each ~1 s during
        # CLI streaming; see ``_run_cli`` for the payload shape.
        self._heartbeat_cb = heartbeat_cb
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
            # Gate the dispatch on the process-wide concurrency cap
            # (``FI_MAX_CONCURRENT_LLM_CALLS``; no-op when unset) so a fleet of
            # quests can't burst past provider rate limits. The slot is held
            # only for this one dispatch and released on return, so retries /
            # fallbacks re-queue rather than deadlock.
            async with _llm_call_slot():
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

    def _trim_messages(
        self, messages: list[dict[str, str]], *, node: str = "",
    ) -> list[dict[str, str]]:
        """Last-resort guard against a runaway prompt. When ``max_prompt_chars``
        is set and the total message size exceeds it, truncate the single
        largest message in the MIDDLE (keeping head + tail, where the task
        framing and the trailing instruction usually live) so the prompt fits.
        No-op when the guard is disabled (0) or the prompt already fits — so
        normal calls pay only one ``sum`` and are otherwise untouched. Returns a
        new list; never mutates the caller's messages."""
        cap = self._max_prompt_chars
        if cap <= 0:
            return messages
        total = sum(len(str(m.get("content", ""))) for m in messages)
        if total <= cap:
            return messages
        trimmed = [dict(m) for m in messages]
        idx = max(
            range(len(trimmed)),
            key=lambda i: len(str(trimmed[i].get("content", ""))),
        )
        content = str(trimmed[idx].get("content", ""))
        overflow = total - cap
        marker = (
            f"\n\n... [FI: {overflow} chars trimmed to fit the "
            f"{cap}-char prompt cap] ...\n\n"
        )
        keep = len(content) - overflow - len(marker)
        if keep <= 0:
            # The largest message alone can't absorb the overflow — hard-cap it.
            new_content = content[: max(0, cap - len(marker))] + marker
        else:
            head = keep // 2
            tail = keep - head
            new_content = content[:head] + marker + (
                content[-tail:] if tail else ""
            )
        trimmed[idx]["content"] = new_content
        _log.warning(
            "[prompt-trim] node=%s prompt %d chars > cap %d — trimmed message "
            "%d to %d chars (dropped ~%d)",
            node or "?", total, cap, idx, len(new_content), overflow,
        )
        return trimmed

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
        # Last-resort prompt-size guard (all transports) — a runaway prompt
        # otherwise blows the context window into a hard 400 / stall.
        messages = self._trim_messages(messages, node=node)
        if self.endpoint.transport == "cli":
            text = await self._chat_cli(
                messages, model_override=model, node=node,
            )
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
            # Jittered backoff: a deterministic ``wait_exponential``
            # synchronises N concurrent quests in a --fleet that all
            # hit a transient upstream blip simultaneously, so they
            # all retry at the same wall-clock instant and re-clog
            # the upstream. Random-exponential spreads them out over
            # a window so the upstream sees a gentler ramp.
            wait=wait_random_exponential(multiplier=1, max=20),
            retry=retry_if_exception(_retry_http_error),
            reraise=True,
        ):
            with attempt:
                # Per-node HTTP read-timeout: heavy nodes (implement/write) get
                # headroom, cheap nodes keep the tight base that catches a hung
                # server fast. Miss → base client timeout.
                http_timeout = self._node_http_timeout_s.get(
                    node, self._http_timeout_s,
                )
                r = await self._http.post(
                    url, json=body, headers=headers, timeout=http_timeout,
                )
                # Raise for any error status; the retry predicate
                # (_retry_http_error) retries only 5xx / 429, letting a 4xx
                # (bad key, quota, content policy, oversized body) surface
                # immediately instead of burning the backoff budget.
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
            # Prefer the IPC socket when available (--serve / --tools
            # path), fall back to the TCP port (chat-spawned per-command
            # bridges still use TCP). The client picks the transport
            # based on which kwarg is non-empty.
            self._bridge = VSCodeBridgeClient(
                host="127.0.0.1",
                port=self.endpoint.vscode_bridge_port,
                socket_path=(self.endpoint.vscode_bridge_socket or None),
                # Wall-clock cap per chat call so a TS-side silent stall
                # (Copilot HTTP/2 hang the inactivity timer missed, or
                # an older .vsix without the timer at all) can't wedge
                # the engine forever on ``await fut``. The bridge
                # raises ``BridgeError("bridge stalled ...")`` on
                # timeout; ``_TRANSIENT_BRIDGE_MARKERS`` recognises it
                # and tenacity below retries the call.
                cli_timeout_s=self._cli_timeout_s,
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

        # Budget: 6 attempts. Each inter-attempt wait is drawn
        # uniformly from ``[0, 2 · 2^attempt]`` seconds, capped at 60
        # — typical expected total ~2 minutes of cumulative backoff,
        # spread randomly so concurrent fleet clients don't synchronise
        # on a shared upstream blip. Sustained Copilot HTTP/2 outages
        # have been observed lasting 30–90 s; the prior 3-attempt /
        # ~14 s budget was too tight and crashed quests on transient
        # upstream issues. The TS-side bridge also retries 4×, so
        # total wall time before a real failure exceeds 2 minutes.
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(6),
                # Jittered backoff (see HTTP path comment): a
                # deterministic schedule synchronises concurrent
                # bridge clients in a --fleet on a shared upstream
                # blip. Random spreads them over the window.
                wait=wait_random_exponential(multiplier=2, max=60),
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
        node: str = "",
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

        ``node`` is the FI engine node name; when set, looks up
        ``node_cli_timeout_s[node]`` for the per-call wall-clock budget
        so reasoning-heavy nodes (implement, execute_reflect) can get
        longer ceilings than fast ones (clarify, ideate).

        Model escalation on retry: on retry attempt 2+, look up
        ``node_model_fallbacks[node]`` and switch to that model. The
        primary motivation is empirically-observed Sonnet 4.6 paralysis
        on long code-gen prompts (extended-thinking spins forever
        without producing text); reproducible escape is to escalate to
        Opus 4.7 on retry. The escalation is per-call and per-node so
        the user's primary model preference is honoured on first try.
        """
        spec = self.endpoint.cli_spec
        if spec is None:  # pragma: no cover — guarded by transport check
            raise RuntimeError("transport=cli but no cli_spec set")
        prompt = _messages_to_text(messages)

        # Pick the per-call total-timeout: node-specific override wins,
        # else the client-level default.
        effective_total_timeout = self._node_cli_timeout_s.get(
            node, self._cli_timeout_s,
        )
        # Effective inactivity timeout. None means "disabled, only the
        # total ceiling applies" — preserved for compat / tests.
        effective_inactivity = (
            self._cli_inactivity_timeout_s
            if self._cli_inactivity_timeout_s is not None
            else effective_total_timeout
        )

        # First-attempt model: per-call override > endpoint default.
        primary_model = (
            model_override if model_override is not None
            else self.endpoint.cli_model_override
        )
        # On retry, escalate to a different model when configured.
        fallback_model = self._node_model_fallbacks.get(node, "") if node else ""

        # Diagnostic visibility: when tenacity catches an exception
        # and decides to retry, the exception text used to be swallowed
        # (the caller saw nothing in run.log until all 4 attempts
        # exhausted). This callback logs each caught exception + which
        # model the next attempt will use, so silent retries become
        # debuggable. The actual log_warning is closed over in the
        # ``_log`` reference below.
        def _retry_log(rs: Any) -> None:
            try:
                exc = rs.outcome.exception() if rs.outcome else None
                next_attempt = rs.attempt_number + 1
                next_model = (
                    fallback_model if (fallback_model and next_attempt >= 2)
                    else primary_model
                )
                _log.warning(
                    "[%s] CLI call attempt %d failed (%s); retrying "
                    "(attempt %d of 4, model=%s)",
                    node or spec.argv[0], rs.attempt_number,
                    repr(exc)[:300] if exc else "no exception captured",
                    next_attempt, next_model or "<cli default>",
                )
            except Exception:
                pass  # never block retries on logging errors

        async for attempt in AsyncRetrying(
            # Normal transients get 4 attempts; a repeated WEDGE bails after 2
            # (an unrecoverable hang — retrying just wedges again, ~2 wasted
            # multi-minute calls) and reraises with switch-provider guidance.
            stop=stop_after_attempt(4) | _stop_on_repeated_wedge,
            # Jittered backoff so concurrent --fleet quests don't all
            # retry the same upstream at the same instant — see
            # the HTTP path's note. wait_random_exponential picks a
            # uniform random value from [0, multiplier * 2^attempt],
            # capped at ``max``, which spreads retries over a window.
            wait=wait_random_exponential(multiplier=1, max=20),
            retry=retry_if_exception(_retry_cli_error),
            reraise=True,
            before_sleep=_retry_log,
        ):
            with attempt:
                # Escalate to fallback model on retry 2+ if configured.
                # The first attempt always uses primary_model so the
                # user's stated preference is tried first.
                attempt_no = attempt.retry_state.attempt_number
                effective_model = (
                    fallback_model
                    if (attempt_no >= 2 and fallback_model)
                    else primary_model
                )
                return await _run_cli(
                    spec, prompt,
                    model=effective_model,
                    timeout_s=effective_total_timeout,
                    inactivity_timeout_s=effective_inactivity,
                    heartbeat_cb=self._heartbeat_cb,
                    node=node,
                )
        raise RuntimeError("unreachable: tenacity reraise=True must raise on exhaustion")


# ---------------------------------------------------------------------------
# Provider fallback chain + per-provider circuit breaker
# ---------------------------------------------------------------------------


def _is_fatal_provider_error(exc: BaseException) -> bool:
    """A provider failure that won't clear within the run — auth/quota/credit
    exhaustion — so the breaker should open immediately rather than after the
    usual failure threshold. Transient blips (5xx, timeouts) trip only after
    repeated failures."""
    if isinstance(exc, _CliTransientError):
        return any(m in str(exc).lower() for m in _CLI_FATAL_MARKERS)
    if isinstance(exc, httpx.HTTPStatusError):
        sc = getattr(getattr(exc, "response", None), "status_code", None)
        return isinstance(sc, int) and sc in (401, 403, 402)
    return False


@dataclass
class _FallbackSlot:
    """One rung of the fallback chain: a provider plus its breaker state."""

    label: str
    # ``None`` for the primary (already constructed); an async factory for
    # lazily-built fallbacks (their proxy, if any, spins up only on first use).
    factory: "Callable[[], Any] | None"
    client: "LLMClient | None" = None
    failures: int = 0
    tripped: bool = False  # circuit open — skipped for the rest of the run

    def record_failure(self, threshold: int, *, fatal: bool) -> None:
        self.failures += 1
        if fatal or self.failures >= threshold:
            self.tripped = True


class FallbackLLMClient:
    """Try an ordered chain of providers so a single provider's outage (rate
    limit, auth failure, proxy crash) doesn't forfeit the whole quest.

    ``chat`` calls the primary; if it terminally fails (after the primary's own
    in-provider tenacity retries), the call moves to the next provider, and so
    on. Each provider has a **circuit breaker**: after ``breaker_threshold``
    failures — or a single auth/quota error — its circuit opens and it is
    skipped for the rest of the run, so subsequent calls jump straight to a
    live provider instead of re-burning the dead one's retry budget on every
    call. Fallbacks are built lazily (nothing is resolved or spawned until the
    primary actually fails), and the provider names of any fallbacks that were
    materialised are recorded in :attr:`built_fallback_providers` so the caller
    can release their proxies on shutdown.

    Presents the read surface the Engine uses on a plain ``LLMClient``
    (``chat`` / ``last_model`` / ``last_usage`` / ``aclose``); ``last_model``
    and ``last_usage`` reflect whichever provider served the most recent call.
    """

    def __init__(
        self,
        primary: "LLMClient",
        factories: "list[tuple[str, Callable[[], Any]]]" = (),
        *,
        breaker_threshold: int = 2,
        log: "logging.Logger | None" = None,
    ) -> None:
        self._log = log or logging.getLogger("fi.provider.fallback")
        primary_label = getattr(
            getattr(primary, "endpoint", None), "provider_name", None,
        ) or "primary"
        self._slots: list[_FallbackSlot] = [
            _FallbackSlot(label=primary_label, factory=None, client=primary)
        ]
        for name, factory in factories:
            self._slots.append(_FallbackSlot(label=name, factory=factory))
        self._threshold = max(1, int(breaker_threshold))
        # Read by the Engine cost logger immediately after each chat().
        self.last_usage: Any = None
        self.last_model: str | None = None
        # Fallback providers actually materialised (for proxy release on close).
        self.built_fallback_providers: list[str] = []

    async def _client_for(self, slot: _FallbackSlot) -> "LLMClient":
        if slot.client is None:
            assert slot.factory is not None
            slot.client = await slot.factory()
            self.built_fallback_providers.append(slot.label)
            self._log.info("[fallback] initialised provider %s", slot.label)
        return slot.client

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        errors: list[tuple[str, BaseException]] = []
        for idx, slot in enumerate(self._slots):
            if slot.tripped:
                continue
            try:
                client = await self._client_for(slot)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # building/resolving the fallback failed
                slot.record_failure(self._threshold, fatal=True)
                self._log.warning(
                    "[fallback] could not initialise %s: %r", slot.label, e,
                )
                errors.append((slot.label, e))
                continue
            try:
                text = await client.chat(messages, **kwargs)
            except asyncio.CancelledError:
                # Cancellation is not a provider failure — never fall back on it.
                raise
            except Exception as e:
                fatal = _is_fatal_provider_error(e)
                slot.record_failure(self._threshold, fatal=fatal)
                more = idx < len(self._slots) - 1
                self._log.warning(
                    "[fallback] provider %s failed (%s)%s; %s",
                    slot.label, type(e).__name__,
                    " [circuit opened]" if slot.tripped else "",
                    "trying next provider" if more else "no more providers",
                )
                errors.append((slot.label, e))
                continue
            # Success — snapshot the serving provider's cost fields.
            slot.failures = 0
            self.last_usage = getattr(client, "last_usage", None)
            self.last_model = getattr(client, "last_model", None)
            if idx > 0:
                self._log.info(
                    "[fallback] request served by %s (primary unavailable)",
                    slot.label,
                )
            return text
        # Every provider failed on this call, or all circuits are already open.
        if errors:
            _, last_err = errors[-1]
            try:
                chain = ", ".join(lbl for lbl, _ in errors)
                last_err.add_note(f"[FI] all providers exhausted: {chain}")
            except Exception:  # pragma: no cover — needs Py<3.11
                pass
            raise last_err
        raise RuntimeError(
            "no LLM providers available: every circuit is open "
            f"({', '.join(s.label for s in self._slots)})"
        )

    async def aclose(self) -> None:
        for slot in self._slots:
            if slot.client is not None:
                try:
                    await slot.client.aclose()
                except Exception:  # pragma: no cover — best-effort cleanup
                    pass
