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
    user's `codex auth login` ChatGPT Plus/Pro OAuth.
  - `claude_cli` — `claude --print --output-format text` reusing the
    user's `claude login` Claude Pro/Max OAuth (no `ANTHROPIC_API_KEY`
    needed; OAuth from the CLI's keychain is honored).

Proxy spawn details:
* `claude_code` — `poetry run python main.py <port>` from
  `FI_CLAUDE_CODE_WRAPPER_DIR` (defaults to `~/claude-code-openai-wrapper`).
* `github_copilot_cli` / `github_copilot_vscode` —
  `npx copilot-api@latest start --port <N> --rate-limit 60 --wait`.
  Both names route to the same proxy.

`ProxySupervisor` is reference-counted so a fleet of N quests using the
same proxy provider shares one proxy process — see Phase H.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import ProviderConfig

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

_PROXY_PROVIDERS = {"claude_code", "github_copilot_cli", "github_copilot_vscode"}


@dataclass(frozen=True)
class _CliSpec:
    """How to invoke a local CLI as a chat endpoint."""

    argv: tuple[str, ...]            # base command; prompt may be appended
    pass_prompt_via: str             # "stdin" | "arg"
    output_via: str                  # "stdout" | "last_message_file"


_CLI_SPECS: dict[str, _CliSpec] = {
    "claude_cli": _CliSpec(
        # `claude --print` prints the response to stdout and exits. Output
        # format "text" keeps it raw; "json" wraps in a JSON envelope. We
        # use "text" and let the engine's `_parse_json_lenient` cope with
        # whatever the model produces.
        argv=("claude", "--print", "--output-format", "text"),
        pass_prompt_via="stdin",
        output_via="stdout",
    ),
    "codex_cli": _CliSpec(
        # `codex exec` runs Codex non-interactively. stdout is the agent
        # log (token counts, tool calls); the final assistant message is
        # written to the file passed via --output-last-message.
        argv=("codex", "exec"),
        pass_prompt_via="arg",
        output_via="last_message_file",
    ),
}
_CLI_PROVIDERS = frozenset(_CLI_SPECS)

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
    transport: str = "http"          # "http" | "cli"
    cli_spec: _CliSpec | None = None  # set when transport == "cli"


@dataclass
class _ProxyHandle:
    name: str
    port: int
    proc: subprocess.Popen[bytes]
    refcount: int = 0


@dataclass
class ProxySupervisor:
    """Reference-counted lifecycle for proxy subprocesses.

    Phase A leaves the spawn paths as `NotImplementedError`; Phase C fills
    them in with the actual `claude-code-openai-wrapper` and `copilot-api`
    invocations.
    """

    _handles: dict[str, _ProxyHandle] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self, provider_name: str) -> _ProxyHandle:
        if provider_name not in _PROXY_PROVIDERS:
            raise ValueError(f"{provider_name!r} is not a proxy provider")
        async with self._lock:
            handle = self._handles.get(provider_name)
            if handle is None:
                # `_spawn` ends with a blocking poll of `/v1/models` (up to
                # 60s). Run it in a worker thread so concurrent quests doing
                # other work don't stall the event loop while one of them
                # waits for the proxy to come up.
                handle = await asyncio.to_thread(self._spawn, provider_name)
                self._handles[provider_name] = handle
            handle.refcount += 1
            return handle

    async def release(self, provider_name: str) -> None:
        async with self._lock:
            handle = self._handles.get(provider_name)
            if handle is None:
                return
            handle.refcount -= 1
            if handle.refcount <= 0:
                handle.proc.terminate()
                try:
                    handle.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    handle.proc.kill()
                self._handles.pop(provider_name, None)

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


async def _run_cli(spec: _CliSpec, prompt: str) -> str:
    argv = list(spec.argv)
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

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"CLI provider binary {argv[0]!r} not found on PATH. "
            f"Install and log in (`claude login` or `codex login`) before using this provider."
        ) from e

    stdout_b, stderr_b = await proc.communicate(stdin_bytes)
    if proc.returncode != 0:
        # Treat as transient (retryable). Auth failures usually appear here
        # too, but in practice OAuth refresh handles them — if a failure
        # persists across 4 retries the error surfaces.
        raise _CliTransientError(
            f"{argv[0]} exited rc={proc.returncode}: "
            f"{stderr_b.decode('utf-8', 'replace')[-500:]}"
        )

    if spec.output_via == "last_message_file":
        assert tmp_out_path is not None
        try:
            content = tmp_out_path.read_text(encoding="utf-8", errors="replace")
        finally:
            tmp_out_path.unlink(missing_ok=True)
    else:
        content = stdout_b.decode("utf-8", errors="replace")
    return content.strip()


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
    if name in _CLI_PROVIDERS:
        # CLI providers exec a local binary per chat call. No URL, no key
        # (the CLI uses its own OAuth keychain). `model` carries the
        # provider name so the LLMClient can look up the spec.
        return ResolvedEndpoint(
            base_url="",
            model=provider.model or name,
            api_key=_NO_KEY_SENTINEL,
            transport="cli",
            cli_spec=_CLI_SPECS[name],
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
    )


class LLMClient:
    """Thin async wrapper that speaks OpenAI Chat Completions.

    Built on `httpx.AsyncClient` rather than the openai SDK directly so
    multiple Engines in one process can share a single connection pool
    cleanly. The request shape is OpenAI-standard.
    """

    def __init__(
        self,
        endpoint: ResolvedEndpoint,
        *,
        http: httpx.AsyncClient | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.endpoint = endpoint
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        if self.endpoint.transport == "cli":
            return await self._chat_cli(messages)
        body: dict[str, Any] = {
            "model": self.endpoint.model,
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
        return data["choices"][0]["message"]["content"]

    async def _chat_cli(self, messages: list[dict[str, str]]) -> str:
        """Exec a local CLI binary with the prompt and return its output.

        Flattens the OpenAI-style `messages` list into a single text prompt
        (most LLM CLIs don't have a separate system/user channel — they
        accept one block of text). Retries on non-zero exit with the same
        exponential backoff as the HTTP path, but only on the broad
        `OSError` family — auth/quota failures raise as `RuntimeError` and
        are NOT retried.
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
                return await _run_cli(spec, prompt)
        raise RuntimeError("unreachable: tenacity reraise=True must raise on exhaustion")
