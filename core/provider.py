"""Unified async LLM client + proxy supervisor.

Every provider eventually presents an OpenAI Chat Completions interface.
Direct providers (openai, codex, gemini, ollama, vllm) point straight at a
`base_url`. Proxy providers are spawned as local subprocesses by
`ProxySupervisor`:

* `claude_code` — `claude-code-openai-wrapper` (Python/poetry, FastAPI).
  Prereqs: clone the repo, `poetry install`, then either
  `claude auth login` (Pro/Max) or `export ANTHROPIC_API_KEY=...`.
  Spawn: `poetry run python main.py <port>` (port is positional or via
  the `PORT` env var).
* `github_copilot_cli` / `github_copilot_vscode` — `copilot-api`
  (Bun/npx). Prereq: `npx copilot-api@latest auth` once. Spawn:
  `npx copilot-api@latest start --port <N> --rate-limit 60 --wait`.
  Both names route to the same proxy — they exist as two config aliases.

`ProxySupervisor` is reference-counted so a fleet of N quests using the
same provider shares one proxy process — see Phase H.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

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


@dataclass
class ResolvedEndpoint:
    base_url: str
    model: str
    api_key: str


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
                handle = self._spawn(provider_name)
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

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
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
        _wait_for_openai_endpoint(port, timeout_s=60)
        return _ProxyHandle(name=provider_name, port=port, proc=proc)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    defaults = _DIRECT_DEFAULTS.get(name)
    if defaults is None:
        raise ValueError(f"unknown provider: {name!r}")
    api_key_env = provider.api_key_env or defaults["api_key_env"]
    api_key = os.environ.get(api_key_env, "") if api_key_env else "not-needed"
    return ResolvedEndpoint(
        base_url=provider.base_url or defaults["base_url"],
        model=provider.model or defaults["model"],
        api_key=api_key or "not-needed",
    )


async def resolve_endpoint_async(
    provider: ProviderConfig,
    supervisor: ProxySupervisor,
) -> ResolvedEndpoint:
    if provider.name not in _PROXY_PROVIDERS:
        return resolve_endpoint(provider)
    handle = await supervisor.acquire(provider.name)
    api_key = os.environ.get(provider.api_key_env or "", "") or "not-needed"
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
        headers = {
            "Authorization": f"Bearer {self.endpoint.api_key}",
            "Content-Type": "application/json",
        }
        r = await self._http.post(url, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
