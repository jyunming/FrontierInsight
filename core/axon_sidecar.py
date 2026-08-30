"""Axon sidecar lifecycle — start once, share across FI surfaces.

Why a sidecar at all
====================
``core.knowledge`` instantiates ``AxonBrain`` in-process. The first
instantiation per Python interpreter is slow: it loads the embedding
model, opens vector indexes, and warms up the BM25 retriever. With
FI's subprocess-per-quest launcher, every new quest pays that cost
again — 5-15 s before the engine actually runs.

Running Axon as a long-lived sidecar (``python -m axon.api``) keeps the
model + indexes hot. FI surfaces (CLI, web, VSCode) can hit
``/health/live`` to confirm it's up, and other tooling (Axon CLI, MCP
server, Streamlit UI) can talk to the same on-disk corpus while FI is
running.

Finding it
==========
The sidecar's port is not fixed: Axon resolves it as ``--port`` >
``AXON_PORT`` > ``config.yaml``'s ``api.port`` > ``8420``. So we don't
guess — ``core.axon_endpoint`` builds an ordered candidate list (env
vars, the running server's own lock file, Axon's config, then the
static defaults) and we probe them in order, first live one wins.

What this module does
=====================
- ``axon_status(host, port)`` — non-blocking health probe. Returns
  ``{"running": bool, "url": str, "ready": bool, "error": str|None,
  "source": str}``. With no arguments it searches every candidate.
- ``ensure_axon_up(host, port)`` — idempotent launcher. If Axon is
  already reachable at any candidate, returns immediately. Otherwise
  spawns ``python -m axon.api`` as a detached background process on the
  *preferred* endpoint (config.yaml's port, not the legacy 8000), polls
  ``/health/live`` until ready, and returns the same status dict.

Spawning on the wrong port is not merely wasteful: current Axon holds a
per-store single-instance lock, so a second server aimed at the same
store exits during startup. Probing before spawning is what keeps that
from turning into a guaranteed boot-timeout on every launch.

We deliberately do **not** track the spawned PID for shutdown. The
sidecar is meant to outlive any single FI invocation so the *next*
CLI quest or web reload starts hot. Users who want to stop Axon can
``taskkill /F /IM python.exe`` matching the Axon command line or use
the OS-native process manager.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, TypedDict

from core.axon_endpoint import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    Candidate,
    candidates,
    preferred_endpoint,
)

_log = logging.getLogger("fi.axon_sidecar")

__all__ = [
    "AxonStatus",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "axon_status",
    "ensure_axon_up",
]


class AxonStatus(TypedDict):
    running: bool
    ready: bool
    url: str
    error: str | None
    # Which candidate answered (or, when nothing did, which one we'd
    # spawn on). Surfaced to the user so "not detected" comes with the
    # list of places we actually looked.
    source: str


# Health payloads are tiny; cap the read so a wrong-service probe can't
# stream a large body into a health check.
_MAX_HEALTH_BODY_BYTES = 4096


def _looks_like_axon(body: bytes | None) -> bool:
    """Reject a 200 that clearly isn't Axon.

    ``/health/live`` returns ``{"status": "alive"}``. Other services live
    on these ports — Axon's own config defaults ``vllm_base_url`` to
    ``localhost:8000/v1``, exactly the port FI used to probe — and a
    passing status code alone would mislabel one of them as a warm
    sidecar. An unreadable or non-JSON body is inconclusive rather than
    wrong, so it stays accepted; only a JSON object that answers with no
    recognisable ``status`` is treated as "some other service".
    """
    if not body:
        return True
    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return True
    if not isinstance(parsed, dict):
        return True
    return isinstance(parsed.get("status"), str)


def _probe(url: str, timeout: float) -> tuple[bool, str | None]:
    """One-shot HTTP GET. Returns (ok, error_message)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if not (200 <= resp.status < 300):
                return False, f"HTTP {resp.status}"
            # Body read is best-effort: the response object may be a
            # stub (tests) or a stream that has already been consumed.
            body: bytes | None = None
            try:
                reader = getattr(resp, "read", None)
                if callable(reader):
                    body = reader(_MAX_HEALTH_BODY_BYTES)
            except Exception:  # noqa: BLE001 - body is a bonus, not a requirement
                body = None
            if not _looks_like_axon(body):
                return False, "responded, but not an Axon health endpoint"
            return True, None
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except (TimeoutError, socket.timeout):
        return False, "timeout"
    except OSError as e:
        return False, str(e)


def _probe_one(host: str, port: int, source: str, timeout: float) -> AxonStatus:
    """Probe a single endpoint's ``/health/live`` + ``/health/ready``."""
    base = f"http://{host}:{port}"
    live_ok, live_err = _probe(f"{base}/health/live", timeout=timeout)
    if not live_ok:
        return AxonStatus(
            running=False, ready=False, url=base, error=live_err, source=source,
        )
    ready_ok, ready_err = _probe(f"{base}/health/ready", timeout=timeout)
    return AxonStatus(
        running=True, ready=ready_ok, url=base,
        error=None if ready_ok else ready_err, source=source,
    )


def axon_status(
    host: str | None = None,
    port: int | None = None,
    *,
    timeout: float = 1.5,
) -> AxonStatus:
    """Probe the Axon sidecar without starting one.

    Hits ``/health/live`` (process up) and ``/health/ready`` (brain
    initialized). Both are cheap on a warm sidecar.

    With no ``host``/``port``, walks the candidate list from
    ``core.axon_endpoint`` and returns the first live one. When none
    answer, the returned status describes the *preferred* endpoint (what
    ``ensure_axon_up`` would spawn) and its ``error`` names every
    candidate tried, so a wrong-port setup is diagnosable from the
    message alone.
    """
    found = list(candidates(host, port))
    if not found:  # pragma: no cover - candidates() always yields defaults
        found = [Candidate(DEFAULT_HOST, DEFAULT_PORT, "default")]

    attempts: list[str] = []
    for cand in found:
        status = _probe_one(cand.host, cand.port, cand.source, timeout)
        if status["running"]:
            return status
        attempts.append(f"{cand.base_url} ({cand.source}): {status['error']}")

    pref_host, pref_port = preferred_endpoint(host, port)
    return AxonStatus(
        running=False,
        ready=False,
        url=f"http://{pref_host}:{pref_port}",
        error="; ".join(attempts),
        source="none reachable",
    )


def ensure_axon_up(
    host: str | None = None,
    port: int | None = None,
    *,
    boot_timeout: float = 30.0,
    poll_interval: float = 0.5,
    offline: bool | None = None,
    models_dir: "os.PathLike[str] | str | None" = None,
    log: Callable[[str], None] | None = None,
) -> AxonStatus:
    """Make sure an Axon API sidecar is reachable.

    Returns immediately if one is already running anywhere on the
    candidate list. Otherwise spawns ``python -m axon.api`` as a fully
    detached background process on the preferred endpoint and polls
    ``/health/live`` until ``boot_timeout`` elapses.

    ``host``/``port`` pin the search and the spawn to one address; left
    unset, both come from ``core.axon_endpoint``. Searching before
    spawning matters more than it used to: Axon now refuses to start a
    second server against a store another server already holds, so
    spawning on a stale port doesn't produce a spare sidecar — it
    produces a process that exits and a guaranteed boot timeout.

    The spawned process inherits ``AXON_HOST`` + ``AXON_PORT`` from
    this call (not the env), so the URL we tell callers about is the
    URL we actually start it on.

    ``offline`` / ``models_dir`` mirror ``knowledge.offline`` /
    ``knowledge.models_dir``: when set, the spawned sidecar gets
    ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` (and ``HF_HOME``
    pointed at the local model cache) so it loads embedding weights from
    disk with no network — the air-gapped path. Without this, the
    sidecar would still try to download even when FI is configured
    offline.

    ``log`` defaults to a friendly stderr printer; pass ``None`` to
    silence, or any callable taking one string.
    """
    if log is None:
        def _default_log(msg: str) -> None:
            print(f"[FI/axon] {msg}", file=sys.stderr)
        log = _default_log

    # The sidecar is launched once at process startup — before any
    # per-quest YAML is parsed — so the cross-process source of truth for
    # offline config is the env, not config. Resolve unset params from
    # ``FI_OFFLINE`` / ``FI_MODELS_DIR`` (same env knobs that seed
    # ``KnowledgeConfig.offline`` / ``.models_dir``). An explicit arg
    # always wins. Kept in sync with ``core.config._offline_env_default``.
    if offline is None:
        offline = os.environ.get("FI_OFFLINE", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
    if models_dir is None:
        models_dir = os.environ.get("FI_MODELS_DIR") or None

    status = axon_status(host, port, timeout=1.0)
    if status["running"]:
        log(
            f"sidecar already up at {status['url']} "
            f"(ready={status['ready']}, found via {status['source']})"
        )
        return status

    # Nothing answered. Spawn on the endpoint Axon itself would pick,
    # not on whatever port FI happened to probe last.
    host, port = preferred_endpoint(host, port)
    log(
        f"no sidecar found ({status['error']}); "
        f"launching `python -m axon.api` on http://{host}:{port}..."
    )

    env = os.environ.copy()
    env["AXON_HOST"] = host
    env["AXON_PORT"] = str(port)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    if models_dir:
        env["HF_HOME"] = os.path.expanduser(str(models_dir))

    # Spawn fully detached so the sidecar outlives this FI process —
    # the *next* CLI quest or web reload then hits a warm one.
    creationflags = 0
    start_new_session = False
    if sys.platform == "win32":
        creationflags = (
            subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        start_new_session = True

    try:
        subprocess.Popen(
            [sys.executable, "-m", "axon.api"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            start_new_session=start_new_session,
            close_fds=True,
        )
    except FileNotFoundError as e:
        log(f"could not launch sidecar — python or axon missing: {e}")
        return AxonStatus(
            running=False, ready=False, url=f"http://{host}:{port}",
            error=str(e), source="spawn failed",
        )
    except Exception as e:  # noqa: BLE001
        log(f"could not launch sidecar: {e}")
        return AxonStatus(
            running=False, ready=False, url=f"http://{host}:{port}",
            error=str(e), source="spawn failed",
        )

    # Poll /health/live until ready or timeout. Pinned to the endpoint
    # we just spawned on — re-running discovery here would rediscover
    # candidates we already know are dead and slow every poll down.
    deadline = time.monotonic() + boot_timeout
    while time.monotonic() < deadline:
        status = _probe_one(host, port, "spawned", timeout=1.0)
        if status["running"]:
            log(f"sidecar reachable at {status['url']} (ready={status['ready']})")
            return status
        time.sleep(poll_interval)

    log(f"sidecar did not come up within {boot_timeout:.0f}s — continuing without it")
    return AxonStatus(
        running=False,
        ready=False,
        url=f"http://{host}:{port}",
        error=f"boot timeout after {boot_timeout:.0f}s",
        source="spawned",
    )
