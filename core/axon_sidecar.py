"""Axon sidecar lifecycle — start once, share across FI surfaces.

Why a sidecar at all
====================
``core.knowledge`` instantiates ``AxonBrain`` in-process. The first
instantiation per Python interpreter is slow: it loads the embedding
model, opens vector indexes, and warms up the BM25 retriever. With
FI's subprocess-per-quest launcher, every new quest pays that cost
again — 5-15 s before the engine actually runs.

Running Axon as a long-lived sidecar (``python -m axon.api``, default
``127.0.0.1:8000``) keeps the model + indexes hot. FI surfaces (CLI,
web, VSCode) can hit ``/health/live`` to confirm it's up, and other
tooling (Axon CLI, MCP server, Streamlit UI) can talk to the same
on-disk corpus while FI is running.

What this module does
=====================
- ``axon_status(host, port)`` — non-blocking health probe. Returns
  ``{"running": bool, "url": str, "ready": bool, "error": str|None}``.
- ``ensure_axon_up(host, port)`` — idempotent launcher. If Axon is
  already on the port, returns immediately. Otherwise spawns
  ``python -m axon.api`` as a detached background process, polls
  ``/health/live`` until ready, and returns the same status dict.

We deliberately do **not** track the spawned PID for shutdown. The
sidecar is meant to outlive any single FI invocation so the *next*
CLI quest or web reload starts hot. Users who want to stop Axon can
``taskkill /F /IM python.exe`` matching the Axon command line or use
the OS-native process manager.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, TypedDict

_log = logging.getLogger("fi.axon_sidecar")

DEFAULT_HOST = os.environ.get("AXON_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("AXON_PORT", "8000"))


class AxonStatus(TypedDict):
    running: bool
    ready: bool
    url: str
    error: str | None


def _probe(url: str, timeout: float) -> tuple[bool, str | None]:
    """One-shot HTTP GET. Returns (ok, error_message)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return (200 <= resp.status < 300, None)
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except (TimeoutError, socket.timeout):
        return False, "timeout"
    except OSError as e:
        return False, str(e)


def axon_status(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    timeout: float = 1.5,
) -> AxonStatus:
    """Probe the Axon sidecar without starting one.

    Hits ``/health/live`` (process up) and ``/health/ready`` (brain
    initialized). Both are cheap on a warm sidecar.
    """
    base = f"http://{host}:{port}"
    live_ok, live_err = _probe(f"{base}/health/live", timeout=timeout)
    if not live_ok:
        return AxonStatus(running=False, ready=False, url=base, error=live_err)
    ready_ok, ready_err = _probe(f"{base}/health/ready", timeout=timeout)
    return AxonStatus(running=True, ready=ready_ok, url=base, error=ready_err)


def ensure_axon_up(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    boot_timeout: float = 30.0,
    poll_interval: float = 0.5,
    log: Callable[[str], None] | None = None,
) -> AxonStatus:
    """Make sure an Axon API sidecar is reachable on ``host:port``.

    Returns immediately if one is already running. Otherwise spawns
    ``python -m axon.api`` as a fully detached background process and
    polls ``/health/live`` until ``boot_timeout`` elapses.

    The spawned process inherits ``AXON_HOST`` + ``AXON_PORT`` from
    this call (not the env), so the URL we tell callers about is the
    URL we actually start it on.

    ``log`` defaults to a friendly stderr printer; pass ``None`` to
    silence, or any callable taking one string.
    """
    if log is None:
        def _default_log(msg: str) -> None:
            print(f"[FI/axon] {msg}", file=sys.stderr)
        log = _default_log

    status = axon_status(host, port, timeout=1.0)
    if status["running"]:
        log(f"sidecar already up at {status['url']} (ready={status['ready']})")
        return status

    log(f"sidecar not reachable at http://{host}:{port}; launching `python -m axon.api`...")

    env = os.environ.copy()
    env["AXON_HOST"] = host
    env["AXON_PORT"] = str(port)

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
        return AxonStatus(running=False, ready=False, url=f"http://{host}:{port}", error=str(e))
    except Exception as e:  # noqa: BLE001
        log(f"could not launch sidecar: {e}")
        return AxonStatus(running=False, ready=False, url=f"http://{host}:{port}", error=str(e))

    # Poll /health/live until ready or timeout.
    deadline = time.monotonic() + boot_timeout
    while time.monotonic() < deadline:
        status = axon_status(host, port, timeout=1.0)
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
    )
