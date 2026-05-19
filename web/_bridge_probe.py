"""Lightweight bridge-availability probe used by the serve-mode
schema endpoints to decide whether to surface ``vscode_extension`` in
the provider picker.

Two transports today, two probes here:

* TCP — the per-chat-command bridges the FI VSCode extension spawns
  with ``--vscode-bridge-port``. Loopback-only.
* IPC (Unix-domain socket on POSIX, named pipe on Windows) — the
  session-long PersistentBridge the extension starts at activation
  with ``--vscode-bridge-socket``. Per-user OS-managed address.

Each probe answers "does the OS accept a connection attempt at this
address" — not "is the listener actually FI's bridge protocol". That
distinction doesn't matter for the UI gate: the existing submit-time
guard catches protocol mismatches at first LLM call, and false
positives at the per-user socket / loopback-port level are
vanishingly rare.

All probes are async, stdlib-only, with a short (default 500 ms)
timeout because they're on the hot path of every ``/api/*/schema``
fetch.
"""

from __future__ import annotations

import asyncio
import os
import sys


async def is_bridge_listening(
    port: int,
    *,
    host: str = "127.0.0.1",
    timeout_s: float = 0.5,
) -> bool:
    """Return True iff a TCP connection to ``host:port`` is accepted
    within ``timeout_s``.

    Returns False on any error (port out of range, refused, timeout)
    so callers can use it as a simple gate without try/except.
    """
    if port <= 0 or port > 65535:
        return False
    try:
        fut = asyncio.open_connection(host, port)
        _reader, writer = await asyncio.wait_for(fut, timeout=timeout_s)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    return True


async def is_socket_listening(
    socket_path: str,
    *,
    timeout_s: float = 0.5,
) -> bool:
    """Return True iff the per-user IPC channel at ``socket_path`` is
    open and accepting connections.

    Cross-platform:

    * POSIX → :func:`asyncio.open_unix_connection`. Fast: an absent
      socket returns ``ENOENT`` immediately; a present-but-not-listening
      one returns ``ECONNREFUSED``; a listening one connects in <1 ms.

    * Windows → :meth:`ProactorEventLoop.create_pipe_connection` on
      ``\\\\.\\pipe\\<name>``. Same fast-fail semantics.

    Returns False on any error so callers can use it as a simple gate
    without try/except.
    """
    if not socket_path:
        return False
    if sys.platform.startswith("win"):
        loop = asyncio.get_event_loop()
        if not hasattr(loop, "create_pipe_connection"):
            return False
        try:
            await asyncio.wait_for(
                loop.create_pipe_connection(  # type: ignore[attr-defined]
                    asyncio.Protocol, socket_path,
                ),
                timeout=timeout_s,
            )
        except (OSError, asyncio.TimeoutError):
            return False
        return True
    # POSIX: the socket file must exist before we even try to connect
    # (open_unix_connection raises FileNotFoundError fast enough but
    # this short-circuits the common "no bridge running" case).
    if not os.path.exists(socket_path):
        return False
    try:
        fut = asyncio.open_unix_connection(socket_path)
        _reader, writer = await asyncio.wait_for(fut, timeout=timeout_s)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    return True
