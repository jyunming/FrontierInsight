"""Canonical path of the per-user FI persistent bridge.

Mirrors :file:`vscode-frontier-insight/src/bridge-path.ts` so both
ends of the bridge always agree on the address.

POSIX:  ``$XDG_RUNTIME_DIR/fi-bridge.sock`` (fallback
``/tmp/fi-bridge-{uid}.sock``) — Unix-domain socket.
Windows: ``\\\\.\\pipe\\fi-bridge-{USERNAME}`` — named pipe.

Both transports give us per-user isolation (so a shared Linux host
or a Windows Terminal Server can host multiple FI users without
collision) plus OS-managed lifecycle (the socket / pipe disappears
when the owning process dies — no stale state to clean up).
"""

from __future__ import annotations

import os
import re
import sys
import tempfile


def persistent_bridge_path() -> str:
    """Return the canonical persistent-bridge address for this OS +
    user. The caller decides whether to open it as a Unix-socket or
    as a Windows named pipe based on :data:`sys.platform`."""
    if sys.platform.startswith("win"):
        user = os.environ.get("USERNAME") or "default"
        user = re.sub(r"[^A-Za-z0-9_-]", "_", user)
        return rf"\\.\pipe\fi-bridge-{user}"
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return os.path.join(xdg, "fi-bridge.sock")
    try:
        uid = os.getuid()
    except AttributeError:  # defensive — getuid is POSIX-only
        uid = "default"
    return os.path.join(tempfile.gettempdir(), f"fi-bridge-{uid}.sock")
