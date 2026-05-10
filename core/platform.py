"""Platform detection (diagnostics only).

The DS daemon discovery code that used to live here is gone — FI no
longer wraps an external research daemon, so there's nothing to spawn.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

System = Literal["linux", "wsl2", "macos", "windows"]


def detect_system() -> System:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    proc_version = Path("/proc/version")
    if proc_version.exists() and "microsoft" in proc_version.read_text().lower():
        return "wsl2"
    return "linux"
