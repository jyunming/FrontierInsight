"""The folder FI keeps its per-person state in: ``~/.frontier-insight`` (``%USERPROFILE%\\.frontier-insight`` on Windows).

``FI_HOME`` overrides it, which is what the tests use so that no test reads or writes the real folder. New per-person
files (the quest index, ``core/quest_index.py``) go through :func:`fi_home`; the VS Code extension has the same rule in
``vscode-frontier-insight/src/fi-home.ts`` — keep the two in step.
"""

from __future__ import annotations

import os
from pathlib import Path


def fi_home() -> Path:
    """``FI_HOME`` when set, else ``~/.frontier-insight``. Not created here: the caller that writes creates it."""
    override = os.environ.get("FI_HOME", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".frontier-insight"
