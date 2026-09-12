"""Marp CLI discovery, matching the pandoc / LaTeX-engine lookups.

The slide generator used a bare ``shutil.which("marp")``, which is PATH-only.
That is fine when marp came from ``npm install -g``, but it cannot see the
standalone binary ``python launch.py --install-marp`` drops into ``tools/`` --
and that installer exists precisely for hosts where npm is unavailable. A
PATH-only lookup would leave those users with a downloaded binary FI refuses
to use.

Tiers mirror ``_pandoc.find_pandoc`` and ``_pdf_engine.find_pdf_engine``:

1. ``marp`` on PATH -- an npm/global install; preferred, since an explicit
   system install is what the user most likely intends.
2. ``<repo_root>/tools/marp[.exe]`` -- the opt-in standalone install.

Note on exports: HTML needs no browser, but PDF / PPTX / PNG do (marp-cli
marks those with a browser icon in its README). So a present marp is
necessary but not always sufficient -- the generator still reports the
chromium-missing case separately.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent


def find_marp(repo_root: Path | None = None) -> str | None:
    """Locate a usable marp executable, or ``None``."""
    on_path = shutil.which("marp")
    if on_path:
        return on_path

    root = repo_root or _DEFAULT_REPO_ROOT
    # Only the platform-appropriate name, for the same reason _pdf_engine
    # gives: a stray marp.exe on POSIX would be found and then fail at exec
    # time with a format error, which is far more confusing than "not found".
    local = root / "tools" / ("marp.exe" if sys.platform == "win32" else "marp")
    if local.is_file():
        return str(local)
    return None
