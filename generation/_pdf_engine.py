"""Shared PDF engine discovery for the paper and poster generators.

Both ``generation/paper.py`` (pandoc + LaTeX engine) and
``generation/poster.py`` (direct pdflatex/tectonic on a beamerposter
``.tex``) need the same 3-tier fallback search:

1. ``pdflatex`` on PATH — canonical MiKTeX / TeX Live install.
2. ``tectonic`` on PATH — single-binary self-bootstrapping LaTeX.
3. ``<repo_root>/tools/tectonic[.exe]`` — opt-in install dropped by
   ``python launch.py --install-tectonic`` (no admin required).

Keeping the lookup in a dedicated module prevents the two generators
from drifting: if ``poster.py`` re-implemented the check with only a
``shutil.which("pdflatex")`` gate, a user who followed the documented
no-admin install path would silently lose ``poster.pdf`` while
``paper.pdf`` succeeded. Audit BLOCK #5 surfaced that drift; this
module is the deduplication that fixes it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Default repo root: ``generation/`` lives one directory below the repo
# root. Callers can pass an explicit ``repo_root`` to
# :func:`find_pdf_engine` to override (existing paper tests pin a fake
# repo by monkeypatching ``generation.paper.REPO_ROOT`` and the
# ``PaperGenerator._find_pdf_engine`` shim forwards it through — see
# ``generation/paper.py``).
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent


def find_pdf_engine(repo_root: Path | None = None) -> tuple[str, str] | None:
    """Locate a LaTeX engine to feed pandoc's ``--pdf-engine`` flag or to
    invoke directly (as poster.py does on its beamerposter ``.tex``).

    Search order (stop at first match):
      1. ``shutil.which("pdflatex")`` — the canonical MiKTeX / TeX Live
         install. Preferred when present because a warm cache compiles
         in ~1-3 s.
      2. ``shutil.which("tectonic")`` — a single-binary self-bootstrapping
         LaTeX. Works on corporate laptops without admin (downloads
         packages on first run, no install step, no GUI prompts).
      3. ``<repo_root>/tools/tectonic.exe`` (or ``./tools/tectonic`` on
         POSIX) — the opt-in install location populated by
         ``python launch.py --install-tectonic``. Most-isolated path,
         used when neither system install is present.

    ``repo_root`` defaults to the repo containing this file. Callers
    pass it explicitly so they can monkeypatch their own module-level
    ``REPO_ROOT`` constant for tests without having to also patch this
    helper.

    Returns ``(engine_name, full_path)`` or ``None`` when no engine is
    reachable. Tectonic is intentionally the fallback — flipping the
    order would penalize the dominant case (existing MiKTeX users with
    a warm cache) to help the corporate-laptop minority.
    """
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        return ("pdflatex", pdflatex)
    tectonic = shutil.which("tectonic")
    if tectonic:
        return ("tectonic", tectonic)
    # Repo-local opt-in install. Check ONLY the platform-appropriate
    # filename — checking the wrong extension first (e.g.
    # ``tectonic.exe`` on POSIX) risks picking up a stray Windows
    # binary in a shared / WSL-mounted repo and crashing the pandoc
    # call with an exec-format error. ``--install-tectonic`` writes
    # under the same platform-appropriate name.
    root = repo_root if repo_root is not None else _DEFAULT_REPO_ROOT
    repo_tectonic = root / "tools" / (
        "tectonic.exe" if sys.platform == "win32" else "tectonic"
    )
    if repo_tectonic.is_file():
        return ("tectonic", str(repo_tectonic))
    return None
