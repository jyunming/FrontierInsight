"""Shared pandoc discovery for the paper generator and the engine pre-flight.

``generation/paper.py`` (markdown -> LaTeX -> PDF) and
``core/engine.py:_preflight_paper_pdf`` both need to answer "can this host
run pandoc?". Both used a bare ``shutil.which("pandoc")``, which is PATH-only
and therefore fails on the exact machines that need help most: locked-down
hosts where the user cannot run an installer or edit the system PATH.

This module widens that to the same 3-tier shape
``_pdf_engine.find_pdf_engine`` already uses for the LaTeX engine, so the two
halves of the PDF pipeline are discoverable the same way:

1. ``pandoc`` on PATH — canonical install (winget / brew / apt). Preferred:
   an explicit system install is what the user most likely intends.
2. ``<repo_root>/tools/pandoc[.exe]`` — drop the portable release archive's
   binary in and it is picked up, mirroring ``tools/tectonic[.exe]``. No
   admin, no PATH edit.
3. **pypandoc's bundled binary** — ``pip install pypandoc_binary`` ships a
   real pandoc executable inside site-packages. This is the only route that
   works with nothing but ``pip``, which makes it the practical answer on a
   restricted machine.

Keeping the lookup here prevents the drift ``_pdf_engine.py``'s docstring
describes: if the pre-flight and the generator disagree about what counts as
"pandoc available", a quest either burns LLM calls before failing, or refuses
to start on a host that would in fact have worked.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ``generation/`` lives one directory below the repo root. Callers may pass an
# explicit ``repo_root`` (tests pin a fake repo the same way they do for
# ``find_pdf_engine``).
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent


# The markdown dialect FI reads a paper in. ONE definition, because both
# renderers must agree on what the writer's text means: the LaTeX path
# (``generation/paper.py``) and the HTML/browser fallback
# (``generation/_html_pdf.py``) render the SAME ``paper.md``, and a dialect
# difference between them shows up as a paper whose two renderings say
# different things.
#
# Beyond pandoc's markdown defaults:
#
# * ``lists_without_preceding_blankline`` — the LLM routinely drops a bullet
#   or numbered list straight after the introducing paragraph with no blank
#   line. Pandoc's default reader requires that blank line and otherwise jams
#   the whole list into the paragraph as inline text ("Foo - a - b - c.").
# * ``autolink_bare_uris`` — a bare reference URL becomes a real ``\url{}``,
#   which the templates' ``xurl`` can break at ANY character. Without it a
#   long wiki-style URL runs straight into the right margin.
# * ``tex_math_single_backslash`` — ``\(x\)`` is inline math and ``\[x\]`` is
#   display math. Pandoc's markdown reader understands ``$x$`` but NOT
#   ``\(x\)`` on its own: it reads ``\(`` as an escaped parenthesis and the
#   commands between the delimiters as raw TeX. Writers use both spellings,
#   and the ``\(...\)`` one silently changed what papers said — LaTeX got
#   ``(R\_0\leq1)`` in text mode and stopped with "Missing $ inserted", while
#   the HTML writer, which cannot emit raw TeX, DELETED the commands and
#   printed "zero when (R_0)" with the relation gone, and table headers as
#   "(R_0) (N) () () ()". Both spellings now parse as math on both paths.
MARKDOWN_READER = (
    "markdown"
    "+lists_without_preceding_blankline"
    "+autolink_bare_uris"
    "+tex_math_single_backslash"
)

# The same dialect for a renderer that has no way to express raw TeX.
#
# ``-raw_tex`` is the difference. With raw TeX enabled, pandoc keeps a stray
# ``\SI{3}{\micro}`` (anything outside math) as a raw-LaTeX inline; the LaTeX
# writer passes it through, but the HTML writer has nowhere to put it and
# drops it WITHOUT a word — the reader then gets a sentence whose meaning
# changed, which is worse than one that looks wrong. Disabled, pandoc reads
# the same text as literal characters and the HTML keeps them visible, so a
# formula the browser path cannot typeset survives as its own source rather
# than as a hole in the sentence. Math is unaffected: ``\(...\)``, ``\[...\]``
# and ``$...$`` are still parsed as math and rendered as MathML.
HTML_MARKDOWN_READER = MARKDOWN_READER + "-raw_tex"


def _pypandoc_pandoc() -> str | None:
    """Absolute path to the pandoc binary bundled by ``pypandoc_binary``.

    Returns ``None`` when pypandoc is absent, or is the plain ``pypandoc``
    distribution that expects a system pandoc it cannot find.

    Looks inside the installed package directory FIRST, rather than trusting
    ``pypandoc.get_pandoc_path()``. That call resolves through PATH and
    returns the bare string ``"pandoc"`` when a system install exists, which
    is not a usable path for us: we only reach this tier when PATH has
    already failed, so a bare name here means "nothing new to offer". The
    bundled executable that ``pypandoc_binary`` ships lands next to
    ``pypandoc/__init__.py`` and is what this tier is actually for.
    """
    try:
        import pypandoc
    except Exception:  # noqa: BLE001 — not installed, or import-time failure
        return None

    pkg_dir = Path(pypandoc.__file__).resolve().parent
    exe_name = "pandoc.exe" if sys.platform == "win32" else "pandoc"
    # ``pypandoc_binary`` ships the executable under ``pypandoc/files/``;
    # check the package root too in case that layout ever flattens.
    for bundled in (pkg_dir / "files" / exe_name, pkg_dir / exe_name):
        if bundled.is_file():
            return str(bundled)

    # Secondary: honour an explicit absolute path pypandoc knows about (e.g.
    # a user-downloaded copy via ``pypandoc.download_pandoc()``). Bare names
    # are rejected — see the docstring.
    try:
        p = pypandoc.get_pandoc_path()
    except Exception:  # noqa: BLE001 — pypandoc present but no binary found
        return None
    if p and Path(p).is_absolute() and Path(p).is_file():
        return str(p)
    return None


def find_pandoc(repo_root: Path | None = None) -> str | None:
    """Locate a usable pandoc executable, or ``None``.

    Order is PATH -> repo-local ``tools/`` -> pypandoc's bundled copy. See the
    module docstring for why each tier exists.
    """
    on_path = shutil.which("pandoc")
    if on_path:
        return on_path

    root = repo_root or _DEFAULT_REPO_ROOT
    # Check ONLY the platform-appropriate name, for the same reason
    # ``_pdf_engine`` does: a stray ``pandoc.exe`` on POSIX would be found and
    # then fail at exec time with a format error, which is a far more
    # confusing failure than "not found".
    local = root / "tools" / ("pandoc.exe" if sys.platform == "win32" else "pandoc")
    if local.is_file():
        return str(local)

    return _pypandoc_pandoc()
