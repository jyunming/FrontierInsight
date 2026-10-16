"""HTML / Chromium PDF fallback for the paper generator.

When no LaTeX engine (pdflatex / tectonic) is reachable, render ``paper.md``
to a PDF *without* LaTeX: pandoc turns the markdown into a Computer-Modern-
styled HTML page (the ``templates/paper/_html/latexlike.css`` theme, with
Latin Modern Roman embedded so the look matches the MiKTeX/tectonic
``article`` output), then a headless system browser prints it to PDF.

Requirements: pandoc (already needed for the LaTeX path) + a Chromium-family
browser (Edge / Chrome / Chromium). Both are near-ubiquitous and need no
admin install, which is the whole point — a locked-down machine that can't
get a TeX distribution can still produce a styled paper.pdf.

Limitation: this can't reproduce a venue's two-column LaTeX class, so it
always renders the single-column house style. It's a fallback, not a
replacement for the venue templates.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from generation._figure_captions import blank_lines_around_figures, numbers_off_figure_captions
from generation._headings import strip_heading_numbers
from generation._keywords import keywords_block
from generation._pandoc import HTML_MARKDOWN_READER
from core.proc_tree import run_tree

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "templates" / "paper" / "_html"
_CSS_PATH = _ASSETS_DIR / "latexlike.css"

# Selectable HTML themes (output.paper_style). ``latexlike`` matches the
# Computer Modern LaTeX article (the no-LaTeX fallback default); ``briefing``
# is the Frontier Insight Research-Briefing brand look.
_THEMES: dict[str, Path] = {
    "latexlike": _ASSETS_DIR / "latexlike.css",
    "briefing": _ASSETS_DIR / "briefing.css",
}


def _run(cmd: list[str], *, timeout: float, cwd: str | None = None) -> "subprocess.CompletedProcess[str]":
    """Run pandoc or the headless browser. A timeout stops it and every process it
    started (the browser's helpers would otherwise keep its output open, and the
    wait for that output had no limit)."""
    return run_tree(cmd, timeout=timeout, cwd=cwd)


def theme_css_path(theme: str) -> Path:
    """Resolve a theme name to its CSS file, defaulting to ``latexlike``."""
    return _THEMES.get(theme, _CSS_PATH)

# Chromium-family executables, by the names they're usually invokable as.
_BROWSER_PATH_NAMES = (
    "msedge", "microsoft-edge", "microsoft-edge-stable",
    "google-chrome", "google-chrome-stable",
    "chromium", "chromium-browser", "chrome",
)


def find_html_browser() -> tuple[str, str] | None:
    """Locate a Chromium-family browser for headless print-to-pdf.

    Returns ``(name, path)`` or ``None``. Checks PATH first, then the
    platform's default install locations (Edge ships with Windows but
    isn't always on a Python child's PATH)."""
    for name in _BROWSER_PATH_NAMES:
        found = shutil.which(name)
        if found:
            return (name, found)

    candidates: list[str] = []
    if sys.platform == "win32":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pfx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            rf"{pfx86}\Microsoft\Edge\Application\msedge.exe",
            rf"{pf}\Microsoft\Edge\Application\msedge.exe",
            rf"{pf}\Google\Chrome\Application\chrome.exe",
            rf"{pfx86}\Google\Chrome\Application\chrome.exe",
        ]
        if local:
            candidates.append(rf"{local}\Google\Chrome\Application\chrome.exe")
    elif sys.platform == "darwin":
        candidates = [
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    for cand in candidates:
        if cand and Path(cand).is_file():
            if "Edge" in cand:
                name = "msedge"
            elif "Chromium" in cand:
                name = "chromium"
            else:
                name = "chrome"
            return (name, cand)
    return None


_FIRST_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


# Math spans (both spellings) and code, removed before scanning for LaTeX the
# HTML writer can't express: commands INSIDE math are exactly the ones that do
# render (as MathML), so counting them would make the warning meaningless.
# The dollar-math alternative follows pandoc's own rule — a non-space just
# inside each ``$`` and no digit after the closing one — so "$5 to $10" is
# prose, not a math span.
_MATH_OR_CODE_RE = re.compile(
    r"```.*?```"
    r"|`[^`\n]+`"
    r"|\$\$.*?\$\$"
    r"|\\\[.*?\\\]"
    r"|\\\(.*?\\\)"
    r"|(?<!\\)\$(?=\S)[^$\n]*?[^\s\\]\$(?!\d)",
    re.DOTALL,
)
# A LaTeX control word: two letters or more, so a markdown escape of a single
# punctuation character (``\_``, ``\&``) isn't reported as a lost formula.
_TEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]{2,}")


def raw_tex_outside_math(markdown: str) -> list[str]:
    """The distinct LaTeX commands sitting OUTSIDE math and code, in order.

    These are what the browser path cannot typeset. They are no longer
    dropped (the reader disables ``raw_tex``, so they print as written), but
    a formula reaching the page as its own source is still a defect worth
    naming in run.log — the writer should have marked it up as math.
    """
    stripped = _MATH_OR_CODE_RE.sub(" ", markdown)
    found: list[str] = []
    for m in _TEX_COMMAND_RE.finditer(stripped):
        if m.group(0) not in found:
            found.append(m.group(0))
    return found


# The line printed at the top of a paper the browser made because LaTeX could not.
NOT_TYPESET_TEXT = (
    "Not typeset: this copy was made without LaTeX, so formulas, tables and page breaks "
    "may look different from a typeset paper."
)
NOT_TYPESET_BANNER_HTML = (
    '<p class="fi-not-typeset" style="border:1px solid #b45309;color:#b45309;'
    'padding:3pt 6pt;margin:0 0 8pt 0;font:bold 9.5pt sans-serif;text-indent:0;text-align:left">'
    + NOT_TYPESET_TEXT + "</p>\n"
)


def _split_title(md_text: str) -> tuple[str, str]:
    """Split off the first ``# Heading`` so pandoc renders it in the title
    block (with the author byline) rather than as an in-body H1. Returns
    ``(title, body)``; ``title`` is "" when there's no leading heading."""
    m = _FIRST_H1_RE.search(md_text)
    if not m:
        return "", md_text
    title = m.group(1).strip()
    body = md_text[:m.start()] + md_text[m.end():]
    return title, body


def _numbered_depth(css_path: Path) -> int:
    """How many heading levels the theme numbers with CSS counters: two for the
    LaTeX-like theme (sections and subsections), none for the briefing theme."""
    try:
        css = css_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    return 2 if "counter(sub)" in css else 1 if "counter(sec)" in css else 0


def render_paper_html_pdf(
    paper_md: Path,
    out_pdf: Path,
    *,
    pandoc_path: str,
    browser: tuple[str, str],
    css_path: Path | None = None,
    log: logging.Logger | None = None,
    timeout_s: float = 180.0,
    author_line: tuple[str, ...] = (),
    not_typeset: str | None = None,
) -> tuple[Path | None, str]:
    """Render ``paper_md`` to ``out_pdf`` via pandoc → styled HTML → headless
    browser. Returns ``(out_pdf, "")`` on success or ``(None, reason)``.

    ``css_path`` selects the theme (``latexlike.css`` by default, or
    ``briefing.css`` for the Research-Briefing look). Runs pandoc with
    ``cwd = out_pdf.parent`` so ``figures/…`` references resolve against the
    copied ``figures/`` dir (same contract the LaTeX path relies on);
    ``--embed-resources`` then inlines the images + any embedded fonts, so
    the printed PDF is fully self-contained.

    ``not_typeset`` is the reason LaTeX did not make this paper (when it is the
    fallback for it): the first page then says, in a visible line, that the
    paper is not typeset. Sections are numbered once: where the theme numbers a
    heading, a section number the writer typed at its start comes off; where it
    does not (the briefing theme), the writer's number stays."""
    log = log or logging.getLogger("frontier_insight.paper")
    browser_name, browser_path = browser
    css_path = css_path or _CSS_PATH
    if not css_path.is_file():
        return None, f"HTML theme missing at {css_path}"

    work = out_pdf.parent
    try:
        from core import paper_facts as _paper_facts

        # FI's fact markers would print as text inside math or code.
        md_text = _paper_facts.without_marks(paper_md.read_text(encoding="utf-8"))
    except OSError as e:
        return None, f"could not read {paper_md}: {e}"
    # Both themes number figures, so the writer's "**Figure 2.**" comes off.
    # A table right under its caption line would print as raw pipes.
    from generation._tables import blank_line_before_tables

    prepared = blank_line_before_tables(blank_lines_around_figures(numbers_off_figure_captions(md_text)))
    # A section number the writer typed comes off only where the theme numbers headings.
    title, body = _split_title(strip_heading_numbers(prepared, numbered_depth=_numbered_depth(css_path)))
    # The keywords line under the abstract becomes a block the themes style.
    keywords, body = keywords_block(body)

    # Resolve the @FONT_DIR@ token to the assets dir's absolute file:/// URL
    # so pandoc --embed-resources can find + inline any vendored OTFs (the
    # latexlike theme uses this; briefing relies on OS-bundled fonts).
    css_text = css_path.read_text(encoding="utf-8").replace(
        "@FONT_DIR@", _ASSETS_DIR.as_uri())
    css_resolved = work / "paper_html_theme.css"
    css_resolved.write_text(css_text, encoding="utf-8")
    body_md = work / "paper_html_body.md"
    body_md.write_text(body, encoding="utf-8")
    html_path = work / "paper_html_source.html"
    # ``author_line`` is the configured author, affiliation, email and link;
    # each set one prints as its own byline row. None set keeps the house
    # byline.
    bylines = [" ".join(str(v).split()) for v in author_line if str(v or "").strip()]
    byline_args = [
        arg for line in (bylines or ["Frontier Insight"])
        for arg in ("--metadata", f"author={line}")
    ]

    # Anything the browser path can't typeset stays visible instead of
    # vanishing (see HTML_MARKDOWN_READER), but say so: a printed ``\SI{...}``
    # means the writer wrote a formula as prose.
    leftover = raw_tex_outside_math(body)
    if leftover:
        log.warning(
            "paper.pdf: the HTML render cannot typeset %d LaTeX command(s) "
            "written outside math (%s); they are printed as written rather "
            "than dropped. Mark them up as math to have them typeset.",
            len(leftover), ", ".join(leftover[:5]),
        )

    banner_args: list[str] = []
    if not_typeset is not None:
        banner = work / "paper_html_banner.html"
        banner.write_text(NOT_TYPESET_BANNER_HTML, encoding="utf-8")
        banner_args = ["--include-before-body", banner.name]

    pandoc_cmd = [
        pandoc_path, body_md.name,
        "--standalone", "--embed-resources", "--mathml",
        # Same markdown dialect as the LaTeX path, minus raw TeX: ``\(x\)``
        # and ``\[x\]`` become MathML like ``$x$`` already did, and TeX this
        # writer can't express is kept as visible text instead of silently
        # deleted.
        f"--from={HTML_MARKDOWN_READER}",
        "--css", css_resolved.name,
        "--metadata", f"title={title or 'Untitled'}",
        *byline_args,
        "--metadata", "pagetitle=paper",
        *banner_args,
        *(["--metadata", "keywords=" + ", ".join(keywords)] if keywords else []),
        "-o", html_path.name,
    ]
    try:
        pr = _run(pandoc_cmd, timeout=timeout_s, cwd=str(work))
    except subprocess.TimeoutExpired:
        return None, "pandoc (markdown→HTML) timed out"
    except OSError as e:
        return None, f"pandoc (markdown→HTML) could not run: {e}"
    if pr.returncode != 0 or not html_path.is_file():
        return None, (
            f"pandoc (markdown→HTML) failed (rc={pr.returncode}): "
            f"{(pr.stderr or '').strip()[:300]}"
        )

    # Headless print-to-pdf. A throwaway --user-data-dir keeps the render off
    # the user's real browser profile and dodges first-run prompts.
    with tempfile.TemporaryDirectory(prefix="fi-html-pdf-") as profile_dir:
        browser_cmd = [
            browser_path, "--headless=new", "--disable-gpu",
            "--no-pdf-header-footer", "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile_dir}",
            f"--print-to-pdf={out_pdf}",
            html_path.as_uri(),
        ]
        try:
            br = _run(browser_cmd, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None, f"{browser_name} print-to-pdf timed out"
        except OSError as e:
            return None, f"{browser_name} could not run: {e}"

    if not out_pdf.is_file() or out_pdf.stat().st_size == 0:
        return None, (
            f"{browser_name} produced no PDF (rc={br.returncode}): "
            f"{(br.stderr or '').strip()[:300]}"
        )
    log.info(
        "paper.pdf: rendered via HTML fallback (pandoc + %s headless), %d bytes",
        browser_name, out_pdf.stat().st_size,
    )
    return out_pdf, ""
