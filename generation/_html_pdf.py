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

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "templates" / "paper" / "_html"
_CSS_PATH = _ASSETS_DIR / "latexlike.css"

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


def render_paper_html_pdf(
    paper_md: Path,
    out_pdf: Path,
    *,
    pandoc_path: str,
    browser: tuple[str, str],
    log: logging.Logger | None = None,
    timeout_s: float = 180.0,
) -> tuple[Path | None, str]:
    """Render ``paper_md`` to ``out_pdf`` via pandoc → styled HTML → headless
    browser. Returns ``(out_pdf, "")`` on success or ``(None, reason)``.

    Runs pandoc with ``cwd = out_pdf.parent`` so ``figures/…`` references
    resolve against the copied ``figures/`` dir (same contract the LaTeX
    path relies on); ``--embed-resources`` then inlines the images + the
    Latin Modern fonts, so the printed PDF is fully self-contained."""
    log = log or logging.getLogger("frontier_insight.paper")
    browser_name, browser_path = browser
    if not _CSS_PATH.is_file():
        return None, f"HTML theme missing at {_CSS_PATH}"

    work = out_pdf.parent
    try:
        md_text = paper_md.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"could not read {paper_md}: {e}"
    title, body = _split_title(md_text)

    # Resolve the @FONT_DIR@ token to the assets dir's absolute file:/// URL
    # so pandoc --embed-resources can find + inline the OTFs.
    css_text = _CSS_PATH.read_text(encoding="utf-8").replace(
        "@FONT_DIR@", _ASSETS_DIR.as_uri())
    css_resolved = work / "paper_html_theme.css"
    css_resolved.write_text(css_text, encoding="utf-8")
    body_md = work / "paper_html_body.md"
    body_md.write_text(body, encoding="utf-8")
    html_path = work / "paper_html_source.html"

    pandoc_cmd = [
        pandoc_path, body_md.name,
        "--standalone", "--embed-resources", "--mathml",
        "--css", css_resolved.name,
        "--metadata", f"title={title or 'Untitled'}",
        "--metadata", "author=Frontier Insight",
        "--metadata", "pagetitle=paper",
        "-o", html_path.name,
    ]
    try:
        pr = subprocess.run(
            pandoc_cmd, capture_output=True, text=True,
            cwd=str(work), timeout=timeout_s,
        )
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
            br = subprocess.run(
                browser_cmd, capture_output=True, text=True, timeout=timeout_s)
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
