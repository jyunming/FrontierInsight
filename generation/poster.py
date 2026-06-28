"""Poster generator.

LLM populates 3 columns of a fixed 36"x48" beamerposter template; we
compile via pdflatex (or tectonic — the 3-tier fallback is shared
with the paper generator via ``generation/_pdf_engine.py``). If no
LaTeX engine is reachable, only ``poster.tex`` is produced and a
``poster_pdf_skipped.md`` diagnostic is written next to it so the
user discovers the skip without grepping run.log.
"""

from __future__ import annotations

import json
import logging
import shutil  # noqa: F401 — existing tests monkeypatch ``poster.shutil.which``
import string
import subprocess
from pathlib import Path

import re as _re

from core.config import Config
from core.engine import (
    QuestArtifacts,
    build_references,
    render_poster_references_latex,
)
from generation.paper import _sanitize_unicode_for_latex

# Emoji / pictographs / dingbats / regional-indicator ranges. pdflatex
# (utf8 inputenc) hard-errors on these ("Unicode character … not set up
# for use with LaTeX"), and an LLM column or a scraped source title can
# easily carry one (a "📊 Global EV Sales Report" citation is what first
# broke poster.pdf). We strip them outright — they carry no meaning the
# poster needs.
_EMOJI_RE = _re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoji + symbols + pictographs
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U00002B00-\U00002BFF"   # misc symbols and arrows
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U00002190-\U000021FF"   # arrows
    "]+",
    flags=_re.UNICODE,
)


def _latex_safe_body(body: str) -> str:
    """Make a fully-substituted poster.tex safe for pdflatex: map common
    Unicode typography to LaTeX equivalents (shared with the paper
    generator) and strip emoji/pictographs that have no LaTeX mapping."""
    return _EMOJI_RE.sub("", _sanitize_unicode_for_latex(body))


# Bare LaTeX specials that hard-error pdflatex when an LLM writes them as
# literal prose in a poster column (a stray 'Microlensing & Timing' / '50%').
# Posters are blocks + itemize, never tabular, so a bare ``&`` is always a
# literal ampersand. Already-escaped specials (``\&``) are skipped via the
# negative lookbehind; backslashes and braces are untouched. A bare
# ``&``/``%``/``#`` inside ``$math$`` would also be escaped, but poster column
# prose doesn't carry math with literal specials, so that's safe here.
_LATEX_TEXT_SPECIAL_RE = _re.compile(r"(?<!\\)([&%#])")


def _escape_latex_text_specials(s: str) -> str:
    """Escape bare ``&``, ``%``, ``#`` in LLM-authored poster column text so
    a literal ampersand or percent doesn't fatally break the compile."""
    return _LATEX_TEXT_SPECIAL_RE.sub(r"\\\1", s or "")


# LaTeX scratch pdflatex/tectonic leave next to poster.pdf — pure compile
# byproducts. We delete them once we have the PDF so the quest dir holds the
# deliverables (poster.pdf + poster.tex), not a pile of poster.aux/.out/…
_POSTER_AUX_EXTS = (
    ".aux", ".out", ".nav", ".snm", ".toc", ".vrb",
    ".fls", ".fdb_latexmk", ".synctex.gz",
)


def _cleanup_poster_artifacts(out_dir: Path, *, keep_log: bool) -> None:
    """Delete the LaTeX scratch files beside poster.pdf, keeping the source
    (poster.tex) and — on a failed compile (``keep_log=True``) — poster.log
    for debugging. Best-effort: a locked file is skipped."""
    exts = list(_POSTER_AUX_EXTS)
    if not keep_log:
        exts.append(".log")
    for ext in exts:
        p = out_dir / f"poster{ext}"
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass


from core.provider import (
    LLMClient,
    ProxySupervisor,
    PROXY_PROVIDERS,
    resolve_endpoint_async,
)
from generation._pdf_engine import find_pdf_engine

_log = logging.getLogger("frontier_insight.poster")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "agents" / "poster.md"
TEMPLATE_PATH = REPO_ROOT / "templates" / "poster" / "poster.tex"
# Frontier Insight brand mark — copied next to poster.tex at compile time
# so the `$brandmark` placeholder can \includegraphics it as a subtle
# bottom-left corner mark (matches the slide/paper branding).
ICON_PNG = REPO_ROOT / "vscode-frontier-insight" / "images" / "fi_glyph_teal.png"

# Header brand LOCKUP for the headline band — the palette-native teal
# glyph + a monospace wordmark (matches the slide deck's bottom-left
# lockup, scaled up for the poster's masthead). Referenced relative to
# the poster's working dir, where the generator copies ``fi_icon.png``.
_POSTER_BRANDLOGO = (
    r"\raisebox{-0.30\height}{\includegraphics[height=1.7cm]{fi_icon.png}}"
    r"\hspace{0.4em}{\ttfamily\bfseries\color{fiteal}\Large FRONTIER INSIGHT}"
)

# Small mark prepended to the Sources band — the same teal glyph raised
# to sit on the band's baseline, a quiet second copy of the masthead
# lockup. Height kept modest so it reads as a corner mark, not a
# competing element.
_POSTER_ICON_TEX = (
    r"\raisebox{-0.26\height}{\includegraphics[height=0.95cm]{fi_icon.png}}"
    r"\hspace{0.45em}"
)


def _poster_rasteriser_available() -> bool:
    """True when poster overflow can be detected — i.e. a PDF rasteriser
    is reachable (PyMuPDF importable, or the ``pdftoppm`` CLI on PATH).
    When False the generator compiles once at a conservative default
    scale instead of auto-fitting, since it can't measure the result."""
    try:
        import fitz  # noqa: F401  (PyMuPDF)
        return True
    except Exception:
        return shutil.which("pdftoppm") is not None


def _render_poster_page(pdf_path: Path, out_dir: Path):
    """Rasterise page 1 of ``pdf_path`` to a PIL image at ~40 DPI. Tries
    PyMuPDF (in-process) first, then the ``pdftoppm`` CLI. Returns
    ``None`` when neither is available or rendering fails — the caller
    treats ``None`` as 'overflow undetectable, assume the page is fine'."""
    try:
        from PIL import Image
    except Exception:
        return None
    try:  # 1) PyMuPDF — no external binary.
        import fitz
        doc = fitz.open(str(pdf_path))
        try:
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(40 / 72, 40 / 72))
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            doc.close()
    except Exception:
        pass
    exe = shutil.which("pdftoppm")  # 2) pdftoppm CLI.
    if exe is None:
        return None
    prefix = out_dir / "_poster_fitcheck"
    try:
        subprocess.run(
            [exe, "-png", "-r", "40", "-singlefile", str(pdf_path), str(prefix)],
            capture_output=True, timeout=60,
        )
        png = prefix.with_suffix(".png")
        if png.is_file():
            img = Image.open(png).copy()
            try:
                png.unlink()
            except OSError:
                pass
            return img
    except Exception:
        pass
    return None


def _poster_overflows(pdf_path: Path, out_dir: Path) -> bool:
    """Detect whether the compiled poster overflows the A1 page.

    beamerposter SILENTLY clips content that exceeds the page height — it
    emits no ``Overfull \\vbox`` warning — so we rasterise page 1 and
    check whether dark content reaches the very bottom of the page. A
    fitting poster leaves a cream margin strip there; an overflowing one
    is clipped flush to the bottom edge. Returns ``False`` ('can't tell,
    assume fine') when no rasteriser is available.

    Calibrated on a known overflow vs. fit poster at ~40 DPI: the darkest
    pixel in the bottom 1%% of the page is ~0-24 when clipped (content)
    and ~203 (cream) when it fits; 130 cleanly separates the two."""
    img = _render_poster_page(pdf_path, out_dir)
    if img is None:
        return False
    g = img.convert("L")
    w, h = g.size
    rows = max(1, int(h * 0.010))
    lo, _hi = g.crop((0, h - rows, w, h)).getextrema()
    return lo < 130


def _copy_poster_icon(out_dir: Path) -> bool:
    """Copy the FI teal glyph into ``out_dir`` as ``fi_icon.png`` so both
    the headline lockup and the Sources-band mark can \\includegraphics it.
    Returns ``True`` on success, ``False`` (no brand mark) if the asset is
    missing or can't be copied — the poster still compiles either way
    because the header lockup is gated on this and the band mark is only
    injected when the copy succeeded."""
    if not ICON_PNG.is_file():
        return False
    try:
        shutil.copyfile(ICON_PNG, out_dir / "fi_icon.png")
        return True
    except OSError:
        return False


def _brand_references_band(references_tex: str, icon_ok: bool) -> str:
    """Prepend the small teal glyph to the start of the Sources band (a
    bottom-left corner mark). Placing it at the *start* of the band keeps
    it co-located with the citations so a long source list can never push
    it off the page. Returns ``references_tex`` unchanged when the icon
    wasn't copied; when the band itself is empty, emits a minimal
    icon-only band so the poster still carries the mark."""
    if not icon_ok:
        return references_tex
    anchor = r"{\scriptsize\textbf{Sources:}"
    if anchor in references_tex:
        return references_tex.replace(anchor, _POSTER_ICON_TEX + anchor, 1)
    if references_tex.strip():
        # Band present but in an unexpected shape — prepend the icon so
        # the mark still renders rather than silently dropping it.
        return _POSTER_ICON_TEX + references_tex
    # No sources at all: emit a minimal icon-only band so the poster is
    # still branded in the bottom-left corner.
    return (
        "\\vspace{0.4em}\\hrule\\vspace{0.3em}\n"
        + _POSTER_ICON_TEX
        + r"{\scriptsize\color{fimuted}\ttfamily FRONTIER INSIGHT}" + "\n"
    )


class PosterGenerator:
    def __init__(self, config: Config) -> None:
        self.config = config

    async def generate(
        self,
        art: QuestArtifacts,
        out_dir: Path,
        *,
        supervisor: ProxySupervisor | None = None,
    ) -> dict[str, Path]:
        if "poster" not in self.config.output.kinds or art.paper_md is None:
            return {}

        paper_md = art.paper_md.read_text(encoding="utf-8")
        figures = []
        if art.figures_dir and art.figures_dir.is_dir():
            figures = sorted(p.name for p in art.figures_dir.iterdir() if p.is_file())
        # safe_substitute (rather than substitute) because the inputs that
        # land here aren't fully trusted: paper_md is LLM-authored prose
        # which may contain stray `$` (LaTeX inline math, currency, etc.),
        # and substitute() raises ValueError on unmatched `$`.
        prompt = string.Template(PROMPT_PATH.read_text(encoding="utf-8")).safe_substitute(
            paper_md=paper_md[:8000],
            figure_list="\n".join(f"- figures/{f}" for f in figures) or "(none)",
        )
        own_supervisor = supervisor is None
        sup = supervisor or ProxySupervisor()
        endpoint = await resolve_endpoint_async(self.config.provider, sup)
        client = LLMClient(endpoint)
        try:
            text = await client.chat([{"role": "user", "content": prompt}], temperature=0.2)
        finally:
            await client.aclose()
            if self.config.provider.name in PROXY_PROVIDERS:
                await sup.release(self.config.provider.name)
            if own_supervisor:
                await sup.shutdown()

        parsed = _lenient_json(text) or {
            "title": "Untitled", "left": "", "right": ""
        }
        # Sources band — built from the quest's actual retrieved
        # literature (web pages carry citable URLs) and injected straight
        # into the template, NOT left to the LLM, so the poster always
        # carries its references even when the 8000-char paper.md slice
        # the LLM saw cut the References section off the end.
        refs = build_references(
            art.raw_state.get("literature") or [],
            audience=self.config.output.audience,
        )
        references_tex = render_poster_references_latex(refs)
        # Copy the teal glyph next to poster.tex, then brand both the
        # masthead (header lockup) and the Sources band (a small corner
        # mark at the start of the band, so a long citation list can't
        # push it off-page). Both are gated on the copy succeeding.
        icon_ok = _copy_poster_icon(out_dir)
        references_tex = _brand_references_band(references_tex, icon_ok)
        brandlogo = _POSTER_BRANDLOGO if icon_ok else ""

        # safe_substitute again: the LLM-supplied left/middle/right columns
        # are LaTeX with arbitrary `$math$` content that would break
        # substitute()'s strict placeholder matcher.
        body = string.Template(TEMPLATE_PATH.read_text(encoding="utf-8")).safe_substitute(
            title=_escape_latex_text_specials(parsed.get("title") or "Untitled"),
            left=_escape_latex_text_specials(parsed.get("left") or ""),
            right=_escape_latex_text_specials(parsed.get("right") or ""),
            references=references_tex,   # already LaTeX-escaped by build step
            brandlogo=brandlogo,
        )
        # Make the substituted LaTeX safe for pdflatex — strip emoji and
        # map Unicode typography that would otherwise hard-error the
        # compile (an LLM column or a scraped citation title can carry a
        # stray 📊 / ≈ / smart-quote).
        body = _latex_safe_body(body)
        poster_tex = out_dir / "poster.tex"
        poster_tex.write_text(body, encoding="utf-8")
        result: dict[str, Path] = {"poster_tex": poster_tex}

        # Use the SAME 3-tier engine fallback the paper generator uses
        # (pdflatex → system tectonic → repo-local tools/tectonic[.exe])
        # so a user who followed the documented no-admin install
        # (``python launch.py --install-tectonic``) gets both
        # ``paper.pdf`` AND ``poster.pdf`` rather than discovering the
        # poster silently skipped while the paper compiled. The shared
        # implementation lives in ``generation/_pdf_engine.py``.
        engine = find_pdf_engine()
        diag_path = out_dir / "poster_pdf_skipped.md"
        if engine is None:
            msg = (
                "no LaTeX engine found (pdflatex or tectonic); poster.pdf "
                "skipped. Run `python launch.py --install-tectonic` for a "
                "no-admin LaTeX install."
            )
            _log.warning(msg)
            diag_path.write_text(
                _render_poster_skip_md(
                    code="no_latex_engine",
                    summary=msg,
                    how_to_fix=(
                        "Easiest no-admin path: run `python launch.py "
                        "--install-tectonic` from the repo root. That drops a "
                        "single self-bootstrapping LaTeX binary (~70 MB) into "
                        "`tools/`; FI auto-detects it on the next quest. "
                        "Standard alternative: install MiKTeX (Windows) or "
                        "TeX Live (macOS/Linux) so `pdflatex` lands on PATH."
                    ),
                ),
                encoding="utf-8",
            )
            result["poster_pdf_skipped"] = diag_path
            return result
        engine_name, engine_path = engine
        _log.info("poster.pdf: using engine=%s at %s", engine_name, engine_path)

        # --- Length-aware auto-fit -------------------------------------
        # beamerposter SILENTLY clips content that overflows the A1 page
        # (no Overfull warning), so we compile at descending ``scale=``
        # values and rasterise each result until the page no longer
        # overflows. When no rasteriser (PyMuPDF / pdftoppm) is reachable
        # we can't measure the result, so we compile ONCE at a
        # conservative default scale instead of the template's maximum.
        # tectonic accepts the same ``.tex`` input as pdflatex; both
        # honour ``-interaction=nonstopmode``. The absolute engine path
        # (mirroring the paper-generator pattern) bypasses any
        # PATH-disagreement between the resolving shell and the child.
        out_pdf = out_dir / "poster.pdf"
        scales = (
            [1.15, 1.06, 0.98, 0.90, 0.83, 0.76]
            if _poster_rasteriser_available() else [1.0]
        )
        for idx, s in enumerate(scales):
            # The template ships ``scale=1.15``; rewrite it for this pass.
            poster_tex.write_text(
                body.replace("scale=1.15", f"scale={s:.2f}", 1), encoding="utf-8",
            )
            try:
                r = subprocess.run(
                    [
                        engine_path,
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        str(poster_tex),
                    ],
                    capture_output=True, text=True, cwd=str(out_dir), timeout=180,
                )
            except subprocess.TimeoutExpired:
                msg = f"{engine_name} poster timeout (>180s); skipped"
                _log.warning(msg)
                diag_path.write_text(
                    _render_poster_skip_md(
                        code=f"{engine_name}_timeout",
                        summary=msg,
                        how_to_fix=(
                            f"The {engine_name} compile took longer than 180s. "
                            f"On a fresh tectonic install the first run downloads "
                            f"CTAN packages (~30 s); retry once the cache is "
                            f"populated. If it consistently times out, raise the "
                            f"timeout in `generation/poster.py`."
                        ),
                    ),
                    encoding="utf-8",
                )
                result["poster_pdf_skipped"] = diag_path
                _cleanup_poster_artifacts(out_dir, keep_log=True)
                return result
            if r.returncode != 0:
                # Both pdflatex and tectonic write LaTeX errors to stdout
                # (stderr usually empty); slice both for the diagnostic.
                # A compile error is scale-independent, so fail fast.
                msg = (
                    f"{engine_name} poster rc={r.returncode}; poster.pdf skipped"
                )
                _log.warning(
                    "%s stdout=%s stderr=%s",
                    msg, r.stdout[-500:], r.stderr[-200:],
                )
                diag_path.write_text(
                    _render_poster_skip_md(
                        code=f"{engine_name}_rc_{r.returncode}",
                        summary=msg,
                        how_to_fix=(
                            f"The LaTeX engine errored on `poster.tex`. The most "
                            f"common cause is a missing `beamerposter` package on "
                            f"a fresh MiKTeX/TeX Live install; tectonic auto-fetches "
                            f"it on first compile. stdout tail (last 500 chars):"
                            f"\n\n```\n{r.stdout[-500:]}\n```"
                        ),
                    ),
                    encoding="utf-8",
                )
                result["poster_pdf_skipped"] = diag_path
                _cleanup_poster_artifacts(out_dir, keep_log=True)
                return result
            if not out_pdf.exists():
                # rc=0 but the engine produced no .pdf on disk. Rare —
                # usually filesystem-level (permission on out_dir, or the
                # engine silently swallowed an internal error). Mirrors the
                # analogous branch in ``PaperGenerator._compile_pdf``.
                msg = (
                    f"{engine_name} returned rc=0 but `poster.pdf` is not on "
                    f"disk in {out_dir}. The subprocess reported success but "
                    f"produced no output file."
                )
                _log.warning(msg)
                diag_path.write_text(
                    _render_poster_skip_md(
                        code="output_missing_after_success",
                        summary=msg,
                        how_to_fix=(
                            f"Most likely a filesystem-level issue: "
                            f"``{out_dir}`` may not be writable by the FI "
                            f"process, or an antivirus/sync tool deleted "
                            f"the file between subprocess exit and our "
                            f"existence check. Verify the directory is "
                            f"writable, re-run the quest, and if the "
                            f"problem persists try running `{engine_name} "
                            f"poster.tex` manually from {out_dir} to "
                            f"isolate whether it's an engine bug or an "
                            f"environment one."
                        ),
                    ),
                    encoding="utf-8",
                )
                result["poster_pdf_skipped"] = diag_path
                _cleanup_poster_artifacts(out_dir, keep_log=True)
                return result
            # Compiled OK at this scale. Accept it unless it overflows AND
            # a smaller scale remains to try.
            at_floor = idx == len(scales) - 1
            if at_floor or not _poster_overflows(out_pdf, out_dir):
                if s != 1.15:
                    _log.info(
                        "poster.pdf: auto-fit to scale=%.2f%s", s,
                        " (scale floor reached)" if at_floor else "",
                    )
                break
            _log.info(
                "poster.pdf: scale=%.2f overflows the A1 page; retrying smaller",
                s,
            )

        result["poster_pdf"] = out_pdf
        # Success — remove any stale skip diagnostic from a prior failed
        # run so the quest dir doesn't show both poster.pdf AND
        # poster_pdf_skipped.md (confusing).
        if diag_path.is_file():
            try:
                diag_path.unlink()
            except OSError:
                pass
        # Drop the LaTeX scratch (.aux/.log/.out/…) now that poster.pdf
        # exists — keep only poster.pdf + poster.tex.
        _cleanup_poster_artifacts(out_dir, keep_log=False)
        return result


def _render_poster_skip_md(*, code: str, summary: str, how_to_fix: str) -> str:
    """Render a ``poster_pdf_skipped.md`` diagnostic. Mirrors the shape
    of ``paper_pdf_skipped.md`` (see ``generation/paper.py``) so a user
    sees the same structure for both skip surfaces — reason code, what
    happened, how to fix it."""
    return (
        f"# poster.pdf was requested but not produced\n\n"
        f"Your `output.kinds` included `poster`, but the poster generator "
        f"couldn't compile a PDF. The `poster.tex` source is still on disk "
        f"next to this file — compile it manually once the prerequisites "
        f"below are in place.\n\n"
        f"## What happened\n\n"
        f"**Reason code:** `{code}`\n\n"
        f"{summary}\n\n"
        f"## How to fix it\n\n"
        f"{how_to_fix}\n\n"
        f"This file is auto-deleted on the next successful poster.pdf compile.\n"
    )


def _lenient_json(text: str) -> dict | None:
    s = text.strip()
    # Strip fence if present.
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0 and s.endswith("```"):
            s = s[nl + 1 : -3].strip()
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                return None
        return None
