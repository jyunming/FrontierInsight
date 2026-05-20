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

from core.config import Config
from core.engine import QuestArtifacts
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
            "title": "Untitled", "left": "", "middle": "", "right": ""
        }
        # safe_substitute again: the LLM-supplied left/middle/right columns
        # are LaTeX with arbitrary `$math$` content that would break
        # substitute()'s strict placeholder matcher.
        body = string.Template(TEMPLATE_PATH.read_text(encoding="utf-8")).safe_substitute(
            title=parsed.get("title") or "Untitled",
            left=parsed.get("left") or "",
            middle=parsed.get("middle") or "",
            right=parsed.get("right") or "",
        )
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

        # tectonic accepts the same ``.tex`` input as pdflatex; both
        # honour ``-interaction=nonstopmode``. Pass the absolute path to
        # the engine binary (mirroring the paper-generator pattern) so
        # we bypass any PATH-disagreement between the resolving shell
        # and the Python child process.
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
            return result
        if r.returncode != 0:
            # Both pdflatex and tectonic write LaTeX errors to stdout
            # (stderr usually empty); slice both for the diagnostic.
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
            return result
        out_pdf = out_dir / "poster.pdf"
        if out_pdf.exists():
            result["poster_pdf"] = out_pdf
            # Success — remove any stale skip diagnostic from a prior
            # failed run so the quest dir doesn't show both poster.pdf
            # AND poster_pdf_skipped.md (confusing).
            if diag_path.is_file():
                try:
                    diag_path.unlink()
                except OSError:
                    pass
        else:
            # rc=0 but the engine produced no .pdf on disk. Rare —
            # usually filesystem-level (permission on out_dir, or the
            # engine silently swallowed an internal error). Mirrors the
            # analogous branch in ``PaperGenerator._compile_pdf`` so the
            # user gets the same skip-diagnostic shape across both
            # output kinds. Without this, the caller sees a missing
            # ``poster_pdf`` and no ``poster_pdf_skipped.md`` — silent
            # partial success.
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
