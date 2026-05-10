"""Poster generator (Phase E-3).

LLM populates 3 columns of a fixed 36"x48" beamerposter template; we
compile via pdflatex. If pdflatex is missing, only `poster.tex` is
produced.
"""

from __future__ import annotations

import json
import logging
import shutil
import string
import subprocess
from pathlib import Path

from core.config import Config
from core.engine import QuestArtifacts
from core.provider import (
    LLMClient,
    ProxySupervisor,
    _PROXY_PROVIDERS,
    resolve_endpoint_async,
)

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
            if self.config.provider.name in _PROXY_PROVIDERS:
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

        if shutil.which("pdflatex") is None:
            _log.warning("pdflatex not on PATH; poster.pdf skipped")
            return result

        try:
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(poster_tex)],
                capture_output=True, text=True, cwd=str(out_dir), timeout=180,
            )
        except subprocess.TimeoutExpired:
            _log.warning("pdflatex poster timeout; skipped")
            return result
        if r.returncode != 0:
            # pdflatex writes diagnostics to stdout; stderr usually empty.
            _log.warning(
                "pdflatex poster rc=%d stdout=%s stderr=%s",
                r.returncode, r.stdout[-500:], r.stderr[-200:],
            )
            return result
        out_pdf = out_dir / "poster.pdf"
        if out_pdf.exists():
            result["poster_pdf"] = out_pdf
        return result


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
