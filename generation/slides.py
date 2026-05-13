"""Slide deck generator (Phase E-2).

Three render targets, each best-effort and independently skipped if
its CLI isn't installed:

1. LLM call: compress `paper.md` into Marp markdown (8-12 slides) →
   ``slides.md``. Always produced as the source of truth.
2. ``marp slides.md -o slides.{html,pdf}`` — produced when the Marp
   CLI is on PATH.
3. ``pandoc slides.md -o slides.pptx`` — produced when pandoc is on
   PATH. Real PowerPoint file the user can open and edit in Office /
   Google Slides / Keynote, satisfying the common "I want an
   actual presentation, not a markdown file" use case.

The Marp YAML frontmatter at the top of slides.md is harmless to
pandoc (it consumes the leading `---\\nmarp: true\\n...\\n---` block as
document metadata) and slide breaks come from the standalone `---`
horizontal rules between slides — same as Marp uses.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import string
from pathlib import Path

from core.config import Config
from core.engine import QuestArtifacts
from core.provider import (
    LLMClient,
    ProxySupervisor,
    PROXY_PROVIDERS,
    resolve_endpoint_async,
)

_log = logging.getLogger("frontier_insight.slides")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "agents" / "slides.md"


class SlideGenerator:
    def __init__(self, config: Config) -> None:
        self.config = config

    async def generate(
        self,
        art: QuestArtifacts,
        out_dir: Path,
        *,
        supervisor: ProxySupervisor | None = None,
    ) -> dict[str, Path]:
        if "slides" not in self.config.output.kinds or art.paper_md is None:
            return {}

        slides_md = await self._author_marp(art, out_dir, supervisor=supervisor)
        result: dict[str, Path] = {"slides_md": slides_md}

        # Marp branch: HTML + PDF rendering. Use shutil.which's resolved
        # path so Windows `.cmd`/`.bat` shims work — `asyncio.create_
        # subprocess_exec` doesn't apply Windows PATHEXT, so spawning
        # bare "marp" fails on systems where marp lives as marp.CMD.
        marp_exe = shutil.which("marp")
        if marp_exe is None:
            _log.warning("marp CLI not on PATH; slides.html/.pdf skipped")
        else:
            for ext in ("html", "pdf"):
                out_path = out_dir / f"slides.{ext}"
                ok = await _run_cli(
                    [marp_exe, str(slides_md), "-o", str(out_path)],
                    cwd=out_dir, label=f"marp {ext}",
                )
                if ok:
                    result[f"slides_{ext}"] = out_path

        # Pandoc branch: real PowerPoint (.pptx). Same .CMD-on-Windows
        # caveat — resolve via shutil.which.
        pandoc_exe = shutil.which("pandoc")
        if pandoc_exe is None:
            _log.warning("pandoc not on PATH; slides.pptx skipped")
        else:
            pptx_path = out_dir / "slides.pptx"
            # `--slide-level=2` makes H2 headings drive slide breaks,
            # matching the structure the LLM produces (deck title at H1,
            # one H2 per slide). Marp's `---` separators also become
            # slide breaks under pandoc's default settings.
            ok = await _run_cli(
                [pandoc_exe, str(slides_md), "--slide-level=2",
                 "-o", str(pptx_path)],
                cwd=out_dir, label="pandoc pptx",
            )
            if ok:
                result["slides_pptx"] = pptx_path

        return result

    async def _author_marp(
        self,
        art: QuestArtifacts,
        out_dir: Path,
        *,
        supervisor: ProxySupervisor | None,
    ) -> Path:
        paper_md = art.paper_md.read_text(encoding="utf-8") if art.paper_md else ""
        figures = []
        if art.figures_dir and art.figures_dir.is_dir():
            figures = sorted(p.name for p in art.figures_dir.iterdir() if p.is_file())
        prompt = string.Template(PROMPT_PATH.read_text(encoding="utf-8")).substitute(
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

        slides_md = out_dir / "slides.md"
        slides_md.write_text(_strip_outer_fence(text), encoding="utf-8")
        _log.info("slides.md written (%d bytes)", slides_md.stat().st_size)
        return slides_md


async def _run_cli(
    argv: list[str], *, cwd: Path, label: str, timeout_s: float = 120.0,
) -> bool:
    """Run a render CLI and log+swallow any failure. Returns True iff
    the process exited 0. The slide generator wants all three target
    formats to be independent — a failing `marp pdf` should not stop
    `pandoc pptx` from running."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        _log.warning("%s timeout (%.0fs); skipped", label, timeout_s)
        return False
    if proc.returncode != 0:
        _log.warning(
            "%s rc=%d stderr=%s",
            label, proc.returncode,
            stderr.decode("utf-8", errors="replace")[-400:],
        )
        return False
    return True


def _strip_outer_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        # remove first fence line and trailing fence
        first_nl = s.find("\n")
        if first_nl > 0:
            s = s[first_nl + 1 : -3].rstrip()
    return s
