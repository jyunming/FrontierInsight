"""Slide deck generator (Phase E-2).

Two steps:
1. LLM call: compress `paper.md` into Marp markdown (8-12 slides).
2. Shell out to the `marp` CLI to render `slides.html` and `slides.pdf`.

Skips cleanly if `marp` CLI is absent — only `slides.md` is produced.
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
        if shutil.which("marp") is None:
            _log.warning("marp CLI not on PATH; slides.html/.pdf skipped")
            return result

        for ext in ("html", "pdf"):
            out_path = out_dir / f"slides.{ext}"
            try:
                proc = await asyncio.create_subprocess_exec(
                    "marp", str(slides_md), "-o", str(out_path),
                    cwd=str(out_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                _log.warning("marp %s timeout; skipped", ext)
                continue
            if proc.returncode != 0:
                _log.warning(
                    "marp %s rc=%d stderr=%s",
                    ext, proc.returncode, stderr.decode("utf-8", errors="replace")[-400:],
                )
                continue
            result[f"slides_{ext}"] = out_path
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


def _strip_outer_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        # remove first fence line and trailing fence
        first_nl = s.find("\n")
        if first_nl > 0:
            s = s[first_nl + 1 : -3].rstrip()
    return s
