"""Speech-script generator (Phase E-4).

One LLM call from `paper.md` (and optionally a slides outline) to a
~10-minute spoken script. No external tools needed.
"""

from __future__ import annotations

import logging
import string
from pathlib import Path

from core.config import Config
from core.engine import QuestArtifacts
from core.provider import (
    LLMClient,
    ProxySupervisor,
    _PROXY_PROVIDERS,
    resolve_endpoint_async,
)

_log = logging.getLogger("frontier_insight.speech")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "agents" / "speech.md"


class SpeechGenerator:
    def __init__(self, config: Config) -> None:
        self.config = config

    async def generate(
        self,
        art: QuestArtifacts,
        out_dir: Path,
        *,
        supervisor: ProxySupervisor | None = None,
    ) -> dict[str, Path]:
        if "speech" not in self.config.output.kinds or art.paper_md is None:
            return {}

        paper_md = art.paper_md.read_text(encoding="utf-8")
        slides_outline = ""
        slides_path = out_dir / "slides.md"
        if slides_path.exists():
            slides_outline = slides_path.read_text(encoding="utf-8")[:4000]

        prompt = string.Template(PROMPT_PATH.read_text(encoding="utf-8")).substitute(
            paper_md=paper_md[:8000],
            slides_outline=slides_outline or "(no slide deck available)",
        )
        own_supervisor = supervisor is None
        sup = supervisor or ProxySupervisor()
        endpoint = await resolve_endpoint_async(self.config.provider, sup)
        client = LLMClient(endpoint)
        try:
            text = await client.chat([{"role": "user", "content": prompt}], temperature=0.3)
        finally:
            await client.aclose()
            if self.config.provider.name in _PROXY_PROVIDERS:
                await sup.release(self.config.provider.name)
            if own_supervisor:
                await sup.shutdown()

        talk_path = out_dir / "talk.md"
        talk_path.write_text(text.strip() + "\n", encoding="utf-8")
        _log.info("talk.md written (%d bytes)", talk_path.stat().st_size)
        return {"speech_md": talk_path}
