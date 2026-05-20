"""Speech-script generator.

One LLM call from `paper.md` (and optionally a slides outline) to a
~10-minute spoken script. No external tools needed.

When the LLM returns an obviously-bad response (short to the point of
being useless, or a refusal token like ``"I'm sorry, I can't"``), the
generator does NOT ship a broken ``talk.md`` — it writes
``speech_skipped.md`` next to where ``talk.md`` would have landed,
following the same diagnostic shape as ``paper_pdf_skipped.md`` (see
#55) and ``poster_pdf_skipped.md`` (see #145). The user discovers
the skip in the result dict + a self-explanatory markdown file
instead of opening a 20-byte ``talk.md`` and wondering what happened.
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
    PROXY_PROVIDERS,
    resolve_endpoint_async,
)
from generation._skip_md import render_skip_md

_log = logging.getLogger("frontier_insight.speech")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "agents" / "speech.md"

# Minimum useful talk-script length. A ~10-minute spoken script is
# roughly 1500 words / 8000+ chars, but we set the floor low (200) to
# only catch obvious failures: empty output, single-sentence
# refusals, or an LLM that returned just whitespace + a closing
# remark. A legitimately short paper's script may run ~1000 chars;
# we don't want to skip those.
_MIN_TALK_CHARS = 200

# Refusal phrases LLMs emit when they decline a request. Matched
# case-insensitively against the raw response. Substring match (not
# whole-line) because the refusal often appears mid-sentence: "I'm
# sorry, but I cannot help with that request." All checked variants
# are common across OpenAI / Anthropic / Mistral / open-source models.
_REFUSAL_TOKENS: tuple[str, ...] = (
    "sorry, i can't",
    "sorry, i cannot",
    "i'm sorry, i can't",
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i cannot help",
    "i'm unable to",
    "i am unable to",
    "i won't",
    "i will not",
    "i can't help with",
    "i cannot assist",
)


def _is_refusal_or_empty(text: str) -> tuple[bool, str]:
    """Return ``(reject, reason)``. ``reject`` is True when the
    response is too short OR contains a refusal token. ``reason`` is
    a human-readable explanation suitable for the diagnostic file."""
    stripped = text.strip()
    if len(stripped) < _MIN_TALK_CHARS:
        return True, (
            f"LLM returned only {len(stripped)} non-whitespace "
            f"characters (threshold: {_MIN_TALK_CHARS}). A 10-minute "
            f"spoken script should be ~8000 chars; anything under "
            f"{_MIN_TALK_CHARS} is almost certainly an empty/aborted "
            f"completion, not a usable talk."
        )
    low = stripped.lower()
    for token in _REFUSAL_TOKENS:
        if token in low:
            return True, (
                f"LLM response contained a refusal phrase "
                f"(matched: {token!r}). The model declined to generate "
                f"the talk script rather than producing one — most "
                f"often a content-policy false positive triggered by "
                f"the paper's topic phrasing."
            )
    return False, ""


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
            if self.config.provider.name in PROXY_PROVIDERS:
                await sup.release(self.config.provider.name)
            if own_supervisor:
                await sup.shutdown()

        talk_path = out_dir / "talk.md"
        diag_path = out_dir / "speech_skipped.md"
        reject, reason = _is_refusal_or_empty(text)
        if reject:
            # Don't ship empty/refusal talk.md. Wipe any stale file
            # from a prior run (so the user doesn't open it expecting
            # this-run content) and write a diagnostic instead.
            _log.warning(
                "speech: LLM response rejected (%s); writing %s",
                reason, diag_path.name,
            )
            if talk_path.is_file():
                try:
                    talk_path.unlink()
                except OSError:
                    pass
            # Embed the first 500 chars of the raw response so a
            # debugging operator can see what the model actually
            # returned without re-running the (paid, slow) chat call.
            raw_preview = text.strip()[:500]
            diag_path.write_text(
                render_skip_md(
                    kind="speech",
                    reason_code="llm_refused_or_empty",
                    summary=(
                        f"{reason}\n\n"
                        f"Raw LLM response (first 500 chars):\n\n"
                        f"```\n{raw_preview}\n```"
                    ),
                    how_to_fix=(
                        "If the response is empty: retry the quest — "
                        "transient completion failures are the common "
                        "cause. If the response is a refusal: the model "
                        "interpreted the paper topic as policy-sensitive; "
                        "switching providers (`provider.name` in YAML) "
                        "or rephrasing the title often resolves it. The "
                        "speech prompt template lives at "
                        "`agents/speech.md` if a permanent tweak is "
                        "needed."
                    ),
                ),
                encoding="utf-8",
            )
            return {"speech_skipped": diag_path}

        # Success — clean up any stale diagnostic from a prior run so
        # the quest dir doesn't show both talk.md AND
        # speech_skipped.md (mirrors the poster pattern).
        if diag_path.is_file():
            try:
                diag_path.unlink()
            except OSError:
                pass
        talk_path.write_text(text.strip() + "\n", encoding="utf-8")
        _log.info("talk.md written (%d bytes)", talk_path.stat().st_size)
        return {"speech_md": talk_path}
