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

# Refusal phrases LLMs emit when they decline a REQUEST FOR HELP.
# Matched case-insensitively against the raw response. Each token
# includes assist/help-with-request language — that's the load-
# bearing constraint that distinguishes a refusal from a legitimate
# script line. Generic phrases like ``"I won't"`` or
# ``"I'm unable to"`` are excluded because a real talk script
# routinely says things like ``"I won't cover X today"`` or
# ``"the model was unable to converge"`` — those would trigger
# false positives and discard usable output.
#
# Additionally we only match these near the START of the response
# (within the first ~400 chars). A refusal sits at the top of the
# message ("I'm sorry, I can't help with that. — apology then
# stop"); a script that mentions "sorry" in a quoted passage 6
# paragraphs in should not be rejected.
_REFUSAL_TOKENS: tuple[str, ...] = (
    "sorry, i can't help",
    "sorry, i cannot help",
    "i'm sorry, i can't help",
    "i'm sorry, but i can't help",
    "i'm sorry, but i cannot help",
    "i cannot help with",
    "i can't help with",
    "i cannot assist",
    "i can't assist",
    "i'm unable to help",
    "i am unable to help",
    "i'm unable to assist",
    "i cannot provide",
    "i can't provide",
)
# Window (in characters) within which a refusal token counts.
# A refusal phrase appearing 1000 chars into a talk script is
# almost certainly a quoted line, not the model declining.
_REFUSAL_SCAN_PREFIX = 400


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
    # Only scan the leading prefix — a refusal sits at the TOP of
    # the response; a later occurrence is almost certainly a quoted
    # script line ("the participant said 'I can't help...'", etc.)
    # and shouldn't trigger a reject.
    head = stripped[:_REFUSAL_SCAN_PREFIX].lower()
    for token in _REFUSAL_TOKENS:
        if token in head:
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
        # Cleanup gate: if "speech" is no longer in output.kinds (user
        # removed it from their YAML between runs), remove any stale
        # ``speech_skipped.md`` left over from a prior run. Mirrors
        # PaperGenerator's cleanup when paper_pdf is dropped from
        # kinds. Without this, the stale diagnostic persists forever
        # after the user opts out of the speech kind.
        if "speech" not in self.config.output.kinds:
            stale = out_dir / "speech_skipped.md"
            if stale.is_file():
                try:
                    stale.unlink()
                except OSError:
                    pass
            return {}
        if art.paper_md is None:
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
            # Pick a code-fence length that doesn't collide with any
            # run of backticks in the embedded preview. The default
            # triple-backtick fence breaks the markdown structure when
            # the LLM response itself contains ``` (common for code-
            # heavy refusals or partial responses). Find the longest
            # run of backticks in the preview and use one more.
            longest_backtick_run = 0
            current_run = 0
            for ch in raw_preview:
                if ch == "`":
                    current_run += 1
                    longest_backtick_run = max(longest_backtick_run, current_run)
                else:
                    current_run = 0
            fence = "`" * max(3, longest_backtick_run + 1)
            diag_path.write_text(
                render_skip_md(
                    requested_kind="speech",
                    display_name="speech (talk.md)",
                    reason_code="llm_refused_or_empty",
                    summary=(
                        f"{reason}\n\n"
                        f"Raw LLM response (first 500 chars):\n\n"
                        f"{fence}\n{raw_preview}\n{fence}"
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
