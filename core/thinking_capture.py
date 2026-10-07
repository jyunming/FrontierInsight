"""The model's own account of its reasoning, when a transport hands it back.

A step of the engine opens a holder (:func:`open_holder`) around one model call; a transport that receives reasoning
text from the model (``reasoning_content`` from an OpenAI-compatible server, the Claude CLI's thinking blocks, Codex's
reasoning summaries, the VS Code bridge's thinking parts) files it with :func:`note_thinking`. Nothing here is
evidence: the text is what the model said about itself, kept in ``.fi/thinking.jsonl`` for a person to read, and never
sealed or checked.

The holder is a task-local ContextVar, so calls made at the same time never read each other's text, and a caller
that opened no holder (a generator calling the client directly) is not affected: :func:`note_thinking` is then a no-op,
and :func:`wanted` is False, so a connection that has to ask for the reasoning (the VS Code bridge) does not ask for
text nobody would keep.
"""
from __future__ import annotations

import contextvars
from typing import Any

THINKING_FILE = "thinking.jsonl"
THINKING_NOTE = "the model's own account of its reasoning, not evidence"
# One line and the whole file are bounded: reasoning models can write tens of thousands of tokens per call.
THINKING_LINE_CHARS = 64_000
THINKING_FILE_BYTES = 32 * 1024 * 1024

# Connections that cannot hand a model's reasoning text back at all, by provider name, with what to tell the user.
CANNOT_RETURN: dict[str, str] = {
    "gemini_cli": "the Gemini CLI's output has no reasoning in it",
    "antigravity_cli": "the Antigravity CLI (agy) reports only how many tokens the model spent thinking, not the text",
    "copilot_cli": "the Copilot CLI is not read for reasoning (use the VS Code connection for that)",
    "claude_code": "the Claude proxy returns only the answer",
    "github_copilot_cli": "the Copilot proxy returns only the answer",
    "github_copilot_vscode": "the Copilot proxy returns only the answer",
}

_HOLDER: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("fi_thinking_holder", default=None)


def open_holder(*, want: bool = True) -> tuple[dict[str, Any], contextvars.Token]:
    """Start collecting for one model call in this task; returns the holder to read afterwards and the token that
    :func:`close_holder` gives back (so a call made inside another one leaves the outer holder as it was). ``want``
    says whether the reasoning will be kept (``output.save_thinking``): a connection asks for it only then."""
    holder: dict[str, Any] = {"text": "", "want": bool(want), "declined": "", "empty_parts": 0, "extension_older": False,
                              "loop": None, "timings": [], "sampling": ""}
    return holder, _HOLDER.set(holder)


def close_holder(token: contextvars.Token) -> None:
    _HOLDER.reset(token)


def as_text(value: object) -> str | None:
    """Reasoning as one string: a string as it is, a list of strings (a VS Code thinking part's ``value`` may be one,
    as may a summary made of several parts) joined; anything else is not reasoning text (``None``)."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return "".join(value)
    return None


def wanted() -> bool:
    """Whether the call being made in this task will keep the model's reasoning: a holder is open and saving is on."""
    holder = _HOLDER.get()
    return holder is not None and bool(holder.get("want", True))


def note_thinking(text: object) -> None:
    """File one attempt's reasoning text. A later attempt of the same call replaces an earlier one: only the attempt
    whose answer was used is kept."""
    holder = _HOLDER.get()
    value = as_text(text)
    if holder is None or value is None:
        return
    holder["text"] = value


def add_thinking(text: object) -> None:
    """Append streamed reasoning text to the call's holder (start a streamed attempt with ``note_thinking("")``)."""
    holder = _HOLDER.get()
    value = as_text(text)
    if holder is None or value is None:
        return
    holder["text"] += value


def note_declined(reason: object) -> None:
    """The connection asked the model for its reasoning and the model refused the request, so it was asked again
    without it (the VS Code bridge says so on its answer): kept on the holder for the engine to say once in run.log."""
    holder = _HOLDER.get()
    if holder is None or not isinstance(reason, str) or not reason.strip():
        return
    holder["declined"] = reason.strip()


def note_empty_parts(count: object) -> None:
    """The connection passed on ``count`` reasoning parts that carried no text (a GPT model's reasoning reaches VS Code
    encrypted only): kept on the holder for the engine to say once in run.log."""
    holder = _HOLDER.get()
    if holder is None or not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        return
    holder["empty_parts"] = count


def note_extension_older() -> None:
    """The FI extension in VS Code is older than this FI expects (it sent no or a lower bridge protocol): kept on the
    holder for the engine to say once."""
    holder = _HOLDER.get()
    if holder is not None:
        holder["extension_older"] = True


def has_holder() -> bool:
    """A model call of an engine step is open in this task (its holder will be read afterwards)."""
    return _HOLDER.get() is not None


# --- a model whose reasoning goes round in circles -----------------------------------------------------------------
#
# A model that thinks at length can fall into a loop: the same few lines again and again until its context is full (one
# real call ran 32 minutes and 240,000 tokens that way). While the reasoning streams in, :func:`repeating_cycle` looks
# at its end; the thresholds are far above anything a derivation does (the same block of up to 60 lines, ten times in a
# row, at least 400 characters; or the same stretch of up to 800 characters ten times in a row). Over the 85 real
# reasoning texts kept by earlier quests only the one real loop is flagged.

LOOP_MIN_REPEATS = 10      # the same block, this many times in a row
LOOP_MIN_CHARS = 400       # ... and a repeated stretch at least this long (a run of "Wait." lines is not enough)
LOOP_MAX_BLOCK_LINES = 60  # a block of up to this many lines
LOOP_CHAR_WINDOW = 8000    # for text with few lines: the last this many characters ...
LOOP_MAX_CHAR_PERIOD = LOOP_CHAR_WINDOW // LOOP_MIN_REPEATS  # ... made of one stretch (up to 800) repeated
LOOP_CHECK_EVERY = 1500    # characters of new reasoning between two looks
LOOP_TAIL_KEPT = 80_000    # the end of the reasoning that is looked at


class ThinkingLoop(Exception):
    """The model's reasoning went round in circles: ``kind`` ``"lines"`` (a block of ``block`` lines repeated
    ``repeats`` times) or ``"chars"`` (a stretch of ``block`` characters), after ``chars`` characters of reasoning. Raised
    by the streamed Ollama call at once (the stream is closed), caught by the same client, which asks again without
    reasoning; it never reaches a step."""

    def __init__(self, *, kind: str, block: int, repeats: int, chars: int) -> None:
        self.kind, self.block, self.repeats, self.chars = kind, block, repeats, chars
        super().__init__(f"the model's reasoning repeated the same {block} {'lines' if kind == 'lines' else 'characters'} "
                         f"{repeats} times in a row ({chars} characters of reasoning)")


def repeating_cycle(text: str) -> tuple[str, int, int] | None:
    """``("lines", block_lines, repeats)`` or ``("chars", block_chars, repeats)`` when the END of ``text`` is one block
    repeated at least :data:`LOOP_MIN_REPEATS` times in a row, else ``None``. Deterministic; the last line is left out
    (it may still be arriving). Lines are compared with their white space collapsed."""
    complete = text[: text.rfind("\n")] if "\n" in text else ""
    lines = [" ".join(x.split()) for x in complete.split("\n") if x.strip()]
    n = len(lines)
    for k in range(1, min(LOOP_MAX_BLOCK_LINES, n // LOOP_MIN_REPEATS) + 1):
        i, run = n - 1, 0
        while i - k >= 0 and lines[i] == lines[i - k]:
            i, run = i - 1, run + 1
        total = run + k
        if total // k >= LOOP_MIN_REPEATS and sum(len(x) for x in lines[n - total:]) >= LOOP_MIN_CHARS:
            return "lines", k, total // k
    tail = text[-LOOP_CHAR_WINDOW:]
    if len(tail) >= LOOP_CHAR_WINDOW:
        fail = [0] * len(tail)  # the prefix function: the smallest period of the whole tail is len - fail[-1]
        for i in range(1, len(tail)):
            j = fail[i - 1]
            while j and tail[i] != tail[j]:
                j = fail[j - 1]
            fail[i] = j + (tail[i] == tail[j])
        period = len(tail) - fail[-1]
        if period <= LOOP_MAX_CHAR_PERIOD:
            return "chars", period, len(tail) // period
    return None


def note_loop(kind: str, block: int, repeats: int, chars: int) -> None:
    """The model's reasoning at this call did not lead to an answer (it went round in circles, or filled the whole
    answer space): FI asked again without reasoning. Kept on the holder for the engine to say once per step, and the
    reasoning of the discarded attempt is not kept."""
    holder = _HOLDER.get()
    if holder is None:
        return
    holder["loop"] = {"kind": kind, "block": block, "repeats": repeats, "chars": chars}
    holder["text"] = ""


def note_timing(line: str) -> None:
    """One plain line about when a streamed call's output began (kept, at most 8 per call, for the engine's run.log)."""
    holder = _HOLDER.get()
    if holder is not None and isinstance(line, str) and len(holder["timings"]) < 8:
        holder["timings"].append(line)


def note_sampling(text: str) -> None:
    """Which sampling a call that asked the model to think used (kept for the engine to say once per model)."""
    holder = _HOLDER.get()
    if holder is not None and isinstance(text, str) and text:
        holder["sampling"] = text
