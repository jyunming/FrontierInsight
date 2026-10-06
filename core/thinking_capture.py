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
    holder: dict[str, Any] = {"text": "", "want": bool(want), "declined": ""}
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
