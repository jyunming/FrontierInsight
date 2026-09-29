"""The model's own account of its reasoning, when a transport hands it back.

A step of the engine opens a holder (:func:`open_holder`) around one model call; a transport that receives reasoning
text from the model (``reasoning_content`` from an OpenAI-compatible server, the Claude CLI's thinking blocks, the
VS Code bridge's thinking parts) files it with :func:`note_thinking`. Nothing here is evidence: the text is what the
model said about itself, kept in ``.fi/thinking.jsonl`` for a person to read, and never sealed or checked.

The holder is a task-local ContextVar, so calls made at the same time never read each other's text, and a caller
that opened no holder (a generator calling the client directly) is not affected: :func:`note_thinking` is then a no-op.
"""
from __future__ import annotations

import contextvars

THINKING_FILE = "thinking.jsonl"
THINKING_NOTE = "the model's own account of its reasoning, not evidence"

_HOLDER: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar("fi_thinking_holder", default=None)


def open_holder() -> dict[str, str]:
    """Start collecting for one model call in this task; returns the holder to read afterwards."""
    holder: dict[str, str] = {"text": ""}
    _HOLDER.set(holder)
    return holder


def close_holder() -> None:
    _HOLDER.set(None)


def note_thinking(text: object) -> None:
    """File one attempt's reasoning text. A later attempt of the same call replaces an earlier one: only the attempt
    whose answer was used is kept."""
    holder = _HOLDER.get()
    if holder is None or not isinstance(text, str):
        return
    holder["text"] = text


def add_thinking(text: object) -> None:
    """Append streamed reasoning text to the call's holder (start a streamed attempt with ``note_thinking("")``)."""
    holder = _HOLDER.get()
    if holder is None or not isinstance(text, str):
        return
    holder["text"] += text
