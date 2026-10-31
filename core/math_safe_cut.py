"""Cutting a text short without cutting inside a LaTeX math span.

A caption or a quoted sentence that FI shortens is typeset later; a cut that leaves a ``$`` (or ``\(``) open makes
the whole paper fail to typeset. :func:`head_before_math` is the one place that rule lives; callers add their own
ending ("...") and their own word-boundary rule on top of it.
"""
from __future__ import annotations


def _spans(text: str) -> list[tuple[int, int | None]]:
    """The math spans of ``text`` as ``(start, end)``; ``end`` is None for a span never closed. ``\$`` is not a delimiter."""
    out: list[tuple[int, int | None]] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("\(", i):
            j = text.find("\)", i + 2)
            out.append((i, None if j < 0 else j + 2))
            i = n if j < 0 else j + 2
        elif text[i] == "\\":
            i += 2
        elif text[i] == "$":
            delim = "$$" if text.startswith("$$", i) else "$"
            j = i + len(delim)
            while j < n and not text.startswith(delim, j):
                j += 2 if text[j] == "\\" else 1
            out.append((i, None if j >= n else j + len(delim)))
            i = n if j >= n else j + len(delim)
        else:
            i += 1
    return out


def head_before_math(text: str, end: int) -> str:
    """``text[:end]``, unless that ends inside a math span: then the text up to just before the span's opening
    delimiter (right-stripped), or, when nothing with a letter or digit would be left, up to the span's closing
    delimiter. Text without math is returned exactly as ``text[:end]``."""
    head = text[:end]
    if "$" not in head and "\(" not in head:
        return head
    for start, stop in _spans(text):
        if start >= end:
            break
        if stop is None or stop > end:
            before = text[:start].rstrip()
            if any(c.isalnum() for c in before) or stop is None:
                return before if stop is not None else head
            return text[:stop]
    return head
