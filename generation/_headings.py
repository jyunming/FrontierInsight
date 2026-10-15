"""A heading's own section number, taken off at render time when the renderer numbers it.

The writer sometimes types ``## 1. Introduction`` and ``### 3.3 Coverage``. The LaTeX
templates number sections themselves (``\\section``), as does the browser theme's CSS,
so the page printed "1 1. Introduction" and "3.3 3.3 Coverage". The number comes off
the heading in the copy that is rendered; ``paper.md`` is left as written.

A number is taken only when it is a section number: it continues the document's own
numbering (the first numbered section is 1, the next 2, ...; a subsection of section 3
is 3.1, then 3.2, ...), and its heading sits at a level the renderer numbers. A count
that is part of a title ("3 Body Problem", "10 Years Later", "2.5 Million Cells") does
not continue that sequence, so it stays. At a level the renderer does not number (the
briefing theme numbers none; LaTeX stops below ``secnumdepth``) the writer's number is
the only one, so it stays too. The forms ``1.``, ``1``, ``1)``, ``1:``, ``3.2``,
``3.2.`` and Roman ``I.`` are read. The title (the first level-1 heading) is never
touched, and neither is anything inside a code fence.
"""

from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<text>.*?)[ \t]*$")
# "3", "3.2", "3.2.1" (one or two digits a part) or a Roman numeral, then ".", ")", ":"
# or nothing (a Roman numeral needs its dot), then the heading's words.
_NUMBER_RE = re.compile(
    r"^(?:(?P<arabic>\d{1,2}(?:\.\d{1,2})*)(?P<sep>[.):])?|(?P<roman>[IVXLC]{1,6})\.)[ \t]+(?=\S)"
)
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_value(text: str) -> int | None:
    total = 0
    for i, ch in enumerate(text):
        value = _ROMAN[ch]
        total += -value if i + 1 < len(text) and _ROMAN[text[i + 1]] > value else value
    return total if total > 0 and _to_roman(total) == text else None


def _to_roman(n: int) -> str:
    out = ""
    for value, letters in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
                           (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out += letters
            n -= value
    return out


def _number(text: str) -> tuple[tuple[int, ...], int] | None:
    """The section number at the start of a heading's text and its length, or None."""
    m = _NUMBER_RE.match(text)
    if m is None:
        return None
    if m.group("roman"):
        value = _roman_value(m.group("roman"))
        return ((value,), m.end()) if value is not None else None
    parts = tuple(int(p) for p in m.group("arabic").split("."))
    return parts, m.end()


def strip_heading_numbers(markdown: str, *, numbered_depth: int = 3) -> str:
    """``markdown`` with each heading's own section number taken off where the renderer
    numbers that heading: ``numbered_depth`` levels from the top section level (the
    highest heading level after the ``#`` title); 0 takes nothing off."""
    lines = markdown.split("\n")
    headings: list[tuple[int, int, str]] = []   # (line index, level, text)
    fenced = False
    for index, line in enumerate(lines):
        if _FENCE_RE.match(line):
            fenced = not fenced
            continue
        m = None if fenced else _HEADING_RE.match(line)
        if m:
            headings.append((index, len(m.group("hashes")), m.group("text")))
    if numbered_depth <= 0 or not headings:
        return markdown
    has_title = headings[0][1] == 1
    title_index = headings[0][0] if has_title else None
    sections = [level for index, level, _text in headings if index != title_index]
    if not sections:
        return markdown
    top = min(sections)

    # The document's own numbering: the last number taken at each depth.
    current: list[int] = []
    for index, level, text in headings:
        if index == title_index or level < top:
            continue
        depth = level - top + 1
        if depth > numbered_depth:
            continue
        found = _number(text)
        if found is None:
            continue
        parts, length = found
        if len(parts) != depth:
            continue
        parent, last = current[:depth - 1], current[depth - 1] if len(current) >= depth else 0
        if list(parts[:-1]) != parent or parts[-1] != last + 1:
            continue
        current = list(parts)
        hashes = "#" * level
        lines[index] = hashes + " " + text[length:]
    return "\n".join(lines)
