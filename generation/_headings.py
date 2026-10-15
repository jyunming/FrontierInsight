"""A heading's own number, taken off at render time.

The writer sometimes types ``## 1. Introduction`` and ``### 3.3 Coverage``. Every
renderer numbers the sections itself (LaTeX's ``\\section``, the browser theme's CSS
counters), so the page printed "1 1. Introduction" and "3.3 3.3 Coverage". The
number comes off the heading in the copy that is rendered; ``paper.md`` is left as
written, and the renderer numbers once.

Only a number that reads as a section number is taken: ``2.`` or ``2.1.`` before
anything, or ``2``, ``2.1`` before a capital letter (``## 2 Methods``, ``### 3.3
Coverage``). A heading that starts with a year or a quantity (``## 2024 outlook``,
``## 10 years later``) is left alone: such a number is more than two digits, or is
followed by a lower-case word. Code fences are skipped.
"""

from __future__ import annotations

import re

# One or two digits per part: "1", "3.3", "2.1.4", with an optional closing dot.
_NUMBER = r"\d{1,2}(?:\.\d{1,2})*"
_HEADING_NUMBER_RE = re.compile(
    r"^(?P<hashes>#{1,6})[ \t]+"
    r"(?:" + _NUMBER + r"\.[ \t]+(?=\S)"
    r"|" + _NUMBER + r"[ \t]+(?=[A-Z]))"
)
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")


def strip_heading_numbers(markdown: str) -> str:
    """``markdown`` with the leading section number taken off each heading."""
    out: list[str] = []
    fenced = False
    for line in markdown.split("\n"):
        if _FENCE_RE.match(line):
            fenced = not fenced
        elif not fenced:
            m = _HEADING_NUMBER_RE.match(line)
            if m:
                line = m.group("hashes") + " " + line[m.end():]
        out.append(line)
    return "\n".join(out)
