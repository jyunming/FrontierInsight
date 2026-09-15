"""A pipe table the way pandoc reads one.

Pandoc starts a pipe table only after a blank line. The writer often puts the
table right under its caption line (``**Table 1.** What it shows`` followed by
``| a | b |``), and pandoc then reads the whole table as more text of the
caption's paragraph: paper.pdf printed the rows as raw pipes. The blank line
goes in at render time; paper.md keeps what the writer wrote.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
# A row of cells between pipes, and the rule under a table's header row
# ("| --- | :---: |", with or without the outer pipes).
_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_RULE_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")


def blank_line_before_tables(markdown: str) -> str:
    """``markdown`` with a blank line above each pipe table that follows a line
    of text directly. A table's header row is a pipe row whose next line is a
    rule. Code blocks are left alone."""
    lines = markdown.split("\n")
    out: list[str] = []
    fence = None
    for i, line in enumerate(lines):
        m = _FENCE_RE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        elif (
            fence is None
            and _ROW_RE.match(line)
            and i + 1 < len(lines)
            and _RULE_RE.match(lines[i + 1])
            and out
            and out[-1].strip()
            and not _ROW_RE.match(out[-1])
        ):
            out.append("")
        out.append(line)
    return "\n".join(out)
