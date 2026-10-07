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


# ---------------------------------------------------------------------------
# A table that fits its column

# The formats whose text is set in two columns (the template's ``twocolumn``).
TWO_COLUMN_FORMATS = frozenset({"ieee_access"})
# How many characters of 10 pt body text a column holds: 3.4 in in two columns,
# 6.5 in in one. Each column of a table also gives up about 2.4 characters to
# the space either side of its cell.
_COLUMN_CHARS = {True: 46.0, False: 92.0}
_CELL_PAD_CHARS = 2.4
# Pandoc wraps a pipe table's cells, with the columns' relative widths taken
# from the dashes under the header, only when some line of the table is longer
# than this (its ``--columns``); otherwise the table keeps the natural width of
# its text and runs on past the column.
_PANDOC_COLUMNS = 72
# What the table is set in when its longest words do not fit side by side at
# the body size: (LaTeX size, how much more text a line holds than at 10 pt).
_SMALLER = (("", 1.0), (r"\footnotesize", 1.25), (r"\scriptsize", 1.5))
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_CAPTION_RE = re.compile(r"^\s*(?::|Table:)\s")
_MAX_WEIGHT = 60


def _cells(row: str) -> list[str]:
    inner = row.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|") and not inner.endswith("\\|"):
        inner = inner[:-1]
    return [c.strip() for c in _CELL_SPLIT.split(inner)]


def _shown(cell: str) -> str:
    """The cell as it prints, roughly: markdown and TeX marks are not characters."""
    return re.sub(r"\s+", " ", re.sub(r"\\[A-Za-z]+|[*`$\\{}]", "", cell.replace("\\|", "|"))).strip()


def _plan_widths(table: list[list[str]], usable: float) -> list[int]:
    """Dashes per column, in proportion to the width each column is given: its longest word at least, then the rest of
    the line by how much text the column holds."""
    count = len(table[0])
    longest_word = [1] * count
    widest = [1] * count
    for row in table:
        for i in range(count):
            text = _shown(row[i]) if i < len(row) else ""
            widest[i] = max(widest[i], len(text))
            for word in text.split():
                longest_word[i] = max(longest_word[i], len(word))
    share = [float(w) for w in longest_word]
    spare = usable - sum(share)
    extra = [max(0.0, min(widest[i], _MAX_WEIGHT) - longest_word[i]) for i in range(count)]
    if spare > 0 and sum(extra) > 0:
        share = [share[i] + spare * extra[i] / sum(extra) for i in range(count)]
    elif spare > 0:
        share = [share[i] + spare / count for i in range(count)]
    total = sum(share)
    return [max(3, round(100 * s / total)) for s in share]


def fit_tables_to_column(markdown: str, *, two_column: bool) -> str:
    """``markdown`` with each pipe table that is wider than its column (or that pandoc would wrap with equal columns)
    given column widths in proportion to its text, so its cells wrap inside the column instead of running over the
    next one. A table whose longest words cannot sit side by side at the body size is set one size smaller. The
    writer's paper.md is not touched; this is the copy pandoc reads. Code blocks are left alone."""
    lines = markdown.split("\n")
    out: list[str] = []
    fence = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FENCE_RE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        if not (
            fence is None and not m and _ROW_RE.match(line)
            and i + 1 < len(lines) and _RULE_RE.match(lines[i + 1])
        ):
            out.append(line)
            i += 1
            continue
        end = i + 2
        while end < len(lines) and _ROW_RE.match(lines[end]):
            end += 1
        block = lines[i:end]
        rows = [_cells(r) for k, r in enumerate(block) if k != 1]
        rule = _cells(block[1])
        count = len(rule)
        if count < 2 or any(len(r) != count for r in rows):
            out.extend(block)
            i = end
            continue
        widest = [max(len(_shown(r[c])) for r in rows) for c in range(count)]
        words = [max([len(w) for r in rows for w in _shown(r[c]).split()] or [1]) for c in range(count)]
        chars = _COLUMN_CHARS[two_column]
        too_wide = sum(widest) > chars - _CELL_PAD_CHARS * count
        if not too_wide and max(len(r) for r in block) <= _PANDOC_COLUMNS:
            out.extend(block)
            i = end
            continue
        size = ""
        scale = 1.0
        for name, factor in _SMALLER:
            size, scale = name, factor
            if sum(words) <= (chars - _CELL_PAD_CHARS * count) * factor:
                break
        usable = (chars - _CELL_PAD_CHARS * count) * scale
        dashes = _plan_widths(rows, usable)
        marks = []
        for cell, d in zip(rule, dashes):
            marks.append((":" if cell.startswith(":") else "") + "-" * d + (":" if cell.endswith(":") else ""))
        fitted = [block[0], "|" + "|".join(marks) + "|", *block[2:]]
        # A caption written right before or after the table belongs to it: the size covers it too.
        tail = end
        j = end
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and _CAPTION_RE.match(lines[j]):
            while j < len(lines) and lines[j].strip():
                j += 1
            tail = j
        head = len(out)
        k = head
        while k > 0 and not out[k - 1].strip():
            k -= 1
        if k > 0 and _CAPTION_RE.match(out[k - 1]):
            while k > 0 and out[k - 1].strip():
                k -= 1
            head = k
        if size:
            out[head:head] = ["```{=latex}", r"\begingroup" + size, "```", ""]
            out.extend(fitted)
            out.extend(lines[end:tail])
            out.extend(["", "```{=latex}", r"\endgroup", "```"])
        else:
            out.extend(fitted)
            out.extend(lines[end:tail])
        i = tail
    return "\n".join(out)
