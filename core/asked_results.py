"""The one table of the results the topic asks for, written by FI from its own records and placed in the paper's results.

Without it the results a topic asks for depend on where the writer happens to put them: a rewrite of the whole paper drops
what the earlier draft placed. FI therefore lists each quantity the topic asks for and FI holds a value for
(:func:`core.goal_coverage.asked_answers`) in a table: the plan's label of the quantity, its value as FI's key fills it
everywhere else in the paper (a ``{{v:...}}`` key, so the value checks see it and the goal check counts it as placed), and
the setting the value is at when the key names one. A quantity FI holds no value for is no row. The table is FI's block
``{{asked_results}}`` (core/paper_facts.py ``BLOCK_NAMES``): it is put in after every write round and replaced, never
repeated, and its text is plain: it carries no word about how FI got the values.

This module only builds the text and finds the place; the engine writes the file (``Engine._place_blocks``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

#: Where the engine keeps the table's text (the block's file, core/paper_facts.py ``_BLOCK_FILES``). ``{n}`` is its number.
FILE = Path(".fi") / "asked_results.md"
CAPTION = "**Table {n}.** The quantities the study set out to determine."
NUMBER = "{n}"

_SNAKE = re.compile(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+")
_CAPTION = re.compile(r"^[ \t]*(?:\*\*|\*|_)?Table[ \t]+(\d+)\b", re.M | re.I)
_BLOCK = re.compile(r"<!-- fi:fact-asked_results -->.*?<!-- /fi:fact-asked_results -->", re.S)
_PLACEHOLDER = re.compile(r"\{\{\s*asked_results\s*\}\}")


def _plain(name: str) -> str:
    """A name as a person reads it: an identifier of the code (``rate_constant``) is words."""
    return name.replace("_", " ") if _SNAKE.fullmatch(name.strip()) else name


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _with_interval(rec: Any, got: Any) -> Any:
    """The key that fills (``got``) as the same key with its interval when FI can fill that too (``mean_ci``, as the writer's
    list of keys offers it), else ``got`` itself: a value is shown with its uncertainty wherever FI holds one."""
    from . import paper_values as _values

    if got is None:
        return None
    head, at, tail = got.key.partition(" @ ")
    if not at or ":" in head or head in rec.answers:
        return got
    with_ci = _values.resolve(_values.placeholder(f"{head}:mean_ci{at}{tail}"), rec)
    return with_ci if with_ci.filled else got


def table(rec: Any, asked: list[dict[str, str]], protocol: dict[str, Any] | None) -> str:
    """The table's markdown (its caption numbered ``{n}``), ``""`` when FI holds a value for none of the quantities
    asked. One row per quantity, in the topic's order; the value cell is the key that fills, so the paper's value
    checks read it as they read any other number. ``rec`` is :class:`core.paper_values.Records`."""
    from . import paper_values as _values
    from . import results_by_setting as _rbs

    if rec is None or not asked:
        return ""
    rows: list[tuple[str, str, str]] = []
    done: set[str] = set()
    varied = set(rec.varied())
    for a in asked:
        if a["metric"] in done:  # two asked quantities answered by one metric are one row
            continue
        got = _with_interval(rec, rec.filling_key(a["metric"]))
        if got is None:
            continue
        done.add(a["metric"])
        try:
            where = _values.parse(_values.placeholder(got.key)).where.items()
        except Exception:  # noqa: BLE001 -- a key that cannot be read again names no setting
            where = []
        # the setting is named only along an axis the study varies and this quantity takes more than one value on
        axes = [(rec.real(k), p.value) for k, p in where if p.kind == "value" and rec.real(k) in varied
                and len(rec.values_of(a["metric"], rec.real(k))) > 1]
        label = _rbs.plain_label(protocol, a["metric"], _plain(a["name"]))
        setting = ", ".join(f"{_plain(rec.label(axis))} = {value}" for axis, value in axes)
        rows.append((_cell(label), _values.placeholder(got.key), _cell(setting)))
    if not rows:
        return ""
    with_setting = any(r[2] for r in rows)
    head = ["Quantity", "Value"] + (["At"] if with_setting else [])
    lines = [CAPTION, "", "| " + " | ".join(head) + " |", "|" + "|".join(["---", "---:"] + (["---"] if with_setting else [])) + "|"]
    for label, value, setting in rows:
        lines.append("| " + " | ".join([label, value] + ([setting] if with_setting else [])) + " |")
    return "\n".join(lines)


def number_at(markdown: str, at: int) -> int:
    """The number of the table placed at offset ``at`` of ``markdown``: the one after the numbered table captions before it,
    unless a caption after it already holds that number (then the one after the highest in the paper), so no number
    is used twice. A paper whose tables carry no number gets 1."""
    own = [m.span() for m in _BLOCK.finditer(markdown)]

    def numbers(a: int, b: int) -> list[int]:
        return [int(m.group(1)) for m in _CAPTION.finditer(markdown, a, b)
                if not any(s <= m.start() < e for s, e in own)]

    before, after = numbers(0, at), numbers(at, len(markdown))
    n = max(before, default=0) + 1
    return n if n not in after else max([*before, *after]) + 1


#: Sections that are never a place for the table of results: the front matter, the set-up of the study, the introduction, the
#: discussion and what follows the body.
_NOT_RESULTS = re.compile(
    r"\b(?:abstract|summary|key ?words?|index terms|highlights|methods?|methodology|materials|set-?up|approach|procedures?"
    r"|protocol|experimental design|introduction|background|related work|preliminar\w*|discussion|conclusions?|limitations?"
    r"|references?|bibliography|works cited|further reading|acknowledg\w*|appendi\w*|supplement\w*)\b", re.I)
_RESULTS_HEAD = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?(?:results?|findings)\b", re.I)


def with_block(markdown: str) -> str:
    """``markdown`` with the placeholder ``{{asked_results}}`` on a line of its own at the end of the section that reports the
    results. That is the section headed Results or Findings, whenever the paper has one (the one with most of FI's value keys
    when several do), even if it holds no key. Only when no section is headed so, the section with most value keys among those
    that are not front matter, set-up, introduction, discussion or back matter. With neither, before the discussion that follows
    the results (else the conclusions, the limitations, the references), else at the end. Nothing is changed when the draft
    holds the block already (:func:`core.paper_facts.place_blocks` takes every earlier one out first)."""
    if _PLACEHOLDER.search(markdown) or _BLOCK.search(markdown):
        return markdown
    from . import best_design_report as _best
    from . import paper_scope as _scope

    sections = [s for s in _scope.split(markdown) if s.index]

    def keys(s: Any) -> int:
        return len(re.findall(r"\{\{(?:v:|calc:|trend:|comparison:)", _scope.text_of(markdown, s)))

    chosen = None
    headed = [s for s in sections if _RESULTS_HEAD.match(s.title)]
    if headed:
        chosen = max(headed, key=lambda s: (keys(s), -s.index))
    else:
        holders = _scope.holds_values(markdown, sections)
        eligible = [s for s in sections if s.index in holders and not _NOT_RESULTS.search(s.title)]
        if eligible:
            chosen = max(eligible, key=lambda s: (keys(s), -s.index))
    if chosen is not None:
        head = markdown[:chosen.end].rstrip()
        tail = markdown[chosen.end:].lstrip("\n")
        return head + "\n\n{{asked_results}}\n\n" + tail if tail else head + "\n\n{{asked_results}}\n"
    return _best.insert_before_discussion(markdown, ["{{asked_results}}"])
