"""Which sections of an earlier draft a round of the paper writer is asked to write again, and putting them back.

A reviewed draft used to go back to the writer with the review and one instruction: write the whole paper again. The
earlier draft was not in the prompt, so text that was fine was lost and new faults appeared. The passage editor
(:mod:`core.paper_patch`) answers a review whose every hit names one sentence; this module answers the rest at the next
size up: FI decides which *sections* the round touches, the writer is shown the earlier draft read-only and writes only
those sections, and FI puts them back by heading. Every other section stays byte for byte as it was.

What this module holds needs no model and no engine state: splitting a draft into sections, saying which section a piece of
text is in or names, choosing the sections to shorten for a page limit, and splicing a reply back. A section is chosen
because a check quotes or locates a passage in it, because the request names it, or because it holds a value of FI's
record that changed since the draft was written (the engine compares FI's own values, not words). A request that maps to
no section is not guessed at: the caller writes the whole paper and says so in the log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import paper_patch

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t#]*$")
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
_QUOTE_RES = (re.compile("“([^”]{25,})”"), re.compile(r'"([^"\n]{25,})"'), re.compile(r"`([^`\n]{25,})`"))
_WORDS_RE = re.compile(r"\S+")
_CLOSEST_COVERAGE = 0.5
_CLOSEST_MARGIN = 0.1
_FIGURE_LINK_RE = re.compile(r"!\[[^\n]*\]\(figures/")
# A section the writer shortens gives up at most this share of its words (the rest of the cut comes from the next section).
_MAX_CUT_SHARE = 0.4
_CUT_WORDS_RE = re.compile(r"\bCut about (\d+) words", re.IGNORECASE)
# The engine's own sections (their text is FI's, written from its records) are never asked for.
_FI_BLOCK_RE = re.compile(r"<!--\s*fi:(?!v\b|fact-)")


@dataclass(frozen=True)
class Section:
    """One ``##`` section of a draft: its heading text and where it lies in the draft (``start`` is the heading line;
    the section runs to the next heading of its level, so its subsections are part of it). The part before the first
    section (the title, the keywords) is ``index 0`` with the title ``""``."""

    index: int
    title: str
    start: int
    end: int


def _norm(text: str) -> str:
    return " ".join(text.split())


def split(markdown: str, end: int | None = None) -> list[Section]:
    """The sections of ``markdown[:end]``, the preamble first. Sections are the ``##`` headings (a heading inside a code
    fence is not one). A draft with no ``##`` heading has only a preamble, so nothing can be mapped to a section."""
    end = len(markdown) if end is None else end
    marks: list[tuple[int, str]] = []
    pos, in_fence = 0, False
    for line in markdown[:end].splitlines(keepends=True):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            m = _HEADING_RE.match(line.rstrip("\r\n"))
            if m and len(m.group(1)) == 2:
                marks.append((pos, m.group(2).strip()))
        pos += len(line)
    starts = [0] + [p for p, _ in marks]
    titles = [""] + [t for _, t in marks]
    ends = starts[1:] + [end]
    return [Section(i, titles[i], starts[i], ends[i]) for i in range(len(starts))]


def text_of(markdown: str, section: Section) -> str:
    return markdown[section.start:section.end]


def holding(markdown: str, sections: list[Section], snippet: str, end: int | None = None) -> int | None:
    """The index of the section that holds ``snippet`` (a claim, a caption, a quoted sentence), or ``None`` when no section
    does or two do. The snippet is looked for as written first, then as the draft's closest sentence
    (:func:`core.paper_patch.locate_passage`), as the passage editor does."""
    snippet = snippet.strip()
    if not snippet:
        return None
    end = len(markdown) if end is None else end
    found = {s.index for s in sections if s.index and snippet in text_of(markdown, s)}
    if not found:
        located, _how = paper_patch.locate_passage(snippet, markdown, end, paper_patch.sentence_units(markdown[:end]))
        if located:
            found = {s.index for s in sections if s.index and located in text_of(markdown, s)}
    return next(iter(found)) if len(found) == 1 else None


def placing(markdown: str, sections: list[Section], snippet: str, end: int | None = None) -> set[int]:
    """The sections a flagged passage can be in: the one that holds it (:func:`holding`), else the up to three whose best
    sentence shares most of the content words of ``snippet`` (a claim the check put in its own words): at least
    :data:`_CLOSEST_COVERAGE` of the words, and every section within :data:`_CLOSEST_MARGIN` of the best one when there are
    at most three. Empty when it is not clear, so FI never points the writer at sections on a weak match."""
    end = len(markdown) if end is None else end
    exact = holding(markdown, sections, snippet, end)
    if exact is not None:
        return {exact}
    words = set(paper_patch._tokens(snippet))
    if len(words) < paper_patch._MIN_CONTENT_WORDS or any(
            s.index and snippet.strip() in text_of(markdown, s) for s in sections):
        return set()
    best: dict[int, float] = {}
    for start, stop, _block in paper_patch.sentence_units(markdown[:end]):
        owner = next((s.index for s in sections if s.index and s.start <= start < s.end), None)
        if owner is not None:
            best[owner] = max(best.get(owner, 0.0), len(words & set(paper_patch._tokens(markdown[start:stop]))) / len(words))
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < _CLOSEST_COVERAGE:
        return set()
    near = {index for index, score in ranked if ranked[0][1] - score <= _CLOSEST_MARGIN}
    return near if len(near) <= 3 else set()


def naming(markdown: str, sections: list[Section], text: str) -> set[int]:
    """The sections a piece of free text (a hit, a weakness, a person's note) names: one whose heading it writes as the
    heading is written (``Discussion``, ``Limitations``), or one that holds a passage it quotes (25 characters or more,
    in quotation marks). A mention in other words names nothing."""
    out: set[int] = set()
    flat = _norm(text)
    for s in sections:
        if not s.index or not s.title:
            continue
        title = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", s.title).strip()
        if len(title) >= 4 and re.search(r"(?<![\w-])" + re.escape(title) + r"(?![\w-])", flat):
            out.add(s.index)
    for rx in _QUOTE_RES:
        for m in rx.finditer(text):
            at = holding(markdown, sections, m.group(1))
            if at is not None:
                out.add(at)
    return out


def words_to_cut(hit: str) -> int | None:
    """The words an ``over_page_limit`` hit asks to cut (FI's own sentence, ``Cut about N words``), else ``None``."""
    m = _CUT_WORDS_RE.search(hit)
    return int(m.group(1)) if m else None


def to_shorten(markdown: str, sections: list[Section], cut: int) -> dict[int, int]:
    """The sections to shorten for a page limit, with the length (in words) each is cut to: the longest first, none taking
    more than :data:`_MAX_CUT_SHARE` of its words, until their shares cover ``cut``. The abstract and FI's own sections are
    left alone. ``{}`` when the sections cannot give ``cut`` words."""
    pool = []
    for s in sections:
        body = text_of(markdown, s)
        if not s.index or s.title.lower() == "abstract" or _FI_BLOCK_RE.search(body):
            continue
        pool.append((len(_WORDS_RE.findall(body)), s.index))
    out: dict[int, int] = {}
    left = cut
    for words, index in sorted(pool, reverse=True):
        if left <= 0:
            break
        take = min(left, int(words * _MAX_CUT_SHARE))
        if take > 0:
            out[index] = words - take
            left -= take
    return out if left <= 0 else {}


def holds_figures(markdown: str, sections: list[Section]) -> set[int]:
    """The sections that hold a figure link (``![caption](figures/...)``): where a caption flagged without a figure's name
    can be, read from the draft's structure."""
    return {s.index for s in sections if s.index and _FIGURE_LINK_RE.search(text_of(markdown, s))}


def holds_values(markdown: str, sections: list[Section]) -> set[int]:
    """The sections that carry a result key of FI's (``{{v:...}}``) or a table of its records: where the results of a run
    are reported, read from the structure and not from any heading's wording."""
    return {s.index for s in sections if s.index and re.search(r"\{\{(?:v:|calc:|results_table)", text_of(markdown, s))}


def changed(before: str, after: str) -> set[int] | None:
    """The sections whose text differs between ``before`` (the draft as it was written) and ``after`` (the same draft with
    FI's values filled in again from the records as they are now): the sections that cite a value that changed. ``None``
    when the two do not have the same sections, so no section can be compared."""
    a, b = split(before), split(after)
    if [s.title for s in a] != [s.title for s in b]:
        return None
    return {x.index for x, y in zip(a, b)
            if x.index and _norm(text_of(before, x)) != _norm(text_of(after, y))}


@dataclass(frozen=True)
class Spliced:
    text: str
    written: list[str]  # the sections taken from the reply
    ignored: list[str]  # the sections the reply gave that were not asked for (dropped)


class ScopeError(Exception):
    """The reply cannot be put back; the message says why, for the run log."""


def splice(markdown: str, sections: list[Section], asked: set[int], reply: str) -> Spliced:
    """``markdown`` with each asked section replaced by the same section of ``reply`` (found by its heading), every other
    byte as it was. A section the reply gives that was not asked for is dropped, not kept: the earlier text stands. Raises
    :class:`ScopeError` when the reply lacks an asked section or gives one empty."""
    given = split(reply)
    by_title: dict[str, list[Section]] = {}
    for g in given:
        if g.index:
            by_title.setdefault(_norm(g.title).lower(), []).append(g)
    wanted = {s.index: s for s in sections if s.index in asked}
    chosen: dict[int, str] = {}
    for index, s in wanted.items():
        matches = by_title.get(_norm(s.title).lower()) or []
        if len(matches) != 1:
            raise ScopeError(f"the reply has {len(matches)} section(s) headed \"{s.title}\", not one")
        body = text_of(reply, matches[0])
        if not _norm(body.split("\n", 1)[1] if "\n" in body else ""):
            raise ScopeError(f"the section \"{s.title}\" in the reply is empty")
        old = text_of(markdown, s)
        trail = old[len(old.rstrip()):]
        chosen[index] = body.rstrip() + (trail or "\n")
    asked_titles = {_norm(s.title).lower() for s in wanted.values()}
    ignored = [g.title for g in given if g.index and _norm(g.title).lower() not in asked_titles]
    out, pos = [], 0
    for s in sections:
        if s.index in chosen:
            out += [markdown[pos:s.start], chosen[s.index]]
            pos = s.end
    out.append(markdown[pos:])
    return Spliced("".join(out), [wanted[i].title for i in sorted(chosen)], ignored)
