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
from dataclasses import dataclass, field

from . import paper_patch

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t#]*$")
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
_QUOTE_RES = (re.compile("“([^”]{25,})”"), re.compile(r'"([^"\n]{25,})"'), re.compile(r"`([^`\n]{25,})`"))
_WORDS_RE = re.compile(r"\S+")
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


def mark(markdown: str) -> tuple[str, dict[str, "Place"]]:
    """``markdown`` as a checker is shown it, with FI's own id before each sentence and each heading marked with its section
    number, and the table of ids: ``<<3.7>>`` is the 7th unit (sentence, caption, table row) of the 3rd ``##`` section.
    A checker that names a passage cites its id; FI reads the passage from this table, never from the checker's words
    (:func:`place`). Headings and code are not numbered."""
    sections = split(markdown)
    units = paper_patch.sentence_units(markdown)
    fenced: list[tuple[int, int]] = []
    pos, opened = 0, None
    for line in markdown.splitlines(keepends=True):
        if _FENCE_RE.match(line):
            if opened is None:
                opened = pos
            else:
                fenced.append((opened, pos + len(line)))
                opened = None
        pos += len(line)
    if opened is not None:
        fenced.append((opened, len(markdown)))
    inserts: list[tuple[int, str]] = []
    table: dict[str, Place] = {}
    count: dict[int, int] = {}
    for start, stop, _block in units:
        text = markdown[start:stop]
        if text.lstrip().startswith("#") or any(a <= start < b for a, b in fenced):
            continue
        owner = next((s.index for s in sections if s.start <= start < s.end), 0)
        count[owner] = count.get(owner, 0) + 1
        key = f"{owner}.{count[owner]}"
        inserts.append((start, f"<<{key}>> "))
        table[key] = Place(key, sections[owner].title, text)
    for sec in sections:
        if sec.index:
            line_end = markdown.find("\n", sec.start)
            inserts.append((len(markdown) if line_end < 0 else line_end, f" <<section {sec.index}>>"))
            table[f"section {sec.index}"] = Place(f"section {sec.index}", sec.title, "")  # the whole section
    out, last = [], 0
    for at, tag in sorted(inserts, key=lambda t: t[0]):
        out += [markdown[last:at], tag]
        last = at
    out.append(markdown[last:])
    return "".join(out), table


@dataclass(frozen=True)
class Place:
    """A passage of the paper by FI's id (:func:`mark`): the id, the title of its section and its exact text."""

    id: str
    section: str
    passage: str


_ID_RE = re.compile(r"section\s*(\d+)|(\d+\.\d+)", re.IGNORECASE)


def read_places(raw: object, table: dict[str, "Place"]) -> list[dict[str, str]]:
    """The passages a checker cited (``raw``: an id or a list of them, ``<<3.7>>`` or ``3.7``) as the records FI keeps:
    ``{"id", "section", "passage"}`` read from FI's own table. An id FI never issued is dropped, so a hit with no valid id
    keeps an empty list and cannot be placed."""
    items = raw if isinstance(raw, list) else [raw] if raw not in (None, "") else []
    out: list[dict[str, str]] = []
    for item in items:
        m = _ID_RE.search(str(item))
        place = table.get(f"section {m.group(1)}" if m and m.group(1) else m.group(2)) if m else None
        if place and not any(o["id"] == place.id for o in out):
            out.append({"id": place.id, "section": place.section, "passage": place.passage})
    return out


def place(markdown: str, sections: list[Section], places: object) -> set[int] | None:
    """The sections of ``markdown`` that hold the passages of ``places`` (records from :func:`read_places`, kept with a
    claim or a hit), or ``None`` when any of them cannot be placed: no record, or its stored text is not in exactly one
    section of this draft (the draft was written again since). FI's stored text is looked up as it is, never matched by
    its words."""
    if not isinstance(places, list) or not places:
        return None
    found: set[int] = set()
    for p in places:
        if isinstance(p, dict) and str(p.get("id") or "").startswith("section "):
            # A cited section: the section whose heading FI stored, when exactly one section of this draft has it.
            same = {s.index for s in sections if s.index and _norm(s.title) == _norm(str(p.get("section") or ""))}
            if len(same) != 1:
                return None
            found |= same
            continue
        text = str(p.get("passage") or "") if isinstance(p, dict) else ""
        if not text.strip():
            return None
        hit = {s.index for s in sections if s.index and text in text_of(markdown, s)}
        if len(hit) != 1:
            return None
        found |= hit
    return found


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
    ignored: list[str]  # the sections the reply gave that were not asked for and did not declare (dropped)
    added: list[str] = field(default_factory=list)  # new sections the writer declared with NEW SECTION: and FI kept


class ScopeError(Exception):
    """The reply cannot be put back; the message says why, for the run log."""


#: The line a writer adds to say that a new section is needed by the fix: ``NEW SECTION: <heading> -- why``.
_NEW_SECTION_RE = re.compile(r"^[ \t]*NEW SECTION:[ \t]*(?P<rest>[^\n]*)$", re.MULTILINE)


def declared_sections(reply: str) -> list[str]:
    """The headings a writer declared as new sections (``NEW SECTION: Heading -- why``), the ``##`` and the reason removed."""
    out = []
    for m in _NEW_SECTION_RE.finditer(reply):
        title = re.split(r"\s+(?:--|—|–)\s+", m.group("rest").strip(), maxsplit=1)[0]
        title = _norm(title.lstrip("#").strip().strip("`"))
        if title:
            out.append(title)
    return out


def splice(markdown: str, sections: list[Section], asked: set[int], reply: str) -> Spliced:
    """``markdown`` with each asked section replaced by the same section of ``reply`` (found by its heading), every other
    byte as it was. A section the reply gives that was not asked for is dropped, not kept: the earlier text stands, unless
    the writer declared it with a ``NEW SECTION: <heading> -- why`` line and the draft has no section of that name: it is
    then kept, right after the asked section it came with. Raises :class:`ScopeError` when the reply lacks an asked
    section or gives one empty."""
    declared = {d.lower() for d in declared_sections(reply)}
    reply = _NEW_SECTION_RE.sub("", reply)
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
    old_titles = {_norm(s.title).lower() for s in sections if s.index}
    ignored, added = [], []
    after: dict[int, str] = {}  # the asked section -> the declared new sections to put right after it
    came_with = next(iter(sorted(wanted)), None)
    for g in given:
        if not g.index:
            continue
        key = _norm(g.title).lower()
        if key in asked_titles:
            came_with = next(i for i, w in wanted.items() if _norm(w.title).lower() == key)
        elif key in declared and key not in old_titles and len(by_title.get(key, [])) == 1 and came_with is not None:
            body = text_of(reply, g)
            if _norm(body.split("\n", 1)[1] if "\n" in body else ""):
                after[came_with] = after.get(came_with, "") + "\n" + body.rstrip() + "\n"
                added.append(g.title)
                continue
            ignored.append(g.title)
        else:
            ignored.append(g.title)
    out, pos = [], 0
    for s in sections:
        if s.index in chosen:
            piece = chosen[s.index]
            if s.index in after:
                trail = piece[len(piece.rstrip()):]
                piece = piece.rstrip() + "\n" + after[s.index].rstrip() + trail
            out += [markdown[pos:s.start], piece]
            pos = s.end
    out.append(markdown[pos:])
    return Spliced("".join(out), [wanted[i].title for i in sorted(chosen)], ignored, added)
