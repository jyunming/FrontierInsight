"""Take a few sentences out of a draft that is a little over its page limit.

A draft over the limit used to go back to the writer with one instruction, write
the whole paper again. Measured on twelve stored quests, that round is the largest
part of the revise loop (a rewrite, then a claim check and a review of it, about
43,000 tokens, 19% of a quest), the overshoot it answers is small (21 of 25
first drafts were 350 words or fewer over, 15 of them 150 or fewer), and a whole
rewrite is the step that changes a correct citation or adds background nobody
backs (see :mod:`core.paper_patch`).

Here the model is only asked which sentences the paper would miss least. It sees
the paper with a number before every sentence it may choose and answers with
numbers; nothing it writes reaches the paper. This module holds the parts that
need no model and no engine state:

* which sentences may be chosen at all. Only sentences of the sections that hold
  background and discussion are offered, and of those never the one that opens a
  paragraph, a list item, a table row, a heading, a figure or a caption, a sentence
  that refers to a figure or a table, one that defines a term or a symbol, one that
  says what the paper sets out to do or finds, or (in an Introduction) why the work
  is needed, one the next sentence points back at, or one that holds a number the
  paper states nowhere else. The Abstract, the Methods and
  the Results are as written, whatever the model answers;
* reading the answer;
* taking sentences out in the model's order until enough words are gone, skipping
  a sentence that would leave a source cited nowhere or a number stated nowhere,
  so that no citation number changes and no number is lost.

Deleting a sentence changes no other character of the paper.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

from . import paper_patch

# The most words a trim may take out. A draft that needs more than this is a
# paper too long for one, and is written again as before.
MAX_WORDS = 400
# The sections sentences may be taken from: the ones that hold background,
# discussion and closing remarks. The Abstract, Methods, Results and the like are
# not on the list, and a paper whose headings match none of them is not trimmed.
_TRIMMABLE_SECTION_RE = re.compile(
    r"introduction|background|related[ \t]+work|literature|prior[ \t]+work|discussion|limitation"
    r"|conclusion|future[ \t]+work|implication|context|overview",
    re.IGNORECASE,
)
_HEADING_LINE_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*$", re.MULTILINE)
_MIN_WORDS = 6  # a shorter sentence is not worth a place in the list
_CITE_BRACKET_RE = re.compile(r"\[((?:W?\d+(?:\s*[–-]\s*\d+)?)(?:\s*[,;]\s*W?\d+(?:\s*[–-]\s*\d+)?)*)\]")
_FIGURE_OR_TABLE_RE = re.compile(r"\b(?:figures?|figs?\.?|tables?)\b", re.IGNORECASE)
# A sentence that introduces a term, a symbol or an abbreviation the text may use
# again: "let", "denote", "defined as", "where $x$ is", "(SIR)".
_DEFINITION_RE = re.compile(
    r"\b(?:let|denote[sd]?|defined?|defines?|we write|stands? for|refers? to|is called|are called"
    r"|abbreviated|acronym)\b|\bwhere\s+\$",
    re.IGNORECASE,
)
_ABBREVIATION_RE = re.compile(r"\([A-Z]{2,}s?\)")
# A sentence about the paper's own work: its aim, its hypothesis, what it did or found. A model
# asked to choose the least valuable sentences took the first ones of the Introduction, this
# statement of the aim among them, on six of six stored papers.
_OWN_WORK_RE = re.compile(
    r"\b(?:we|our|ours|us|this (?:paper|study|work|article|research|manuscript|analysis)"
    r"|the present (?:paper|study|work|analysis)|here)\b",
    re.IGNORECASE,
)
# A sentence that opens by pointing back at the one before it ("However, ...", "This ...", "It ...").
# Taking that one out leaves the pointer with nothing to point at.
_POINTS_BACK_RE = re.compile(
    r"^(?:however|moreover|furthermore|additionally|also|therefore|thus|consequently|hence|in contrast"
    r"|conversely|similarly|likewise|nevertheless|nonetheless|instead|meanwhile|this|these|those|such|that"
    r"|it|they|their|its|the latter|the former|both|in addition|as a result|for this reason)\b",
    re.IGNORECASE,
)
# The sentence of an Introduction that says why the work is needed: what the standard
# treatment misses ("However, real epidemics are stochastic", "While the model gives a
# threshold, it fails to capture ..."). A model asked for the least valuable sentences chose
# it on five of six stored papers, since it reads like background.
_INTRO_SECTION_RE = re.compile(
    r"introduction|background|related[ \t]+work|literature|prior[ \t]+work|overview|context", re.IGNORECASE,
)
_GAP_RE = re.compile(
    r"^(?:however|but|yet|still|while|although|though|nevertheless|nonetheless|despite)\b"
    r"|\b(?:fails?|failed|ignores?|ignored|neglects?|overlooks?|lacks?|gaps?|unclear|unknown|little is known"
    r"|remains? (?:unclear|unknown|open))\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
_LIST_MARK_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+$")
_NOT_PROSE_PREFIXES = ("|", "![", "#", "```", "~~~", "$$", "<", ">", "**Keywords")


@dataclass(frozen=True)
class Sentence:
    """A sentence the model may choose."""

    n: int  # the number the model is shown, from 1
    start: int  # where it is in the paper's text
    end: int
    text: str
    words: int
    cites: tuple[str, ...]  # the source labels it cites, one entry per citation
    numbers: tuple[str, ...]  # the numbers it states, one entry per occurrence


@dataclass(frozen=True)
class Trimmed:
    text: str  # the paper with the sentences taken out
    taken: tuple[Sentence, ...]
    words: int  # how many words those are


def _words(text: str) -> int:
    return len(text.split())


def _labels(text: str) -> list[str]:
    """The source labels ``text`` cites, ranges spelled out (``[2-4]`` is 2, 3, 4)."""
    out: list[str] = []
    for match in _CITE_BRACKET_RE.finditer(text):
        for part in re.split(r"[,;]", match.group(1)):
            part = part.strip()
            if part.startswith("W"):
                out.append(part)
                continue
            ends = [int(x) for x in re.split(r"[–-]", part) if x.strip()]
            out += [str(n) for n in range(ends[0], min(ends[-1], ends[0] + 50) + 1)]
    return out


def _numbers(text: str) -> list[str]:
    """The numbers ``text`` states, its citation brackets left out."""
    return _NUMBER_RE.findall(_CITE_BRACKET_RE.sub(" ", text))


def _uses(cites: list[str] | tuple[str, ...], numbers: list[str] | tuple[str, ...]) -> Counter[str]:
    """What a text needs the rest of the paper to keep: each citation and each
    number, told apart (the citation ``[1]`` is not the number 1)."""
    return Counter([f"cite {c}" for c in cites] + [f"number {n}" for n in numbers])


def _section_of(headings: list[tuple[int, int, str]], pos: int) -> str:
    """The heading of the section a position is in: the shallowest heading that
    encloses it below the paper's title (the title is the first heading), or the
    title itself before the first section."""
    stack: list[tuple[int, str]] = []
    for at, level, text in headings:
        if at > pos:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
    if not stack:
        return ""
    return stack[1][1] if len(stack) > 1 else stack[0][1]


def candidates(paper: str, end: int) -> list[Sentence]:
    """The sentences of ``paper[:end]`` that may be taken out, numbered from 1 in
    the order they stand. See the module docstring for what is never offered."""
    region = paper[:end]
    headings = [(m.start(), len(m.group(1)), m.group(2)) for m in _HEADING_LINE_RE.finditer(region)]
    stated = Counter(_numbers(region))
    seen_blocks: set[int] = set()
    out: list[Sentence] = []
    units = paper_patch.sentence_units(region)
    for i, (start, stop, block) in enumerate(units):
        text = region[start:stop]
        if text.lstrip().startswith(_NOT_PROSE_PREFIXES):
            continue  # a heading, a table row or a figure does not open a paragraph
        first = block not in seen_blocks
        seen_blocks.add(block)
        if first:
            continue
        if _OWN_WORK_RE.search(text):
            continue  # what the paper sets out to do, did or found
        following = units[i + 1] if i + 1 < len(units) else None
        if following and following[2] == block and _POINTS_BACK_RE.match(region[following[0]:following[1]]):
            continue  # the next sentence points back at this one
        section = _section_of(headings, start)
        if not _TRIMMABLE_SECTION_RE.search(section):
            continue
        if _INTRO_SECTION_RE.search(section) and _GAP_RE.search(text):
            continue  # why the work is needed
        if _LIST_MARK_RE.match(region[region.rfind("\n", 0, start) + 1:start]):
            continue  # a list item: taking it out would leave its bullet behind
        if _words(text) < _MIN_WORDS:
            continue
        if _FIGURE_OR_TABLE_RE.search(text) or _DEFINITION_RE.search(text) or _ABBREVIATION_RE.search(text):
            continue
        numbers = _numbers(text)
        if any(stated[n] - k < 1 for n, k in Counter(numbers).items()):
            continue  # a number this sentence alone states
        out.append(Sentence(len(out) + 1, start, stop, text, _words(text), tuple(_labels(text)), tuple(numbers)))
    return out


def marked_body(paper: str, end: int, sentences: list[Sentence]) -> str:
    """``paper[:end]`` with ``<<n: k words>>`` before each sentence that may be
    chosen: what the model is shown."""
    pieces: list[str] = []
    last = 0
    for s in sentences:
        pieces += [paper[last:s.start], f"<<{s.n}: {s.words} words>> "]
        last = s.start
    pieces.append(paper[last:end])
    return "".join(pieces)


def parse_ranking(reply: str, count: int) -> list[int]:
    """The sentence numbers a reply lists, in its order, without repeats and
    without a number that is not one of the ``count`` sentences offered. Raises
    :class:`paper_patch.PatchError` when the reply is not a list of numbers."""
    data = paper_patch._load_json((reply or "").strip())  # noqa: SLF001 — the reader the edits use
    if isinstance(data, dict):
        data = data.get("delete")
    if not isinstance(data, list):
        raise paper_patch.PatchError("the reply is not a list of sentence numbers")
    ranked: list[int] = []
    for item in data:
        found = None if isinstance(item, bool) else re.search(r"\d+", str(item))
        n = int(found.group(0)) if found else 0
        if 1 <= n <= count and n not in ranked:
            ranked.append(n)
    if not ranked:
        raise paper_patch.PatchError("the reply names no sentence that was offered")
    return ranked


def choose(
    paper: str, end: int, sentences: list[Sentence], ranking: list[int], words: int, *, partial: bool = False,
) -> Trimmed | None:
    """The paper with sentences taken out in ``ranking`` order until at least
    ``words`` words are gone, or ``None`` when the list runs out first (or
    ``words`` is over :data:`MAX_WORDS`). With ``partial``, a list that runs out
    returns what it took, when it took anything.

    A sentence is skipped when it holds the last remaining citation of a source
    (without it the source would be listed and cited nowhere, and the sources
    after it would change number) or the last remaining statement of a number."""
    if words > MAX_WORDS:
        return None
    by_number = {s.n: s for s in sentences}
    left = _uses(_labels(paper[:end]), _numbers(paper[:end]))
    taken: list[Sentence] = []
    gone = 0
    for n in ranking:
        s = by_number.get(n)
        if s is None:
            continue
        uses = _uses(s.cites, s.numbers)
        if any(left[what] - k < 1 for what, k in uses.items()):
            continue
        left.subtract(uses)
        taken.append(s)
        gone += s.words
        if gone >= words:
            break
    if gone < words and not (partial and taken):
        return None
    return Trimmed(_delete(paper, [(s.start, s.end) for s in taken]), tuple(taken), gone)


def _delete(text: str, spans: list[tuple[int, int]]) -> str:
    """``text`` without ``spans``, and the spaces that set each one off from its
    neighbour: the ones after it, or, for the last sentence of a line, the ones
    before it."""
    for start, stop in sorted(spans, reverse=True):
        after = stop
        while after < len(text) and text[after] in " \t":
            after += 1
        if after < len(text) and text[after] != "\n":
            stop = after
        else:
            while start > 0 and text[start - 1] in " \t":
                start -= 1
        text = text[:start] + text[stop:]
    return text


def summary(trimmed: Trimmed) -> str:
    """The sentences taken out, for the run log, each cut to its first words."""
    return json.dumps([" ".join(s.text.split()[:9]) + " …" for s in trimmed.taken], ensure_ascii=False)
