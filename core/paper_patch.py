"""Edit the passages a review named and leave the rest of the paper as it is.

A reviewed draft used to go back to the writer with the whole review and one
instruction: write the whole paper again. Measured on the stored quests, that
costs more than the flags it answers:

* every whole rewrite adds background sentences with citations their sources do
  not back, so ``unsupported_claim`` survives the rewrite limit (28 of 38 graded
  papers ended at the iteration cap still flagged);
* a rewrite triggered by a figure flag swapped two citation numbers ("described
  by Allen [2]" for Thompson) in a paper whose first draft had them right;
* a rewrite also loses text that was correct, and one deleted all three figures.

When every must-fix hit names one passage of the paper (a claim, a caption, a
number, a statistic), the writer is instead given the earlier draft and only
those passages and returns edits, ``[{"find": ..., "replace": ...}]``. This
module holds the parts that need no model and no engine state: finding the
passage a check means, reading the reply, and applying the edits. An edit is
applied only when its ``find`` occurs exactly once in the draft's text, and
every character outside the edits stays as it was. Anything else raises
:class:`PatchError`, and the caller writes the whole paper again for that round.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

# The must-fix hits that name one passage of the paper, and what each names.
# ``over_page_limit`` (the whole paper must be shorter), ``figure_missing`` (the
# write step puts a figure back) and ``citations_unchecked`` (nothing was
# located at all) are not here: they are answered by writing the paper again.
PATCHABLE_HITS = {
    "unsupported_claim": "claim",
    "figure_caption": "caption",
    "unsourced_number": "number",
    "mislabelled_statistic": "statistic",
}
_NOT_A_PASSAGE = {
    "over_page_limit": "the draft is over the page limit, which needs the whole paper shortened",
    "figure_missing": "a planned figure is missing from the draft, which the write step puts back",
    "citations_unchecked": "the claim check did not run, so no passage was named",
}

# A passage the engine cannot match to a sentence is still handed to the model;
# one it matches is handed over as the paper's own words. A sentence must cover
# this share of the flagged text's content words to count as the one meant.
_MIN_COVERAGE = 0.8
_MIN_CONTENT_WORDS = 4
# Two candidates this close to each other in score are one too many: the
# engine says nothing rather than point at the wrong sentence.
_AMBIGUITY_MARGIN = 0.05
# Edits that between them replace more than this share of the draft are a whole
# paper written again in disguise.
_MAX_REPLACED_SHARE = 0.4

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\\$])")
_LIST_MARK_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_IMAGE_RE = re.compile(r"!\[(?P<alt>(?:[^\[\]]|\[[^\[\]]*\])*)\]\((?P<src>[^)\s]+)")
_HEADING_RE = re.compile(r"^#{1,6}[ \t].*$", re.MULTILINE)
_CITE_RE = re.compile(r"\[(?:W?\d+(?:\s*[–-]\s*\d+)?(?:\s*[,;]\s*W?\d+(?:\s*[–-]\s*\d+)?)*)\]")
_LATEX_DELIM_RE = re.compile(r"\\[()\[\]]|\$")
_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_FIGURE_FILE_RE = re.compile(r"figures/([^\s\"'`)]+)")
_FURTHER_NUMBERS_RE = re.compile(r"A further \d+ number")
_STOPWORDS = frozenset(
    "a an the of and or to in on at by for with from as is are was were be been it its "
    "this that these those which than then so if not no".split()
)


class PatchError(Exception):
    """The edits cannot be used; the message says why, for the run log."""


@dataclass(frozen=True)
class Passage:
    """One thing a check flagged, and where in the draft it is."""

    kind: str  # "claim" | "caption" | "number" | "statistic"
    flagged: str  # what the check said
    why: str = ""  # its reason, when it gave one
    located: str = ""  # the draft's own words for it, exactly as the draft has them
    how: str = ""  # "exact" | "sentence" | "caption" | "" (not located)


@dataclass(frozen=True)
class Edit:
    find: str
    replace: str


@dataclass(frozen=True)
class Patched:
    text: str
    applied: int
    ignored: int  # edits whose replacement equals what they found
    removed_chars: int
    added_chars: int


# --- finding a flagged passage in the draft -----------------------------------------


def _tokens(text: str) -> list[str]:
    """The content words of ``text``, for comparing a paraphrase with a sentence:
    citations, LaTeX delimiters and markdown emphasis dropped, lower case."""
    text = _LATEX_DELIM_RE.sub(" ", _CITE_RE.sub(" ", text))
    text = re.sub(r"[*_`]+", "", text).lower()
    return [w for w in _WORD_RE.findall(text) if w not in _STOPWORDS]


def sentence_units(text: str) -> list[tuple[int, int, int]]:
    """``(start, end, block)`` for each sentence-like unit of ``text``: the
    spans of the raw text, so a unit is always an exact substring of it.

    A block is what a blank line separates. A table row, a heading, a figure
    line and a code fence are one unit each; a list item is split into its
    sentences; consecutive lines of prose are read as one paragraph."""
    units: list[tuple[int, int, int]] = []
    block = 0
    run: list[int] | None = None  # [start, end] of the prose lines being collected

    def sentences(start: int, end: int) -> None:
        seg = text[start:end]
        mark = _LIST_MARK_RE.match(seg)
        cut = start + (mark.end() if mark else 0)
        for m in _SENTENCE_END_RE.finditer(seg):
            if start + m.start() > cut:
                units.append((cut, start + m.start(), block))
            cut = start + m.end()
        if end > cut:
            units.append((cut, end, block))

    def flush() -> None:
        nonlocal run
        if run is not None:
            sentences(run[0], run[1])
            run = None

    offset = 0
    for line in text.split("\n"):
        start, end = offset, offset + len(line)
        offset = end + 1
        stripped = line.strip()
        if not stripped:
            flush()
            block += 1
            continue
        lead = start + (len(line) - len(line.lstrip()))
        if stripped.startswith(("|", "![", "#", "```", "~~~")):
            flush()
            units.append((lead, start + len(line.rstrip()), block))
        elif _LIST_MARK_RE.match(line):
            flush()
            run = [lead, start + len(line.rstrip())]
        elif run is None:
            run = [lead, start + len(line.rstrip())]
        else:
            run[1] = start + len(line.rstrip())
    flush()
    return units


def locate_passage(flagged: str, paper: str, end: int, units: list[tuple[int, int, int]]) -> tuple[str, str]:
    """``(text, how)``: the draft's own words for what a check flagged, or
    ``("", "")``. The checks quote or paraphrase, so an exact substring is the
    exception: ``exact`` is the flagged text found as it stands, ``sentence`` is
    the one sentence (or two adjacent ones) whose content words cover it."""
    wanted = flagged.strip().strip('"“”').strip()
    region = paper[:end]
    if not wanted:
        return "", ""
    if wanted in region:
        return wanted, "exact"
    words = set(_tokens(wanted))
    if len(words) < _MIN_CONTENT_WORDS:
        return "", ""
    unit_words = [set(_tokens(region[s:e])) for s, e, _b in units]
    scored: list[tuple[float, float, int, int]] = []  # coverage, closeness, start, end
    for i, (s, _e, block) in enumerate(units):
        for j in (i, i + 1):
            if j >= len(units) or units[j][2] != block:
                continue
            have = unit_words[i] | unit_words[j] if j != i else unit_words[i]
            common = len(words & have)
            if not have or common / len(words) < _MIN_COVERAGE:
                continue
            scored.append((common / len(words), common / len(have | words), s, units[j][1]))
    if not scored:
        return "", ""
    scored.sort(key=lambda c: (-c[0], -c[1], c[3] - c[2]))
    best = scored[0]
    for other in scored[1:]:
        apart = other[3] <= best[2] or other[2] >= best[3]
        if apart and other[0] >= best[0] - _AMBIGUITY_MARGIN and other[1] >= best[1] - _AMBIGUITY_MARGIN:
            return "", ""
    return region[best[2]:best[3]], "sentence"


def caption_in(paper: str, end: int, name: str) -> str:
    """The caption (the text in ``![...]``) of the figure file ``name``, or ""."""
    for m in _IMAGE_RE.finditer(paper[:end]):
        if PurePosixPath(m.group("src")).name == name:
            return m.group("alt")
    return ""


def plan_passages(
    paper: str, end: int, *, hits: list[tuple[str, str]],
    claims: list[dict[str, str]], captions: list[str],
) -> tuple[list[Passage], str]:
    """The passages to edit, or ``([], reason)`` when this round is to write the
    whole paper again.

    ``hits`` are the review's must-fix hits as ``(name, text)``; ``claims`` are
    the unsupported claims (``{"claim", "evidence"}``) the claim check listed;
    ``captions`` are the figure-caption findings. ``end`` is where the draft's
    source lists begin: those are written by the engine, so nothing in them is
    edited. A round qualifies only when every hit names one passage."""
    if not hits:
        return [], "no must-fix hit names a passage"
    names = list(dict.fromkeys(name for name, _ in hits))
    for name in names:
        if name not in PATCHABLE_HITS:
            return [], _NOT_A_PASSAGE.get(name, f"{name or 'a hit'} does not name a passage")
    if "unsupported_claim" in names and not claims:
        return [], "the review flags unsupported claims but the claim check lists none"
    if "figure_caption" in names and not captions:
        return [], "the review flags a figure caption but no finding names the figure"
    if any(_FURTHER_NUMBERS_RE.search(text) for _n, text in hits):
        return [], "the check lists only some of the numbers it could not trace"
    units = sentence_units(paper[:end])
    passages: list[Passage] = []
    for claim in claims:
        text = str(claim.get("claim") or "").strip()
        if text:
            located, how = locate_passage(text, paper, end, units)
            passages.append(Passage("claim", text, str(claim.get("evidence") or "").strip(), located, how))
    for finding in captions:
        m = _FIGURE_FILE_RE.search(finding)
        caption = caption_in(paper, end, m.group(1)) if m else ""
        passages.append(Passage("caption", finding, "", caption, "caption" if caption else ""))
    for name, text in hits:
        if PATCHABLE_HITS[name] not in ("number", "statistic"):
            continue
        body = text.split(":", 1)[1].strip() if ":" in text[:60] else text.strip()
        message, sep, context = body.rpartition(" — “")
        context = context.rstrip("”").strip() if sep else ""
        located, how = locate_passage(context, paper, end, units) if context else ("", "")
        passages.append(Passage(PATCHABLE_HITS[name], context or body, message if sep else "", located, how))
    return (passages, "") if passages else ([], "no passage was named")


_KIND_LABEL = {
    "claim": "A claim that neither the study's results nor a cited source backs",
    "caption": "A caption that describes what its figure does not show",
    "number": "A number that nothing in the run accounts for",
    "statistic": "A statistic the paper describes as something the run did not compute",
}


def format_passages(passages: list[Passage]) -> str:
    """The write-patch prompt's ``$passages_block``. The draft's own words go in
    as a JSON string, so the escaping of a backslash or a quote is already the
    one ``find`` needs."""
    blocks: list[str] = []
    for n, p in enumerate(passages, 1):
        lines = [f"### {n}. {_KIND_LABEL[p.kind]}", f"Flagged: {p.flagged}"]
        if p.why:
            lines.append(f"Reason: {p.why}")
        if p.located:
            lead = {
                "exact": "In the draft, exactly",
                "caption": "The caption in the draft, exactly (`find` this text, not the figure link)",
                "sentence": "The sentence in the draft that the flag most likely means",
            }[p.how]
            lines.append(f"{lead}: {json.dumps(p.located, ensure_ascii=False)}")
        else:
            lines.append(
                "In the draft: not found by the engine. Find the passage that says this and "
                "copy it exactly."
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# --- reading the model's reply -------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:\w+)?\s*\n(.*?)\n?```\s*$", re.DOTALL)
# A backslash JSON has no escape for (a LaTeX ``\(``, ``\gamma``): doubled, it is
# the backslash the model meant. The valid pairs are matched first, so they stay.
_ESCAPE_OR_BACKSLASH_RE = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def _load_json(text: str) -> object | None:
    body = text.strip()
    if m := _FENCE_RE.match(body):
        body = m.group(1).strip()
    candidates = [body]
    for opener, closer in (("[", "]"), ("{", "}")):
        lo, hi = body.find(opener), body.rfind(closer)
        if 0 <= lo < hi:
            candidates.append(body[lo:hi + 1])
    for candidate in candidates:
        for attempt in (candidate, _ESCAPE_OR_BACKSLASH_RE.sub(
            lambda m: m.group(0) if len(m.group(0)) > 1 else "\\\\", candidate,
        )):
            try:
                return json.loads(attempt)
            except ValueError:
                continue
    return None


def parse_edits(reply: str) -> list[Edit]:
    """The edits in the model's reply: a JSON array of ``{"find", "replace"}``,
    or an object holding it as ``edits``. Raises :class:`PatchError` with the
    reason when the reply is a whole paper, is not JSON, or holds no edit."""
    text = (reply or "").strip()
    if not text:
        raise PatchError("the reply is empty")
    if text.startswith("#"):
        # A paper opens with its title; JSON never opens with a heading. Checked
        # first because a paper's ``[3]`` is, on its own, a JSON list.
        raise PatchError("the reply is a whole paper, not edits")
    data = _load_json(text)
    if data is None:
        if len(text) > 2000:
            raise PatchError("the reply is a whole paper, not edits")
        raise PatchError("the reply is not JSON")
    if isinstance(data, dict):
        data = data.get("edits")
    if not isinstance(data, list):
        raise PatchError("the reply is JSON but not a list of edits")
    edits: list[Edit] = []
    for n, item in enumerate(data, 1):
        if not isinstance(item, dict) or not isinstance(item.get("find"), str):
            raise PatchError(f"edit {n} is not a find/replace pair")
        replace = item.get("replace")
        if replace is not None and not isinstance(replace, str):
            raise PatchError(f"edit {n} has a replacement that is not text")
        edits.append(Edit(item["find"], replace or ""))
    if not edits:
        raise PatchError("the reply holds no edits")
    return edits


# --- applying them -------------------------------------------------------------------


def _latex_from_json_escapes(text: str) -> str:
    """``text`` with the control characters a JSON escape made of a LaTeX
    command put back: ``\\beta`` written with one backslash reads as a backspace
    and ``eta``, ``\\times`` as a tab. (``\\nu`` reads as a newline and is not
    guessed at.) Tried only for a ``find`` that is not in the draft as it came."""
    return text.replace("\x08", "\\b").replace("\x0c", "\\f").replace("\t", "\\t").replace("\r", "\\r")


def _once(region: str, find: str) -> tuple[int, str]:
    """Where ``find`` is in ``region``, and ``""``; or ``-1`` and why not."""
    count = region.count(find)
    if count == 1:
        return region.index(find), ""
    return -1, "is not in the draft" if count == 0 else f"is in the draft {count} times"


def _shape(text: str) -> tuple[list[str], list[str]]:
    """What no edit may change: the figures the text links and its headings."""
    return [m.group("src") for m in _IMAGE_RE.finditer(text)], _HEADING_RE.findall(text)


def apply_edits(paper: str, end: int, edits: list[Edit]) -> Patched:
    """``paper`` with ``edits`` applied to the text before ``end``, and every
    other character as it was.

    Each ``find`` must occur exactly once in that text, and the edits must not
    overlap. All are located in the paper as it is, then applied together, so
    the order they are given in does not matter. Raises :class:`PatchError` when
    one cannot be applied, when none changes anything, when they replace most of
    the paper, or when the result has a different set of figure links or
    headings."""
    region = paper[:end]
    located: list[tuple[int, int, str, str]] = []  # start, end, find, replace
    ignored = 0
    for n, edit in enumerate(edits, 1):
        find, replace = edit.find, edit.replace
        if not find.strip():
            raise PatchError(f"edit {n} has nothing to find")
        at, problem = _once(region, find)
        if at < 0 and (fixed := _latex_from_json_escapes(find)) != find:
            at_fixed, _ = _once(region, fixed)
            if at_fixed >= 0:
                at, problem, find, replace = at_fixed, "", fixed, _latex_from_json_escapes(replace)
        if at < 0:
            raise PatchError(f"edit {n}: the text to find {problem}: {find[:70]!r}")
        if replace == find:
            ignored += 1
            continue
        located.append((at, at + len(find), find, replace))
    if not located:
        raise PatchError("the edits change nothing")
    located.sort()
    for (_s0, e0, _f0, _r0), (s1, _e1, f1, _r1) in zip(located, located[1:]):
        if s1 < e0:
            raise PatchError(f"two edits overlap at: {f1[:70]!r}")
    removed = sum(e - s for s, e, _f, _r in located)
    if removed > _MAX_REPLACED_SHARE * max(len(region), 1):
        raise PatchError(f"the edits replace {removed} of the draft's {len(region)} characters")
    pieces: list[str] = []
    last = 0
    for start, stop, _find, replace in located:
        pieces += [region[last:start], replace]
        last = stop
    pieces.append(region[last:])
    edited = "".join(pieces)
    if _shape(edited) != _shape(region):
        raise PatchError("an edit changed a figure link or a heading")
    return Patched(edited + paper[end:], len(located), ignored, removed, sum(len(r) for _s, _e, _f, r in located))
