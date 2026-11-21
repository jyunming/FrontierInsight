"""Words of the paper that FI's records say nothing for, found so QUALITY.md can say FI did not check them.

FI states two facts itself, from its own records, and the writer is told not to: what the search for the best design
found against the baseline (the first line of the "Best design found" section, ``core/best_design_report.py``) and which
checks of the code against a known answer did not pass (the Limitations, ``core/paper_facts.py::known_answer_items``).
A writer can still word either in its own sentence, and a reading of free text cannot tell which of such sentences is
true, so this decides nothing: it never asks for a rewrite, never takes a sentence out and never stops a quest. It finds
the sentences that touch the two facts while FI's record says the opposite of a plain reading (so that a person is told
they were not checked), and :mod:`core.quality_report` lists them.

* a sentence that calls a design optimised, improved or best while the record says the search did not beat the baseline
  (or the check of it found it no better, no feasible design, or not a local optimum);
* a sentence of the writer's that names a figure as the optimised design when the record has no optimised design (the
  figures and their captions are FI's, so only a sentence of the writer's is read);
* a sentence that calls a result validated or confirmed against known answers while those checks are recorded as
  unconfirmed or failed.

A sentence is listed only when it asserts the claim with none of the words that hedge or deny it; a missing record, an
unreadable one or a sentence with any hedge in it lists nothing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .best_design_report import not_better_why

_OPTIMISED = re.compile(
    r"\b(?:optimi[sz]ed|optimal|optimum|best|improved)\s+(?:design|configuration|geometry|layout|setting|solution|shape|"
    r"arrangement|parameters?|variables?)s?\b"
    r"|\bfound\s+(?:the\s+|an?\s+)?(?:optimal|optimum|best)\b"
    r"|\bthe\s+optimum\b"
    r"|\boptimi[sz]ation\s+(?:succeeded|improved|found|achieved|reduced|increased)\b"
    r"|\b(?:improves?|improved|outperforms?|outperformed|beats?|beat)\s+(?:on\s+|over\s+|upon\s+)?the\s+"
    r"(?:baseline|reference|current|original|initial)\b",
    re.IGNORECASE)
_VALIDATED = re.compile(
    r"\b(?:validated|verified|confirmed|benchmarked)\b[^.]{0,60}\b(?:known|analytic\w*|exact|reference|closed[- ]form)\b"
    r"[^.]{0,30}\b(?:answers?|solutions?|results?|values?|limits?|cases?)\b"
    r"|\bknown[- ]answer\s+(?:checks?|tests?)\b[^.]{0,40}\b(?:pass\w*|agree\w*|confirm\w*|matched|met)\b"
    r"|\b(?:simulation|model|code|solver|implementation)\b[^.]{0,30}\b(?:was|is|has\s+been|were)\s+(?:successfully\s+)?"
    r"(?:validated|verified)\b",
    re.IGNORECASE)
#: Any of these in the sentence means it is not asserting the claim plainly: it denies, doubts or only supposes it.
_HEDGE = re.compile(
    r"\b(?:not|no|nor|never|neither|cannot|can't|couldn't|could|did\s+not|does\s+not|do\s+not|didn't|doesn't|isn't|aren't|"
    r"wasn't|weren't|without|unable|fail\w*|whether|unclear|unverified|unconfirmed|if|would|might|may|should|"
    r"hypothes\w+|candidate|any|unless|until|rather\s+than|instead|exploratory|caveat\w*|limitation\w*)\b|n't",
    re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r"[,;:()]|\b(?:but|however|although|though|while|whereas|yet)\b|\s[-\u2013\u2014]\s", re.IGNORECASE)
_SAME_AS_BASELINE = re.compile(r"\b(?:same|identical|itself|equal|equals|unchanged)\b", re.IGNORECASE)
_CAPTION_OPTIMISED = re.compile(r"\b(?:optimi[sz]ed|optimal|optimum|best)\s+design\b|\bthe\s+optimum\b", re.IGNORECASE)
_CAPTION = re.compile(r"^\s*(?:\*\*|_|\*)?\s*(?:figure|fig\.)\s*\d+", re.IGNORECASE)
_UNCONFIRMED_ORACLE = frozenset({"went_on_failing", "stopped", "failed", "unconfirmed"})


def _sentences(markdown: str) -> list[str]:
    """The paper's prose, one sentence per item: no code, no headings, no tables, no image lines."""
    out: list[str] = []
    fenced = False
    for line in (markdown or "").splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced or not line.strip() or line.lstrip().startswith(("#", "|", "![", "<!--", "---")):
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return out


def _asserted(sentence: str, pattern: re.Pattern[str]) -> bool:
    """Whether ``sentence`` asserts what ``pattern`` finds: some match of it with no hedge or negation in its own clause
    (a clause ends at a comma, semicolon, colon, bracket or "but"). A hedge elsewhere in the sentence does not govern it."""
    for m in pattern.finditer(sentence):
        start = max((c.end() for c in _CLAUSE_BREAK.finditer(sentence, 0, m.start())), default=0)
        stop = next((c.start() for c in _CLAUSE_BREAK.finditer(sentence, m.end())), len(sentence))
        if not _HEDGE.search(sentence[start:stop]):
            return True
    return False


def _shown(sentence: str, cap: int = 160) -> str:
    one = " ".join(sentence.split())
    return one if len(one) <= cap else one[:cap - 3] + "..."


#: What each kind of listed sentence is about, in the words QUALITY.md uses.
SEARCH, FIGURE, KNOWN_ANSWER = "search", "figure", "known_answer"
_ABOUT = {SEARCH: "whether the search beat the baseline",
          FIGURE: "which design a figure shows",
          KNOWN_ANSWER: "whether the results were checked against a known answer"}


def unchecked(paper_md: str, *, best_design: Any = None, optimum_check: Any = None,
              oracle: Any = None) -> list[dict[str, str]]:
    """The sentences of ``paper_md`` that word one of the two facts FI states itself in a way its record says is not so,
    one entry each: ``kind`` (:data:`SEARCH`, :data:`FIGURE`, :data:`KNOWN_ANSWER`), ``sentence`` and ``says`` (what FI's
    record says, in plain words). Nothing here changes the paper."""
    out: list[dict[str, str]] = []
    sentences = _sentences(paper_md)
    why_not = not_better_why(best_design, optimum_check)
    if why_not:
        first = next((s for s in sentences if _asserted(s, _OPTIMISED) and not _SAME_AS_BASELINE.search(s)), None)
        if first:
            out.append({"kind": SEARCH, "sentence": first, "says": why_not})
        # An image's own caption is FI's (written from the figure's record), so only a sentence of the writer's is read.
        caption = next((s for s in sentences if _CAPTION.match(s) and _asserted(s, _CAPTION_OPTIMISED)), None)
        if caption:
            out.append({"kind": FIGURE, "sentence": caption, "says": f"{why_not}, so no optimised design was computed to draw"})
    status = str((oracle or {}).get("status") or "") if isinstance(oracle, dict) else ""
    if status in _UNCONFIRMED_ORACLE:
        first = next((s for s in sentences if _asserted(s, _VALIDATED)), None)
        if first:
            out.append({"kind": KNOWN_ANSWER, "sentence": first,
                        "says": "the known-answer checks were unconfirmed or failed"})
    return out


_SHORT = {SEARCH: "no better design was found", FIGURE: "no optimised design was computed",
          KNOWN_ANSWER: "those checks did not pass"}


def disclosure(entry: dict[str, str], cap: int = 55) -> str:
    """The line QUALITY.md gives for one entry of :func:`unchecked`; the sentence is shown up to ``cap`` characters."""
    return (f"the paper's own words about {_ABOUT.get(entry['kind'], 'a result')} were not checked by FI: "
            f"\"{_shown(entry['sentence'], cap)}\" (FI's record: {_SHORT.get(entry['kind'], entry['says'])})")


# --- a sentence the claim check found unsupported that states a number ----------------------------------------------

#: FI's own blocks (the Limitations, the run record ...) and the values it filled: a block is never read as the writer's
#: sentence, and a value mark is masked while the sentences are found so that nothing inside it splits a sentence.
_FI_BLOCK = re.compile(r"<!-- fi:fact-([\w-]+) -->.*?<!-- /fi:fact-\1 -->", re.S)
_FI_MARK = re.compile(r"<!-- fi:v(?:-span)? .*?-->.*?<!-- /fi:v(?:-span)? -->", re.S)
#: A sentence is read as the one a claim is about only when it is at most this many times as long as the claim, in words.
_SIZE_RATIO = 3
_NOT_PROSE = ("|", "#", "![", "```", "~~~")


_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
_LIST_ITEM_LEAD = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+$")


def _fenced(text: str) -> list[tuple[int, int]]:
    """The spans of fenced code blocks (the fence lines included); an unclosed fence runs to the end."""
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    marker = ""
    pos = 0
    for line in text.split("\n"):
        m = _FENCE.match(line)
        if m and open_at is None:
            if not (m.group(1)[0] == "`" and "`" in m.group(2)):  # a backtick fence's info string holds no backtick
                open_at, marker = pos, m.group(1)
        elif (m and open_at is not None and m.group(1)[0] == marker[0] and len(m.group(1)) >= len(marker)
              and not m.group(2).strip()):
            spans.append((open_at, pos + len(line)))
            open_at = None
        pos += len(line) + 1
    if open_at is not None:
        spans.append((open_at, len(text)))
    return spans


def _numbers(text: str) -> list[str]:
    return sorted(_NUMBER.findall(text))


def _cut(text: str, spans: list[tuple[int, int]]) -> str:
    """``text`` without ``spans`` (a sentence each). A sentence that is all of a list item's text takes the item's marker and
    line with it, so no bare marker is left; any other leaves nothing but the space that set it off."""
    from . import paper_trim

    for start, end in sorted(spans, reverse=True):
        ls = text.rfind("\n", 0, start) + 1
        le = text.find("\n", end)
        le = len(text) if le == -1 else le
        if _LIST_ITEM_LEAD.match(text[ls:start]) and not text[end:le].strip():
            text = text[:ls] + text[le + 1:] if le < len(text) else text[:max(0, ls - 1)]
        else:
            text = paper_trim._delete(text, [(start, end)])
    return text


def take_out_unsupported_numbers(
        paper_md: str, claims: list[dict[str, Any]], *, same: Callable[[str, str], bool],
        holds_number: Callable[[str], bool], plain: Callable[[str], str],
        keyed: Callable[[str], str] | None = None,
        confirm: Callable[[dict[str, Any], str], bool] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """``paper_md`` without the writer's sentences of the ``unsupported`` claims in ``claims``, and one entry for each
    sentence taken out (``claim``, ``evidence``, ``sentence``, and the claim's ``fault`` and ``keys`` when it has them).

    Which claims go is the caller's choice, made from each claim's fault alone (core/engine.py ``_CLAIM_FAULTS_TAKEN_OUT``:
    a number not in FI's records, a true value under a wrong setting, label or interval, a source that backs nothing, a
    verdict FI could not read); the engine passes every such claim and ``holds_number`` always true, so no wording or
    number of the sentence is read to decide. Only for a claim judged before faults were named (a record an older FI wrote)
    does the engine pass the rule that FI applied: the sentence goes when it states a number. A sentence like that is not delivered: FI cannot say which value or source
    the writer meant, so it does not put one in, and the Limitations say that it was taken out. A claim whose fault keeps
    its sentence (its words go beyond the record) stays, and is stated as unconfirmed in the Limitations.

    A sentence is cut only when it is identified exactly, never by likeness of wording: its keyed text must equal the claim's
    own passage (``where`` in the claim list, the writer's sentence with its keys). A claim with no passage, or none that equals
    a sentence of the paper, cuts nothing and stays stated as unconfirmed. Only running
    prose is read: FI's blocks, fenced code, tables, headings and figure lines are not the writer's sentences.

    ``holds_number(sentence)``: whether a placed sentence is cut (with FI's value marks still in it); ``plain(sentence)``
    is the sentence as a reader sees it; ``keyed(sentence)`` is the sentence with FI's values as the writer's keys;
    ``confirm(claim, sentence)``, when given, must also hold for a placed sentence (``sentence`` as a reader sees it)."""
    wanted = [c for c in claims if isinstance(c, dict) and c.get("basis") == "unsupported" and str(c.get("claim") or "").strip()]
    if not wanted or not paper_md:
        return paper_md, []
    units = _prose_units(paper_md, plain, keyed)
    chosen: dict[tuple[int, int], dict[str, Any]] = {}
    for claim in wanted:
        text = str(claim["claim"])
        for start, end, sentence, _k in _units_of_claim(claim, units):  # no passage equal to a sentence: nothing is cut
            if holds_number(paper_md[start:end]) and (confirm is None or confirm(claim, sentence)):
                entry = {"claim": text.strip(), "evidence": str(claim.get("evidence") or "").strip(), "sentence": sentence}
                # Why it went stays with it: the check's fault and the keys of FI's values it is about.
                if claim.get("fault"):
                    entry["fault"] = str(claim["fault"])
                if isinstance(claim.get("keys"), list) and claim["keys"]:
                    entry["keys"] = [str(k) for k in claim["keys"]]
                chosen.setdefault((start, end), entry)
    if not chosen:
        return paper_md, []
    return _cut(paper_md, list(chosen)), [chosen[k] for k in sorted(chosen)]


def _prose_units(paper_md: str, plain: Callable[[str], str],
                 keyed: Callable[[str], str] | None) -> list[tuple[int, int, str, str]]:
    """The writer's sentences of ``paper_md``: ``(start, end, as a reader sees it, with its keys)``. Only running prose:
    FI's blocks, fenced code, tables, headings and figure lines are not the writer's sentences."""
    from . import paper_patch

    blocks = [m.span() for m in _FI_BLOCK.finditer(paper_md)] + _fenced(paper_md)
    masked = _FI_MARK.sub(lambda m: "\x00" * len(m.group(0)), paper_md)
    units: list[tuple[int, int, str, str]] = []
    for start, end, _block in paper_patch.sentence_units(masked):
        raw = paper_md[start:end]
        if not raw.strip() or raw.lstrip().startswith(_NOT_PROSE) or any(a <= start < b for a, b in blocks):
            continue
        # A unit that is nothing but FI's own filled text (a value mark and its punctuation) is not the writer's sentence.
        if not re.sub(r"[\s\x00.,;:!?()\-]+", "", masked[start:end]):
            continue
        units.append((start, end, plain(raw), " ".join((keyed(raw) if keyed else raw).split())))
    return units


def _units_of_claim(claim: dict[str, Any],
                    units: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str, str]]:
    """The units whose keyed text equals one of the claim's own passages (``where``): FI's exact identification."""
    passages = {" ".join(str(w.get("passage") or "").split()) for w in claim.get("where") or [] if isinstance(w, dict)}
    passages.discard("")
    return [u for u in units if u[3] in passages]


def placed_sentences(paper_md: str, claims: list[dict[str, Any]], *, plain: Callable[[str], str],
                     keyed: Callable[[str], str] | None = None) -> list[tuple[int, int]]:
    """``(start, end)`` of each writer's sentence of ``paper_md`` that one of ``claims`` names exactly (its keyed text equals
    the claim's own passage, as for :func:`take_out_unsupported_numbers`), in order, each once. A claim with no passage
    equal to a sentence names none."""
    units = _prose_units(paper_md, plain, keyed) if paper_md and claims else []
    found = {(u[0], u[1]) for c in claims if isinstance(c, dict) for u in _units_of_claim(c, units)}
    return sorted(found)
