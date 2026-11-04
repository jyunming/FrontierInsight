"""The paper may not claim what FI's own records contradict.

FI keeps records of what happened: the search for the best design (``results/best_design.json``) and FI's check of it at
finer settings (``needs/OPTIMUM_CHECK.json``) and the known-answer checks (``needs/ORACLE_CHECK.json``). This reads the finished paper against them and returns plain must-fix hits for what they contradict. Three
kinds, each decided from a record and each worded conservatively: a sentence is flagged only when it asserts the claim
with none of the words that hedge or deny it, and only when the record says the opposite.

* the paper says a design was optimised, improved or found best, while the record says the search did not beat the
  baseline (or the check of it found it no better, no feasible design, or not a local optimum);
* the paper's prose names a figure as showing the optimised design when the record has no optimised design (the
  figures and their captions are FI's, so only a sentence of the writer's is read);
* the paper calls a result validated or confirmed against known answers while those checks are recorded as unconfirmed
  or failed.

(A value of a quantity FI left out of the results is not on the sheet of keys the writer is given, so a number the writer
types for it is a typed number: ``core/typed_numbers.py`` finds it.)

When unsure, nothing is flagged: a missing record, an unreadable one or a sentence with any hedge in it gives no hit.

The reading of the sentence is a reading of free text, so it can flag a sentence that says what the record says (in live
quests it flagged "the best design found is the baseline" and "nothing ... is validated against that known answer"). So a
hit never stops the quest: the paper is written again with the contradiction named, and a sentence still flagged after
those rewrites is taken out of the paper (:func:`take_out`) and listed in QUALITY.md. A claim the records contradict is
never delivered either way, and a sentence flagged wrongly costs that sentence, not the paper.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

HIT = "record_contradiction"
#: What FI took out of the paper after its rewrites (QUALITY.md prints it).
TAKEN_OUT = ".fi/record_claims_taken_out.json"

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
_NOT_BETTER_VERDICTS = frozenset({"improvement_not_shown", "infeasible", "not_local_optimum"})
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


def _shown(sentence: str) -> str:
    one = " ".join(sentence.split())
    return one if len(one) <= 160 else one[:157] + "..."


def search_not_better(best_design: Any, optimum_check: Any) -> str:
    """Why FI's record says the search did not find a better design than the baseline, in plain words; ``""`` when it
    says nothing of the kind (no search, a better design, or a check that could not decide)."""
    if not isinstance(best_design, dict):
        return ""
    improvement = best_design.get("improvement")
    verdict = str((optimum_check or {}).get("verdict") or (best_design.get("check") or {}).get("verdict") or "")
    if verdict in _NOT_BETTER_VERDICTS:
        says = str((optimum_check or {}).get("says") or (best_design.get("check") or {}).get("says") or "").strip()
        return f"FI's check of the best design at finer settings ended '{verdict.replace('_', ' ')}'" + (
            f" ({says[:160]})" if says else "")
    if best_design.get("best") is None and "best" in best_design:
        return "the search found no design that meets every limit, so there is no best design"
    if isinstance(improvement, dict) and improvement.get("better") is False:
        return "the search found no design better than the baseline"
    return ""


def contradictions(paper_md: str, *, best_design: Any = None, optimum_check: Any = None, oracle: Any = None) -> list[str]:
    """The must-fix hits for what ``paper_md`` claims against FI's records, each starting ``record_contradiction:`` and
    saying in plain words what the paper says, what the record says and what to do. Empty when nothing is contradicted."""
    return [hit for hit, _span in found(paper_md, best_design=best_design, optimum_check=optimum_check, oracle=oracle)]


def take_out(paper_md: str, spans: list[str]) -> tuple[str, list[str]]:
    """``paper_md`` with each of ``spans`` (a sentence, as :func:`found` gives them) taken out once, and the spans that
    were there. A sentence leaves the rest of its line."""
    text, taken = paper_md or "", []
    for span in spans:
        if not span or span not in text:
            continue
        at = text.index(span)
        end = at + len(span)
        while end < len(text) and text[end] in " \t":
            end += 1
        text = text[:at] + text[end:]
        taken.append(span)
    return text, taken


def found(paper_md: str, *, best_design: Any = None, optimum_check: Any = None,
          oracle: Any = None) -> list[tuple[str, str]]:
    """:func:`contradictions`, each hit with the sentence of the paper it is about, so that sentence can be taken out
    (:func:`take_out`)."""
    hits: list[tuple[str, str]] = []
    sentences = _sentences(paper_md)

    # (a) a design called optimised, improved or best when the record says it did not beat the baseline.
    why_not = search_not_better(best_design, optimum_check)
    if why_not:
        for s in sentences:
            if _asserted(s, _OPTIMISED) and not _SAME_AS_BASELINE.search(s):
                hits.append((f"{HIT}: the paper says \"{_shown(s)}\", but FI's record of the search says {why_not}. "
                             "Remove the claim, or say plainly what the record shows (no better design was found).", s))
                break

    # (b) a sentence of the writer's that names a figure as the optimised design when there is none. An image's own
    # caption is FI's (written from the figure's record), so it is not read here.
    if why_not:
        for c, span in [(s, s) for s in sentences if _CAPTION.match(s)]:
            if _asserted(c, _CAPTION_OPTIMISED):
                hits.append((f"{HIT}: a figure is captioned \"{_shown(c)}\", but FI's record of the search says "
                             f"{why_not}, so no optimised design was computed to draw. Take the figure out, or caption "
                             "what it does show.", span))
                break

    # (c) validated or confirmed against known answers while those checks are recorded as unconfirmed or failed.
    status = str((oracle or {}).get("status") or "") if isinstance(oracle, dict) else ""
    if status in _UNCONFIRMED_ORACLE:
        for s in sentences:
            if _asserted(s, _VALIDATED):
                hits.append((f"{HIT}: the paper says \"{_shown(s)}\", but FI's record of the known-answer checks says "
                             "they were unconfirmed or failed (needs/ORACLE_CHECK.json). Do not call the result validated "
                             "or confirmed against known answers; say plainly that those checks did not pass.", s))
                break
    return hits


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
        keyed: Callable[[str], str] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """``paper_md`` without the writer's sentences that the claim check judged ``unsupported`` and that state a number,
    and one entry for each sentence taken out (``claim``, ``evidence``, ``sentence``).

    The claim check holds every number of a result to FI's records: a number that is not in them, or is the number of
    another setting or of the whole run where the sentence says one setting, makes the claim ``unsupported`` (the verdict is
    the check's, no wording is read here). A sentence like that is not delivered: FI cannot say which value the writer
    meant, so it does not put one in, and the Limitations say that it was taken out. An ``unsupported`` claim with no number
    (a statement no source backs, a remark the check could not tie to a result) stays, and is stated as unconfirmed in the
    Limitations as before.

    A sentence is cut only when it is identified exactly, never by likeness of wording: its keyed text must equal the claim's
    own passage (``where`` in the claim list, the writer's sentence with its keys). A claim with no passage, or none that equals
    a sentence of the paper, cuts nothing and stays stated as unconfirmed. Only running
    prose is read: FI's blocks, fenced code, tables, headings and figure lines are not the writer's sentences.

    ``holds_number(sentence)``: whether the sentence (with FI's value marks still in it) states a number; ``plain(sentence)``
    is the sentence as a reader sees it; ``keyed(sentence)`` is the sentence with FI's values as the writer's keys."""
    from . import paper_patch

    wanted = [c for c in claims if isinstance(c, dict) and c.get("basis") == "unsupported" and str(c.get("claim") or "").strip()]
    if not wanted or not paper_md:
        return paper_md, []
    blocks = [m.span() for m in _FI_BLOCK.finditer(paper_md)] + _fenced(paper_md)
    masked = _FI_MARK.sub(lambda m: "\x00" * len(m.group(0)), paper_md)
    units: list[tuple[int, int, str, str]] = []  # start, end, as the reader sees it, with its keys
    for start, end, _block in paper_patch.sentence_units(masked):
        raw = paper_md[start:end]
        if not raw.strip() or raw.lstrip().startswith(_NOT_PROSE) or any(a <= start < b for a, b in blocks):
            continue
        # A unit that is nothing but FI's own filled text (a value mark and its punctuation) is not the writer's sentence.
        if not re.sub(r"[\s\x00.,;:!?()\-]+", "", masked[start:end]):
            continue
        units.append((start, end, plain(raw), " ".join((keyed(raw) if keyed else raw).split())))
    chosen: dict[tuple[int, int], dict[str, Any]] = {}
    for claim in wanted:
        text = str(claim["claim"])
        passages = {" ".join(str(w.get("passage") or "").split()) for w in claim.get("where") or [] if isinstance(w, dict)}
        passages.discard("")
        hits = [u for u in units if u[3] in passages]  # no passage, or none equal to a sentence: nothing is cut
        for start, end, sentence, _k in hits:
            if holds_number(paper_md[start:end]):
                chosen.setdefault((start, end), {"claim": text.strip(), "evidence": str(claim.get("evidence") or "").strip(),
                                                 "sentence": sentence})
    if not chosen:
        return paper_md, []
    return _cut(paper_md, list(chosen)), [chosen[k] for k in sorted(chosen)]
