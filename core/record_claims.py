"""The paper may not claim what FI's own records contradict.

FI keeps records of what happened: the search for the best design (``results/best_design.json``) and FI's check of it at
finer settings (``needs/OPTIMUM_CHECK.json``), the known-answer checks (``needs/ORACLE_CHECK.json``) and the figures the
run drew. This reads the finished paper against them and returns plain must-fix hits for what they contradict. Three
kinds, each decided from a record and each worded conservatively: a sentence is flagged only when it asserts the claim
with none of the words that hedge or deny it, and only when the record says the opposite.

* the paper says a design was optimised, improved or found best, while the record says the search did not beat the
  baseline (or the check of it found it no better, no feasible design, or not a local optimum);
* the paper embeds a figure the run did not draw, or captions a figure as showing the optimised design when the record
  has no optimised design;
* the paper calls a result validated or confirmed against known answers while those checks are recorded as unconfirmed
  or failed.

When unsure, nothing is flagged: a missing record, an unreadable one or a sentence with any hedge in it gives no hit.
"""

from __future__ import annotations

import re
from typing import Any

HIT = "record_contradiction"

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
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)")
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


def contradictions(paper_md: str, *, best_design: Any = None, optimum_check: Any = None, oracle: Any = None,
                   figures_on_disk: set[str] | None = None) -> list[str]:
    """The must-fix hits for what ``paper_md`` claims against FI's records, each starting ``record_contradiction:`` and
    saying in plain words what the paper says, what the record says and what to do. Empty when nothing is contradicted.
    ``figures_on_disk``: the file names of the figures the run drew (``None``: not known, so no figure is judged)."""
    hits: list[str] = []
    sentences = _sentences(paper_md)

    # (a) a design called optimised, improved or best when the record says it did not beat the baseline.
    why_not = search_not_better(best_design, optimum_check)
    if why_not:
        for s in sentences:
            if _asserted(s, _OPTIMISED) and not _SAME_AS_BASELINE.search(s):
                hits.append(f"{HIT}: the paper says \"{_shown(s)}\", but FI's record of the search says {why_not}. "
                            "Remove the claim, or say plainly what the record shows (no better design was found).")
                break

    # (b) a figure the run did not draw, or one captioned as the optimised design when there is none.
    if figures_on_disk is not None:
        for m in _IMAGE.finditer(paper_md or ""):
            path = m.group(2).strip()
            name = path.replace("\\", "/").rsplit("/", 1)[-1]
            if "figures/" in path.replace("\\", "/") and name and name not in figures_on_disk:
                hits.append(f"{HIT}: the paper shows {path}, but the run drew no figure of that name (FI's record of the "
                            "run's figures does not have it). Take the figure out, or show one the run drew.")
    if why_not:
        captions = [m.group(1) for m in _IMAGE.finditer(paper_md or "")] + [
            s for s in sentences if _CAPTION.match(s)]
        for c in captions:
            if _asserted(c, _CAPTION_OPTIMISED):
                hits.append(f"{HIT}: a figure is captioned \"{_shown(c)}\", but FI's record of the search says {why_not}, "
                            "so no optimised design was computed to draw. Take the figure out, or caption what it does show.")
                break

    # (c) validated or confirmed against known answers while those checks are recorded as unconfirmed or failed.
    status = str((oracle or {}).get("status") or "") if isinstance(oracle, dict) else ""
    if status in _UNCONFIRMED_ORACLE:
        for s in sentences:
            if _asserted(s, _VALIDATED):
                hits.append(f"{HIT}: the paper says \"{_shown(s)}\", but FI's record of the known-answer checks says they "
                            "were unconfirmed or failed (needs/ORACLE_CHECK.json). Do not call the result validated or "
                            "confirmed against known answers; say plainly that those checks did not pass.")
                break
    return hits
