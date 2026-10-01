"""Who accepted a result, and the one question a person answers before they do.

An accept that no person looked at (``pauses.auto_accept_on_pass``, or a quest run with ``pauses.review: off``) is
recorded as automatic, and the evidence level then stops one below ``publication_ready`` with the gap
:data:`NO_PERSON_GAP` (:func:`core.evidence.assess`). Before a person accepts, every interface (the terminal, the web
page, VS Code) shows what the result does not guarantee and its most important evidence gaps (:func:`shown`), and asks
:data:`QUESTION`. The answer is recorded with the accept (``acceptance`` in the quest's state, ``needs/EVIDENCE.json``
and a ``result_accepted`` event in the decision trace). "No" does not accept: the person refines, or looks again.

Pure: no model, no files.
"""

from __future__ import annotations

from typing import Any

QUESTION = "Do the main numbers match what you expected?"

#: The answers, in the order every interface shows them: (id, label).
CHOICES: tuple[tuple[str, str], ...] = (
    ("yes", "Yes"),
    ("partly", "Partly"),
    ("no", "No"),
    ("not_checked", "I did not check"),
)
ANSWERS = tuple(c for c, _ in CHOICES)
#: The answers an accept can carry ("no" never accepts).
ACCEPTING = ("yes", "partly", "not_checked")
LABELS = dict(CHOICES)

# What a person may type (the CLI's ``--accept <answer>``, the terminal prompt).
_ALIASES = {
    "y": "yes", "yes": "yes",
    "p": "partly", "partly": "partly", "partial": "partly", "partially": "partly",
    "n": "no", "no": "no",
    "not_checked": "not_checked", "not-checked": "not_checked", "notchecked": "not_checked",
    "unchecked": "not_checked", "not checked": "not_checked", "did not check": "not_checked",
    "i did not check": "not_checked", "didnt-check": "not_checked", "didn't check": "not_checked",
    "c": "not_checked",
}

#: The gap an automatic accept leaves below ``publication_ready``.
NO_PERSON_GAP = "no person reviewed the result before it was accepted"
#: The gap the evidence names while the review is waiting (it is the decision being made, so it is not shown then).
WAITING_GAP = "the review is waiting for your decision (accept, reject or refine)"

#: What happens when the answer is "no".
NOT_ACCEPTED_ON_NO = (
    "You said the main numbers do not match what you expected, so the result was not accepted. Say what is wrong "
    "with a refine (your notes go to the writing step, or to the design if a point needs a new experiment), or look "
    "at the paper again and decide then."
)

#: How many items of each list are shown before an accept.
MAX_NOT_GUARANTEED = 2
MAX_GAPS = 3
_MAX_CHARS = 240


def parse_answer(text: Any) -> str | None:
    """The answer id for what a person typed or sent (``None`` when it is not one of the answers)."""
    key = " ".join(str(text or "").strip().lower().replace("’", "'").split())
    if not key:
        return None
    return _ALIASES.get(key) or _ALIASES.get(key.replace(" ", "_"))


def problem(answer: Any) -> str | None:
    """Why an accept cannot go through as given (``None`` when it can): it must carry an answer to :data:`QUESTION`,
    and "no" does not accept."""
    got = parse_answer(answer.get("answer")) if isinstance(answer, dict) else None
    if got is None:
        return (f"the accept did not answer the question asked before accepting ({QUESTION}), so the result was not "
                "accepted")
    if got == "no":
        return NOT_ACCEPTED_ON_NO
    return None


def by_person(answer: dict[str, Any], via: str) -> dict[str, Any]:
    """The acceptance record of a person's accept. ``via`` (the interface) is the one the answer names, else the
    engine's own name for the path it came by; whatever ``by`` the answer claims is never taken."""
    return {
        "by": "person",
        "via": str(answer.get("via") or via or "unknown")[:40],
        "question": QUESTION,
        "answer": parse_answer(answer.get("answer")),
    }


def automatic(via: str) -> dict[str, Any]:
    """The acceptance record of an accept no person made."""
    return {"by": "automatic", "via": via}


def stamp(answer: dict[str, Any], via: str) -> dict[str, Any]:
    """A person's review answer as the engine resumes the pause with it: an accept carries the acceptance record
    (:func:`by_person`); whatever acceptance record the answer brought with it is dropped."""
    out = {k: v for k, v in answer.items() if k != "acceptance"}
    if str(answer.get("action") or "").strip().lower() == "accept":
        out["acceptance"] = by_person(answer, via)
    return out


def accepted_by(state: dict[str, Any], *, pending: bool, paper_sha256: str = "") -> str | None:
    """``person`` or ``automatic`` for a result whose review accepted it, ``None`` while the review is waiting for a
    decision or when the review did not accept. No acceptance record counts as automatic: nobody is recorded as having
    looked; so does a person's accept of another paper than the one there is now (``paper_sha256``, when both the
    record and the caller name one)."""
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    if pending or not review or review.get("verdict") != "accept":
        return None
    record = state.get("acceptance")
    if not (isinstance(record, dict) and record.get("by") == "person"):
        return "automatic"
    accepted = str(record.get("paper_sha256") or "")
    if "paper_sha256" in record and (not accepted or (paper_sha256 and accepted != paper_sha256)):
        return "automatic"
    return "person"


def _short(text: Any) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= _MAX_CHARS else s[: _MAX_CHARS - 1].rstrip() + "…"


def _unique(items: list[str], limit: int, seen: set[str]) -> tuple[list[str], int]:
    out: list[str] = []
    extra = 0
    for item in items:
        text = _short(item)
        key = text.lower().rstrip(".")
        if not text or key in seen:
            continue
        seen.add(key)
        if len(out) < limit:
            out.append(text)
        else:
            extra += 1
    return out, extra


def shown(record: dict[str, Any] | None) -> dict[str, Any]:
    """What a person sees before they accept, from the quest's evidence record: what the result does not guarantee
    (the blind spots of the level it reached, at most :data:`MAX_NOT_GUARANTEED`), its most important gaps (the next
    level's first, then the ones above it, at most :data:`MAX_GAPS`; never the "waiting for your decision" gap, which
    is the decision being made, and neither is "no person reviewed it"), how many more there are, the question and its
    answers. Nothing is repeated."""
    from .evidence import INFO, LEVELS

    unknown = not isinstance(record, dict) or bool(record.get("assessment_failed"))
    record = record if isinstance(record, dict) else {}
    status = str(record.get("status") or "not_executed")
    blind = list(INFO.get(status, INFO[LEVELS[0]])["known_blind_spots"])
    if unknown:
        # The assessment itself failed (or there is none): nothing is claimed, and the run may well have produced results.
        blind = ["The evidence could not be worked out (see .fi/run.log), so nothing here says how far the result was "
                 "checked."]
    elif status == "not_executed":
        blind = ["The experiment has not produced results, so nothing is shown about them."]
    all_gaps = record.get("all_gaps") if isinstance(record.get("all_gaps"), dict) else {}
    ordered: list[str] = []
    start = LEVELS.index(status) + 1 if status in LEVELS else 0
    for level in LEVELS[start:]:
        ordered.extend(str(g) for g in all_gaps.get(level) or [])
    ordered.extend(str(g) for g in record.get("gaps") or [])  # a record without all_gaps (an older one)
    # The decision being made is not one of its own gaps.
    ordered = [g for g in ordered if g.strip() and g.strip() not in (WAITING_GAP, NO_PERSON_GAP)]
    seen: set[str] = set()
    not_guaranteed, _ = _unique(blind, MAX_NOT_GUARANTEED, seen)
    gaps, more = _unique(ordered, MAX_GAPS, seen)
    return {
        "reached": ("" if unknown else
                    INFO.get(status, {}).get("assurance_claim") or "The experiment has not produced results."),
        "not_guaranteed": not_guaranteed,
        "gaps": gaps,
        "more_gaps": more,
        "question": QUESTION,
        "choices": [{"id": c, "label": label} for c, label in CHOICES],
    }


def lines(block: dict[str, Any] | None) -> list[str]:
    """``block`` (:func:`shown`) as short plain lines for a terminal or a card."""
    if not isinstance(block, dict):
        return []
    out: list[str] = []
    if block.get("reached"):
        out.append(f"What the checks show: {block['reached']}")
    for item in block.get("not_guaranteed") or []:
        out.append(f"Not guaranteed: {item}")
    for item in block.get("gaps") or []:
        out.append(f"Gap: {item}")
    if block.get("more_gaps"):
        out.append(f"(+{block['more_gaps']} more gap(s) in needs/EVIDENCE.json)")
    return out
