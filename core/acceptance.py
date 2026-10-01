"""Who accepted a result, and the one question a person answers before they do.

An accept that no person looked at (``pauses.auto_accept_on_pass``, or a quest run with ``pauses.review: off``) is
recorded as automatic, and the evidence level then stops one below ``publication_ready`` with the gap
:data:`NO_PERSON_GAP` (:func:`core.evidence.assess`). Before a person accepts, every interface (the terminal, the web
page, VS Code) shows what the result does not guarantee and its most important evidence gaps (:func:`shown`), and asks
:data:`QUESTION`. Only an explicit "yes" lets the result reach ``publication_ready``:

- "yes": the person reviewed the evidence record and accepts the claims and the limits listed.
- "partly": accepted with a short note (required) saying what is not accepted; the note is recorded and kept as a gap
  (:func:`partly_gap`) until the paper changes (the limit written into the claims, or the problem fixed) and a person
  accepts the new paper with "yes".
- "I did not check": the quest finishes and exports, with the gap :data:`NOT_CHECKED_GAP`.
- "no": nothing is accepted; the person refines, or looks again.

The answer is recorded as a receipt (:func:`receipt`: the paper and the evidence record it was given for, who, when,
through which interface, the answer, the note and the limits that were listed) in the quest's state,
``needs/EVIDENCE.json`` and a ``result_accepted`` event in the decision trace.

Pure: no model, no files.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any

QUESTION = "Have you reviewed the evidence record, and do you accept these claims and the limits listed?"

#: The answers, in the order every interface shows them: (id, label).
CHOICES: tuple[tuple[str, str], ...] = (
    ("yes", "Yes"),
    ("partly", "Partly (say what you do not accept)"),
    ("no", "No"),
    ("not_checked", "I did not check"),
)
ANSWERS = tuple(c for c, _ in CHOICES)
#: The answers an accept can carry ("no" never accepts).
ACCEPTING = ("yes", "partly", "not_checked")
LABELS = dict(CHOICES)

#: The gap a person's "I did not check" leaves below ``publication_ready``.
NOT_CHECKED_GAP = "no person reviewed the evidence before accepting"
#: How the gap a person's "partly" leaves begins (:func:`partly_gap` adds the note).
PARTLY_GAP = "a person accepted the result only in part"
#: The gap an accept given in answer to an earlier question leaves (it did not say the evidence was reviewed).
EARLIER_QUESTION_GAP = ("the person who accepted it answered an earlier question (whether the numbers matched what "
                        "they expected), not whether they reviewed the evidence")
#: Why a "partly" without its note is not an accept.
NOTE_NEEDED = ("\"partly\" needs a short note saying what you do not accept, so the result was not accepted: give "
               "what you do not accept with the answer")
_MAX_NOTE = 500
_MAX_WHO = 80

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
    "You answered no (you do not accept these claims and their limits), so the result was not accepted. Say what is "
    "wrong with a refine (your notes go to the writing step, or to the design if a point needs a new experiment), or "
    "look at the paper and the evidence again and decide then."
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


def note_of(answer: Any) -> str:
    """The note that goes with an answer (what is not accepted), on one line and capped; ``""`` when there is none."""
    note = answer.get("note") if isinstance(answer, dict) else None
    text = " ".join(str(note or "").split())
    return text if len(text) <= _MAX_NOTE else text[: _MAX_NOTE - 1].rstrip() + "…"


def problem(answer: Any) -> str | None:
    """Why an accept cannot go through as given (``None`` when it can): it must carry an answer to :data:`QUESTION`,
    "no" does not accept, and "partly" needs its note."""
    got = parse_answer(answer.get("answer")) if isinstance(answer, dict) else None
    if got is None:
        return (f"the accept did not answer the question asked before accepting ({QUESTION}), so the result was not "
                "accepted")
    if got == "no":
        return NOT_ACCEPTED_ON_NO
    if got == "partly" and not note_of(answer):
        return NOTE_NEEDED
    return None


def now() -> str:
    """The time now, as the receipts write it (UTC, ISO 8601)."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _time_of(value: Any) -> str:
    """``value`` when it is an ISO 8601 time with a time zone (when the interface took the answer), else now."""
    text = str(value or "").strip()
    try:
        parsed = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return now()
    if parsed.tzinfo is None or parsed > _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(minutes=5):
        return now()  # no time zone, or a time still to come: not when anyone answered
    return parsed.isoformat()


def receipt(answer: dict[str, Any], via: str, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """The receipt of a person's accept: who (the name the interface gives: ``--approve-as`` or the login name at the
    terminal, the name typed on the web page, the login name in VS Code; "not given" when it gives none), when, through
    which interface (the one the answer names, else the engine's own name for the path it came by), the question, the
    answer and its note (kept only with "partly"), and what the person was shown with the question: the fingerprint of
    the evidence record (``evidence_sha256``: the one the interface sends back as ``shown_evidence_sha256``, else the
    one in the review ``snapshot`` handed to it) and the limits listed, word for word (``limits_shown``, from that
    snapshot). When the fingerprint sent back differs from the snapshot's (the record was worked out again on resume
    and came out different), the snapshot's is kept as ``evidence_sha256_at_accept`` and its limits as
    ``limits_at_accept``, and ``limits_shown`` is left empty. The paper it covers (``paper_sha256``) is added where
    the paper is read (``Engine._node_human_feedback``). Whatever ``by`` the answer claims is never taken."""
    block = (snapshot or {}).get("before_accept") if isinstance(snapshot, dict) else None
    block = block if isinstance(block, dict) else None
    who = " ".join(str(answer.get("who") or "").split())[:_MAX_WHO]
    got = parse_answer(answer.get("answer"))
    shown_now = str((block or {}).get("evidence_sha256") or "")
    # The fingerprint the interface displayed (it sends it back), else the one handed to it now.
    shown_then = str(answer.get("shown_evidence_sha256") or "").strip().lower()
    shown_then = shown_then if len(shown_then) == 64 and all(c in "0123456789abcdef" for c in shown_then) else ""
    out = {
        "by": "person",
        "via": str(answer.get("via") or via or "unknown")[:40],
        "who": who or "not given",
        "at": _time_of(answer.get("at")),
        "question": QUESTION,
        "answer": got,
        # Only "partly" says what is not accepted: a note sent with any other answer is not kept.
        "note": note_of(answer) if got == "partly" else "",
        "evidence_sha256": shown_then or shown_now,
        "limits_shown": lines(block),
    }
    if shown_then and shown_now and shown_then != shown_now:
        # The evidence was worked out again when the quest resumed and came out different from what was shown: the
        # limits worked out now are not the ones the person read, so they are kept under their own name.
        out["evidence_sha256_at_accept"] = shown_now
        out["limits_at_accept"] = out.pop("limits_shown")
        out["limits_shown"] = []
    return out


def automatic(via: str) -> dict[str, Any]:
    """The acceptance record of an accept no person made."""
    return {"by": "automatic", "via": via}


def stamp(answer: dict[str, Any], via: str, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """A person's review answer as the engine resumes the pause with it: an accept carries its receipt
    (:func:`receipt`); whatever acceptance record the answer brought with it is dropped."""
    out = {k: v for k, v in answer.items() if k != "acceptance"}
    if str(answer.get("action") or "").strip().lower() == "accept":
        out["acceptance"] = receipt(answer, via, snapshot)
    return out


def partly_gap(note: str) -> str:
    """The gap a person's "partly" leaves, with what they did not accept."""
    return (f"{PARTLY_GAP}; not accepted: {note or '(no note was recorded)'} (write that limit into the paper's "
            "claims, or fix the problem, then accept the new paper with yes)")


def review_gap(record: Any) -> str | None:
    """The gap a person's accept leaves below ``publication_ready`` (``None`` for an explicit "yes" to
    :data:`QUESTION`): "I did not check" and any answer that is not a yes leave :data:`NOT_CHECKED_GAP`, "partly" its
    note (:func:`partly_gap`), and an answer to an earlier question (a record that names another one)
    :data:`EARLIER_QUESTION_GAP`."""
    record = record if isinstance(record, dict) else {}
    question = str(record.get("question") or "").strip()
    if question and question != QUESTION:
        return EARLIER_QUESTION_GAP
    answer = parse_answer(record.get("answer"))
    if answer == "yes":
        return None
    if answer == "partly":
        return partly_gap(note_of(record))
    return NOT_CHECKED_GAP


def mark(record: Any) -> str:
    """One sentence for the summary line when a person's accept was not a plain yes and its gap is not the first one
    shown ("" otherwise), so the note is seen wherever the one line is. Said only when the record itself holds that
    gap: a record worked out before this rule (read back as it was) is not described by it."""
    if not isinstance(record, dict) or record.get("accepted_by") != "person":
        return ""
    gap = review_gap(record.get("acceptance"))
    gaps = record.get("gaps") or []
    held = [str(g) for level in (record.get("all_gaps") or {}).values() if isinstance(level, list) for g in level]
    if not gap or gap not in held or (gaps and gaps[0] == gap):
        return ""
    if gap == NOT_CHECKED_GAP:
        return "The person who accepted it did not review the evidence."
    if gap == EARLIER_QUESTION_GAP:
        return "The person who accepted it answered an earlier question, not whether they reviewed the evidence."
    return f"Accepted only in part; not accepted: {note_of(record.get('acceptance')) or '(no note was recorded)'}."


def evidence_sha256(record: Any) -> str:
    """The fingerprint of an evidence record (its JSON, keys sorted); ``""`` when there is none."""
    if not isinstance(record, dict):
        return ""
    data = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8", errors="replace")
    return hashlib.sha256(data).hexdigest()


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
    is the decision being made, and neither are the gaps an accept leaves), how many more there are, the question, its
    answers and the record's fingerprint (:func:`evidence_sha256`). Nothing is repeated."""
    from .evidence import INFO, LEVELS

    unknown = not isinstance(record, dict) or bool(record.get("assessment_failed"))
    fingerprint = evidence_sha256(record)
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
    # The decision being made is not one of its own gaps (nor an earlier accept's).
    ordered = [g for g in ordered if g.strip()
               and g.strip() not in (WAITING_GAP, NO_PERSON_GAP, NOT_CHECKED_GAP, EARLIER_QUESTION_GAP)
               and not g.strip().startswith(PARTLY_GAP)]
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
        # Which evidence record the person is asked about: the receipt of their accept carries it.
        "evidence_sha256": fingerprint,
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
