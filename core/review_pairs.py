"""One model writes, a different model reviews: the parts the three reviewed stages (plan, design, code) share.

At each key stage the work is written by the main model (or the model a person routes that step to) and read by a reviewer
that can be another model. Each reviewer has its own step, so a person can route them separately:

* the plan's checks: ``oracle_review`` (core/oracle_review.py),
* the design: ``design_review`` (the methodology audit, ``Engine._audit_design``),
* the code: ``code_review`` (core/code_review.py), which follows ``oracle_review`` while it is not set.

A step that is not set keeps today's behaviour: the design is audited by the main model (recorded as a self-review, never
as an independent one), the code is read by whatever reads the checks.

A reviewer's job is to FIND PROBLEMS, not to agree or disagree: every finding says where it is, what is wrong, why, how
to fix it and how serious it is, and the reply also lists what the reviewer checked and found no problem with, so that
silence is never taken as approval. A reply with no finding and no list of what was checked is no review: it is asked
once more, and then recorded as not reviewed.
"""

from __future__ import annotations

import re
from typing import Any, Callable

#: The step a person routes (``provider.node_providers.<step>`` / ``provider.node_models.<step>``) to have each stage read.
READER_STEP = {"checks": "oracle_review", "design": "design_review", "code": "code_review"}
#: The steps a call is made under when its own step is not set: the design audit keeps its old name (so does every
#: record of it); the code reading goes to the step that reads the checks (``oracle_review.code`` is routed by the key
#: ``oracle_review``, and its calls are recorded apart from the reading of the plan's checks).
FALLBACK_NODE = {"design_review": "design_self_critique", "code_review": "oracle_review.code"}


def node_for(step: str, configured: Callable[[str], Any]) -> str:
    """The node a reviewer's calls are made (and recorded) under: its own step when a model or provider is set for it,
    else the step it falls back to (:data:`FALLBACK_NODE`). ``configured(step)`` says whether ``step`` is set."""
    return step if configured(step) else FALLBACK_NODE.get(step, step)


# --- findings: where, what, why, how to fix, how serious -----------------------------------------------------------------

SEVERITIES = ("high", "medium", "low")
_SEVERITY_WORDS = {
    "high": "high", "critical": "high", "major": "high", "severe": "high", "blocker": "high", "blocking": "high",
    "medium": "medium", "moderate": "medium", "mid": "medium",
    "low": "low", "minor": "low", "nit": "low", "trivial": "low", "info": "low",
}
#: What a reviewer writes for "nothing": not a finding.
_NOTHING = {"", "none", "n/a", "na", "-", "no", "nothing", "empty", "null", "not applicable", "none needed", "no problem",
            "no problems", "no issue", "no issues", "looks fine", "looks good", "all good", "ok", "fine", "agree"}


def _text(value: Any, limit: int = 400) -> str:
    """One line, no backtick, bounded; empty for a reviewer's way of saying nothing."""
    text = " ".join(str(value if value is not None else "").replace("`", "'").split())[:limit]
    return "" if text.lower().rstrip(".") in _NOTHING else text


def _first(item: dict[str, Any], *keys: str, limit: int = 400) -> str:
    for key in keys:
        got = _text(item.get(key), limit)
        if got:
            return got
    return ""


def severity_of(value: Any) -> str:
    return _SEVERITY_WORDS.get(" ".join(str(value or "").lower().split()).rstrip("."), "medium")


def finding(item: Any) -> dict[str, Any] | None:
    """One finding as ``{where, what, why, fix, severity, quantity}``, or ``None`` when it does not say what is wrong.
    The keys a reviewer is likely to use for the same thing are read as one (``problem`` / ``issue`` for ``what``)."""
    if isinstance(item, str):
        item = {"what": item}
    if not isinstance(item, dict):
        return None
    what = _first(item, "what", "problem", "issue", "objection", "finding", "description")
    if not what:
        return None
    return {
        "where": _first(item, "where", "location", "at", "file", "check", "step", limit=160),
        "what": what,
        "why": _first(item, "why", "reason", "because"),
        "fix": _first(item, "fix", "how_to_fix", "suggestion", "how"),
        "severity": severity_of(item.get("severity")),
        "quantity": _first(item, "quantity", "name", limit=80),
    }


def findings(value: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    """The findings in a reply's list (at most ``limit``; an entry that says nothing is dropped)."""
    out = [f for f in (finding(i) for i in (value if isinstance(value, list) else [])) if f is not None]
    return out[:limit]


def checked(value: Any, *, limit: int = 40) -> list[str]:
    """What a reviewer says it checked and found no problem with: one short line each (a list of strings, or of
    ``{item, note}`` mappings)."""
    out: list[str] = []
    for i in value if isinstance(value, list) else []:
        if isinstance(i, dict):
            name = _first(i, "item", "check", "name", "what", "where", limit=120)
            note = _first(i, "note", "why", "evidence", limit=200)
            line = f"{name}: {note}" if name and note else (name or note)
        else:
            line = _text(i, 300)
        if line and line not in out:
            out.append(line)
    return out[:limit]


def said_nothing(found: list[Any], listed_checked: list[str], *, other_problem: bool = False,
                 evidence: bool = False) -> bool:
    """Whether a reply is no review: it names no problem and says nothing it checked. ``other_problem``: the reply's
    own per-item verdicts name a problem; ``evidence``: its per-item verdicts show what it looked at (a reason, or the
    file and line where it found the thing), which is a list of what it checked."""
    return not found and not listed_checked and not other_problem and not evidence


#: Added to the request when the reviewer's first reply listed no problem and did not say what it checked.
REASK = (
    "\n\n---\n\n# Your previous reply was no review\n\n"
    "Your previous reply named no problem and did not say what you checked, so it cannot be told from a reply that did "
    "not look. A reviewer's job is to FIND PROBLEMS, not to agree: look again, and answer the same request in the same "
    "shape. Put every problem you find under `findings` (each with where it is, what is wrong, why it is wrong and how to "
    "fix it), and put every item you checked and found no problem with under `checked`, one line each saying what you "
    "looked at. Do not leave both empty."
)
#: The same, as a reason in a record.
NOT_A_REVIEW = "its reply named no problem and did not say what it checked, even when asked again"


def sentence(f: dict[str, Any]) -> str:
    """One finding as one plain sentence (for the request back to the writer, run.log and the reports)."""
    where = f"{f['where']}: " if f.get("where") else ""
    why = f" ({f['why']})" if f.get("why") else ""
    fix = f" Fix: {f['fix']}" if f.get("fix") else ""
    return f"{where}{f['what'].rstrip('.')}{why}.{fix}"


# --- no expected value in a repair ------------------------------------------------------------------------------------

_ID_RE = re.compile(r"(?<![A-Za-z_\d])(?:lines?|l|e|r|eq|equation|step)\s*\.?\s*#?\s*\d+(?:\s*[-,]\s*\d+)*(?![A-Za-z_\d.])", re.I)
#: A power, a root or a multiple of pi names an operation, not a value.
_OPERATION_RE = re.compile(r"(?:\*\*|\^)\s*[-+]?\d+|sqrt\(\s*\d+\s*\)|\d+\s*\*?\s*pi|pi\s*\*\s*\d+", re.I)
_NUMBER_RE = re.compile(r"(?<![A-Za-z_\d])\.?\d")


def has_value(text: str) -> bool:
    """Whether ``text`` carries a number a quantity could be written to equal (not a line number, an id such as E3 or R2,
    or a power or root): a reviewer who writes "it should be 0.25" is giving an expected value, and that never goes into
    a repair."""
    return bool(_NUMBER_RE.search(_OPERATION_RE.sub(" ", _ID_RE.sub(" ", text or ""))))


def _without_value_sentences(text: str) -> str:
    return " ".join(s for s in re.split(r"(?<=[.;])\s+", text or "") if s and not has_value(s))


def without_values(f: dict[str, Any]) -> dict[str, Any]:
    """``f`` with every sentence that carries a number a quantity could equal left out; the finding stays, with the rest
    of what it says (and, when nothing of ``what`` is left, a pointer to the plan)."""
    out = {**f, "why": _without_value_sentences(f.get("why", "")), "fix": _without_value_sentences(f.get("fix", ""))}
    out["what"] = _without_value_sentences(f["what"]) or "this is not computed as the plan defines it"
    return out
