"""What past attempts would recommend at a decision — recorded, never acted on (shadow mode).

The attempt records (core/attempt_records.py) say what each quest tried and under which conditions. This module reads
them back: at a few decision points (a plan put forward, a design about to be implemented, an experiment about to run,
a script about to be repaired) it compares the conditions of the decision with past attempts that failed and writes
what it *would* recommend to ``.fi/shadow_recommendations.jsonl``. Nothing reads that file to decide anything: the
quest runs exactly as it would without it. ``fi tools shadow-report`` scores the recommendations against what happened.

Why only recorded: a failure is a fact about one attempt under one set of conditions, not about a method in general.
Whether acting on these recommendations raises the share of usable results, lowers cost, or suppresses exploration
that would have succeeded has not been shown; until a prospective, preregistered comparison (memory on/off x
exploration constrained/free) shows benefit without more false accepts, they stay a record.

The actions, from strongest to weakest:

- ``BLOCK``: the same complete conditions (same model and settings, FI version, prompts, code, environment, inputs,
  protocol, design, budget) failed the same way at least :data:`REPRODUCED` times — a reproducible hazard. Only before
  an experiment runs, where the code is known; a plan or a repair can be matched at most as ``VERIFY``.
- ``VERIFY``: the same complete conditions failed once, or failed with everything the same but the model or FI's
  version (a different model is not the same test).
- ``INFO``: a similar failure — the same design or protocol or code under other conditions, or a match that rests on
  an incomplete record or an older record schema.
- ``IGNORE``: nothing comparable failed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import attempt_records as _attempts

SHADOW = "shadow_recommendations.jsonl"
SHADOW_LOST = "shadow_recommendations.lost"
ACTIONS = ("BLOCK", "VERIFY", "INFO", "IGNORE")
DECISIONS = ("plan", "implement", "execute", "repair")
#: How many times the same complete conditions must have failed the same way to count as reproducible.
REPRODUCED = 2
#: Outcomes of an attempt that are failures (core/attempt_records.py::OUTCOMES without the non-failures).
FAILURES = ("process_error", "protocol_mismatch", "oracle_failure")
#: Only this many quests (the most recently changed) are read from an output root.
MAX_QUESTS = 500

# The parts of a context compared, by group. ``model`` and ``fi`` may differ for VERIFY; everything else must match.
_MODEL_FIELDS = ("provider", "model", "settings")
_FI_FIELDS = ("fi",)
_REST_BY_DECISION: dict[str, tuple[str, ...]] = {
    # Before code exists: the question, the design, the protocol, the prompts, the inputs.
    "plan": ("prompts", "question", "protocol", "metric_specs", "inputs", "data", "skills", "budget"),
    "implement": ("prompts", "question", "protocol", "metric_specs", "inputs", "data", "skills", "budget"),
    # With code: all of it.
    "execute": ("prompts", "question", "protocol", "metric_specs", "inputs", "data", "skills", "budget", "code",
                "environment", "dependency_lock"),
    "repair": ("prompts", "question", "protocol", "metric_specs", "inputs", "data", "skills", "budget", "code",
               "environment", "dependency_lock"),
}


def _part(context: dict[str, Any], name: str) -> Any:
    """One comparable part of a context (hashes where the context has them)."""
    if name == "fi":
        fi = context.get("fi") or {}
        return fi.get("source_sha256") if isinstance(fi, dict) else None
    if name == "prompts":
        return context.get("prompts_sha256")
    if name == "question":
        q = context.get("question") or {}
        return (q.get("topic_sha256"), q.get("design_sha256")) if isinstance(q, dict) else None
    if name == "protocol":
        return context.get("protocol_sha256")
    if name == "metric_specs":
        return context.get("metric_specs_sha256")
    if name in ("inputs", "data"):
        v = context.get(name) or {}
        return v.get("sha256") if isinstance(v, dict) else None
    if name == "environment":
        return context.get("environment_sha256")
    if name == "dependency_lock":
        return context.get("dependency_lock_sha256")
    if name == "code":
        return json.dumps(context.get("code") or {}, sort_keys=True)
    if name == "skills":
        return tuple(context.get("skills") or [])
    if name == "budget":
        return json.dumps(context.get("budget") or {}, sort_keys=True, default=str)
    if name == "settings":
        return json.dumps(context.get("settings") or {}, sort_keys=True, default=str)
    return context.get(name)


def _answered_model(record: dict[str, Any]) -> tuple[Any, Any]:
    answered = record.get("answered_by") if isinstance(record.get("answered_by"), dict) else {}
    return answered.get("provider"), answered.get("model")


def model_family(context: dict[str, Any]) -> str:
    """``provider/model`` of a context, for grouping (``?`` for what is not known)."""
    return f"{context.get('provider') or '?'}/{context.get('model') or '?'}"


@dataclass
class Attempt:
    """One past attempt that failed, as the index keeps it."""
    record_id: str
    quest_id: str
    outcome: str
    failure: str
    context: dict[str, Any]
    current_schema: bool
    complete: bool
    answered: tuple[Any, Any] = (None, None)
    at: float = 0.0


def failure_class(record: dict[str, Any]) -> str:
    """The failure as one short label: the outcome, and for a stop the check that stopped it."""
    outcome = str(record.get("outcome") or record.get("execution_status") or "unknown")
    if record.get("kind") == "stop" and record.get("pause"):
        return f"{outcome}:{record['pause']}"
    if record.get("kind") == "quest" and record.get("error"):
        return f"crashed:{record['error']}"
    return outcome


def _as_attempt(record: dict[str, Any]) -> Attempt | None:
    kind = record.get("kind")
    failed = (kind in ("run", "stop") and record.get("outcome") in FAILURES) or (
        kind == "quest" and record.get("execution_status") == "crashed")
    context = record.get("context")
    if not failed or not isinstance(context, dict) or not context:
        return None
    return Attempt(
        record_id=str(record.get("record_id") or ""), quest_id=str(record.get("quest_id") or ""),
        outcome=str(record.get("outcome") or record.get("execution_status") or ""), failure=failure_class(record),
        context=context, current_schema=record.get("schema") == _attempts.SCHEMA,
        complete=bool(context.get("complete")), answered=_answered_model(record),
        at=float(record.get("at") or 0.0),
    )


class Index:
    """The failed attempts of the quests under one output root, read lazily and re-read when a file changes."""

    def __init__(self, output_root: Path | None, *, own_fi_dir: Path | None = None) -> None:
        self.output_root = Path(output_root) if output_root else None
        self.own_fi_dir = Path(own_fi_dir) if own_fi_dir else None
        self._files: dict[str, tuple[float, int, list[Attempt]]] = {}

    def _paths(self) -> list[Path]:
        paths: list[Path] = []
        if self.output_root is not None and self.output_root.is_dir():
            try:
                quests = sorted((p for p in self.output_root.iterdir() if (p / ".fi" / _attempts.ATTEMPTS).is_file()),
                                key=lambda p: (p / ".fi" / _attempts.ATTEMPTS).stat().st_mtime, reverse=True)
            except OSError:
                quests = []
            paths += [q / ".fi" / _attempts.ATTEMPTS for q in quests[:MAX_QUESTS]]
        if self.own_fi_dir is not None:
            own = self.own_fi_dir / _attempts.ATTEMPTS
            if own not in paths:
                paths.append(own)
        return paths

    def attempts(self) -> list[Attempt]:
        out: list[Attempt] = []
        for path in self._paths():
            try:
                st = path.stat()
            except OSError:
                continue
            key = str(path)
            cached = self._files.get(key)
            if cached is None or cached[0] != st.st_mtime or cached[1] != st.st_size:
                found = [a for a in (_as_attempt(r) for r in _attempts.read(path.parent, path.name)) if a is not None]
                cached = (st.st_mtime, st.st_size, found)
                self._files[key] = cached
            out += cached[2]
        return out


@dataclass
class Recommendation:
    action: str
    reason: str
    matched_attempt_ids: list[str] = field(default_factory=list)
    fields_matched: list[str] = field(default_factory=list)
    fields_differed: list[str] = field(default_factory=list)
    failure_class: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "matched_attempt_ids": self.matched_attempt_ids,
                "fields_matched": self.fields_matched, "fields_differed": self.fields_differed,
                "failure_class": self.failure_class}


def _compare(candidate: dict[str, Any], past: dict[str, Any], names: tuple[str, ...]) -> tuple[list[str], list[str]]:
    same, differ = [], []
    for name in names:
        # Both without the part (no data folder, no lock file) is the same; whether a part that must be there is
        # there is what ``complete`` says, and only complete contexts can match as the same conditions.
        a, b = _part(candidate, name), _part(past, name)
        (same if a == b else differ).append(name)
    return same, differ


def recommend(candidate: dict[str, Any], attempts: list[Attempt], *, decision: str) -> Recommendation:
    """What the past failures in ``attempts`` would recommend for a decision made under ``candidate`` (a context from
    :func:`core.attempt_records.context_fingerprint`). Pure: it reads nothing and changes nothing."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}; got {decision!r}")
    rest = _REST_BY_DECISION[decision]
    candidate_ok = bool(candidate.get("complete"))
    best: dict[str, list[tuple[Attempt, list[str], list[str]]]] = {a: [] for a in ACTIONS}
    for past in attempts:
        m_same, m_differ = _compare(candidate, past.context, _MODEL_FIELDS)
        f_same, f_differ = _compare(candidate, past.context, _FI_FIELDS)
        r_same, r_differ = _compare(candidate, past.context, rest)
        same, differ = m_same + f_same + r_same, m_differ + f_differ + r_differ
        # Only a complete context of the current schema, on both sides, can stand for "the same conditions".
        trusted = candidate_ok and past.complete and past.current_schema
        if not r_differ and trusted and not (m_differ or f_differ):
            best["BLOCK"].append((past, same, differ))  # exactly the same: BLOCK when reproduced, else VERIFY
        elif not r_differ and trusted:
            best["VERIFY"].append((past, same, differ))  # another model or FI version: at most a check
        elif not r_differ or any(n in r_same for n in ("question", "protocol", "code")):
            best["INFO"].append((past, same, differ))  # similar, or resting on an incomplete / older record
    # BLOCK: the same complete conditions, reproduced, before an experiment runs.
    exact = best["BLOCK"]
    if exact:
        by_failure: dict[str, list[tuple[Attempt, list[str], list[str]]]] = {}
        for item in exact:
            by_failure.setdefault(item[0].failure, []).append(item)
        failure, items = max(by_failure.items(), key=lambda kv: len(kv[1]))
        if decision == "execute" and len(items) >= REPRODUCED:
            return Recommendation("BLOCK", f"the same conditions failed the same way {len(items)} times ({failure})",
                                  [i[0].record_id for i in items], items[0][1], items[0][2], failure)
        return Recommendation("VERIFY", f"the same conditions failed before ({failure})"
                              + ("" if decision == "execute" else f"; at {decision} it is at most a check"),
                              [i[0].record_id for i in items], items[0][1], items[0][2], failure)
    if best["VERIFY"]:
        items = best["VERIFY"]
        first = items[0]
        return Recommendation("VERIFY", f"the same conditions but another model or FI version failed ({first[0].failure})",
                              [i[0].record_id for i in items], first[1], first[2], first[0].failure)
    if best["INFO"]:
        items = best["INFO"]
        first = items[0]
        why = ("a similar attempt failed" if candidate_ok else
               "a similar attempt failed (the conditions of this decision are not fully known)")
        return Recommendation("INFO", f"{why} ({first[0].failure})", [i[0].record_id for i in items[:20]],
                              first[1], first[2], first[0].failure)
    return Recommendation("IGNORE", "nothing comparable failed")


def record(fi_dir: Path, quest_id: str, *, decision: str, taken: str, candidate: dict[str, Any],
           recommendation: Recommendation) -> str | None:
    """Append one shadow recommendation; its id, or ``None`` when it could not be written (counted)."""
    rid = _attempts.append(fi_dir, SHADOW, {
        "kind": "shadow", "quest_id": quest_id, "decision": decision, "taken": taken,
        "model_family": model_family(candidate), "context_complete": bool(candidate.get("complete")),
        "context_sha256": _attempts._json_sha({k: v for k, v in candidate.items() if k != "missing"}),
        "lineage": candidate.get("lineage"), **recommendation.as_dict(),
    })
    if rid is None:
        _attempts.count_lost(fi_dir, SHADOW_LOST)
    return rid


# ---- scoring ------------------------------------------------------------------------------------------------------

def _outcome_after(rec: dict[str, Any], attempts: list[dict[str, Any]]) -> str | None:
    """What the decision a recommendation was made at came to: the next run of that quest after it (for a plan, the
    quest's end); ``None`` when nothing followed yet."""
    at = float(rec.get("at") or 0.0)
    later = sorted((r for r in attempts if float(r.get("at") or 0.0) >= at), key=lambda r: float(r.get("at") or 0.0))
    for r in later:
        if rec.get("decision") != "plan" and r.get("kind") in ("run", "stop"):
            return str(r.get("outcome") or "")
        if r.get("kind") == "quest":
            status = str(r.get("execution_status") or "")
            if status == "crashed":
                return "process_error"
            if rec.get("decision") == "plan":
                return "accepted" if r.get("review_status") == "accepted" else status or "inconclusive"
    return None


@dataclass
class Score:
    counts: dict[str, dict[str, int]]
    by_family: dict[str, dict[str, dict[str, int]]]
    lost: int
    quests: int


def score(output_root: Path) -> Score:
    """For every shadow recommendation under ``output_root``: was the decision it was made at followed by a failure?
    Counts by action: ``right`` (a BLOCK/VERIFY followed by a failure, an IGNORE/INFO followed by none), ``wrong``,
    ``open`` (nothing followed yet)."""
    counts: dict[str, dict[str, int]] = {a: {"right": 0, "wrong": 0, "open": 0} for a in ACTIONS}
    by_family: dict[str, dict[str, dict[str, int]]] = {}
    lost = quests = 0
    root = Path(output_root)
    if not root.is_dir():
        return Score(counts, by_family, lost, quests)
    for quest in sorted(p for p in root.iterdir() if (p / ".fi" / SHADOW).is_file()):
        quests += 1
        fi = quest / ".fi"
        lost += _attempts.lost(fi, SHADOW_LOST)
        attempts = _attempts.read(fi, _attempts.ATTEMPTS)
        for rec in _attempts.read(fi, SHADOW):
            action = str(rec.get("action") or "")
            if action not in counts:
                continue
            outcome = _outcome_after(rec, attempts)
            if outcome is None:
                verdict = "open"
            else:
                failed = outcome in FAILURES
                warned = action in ("BLOCK", "VERIFY")
                verdict = "right" if failed == warned else "wrong"
            counts[action][verdict] += 1
            fam = by_family.setdefault(str(rec.get("model_family") or "?/?"),
                                       {a: {"right": 0, "wrong": 0, "open": 0} for a in ACTIONS})
            fam[action][verdict] += 1
    return Score(counts, by_family, lost, quests)


def report_lines(result: Score) -> list[str]:
    """The score in plain lines: counts only, no claim beyond them."""
    lines = [f"Shadow recommendations in {result.quests} quest(s) — recorded only; none changed what a quest did."]
    total = sum(sum(v.values()) for v in result.counts.values())
    if not total:
        return lines + ["No recommendation recorded yet."]
    for action in ACTIONS:
        c = result.counts[action]
        n = sum(c.values())
        if not n:
            continue
        if action in ("BLOCK", "VERIFY"):
            lines.append(f"  {action}: {n} — followed by a failure {c['right']}, not {c['wrong']}, not known yet {c['open']}")
        else:
            lines.append(f"  {action}: {n} — followed by no failure {c['right']}, by a failure {c['wrong']}, "
                         f"not known yet {c['open']}")
    for fam, per in sorted(result.by_family.items()):
        n = sum(sum(v.values()) for v in per.values())
        warned = sum(per[a]["right"] + per[a]["wrong"] for a in ("BLOCK", "VERIFY"))
        lines.append(f"  {fam}: {n} recommendation(s), {warned} would have warned")
    if result.lost:
        lines.append(f"  {result.lost} recommendation(s) could not be written.")
    lines.append("These counts do not show that acting on the recommendations would help: that needs a planned "
                 "comparison.")
    return lines
