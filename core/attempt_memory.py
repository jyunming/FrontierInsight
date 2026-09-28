"""What past attempts would recommend at a decision — recorded, never acted on (shadow mode).

The attempt records (core/attempt_records.py) say what each quest tried and under which conditions. This module reads
them back: at four decisions (a plan put forward, a design about to be implemented, an experiment about to run, a
script about to be repaired) it compares the conditions *that decision depends on* with past attempts that failed and
writes what it would recommend to ``.fi/shadow_recommendations.jsonl``. Nothing reads that file to decide anything:
no route, state, prompt or trace changes. ``fi tools shadow-report`` scores the recommendations against what followed.

Why only recorded: a failure is a fact about one attempt under one set of conditions, not about a method in general.
Whether acting on these recommendations raises the share of usable results or lowers cost, without more false accepts
and without suppressing exploration that would have worked, has not been shown; that needs a prospective, planned
comparison (memory on/off x exploration constrained/free) before any of them may change what a quest does.

**What each decision is compared on** (:data:`KEYS`): only what that decision depends on, never the whole quest.

- ``plan``: the topic, the inputs and data, the selected skills, the policy (rigor profile, review panel, result use),
  the design prompt, and the model that answered the plan. With the design's features (metric ids, grid axes) also
  the same it is a check (``VERIFY``); with them different, a note.
- ``implement``: the design, the protocol and its metric specs, the packages asked for, the implement prompts, and the
  model the implement step is asked on.
- ``execute``: the code, the packages asked for, the inputs and data, the protocol, runs per setting, the time limit,
  the replicate seeds, and FI's trial harness. No model and no prompt: the same code is the same test whoever wrote it.
- ``repair``: the code and the failure signature (return code, exception type, the check that stopped it). The run
  that triggered the repair, and the repair lineage of the same design in the same quest, are never matched.

The environment and dependency lock are compared only where both sides know them (a first run writes its
environment during the run). A part that is not known on either side makes a match information only.

**Actions**: ``BLOCK`` only before an experiment runs, when the same key failed with the same non-transient signature
at least :data:`REPRODUCED` times (a timeout, running out of memory or a provider/transport error never counts);
``VERIFY`` when the same key failed once, or at a plan / implement / repair; ``INFO`` for a similar failure or one that
rests on an unknown part or an older record; ``IGNORE`` when nothing comparable failed.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import attempt_records as _attempts

SHADOW = "shadow_recommendations.jsonl"
SHADOW_LOST = "shadow_recommendations.lost"
#: The shadow record ids waiting for the run, stop or quest record they are about (lineage, not wall clock).
SHADOW_PENDING = "shadow_pending.json"
ACTIONS = ("BLOCK", "VERIFY", "INFO", "IGNORE")
DECISIONS = ("plan", "implement", "execute", "repair")
REPRODUCED = 2
FAILURES = ("process_error", "protocol_mismatch", "oracle_failure")
#: A decision's recommendation must be worked out within this many seconds, or it is dropped (counted, not written).
DEADLINE_S = 5.0
#: Only this many quests (the most recently changed) are read from an output root.
MAX_QUESTS = 500

KEYS: dict[str, tuple[str, ...]] = {
    "plan": ("topic", "inputs", "data", "skills", "policy", "prompt_plan", "model_plan"),
    "implement": ("design", "protocol", "metric_specs", "deps", "prompt_implement", "model_implement"),
    "execute": ("code", "deps", "inputs", "data", "protocol", "runs_per_setting", "timeout_s", "seeds", "harness"),
    "repair": ("code", "signature"),
}
#: Compared only when both sides know them.
OPTIONAL: dict[str, tuple[str, ...]] = {
    "plan": (), "implement": ("environment", "lock"), "execute": ("environment", "lock"), "repair": (),
}
#: Parts that, when the same, make a failure "similar" (INFO) even if the rest differs.
ANCHORS: dict[str, tuple[str, ...]] = {
    "plan": ("topic",), "implement": ("design", "protocol"), "execute": ("code", "protocol"), "repair": ("code",),
}
_MODEL_PARTS = ("model_plan", "model_implement")

_EXC_RE = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Timeout))\b", re.MULTILINE)
_TRANSIENT_WORDS = ("timeout", "timedout", "memoryerror", "connection", "transport", "remoteprotocol", "ratelimit",
                    "http", "transient", "axonunavailable", "cancelled", "keyboardinterrupt", "brokenpipe",
                    "temporar", "unavailable", "oserror")
_TRANSIENT_RC = (-9, -15, 137, 143)


def failure_signature(*, returncode: Any = None, stderr: str = "", timed_out: bool = False, pause: str | None = None,
                      error: str | None = None) -> dict[str, Any]:
    """How an attempt failed, as a signature: the return code, the exception type (the last one a traceback names in
    ``stderr``, or ``error``), the check that stopped it (``pause``), and whether it is ``transient`` (a time limit,
    running out of memory, a signal kill, a provider or transport error) — a transient failure is never a reproducible
    hazard."""
    exc = error
    if not exc and stderr:
        found = _EXC_RE.findall(str(stderr)[-6000:])
        exc = found[-1] if found else None
    rc = returncode if isinstance(returncode, int) and not isinstance(returncode, bool) else None
    transient = bool(timed_out) or rc in _TRANSIENT_RC or any(w in str(exc or "").lower() for w in _TRANSIENT_WORDS)
    return {"returncode": rc, "exception": exc, "pause": pause or None, "transient": transient}


def _sig_key(sig: dict[str, Any] | None) -> str | None:
    if not isinstance(sig, dict):
        return None
    return json.dumps({k: sig.get(k) for k in ("returncode", "exception", "pause")}, sort_keys=True)


def _present(v: Any) -> Any:
    """A folder manifest's hash, ``"absent"`` for no folder, ``None`` (unknown) for one not all hashed."""
    if v is None:
        return "absent"
    if isinstance(v, dict):
        return v.get("sha256") if v.get("complete") else None
    return None


def _model(ctx: dict[str, Any], nodes: tuple[str, ...], requested: str | None) -> tuple[str | None, bool]:
    """The model that answered one of ``nodes`` in ``ctx`` (and whether its connection named it); else the model the
    step is asked on (``requested``, unreported)."""
    used = ctx.get("models_used") if isinstance(ctx.get("models_used"), dict) else {}
    for n in nodes:
        e = used.get(n)
        if isinstance(e, dict) and e.get("model"):
            return f"{e.get('provider') or '?'}/{e.get('model')}", bool(e.get("reported"))
    return (requested, False) if requested else (None, False)


def _prompt(ctx: dict[str, Any], prefix: str) -> str | None:
    shas = ctx.get("prompt_shas") if isinstance(ctx.get("prompt_shas"), dict) else None
    if shas is None:
        return None
    chosen = {k: v for k, v in shas.items() if k == prefix or k.startswith(prefix + "_")}
    return _attempts._json_sha(chosen) if chosen else "absent"


def parts(ctx: dict[str, Any], decision: str, *, signature: dict[str, Any] | None = None,
          requested_model: str | None = None) -> dict[str, Any]:
    """The comparable parts of a context for one decision (hashes or plain values; ``None`` = not known)."""
    ctx = ctx if isinstance(ctx, dict) else {}
    q = ctx.get("question") if isinstance(ctx.get("question"), dict) else {}
    pol = ctx.get("policy") if isinstance(ctx.get("policy"), dict) else {}
    budget = ctx.get("budget") if isinstance(ctx.get("budget"), dict) else {}
    code = ctx.get("code") if isinstance(ctx.get("code"), dict) else {}
    all_parts: dict[str, Any] = {
        "topic": q.get("topic_sha256"),
        "design": q.get("design_sha256"),
        "inputs": _present(ctx.get("inputs")),
        "data": _present(ctx.get("data")),
        "skills": json.dumps(sorted(ctx.get("skills") or [])),
        "policy": _attempts._json_sha({k: pol.get(k) for k in ("rigor_profile", "review_panel", "result_use")})
        if pol else None,
        "prompt_plan": _prompt(ctx, "design"),
        "prompt_implement": _prompt(ctx, "implement"),
        "protocol": ctx.get("protocol_sha256") or "absent",
        "metric_specs": ctx.get("metric_specs_sha256") or "absent",
        "deps": ctx.get("deps_sha256") or "absent",
        "code": _attempts._json_sha(code) if code else None,
        # With no protocol there is no runs-per-setting to know: "absent", not unknown.
        "runs_per_setting": budget.get("runs_per_setting") if ctx.get("protocol_sha256") else
        (budget.get("runs_per_setting") or "absent"),
        "timeout_s": budget.get("timeout_s"),
        "seeds": json.dumps([budget.get("execute_replicates"), budget.get("replicate_seed_stride")]),
        "harness": ctx.get("harness_sha256"),
        "environment": ctx.get("environment_sha256"),
        "lock": ctx.get("dependency_lock_sha256"),
        "signature": _sig_key(signature),
        "features": json.dumps(ctx.get("design_features"), sort_keys=True) if ctx.get("design_features") else None,
    }
    all_parts["model_plan"], all_parts["model_plan_reported"] = _model(ctx, ("plan", "design"), requested_model)
    all_parts["model_implement"], all_parts["model_implement_reported"] = _model(
        ctx, ("implement", "implement_body", "implement_outline"), requested_model)
    keep = (*KEYS[decision], *OPTIONAL[decision], *(("features",) if decision == "plan" else ()),
            *(f"{m}_reported" for m in _MODEL_PARTS if m in KEYS[decision]))
    return {k: all_parts.get(k) for k in keep}


@dataclass
class Attempt:
    """One past failed attempt, kept compactly: its ids, signature and the parts of each decision."""
    record_id: str
    quest_id: str
    kind: str
    outcome: str
    signature: dict[str, Any]
    current_schema: bool
    design_revision: Any
    decision_parts: dict[str, dict[str, Any]]
    model_family: str = "?/?"


def _as_attempt(record: dict[str, Any]) -> Attempt | None:
    kind = str(record.get("kind") or "")
    failed = (kind in ("run", "stop") and record.get("outcome") in FAILURES) or (
        kind == "quest" and record.get("execution_status") == "crashed")
    ctx = record.get("context")
    if not failed or not isinstance(ctx, dict) or not ctx:
        return None
    sig = record.get("failure_signature") if isinstance(record.get("failure_signature"), dict) else None
    if sig is None:
        sig = failure_signature(pause=record.get("pause") if kind == "stop" else None,
                                error=record.get("error") if kind == "quest" else None,
                                returncode=record.get("returncode"))
    lineage = ctx.get("lineage") if isinstance(ctx.get("lineage"), dict) else {}
    return Attempt(
        record_id=str(record.get("record_id") or ""), quest_id=str(record.get("quest_id") or ""), kind=kind,
        outcome=str(record.get("outcome") or record.get("execution_status") or ""), signature=sig,
        current_schema=record.get("schema") == _attempts.SCHEMA, design_revision=lineage.get("design_revision"),
        decision_parts={d: parts(ctx, d, signature=sig) for d in DECISIONS},
        model_family=f"{ctx.get('provider') or '?'}/{ctx.get('model') or '?'}",
    )


class Index:
    """The failed attempts of the quests under one output root, as compact summaries, re-read per file when it
    changes. Thread-safe: every engine in a process shares one per output root (:func:`index_for`)."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)
        self._lock = threading.Lock()
        self._files: dict[str, tuple[float, int, list[Attempt]]] = {}

    def _paths(self, extra: Path | None) -> list[Path]:
        paths: list[Path] = []
        try:
            quests = [p for p in self.output_root.iterdir() if (p / ".fi" / _attempts.ATTEMPTS).is_file()]
            quests.sort(key=lambda p: (p / ".fi" / _attempts.ATTEMPTS).stat().st_mtime, reverse=True)
        except OSError:
            quests = []
        paths += [q / ".fi" / _attempts.ATTEMPTS for q in quests[:MAX_QUESTS]]
        if extra is not None and (extra / _attempts.ATTEMPTS) not in paths:
            paths.append(extra / _attempts.ATTEMPTS)
        return paths

    def attempts(self, *, also: Path | None = None) -> list[Attempt]:
        out: list[Attempt] = []
        for path in self._paths(also):
            try:
                st = path.stat()
            except OSError:
                continue
            key = str(path)
            with self._lock:
                cached = self._files.get(key)
            if cached is None or cached[0] != st.st_mtime or cached[1] != st.st_size:
                found = [a for a in (_as_attempt(r) for r in _attempts.read(path.parent, path.name)) if a is not None]
                cached = (st.st_mtime, st.st_size, found)
                with self._lock:
                    self._files[key] = cached
            out += cached[2]
        return out


# The one piece of shared state in this module: a read-only, derived cache of other quests' failure summaries, one per
# output root, guarded by a lock so N engines in one process (a fleet) read each file once. It holds nothing an engine
# writes or decides with.
_INDEXES: dict[str, Index] = {}
_INDEXES_LOCK = threading.Lock()


def index_for(output_root: Path) -> Index:
    key = str(Path(output_root).resolve())
    with _INDEXES_LOCK:
        found = _INDEXES.get(key)
        if found is None:
            found = _INDEXES[key] = Index(Path(output_root))
        return found


@dataclass
class Recommendation:
    action: str
    reason: str
    matched_attempt_ids: list[str] = field(default_factory=list)
    fields_matched: list[str] = field(default_factory=list)
    fields_differed: list[str] = field(default_factory=list)
    fields_unknown: list[str] = field(default_factory=list)
    failure_signature: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "matched_attempt_ids": self.matched_attempt_ids,
                "fields_matched": self.fields_matched, "fields_differed": self.fields_differed,
                "fields_unknown": self.fields_unknown, "failure_signature": self.failure_signature}


def _compare(cand: dict[str, Any], past: dict[str, Any], decision: str) -> tuple[list[str], list[str], list[str]]:
    same, differ, unknown = [], [], []
    for name in KEYS[decision]:
        a, b = cand.get(name), past.get(name)
        if a is None or b is None:
            unknown.append(name)
        elif a == b:
            same.append(name)
        else:
            differ.append(name)
    for name in OPTIONAL[decision]:
        a, b = cand.get(name), past.get(name)
        if a is not None and b is not None:
            (same if a == b else differ).append(name)
    return same, differ, unknown


def recommend(candidate: dict[str, Any], attempts: list[Attempt], *, decision: str,
              exclude_ids: set[str] | None = None, exclude_lineage: tuple[str, Any] | None = None) -> Recommendation:
    """What past failures would recommend for ``decision``, whose comparable parts are ``candidate`` (from
    :func:`parts`). ``exclude_ids`` / ``exclude_lineage`` (quest id, design revision) are never matched: a repair is not
    compared with the run that triggered it or the repairs of the same design. Pure."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}; got {decision!r}")
    exclude_ids = exclude_ids or set()
    exact: list[tuple[Attempt, list[str], list[str]]] = []
    other_model: list[tuple[Attempt, list[str], list[str]]] = []
    similar: list[tuple[Attempt, list[str], list[str], list[str]]] = []
    for past in attempts:
        if past.record_id in exclude_ids:
            continue
        if exclude_lineage is not None and (past.quest_id, past.design_revision) == exclude_lineage:
            continue
        pp = past.decision_parts.get(decision) or {}
        same, differ, unknown = _compare(candidate, pp, decision)
        trusted = past.current_schema and not unknown
        model_only = bool(differ) and all(d in _MODEL_PARTS for d in differ)
        if decision == "plan" and not differ and trusted and candidate.get("features") != pp.get("features"):
            similar.append((past, same, ["features"], unknown))
        elif not differ and trusted:
            exact.append((past, same, differ))
        elif model_only and trusted:
            other_model.append((past, same, differ))
        elif not differ or any(a in same for a in ANCHORS[decision]):
            similar.append((past, same, differ, unknown))
    if exact:
        groups: dict[str, list[tuple[Attempt, list[str], list[str]]]] = {}
        for item in exact:
            groups.setdefault(_sig_key(item[0].signature) or "?", []).append(item)
        items = max(groups.values(), key=len)
        sig = items[0][0].signature
        ids = [i[0].record_id for i in items]
        if decision == "execute" and len(items) >= REPRODUCED and not sig.get("transient"):
            return Recommendation("BLOCK", f"the same conditions failed the same way {len(items)} times",
                                  ids, items[0][1], items[0][2], [], sig)
        why = "the same conditions failed before"
        if decision == "execute" and sig.get("transient"):
            why += " (a time limit, memory or connection failure, which is not taken as reproducible)"
        elif decision != "execute":
            why += f"; at {decision} it is at most a check"
        return Recommendation("VERIFY", why, [i[0].record_id for i in exact], items[0][1], items[0][2], [], sig)
    if other_model:
        first = other_model[0]
        return Recommendation("VERIFY", "the same conditions failed under another model",
                              [i[0].record_id for i in other_model], first[1], first[2], [], first[0].signature)
    if similar:
        first = similar[0]
        why = "a similar attempt failed" + (" (some conditions are not known)" if first[3] else "")
        return Recommendation("INFO", why, [i[0].record_id for i in similar[:20]], first[1], first[2], first[3],
                              first[0].signature)
    return Recommendation("IGNORE", "nothing comparable failed")


def record(fi_dir: Path, quest_id: str, *, record_id: str, decided_at: float, decision: str, taken: str,
           candidate: dict[str, Any], context_missing: list[str], model_family: str,
           recommendation: Recommendation, excluded: list[str]) -> str | None:
    """Append one shadow recommendation (every compared part's value and what the context could not work out, so the
    rules can be scored again offline); its id, or ``None`` when it could not be written (counted)."""
    rid = _attempts.append(fi_dir, SHADOW, {
        "kind": "shadow", "quest_id": quest_id, "record_id": record_id, "at": decided_at, "decision": decision,
        "taken": taken, "model_family": model_family, "parts": candidate, "context_missing": list(context_missing),
        "excluded_attempt_ids": excluded, **recommendation.as_dict(),
    })
    if rid is None:
        _attempts.count_lost(fi_dir, SHADOW_LOST)
    return rid


# ---- lineage: which run, stop or quest record a recommendation is about ------------------------------------------

def _read_pending(fi_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads((fi_dir / SHADOW_PENDING).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_pending(fi_dir: Path, data: dict[str, Any]) -> None:
    try:
        fi_dir.mkdir(parents=True, exist_ok=True)
        (fi_dir / SHADOW_PENDING).write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def stamp(fi_dir: Path, decision: str, record_id: str) -> None:
    """Remember a recommendation until the record it is about is written: a plan's until the quest ends, the others'
    until the next run or stop."""
    data = _read_pending(fi_dir)
    if decision == "plan":
        data["plan"] = record_id
    else:
        data["pending"] = [*list(data.get("pending") or []), record_id]
    _write_pending(fi_dir, data)


def take(fi_dir: Path, kind: str) -> list[str]:
    """The recommendation ids a run / stop / quest record is about, cleared as they are used."""
    data = _read_pending(fi_dir)
    if not data:
        return []
    ids = list(data.get("pending") or [])
    if data.get("plan"):
        ids.append(str(data["plan"]))
    data["pending"] = []
    if kind == "quest":
        data["plan"] = None
    _write_pending(fi_dir, data)
    return ids


# ---- scoring ------------------------------------------------------------------------------------------------------

def _usable(quest_record: dict[str, Any]) -> bool:
    return (str(quest_record.get("execution_status") or "") in ("completed", "data_analysis", "no_experiment_by_design")
            and quest_record.get("review_status") == "accepted")


def _followed_by_failure(rec: dict[str, Any], attempts: list[dict[str, Any]]) -> bool | None:
    """Whether the decision a recommendation was made at failed, from the records stamped with its id: a plan fails when
    the quest stops at a check or ends without a usable result (an accepted review of a finished run); an implement,
    execute or repair when the run or stop it led to failed. ``None`` when nothing it led to is recorded yet."""
    rid = rec.get("record_id")
    about = [r for r in attempts if rid in (r.get("parent_shadow_ids") or [])]
    if rec.get("decision") == "plan":
        if any(r.get("kind") == "stop" and r.get("outcome") in FAILURES for r in about):
            return True
        quest = next((r for r in about if r.get("kind") == "quest"), None)
        return None if quest is None else not _usable(quest)
    first = next((r for r in about if r.get("kind") in ("run", "stop")), None)
    if first is None:
        quest = next((r for r in about if r.get("kind") == "quest"), None)
        return None if quest is None else quest.get("execution_status") == "crashed"
    return first.get("outcome") in FAILURES


@dataclass
class Score:
    counts: dict[str, dict[str, dict[str, int]]]
    by_family: dict[str, dict[str, int]]
    lost: int
    quests: int


def score(output_root: Path) -> Score:
    """For every shadow recommendation under ``output_root``, by decision and action: ``failed`` (the decision was
    followed by a failure), ``ok`` (it was not), ``open`` (nothing recorded yet)."""
    counts = {d: {a: {"failed": 0, "ok": 0, "open": 0} for a in ACTIONS} for d in DECISIONS}
    by_family: dict[str, dict[str, int]] = {}
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
            decision, action = str(rec.get("decision") or ""), str(rec.get("action") or "")
            if decision not in counts or action not in ACTIONS:
                continue
            outcome = _followed_by_failure(rec, attempts)
            counts[decision][action]["open" if outcome is None else ("failed" if outcome else "ok")] += 1
            fam = by_family.setdefault(str(rec.get("model_family") or "?/?"), {"recommendations": 0, "warned": 0})
            fam["recommendations"] += 1
            fam["warned"] += int(action in ("BLOCK", "VERIFY"))
    return Score(counts, by_family, lost, quests)


def report_lines(result: Score) -> list[str]:
    """The score in plain lines: counts only, with the rate of failures after an IGNORE as the base rate."""
    lines = [f"Shadow recommendations in {result.quests} quest(s): recorded only — no route or state was changed."]
    if not any(sum(c.values()) for d in result.counts.values() for c in d.values()):
        return lines + ["No recommendation recorded yet."]
    for decision in DECISIONS:
        per = result.counts[decision]
        if not any(sum(c.values()) for c in per.values()):
            continue
        lines.append(f"  at {decision}:")
        for action in ACTIONS:
            c = per[action]
            n = sum(c.values())
            if n:
                lines.append(f"    {action}: {n} — followed by a failure {c['failed']}, by none {c['ok']}, "
                             f"not known yet {c['open']}")
        ign = per["IGNORE"]
        known = ign["failed"] + ign["ok"]
        if known:
            lines.append(f"    base rate: with nothing to warn about, a failure followed {ign['failed']} of {known} "
                         f"({100 * ign['failed'] / known:.0f}%)")
    for fam, c in sorted(result.by_family.items()):
        lines.append(f"  {fam}: {c['recommendations']} recommendation(s), {c['warned']} would have warned")
    if result.lost:
        lines.append(f"  {result.lost} recommendation(s) were not written (worked out too late, or could not be).")
    lines.append("These counts do not show that acting on the recommendations would help: that needs a planned "
                 "comparison.")
    return lines
