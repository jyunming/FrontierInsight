"""Scoring benchmark runs: whether the answer FI computed is right, how far the result got, and which check caught what.

What is read, and what is never read:

* The answer is FI's own record of the trials (``raw/trials.json``, written by FI's harness from what each trial
  returned), checked against the hashes FI kept of it (``.fi/trials/run.json``): a record edited after FI wrote it is
  ``unscorable``, not right or wrong. Nothing the model printed (``RESULT_JSON``, the paper, the PDF) is read for it.
* How far the result got is the evidence record as every surface reads it (``core.evidence.read``, its seal checked).
  "Would be published" is ``publication_ready``, or one level below it when the only gap left is that no person
  accepted it (benchmark runs are accepted automatically on a passing review: runner.py).
* Which check flagged a run comes from each check's own record (needs/*.json, paper/*_audit.json, the receipts, the
  claim ledger, the retraction lookups) and the evidence gaps, sorted into the gates of catalogue.GATES; the first one in
  that order is where the error was first caught. A gap the run's control (the same copy run again from the same step
  with nothing planted) also has is the fork's doing, not a detection, and is left out.

A literature task (Q11) is scored on structure only: the retracted paper is not cited, the studies the answer rests on
were found, and most claims are grounded.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from core import acceptance as _acceptance
from core import evidence as _evidence
from core import receipts as _receipts
from core import replay as _replay
from core import retractions as _retractions
from core import trial_runner as _trials

from . import answers as _answers
from . import catalogue
from . import plant as _plant
from . import runner as _runner

# A provider outage or a spent quota, in the words a stop or a failure uses: not a check of FI's.
INFRA_RE = re.compile(r"\brate.?limit|\bquota\b|\b429\b(?!\d)|too many requests|\boverloaded|\btimed out\b"
                      r"|\bcould not reach\b|\bconnect(?:ion)?(?:error|timeout)\b|\bconnection (?:error|refused|reset)\b"
                      r"|\b(?:read|write|pool)timeout\b|\btimeouterror\b|\busage limit\b|\bweekly limit\b", re.I)


def _json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """The Wilson score interval of k successes in n (95% by default); ``None`` for n = 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# --- the answer, from FI's own record ---------------------------------------------------------------------------------

def engine_values(root: Path, metric: str, cell: dict[str, Any]) -> tuple[list[float] | None, str]:
    """The values FI recorded for ``metric`` in the one cell matching ``cell``: ``(values, "")`` or ``(None, why)``."""
    summary_path = root / _trials.RAW_DIRNAME / _trials.SUMMARY_NAME
    record = _json(root / _trials.RUN_RECORD)
    if not summary_path.is_file() or not isinstance(record, dict):
        return None, "FI has no trial record for this run (raw/trials.json, .fi/trials/run.json)"
    if record.get("files") != _trials._file_hashes(root):
        return None, "FI's trial record was changed after FI wrote it"
    summary = _json(summary_path) or {}
    matching = [c for c in summary.get("cells") or [] if _answers.cell_matches(c.get("cell") or {}, cell)]
    if len(matching) != 1:
        return None, f"{len(matching)} of FI's settings match {cell} (exactly one must)"
    metrics = matching[0].get("metrics") or {}
    found = next((v for k, v in metrics.items() if str(k).lower() == metric.lower()), None)
    if not isinstance(found, dict):
        return None, f"FI's record of {matching[0].get('key')} has no metric {metric!r} (it has {', '.join(metrics) or 'none'})"
    values = [float(v) for v in found.get("values") or [] if isinstance(v, (int, float)) and math.isfinite(v)]
    if not values:
        return None, f"FI recorded no finite value of {metric!r} in {matching[0].get('key')}"
    return values, ""


def check_answers(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    """Each answer of a simulation task against FI's record; ``correct`` is True, False, or None (unscorable)."""
    rows = []
    for a in task.get("answers") or []:
        values, why = engine_values(root, a["metric"], a["cell"])
        if values is None:
            rows.append({"metric": a["metric"], "cell": a["cell"], "expected": a["expected"], "got": None, "ok": None,
                         "why": why})
            continue
        got = values[0] if a["statistic"] == "value" and len(values) == 1 else sum(values) / len(values)
        rows.append({"metric": a["metric"], "cell": a["cell"], "expected": a["expected"], "got": got,
                     "n": len(values), "ok": _answers.within(got, a), "why": ""})
    oks = [r["ok"] for r in rows]
    correct = None if not rows or any(o is None for o in oks) else all(oks)
    return {"correct": correct, "values": rows}


def _paper_text(root: Path) -> str:
    parts = []
    for rel in ("paper/paper.md", "paper.md"):
        path = root / rel
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
            break
    for path in sorted((root / "paper").glob("*.bib")) + sorted((root / "paper").glob("*.ris")):
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def check_structure(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    """A literature task's structural answer: no retracted paper cited, the must-find studies found, claims grounded."""
    s = task["structural"]
    paper = _paper_text(root).lower()
    searched = json.dumps(_json(root / ".fi" / "literature_queries.json") or {}).lower()
    cited_bad = [r["doi"] for r in s["must_not_cite"] if r["doi"].lower() in paper]
    found = [r["doi"] for r in s["must_find"] if r["doi"].lower() in searched or r["doi"].lower() in paper]
    recall = len(found) / len(s["must_find"]) if s["must_find"] else 1.0
    claims = _json(root / "paper" / "claims.json") or {}
    total = int(claims.get("total") or 0)
    share = (int(claims.get("grounded") or 0) / total) if total else None
    correct = (not cited_bad and recall >= float(s["min_recall"])
               and (share is None or share >= float(s["min_supported_share"])))
    return {"correct": correct, "cites_retracted": cited_bad, "found": found, "recall": recall,
            "supported_share": share}


# --- where it was caught ----------------------------------------------------------------------------------------------

def gate_of(level: str, gap: str) -> str | None:
    """The gate an evidence gap belongs to; ``None`` for the accept-by-no-person gap (not a check)."""
    text = gap.lower()
    if gap == _acceptance.NO_PERSON_GAP:
        return None
    if level == "executed":
        return "run"
    if level == "internally_reconciled":
        return "numbers"
    if "retract" in text:
        return "retractions"
    if level == "protocol_runtime_matched":
        if "numeric" in text:
            return "numeric_warnings"
        if "manifest" in text or "trial record" in text or "trials" in text:
            return "run_record"
        if "best design" in text or "finer" in text:
            return "optimum"
        return "protocol"
    if level == "independently_validated":
        return "optimum" if ("best design" in text or "finer numerical" in text) else "oracle"
    if level == "statistically_adequate":
        return "statistics"
    if "claim check" in text:
        return "claim_check"
    if "evidence gate" in text:
        return "evidence_gate"
    if "methodology audit" in text:
        return "design_audit"
    if "review" in text or "must-fix" in text:
        return "review"
    if "trace" in text or "seal" in text:
        return "trace"
    return "other"


def _finding_text(f: Any) -> str:
    if isinstance(f, dict):
        return str(f.get("message") or f.get("token") or f.get("kind") or json.dumps(f, sort_keys=True))
    return str(f)


def _record_flags(root: Path) -> dict[str, str]:
    """The gates whose own record says this run failed them, with what the record found (the findings themselves,
    so a planted run's finding is told apart from a different one its control also had)."""
    needs = root / "needs"
    flags: dict[str, str] = {}
    retracted = _retractions.retracted_in_record(root / ".fi")
    if retracted:
        flags["retractions"] = "retracted source(s) found: " + "; ".join(sorted(retracted))[:600]
    for gate, name in (("oracle", "ORACLE_CHECK.json"), ("protocol", "PROTOCOL_CHECK.json"),
                       ("run_record", "RUN_MANIFEST_CHECK.json"), ("optimum", "OPTIMUM_CHECK.json")):
        rec = _json(needs / name)
        if isinstance(rec, dict) and rec.get("status") not in (None, "ok", "not_applicable", "single_script"):
            problems = rec.get("problems") or rec.get("differences") or []
            flags[gate] = f"{name}: {rec.get('status')}" + (
                ": " + "; ".join(sorted(_finding_text(p) for p in problems))[:600] if problems else "")
    found = []
    for name in ("numeric_audit.json", "provenance_audit.json", "statistics_audit.json"):
        rec = _json(root / "paper" / name)
        if isinstance(rec, dict) and rec.get("ok") is False:
            found += [f"paper/{name}: {_finding_text(f)}" for f in rec.get("findings") or []] or [f"paper/{name}"]
    if found:
        flags["numbers"] = "; ".join(sorted(found))[:1200]
    claims = _json(root / "paper" / "claims.json")
    if isinstance(claims, dict) and claims.get("unsupported"):
        flags["claim_check"] = "paper/claims.json, unsupported: " + "; ".join(
            sorted(str(c) for c in claims["unsupported"]))[:1200]
    for check in ("evidence_gate", "claim_check", "design_audit"):
        status, _rec, problem = _receipts.read(root, check)
        if status in ("fail", "unknown") and not problem:
            flags.setdefault(check, f"needs/receipts/{check}.json: {status}")
    return flags


def _stop_gate(text: str) -> str:
    t = text.lower()
    for gate, words in (("oracle", ("oracle", "reference check", "check of correctness")),
                        ("protocol", ("protocol",)), ("run_record", ("trial", "run record", "manifest")),
                        ("numbers", ("number", "provenance")), ("claim_check", ("claim",)),
                        ("evidence_gate", ("evidence",)), ("review", ("review",))):
        if any(w in t for w in words):
            return gate
    return "run"


def _gaps(record: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not isinstance(record, dict):
        return []
    return [(level, str(g)) for level, gaps in (record.get("all_gaps") or {}).items() for g in gaps]


def _tokens(bench: Path) -> tuple[int, int]:
    """(tokens, calls) of the model calls this run really made: the ``real`` rows of its ``calls.jsonl`` (a replayed
    or planted answer cost nothing now, whatever the recording says it cost then)."""
    tokens = calls = 0
    path = bench / "calls.jsonl"
    if not path.is_file():
        return 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("source") not in ("real", "error"):
            continue
        calls += 1
        usage = row.get("usage") or {}
        tokens += int(usage.get("total_tokens") or (int(usage.get("prompt_tokens") or 0)
                                                    + int(usage.get("completion_tokens") or 0)))
    return tokens, calls


def _cites_source(root: Path, hits: list[dict[str, Any]]) -> bool:
    """Whether the paper rests on one of ``hits`` (a planted search result): its DOI or title is in the paper, or a
    claim of the paper quotes its text."""
    paper = _paper_text(root).lower()
    claims = (_json(root / "paper" / "claims.json") or {}).get("claims") or []
    for hit in hits:
        meta = hit.get("metadata") or {}
        doi, title = str(meta.get("doi") or "").lower(), str(meta.get("title") or "")[:40].lower()
        if (doi and doi in paper) or (title and title in paper):
            return True
        text = " ".join(str(hit.get("content") or "").split()).lower()
        if any(len(q := " ".join(str(c.get("quote") or "").split()).lower()) >= 20 and q in text for c in claims):
            return True
    return False


_DIGITS = re.compile(r"\d+(?:\.\d+)?")


def _same_reason(text: str) -> str:
    """A flag's words with its numbers taken out: "3 finding(s)" in a planted run and "1 finding(s)" in its control
    are the same reason."""
    return _DIGITS.sub("#", text).strip().lower()


def score_run(run_dir: Path, task: dict[str, Any], *, control: dict[str, Any] | None = None,
              role: str | None = None, needs_control: bool = False) -> dict[str, Any]:
    """One run's outcome (see the module docstring). ``control``: the scored control of the same copy and step;
    ``needs_control``: a planted run with none is not counted (nothing shows the copy alone would be published)."""
    run_dir = Path(run_dir)
    root = _runner.quest_root(run_dir)
    _out, bench = _runner.paths(run_dir)
    meta = _json(bench / "run.json") or {}
    plant = _plant.read_record(bench) or {}
    error = plant.get("error")
    role = role or ("control" if plant.get("control") else "planted" if error else "clean")
    stop_text = ""
    for name in ("NEXT_STEP.md", "quest_failed.md"):
        if (root / name).is_file():
            stop_text = (root / name).read_text(encoding="utf-8", errors="replace")
            break
    if meta.get("error") and not stop_text:
        stop_text = str(meta["error"])
    infra = bool(stop_text) and bool(INFRA_RE.search(stop_text))
    record = _evidence.read(root) if (root / "needs" / "EVIDENCE.json").is_file() else None
    level = (record or {}).get("status") if not stop_text else None
    gaps = _gaps(record)
    ready_or_one_below = isinstance(record, dict) and (
        record.get("status") == "publication_ready"
        or (record.get("status") == "statistically_adequate" and list(record.get("gaps") or []) == [_acceptance.NO_PERSON_GAP]))
    would_publish = bool(ready_or_one_below) and not stop_text

    # Flags: the checks' own records and the evidence gaps, less what the control run also has (the same gate for the
    # same reason, its numbers aside). A gate flagged only for a reason the control shares is kept apart, not dropped.
    # A check's record is compared by what it found (its findings, word for word); an evidence gap by its words with
    # its counts left out, except the paper-number audits' gap, which is only a count (their findings decide).
    control_records = set((control or {}).get("_record_flags", {}).items())
    control_gaps = {(gate_of(lvl, gap) or "", gap if lvl == "internally_reconciled" else _same_reason(gap))
                    for lvl, gap in (control or {}).get("_gaps", [])}
    flags: dict[str, str] = {}
    shared: dict[str, str] = {}
    for g, why in _record_flags(root).items():
        (shared if (g, why) in control_records else flags)[g] = why
    for lvl, gap in gaps:
        if stop_text and lvl == "executed":
            continue  # a stopped quest has no results: that follows from the stop, it is not a check of its own
        gate = gate_of(lvl, gap)
        if not gate or gate in flags:
            continue
        if (gate, gap if lvl == "internally_reconciled" else _same_reason(gap)) in control_gaps or gate in shared:
            shared.setdefault(gate, gap)
        else:
            flags.setdefault(gate, gap)
    if stop_text and not infra and not flags:
        # The stop's own words, when no check's record names what stopped it.
        flags[_stop_gate(stop_text)] = stop_text.splitlines()[0][:300]
    flagged = [g for g in catalogue.GATES if g in flags]

    if task.get("kind") == "literature":
        answer = check_structure(root, task)
    else:
        answer = check_answers(root, task)
    divergences = int(meta.get("divergences") or 0)
    events = _replay.read_events(bench)
    valid: bool | None = None
    why_not = ""
    if error:
        validity = _json(bench / "validity.json")
        if isinstance(validity, dict) and validity.get("valid") is not None:
            valid = bool(validity["valid"])
            why_not = "" if valid else str(validity.get("why") or "an equivalent change")
        elif error in ("R1", "R2"):
            valid = True  # the paper states a number the run did not compute, by construction (plant.py)
        elif error == "L1":
            # Exercised when the same source, never retracted, is one the paper rests on in the control: then the
            # retracted one had the same chance to be used, whatever FI then did with it.
            valid = bool(control.get("cites_planted")) if control else None
            why_not = "" if valid else ("the control's paper does not rest on the planted kind of source"
                                        if control else "there is no control run of the same task")
        if valid and plant.get("answers") and not any(e.get("event") == "planted" for e in events):
            valid, why_not = None, "the planted answer was never asked for"
        if valid and divergences:
            valid, why_not = None, f"{divergences} call(s) were not in the recording: the run did not follow it"
        if valid and needs_control and control is None:
            valid, why_not = None, "there is no control run of the same copy and step"
        if valid and control is not None and not control.get("would_publish"):
            valid, why_not = None, "its control would not be published either: this step cannot be measured by a copy"
    tokens, calls = _tokens(bench)
    cites_planted = _cites_source(root, plant.get("hits") or []) if plant.get("hits") else None
    # A planted error let through: the result would be published. For L1, only when the paper also rests on the
    # retracted source: FI keeping it out of a published paper is the check working.
    let_through = would_publish and (error != "L1" or bool(cites_planted))
    return {
        "run": run_dir.name, "task": task["task"], "error": error, "role": role, "mode": meta.get("mode"),
        "from_step": meta.get("from_step"), "answer": answer, "evidence_level": level,
        "would_publish": would_publish, "accepted": bool((record or {}).get("accepted_by")) and not stop_text,
        "at_least_validated": bool(level) and level in _evidence.LEVELS
        and _evidence.LEVELS.index(level) >= _evidence.LEVELS.index("independently_validated"),
        "stopped": stop_text.splitlines()[0][:300] if stop_text else "", "infrastructure_failure": infra,
        "flagged": flagged, "flag_reasons": {g: flags[g] for g in flagged},
        "first_gate": flagged[0] if flagged else None, "flagged_as_control": sorted(shared),
        "valid": valid, "not_counted_because": why_not if error and valid is not True else "",
        "cites_planted": cites_planted, "let_through": let_through if error else None,
        "divergences": divergences, "tokens": tokens, "calls": calls,
        "seconds": meta.get("seconds"),
        # Kept for the runs that use this one as their control; dropped from the report.
        "_gaps": [list(g) for g in gaps], "_record_flags": _record_flags(root),
    }


# --- the benchmark's numbers ------------------------------------------------------------------------------------------

def _rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": list(ci) if ci else None}


def _let(o: dict[str, Any]) -> bool:
    """Whether a planted error was let through (score_run's ``let_through``; ``would_publish`` for an older outcome)."""
    return bool(o["let_through"] if o.get("let_through") is not None else o["would_publish"])


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """The design's metrics over scored runs (design section 4)."""
    planted = [o for o in outcomes if o["role"] == "planted" and o.get("valid") is True]
    clean = [o for o in outcomes if o["role"] == "clean"]
    by_error: dict[str, dict[str, Any]] = {}
    for err in sorted({o["error"] for o in planted}):
        runs = [o for o in planted if o["error"] == err]
        by_error[err] = _rate(sum(_let(o) for o in runs), len(runs))
    false_pass = {"by_error": by_error, "total": _rate(sum(_let(o) for o in planted), len(planted)),
                  "not_valid": sum(1 for o in outcomes if o["role"] == "planted" and o.get("valid") is not True)}
    right = [o for o in clean if o["answer"].get("correct") is True and not o["infrastructure_failure"]]
    false_block = _rate(sum(not o["would_publish"] for o in right), len(right))
    published = [o for o in clean if o["would_publish"] and o["answer"].get("correct") is not None]
    after_publication = _rate(sum(o["answer"]["correct"] is False for o in published), len(published))
    matrix: dict[str, dict[str, int]] = {}
    for o in planted:
        row = matrix.setdefault(o["error"], {})
        # Held back, but only by checks its control failed the same way: listed apart, not as a detection of a gate.
        key = o["first_gate"] or ("control_also" if o.get("flagged_as_control") and not _let(o) else "none")
        row[key] = row.get(key, 0) + 1
        if not _let(o):
            row["any"] = row.get("any", 0) + 1
    calibration: dict[str, dict[str, Any]] = {}
    for o in outcomes:
        if o["answer"].get("correct") is None:
            continue
        level = o["evidence_level"] or ("stopped" if o["stopped"] else "none")
        c = calibration.setdefault(level, {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += int(o["answer"]["correct"] is True)
    for level, c in calibration.items():
        # Under 10 runs a rate says little: only the counts are given.
        c.update({"rate": c["correct"] / c["n"], "ci95": list(wilson(c["correct"], c["n"]) or ())} if c["n"] >= 10 else {})
    return {
        "false_pass": false_pass, "false_block": false_block, "error_after_publication": after_publication,
        "detection": matrix, "calibration": calibration,
        "cost": {"tokens": sum(o["tokens"] for o in outcomes), "calls": sum(o["calls"] for o in outcomes),
                 "seconds": round(sum(float(o.get("seconds") or 0) for o in outcomes), 1),
                 "divergences": sum(o["divergences"] for o in outcomes)},
        "runs": len(outcomes),
    }


def public(outcome: dict[str, Any]) -> dict[str, Any]:
    """An outcome without the fields only a run's control needs."""
    return {k: v for k, v in outcome.items() if not k.startswith("_")}


def _plain_cell(cell: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {v}" for k, v in cell.items())


def answer_line(outcome: dict[str, Any]) -> str:
    """The answer in one plain line: what FI computed against what it should be."""
    a = outcome["answer"]
    if "values" in a:
        parts = []
        for v in a["values"]:
            got = "?" if v["got"] is None else f"{v['got']:.4g}"
            parts.append(f"{v['metric']} at {_plain_cell(v['cell'])}: {got} (expected {v['expected']})"
                         + (f" [{v['why']}]" if v["why"] else ""))
        return "; ".join(parts)
    return (f"retracted cited: {', '.join(a['cites_retracted']) or 'none'}; found {len(a['found'])} "
            f"(recall {a['recall']:.2f}); grounded share {a['supported_share']}")
