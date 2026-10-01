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
from pathlib import Path
from typing import Any

from core import acceptance as _acceptance
from core import evidence as _evidence
from core import receipts as _receipts
from core import retractions as _retractions
from core import trial_runner as _trials

from . import answers as _answers
from . import catalogue
from . import plant as _plant
from . import runner as _runner

INFRA_WORDS = ("rate limit", "quota", "429", "timed out", "could not reach", "connection", "usage limit")


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


def _record_flags(root: Path) -> dict[str, str]:
    """The gates whose own record says this run failed them, with the record's words."""
    needs = root / "needs"
    flags: dict[str, str] = {}
    retracted = _retractions.retracted_in_record(root / ".fi")
    if retracted:
        flags["retractions"] = "retracted source(s) found: " + "; ".join(retracted)[:300]
    for gate, name in (("oracle", "ORACLE_CHECK.json"), ("protocol", "PROTOCOL_CHECK.json"),
                       ("run_record", "RUN_MANIFEST_CHECK.json"), ("optimum", "OPTIMUM_CHECK.json")):
        rec = _json(needs / name)
        if isinstance(rec, dict) and rec.get("status") not in (None, "ok", "not_applicable", "single_script"):
            flags[gate] = f"{name}: {rec.get('status')}"
    for name in ("numeric_audit.json", "provenance_audit.json", "statistics_audit.json"):
        rec = _json(root / "paper" / name)
        if isinstance(rec, dict) and rec.get("ok") is False:
            flags.setdefault("numbers", f"paper/{name}: {len(rec.get('findings') or [])} finding(s)")
    claims = _json(root / "paper" / "claims.json")
    if isinstance(claims, dict) and claims.get("unsupported"):
        flags["claim_check"] = f"paper/claims.json: {len(claims['unsupported'])} unsupported claim(s)"
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


def _tokens(root: Path, since: float | None) -> tuple[int, int]:
    """(tokens, calls) of the model calls this run made (``.fi/cost.jsonl`` rows after ``since``)."""
    tokens = calls = 0
    path = root / ".fi" / "cost.jsonl"
    if not path.is_file():
        return 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or "node" not in row or (since is not None and float(row.get("ts") or 0) < since):
            continue
        calls += 1
        usage = row.get("usage") or {}
        tokens += int(usage.get("total_tokens") or (int(usage.get("prompt_tokens") or 0)
                                                    + int(usage.get("completion_tokens") or 0)))
    return tokens, calls


def _l1_exercised(root: Path, plant: dict[str, Any]) -> bool:
    """Whether the planted retracted paper reached the paper: cited by DOI or title, or a claim rests on it."""
    paper = _paper_text(root).lower()
    if str(plant.get("doi") or "").lower() in paper or str(plant.get("title") or "")[:40].lower() in paper:
        return True
    claims = _json(root / "paper" / "claims.json") or {}
    return any("retracted" in str(c.get("evidence") or "").lower() for c in claims.get("claims") or [])


def score_run(run_dir: Path, task: dict[str, Any], *, control: dict[str, Any] | None = None,
              role: str | None = None) -> dict[str, Any]:
    """One run's outcome (see the module docstring). ``control``: the scored control of the same copy and step."""
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
    infra = bool(stop_text) and any(w in stop_text.lower() for w in INFRA_WORDS)
    record = _evidence.read(root) if (root / "needs" / "EVIDENCE.json").is_file() else None
    level = (record or {}).get("status") if not stop_text else None
    gaps = _gaps(record)
    ready_or_one_below = isinstance(record, dict) and (
        record.get("status") == "publication_ready"
        or (record.get("status") == "statistically_adequate" and list(record.get("gaps") or []) == [_acceptance.NO_PERSON_GAP]))
    would_publish = bool(ready_or_one_below) and not stop_text

    # Flags: the checks' own records and the evidence gaps, less what the control run also has.
    control_gaps = {(g[0], g[1]) for g in (control or {}).get("_gaps", [])}
    control_flags = set((control or {}).get("_record_flags", {}))
    flags = {g: why for g, why in _record_flags(root).items() if g not in control_flags}
    for lvl, gap in gaps:
        if (lvl, gap) in control_gaps or (stop_text and lvl == "executed"):
            continue  # a stopped quest has no results: that follows from the stop, it is not a check of its own
        gate = gate_of(lvl, gap)
        if gate:
            flags.setdefault(gate, gap)
    if stop_text and not infra and not flags:
        # The stop's own words, when no check's record names what stopped it.
        flags[_stop_gate(stop_text)] = stop_text.splitlines()[0][:300]
    flagged = [g for g in catalogue.GATES if g in flags]

    if task.get("kind") == "literature":
        answer = check_structure(root, task)
    else:
        answer = check_answers(root, task)
    valid: bool | None = None
    if error:
        validity = _json(bench / "validity.json")
        if isinstance(validity, dict) and validity.get("valid") is not None:
            valid = bool(validity["valid"])
        elif error in ("R1", "R2"):
            valid = True  # the paper states a number the run did not compute, by construction (plant.py)
        elif error == "L1":
            valid = _l1_exercised(root, plant)
    started = meta.get("started_at")
    tokens, calls = _tokens(root, float(started) if started else None)
    return {
        "run": run_dir.name, "task": task["task"], "error": error, "role": role, "mode": meta.get("mode"),
        "from_step": meta.get("from_step"), "answer": answer, "evidence_level": level,
        "would_publish": would_publish, "accepted": bool((record or {}).get("accepted_by")) and not stop_text,
        "at_least_validated": bool(level) and level in _evidence.LEVELS
        and _evidence.LEVELS.index(level) >= _evidence.LEVELS.index("independently_validated"),
        "stopped": stop_text.splitlines()[0][:300] if stop_text else "", "infrastructure_failure": infra,
        "flagged": flagged, "flag_reasons": {g: flags[g] for g in flagged},
        "first_gate": flagged[0] if flagged else None, "valid": valid,
        "divergences": int(meta.get("divergences") or 0), "tokens": tokens, "calls": calls,
        "seconds": meta.get("seconds"),
        # Kept for the runs that use this one as their control; dropped from the report.
        "_gaps": [list(g) for g in gaps], "_record_flags": _record_flags(root),
    }


# --- the benchmark's numbers ------------------------------------------------------------------------------------------

def _rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": list(ci) if ci else None}


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """The design's metrics over scored runs (design section 4)."""
    planted = [o for o in outcomes if o["role"] == "planted" and o.get("valid") is True]
    clean = [o for o in outcomes if o["role"] == "clean"]
    by_error: dict[str, dict[str, Any]] = {}
    for err in sorted({o["error"] for o in planted}):
        runs = [o for o in planted if o["error"] == err]
        by_error[err] = _rate(sum(o["would_publish"] for o in runs), len(runs))
    false_pass = {"by_error": by_error, "total": _rate(sum(o["would_publish"] for o in planted), len(planted)),
                  "not_valid": sum(1 for o in outcomes if o["role"] == "planted" and o.get("valid") is not True)}
    right = [o for o in clean if o["answer"].get("correct") is True and not o["infrastructure_failure"]]
    false_block = _rate(sum(not o["would_publish"] for o in right), len(right))
    published = [o for o in clean if o["would_publish"] and o["answer"].get("correct") is not None]
    after_publication = _rate(sum(o["answer"]["correct"] is False for o in published), len(published))
    matrix: dict[str, dict[str, int]] = {}
    for o in planted:
        row = matrix.setdefault(o["error"], {})
        key = o["first_gate"] or "none"
        row[key] = row.get(key, 0) + 1
        if not o["would_publish"]:
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
