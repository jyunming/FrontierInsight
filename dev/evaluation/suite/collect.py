"""Read every quest of a round and write ``round.json`` and ``round.md``.

Everything comes from records FI wrote (``core/quality_report.py`` is rebuilt from the quest's own files; a stored
``.fi/quality.json`` is not trusted, because an older one lacks fields), plus the round manifest (who ran when) and the
round's ``grades.json`` (filled by a grader, see ``grade.py``). Nothing in a quest folder is changed.

``grades.json`` is ``{topic: {"g1": "pass|fail...", "g2": ..., "g3": ..., "mean": 92.6, "d": {"D1": 100, ...},
"stop_correct": true|false, "class": <taxonomy class>, "stage": ..., "class_note": ..., "notes": ...}}``. Every key is
optional; a missing grade leaves the cell empty, never guessed.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from core import quality_report as qr

from . import runner as _runner
from . import taxonomy

SCHEMA = "fi.suite.round/v1"
GRADES = "grades.json"
OUTCOMES = {"finished": "paper", "honest_stop": "honest_stop", "paused": "pause", "crashed": "crash",
            "running": "running", "unknown": "unknown"}
_PASS = ("g1", "g2", "g3")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _iso(text: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None


def load_grades(round_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(k): v for k, v in _dict(_read_json(Path(round_dir) / GRADES)).items() if isinstance(v, dict)}


def is_pass(value: Any) -> bool:
    return str(value or "").strip().lower().startswith("pass")


def is_fail(value: Any) -> bool:
    return str(value or "").strip().lower().startswith("fail")


def latest_quest(folder: Path) -> Path | None:
    """The quest folder under a topic's output folder (the newest, if a topic was run twice)."""
    quests = qr.quest_dirs([folder])
    if not quests:
        return None
    return max(quests, key=lambda p: max((x.stat().st_mtime for x in (p / ".fi").glob("audit.jsonl")), default=p.stat().st_mtime))


def _stuck_kind(quest: Path) -> dict[str, Any]:
    return _dict(_read_json(quest / "needs" / "STUCK.json"))


def code_facts(quest: Path, report: dict[str, Any]) -> dict[str, Any]:
    """The report's code metrics, with "was the final code read" decided from the code_review.json fields themselves: an
    older record has no ``final_read`` key, and the report would then call every final version unread."""
    code = dict(_dict(report.get("code")))
    cr = _dict(_read_json(quest / ".fi" / "code_review.json"))
    if "final_read" in cr:
        read = bool(cr["final_read"])
    elif cr.get("version") and cr.get("read_version"):
        read = cr["version"] == cr["read_version"]
    elif cr:
        read = False
    else:
        read = None
    code["final_version_read"] = read
    code["skipped"] = str(cr.get("skipped") or "")
    return code


def facts_of(quest: Path, report: dict[str, Any], outcome: str, grade: dict[str, Any] | None) -> dict[str, Any]:
    """What the taxonomy rules read, from the report and the quest's own record files."""
    fi, needs = quest / ".fi", quest / "needs"
    evidence = _dict(_read_json(needs / "EVIDENCE.json"))
    gaps = [str(g) for lst in _dict(evidence.get("all_gaps")).values() for g in _list(lst)]
    covered = _dict(_read_json(fi / "topic_covered.json"))
    typed = _dict(_read_json(fi / "typed_results.json"))
    statics = {k: int(_dict(v).get("used") or 0) for k, v in _dict(_read_json(fi / "static_checks.json")).items()}
    manifest_stop = _dict(_read_json(fi / "run_manifest_stop.json"))
    failed_md = ""
    try:
        failed_md = (quest / "quest_failed.md").read_text(encoding="utf-8")
    except OSError:
        pass
    failure = _dict(report.get("failure"))
    return {
        "outcome": outcome, "stage": str(failure.get("stage") or ""), "chain": [str(c) for c in _list(failure.get("chain"))],
        "headline": report["outcome"]["line"], "stuck": _stuck_kind(quest), "pause": _dict(_read_json(fi / "pause.json")),
        "failed_md": failed_md, "run_manifest_stop": "; ".join(str(p)[:200] for p in _list(manifest_stop.get("problems"))[:1]),
        "typed_results_used": int(typed.get("used") or 0), "static_fired": {k: v for k, v in statics.items() if v > 0},
        "evidence_gaps": gaps, "evidence_status": str(evidence.get("status") or ""),
        "evidence_executed": _dict(evidence.get("levels")).get("executed"),
        "evidence_flaw": any(taxonomy._EVIDENCE_GATE.search(g) for g in gaps), "code": code_facts(quest, report),
        "coverage_ever": [str(x) for x in _list(covered.get("ever_missing"))],
        "coverage_left": [str(x) for x in _list(covered.get("left"))], "grade": grade or {},
    }


def record_contradiction(facts: dict[str, Any]) -> str:
    """Why a DELIVERED paper contradicts FI's own record, or "" (see trend.py for how it is counted)."""
    if facts["outcome"] != "paper":
        return ""
    if facts.get("evidence_status") == "not_executed" or facts.get("evidence_executed") is False:
        return "EVIDENCE.json says the experiment was not executed"
    if facts.get("stage") in ("checks", "code", "run", "analysis"):
        chain = facts.get("chain") or []
        return f"a {facts['stage']} problem was not fixed: {chain[0] if chain else ''}"[:240]
    return ""


def paper_status(outcome: str, grade: dict[str, Any] | None, contradiction: str) -> str | None:
    """``trustworthy`` (G1-G3 pass, no contradiction), ``fake`` (G1 fails or the record contradicts the paper),
    ``flawed`` (a grade was given and is not a pass, and G1 is not the one failing), ``ungraded`` (some of G1-G3 is not
    graded yet and none has failed: the missing grades may still decide it either way); ``None`` when no paper was
    delivered. A paper is ``flawed`` only on a grade that is actually there: a missing grade is never read as a failure."""
    if outcome != "paper":
        return None
    g = grade or {}
    if is_fail(g.get("g1")):
        return "fake"
    # A contradiction from an unfixed problem makes the paper fake only if the grader says it changes a reported number.
    # If the grader says unfixed_changes_numbers is False, the paper is judged by its grades like any other paper.
    if contradiction:
        # The "not executed" case is always fake, regardless of the unfixed_changes_numbers judgment.
        if contradiction.startswith("EVIDENCE.json says the experiment was not executed"):
            return "fake"
        # For unfixed checks/code/run/analysis problems, check the grader's judgment.
        # If unfixed_changes_numbers is False, do NOT return "fake"; fall through to the grade-based decision.
        # If it's True or None (missing), return "fake".
        if g.get("unfixed_changes_numbers") is False:
            pass  # Fall through to the grade-based decision below
        else:
            # unfixed_changes_numbers is True or None (missing)
            return "fake"
    if all(is_pass(g.get(k)) for k in _PASS):
        # a paper may do part of the topic, but only if it says plainly which parts it does not do; a report that does not
        # say how many parts were left out (no `PARTS NOT DONE` line) has not decided it yet
        if g.get("parts_not_done") is None:
            return "ungraded"
        if (g.get("parts_not_done") or 0) > 0 and g.get("parts_stated") is False:
            return "flawed"
        return "trustworthy"
    if any(_given(g.get(k)) and not is_pass(g.get(k)) for k in _PASS):
        return "flawed"
    return "ungraded"


def _given(value: Any) -> bool:
    return bool(str(value or "").strip())


def stop_status(outcome: str, grade: dict[str, Any] | None, cls: dict[str, Any] | None) -> str | None:
    """For an honest stop: ``false`` (a grader says the stop was wrong, or set the cause to an FI bug), ``correct``
    (a grader says it was right) or ``unjudged``; ``None`` for other outcomes."""
    if outcome != "honest_stop":
        return None
    g = grade or {}
    if g.get("stop_correct") is False or g.get("class") == "fi_bug":
        return "false"
    if g.get("stop_correct") is True:
        return "correct"
    return "unjudged"


def topic_record(topic: str, entry: dict[str, Any], grade: dict[str, Any] | None, *, running: bool | None = None,
                 builder: Callable[..., dict[str, Any]] = qr.build) -> dict[str, Any]:
    ended = entry.get("state") in _runner.ENDED
    quest = latest_quest(Path(entry["quest_root"])) if entry.get("quest_root") else None
    rec: dict[str, Any] = {"topic": topic, "quest_id": None, "folder": str(quest) if quest else None,
                           "state": entry.get("state"), "pid": entry.get("pid"), "grade": grade}
    started, finished = _iso(entry.get("started_at")), _iso(entry.get("ended_at"))
    rec["wall_seconds"] = round((finished - started).total_seconds()) if started and finished else None
    if quest is None:
        rec.update(outcome="crash" if ended else "running" if entry.get("state") == "running" else "queued",
                   outcome_line="No quest folder was written." if ended else "Not finished.", needs_person=False)
        facts = {"outcome": rec["outcome"], "stage": "", "chain": [], "headline": rec["outcome_line"], "grade": grade or {}}
        rec["class"] = taxonomy.classify(facts) if ended else None
        rec.update(paper_status=None, stop_status=None, contradiction="")
        return rec
    report = builder(quest, running=running if running is not None else (entry.get("state") == "running"))
    outcome = OUTCOMES.get(report["outcome"]["kind"], "unknown")
    if outcome == "unknown" and ended:
        outcome = "crash"  # the process is gone and wrote no end record
        report["outcome"]["line"] = "The process ended without writing an end record."
    facts = facts_of(quest, report, outcome, grade)
    contradiction = record_contradiction(facts)
    cls = taxonomy.classify(facts)
    code = facts["code"]
    cost = _dict(report.get("cost"))
    problems = []
    for s in _list(report.get("stages")):
        for item in _list(_dict(s.get("problems")).get("items")):
            problems.append({"stage": s["stage"], "found_by": item.get("found_by"), "text": str(item.get("text"))[:200],
                             "status": item.get("status")})
    rec.update(
        quest_id=report.get("quest_id"), outcome=outcome, outcome_line=report["outcome"]["line"],
        needs_person=bool(report["outcome"].get("needs_person")), evidence_line=report["outcome"].get("evidence", ""),
        stage=facts["stage"], failure_chain=facts["chain"][:8], **{"class": cls},
        checks_fired=problems,
        problems=report.get("problems"),
        code={"final_read": code.get("final_version_read"), "converged": code.get("repairs_converged"),
              "open_per_reading": code.get("open_per_reading"), "open_high_per_reading": code.get("open_high_per_reading"),
              "functions_done": code.get("functions_done"), "functions_total": code.get("functions_total"), "skipped": code.get("skipped"),
              "repair_requests": code.get("repair_requests")},
        coverage={"ever_missing": facts["coverage_ever"], "left": facts["coverage_left"]},
        static_checks=facts["static_fired"], typed_results_used=facts["typed_results_used"],
        evidence={"status": facts["evidence_status"], "gaps": facts["evidence_gaps"][:6]},
        models=cost.get("by_model"), calls=cost.get("calls"), prompt_tokens=cost.get("prompt_tokens"),
        completion_tokens=cost.get("completion_tokens"), usd=cost.get("usd"), work_seconds=report.get("seconds"),
        config=report.get("config"), contradiction=contradiction,
        paper_status=paper_status(outcome, grade, contradiction), stop_status=stop_status(outcome, grade, cls))
    return rec


def metrics(topics: list[dict[str, Any]]) -> dict[str, Any]:
    """The round's suite metrics, counted from the topic records."""
    def n(pred: Callable[[dict[str, Any]], bool]) -> int:
        return sum(1 for t in topics if pred(t))

    return {
        "topics": len(topics), "papers": n(lambda t: t["outcome"] == "paper"),
        "honest_stops": n(lambda t: t["outcome"] == "honest_stop"),
        "pauses_needing_a_person": n(lambda t: t["outcome"] == "pause" and t.get("needs_person", True)),
        "crashes": n(lambda t: t["outcome"] == "crash"), "unfinished": n(lambda t: t["outcome"] in ("running", "queued", "unknown")),
        "papers_trustworthy": n(lambda t: t.get("paper_status") == "trustworthy"),
        "papers_fake": n(lambda t: t.get("paper_status") == "fake"),
        "papers_flawed": n(lambda t: t.get("paper_status") == "flawed"),
        "papers_ungraded": n(lambda t: t.get("paper_status") == "ungraded"),
        "stops_false": n(lambda t: t.get("stop_status") == "false"),
        "stops_correct": n(lambda t: t.get("stop_status") == "correct"),
        "stops_unjudged": n(lambda t: t.get("stop_status") == "unjudged"),
    }


def collect_round(round_dir: Path, *, builder: Callable[..., dict[str, Any]] = qr.build,
                  alive: Callable[[int], bool] = _runner.default_alive) -> dict[str, Any]:
    """Build ``round.json`` and ``round.md`` of a round from its manifest, its quests and its grades."""
    rdir = Path(round_dir)
    manifest = _runner.status(rdir, alive=alive)
    grades = load_grades(rdir)
    topics = [topic_record(t, e, grades.get(t), builder=builder) for t, e in manifest["topics"].items()]
    data = {
        "schema": SCHEMA, "round": manifest["round"], "group": manifest.get("group"), "control": bool(manifest.get("control")),
        "fi": manifest.get("fi"), "fi_git": manifest.get("fi_git"), "fi_note": manifest.get("fi_note"),
        "created_at": manifest.get("created_at"), "collected_at": _runner.now(), "topics": topics, "metrics": metrics(topics),
    }
    (rdir / "round.json").write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")
    (rdir / "round.md").write_text(render_round(data), encoding="utf-8")
    return data


# ----------------------------------------------------------------------------------------------- rendering

def _cell(text: Any) -> str:
    return str(text if text not in (None, "") else "-").replace("|", "/").replace("\n", " ")


def _tok(rec: dict[str, Any]) -> str:
    if rec.get("prompt_tokens") is None:
        return "-"
    return f"{rec['prompt_tokens']:,} / {rec['completion_tokens']:,}"


def _grade_cell(rec: dict[str, Any]) -> str:
    g = rec.get("grade") or {}
    if not g:
        return "to grade" if rec.get("outcome") == "paper" else "-"
    parts = [f"G1 {g.get('g1', '?')}", f"G2 {g.get('g2', '?')}", f"G3 {g.get('g3', '?')}"]
    return ", ".join(parts) + (f"; mean {g['mean']}" if g.get("mean") is not None else "")


def render_round(data: dict[str, Any]) -> str:
    m = data["metrics"]
    fi = data.get("fi_git") or {}
    lines = [f"# Round {data['round']}" + (" (control)" if data.get("control") else ""), "",
             f"FI {fi.get('short', 'unknown')}{' (uncommitted changes)' if fi.get('dirty') else ''}"
             + (f" - {data['fi_note']}" if data.get("fi_note") else "") + f". Collected {data['collected_at']}.", "",
             "## Suite metrics", "",
             f"- Topics: {m['topics']}; papers delivered: {m['papers']}; honest stops: {m['honest_stops']}; "
             f"pauses that needed a person: {m['pauses_needing_a_person']}; crashes: {m['crashes']}; unfinished: {m['unfinished']}.",
             f"- Papers: {m['papers_trustworthy']} trustworthy, {m['papers_fake']} fake, {m['papers_flawed']} flawed, "
             f"{m['papers_ungraded']} not graded yet.",
             f"- Honest stops: {m['stops_false']} judged false, {m['stops_correct']} judged right, {m['stops_unjudged']} not judged.",
             "", "## Per topic", "",
             "| Topic | Outcome | Failing stage | Failure class | Rubric grade (G1/G2/G3, mean) | Tokens (prompt / completion) | Work time |",
             "|---|---|---|---|---|---|---|"]
    for t in data["topics"]:
        c = t.get("class") or {}
        work = t.get("work_seconds")
        lines.append(f"| {_cell(t['topic'])} | {_cell(t['outcome'])}{' (' + t['paper_status'] + ')' if t.get('paper_status') else ''}"
                     f"{' (stop ' + t['stop_status'] + ')' if t.get('stop_status') else ''} | {_cell(c.get('stage') or t.get('stage'))} | "
                     f"{_cell(taxonomy.CLASSES.get(c.get('class'), '')) if c else '-'} | {_cell(_grade_cell(t))} | {_tok(t)} | "
                     f"{qr._span(int(work)) if work else '-'} |")
    for t in data["topics"]:
        lines += ["", f"### {t['topic']}", "", t.get("outcome_line") or ""]
        c = t.get("class")
        if c:
            lines.append(f"- Class: {taxonomy.CLASSES.get(c['class'])} at stage `{c['stage'] or '?'}` ({c['rule']}): {c['evidence']}")
            lines += [f"- Also: {a}" for a in c.get("also", [])]
        if t.get("contradiction"):
            lines.append(f"- Record contradiction: {t['contradiction']}")
        code = t.get("code") or {}
        if code.get("functions_total") or code.get("open_per_reading"):
            lines.append(f"- Code: functions written {code.get('functions_done')}/{code.get('functions_total')}; final version read: "
                         f"{code.get('final_read')}; repairs converged: {code.get('converged')}; open findings per reading: "
                         f"{code.get('open_per_reading')} (high: {code.get('open_high_per_reading')}).")
        cov = t.get("coverage") or {}
        if cov.get("ever_missing") or cov.get("left"):
            lines.append(f"- Topic coverage: {len(cov.get('ever_missing') or [])} requirement(s) missing at some reading, "
                         f"{len(cov.get('left') or [])} left: {', '.join((cov.get('left') or [])[:6]) or 'none'}.")
        if t.get("static_checks"):
            lines.append("- Static checks that fired: " + ", ".join(f"{k} x{v}" for k, v in t["static_checks"].items()) + ".")
        fired = t.get("checks_fired") or []
        if fired:
            lines.append(f"- Problems the checks found: {len(fired)} (" + "; ".join(
                f"{p['stage']}: {p['found_by']} -> {p['status']}" for p in fired[:5]) + (" ..." if len(fired) > 5 else "") + ").")
        models = t.get("models") or {}
        if models:
            lines.append("- Tokens per model: " + "; ".join(f"{k} {v['calls']} calls, {v['prompt_tokens']:,} / {v['completion_tokens']:,}"
                                                              for k, v in models.items()) + ".")
    return "\n".join(lines) + "\n"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
