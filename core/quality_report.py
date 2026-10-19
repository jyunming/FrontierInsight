"""The quest's quality report: where it went wrong, who caught it, what was fixed, and what it cost.

``QUALITY.md`` (plain words, at the quest root) and ``.fi/quality.json`` (the same, structured) are written at every stop
a quest reaches: finished, an honest stop, a pause, a crash (``Engine.run``'s last step calls :func:`write_quietly`). The
report only READS records FI already wrote (the model-call log, the audit trace, the checks' own files, ``needs/``,
``paper/``); it never calls a model, never changes the quest, and a missing or old-format file just leaves its row
shorter. The suggestions are rules over those records (each one names the record line it comes from), not a model's
opinion.

``fi tools quality <folder>...`` reads the same records (or ``.fi/quality.json``) of many quests and prints one row per
quest so configurations can be compared (:func:`table`).
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import review_pairs as _pairs

SCHEMA = "fi.quality/v1"
STAGES = ("plan", "checks", "design", "code", "run", "analysis", "paper", "review")
STAGE_NAME = {"plan": "Plan", "checks": "Checks", "design": "Design", "code": "Code", "run": "Run",
              "analysis": "Analysis", "paper": "Paper", "review": "Review"}
#: Which stage a model call belongs to (the step name, without any ``.suffix``).
_NODE_STAGE = {
    "clarify": "plan", "ideate": "plan", "ideate_query": "plan", "ideate_reflect": "plan", "ideate_tournament": "plan",
    "literature_query": "plan", "literature_foundational": "plan", "literature_screen": "plan", "source_router": "plan",
    "select_skills": "plan", "plan": "plan", "plan_revise": "plan", "plan_criteria": "plan",
    "oracle_review": "checks", "implement_oracle": "checks", "oracle_recompute": "checks",
    "design": "design", "design_self_critique": "design", "design_review": "design",
    "implement_outline": "code", "implement": "code", "implement_body": "code", "implement_function": "code",
    "implement_function_repair": "code", "implement_module_repair": "code", "code_review_repair": "code",
    "code_review": "code", "oracle_review.code": "code", "code_review.repair": "code", "oracle_review.code.repair": "code",
    "oracle_review.topic": "plan",
    "execute_reflect": "run", "improve": "run",
    "analyze": "analysis", "cross_check": "analysis",
    "write": "paper", "figures": "paper", "claim_check": "paper", "citation_check": "paper",
    "oracle_review.citations": "paper",
    "review": "review",
}
#: The steps that write a stage's work, and the steps that read it (the step a person routes with
#: ``provider.node_models.<step>``).
_WRITERS = {"plan": ("plan", "plan_revise"), "checks": ("implement_oracle",), "design": ("design",),
            "code": ("implement", "implement_outline", "implement_body", "implement_function",
                     "implement_function_repair", "implement_module_repair", "code_review_repair"),
            "run": ("execute_reflect", "improve"), "analysis": ("analyze",), "paper": ("write", "figures"),
            "review": ("review",)}
_READERS = {"plan": ("oracle_review.topic",), "checks": ("oracle_review", "oracle_recompute"),
            "design": ("design_review", "design_self_critique"), "code": ("code_review", "oracle_review.code"),
            "analysis": ("cross_check",), "paper": ("claim_check", "citation_check", "oracle_review.citations")}
#: A stage's step to name when a person asks for another model to write (or read) it.
_KEY = {"plan": "plan", "checks": "oracle_review", "design": "design", "code": "implement", "run": "execute_reflect",
        "analysis": "analyze", "paper": "write", "review": "review"}
_RECORD_FILES = ("model_calls.jsonl", "audit.jsonl", "oracle_review.json", "code_review.json", "equation_tests.json",
                 "function_steps.json", "second_impl.json")


def _is_choice(text: str) -> bool:
    """A record's advice is quoted only when it is a genuine choice (a model, more time), never a request to fix code or
    edit a file: people do not debug."""
    low = text.lower()
    if any(w in low for w in ("fix ", "edit", "debug", "script", "code", "file", "rewrite", "by hand", "yourself", "patch")):
        return False
    return any(w in low for w in ("model", "provider.", "timeout", "time limit", "more time", "faster"))


# ----------------------------------------------------------------------------------------------------- reading

def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except OSError:
        pass
    return out


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _one(text: Any, limit: int = 160) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _epoch(ts: Any) -> float | None:
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _stage_of(node: str) -> str | None:
    node = str(node or "")
    return _NODE_STAGE.get(node) or _NODE_STAGE.get(node.split(".")[0])


def _reads(stage: str, node: Any) -> bool:
    """Whether a call of ``node`` is a reviewer's call for ``stage`` (``oracle_review.code`` is the code's, not the
    plan's checks'; any other ``oracle_review.<x>`` is the checks')."""
    node = str(node or "")
    names = _READERS.get(stage, ())
    return node in names or (node.split(".")[0] in names and node not in _NODE_STAGE)


def _model_name(row: dict[str, Any]) -> str:
    return str(row.get("served_model") or row.get("requested_model") or row.get("model") or "").strip()


def _who(row: dict[str, Any]) -> str:
    model = _model_name(row)
    provider = str(row.get("provider") or "").strip()
    return f"{provider}/{model}" if provider and model else (model or provider or "unknown")


class _Quest:
    """Every record of one quest, read once. Any of them may be missing."""

    def __init__(self, root: Path) -> None:
        self.root = root
        fi = root / ".fi"
        self.calls = _jsonl(fi / "model_calls.jsonl")
        self.audit = _jsonl(fi / "audit.jsonl")
        self.oracle_review = _dict(_json(fi / "oracle_review.json"))
        self.code_review = _dict(_json(fi / "code_review.json"))
        self.equation_tests = _dict(_json(fi / "equation_tests.json"))
        self.function_steps = _dict(_json(fi / "function_steps.json"))
        self.second_impl = _dict(_json(fi / "second_impl.json"))
        self.search_asked = _dict(_json(fi / "search_block_asked.json"))
        self.protocol_asked = _dict(_json(fi / "protocol_part_asked.json"))
        self.pilot = _dict(_json(fi / "pilot_check.json"))
        self.trial_checks = _dict(_json(fi / "trial_checks.json"))
        self.drafted_asked = _dict(_json(fi / "drafted_parts_asked.json"))
        self.design_acted = _list(_json(fi / "design_audit_acted.json"))
        self.pause = _dict(_json(fi / "pause.json"))
        self.cost = _dict(_json(fi / "cost.summary.json"))
        self.stuck = _dict(_json(root / "needs" / "STUCK.json"))
        self.critique = [x for x in _list(_json(root / "needs" / "DESIGN_CRITIQUE.json")) if isinstance(x, dict)]
        self.oracle_check = _dict(_json(root / "needs" / "ORACLE_CHECK.json"))
        self.protocol_check = _dict(_json(root / "needs" / "PROTOCOL_CHECK.json"))
        self.result_check = _dict(_json(root / "needs" / "RESULT_CHECK.json"))
        self.evidence = _dict(_json(root / "needs" / "EVIDENCE.json"))
        self.summary = _dict(_json(root / "frontier_insight_summary.json"))
        self.paper_check = _dict(_json(root / ".fi" / "paper_record_check.json"))
        self.numeric_warnings = _dict(_json(root / "needs" / "NUMERIC_WARNINGS.json"))
        self.receipts = {p.stem: _dict(_json(p)) for p in sorted((root / "needs" / "receipts").glob("*.json"))} \
            if (root / "needs" / "receipts").is_dir() else {}
        self.failed_md = _text(root / "quest_failed.md")
        self.config = _read_yaml(root / "config.yaml")

    def checks(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.audit if e.get("kind") == "check_result" and e.get("check") == name]

    def decisions(self, node: str) -> list[dict[str, Any]]:
        return [e for e in self.audit if e.get("kind") == "route_decision" and e.get("node") == node]


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
        data = yaml.safe_load(_text(path))
    except Exception:  # noqa: BLE001 -- a config that cannot be read leaves the config line shorter
        return {}
    return data if isinstance(data, dict) else {}


# ------------------------------------------------------------------------------------------------------ stages

def _problem(stage: str, text: str, by: str, status: str, source: str) -> dict[str, str]:
    return {"stage": stage, "text": _one(text, 220), "found_by": by, "status": status, "source": source}


def _pending_names(problems: list[Any]) -> set[str]:
    names = set()
    for p in problems:
        m = re.search(r"oracle '([^']+)'", str(p))
        names.add(m.group(1) if m else _one(p, 60))
    return names


def _oracle_problems(q: _Quest) -> list[dict[str, str]]:
    """The known-answer checks the engine judged: what failed, and whether the last look still had it failing."""
    rounds: list[tuple[str, list[Any], str]] = []
    for e in q.checks("oracle"):
        rounds.append((str(e.get("status") or ""), _list(e.get("problems")), f".fi/audit.jsonl seq {e.get('seq')}"))
    if not rounds:
        for a in _list(q.oracle_check.get("attempts")):
            a = _dict(a)
            rounds.append(("ok" if not a.get("problems") else "failing", _list(a.get("problems")),
                           f"needs/ORACLE_CHECK.json attempt {a.get('attempt')}"))
        if rounds and q.oracle_check.get("status"):
            rounds[-1] = (str(q.oracle_check["status"]), rounds[-1][1], rounds[-1][2])
    if not rounds:
        return []
    out: list[dict[str, str]] = []
    seen: dict[str, int] = {}
    first_text: dict[str, tuple[str, str]] = {}
    for status, problems, src in rounds:
        for p in problems:
            key = next(iter(_pending_names([p])))
            seen[key] = seen.get(key, 0) + 1
            first_text.setdefault(key, (str(p), src))
    last_status, last_problems, _ = rounds[-1]
    still = _pending_names(last_problems)
    for key, (text, src) in first_text.items():
        if key in still:
            status = "left out" if last_status == "went_on_failing" else "not fixed"
        else:
            status = "fixed"
        tries = f" (failed {seen[key]} times)" if seen[key] > 1 else ""
        out.append(_problem("run", text + tries, "FI's own known-answer check", status, src))
    return out


def _checks_problems(q: _Quest) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    orv = q.oracle_review
    reviewer = str(orv.get("reviewer") or "a second model")
    for f in _list(orv.get("findings")):
        text = _one(f, 200)
        text = re.sub(r"^a check to add:.*", "the reviewer proposed a check that was missing", text)
        out.append(_problem("checks", text, f"{reviewer} (reads the checks)",
                            "fixed" if orv.get("written") else "left out", ".fi/oracle_review.json"))
    eq = q.equation_tests
    eqs = _dict(eq.get("equations"))
    repairs = _dict(eq.get("repairs"))
    for item in _list(eq.get("failed")):
        item = _dict(item)
        eid = str(item.get("id") or "")
        ok = _dict(eqs.get(eid)).get("state") == "ok"
        n = _int(repairs.get(eid))
        out.append(_problem("checks", f"equation {eid}: its function {item.get('function') or ''} failed its test"
                            + (f", repaired {n} time{'s' if n != 1 else ''}" if n else ""),
                            "FI's equation test", "fixed" if ok else "not fixed", ".fi/equation_tests.json"))
    for eid in _list(eq.get("not_tested")):
        out.append(_problem("checks", f"equation {eid} has no test", "FI's equation test", "left out",
                            ".fi/equation_tests.json"))
    for note in _list(q.protocol_check.get("notes")):
        out.append(_problem("checks", str(note), "FI's reading of the code (a note, not a stop)", "left out",
                            "needs/PROTOCOL_CHECK.json"))
    return out


def _code_problems(q: _Quest) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    cr = q.code_review
    reviewer = str(cr.get("reviewer") or "a second model")
    final = bool(cr.get("final"))
    taken_out = {str(c.get("quantity")) for c in map(_dict, _list(q.result_check.get("checks"))) if c.get("quantity")}
    for p in _list(cr.get("problems")):
        p = _dict(p)
        if p.get("kind") in ("dependent variable", "metric") and str(p.get("name") or "") in taken_out:
            out.append(_problem("code", f"the plan's {p.get('kind')} '{_one(p.get('text'), 100)}' is "
                                f"{p.get('status') or 'missing'} in the code; the paper leaves it out",
                                f"{reviewer} (reads the code)", "left out", "needs/RESULT_CHECK.json"))
            continue
        if p.get("kind") == "logic":
            where = f" ({p['where']})" if p.get("where") else ""
            out.append(_problem("code", f"{_one(p.get('text'), 140)}{where} [{p.get('severity') or 'medium'}]",
                                f"{reviewer} (checks the code's logic)",
                                "not fixed" if final else "fixed", ".fi/code_review.json"))
            continue
        out.append(_problem("code", f"the plan's {p.get('kind') or 'requirement'} '{_one(p.get('text'), 100)}' is "
                            f"{p.get('status') or 'missing'} in the code", f"{reviewer} (reads the code)",
                            "not fixed" if final else "fixed", ".fi/code_review.json"))
    for item in _list(cr.get("not_reviewed")):
        out.append(_problem("code", f"not answered by the reader: {_one(item, 120)}", "the code review", "left out",
                            ".fi/code_review.json"))
    fs = q.function_steps
    for name, f in _dict(fs.get("functions")).items():
        f = _dict(f)
        attempts = _int(f.get("attempts"))
        eqn = f.get("equation") or ""
        if f.get("status") == "failed":
            out.append(_problem("code", f"function {name} (equation {eqn}) did not pass its checks after {attempts} "
                                "tries" + (f": {_one(f.get('error'), 120)}" if f.get("error") else ""),
                                "FI's equation test", "not fixed", ".fi/function_steps.json"))
        elif f.get("status") == "done" and attempts > 1:
            out.append(_problem("code", f"function {name} (equation {eqn}) needed {attempts} tries",
                                "FI's equation test", "fixed", ".fi/function_steps.json"))
    # A second, independently written version of a key function (core/second_impl.py): said in plain words, never a stop.
    for name, f in _dict(q.second_impl.get("functions")).items():
        f = _dict(f)
        if f.get("status") in ("unresolved", "unconfirmed") and f.get("limitation"):
            out.append(_problem("code", _one(f.get("limitation"), 220), "a second, independently written version",
                                "not fixed" if f.get("status") == "unresolved" else "left out", ".fi/second_impl.json"))
        elif _dict(f.get("adjudication")).get("outcome") == "repair accepted":
            out.append(_problem("code", f"function {name} differed from a second, independently written version; the first "
                                "was wrong and was repaired", "a second, independently written version", "fixed",
                                ".fi/second_impl.json"))
    return out


def _design_problems(q: _Quest) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    critic = _first_model(q.calls, ("design_review", "design_self_critique")) or "a second look"
    for it in q.critique:
        kind = _dict(it.get("review")).get("kind")
        label = "design review" if kind == "independent_review" else "design self-critique"
        for o in _list(it.get("objections")):
            o = _dict(o)
            out.append(_problem("design", o.get("objection") or o.get("check") or "", f"{critic} ({label})",
                                "fixed" if o.get("fix") else "not fixed", "needs/DESIGN_CRITIQUE.json"))
        if it.get("status") not in (None, "ok") and it.get("failure"):
            out.append(_problem("design", f"the design's self-critique could not be used: {_one(it['failure'], 120)}",
                                "FI", "left out", "needs/DESIGN_CRITIQUE.json"))
    for rec in (q.receipts.get("design_audit"),):
        if rec and rec.get("status") == "fail":
            detail = re.sub(r"^it objected to the design that ran:\s*", "", str(rec.get("detail") or ""))
            parts = [x.strip() for x in re.split(r";\s+", detail) if x.strip()] or [detail]
            for part in parts:
                out.append(_problem("design", part, "the design audit",
                                    "fixed" if q.design_acted else "not fixed", "needs/receipts/design_audit.json"))
    substituted = [str(x) for x in _list(q.drafted_asked.get("substituted"))]
    if substituted:
        names = {"grid": "settings to vary", "metrics": "metrics", "runs_per_setting": "runs per setting"}
        listed = ", ".join(names.get(x, x.replace("_", " ")) for x in substituted)
        out.append(_problem("design", f"the redesign's {listed} could not be read, so FI ran the plan's own instead; "
                                      "the results belong to the plan's settings, not to what the redesign meant",
                            "FI's design check", "left out", ".fi/drafted_parts_asked.json"))
    return out


def _plan_problems(q: _Quest) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    sb = q.search_asked
    if sb:
        out.append(_problem("plan", "the plan did not say how the search for the best design runs", "FI's plan check",
                            "fixed" if sb.get("outcome") == "kept" else "not fixed", ".fi/search_block_asked.json"))
    pp = q.protocol_asked
    if pp:
        parts = ", ".join(str(x) for x in _list(pp.get("parts"))) or "a part"
        out.append(_problem("plan", f"the plan left out its {parts}", "FI's plan check",
                            "fixed" if pp.get("outcome") in ("kept", "told in the plan") else "not fixed",
                            ".fi/protocol_part_asked.json"))
    if q.pause.get("kind") == "plan":
        out.append(_problem("plan", q.pause.get("headline") or "the plan cannot be run as written", "FI's plan check",
                            "not fixed", ".fi/pause.json"))
    return out


def _flagged_check_problems(q: _Quest) -> list[dict[str, str]]:
    """The known-answer checks that passed but that a second model's reading of the code flags as computed by flagged code:
    not shown, so they do not count as evidence (``core/code_review.py:flagged_checks``)."""
    from . import code_review as _cr
    from . import oracle_check as _oc

    names = [str(j["name"]) for j in _oc.last_judged(q.oracle_check)]
    return [_problem("run", f"the check {n!r} is {_oc.not_shown_words(text)}", "a second model reading the code", "not fixed",
                     ".fi/code_review.json")
            for n, text in _cr.flagged_checks(q.root, q.root / "code", names).items()]


def _run_problems(q: _Quest) -> list[dict[str, str]]:
    out = [*_oracle_problems(q), *_flagged_check_problems(q)]
    crashes = q.decisions("execute_reflect")
    retries = [d for d in crashes if str(d.get("chosen")) == "retry"]
    ended_ok = bool(crashes) and str(crashes[-1].get("chosen")) == "proceed"
    for i, d in enumerate(retries, 1):
        out.append(_problem("run", f"the script stopped with an error (try {i})", "the run itself",
                            "not fixed" if i == len(retries) and not ended_ok else "fixed",
                            f".fi/audit.jsonl seq {d.get('seq')}"))
    manifest = q.checks("run_manifest")
    # ``warned`` (engine.run_manifest_check: warn, the default) is recorded and not acted on: it is listed as not fixed.
    if any(e.get("status") in ("repairing", "stopped", "warned") for e in manifest):
        first = next(e for e in manifest if e.get("status") in ("repairing", "stopped", "warned"))
        last = manifest[-1].get("status")
        found_by = "FI's run-record check" + (" (recorded only)" if last == "warned" else "")
        out.append(_problem("run", _one((_list(first.get("problems")) or ["the run did not report what the protocol fixed"])[0], 200),
                            found_by, "fixed" if last in ("ok", "pass", "not_applicable") else "not fixed",
                            f".fi/audit.jsonl seq {first.get('seq')}"))
    # What the analysis did not print of trials FI holds whole is not a difference in the results: it is here, not in the paper.
    unprinted = _list(_dict(_json(q.root / "needs" / "RUN_MANIFEST_CHECK.json")).get("unprinted"))
    if unprinted:
        out.append(_problem("run", _one(f"the analysis prints no per-trial list for {len(unprinted)} metric(s) FI holds trial "
                                        f"by trial; first: {unprinted[0]}", 300),
                            "FI's run-record check (recorded only)", "not fixed", "needs/RUN_MANIFEST_CHECK.json"))
    pilot = q.pilot
    if _list(pilot.get("problems")):
        out.append(_problem("run", _one(str(_list(pilot["problems"])[0]), 200), "FI's pilot check",
                            "fixed" if pilot.get("outcome") == "passed" else "not fixed", ".fi/pilot_check.json"))
    out += _precision_not_judged(q)
    stop = _dict(_json(q.root / ".fi" / "run_manifest_stop.json"))
    if stop and not any(p["found_by"].startswith("FI's run-record check") for p in out):
        out.append(_problem("run", stop.get("headline") or stop.get("problem") or "the run did not report what the protocol fixed",
                            "FI's run-record check", "not fixed", ".fi/run_manifest_stop.json"))
    return out


def _precision_not_judged(q: _Quest) -> list[dict[str, str]]:
    """The plan's precision target is the half-width of a 95% interval from repeated runs of a setting. When each setting
    ran once (FI's record of the trials it ran: as many trials as settings), there is no interval and FI judged nothing
    against the target; without this line nothing said so."""
    from . import frozen_protocol as _frozen

    protocol = _frozen.protocol_of(q.root) or {}
    precision = _dict(protocol.get("precision"))
    target = precision.get("target_half_width")
    if not isinstance(target, (int, float)) or isinstance(target, bool):
        return []
    ran = _dict(_dict(_json(q.root / ".fi" / "run_facts.json")).get("trials_run"))
    trials, cells = ran.get("trials"), ran.get("cells")
    if not (isinstance(trials, int) and isinstance(cells, int) and 0 < trials <= cells):
        return []
    metric = str(precision.get("metric") or "").strip()
    on = f" on `{metric}`" if metric else ""
    return [_problem("run", f"the plan's precision target (a 95% interval of ±{target:g}{on}) was not judged: each setting "
                            f"ran once ({trials} trials for {cells} settings), so there is no interval to hold to it",
                     "FI's precision check", "left out", ".fi/run_facts.json")]


def _pilot_summary(q: _Quest) -> dict[str, Any] | None:
    """What the pilot before the full run showed and how many repairs it caused, or ``None`` when no pilot ran."""
    p = q.pilot
    if not p or not p.get("outcome"):
        return None
    repairs = _int(p.get("repairs"))
    trials, settings = _int(p.get("trials")), _int(p.get("settings"))
    read = f"{trials} trial(s) at {settings} setting(s) were read before the full run"
    if p["outcome"] == "passed":
        said = f"{read}; nothing was wrong" + (" once the simulation was repaired" if repairs else "")
    elif p["outcome"] == "stopped":
        said = f"{read}; they showed the simulation does not measure what the protocol names, and its repairs did not remove it"
    else:
        said = f"{read}; they showed the simulation does not measure what the protocol names, and it was sent back to be repaired"
    return {"outcome": p["outcome"], "trials": trials, "settings": settings, "repairs": repairs,
            "problems": [str(x) for x in _list(p.get("problems"))], "says": said}


def _trial_checks_summary(q: _Quest) -> dict[str, Any] | None:
    """FI's own checks of what the trials gave (core/trial_checks.py): the settings with failed trials, those left out, each
    reported interval or standard error judged against the true values, and what could not be checked. ``None`` when the
    checks did not run (no trials of FI's own)."""
    rec = q.trial_checks
    if not rec:
        return None
    if rec.get("not_run"):
        from . import trial_checks as _tc

        why = _tc.NOT_RUN.get(str(rec["not_run"]), str(rec["not_run"]))
        return {"action": "", "left_out": [], "checked": False, "faults": 0, "not_run": True,
                "lines": [f"FI's checks of the trials did not run: {why}"]}
    fails =[f for f in _list(rec.get("failures")) if isinstance(f, dict)]
    left = [str(k) for k in _list(rec.get("left_out_settings"))]
    cal = _dict(rec.get("calibration"))
    quantities = [x for x in _list(cal.get("quantities")) if isinstance(x, dict)]
    lines = []
    if fails:
        lines.append(f"{len(fails)} of {_int(rec.get('settings'))} setting(s) had failed trials"
                     + (f"; {len(left)} left out of the results because most of their trials failed ({', '.join(left[:6])})"
                        if left else ""))
        lines += [f"{f.get('key') or 'the one setting'}: {f.get('failed')} of {f.get('planned')} failed"
                  + (f" ({_one(_list(f.get('reasons'))[0][0], 160)})" if _list(f.get("reasons"))
                     and isinstance(_list(f.get("reasons"))[0], list) else "") for f in fails[:8]]
    else:
        lines.append(f"no trial failed in the {_int(rec.get('settings'))} setting(s)")
    for x in quantities:
        state = (("not the multiplier the plan fixes (a fault)" if x.get("kind") == "multiplier" else "grossly wrong (a fault)")
                 if x.get("fault") else
                 f"below the level in {_int(x.get('below'))} setting(s): reported, not a fault" if _int(x.get("below"))
                 else f"above the level in {_int(x.get('above'))} setting(s) (wider than need be): reported, not a fault"
                 if _int(x.get("above")) else "as it should be")
        lines.append(f"{_one(x.get('says'), 300)} - {state}")
    from . import trial_checks as _tc

    for u in [u for u in _list(rec.get("undefined")) if isinstance(u, dict)][:8]:
        lines.append(_one(_tc.said_undefined(u), 300) + ("" if u.get("all") else " (not a failed trial; the paper says so)"))
    not_checked = [str(n) for n in _list(cal.get("not_checked"))]
    if not cal.get("checked"):
        lines.append("calibration not checked: " + _one("; ".join(not_checked), 400))
    elif not_checked:
        lines += [f"not checked: {_one(n, 200)}" for n in not_checked[:4]]
    return {"action": str(rec.get("action") or ""), "left_out": left, "checked": bool(cal.get("checked")),
            "faults": sum(1 for x in quantities if x.get("fault")), "lines": lines}


def _analysis_problems(q: _Quest) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for e in q.checks("numeric_warnings"):
        if e.get("status") == "block":
            out.append(_problem("analysis", _one((_list(e.get("problems")) or [e.get("summary")])[0], 200),
                                "FI's number check", "not fixed" if q.stuck else "fixed", f".fi/audit.jsonl seq {e.get('seq')}"))
            break
    if q.stuck and _stuck_stage(q) == "analysis":
        out.append(_problem("analysis", q.stuck.get("problem") or "the analysis could not be read", "FI", "not fixed",
                            "needs/STUCK.json"))
    return out


def _paper_problems(q: _Quest) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    claim_model = _first_model(q.calls, ("claim_check",)) or "claim check"
    rec = q.receipts.get("claim_check")
    if rec and rec.get("status") == "fail":
        out.append(_problem("paper", rec.get("detail") or "claims in the draft are unsupported",
                            f"{claim_model} (claim check)",
                            "not fixed" if not _paper_exists(q.root) else "left out", "needs/receipts/claim_check.json"))
    for r in _list(q.paper_check.get("removed_numbers")):
        r = _dict(r)
        out.append(_problem("paper", f"the number {r.get('number')} was typed by the writer, not filled by FI from its records, "
                            f"and was taken out: {_one(r.get('sentence'), 200)}",
                            "FI's check of the paper's numbers", "left out", ".fi/paper_record_check.json"))
    for r in _list(q.paper_check.get("taken_out_values")):
        r = _dict(r)
        out.append(_problem("paper", f"FI could not fill the value {r.get('key')} from its records ({r.get('how')}): "
                            f"{_one(r.get('sentence'), 200)}",
                            "FI's check of the paper's numbers", "left out", ".fi/paper_record_check.json"))
    if q.stuck and _stuck_stage(q) == "paper":
        out.append(_problem("paper", q.stuck.get("problem") or "the paper contradicts FI's records",
                            "FI's records", "not fixed", "needs/STUCK.json"))
    pdf = _paper_pdf(q)
    if pdf and not pdf["typeset"] and _paper_exists(q.root):
        out.append(_problem("paper", "paper.pdf is not typeset (a browser drew it): " + _one(pdf["reason"], 200),
                            "FI's PDF render", "not fixed", ".fi/paper_render.json"))
    return out


def _paper_pdf(q: _Quest) -> dict[str, Any] | None:
    """How ``paper.pdf`` was made (``.fi/paper_render.json``, written by the paper generator): typeset by LaTeX or
    drawn by the browser, every character set in another form or font than the text's, and any no font here could
    set. ``None`` when there is no record."""
    try:
        from generation.paper import read_render_record
        rec = _dict(read_render_record(q.root))
    except Exception:  # noqa: BLE001 — the report never fails over one record
        return None
    if not rec:
        return None
    return {"typeset": bool(rec.get("typeset")), "engine": str(rec.get("engine") or ""),
            "reason": str(rec.get("reason") or ""), "pages": rec.get("pages"), "format": rec.get("format"),
            "glyphs_replaced": [_dict(g) for g in _list(rec.get("glyphs_replaced"))],
            "glyphs_not_set": [_dict(g) for g in _list(rec.get("glyphs_not_set"))]}


def _citations(q: _Quest) -> dict[str, Any] | None:
    """The citation check's ledger of the delivered paper (``paper/citations.json``): every citation with the passage that
    backs it or what was done instead, and the counts. ``None`` when the check did not run."""
    data = _json(q.root / "paper" / "citations.json")
    return data if isinstance(data, dict) and isinstance(data.get("citations"), list) else None


def _citation_problems(q: _Quest) -> list[dict[str, str]]:
    """Each citation the retrieved text did not support: found by the citation reader, fixed by FI."""
    data = _citations(q)
    if not data:
        return []
    reader = str(data.get("reader") or "") or "the citation reader"
    out: list[dict[str, str]] = []
    for r in map(_dict, _list(data.get("citations"))):
        if r.get("status") == "supported" or str(r.get("action") or "").startswith("added"):
            continue
        fixed = r.get("status") != "unchecked"
        out.append(_problem("paper", f"[{r.get('label')}] {_one(r.get('sentence'), 90)}: {_one(r.get('problem'), 140)}",
                            f"{reader} (citation check)", "fixed" if fixed else "not fixed", "paper/CITATIONS.md"))
    return out


def _review_problems(q: _Quest) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    reviewer = _first_model(q.calls, ("review",)) or "the review"
    ds = q.decisions("review")
    for i, d in enumerate(ds):
        facts = _dict(d.get("facts"))
        if str(facts.get("verdict")) not in ("revise", "reject"):
            continue
        hits = {str(h) for h in _list(facts.get("must_flag_hits"))}
        if i + 1 < len(ds):
            nxt = _dict(ds[i + 1].get("facts"))
            nxt_hits = {str(h) for h in _list(nxt.get("must_flag_hits"))}
            fixed = str(nxt.get("verdict")) == "accept" or bool(hits - nxt_hits)
        else:
            fixed = False  # the last look still asked for a revision (the rounds allowed were spent)
        out.append(_problem("review", "the review asked for a revision" + (f": {', '.join(sorted(hits))}" if hits else "")
                            + ("" if i + 1 < len(ds) else " (the rounds allowed were spent; delivered as it was)"),
                            f"{reviewer} (review)", "fixed" if fixed else "not fixed", f".fi/audit.jsonl seq {d.get('seq')}"))
    return out


def _call_problems(q: _Quest) -> list[dict[str, str]]:
    """Model calls that got no usable answer (a timeout, a refusal by the provider), per step: fixed when the step
    got an answer afterwards."""
    out: list[dict[str, str]] = []
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in q.calls:
        by_node[str(r.get("node", "")).split(".")[0]].append(r)
    for node, rows in by_node.items():
        stage = _stage_of(node)
        bad = [r for r in rows if r.get("outcome") not in (None, "ok")]
        if not stage or not bad:
            continue
        kinds = ", ".join(f"{k} x{n}" for k, n in Counter(str(r["outcome"]) for r in bad).most_common())
        out.append(_problem(stage, f"the model did not answer the step '{node}': {kinds}", "the provider",
                            "fixed" if rows[-1].get("outcome") in (None, "ok") else "not fixed", ".fi/model_calls.jsonl"))
    return out


def _first_model(calls: list[dict[str, Any]], nodes: tuple[str, ...]) -> str:
    c = Counter(_model_name(r) for r in calls if str(r.get("node", "")).split(".")[0] in nodes and _model_name(r))
    return c.most_common(1)[0][0] if c else ""


def _paper_exists(root: Path) -> bool:
    return (root / "paper" / "paper.md").is_file() or (root / "paper.md").is_file()


def _stuck_stage(q: _Quest) -> str:
    text = str(q.stuck.get("problem") or "")
    if "contradict" in text or "claims" in text:
        return "paper"
    if "main quantity" in text or "typed in" in text or "shown by a failing test" in text or "text still could not be checked" in text:
        return "code"
    if "no number" in text or "could not run" in text or "the same result" in text or "time allowed" in text \
            or "nothing was run" in text:
        return "run"
    return "analysis"


def _durations(audit: list[dict[str, Any]]) -> dict[str, float]:
    """Seconds spent per stage, from each step's start to its end (or the next step's start) in the audit trace."""
    totals: dict[str, float] = defaultdict(float)
    open_at: dict[str, float] = {}
    for e in audit:
        node, t = str(e.get("node") or ""), _epoch(e.get("ts"))
        if t is None:
            continue
        kind = e.get("kind")
        if kind == "node_started":
            open_at[node] = t
        elif kind in ("node_completed", "node_paused", "node_failed") and node in open_at:
            stage = _stage_of(node)
            if stage:
                totals[stage] += max(0.0, t - open_at.pop(node))
    return dict(totals)


# ------------------------------------------------------------------------------------------------------ report

def _stage_models(q: _Quest, stage: str) -> tuple[list[str], str, bool | None, str]:
    """(writers, reviewer, same model?, note) for one stage."""
    writers = Counter(_who(r) for r in q.calls if str(r.get("node", "")).split(".")[0] in _WRITERS.get(stage, ()))
    wrote = [w for w, _ in writers.most_common()]
    reader = ""
    same: bool | None = None
    note = ""
    if stage == "checks":
        orv = q.oracle_review
        reader = str(orv.get("reviewer") or "")
        if not wrote and orv.get("planner"):
            wrote = [str(orv["planner"])]
        if orv.get("same_model") is not None:
            same = bool(orv.get("same_model"))
    elif stage == "code":
        cr = q.code_review
        reader = str(cr.get("reviewer") or "")
        skipped = str(cr.get("skipped") or "")
        if skipped and not reader:
            m = re.search(r"\(([^()]+)\)\s*$", skipped)
            if "the reader is the model that wrote" in skipped:
                same, reader = True, (m.group(1) if m else "")
            note = skipped
        elif cr.get("same_model") is not None:
            same = bool(cr.get("same_model"))
        if cr.get("skipped") and not note:
            note = str(cr["skipped"])
    elif stage == "run":
        reader = "the engine's own checks (no model)"
    elif stage in _READERS:
        readers = Counter(_who(r) for r in q.calls if _reads(stage, r.get("node")))
        reader = readers.most_common(1)[0][0] if readers else ""
        if reader and wrote:
            same = _bare(reader) == _bare(wrote[0])
    if stage in ("design", "analysis", "paper") and reader and wrote and same is None:
        same = _bare(reader) == _bare(wrote[0])
    if stage == "design" and q.critique:
        # What the design audit recorded about who read the design against who wrote it (an independent review, or the
        # writer's own self-review) is what the report says; the call log only fills in when it recorded nothing.
        kind = _dict(q.critique[-1].get("review")).get("kind")
        if kind in ("independent_review", "self_review"):
            same = kind == "self_review"
    return wrote, reader, same, note


def _review_cost(q: _Quest, stage: str) -> dict[str, Any] | None:
    """What a stage's reading cost: the calls its reviewer made (``None`` for a stage no model reads), their tokens, and
    their price when the cost record has one (``cost.summary.json``, per step). A person reads this beside who the reader
    was to see which readings are worth a stronger model."""
    if stage not in _READERS:
        return None
    rows = [r for r in q.calls if _reads(stage, r.get("node"))]
    if stage == "plan" and not rows:
        return None  # the plan is read only by the reading of the topic, which a quest may not have made
    usd: float | None = None
    for node, bucket in _dict(q.cost.get("by_node")).items():
        price = _dict(bucket).get("cost_usd")
        if _reads(stage, node) and isinstance(price, (int, float)):
            usd = (usd or 0.0) + float(price)
    return {"calls": len(rows), "failed_calls": sum(1 for r in rows if r.get("outcome") not in (None, "ok")),
            "prompt_tokens": sum(_int(_dict(r.get("usage")).get("prompt_tokens")) for r in rows),
            "completion_tokens": sum(_int(_dict(r.get("usage")).get("completion_tokens")) for r in rows),
            "usd": round(usd, 4) if usd is not None else None}


def _bare(who: str) -> str:
    # The provider is dropped on purpose: one model reached through two providers is still one model, and a reading by it
    # is not an independent one. FI's own check does the same (``oracle_review.same_model`` ignores the provider), so this
    # report must agree with it.
    return who.split("/", 1)[-1]


def _evidence_line(q: _Quest) -> str:
    record = _dict(q.summary.get("evidence")) or q.evidence
    if not record:
        return ""
    try:
        from . import evidence as _evidence
        return _evidence.summary_line(record)
    except Exception:  # noqa: BLE001 -- an old record leaves the plain status
        return str(record.get("status") or "")


def _outcome(q: _Quest) -> dict[str, Any]:
    root = q.root
    marks = {"stop": _mtime(root / "needs" / "STUCK.json") if q.stuck else 0.0,
             "crash": _mtime(root / "quest_failed.md") if q.failed_md else 0.0,
             "pause": _mtime(root / ".fi" / "pause.json") if q.pause else 0.0,
             "paper": _mtime(root / "paper" / "paper.md") or _mtime(root / "paper.md")}
    kind = max(marks, key=lambda k: marks[k]) if any(marks.values()) else ""
    ev = _evidence_line(q)
    out: dict[str, Any] = {"kind": "unknown", "line": "No end record was found: the quest may still be running, or it "
                           "stopped before FI could write one.", "evidence": ev, "needs_person": False}
    if kind == "paper":
        out.update(kind="finished", line="A paper was delivered.")
    elif kind == "stop":
        out.update(kind="honest_stop", line="FI stopped without a paper, on purpose: "
                   + _one(q.stuck.get("problem") or q.stuck.get("say"), 300) + ".")
    elif kind == "crash":
        what = re.search(r"\*\*What happened:\*\*\s*(.+)", q.failed_md)
        node = re.search(r"\*\*Failing node:\*\*\s*`?([A-Za-z0-9_.]+)", q.failed_md)
        out.update(kind="crashed", line="The quest stopped on an error: "
                   + (_one(what.group(1), 300) if what else "see quest_failed.md.")
                   + (f" (step: {node.group(1)})" if node else ""), failing_node=node.group(1) if node else "")
    elif kind == "pause":
        decide = _one(q.pause.get("decide"), 200)
        out.update(kind="paused", line=f"Paused: {_one(q.pause.get('headline'), 200)}."
                   + (f" It waits for you: {decide}" if decide else ""), needs_person=True)
    return out


def _failure(q: _Quest, outcome: dict[str, Any], problems: list[dict[str, str]]) -> dict[str, Any] | None:
    kind = outcome["kind"]
    stage = ""
    if kind == "honest_stop":
        stage = _stuck_stage(q)
    elif kind == "crashed":
        stage = _stage_of(outcome.get("failing_node", "")) or ""
    elif kind == "paused":
        stage = {"plan": "plan", "oracle": "run", "manifest": "run", "model_output": _stage_of(
            str(_last_node(q, "node_paused"))) or "plan", "review": "review"}.get(str(q.pause.get("kind")), "")
        stage = stage or _stage_of(_last_node(q, "node_paused")) or ""
    else:
        for wanted in ("not fixed", "left out"):
            bad = sorted((p for p in problems if p["status"] == wanted), key=lambda p: STAGES.index(p["stage"]))
            if bad:
                stage = bad[0]["stage"]
                break
    if not stage:
        return None
    chain = [f"{p['found_by']}: {p['text']} -> {p['status']}" for p in problems if p["stage"] == stage]
    if kind in ("honest_stop", "crashed", "paused"):
        chain.append(outcome["line"])
    if kind == "honest_stop":
        chain = [*[f"FI tried: {_one(t, 200)}" for t in _list(q.stuck.get("tried"))], *chain]
    return {"stage": stage, "chain": chain}


def _last_node(q: _Quest, kind: str) -> str:
    for e in reversed(q.audit):
        if e.get("kind") == kind and e.get("node"):
            return str(e["node"])
    return ""


def _suggestions(q: _Quest, stages: dict[str, dict[str, Any]], outcome: dict[str, Any],
                 failure: dict[str, Any] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    def add(text: str, source: str) -> None:
        if not any(s["text"] == text for s in out):
            out.append({"text": text, "source": source})

    # What the record itself says to do comes first.
    if outcome["kind"] == "crashed":
        m = re.search(r"\*\*What to do:\*\*\s*(.+)", q.failed_md)
        if m and _is_choice(m.group(1)):
            add(_one(m.group(1), 300), "quest_failed.md")
    if outcome["kind"] == "paused" and _is_choice(str(q.pause.get("recommended") or "")):
        add(_one(q.pause["recommended"], 300), ".fi/pause.json")
    # A step that timed out or was limited, again and again.
    errors: dict[tuple[str, str], int] = Counter()
    for r in q.calls:
        if r.get("outcome") not in (None, "ok"):
            errors[(str(r.get("node", "")).split(".")[0], str(r["outcome"]))] += 1
    for (node, what), n in sorted(errors.items(), key=lambda kv: -kv[1]):
        if n < 2:
            continue
        if what == "ReadTimeout":
            add(f"The step '{node}' timed out {n} times: raise provider.node_http_timeout_s for it, or give it a faster "
                f"model (provider.node_models.{node}).", ".fi/model_calls.jsonl")
        elif what == "HTTPStatusError":
            add(f"The step '{node}' was refused by the provider {n} times (an HTTP error, such as a rate limit): wait, or "
                f"route that step to another provider (provider.node_providers.{node}).", ".fi/model_calls.jsonl")
        else:
            add(f"The step '{node}' ended with '{what}' {n} times: try another model for it "
                f"(provider.node_models.{node}).", ".fi/model_calls.jsonl")
    # A stage read by the model that wrote it.
    order = [failure["stage"]] if failure else []
    order += [s for s in STAGES if s not in order]
    for stage in order:
        row = stages[stage]
        if row["same_model"] and stage in _READERS:
            reader_key = _pairs.READER_STEP.get(stage) or _READERS[stage][0]
            if row["problems"]["total"] or stage == (failure or {}).get("stage") or stage == "code":
                add(f"The {STAGE_NAME[stage].lower()} stage was read by the model that wrote it "
                    f"({row['reviewed_by'] or 'the same model'}): give the reading step another provider's model "
                    f"(provider.node_providers.{reader_key}: name, model, base_url, api_key_env)"
                    + (" - until it is set, the code is read by `oracle_review`, the step that reads the plan's checks"
                       if stage == "code" else "") + ".",
                    {"checks": ".fi/oracle_review.json", "code": ".fi/code_review.json"}.get(stage, ".fi/model_calls.jsonl"))
    # The stage with the most problems left.
    left = {s: stages[s]["problems"]["not_fixed"] + stages[s]["problems"]["left_out"] for s in STAGES}
    worst = max(left, key=lambda s: left[s])
    if left[worst] >= 2 and worst in ("code", "design", "plan", "run"):
        who = f" (now {stages[worst]['wrote'][0]})" if stages[worst]["wrote"] else ""
        add(f"Most problems left are in the {STAGE_NAME[worst].lower()}: try a stronger model for it{who} "
            f"(provider.node_models.{_KEY[worst]}, or another provider with provider.node_providers.{_KEY[worst]}).", "QUALITY: problems per stage")
    # A check that kept failing for a reason the record names.
    repeated = [p for p in _oracle_problems(q) if "failed " in p["text"] and "times" in p["text"] and p["status"] != "fixed"]
    for p in repeated[:2]:
        add(f"A known-answer check kept failing ({_one(p['text'], 170)}). Either the check or the code is wrong; "
            f"try another model for the code (provider.node_models.implement) or the plan (provider.node_models.plan).",
            p["source"])
    # A pause or an honest stop with no record advice: the stage's own model.
    if outcome["kind"] == "honest_stop" and not any("provider.node_models" in s["text"] for s in out):
        stage = failure["stage"] if failure else "analysis"
        add(f"Run it again with another model for the {STAGE_NAME[stage].lower()} (provider.node_models.{_KEY[stage]}).",
            "needs/STUCK.json")
    return out


def _cost(q: _Quest) -> dict[str, Any]:
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "failed_calls": 0})
    for r in q.calls:
        row = by_model[_who(r)]
        u = _dict(r.get("usage"))
        row["calls"] += 1
        row["prompt_tokens"] += _int(u.get("prompt_tokens"))
        row["completion_tokens"] += _int(u.get("completion_tokens"))
        row["failed_calls"] += 0 if r.get("outcome") in (None, "ok") else 1
    if not by_model and _dict(q.cost.get("by_node")):
        by_model["(not recorded per model)"] = {"calls": _int(q.cost.get("total_requests")),
                                                "prompt_tokens": _int(q.cost.get("total_prompt_tokens")),
                                                "completion_tokens": _int(q.cost.get("total_completion_tokens")),
                                                "failed_calls": 0}
    usd = q.cost.get("total_cost_usd")
    return {"by_model": dict(by_model),
            "calls": sum(m["calls"] for m in by_model.values()),
            "prompt_tokens": sum(m["prompt_tokens"] for m in by_model.values()),
            "completion_tokens": sum(m["completion_tokens"] for m in by_model.values()),
            "usd": usd if isinstance(usd, (int, float)) else None}


def _config(q: _Quest, stages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prov = _dict(q.config.get("provider"))
    c = Counter(_who(r) for r in q.calls)
    main = str(prov.get("model") or "")
    if main and prov.get("name"):
        main = f"{prov['name']}/{main}"
    main = main or (c.most_common(1)[0][0] if c else "")
    routed: dict[str, str] = {}
    for node, model in _dict(prov.get("node_models")).items():
        routed[str(node)] = str(model)
    if not routed:
        per_node: dict[str, Counter] = defaultdict(Counter)
        for r in q.calls:
            per_node[str(r.get("node", "")).split(".")[0]][_who(r)] += 1
        for node, cnt in per_node.items():
            who = cnt.most_common(1)[0][0]
            if who != main and node in {n for names in (*_WRITERS.values(), *_READERS.values()) for n in names}:
                routed[node] = who
    return {"main_model": main, "routed": dict(sorted(routed.items()))}


def _run_started(q: _Quest) -> float | None:
    """When the current run began: the latest ``quest_started`` in the audit trace (a resume writes one too)."""
    for e in reversed(q.audit):
        if e.get("kind") == "quest_started":
            return _epoch(e.get("ts"))
    return None


def _stamp(t: float | None) -> str | None:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds") if t else None


def _day(iso: Any) -> str:
    return str(iso or "an unknown time").replace("T", " ")[:16]


def _code_metrics(q: _Quest) -> dict[str, Any]:
    """What the code step measured about itself: how many functions of the model's package were written and passed their
    check (``.fi/function_steps.json``), and what the second model's reading of the code came to (``.fi/code_review.json``):
    whether the final version was read, the open findings in each reading and whether the repairs converged."""
    functions = [f for f in _dict(q.function_steps.get("functions")).values() if isinstance(f, dict)]
    cr = q.code_review
    readings = [r for r in _list(cr.get("readings")) if isinstance(r, dict)]
    return {"functions_done": sum(1 for f in functions if f.get("status") == "done"), "functions_total": len(functions),
            "final_version_read": bool(cr.get("final_read")) if (readings or cr.get("read_version") or cr.get("skipped")
                                                                  or cr.get("error")) else None,
            "open_per_reading": [_int(r.get("open")) for r in readings],
            "open_high_per_reading": [_int(r.get("high")) for r in readings],
            "repairs_converged": cr.get("converged") if isinstance(cr.get("converged"), bool) else None,
            "repair_requests": _int(cr.get("repair_calls")), "repair_rounds": _int(cr.get("send_backs")),
            "repairs_stopped": str(cr.get("repairs_stopped") or ""),
            "issues": _issue_rows(cr, getattr(q, "root", None)), "issue_summary": _issue_summary(cr),
            "open_changing_per_reading": [_int(r.get("changing", r.get("high"))) for r in readings]}


def _issue_rows(cr: dict[str, Any], root: Path | None = None) -> list[dict[str, Any]]:
    """One row per issue of the code reader (a stable id; :mod:`core.code_findings`): what it concerns, whether it can change
    a reported result, whether a failing test showed it, whether and by which model it was repaired, and whether it is
    still open after the last reading."""
    from . import code_findings as _issues
    from . import code_reach as _code_reach

    never_run = _code_reach.never_run_in(root) if root is not None else (lambda _f, _g: False)
    rows = []
    for issue in _issues.ranked(cr, only_open=False):
        outcome = str(issue.get("outcome") or "")
        readings = [r for r in _list(issue.get("readings")) if isinstance(r, dict)]
        still = bool(readings) and readings[-1].get("status") == "open" and outcome != _issues.FIXED
        rows.append({
            "id": str(issue.get("id") or ""), "what": _one(issue.get("text"), 110),
            "where": " ".join(x for x in (str(issue.get("file") or ""), str(issue.get("function") or "")) if x),
            "changes_result": bool(issue.get("changes_result")),
            "demonstrated": issue.get("demonstrated") if isinstance(issue.get("demonstrated"), bool) else None,
            "repaired": outcome == _issues.FIXED, "repaired_by": str(issue.get("fixed_by") or ""),
            "open": still, "outcome": outcome, "plan_problem": bool(issue.get("plan_conflict")) or outcome == _issues.PLAN_PROBLEM,
            "reason": _one(issue.get("reason"), 140), "seen_in": len(readings),
            "reported_as": [str(k) for k in _list(issue.get("reported_as"))],
            "at_gate": issue.get("at_gate") if isinstance(issue.get("at_gate"), dict) else {},
            "never_run": bool(issue.get("never_run")) or never_run(issue.get("file"), issue.get("function"))})
    return rows


def _issue_summary(cr: dict[str, Any]) -> dict[str, Any]:
    from . import code_findings as _issues

    summary = _issues.summary(cr)
    return {**summary, "line": _issues.convergence_line(cr)}


def _left_out_of_paper(q: _Quest) -> list[dict[str, Any]]:
    """The quantities FI took out of the results the paper may use, and why (``needs/RESULT_CHECK.json``): one entry per
    quantity, with the names it was found under in the results."""
    out: list[dict[str, Any]] = []
    record = q.result_check
    named = {str(k) for c in map(_dict, _list(record.get("checks"))) for k in _list(c.get("keys"))}
    for c in map(_dict, _list(record.get("checks"))):
        if c.get("quantity"):
            out.append({"quantity": str(c["quantity"]), "why": _one(c.get("note"), 300),
                        "names_in_results": [str(k) for k in _list(c.get("keys"))]})
    rest = [str(k) for k in _list(record.get("removed_quantities")) if str(k) not in named]
    if rest:
        out.append({"quantity": ", ".join(rest[:6]), "why": _one(record.get("note"), 300), "names_in_results": rest})
    return out


def _paper_records(q: _Quest) -> dict[str, Any]:
    """What FI recorded about the run that the paper itself does not carry (``core/paper_facts.py``): how the result was
    reached (design revisions, runs), the run record (seed, measured run time, trial counts, a reduced size, the topic's
    requirements not done, reviewer objections), the open points in the code, and what FI's check of the paper against these
    records found the last time it ran. Each part is empty when its record says nothing."""
    from . import code_review as _cr
    from . import disclosure as _d
    from . import paper_facts as _pf

    root = q.root
    engine = _dict(q.config.get("engine"))
    topic = str(q.config.get("topic") or "")
    out: dict[str, Any] = {"how_reached": "", "run_record": "", "code_findings": "", "paper_check": {}}
    # The study as the quest resolved it (clarify can decide a quest needs no simulation, or is a survey, when the YAML did
    # not say so): the last design route records both. A quest with no such record falls back to the YAML.
    resolved = next((_dict(e.get("facts")) for e in reversed(q.decisions("design")) if _dict(e.get("facts"))), {})
    no_sim = bool(resolved.get("no_simulation", engine.get("no_simulation")))
    survey = bool(resolved.get("survey_mode", engine.get("survey_mode")))
    try:
        out["how_reached"] = _d.paragraph(root, no_simulation=no_sim, survey=survey).replace(_d.LEAD + " ", "")
        wall, least = _pf.measured_run_time(root, None)
        out["run_record"] = _d.run_record(root, topic=topic, duration_s=wall).replace(_d._RECORD_LEAD + " ", "")
        if wall is None and least is not None:
            out["run_record"] += (f" FI did not record how long the whole experiment took; the trials alone took at least "
                                  f"{_d._plain_seconds(least)}.")
        out["code_findings"] = _cr.findings_block(root, root / "code").replace(
            "**Open points in the code that produced these results.** ", "")
        kept = _pf.kept_check(root)
        out["paper_check"] = {"hits": [str(h) for h in _list(kept.get("hits"))],
                              "unmet_items": [_dict(i) for i in _list(kept.get("unmet_items"))],
                              "unfilled": [str(u) for u in _list(kept.get("unfilled"))],
                              "removed": [_dict(r) for r in _list(kept.get("removed_numbers"))],
                              "taken_out": [_dict(r) for r in _list(kept.get("taken_out_values"))],
                              "not_filled": str(kept.get("not_filled_by_fi") or "")}
    except Exception:  # noqa: BLE001 -- a record that cannot be read leaves its part out
        pass
    return out


def build(quest_root: Path | str, *, running: bool | None = None) -> dict[str, Any]:
    """The report of one quest, from its records. Never raises for a missing or old-format record.

    ``running`` says whether the quest is running now; ``None`` finds out the way the quest list does
    (``quest_title.looks_running``). A running quest's outcome is "running", never an earlier run's stop or paper."""
    root = Path(quest_root)
    q = _Quest(root)
    if running is None:
        try:
            from . import quest_title
            running = quest_title.looks_running(root)
        except Exception:  # noqa: BLE001
            running = False
    started = _run_started(q)
    problems: list[dict[str, str]] = [*_plan_problems(q), *_checks_problems(q), *_design_problems(q), *_code_problems(q),
                                      *_run_problems(q), *_analysis_problems(q), *_paper_problems(q), *_citation_problems(q),
                                      *_review_problems(q),
                                      *_call_problems(q)]
    seconds = _durations(q.audit)
    stages: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        wrote, reader, same, note = _stage_models(q, stage)
        calls = [r for r in q.calls if _stage_of(str(r.get("node", ""))) == stage]
        mine = [p for p in problems if p["stage"] == stage]
        count = Counter(p["status"].replace(" ", "_") for p in mine)
        stages[stage] = {
            "stage": stage, "wrote": wrote, "reviewed_by": reader, "same_model": same, "review_note": note,
            "problems": {"total": len(mine), "fixed": count["fixed"], "not_fixed": count["not_fixed"],
                         "left_out": count["left_out"], "items": mine},
            "calls": len(calls), "failed_calls": sum(1 for r in calls if r.get("outcome") not in (None, "ok")),
            "prompt_tokens": sum(_int(_dict(r.get("usage")).get("prompt_tokens")) for r in calls),
            "completion_tokens": sum(_int(_dict(r.get("usage")).get("completion_tokens")) for r in calls),
            "seconds": round(seconds[stage]) if stage in seconds else None,
            "review_cost": _review_cost(q, stage),
        }
    outcome = _outcome(q)
    if running:
        previous = _dict(_json(root / ".fi" / "quality.json")).get("written_at")
        outcome = {"kind": "running", "needs_person": False, "evidence": "",
                   "line": "The quest is running now" + (f" (this run started {_day(_stamp(started))})" if started else "")
                   + ". This report is built from the records so far; the outcome is not known yet."
                   + (f" The QUALITY.md in the quest folder is from the previous run of {_day(previous)}." if previous else "")}
    failure = None if running else _failure(q, outcome, problems)
    totals = Counter(p["status"].replace(" ", "_") for p in problems)
    total = len(problems)
    quest_id = str(q.summary.get("quest_id") or (q.audit[0].get("quest_id") if q.audit else "") or root.name)
    return {
        "schema": SCHEMA, "written_at": _stamp(time.time()), "run_started_at": _stamp(started), "running": bool(running),
        "quest_id": quest_id, "title": str(q.summary.get("title") or q.config.get("title") or ""),
        "outcome": outcome, "stages": [stages[s] for s in STAGES], "failure": failure, "code": _code_metrics(q),
        "pilot": _pilot_summary(q),
        "trial_checks": _trial_checks_summary(q),
        "citations": _citations(q),
        "paper_pdf": _paper_pdf(q),
        "left_out_of_paper": _left_out_of_paper(q),
        "paper_records": _paper_records(q),
        "suggestions": _suggestions(q, stages, outcome, failure), "cost": _cost(q), "config": _config(q, stages),
        "problems": {"total": total, "fixed": totals["fixed"], "not_fixed": totals["not_fixed"],
                     "left_out": totals["left_out"],
                     "fixed_rate": round(totals["fixed"] / total, 3) if total else None},
        "seconds": sum(seconds.values()) if seconds else None,
    }


# ------------------------------------------------------------------------------------------------- rendering

def _n(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def _span(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def _issue_lines(code: dict[str, Any]) -> list[str]:
    """The code reader's findings by stable id: shown by a test or not, repaired and by which model, still open and whether
    it changes a result. The convergence statement counts the issues a test showed to change a result, not findings per
    reading (the same issue comes back reworded)."""
    summary = code.get("issue_summary") or {}
    lines = ["", "### Findings of the code reader, by id", ""]
    if summary.get("line"):
        lines += [summary["line"] + ".", ""]
    models = summary.get("fixed_by") or {}
    if models:
        lines += ["Repaired by: " + ", ".join(f"{m} ({n})" for m, n in sorted(models.items())) + ".", ""]
    lines += ["| id | where | what | changes a result | shown by a failing test | repaired | still open |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in code["issues"][:40]:
        if row["demonstrated"] is False:
            shown = ("no: the test passed on the code, so it was not repaired" if "passes on the current code" in row["reason"]
                     else "no usable test could be written, so it was not repaired")
        else:
            shown = "yes" if row["demonstrated"] else "not tested"
        repaired = f"yes, by {row['repaired_by']}" if row["repaired"] else ("no" + (f" ({row['reason']})" if row["reason"] else ""))
        if row["plan_problem"]:
            still = "plan problem: the plan contradicts itself, asked of the plan's model"
        elif row["open"] and row.get("never_run"):
            still = "yes, but the run never calls this code, so it changes nothing reported; it is kept here, not in the paper"
        elif row["open"] and row.get("at_gate", {}).get("outcome") == "left_out":
            keys = ", ".join(str(k) for k in _list(row["at_gate"].get("keys"))[:3])
            still = f"yes: its test still fails, so FI left {keys} out of the results the paper uses, and the paper says so"
        elif row["open"] and row.get("at_gate", {}).get("outcome") == "not_computed":
            still = "yes: the code does not compute it, and the paper says so as a part not done"
        elif row["open"]:
            still = ("yes, and it changes a result" if row["changes_result"] else
                     f"yes, but the results hold it as {', '.join(row['reported_as'][:3])}: a difference of names"
                     if row.get("reported_as") else "yes, but it changes nothing reported")
        else:
            still = "no"
        lines.append(f"| {_cell(row['id'])} | {_cell(row['where'] or '-')} | {_cell(row['what'])} | "
                     f"{'yes' if row['changes_result'] else 'no'} | {_cell(shown)} | {_cell(repaired)} | {_cell(still)} |")
    if len(code["issues"]) > 40:
        lines.append(f"(+{len(code['issues']) - 40} more in .fi/code_review.json)")
    return lines


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _paper_pdf_line(pdf: dict[str, Any]) -> str:
    pages = f", {pdf['pages']} pages" if pdf.get("pages") else ""
    if not pdf["typeset"]:
        return f"paper.pdf is **not typeset**: a browser drew it{pages}. Why: {_one(pdf['reason'], 300)}"
    changed = len(pdf["glyphs_replaced"])
    return (f"paper.pdf is typeset by {pdf['engine'] or 'LaTeX'}{pages}."
            + (f" {changed} character(s) are printed in another form or font than the text's:" if changed else ""))


def _citations_line(c: dict[str, Any]) -> str:
    n = c.get("counts") or {}
    who = (f" Read by {c.get('reader')}" + (", the model that wrote the paper." if c.get("reader_is_writer") else ".")
           if c.get("reader") else "")
    return (f"{n.get('citations', 0)} citation(s): {n.get('supported', 0)} backed by a passage of the retrieved text as "
            f"written, {n.get('found_unsupported', 0)} found unsupported and fixed ({n.get('replaced', 0)} cited to another "
            f"source, {n.get('reworded', 0)} reworded, {n.get('citation_dropped', 0)} citation dropped, "
            f"{n.get('sentence_dropped', 0)} sentence dropped)"
            + (f", {n.get('unchecked')} could not be checked" if n.get("unchecked") else "") + "." + who)


def _paper_record_lines(records: dict[str, Any]) -> list[str]:
    """The section of what FI recorded about the run that the paper does not carry, and what its check of the paper found."""
    labels = (("how_reached", "How this result was reached"), ("run_record", "What FI recorded about the run"),
              ("code_findings", "Open points in the code that produced the results"))
    parts = [(title, records.get(key)) for key, title in labels if records.get(key)]
    check = records.get("paper_check") or {}
    unmet = [i for i in check.get("unmet_items") or [] if isinstance(i, dict)]
    hits = [h for h in check.get("hits") or [] if h]
    unfilled = [u for u in check.get("unfilled") or [] if u]
    removed = [r for r in check.get("removed") or [] if isinstance(r, dict)]
    taken_out = [r for r in check.get("taken_out") or [] if isinstance(r, dict)]
    not_filled = str(check.get("not_filled") or "")
    if not parts and not unmet and not hits and not unfilled and not removed and not taken_out and not not_filled:
        return []
    lines = ["", "## What FI recorded that the paper does not carry", "",
             "These are FI's records of the run. The paper's seed, measured run time, time limit, trial counts and the "
             "parts the study did not do are filled in by FI from these records; the full records are kept here, not "
             "pasted into the paper.", ""]
    for title, text in parts:
        lines += [f"### {title}", "", str(text).strip(), ""]
    if unmet or hits:
        lines += ["### Where the paper still disagrees with these records", "",
                  "FI sent the paper back to be rewritten for these and the review rounds ended before they were settled:", ""]
        lines += [f"- Not in the paper's Limitations: {_cell(str(i.get('text') or i.get('name') or ''))} "
                  f"(record `{_cell(str(i.get('id') or i.get('kind') or ''))}`)" for i in unmet]
        ids = {str(i.get("id")) for i in unmet}
        lines += [f"- {_cell(_one(h, 300))}" for h in hits if not any(i and i in h for i in ids)]
        lines.append("")
    if removed:
        lines += ["### Numbers taken out of the paper", "",
                  "The writer typed these numbers instead of using the results FI fills in from its own records, and the "
                  "review rounds ended before the writer fixed them, so each is replaced by \"[number removed]\" in the paper:",
                  ""]
        lines += [f"- {_cell(str(r.get('number')))}: {_cell(_one(r.get('sentence'), 240))}" for r in removed]
        lines.append("")
    if taken_out:
        lines += ["### Values taken out of the paper", "",
                  "The paper asked for these values, but FI has no record that fills them and the review rounds ended before "
                  "the writer fixed the key. FI put no number in their place: where a sentence carried only such values it "
                  "is removed, and anywhere else the value is replaced by \"[number removed]\":", ""]
        lines += [f"- {_cell(str(r.get('how')))}: {_cell(_one(r.get('sentence'), 240))} (`{_cell(str(r.get('key')))}`)"
                  for r in taken_out]
        lines.append("")
    if not_filled:
        lines += ["### Result numbers in this paper", "",
                  ("The study has no per-setting trial record, so FI could not fill in the paper's result numbers from its "
                   "records, and they were not checked against the plan." if not_filled == "no_records" else
                   "This paper was written before FI filled in its result numbers from its own records, so its numbers were "
                   "not checked against the plan; run the write step again to have them filled in."), ""]
    if unfilled:
        lines += ["### Values FI could not fill in", "",
                  "The paper asks for these values, but FI has no record of them, so the paper reads \"not recorded\" there:",
                  ""]
        lines += [f"- `{_cell(u)}`" for u in unfilled]
        lines.append("")
    return lines


def render(report: dict[str, Any]) -> str:
    """QUALITY.md: plain words for a scientist."""
    o = report["outcome"]
    lines = ["# Quality report", "", f"Written {_day(report.get('written_at'))}"
             + (f", for the run that started {_day(report['run_started_at'])}" if report.get("run_started_at") else "") + ".", "",
             f"Quest `{report['quest_id']}`" + (f" - {report['title']}" if report.get("title") else ""),
             "", "## Outcome", "", o["line"] + (f" Evidence: {o['evidence']}" if o.get("evidence") else ""), "",
             "## Stage by stage", "",
             "| Stage | Written by | Read by | Reading cost | Problems found | Fixed / not fixed / left out | Calls | Tokens (prompt/completion) | Time |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in report["stages"]:
        p = s["problems"]
        reader = s["reviewed_by"] or "-"
        if s["same_model"] and s["reviewed_by"]:
            reader += " (same model)"
        elif s["review_note"] and not s["reviewed_by"]:
            reader = _one(s["review_note"], 60)
        first = "; ".join(f"{_one(i['text'], 90)} [{i['found_by']}]" for i in p["items"][:3])
        more = f" (+{p['total'] - 3} more)" if p["total"] > 3 else ""
        rc = s.get("review_cost")
        cost = "-" if not rc or not rc["calls"] else (
            f"{rc['calls']} call{'s' if rc['calls'] != 1 else ''}, {_n(rc['prompt_tokens'])} / {_n(rc['completion_tokens'])} tokens"
            + (f", ${rc['usd']:.2f}" if rc.get("usd") is not None else ""))
        lines.append(f"| {STAGE_NAME[s['stage']]} | {_cell(', '.join(s['wrote']) or '-')} | {_cell(reader)} | {_cell(cost)} | "
                     f"{p['total']}{(': ' + _cell(first) + more) if first else ''} | "
                     f"{p['fixed']} / {p['not_fixed']} / {p['left_out']} | {s['calls']}"
                     f"{(' (' + str(s['failed_calls']) + ' failed)') if s['failed_calls'] else ''} | "
                     f"{_n(s['prompt_tokens'])} / {_n(s['completion_tokens'])} | {_span(s['seconds'])} |")
    code = report.get("code") or {}
    if code.get("functions_total"):
        lines += ["", f"Functions written and tested: {code['functions_done']} of {code['functions_total']}."]
    if code.get("final_version_read") is not None:
        lines += ["", "The final version of the code was " + ("" if code["final_version_read"] else "not ")
                  + "read by a second model."
                  + (" Open findings per reading: " + ", ".join(
                      f"{n} ({h} high-severity)" for n, h in zip(code["open_per_reading"], code["open_high_per_reading"]))
                     + "." if code.get("open_per_reading") else "")
                  + ("" if code.get("repairs_converged") is None else
                     " The repairs " + ("converged." if code["repairs_converged"] else
                                        "did not converge (they were stopped when a round did not reduce the open "
                                        "findings that change a result)."))
                  + (f" Repairs ended because {code['repairs_stopped']}; {code['repair_rounds']} round(s) and "
                     f"{code['repair_requests']} request(s) were used." if code.get("repairs_stopped") else "")]
    if code.get("issues"):
        lines += _issue_lines(code)
    pilot = report.get("pilot")
    if pilot:
        lines += ["", f"Pilot check: {pilot['says']}. Repairs it caused: {pilot['repairs']}."]
    trials = report.get("trial_checks")
    if trials:
        lines += ["", "## FI's checks of the trials", ""]
        if trials.get("not_run"):
            lines += [f"{line}." for line in trials["lines"]]
        else:
            lines += ["FI counted the failed trials of each setting and measured the intervals and standard errors the "
                      "simulation reports against the true values it was given:"]
            lines += [f"- {line}" for line in trials["lines"]]
    lines += ["", "## Where it failed", ""]
    f = report.get("failure")
    if f:
        lines.append(f"The first stage where something could not be fixed: **{STAGE_NAME[f['stage']]}**.")
        lines += ["", *[f"{i}. {_one(c, 300)}" for i, c in enumerate(f["chain"], 1)]]
    else:
        lines.append("Nothing was left unfixed that stopped the quest." if o["kind"] == "finished"
                     else "The records name no stage where the quest could not go on.")
    cites = report.get("citations")
    if cites:
        lines += ["", "## Citations", "", _citations_line(cites), ""]
        for r in cites["citations"]:
            if str(r.get("action") or "").startswith("added"):
                lines.append(f"- [{r['label']}] cited in the fixed sentence: \"{_one(r.get('passage'), 200)}\"")
                continue
            what = (f"backed by: \"{_one(r.get('passage'), 200)}\"" if r["status"] == "supported"
                    else f"{str(r['status']).replace('_', ' ')} ({_one(r.get('problem'), 160)})")
            lines.append(f"- [{r['label']}] {_one(r.get('sentence'), 90)} -- {what}")
    pdf = report.get("paper_pdf")
    if pdf:
        lines += ["", "## The PDF", "", _paper_pdf_line(pdf)]
        lines += [f"- `{_cell(str(g.get('glyph', '')))}` ({g.get('code_point', '')}) printed as "
                  f"{_cell(str(g.get('became', '')))}, {g.get('count', 1)} time(s)" for g in pdf["glyphs_replaced"]]
        lines += [f"- `{_cell(str(g.get('glyph', '')))}` ({g.get('code_point', '')}): no font here has it"
                  for g in pdf["glyphs_not_set"]]
    if report.get("left_out_of_paper"):
        lines += ["", "## Left out of the paper", "",
                  "These quantities are not in the paper's results; FI checks that the paper says so:", ""]
        lines += [f"- `{_cell(x['quantity'])}`: {_cell(x['why'])}" for x in report["left_out_of_paper"]]
    lines += _paper_record_lines(report.get("paper_records") or {})
    lines += ["", "## What to try", ""]
    if report["suggestions"]:
        lines += [f"- {s['text']} (from {s['source']})" for s in report["suggestions"]]
    else:
        lines.append("Nothing in the records points to a change.")
    c = report["cost"]
    lines += ["", "## Cost", "", f"Total: {_n(c['calls'])} model calls, {_n(c['prompt_tokens'])} prompt and "
              f"{_n(c['completion_tokens'])} completion tokens"
              + (f", ${c['usd']:.2f}" if c.get("usd") is not None else ", price not known")
              + (f", {_span(report['seconds'])} of work" if report.get("seconds") else "") + ".", "",
              "| Model | Calls | Failed calls | Prompt tokens | Completion tokens |", "|---|---|---|---|---|"]
    for name, m in sorted(c["by_model"].items(), key=lambda kv: -kv[1]["calls"]):
        lines.append(f"| {_cell(name)} | {m['calls']} | {m['failed_calls']} | {_n(m['prompt_tokens'])} | {_n(m['completion_tokens'])} |")
    lines += ["", "Read from the records FI wrote while it ran; no model was asked and nothing was changed. "
              "A suggestion names the record it comes from.", ""]
    return "\n".join(lines)


def write(quest_root: Path | str, out_dir: Path | str | None = None, *, running: bool | None = None) -> dict[str, Any]:
    """Build the report and write ``QUALITY.md`` and ``quality.json``: at the quest root and in ``.fi/``, or both into
    ``out_dir`` when one is given (the quest is then left untouched)."""
    root = Path(quest_root)
    report = build(root, running=running)
    if out_dir is None:
        md, js = root / "QUALITY.md", root / ".fi" / "quality.json"
    else:
        target = Path(out_dir)
        md, js = target / "QUALITY.md", target / "quality.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render(report), encoding="utf-8")
    js.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report


def write_quietly(quest_root: Path | str) -> None:
    """The engine's call: a report that cannot be written never changes the quest."""
    try:
        write(quest_root, running=False)  # the engine calls this as its run ends: the run is over
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------------------------- the cross-quest table

def quest_dirs(folders: list[Path | str]) -> list[Path]:
    """Every quest under the folders (a quest folder itself counts): a folder with a ``.fi`` or ``needs`` inside."""
    found: list[Path] = []

    def is_quest(p: Path) -> bool:
        return (p / ".fi").is_dir() or (p / "needs").is_dir()

    def walk(p: Path, depth: int) -> None:
        if is_quest(p):
            found.append(p)
            return
        if depth <= 0:
            return
        try:
            kids = sorted(k for k in p.iterdir() if k.is_dir() and not k.name.startswith("."))
        except OSError:
            return
        for k in kids:
            walk(k, depth - 1)

    for folder in folders:
        walk(Path(folder), 3)
    seen: set[Path] = set()
    return [p for p in found if not (p.resolve() in seen or seen.add(p.resolve()))]


def load(quest_root: Path) -> dict[str, Any]:
    """The stored report when there is one, else one built from the records (never written)."""
    stored = _dict(_json(Path(quest_root) / ".fi" / "quality.json"))
    if stored.get("schema") != SCHEMA:
        return build(quest_root)
    # A quest that is running now has moved on from its stored report: say so, built from the records so far.
    if build_running(Path(quest_root)):
        return build(quest_root, running=True)
    return stored


def build_running(root: Path) -> bool:
    try:
        from . import quest_title
        return bool(quest_title.looks_running(root))
    except Exception:  # noqa: BLE001
        return False


def _row(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    cfg = report.get("config") or {}
    per = {s["stage"]: s["problems"]["total"] for s in report["stages"]}
    p = report["problems"]
    fail = report.get("failure") or {}
    c = report["cost"]
    return {
        "quest": report["quest_id"], "folder": str(root), "outcome": report["outcome"]["kind"],
        "main_model": cfg.get("main_model", ""),
        "routed_steps": "; ".join(f"{k}={v}" for k, v in (cfg.get("routed") or {}).items()),
        "failing_stage": fail.get("stage", ""),
        **{f"problems_{s}": per.get(s, 0) for s in STAGES},
        "problems_total": p["total"], "fixed_rate": "" if p["fixed_rate"] is None else f"{p['fixed_rate']:.0%}",
        "prompt_tokens": c["prompt_tokens"], "completion_tokens": c["completion_tokens"],
        "minutes": "" if report.get("seconds") is None else round(report["seconds"] / 60, 1),
    }


def table(folders: list[Path | str]) -> tuple[list[dict[str, Any]], str, str]:
    """One row per quest under the folders: (rows, Markdown, CSV)."""
    rows = [_row(d, load(d)) for d in quest_dirs(folders)]
    if not rows:
        return [], "No quests found.\n", ""
    cols = list(rows[0])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    shown = ["quest", "outcome", "main_model", "routed_steps", "failing_stage", *[f"problems_{s}" for s in STAGES],
             "fixed_rate", "prompt_tokens", "completion_tokens", "minutes"]
    head = {"quest": "Quest", "outcome": "Outcome", "main_model": "Main model", "routed_steps": "Routed steps",
            "failing_stage": "Failed at", "fixed_rate": "Fixed", "prompt_tokens": "Prompt tokens",
            "completion_tokens": "Completion tokens", "minutes": "Minutes",
            **{f"problems_{s}": f"Problems: {s}" for s in STAGES}}
    md = ["| " + " | ".join(head[c] for c in shown) + " |", "|" + "---|" * len(shown)]
    for r in rows:
        md.append("| " + " | ".join(_cell(str(r[c]) if c != "quest" else _one(r[c], 48)) for c in shown) + " |")
    return rows, "\n".join(md) + "\n", buf.getvalue()


def write_table(folders: list[Path | str], out_dir: Path | str) -> tuple[list[dict[str, Any]], str]:
    rows, md, csv_text = table(folders)
    if rows:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "quality_table.md").write_text(md, encoding="utf-8")
        (out / "quality_table.csv").write_text(csv_text, encoding="utf-8", newline="")
    return rows, md
