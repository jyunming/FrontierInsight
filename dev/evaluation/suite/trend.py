"""The version-over-version table (``trend.md``) and the failure Pareto (``pareto.md``), from every ``round.json``.

Definitions (also printed in the files):

* **trustworthy paper**: a delivered paper graded G1, G2 and G3 all pass, whose run record does not contradict it.
* **fake paper**: a delivered paper that fails G1, or whose own record contradicts it (the experiment was not executed, or a
  checks/code/run/analysis problem is recorded as not fixed).
* **false stop**: an honest stop a grader marked as wrong (``stop_correct: false``) or whose cause a grader set to an FI bug.
* Rates are over ALL topics run in the round (a crash or a pause counts as not trustworthy).

Done criteria: trustworthy >= 70% of topics, fake papers = 0, false stops <= 10% of topics. A criterion is **met** or
**not met** only when the data decide it whatever the still ungraded papers and unjudged stops turn out to be; otherwise
it reads **not yet known** (with the number waiting for a grader). A round with no topics decides nothing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from core import quality_report as qr

from . import taxonomy

TRUSTWORTHY_MIN = 0.70
FALSE_STOP_MAX = 0.10


def load_rounds(rounds_dir: Path) -> list[dict[str, Any]]:
    """Every ``round.json`` under ``rounds_dir``, oldest first (by creation time, then name)."""
    rounds = []
    for path in sorted(Path(rounds_dir).glob("*/round.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("topics") is not None:
            rounds.append(data)
    rounds.sort(key=lambda r: (str(r.get("created_at") or ""), str(r.get("round"))))
    return rounds


def _verdict(low: float, high: float, ok: Any) -> str:
    """met / not met / not yet known, for a rate somewhere in [low, high] against ``ok(rate)``."""
    if ok(low) and ok(high):
        return "met"
    if not ok(low) and not ok(high):
        return "not met"
    return "not yet known"


def criteria(m: dict[str, Any]) -> dict[str, str]:
    n = m["topics"]
    if not n:
        return {"trustworthy": "not yet known", "fake": "not yet known", "false_stops": "not yet known"}
    trust_low = m["papers_trustworthy"] / n
    trust_high = (m["papers_trustworthy"] + m["papers_ungraded"]) / n
    fake_low = m["papers_fake"]
    fake_high = m["papers_fake"] + m["papers_ungraded"]
    stop_low = m["stops_false"] / n
    stop_high = (m["stops_false"] + m["stops_unjudged"]) / n
    unfinished = m.get("unfinished", 0)
    if unfinished:  # a topic still running could turn out either way
        trust_high = min(1.0, trust_high + unfinished / n)
        fake_high += unfinished
        stop_high = min(1.0, stop_high + unfinished / n)
    return {"trustworthy": _verdict(trust_low, trust_high, lambda r: r >= TRUSTWORTHY_MIN),
            "fake": _verdict(fake_low, fake_high, lambda c: c == 0),
            "false_stops": _verdict(stop_low, stop_high, lambda r: r <= FALSE_STOP_MAX)}


def _sum(topics: list[dict[str, Any]], key: str) -> int | None:
    vals = [t[key] for t in topics if isinstance(t.get(key), (int, float))]
    return int(sum(vals)) if vals else None


def trend_row(rd: dict[str, Any]) -> dict[str, Any]:
    m = rd["metrics"]
    topics = rd["topics"]
    fi = rd.get("fi_git") or {}
    tokens = None
    if any(t.get("prompt_tokens") is not None for t in topics):
        tokens = sum(int(t.get("prompt_tokens") or 0) + int(t.get("completion_tokens") or 0) for t in topics)
    usd = [t["usd"] for t in topics if isinstance(t.get("usd"), (int, float))]
    wall = _sum(topics, "wall_seconds")
    return {"round": rd["round"], "control": bool(rd.get("control")), "fi": fi.get("short") or "unknown",
            "fi_dirty": fi.get("dirty"), "fi_note": rd.get("fi_note"), "metrics": m, "criteria": criteria(m),
            "tokens": tokens, "usd": round(sum(usd), 2) if usd else None,
            "seconds": wall if wall is not None else _sum(topics, "work_seconds"),
            "seconds_kind": "wall" if wall is not None else "work"}


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.0%})" if n else "-"


def render_trend(rows: list[dict[str, Any]]) -> str:
    lines = ["# Trend: one row per round", "", "Counts are over every topic run in the round. Papers: delivered. Trustworthy: graded G1, G2, G3 "
             "pass and the record agrees. Fake: fails G1 or the record contradicts it. False stops: an honest stop a grader marked wrong. "
             "A figure in brackets is still waiting for a grader.", "",
             "| Round | FI version | Topics | Papers | Trustworthy | Fake | False stops | Honest stops | Pauses (person) | Crashes | Tokens | Cost | Time | Done? |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, c = r["metrics"], r["criteria"]
        n = m["topics"]
        fi = r["fi"] + ("+dirty" if r["fi_dirty"] else "") + (" (?)" if r.get("fi_note") else "")
        trust = _pct(m["papers_trustworthy"], n) + (f" [{m['papers_ungraded']} to grade]" if m["papers_ungraded"] else "")
        fake = str(m["papers_fake"]) + (f" [{m['papers_ungraded']} to grade]" if m["papers_ungraded"] else "")
        stops = str(m["stops_false"]) + (f" [{m['stops_unjudged']} not judged]" if m["stops_unjudged"] else "")
        done = "yes" if all(v == "met" for v in c.values()) else (
            "no" if any(v == "not met" for v in c.values()) else "not yet known")
        t = r["seconds"]
        lines.append(f"| {r['round']}{' (control)' if r['control'] else ''} | {fi} | {n} | {m['papers']} | {trust} | {fake} | {stops} | "
                     f"{m['honest_stops']} | {m['pauses_needing_a_person']} | {m['crashes']} | "
                     f"{'-' if r['tokens'] is None else format(r['tokens'], ',')} | "
                     f"{'-' if r['usd'] is None else '$' + format(r['usd'], '.2f')} | {'-' if t is None else qr._span(t)}"
                     f"{'' if t is None or r['seconds_kind'] == 'wall' else ' (work)'} | {done} |")
    lines += ["", "## Done criteria by round", "",
              f"Trustworthy >= {TRUSTWORTHY_MIN:.0%} of topics; fake papers = 0; false stops <= {FALSE_STOP_MAX:.0%} of topics.", "",
              "| Round | Trustworthy >= 70% | Fake = 0 | False stops <= 10% |", "|---|---|---|---|"]
    for r in rows:
        c = r["criteria"]
        lines.append(f"| {r['round']} | {c['trustworthy']} | {c['fake']} | {c['false_stops']} |")
    notes = [f"- {r['round']}: FI version {r['fi']} - {r['fi_note']}" for r in rows if r.get("fi_note")]
    if notes:
        lines += ["", "## Notes on the FI version", "", *notes]
    return "\n".join(lines) + "\n"


def pareto(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Failures across all rounds: per class, per stage x class, and the contributing flags."""
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flags: dict[str, list[str]] = defaultdict(list)
    for rd in rounds:
        for t in rd["topics"]:
            c = t.get("class")
            if not c:
                continue
            item = {"round": rd["round"], "topic": t["topic"], "quest": t.get("quest_id"), "outcome": t["outcome"],
                    "stage": c["stage"] or "?", "rule": c["rule"], "evidence": c["evidence"]}
            by_class[c["class"]].append(item)
            for a in c.get("also", []):
                flags[a.split(" (")[0] if a.startswith("static") else a].append(f"{rd['round']}/{t['topic']}")
    return {"by_class": by_class, "flags": flags}


def render_pareto(rounds: list[dict[str, Any]]) -> str:
    p = pareto(rounds)
    total = sum(len(v) for v in p["by_class"].values())
    lines = ["# Failure Pareto (all rounds so far)", "",
             f"{total} failure(s) in {len(rounds)} round(s). A failure is a run that delivered no paper, or a paper that failed a gate or "
             "whose record contradicts it. Each is placed by stage and cause class with a stated rule (see `dev/evaluation/suite/taxonomy.py`); "
             "a run no rule fits is `unclassified`, never guessed.", ""]
    if not total:
        return "\n".join(lines + ["No failures recorded yet.", ""])
    order = sorted(p["by_class"], key=lambda k: (-len(p["by_class"][k]), k))
    lines += ["| Cause class | Count | Share | Cumulative |", "|---|---|---|---|"]
    run = 0
    for k in order:
        run += len(p["by_class"][k])
        lines.append(f"| {taxonomy.CLASSES.get(k, k)} | {len(p['by_class'][k])} | {len(p['by_class'][k]) / total:.0%} | {run / total:.0%} |")
    stages = defaultdict(lambda: defaultdict(int))
    for k, items in p["by_class"].items():
        for it in items:
            stages[it["stage"]][k] += 1
    cols = list(order)
    lines += ["", "## Stage by cause class", "",
              "| Stage | " + " | ".join(taxonomy.CLASSES.get(k, k) for k in cols) + " | Total |", "|---|" + "---|" * (len(cols) + 1)]
    for s in [*taxonomy.STAGES, "?"]:
        if s in stages:
            lines.append(f"| {s} | " + " | ".join(str(stages[s].get(k, "")) or "" for k in cols) + f" | {sum(stages[s].values())} |")
    for k in order:
        lines += ["", f"## {taxonomy.CLASSES.get(k, k)} ({len(p['by_class'][k])})", ""]
        for it in p["by_class"][k]:
            lines.append(f"- `{it['round']}/{it['topic']}` ({it['outcome']}, stage {it['stage']}, {it['rule']}; quest `{it['quest']}`): "
                         f"{it['evidence']}")
    if p["flags"]:
        lines += ["", "## Contributing flags (any run, whatever its primary class)", "", "| Flag | Runs |", "|---|---|"]
        for flag, runs in sorted(p["flags"].items(), key=lambda kv: (-len(kv[1]), kv[0])):
            lines.append(f"| {flag.replace('|', '/')} | {len(runs)}: {', '.join(runs)} |")
    return "\n".join(lines) + "\n"


def write_reports(rounds_dir: Path) -> tuple[Path, Path]:
    rounds = load_rounds(rounds_dir)
    trend_path, pareto_path = Path(rounds_dir) / "trend.md", Path(rounds_dir) / "pareto.md"
    trend_path.write_text(render_trend([trend_row(r) for r in rounds]), encoding="utf-8")
    pareto_path.write_text(render_pareto(rounds), encoding="utf-8")
    return trend_path, pareto_path
