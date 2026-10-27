"""Score one FI tree on a fixed list of recorded cases, so that a change can be judged against a baseline.

A case is one recorded input whose right outcome is known: a quest that stopped and should not have, one that must still
stop, a planted fault that must be caught, a clean record that must not be flagged. The cases file lives outside the
repository (it names real quests); this module only runs it. Each case names a *probe*: a script that replays that
input on the tree under test and prints, as its last line, ``ACCEPT_JSON: {"got": <outcome>, "detail": <text>}``. The
probe is how the case is replayed with today's code; when a later change removes the code a probe calls, the probe is
rewritten and the case and its expected outcome stay as they are.

    python -m dev.evaluation.acceptance --tree C:/dev/fi_wt_x --cases cases.yaml --out DIR [--against OLD/score.json]

The score (``score.json`` and ``score.md`` in ``--out``) holds every case's expected and measured outcome and the totals:
false stops, missed stops, faults caught and missed, false flags, other failures, not measured (the probe failed or
printed no outcome) and cases with no probe yet. ``--against`` lists the cases that passed in the older score and do not
now, which is what decides whether a change may be kept.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

MARK = "ACCEPT_JSON:"

#: How a measured outcome that differs from the expected one is counted, by the expected outcome.
_FAILURE_KIND = {"go_on": "false_stop", "stop": "missed_stop", "catch": "missed_fault", "clean": "false_flag"}


@dataclass
class Case:
    id: str
    expect: str
    quest: str = ""
    probe: str = ""
    args: list[str] = field(default_factory=list)
    group: str = ""
    note: str = ""
    timeout_s: int = 600


@dataclass
class Result:
    id: str
    group: str
    expect: str
    got: str  # the probe's outcome, "not_measured" or "no_probe"
    passed: bool
    kind: str  # "pass", a key of _FAILURE_KIND's values, "other_fail", "not_measured" or "no_probe"
    detail: str = ""
    seconds: float = 0.0


def load(path: Path) -> tuple[list[Case], Path, Path]:
    """``(cases, quest root, probe folder)`` from a cases file. Relative quest paths are read under ``root`` and relative
    probe paths under ``probes`` (both default to the cases file's folder)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = Path(data.get("root") or path.parent)
    probes = Path(data.get("probes") or path.parent)
    cases: list[Case] = []
    seen: set[str] = set()
    for row in data.get("cases") or []:
        if not isinstance(row, dict) or not row.get("id") or not row.get("expect"):
            raise ValueError(f"a case needs an id and an expected outcome: {row!r}")
        if row["id"] in seen:
            raise ValueError(f"two cases share the id {row['id']!r}")
        seen.add(row["id"])
        cases.append(Case(id=str(row["id"]), expect=str(row["expect"]), quest=str(row.get("quest") or ""),
                          probe=str(row.get("probe") or ""), args=[str(a) for a in row.get("args") or []],
                          group=str(row.get("group") or ""), note=str(row.get("note") or ""),
                          timeout_s=int(row.get("timeout_s") or 600)))
    return cases, root, probes


def outcome(stdout: str) -> dict[str, Any] | None:
    """The last ``ACCEPT_JSON:`` line a probe printed, read; ``None`` when there is none or it is not a JSON object with
    a ``got``."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARK):
            try:
                value = json.loads(line[len(MARK):])
            except ValueError:
                return None
            return value if isinstance(value, dict) and value.get("got") else None
    return None


def judge(case: Case, got: str, detail: str = "", seconds: float = 0.0) -> Result:
    if got in ("not_measured", "no_probe"):
        return Result(case.id, case.group, case.expect, got, False, got, detail, seconds)
    if got == case.expect:
        return Result(case.id, case.group, case.expect, got, True, "pass", detail, seconds)
    kind = _FAILURE_KIND.get(case.expect, "other_fail")
    if kind == "false_stop" and got != "stop":
        kind = "other_fail"  # a run that should go on and did something else than stop is not a false stop
    return Result(case.id, case.group, case.expect, got, False, kind, detail, seconds)


def run_case(case: Case, tree: Path, root: Path, probes: Path, out: Path) -> Result:
    if not case.probe:
        return judge(case, "no_probe", case.note)
    probe = Path(case.probe) if Path(case.probe).is_absolute() else probes / case.probe
    quest = (Path(case.quest) if Path(case.quest).is_absolute() else root / case.quest) if case.quest else Path()
    work = out / "cases" / case.id
    work.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    try:
        done = subprocess.run([sys.executable, str(probe), str(tree), str(quest), *case.args], cwd=work,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=case.timeout_s, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return judge(case, "not_measured", f"the probe ran past {case.timeout_s} s", time.monotonic() - began)
    seconds = time.monotonic() - began
    (work / "stdout.txt").write_text(done.stdout, encoding="utf-8")
    (work / "stderr.txt").write_text(done.stderr, encoding="utf-8")
    read = outcome(done.stdout)
    if read is None:
        tail = (done.stderr.strip().splitlines() or ["no output"])[-1]
        return judge(case, "not_measured", f"no outcome printed (exit {done.returncode}): {tail[:300]}", seconds)
    return judge(case, str(read["got"]), str(read.get("detail") or "")[:500], seconds)


def totals(results: list[Result]) -> dict[str, int]:
    out = {"cases": len(results), "measured": 0, "passed": 0, "false_stop": 0, "missed_stop": 0, "faults": 0,
           "faults_caught": 0, "missed_fault": 0, "false_flag": 0, "other_fail": 0, "not_measured": 0, "no_probe": 0}
    for r in results:
        out["measured"] += r.kind not in ("not_measured", "no_probe")
        out["passed"] += r.passed
        if r.expect == "catch" and r.kind not in ("not_measured", "no_probe"):
            out["faults"] += 1
            out["faults_caught"] += r.passed
        if r.kind != "pass":
            out[r.kind] += 1
    return out


def regressions(old: dict[str, Any], results: list[Result]) -> list[str]:
    """The ids of the cases that passed in the older score and do not now (a case the older score lacks is not one)."""
    before = {r["id"]: r for r in old.get("results") or []}
    return sorted(r.id for r in results if before.get(r.id, {}).get("passed") and not r.passed)


def core_lines(tree: Path) -> int:
    """The number of lines of FI's own code (``core/**/*.py``) in the tree: the rebuild must make this smaller."""
    total = 0
    for path in (tree / "core").rglob("*.py"):
        try:
            total += len(path.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError):
            continue
    return total


def head(tree: Path) -> str:
    try:
        done = subprocess.run(["git", "-C", str(tree), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return done.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def score(tree: Path, cases_file: Path, out: Path, *, only: set[str] | None = None, jobs: int = 2,
          against: Path | None = None) -> dict[str, Any]:
    cases, root, probes = load(cases_file)
    if only:
        cases = [c for c in cases if c.id in only or c.group in only]
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(lambda c: run_case(c, tree, root, probes, out), cases))
    report: dict[str, Any] = {"tree": str(tree), "head": head(tree), "cases_file": str(cases_file),
                              "when": time.strftime("%Y-%m-%d %H:%M:%S"), "core_lines": core_lines(tree),
                              "totals": totals(results), "results": [asdict(r) for r in results]}
    if against is not None:
        old = json.loads(against.read_text(encoding="utf-8"))
        report["against"] = {"score": str(against), "head": old.get("head"), "regressions": regressions(old, results),
                             "core_lines_before": old.get("core_lines")}
    (out / "score.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "score.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(report: dict[str, Any]) -> str:
    t = report["totals"]
    lines = [f"# Acceptance score: {report.get('head') or report['tree']} ({report['when']})", "",
             f"cases {t['cases']}, measured {t['measured']}, passed {t['passed']}; false stops {t['false_stop']}, "
             f"missed stops {t['missed_stop']}, faults caught {t['faults_caught']} of {t['faults']}, false flags "
             f"{t['false_flag']}, other failures {t['other_fail']}, not measured {t['not_measured']}, no probe yet "
             f"{t['no_probe']}; core lines {report['core_lines']}", ""]
    if "against" in report:
        a = report["against"]
        lines += [f"Against {a.get('head')}: {len(a['regressions'])} case(s) passed before and do not now"
                  + (": " + ", ".join(a["regressions"]) if a["regressions"] else "")
                  + f"; core lines {a.get('core_lines_before')} -> {report['core_lines']}", ""]
    lines += ["| case | group | expected | got | result | detail |", "|---|---|---|---|---|---|"]
    for r in report["results"]:
        detail = str(r["detail"]).replace("|", "/").replace("\n", " ")[:160]
        lines.append(f"| {r['id']} | {r['group']} | {r['expect']} | {r['got']} | {r['kind']} | {detail} |")
    return "\n".join(lines) + "\n"
