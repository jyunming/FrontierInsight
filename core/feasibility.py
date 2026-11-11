"""Trying the plan on a small case before the study: FI's record of what the trial showed (``needs/FEASIBILITY.json``).

With ``pauses.plan: after_trial`` the quest goes through its usual steps up to the line before the protocol is frozen,
without the reading of the code: FI runs the simulation on a small case (:mod:`core.small_run`, with its crash repairs and
its one send-back per exact fault), times the whole study with its pilot (:mod:`core.run_estimate`) and stops. This module
turns what those steps already recorded into one plain sentence per check and a verdict; it never calls a model, never runs
anything and reads no value of a result beyond whether it is a finite number. The engine's side (where the trial runs, the
stop, the resume) is ``RunChecksMixin._try_plan_first``.

The six checks, each from FI's own records:

* ``runs`` -- the small run (``.fi/small_run.json``): it loaded and most trials of every setting ran.
* ``numbers`` -- each declared number had a finite value at one setting FI tried or more; the pilot's ``undefined`` or
  ``stuck`` finding that its repairs left (``.fi/pilot_check.json``) fails it.
* ``results`` -- no exact fault of the small run (a number returned per value where the plan declares one per run, a
  declared result no trial returned with every trial whole, a fault of the analysis whatever the data) is left.
* ``moves`` -- every setting the study compares was asked for by the code; one nothing asks for and the code's text never
  looks up, or the pilot's flat output, fails it.
* ``time`` -- FI's timing (``.fi/run_estimate.json``): the whole study fits the time allowed.
* ``data`` -- the files the study was given are there.

A ``fail`` is only what FI's checks class as exact and that is still there after its one send-back or the repairs; what a
small sample can explain is ``not_shown``, never ``fail``. ``feasible`` needs every check passed and every goal answered;
the goals are not read in this build (``goals.state`` is ``could_not_judge``), so the verdict is ``feasible`` only once
that reading exists, ``not_feasible`` when a check fails and ``unsure`` otherwise.

The stop and the resume: the record is written first (a temporary file renamed), then the marker
``.fi/paused_at_trial.flag`` (the record's plan hash, code version and inputs hash), then the pause. On a resume
:func:`on_resume` says whether the trial is made again (the record is missing or partial, or the plan or the code changed)
or the study goes on; ``--approve-as <who>`` on that resume is kept (:func:`note_approval`) and the freeze names the person
and every change FI's checks made to the protocol after the trial (:func:`changed_after_trial`). Nothing here knows a
field of science.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

SCHEMA = "fi.feasibility/v1"
RECORD = Path("needs") / "FEASIBILITY.json"
#: Older records (one per trial) and FI's own files of the trial: the protocol that was tried, a person's approval.
OLDER = Path(".fi") / "feasibility"
TRIED = OLDER / "tried_protocol.json"
APPROVAL = OLDER / "approved.json"
MARKER = Path(".fi") / "paused_at_trial.flag"
#: The values of results taken from other studies (written by FI when a study declares them; absent otherwise).
STUDY_INPUTS = Path("inputs") / "study_inputs.json"
MODE = "after_trial"

PASS, FAIL, NOT_SHOWN, NOT_APPLICABLE = "pass", "fail", "not_shown", "not_applicable"
FEASIBLE, NOT_FEASIBLE, UNSURE = "feasible", "not_feasible", "unsure"
CHECKS = ("runs", "numbers", "results", "moves", "time", "data")
#: Each check in a few plain words, for a sentence that names what could not be shown.
_WHAT = {"runs": "that the simulation runs", "numbers": "that its numbers are finite",
         "results": "that every result comes back in its declared form", "moves": "that the results follow the settings",
         "time": "that it fits the time allowed", "data": "that what it reads is there"}
NOT_TRIED_YET = "FI does not try studies with no simulation yet"
GOALS_NOT_READ = "FI does not read the study's goals in a trial yet."
SMALL = ".fi/small_run.json"
PILOT = ".fi/pilot_check.json"
ESTIMATE = ".fi/run_estimate.json"
MOVES = ".fi/design_moves.json"


def asked(config: Any) -> bool:
    """Whether this quest tries its plan before the study (``pauses.plan: after_trial``). A study that does not ask
    behaves exactly as before: no trial, no stop, no model call."""
    return str(getattr(getattr(config, "pauses", None), "plan", "") or "") == MODE


# --- reading FI's records ----------------------------------------------------------------------------------------------


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite(value: Any) -> bool:
    """A finite number, or a mapping or list holding one (a number per value of what a run compares inside it)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return any(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_finite(v) for v in value)
    return False


def _settings(small: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in small.get("settings") or [] if isinstance(s, dict)]


def _names(items: list[str]) -> str:
    return ", ".join(f"`{n}`" for n in items)


def _check(cid: str, result: str, says: str, records: list[str], small_change: str = "") -> dict[str, Any]:
    out = {"id": cid, "result": result, "says": says, "records": records}
    if small_change:
        out["small_change"] = small_change
    return out


def _settings_count(n: int) -> str:
    return f"{n} setting{'s' if n != 1 else ''}"


def _first_sentence(text: str, limit: int = 400) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _pilot_stopped(pilot: dict[str, Any], kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    """The pilot's findings of these kinds that its repairs left (the pilot record's ``outcome: stopped``)."""
    if pilot.get("outcome") != "stopped":
        return []
    return [p for p in pilot.get("problem_items") or [] if isinstance(p, dict) and str(p.get("kind")) in kinds]


# --- the six checks ----------------------------------------------------------------------------------------------------


def check_runs(small: dict[str, Any], *, timing_failed: str = "") -> dict[str, Any]:
    """``runs``: from the small run. ``timing_failed``: a setting failed while FI timed the study and no repair was left
    (the sizing's own words)."""
    records = [SMALL]
    if timing_failed:
        return _check("runs", FAIL, "The simulation failed while FI timed the study, and the repairs FI may make are "
                      f"used up: {_first_sentence(timing_failed)}.", records)
    if not small:
        return _check("runs", NOT_SHOWN, "FI did not run the simulation on a small case: it does not give FI one function "
                      "to call for each trial.", records)
    settings = _settings(small)
    if small.get("crashed"):
        return _check("runs", FAIL, "The simulation did not run to the end at the settings FI tried, after the repairs "
                      f"FI may make: {_first_sentence(small.get('says'))}.", records)
    trials = sum(int(s.get("planned") or 0) for s in settings)
    failed = sum(int(s.get("failed") or 0) for s in settings)
    if any(s.get("timed_out") for s in settings):
        late = sum(1 for s in settings if s.get("timed_out"))
        return _check("runs", NOT_SHOWN, f"The small run's time ran out in {late} of the {_settings_count(len(settings))} "
                      "FI tried, so FI did not see every one run to the end.", records)
    return _check("runs", PASS, f"The simulation ran to the end at the {_settings_count(len(settings))} FI tried "
                  f"({trials} trial{'s' if trials != 1 else ''}" + (f"; {failed} failed" if failed else "") + ").", records)


def _declared_numbers(protocol: dict[str, Any]) -> list[str]:
    return [str(m.get("id")) for m in protocol.get("metrics") or [] if isinstance(m, dict) and str(m.get("id") or "").strip()]


def _values_of(setting: dict[str, Any], name: str) -> list[Any] | None:
    from . import metric_spec as _metric_spec

    values = _dict(setting.get("values"))
    for form in _metric_spec.result_forms(name):
        if form in values:
            got = values[form]
            return got if isinstance(got, list) else [got]
    return None


def check_numbers(protocol: dict[str, Any], small: dict[str, Any], pilot: dict[str, Any]) -> dict[str, Any]:
    """``numbers``: each declared number counted from the small run's own values. Fail only for what the pilot found and its
    repairs left (no finite value in any pilot trial, or proportions all stuck at one bound)."""
    from . import small_run as _small

    records = [PILOT, SMALL]
    left = _pilot_stopped(pilot, ("undefined", "stuck"))
    if left:
        return _check("numbers", FAIL, "After the repairs FI may make, the pilot still found: "
                      + "; ".join(_first_sentence(p.get("says"), 300) for p in left[:3]) + ".", records)
    settings = _settings(small)
    if not settings:
        return _check("numbers", NOT_SHOWN, "FI did not run the simulation on a small case, so it saw no number.", records)
    notes: list[str] = []
    none_finite: list[str] = []
    seen = 0
    for name in _declared_numbers(protocol):
        per = [v for v in (_values_of(s, name) for s in settings) if v is not None]
        if not per:
            continue  # never returned: the ``results`` check, or a number the analysis computes afterwards
        seen += 1
        bad = sum(1 for v in per if not _finite(v))
        if bad == len(per):
            none_finite.append(name)
        elif bad:
            notes.append(f"`{name}`: not finite at {bad} of {_settings_count(len(per))}")
    analysis = _dict(small.get("analysis"))
    no_value = [str(m.get("id")) for m in analysis.get("no_value") or [] if isinstance(m, dict) and m.get("id")]
    block = protocol.get("optimisation") if isinstance(protocol.get("optimisation"), dict) else None
    baseline = [f.quantity for f in _small.judge_baseline(block, small) if not f.send and not f.fact]
    extra = "".join(f" {n}." for n in notes)
    if none_finite or no_value or baseline:
        parts = []
        if none_finite:
            parts.append(f"{_names(none_finite)} had no finite value at any of the {_settings_count(len(settings))} FI tried")
        if no_value:
            parts.append(f"FI's analysis of the small run gives no value of {_names(no_value)}")
        if baseline:
            parts.append(f"{_names(baseline)} is not finite at the search's starting design (other designs may differ)")
        return _check("numbers", NOT_SHOWN, "; ".join(parts) + f"; a few trials can explain this, so it is not shown "
                      "either way." + extra, records)
    if not seen:
        return _check("numbers", NOT_SHOWN, "None of the numbers the plan declares came back from the small run's trials "
                      "(they may be computed afterwards by the analysis).", records)
    return _check("numbers", PASS, f"Every number the plan declares had a finite value at one or more of the "
                  f"{_settings_count(len(settings))} FI tried." + extra, records)


def _exact_faults(protocol: dict[str, Any], small: dict[str, Any], *, docker: bool) -> tuple[list[Any], list[Any]]:
    """``(exact faults left, faults a small sample can explain)`` of the small run, from its record alone."""
    from . import pipeline_gate as _gate
    from . import small_run as _small

    block = protocol.get("optimisation") if isinstance(protocol.get("optimisation"), dict) else None
    faults = list(_small.judge_baseline(block, small)) if block is not None else []
    named = {f.quantity for f in faults if f.send}
    faults += [f for f in _small.shape_faults(protocol, small) if f.quantity not in named]
    named |= {f.quantity for f in faults if f.send}
    faults += [f for f in _small.missing_outputs(protocol, small) if f.quantity not in named]
    kept = _dict(small.get("analysis"))
    if kept:
        got = _gate.GateResult(**{k: v for k, v in kept.items() if k in _gate.GateResult.__dataclass_fields__})
        split = _small.analysis_split(got, docker=docker)
        if split is not None:
            faults.append(split)
    return [f for f in faults if f.send], [f for f in faults if not f.send and f.kind == _small.ANALYSIS]


def check_results(protocol: dict[str, Any], small: dict[str, Any], pilot: dict[str, Any], *,
                  docker: bool = False) -> dict[str, Any]:
    """``results``: an exact fault of the small run still there (after its one send-back), or the pilot's finding that
    the trial function returns a mean of many trials, fails it."""
    from . import small_run as _small

    records = [SMALL]
    aggregate = _pilot_stopped(pilot, ("aggregate",))
    if aggregate:
        return _check("results", FAIL, "After the repairs FI may make, the pilot still found: "
                      + _first_sentence(aggregate[0].get("says"), 300) + ".", [PILOT, *records])
    if not _settings(small):
        return _check("results", NOT_SHOWN, "FI did not run the simulation on a small case, so it saw no result.", records)
    exact, maybe = _exact_faults(protocol, small, docker=docker)
    if exact:
        sent = set(_small.sent(small)["faults"])
        return _check("results", FAIL, " ".join(_first_sentence(f.why, 300) for f in exact[:3])
                      + (" It was sent back to be repaired once and is still there." if any(f.key in sent for f in exact)
                         else " No send-back of the small run was left."), records)
    if maybe:
        return _check("results", NOT_SHOWN, _first_sentence(maybe[0].why, 300), records)
    if not _small._whole(small):
        return _check("results", NOT_SHOWN, "Not every trial of the small run ran to the end, so a result no trial "
                      "returned is not a fact either way.", records)
    return _check("results", PASS, "Every result the plan asks for came back, in the form it declares.", records)


def _compared(protocol: dict[str, Any]) -> list[str]:
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    return [str(k) for k, v in grid.items() if isinstance(v, list) and len(v) >= 2]


def check_moves(protocol: dict[str, Any], small: dict[str, Any], pilot: dict[str, Any], moves: dict[str, Any], *,
                unread: list[str] | None = None, conditional: list[str] | None = None) -> dict[str, Any]:
    """``moves``: ``unread`` is FI's finding at the end of the trial that a setting the study compares was asked for by
    nothing and the code's text never looks it up (still there after its repair); ``conditional``: one the code reads only
    under a condition the small run did not reach."""
    records = [SMALL, PILOT, MOVES]
    if unread:
        return _check("moves", FAIL, f"Nothing in the code asked for {_names(unread)}, a setting the study compares, and "
                      "its text never looks it up; it was sent back to be repaired and is still so.", records)
    flat = _pilot_stopped(pilot, ("flat", "constant"))
    if flat:
        return _check("moves", FAIL, "After the repairs FI may make, the pilot still found: "
                      + "; ".join(_first_sentence(p.get("says"), 300) for p in flat[:2]) + ".", records)
    no_change = [str(v.get("name")) for v in moves.get("variables") or []
                 if isinstance(v, dict) and v.get("verdict") == "no_change"]
    compared = _compared(protocol)
    if not compared and not moves.get("variables"):
        return _check("moves", NOT_APPLICABLE, "The study compares no setting.", records)
    reads = _dict(small.get("reads"))
    if conditional:
        return _check("moves", NOT_SHOWN, f"The code reads {_names(conditional)} only under a condition the small run did "
                      "not reach, so FI did not see it asked for.", records)
    if no_change:
        return _check("moves", NOT_SHOWN, f"Moving {_names(no_change)} to the two ends changed nothing FI could see; that "
                      "does not show the simulation ignores it.", records)
    if compared and (not small or not reads.get("measurable", False)):
        why = str(reads.get("why") or "FI did not run the simulation on a small case")
        return _check("moves", NOT_SHOWN, f"FI could not see which settings the code asked for ({why}).", records)
    said = ("Each setting the study compares was asked for by the code at the settings FI tried" if compared
            else "Each design variable FI moved changed the objective")
    if pilot.get("outcome") == "passed" and compared:
        said += ", and the pilot found no result that stays the same across them"
    return _check("moves", PASS, said + ".", records)


def check_time(estimate: dict[str, Any]) -> dict[str, Any]:
    """``time``: FI's timing record. Fail for the sizing's stop, a study FI would run only up to the limit, or one over the
    topic's own limit even at the fewest runs; not shown when it could not be timed or only a lower bound under the limit
    was."""
    from . import run_estimate as _estimate

    records = [ESTIMATE]
    if not estimate or estimate.get("fits") is None:
        return _check("time", NOT_SHOWN, "FI could not time the study before it runs.", records)
    limit = float(estimate.get("limit_s") or 0.0)
    took = float(estimate.get("estimate_s") or 0.0)
    allowed = _estimate.plain_duration(limit) if limit else "the time allowed"
    if estimate.get("stopped"):
        return _check("time", FAIL, f"The whole study does not fit the time allowed: {_first_sentence(estimate['stopped'])}.",
                      records)
    cut = _dict(estimate.get("truncated"))
    if cut:
        fit = _dict(cut.get("fit"))
        reach = (f"; about {int(fit['settings_that_fit']):,} of its {int(fit.get('settings') or cut.get('settings') or 0):,} "
                 "settings fit" if fit.get("settings_that_fit") else "")
        return _check("time", FAIL, f"The whole study does not fit the {allowed} allowed: it would take "
                      f"{'at least ' if cut.get('at_least') else 'about '}"
                      f"{_estimate.plain_duration(float(cut.get('estimate_s') or took))}{reach}; FI would run its settings "
                      "in the plan's order only up to the limit.", records)
    reduced = _dict(estimate.get("reduced"))
    over = _dict(estimate.get("over_topic_limit")) or (reduced if reduced.get("over_topic_limit") else {})
    change = ""
    if reduced.get("from") and reduced.get("to"):
        change = (f"fewer runs per setting: {_dict(reduced['from']).get('runs_per_setting')} -> "
                  f"{_dict(reduced['to']).get('runs_per_setting')}, to fit {allowed}")
    if over or estimate.get("fits") is False:
        return _check("time", FAIL, f"The whole study does not fit the {allowed} allowed: about "
                      f"{_estimate.plain_duration(took)} even with the fewest runs per setting the plan allows.", records,
                      change)
    if estimate.get("at_least"):
        return _check("time", NOT_SHOWN, f"FI timed only part of the study: at least {_estimate.plain_duration(took)}, "
                      f"under the {allowed} allowed; the rest was not timed.", records, change)
    said = f"The whole study fits the time allowed: about {_estimate.plain_duration(took)} of the {allowed}"
    if change:
        said += f", with {_dict(reduced['to']).get('runs_per_setting')} runs per setting instead of " \
                f"{_dict(reduced['from']).get('runs_per_setting')}"
    return _check("time", PASS, said + ".", records, change)


def supplied_files(quest_root: Path | str) -> int:
    """The files a person or FI put in ``inputs/`` for the study (papers and FI's own value file not counted)."""
    base = Path(quest_root) / "inputs"
    if not base.is_dir():
        return 0
    n = 0
    for path in base.rglob("*"):
        rel = path.relative_to(base)
        if path.is_file() and rel.parts[0] != "papers" and Path("inputs", *rel.parts) != STUDY_INPUTS:
            n += 1
    return n


def check_data(quest_root: Path | str) -> dict[str, Any]:
    """``data``: the files the study was given. (Results taken from another study are declared and checked by a later
    part of this design; with none declared, nothing of them is said.)"""
    n = supplied_files(quest_root)
    if not n:
        return _check("data", NOT_APPLICABLE, "The study reads no supplied file.", ["inputs/"])
    return _check("data", PASS, f"The study's supplied files are there: {n} file{'s' if n != 1 else ''} in inputs/.",
                  ["inputs/"])


# --- the verdict and the record -----------------------------------------------------------------------------------------


def goals_not_read() -> dict[str, Any]:
    """The goals part of the record while FI does not read the goals in a trial: could not be judged, said."""
    return {"state": "could_not_judge", "by_model": None, "items": [], "why": GOALS_NOT_READ}


def verdict(checks: list[dict[str, Any]], goals: dict[str, Any]) -> str:
    results = [str(c.get("result")) for c in checks]
    if FAIL in results:
        return NOT_FEASIBLE
    if all(r in (PASS, NOT_APPLICABLE) for r in results) and goals.get("state") == "answered":
        return FEASIBLE
    return UNSURE


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text[:2] != text[:2].upper() else text


def line(checks: list[dict[str, Any]], goals: dict[str, Any], said: str) -> str:
    """The record's one plain sentence."""
    if said == NOT_FEASIBLE:
        failed = next(c for c in checks if c.get("result") == FAIL)
        return "Not feasible: " + _lower_first(str(failed.get("says") or "")).rstrip(".") + "."
    if said == FEASIBLE:
        return "Feasible: every check FI made passed and every goal is answered."
    unknown = [_WHAT[str(c["id"])] for c in checks if c.get("result") == NOT_SHOWN and str(c.get("id")) in _WHAT]
    parts = ["nothing FI checked failed"]
    if unknown:
        parts.append("FI could not show " + (", ".join(unknown[:-1]) + " or " + unknown[-1] if len(unknown) > 1
                                              else unknown[0]))
    if goals.get("state") != "answered":
        parts.append(_lower_first(str(goals.get("why") or "not every goal is answered").rstrip(".")))
    return "Unsure: " + "; ".join(parts) + "."


def inputs_sha256(quest_root: Path | str) -> str:
    """The hash of the values taken from other studies (``inputs/study_inputs.json``), ``""`` when there is none."""
    path = Path(quest_root) / STUDY_INPUTS
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def spent(quest_root: Path | str, since: float | None = None) -> dict[str, Any]:
    """What the quest spent up to now (``.fi/cost.jsonl``): seconds from its first model call (or ``since``), calls,
    tokens."""
    rows: list[dict[str, Any]] = []
    try:
        for raw in (Path(quest_root) / ".fi" / "cost.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except OSError:
        pass
    stamps = [float(r["ts"]) for r in rows if isinstance(r.get("ts"), (int, float))]
    start = since if since is not None else (min(stamps) if stamps else None)
    tokens = sum(int(_dict(r.get("usage")).get("total_tokens") or 0) for r in rows)
    return {"seconds": round(time.time() - start) if start else 0, "calls": len(rows), "tokens": tokens}


def build(quest_root: Path | str, protocol: dict[str, Any] | None, *, quest_id: str, plan_sha256: str, code_version: str,
          docker: bool = False, timing_failed: str = "", unread: list[str] | None = None,
          conditional: list[str] | None = None, stopped: str = "", stopped_kind: str = "") -> dict[str, Any]:
    """The record of one trial, from FI's records on disk and the facts the engine found at its end. ``stopped``: the
    timing's stop, taken as a record (``stopped_kind`` ``too_long`` for the time; anything else is a fault its records
    name). A stop that no check's record shows fails ``time`` (too long) or ``runs`` with its own words."""
    root = Path(quest_root)
    protocol = protocol if isinstance(protocol, dict) else {}
    small = _dict(_json(root / SMALL))
    pilot = _dict(_json(root / PILOT))
    estimate = _dict(_json(root / ESTIMATE))
    moves = _dict(_json(root / MOVES))
    checks = [check_runs(small, timing_failed=timing_failed), check_numbers(protocol, small, pilot),
              check_results(protocol, small, pilot, docker=docker),
              check_moves(protocol, small, pilot, moves, unread=unread, conditional=conditional),
              check_time(estimate), check_data(root)]
    if stopped and not any(c["result"] == FAIL for c in checks):
        cid = "time" if stopped_kind == "too_long" else "runs"
        checks = [_check(cid, FAIL, f"FI's check before the study stopped it: {_first_sentence(stopped)}.",
                         c["records"]) if c["id"] == cid else c for c in checks]
    return _record(root, checks, quest_id=quest_id, plan_sha256=plan_sha256, code_version=code_version,
                   estimate=estimate)


def not_tried(quest_root: Path | str, *, quest_id: str, plan_sha256: str, why: str, data_line: str) -> dict[str, Any]:
    """The record of a study FI does not try: every check ``not_applicable`` with ``why``, one line of what it was given,
    verdict ``unsure``, so the caller still gets one record and decides."""
    checks = [_check(c, NOT_APPLICABLE, why + ".", []) for c in CHECKS if c != "data"]
    checks.append(_check("data", NOT_APPLICABLE, data_line, ["inputs/", "data/"]))
    record = _record(Path(quest_root), checks, quest_id=quest_id, plan_sha256=plan_sha256, code_version="", estimate={})
    record.update(verdict=UNSURE, line=f"Unsure: {why}.")
    return record


def _record(root: Path, checks: list[dict[str, Any]], *, quest_id: str, plan_sha256: str, code_version: str,
            estimate: dict[str, Any]) -> dict[str, Any]:
    goals = goals_not_read()
    said = verdict(checks, goals)
    small_changes = [str(c["small_change"]) for c in checks if c.get("small_change")]
    big_changes = [str(c["says"]) for c in checks if c.get("result") == FAIL]
    return {
        "schema": SCHEMA, "quest_id": quest_id, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plan_sha256": plan_sha256, "code_version": code_version, "inputs_sha256": inputs_sha256(root),
        "verdict": said, "line": line(checks, goals, said), "checks": checks, "goals": goals,
        "stand_ins": [], "small_changes": small_changes, "big_changes": big_changes,
        "estimate": {"seconds": float(estimate.get("estimate_s") or 0.0), "allowed_s": float(estimate.get("limit_s") or 0.0),
                     "at_least": bool(estimate.get("at_least")), "other_studies_running": None},
        "changed_after_trial": None, "cost": spent(root), "versions": [],
    }


# --- writing, the marker and the resume ---------------------------------------------------------------------------------


def _write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read(quest_root: Path | str) -> dict[str, Any]:
    return _dict(_json(Path(quest_root) / RECORD))


def complete(record: dict[str, Any]) -> bool:
    """A record a resume can rely on: its schema, a verdict and every check."""
    ids = {str(c.get("id")) for c in record.get("checks") or [] if isinstance(c, dict)}
    return record.get("schema") == SCHEMA and record.get("verdict") in (FEASIBLE, NOT_FEASIBLE, UNSURE) \
        and set(CHECKS) <= ids


def write(quest_root: Path | str, record: dict[str, Any], tried: dict[str, Any] | None) -> Path:
    """Step (1) of the stop: the record, written to a temporary file and renamed. An older record goes under
    ``.fi/feasibility/``; the protocol that was tried is kept beside it for the freeze's comparison."""
    root = Path(quest_root)
    path = root / RECORD
    old = read(root)
    if old:
        n = len(list((root / OLDER).glob("record_*.json"))) + 1
        _write_atomic(root / OLDER / f"record_{n:03d}.json", old)
    _write_atomic(root / TRIED, {"plan_sha256": record.get("plan_sha256"), "protocol": tried or {}})
    _write_atomic(path, record)
    return path


def write_marker(quest_root: Path | str, record: dict[str, Any]) -> None:
    """Step (2) of the stop: the marker holds what the trial was made on."""
    _write_atomic(Path(quest_root) / MARKER, {"plan_sha256": record.get("plan_sha256"),
                                               "code_version": record.get("code_version"),
                                               "inputs_sha256": record.get("inputs_sha256"), "went_on": False})


def read_marker(quest_root: Path | str) -> dict[str, Any] | None:
    path = Path(quest_root) / MARKER
    if not path.is_file():
        return None
    return _dict(_json(path))


AGAIN, DONE = "again", "done"


def on_resume(quest_root: Path | str, *, plan_sha256: str, code_version: str) -> str:
    """What the execute step does when it runs again with the marker on disk, in the design's order: the record missing
    or partial, or the plan or the code changed since the trial -> :data:`AGAIN` (the trial is made again; its parts are
    reused by their keys); otherwise :data:`DONE` (the full checks and the study). Once the study has gone on after the
    trial (``went_on``), the trial is never made again: a repair during the run is not a person's change of the plan.

    Results taken from other studies (a stand-in still in place; a value changed since the trial) are a later part of this
    design: until then a changed value file makes the trial again."""
    marker = read_marker(quest_root)
    if marker is None:
        return AGAIN
    if marker.get("went_on"):
        return DONE
    record = read(quest_root)
    if not complete(record) or marker.get("plan_sha256") != plan_sha256 or marker.get("code_version") != code_version:
        return AGAIN
    if str(marker.get("inputs_sha256") or "") != inputs_sha256(quest_root):
        return AGAIN
    return DONE


def went_on(quest_root: Path | str) -> None:
    """The study goes on after the trial: the marker says so, so the run's own repairs never stop it at the trial again."""
    marker = read_marker(quest_root)
    if marker is not None and not marker.get("went_on"):
        _write_atomic(Path(quest_root) / MARKER, {**marker, "went_on": True})


def note_approval(quest_root: Path | str, who: str) -> bool:
    """``--approve-as <who>`` on the resume after a trial: kept for the freeze, for the plan that was tried. Nothing when
    the quest did not stop after a trial, or the study already went on."""
    who = " ".join(str(who or "").split())
    marker = read_marker(quest_root)
    if not who or marker is None or marker.get("went_on"):
        return False
    _write_atomic(Path(quest_root) / APPROVAL, {"who": who, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                 "plan_sha256": marker.get("plan_sha256")})
    return True


def approval(quest_root: Path | str) -> dict[str, Any] | None:
    """The person's approval of the tried plan, only when it was given for the plan the current marker holds."""
    got = _dict(_json(Path(quest_root) / APPROVAL))
    marker = read_marker(quest_root) or {}
    if got.get("who") and got.get("plan_sha256") and got.get("plan_sha256") == marker.get("plan_sha256"):
        return got
    return None


def _changed_by(key: str) -> str:
    """Which of FI's checks makes a change at this place of the protocol."""
    head = key.split(".", 1)[0].split("[", 1)[0]
    if "source" in key or "reference" in key:
        return "a source FI filled in for a check"
    if head == "oracles":
        return "FI's known-answer checks"
    if head == "runs_per_setting":
        return "FI's timing (fewer runs per setting)"
    if head == "optimisation":
        return "FI's timing of the search"
    return "FI's checks before the run"


def changed_after_trial(quest_root: Path | str, frozen: dict[str, Any] | None) -> list[dict[str, str]] | None:
    """Each difference between the protocol FI tried and the one it freezes, with the check that made it; ``None`` when
    no protocol was tried."""
    from . import frozen_protocol as _frozen

    tried = _dict(_json(Path(quest_root) / TRIED))
    if not tried:
        return None
    return [{"change": diff, "by": _changed_by(diff.split(":", 1)[0])}
            for diff in _frozen.diff(tried.get("protocol") or {}, frozen or {})]


def record_changes(quest_root: Path | str, changes: list[dict[str, str]] | None) -> None:
    """At the freeze: the record says what changed after the trial (an empty list: nothing)."""
    record = read(quest_root)
    if not record or changes is None:
        return
    try:
        _write_atomic(Path(quest_root) / RECORD, {**record, "changed_after_trial": changes})
    except OSError:
        pass


def approved_by(quest_root: Path | str, who: str, plan_sha256: str, changes: list[dict[str, str]] | None) -> str:
    """The freeze record's ``approved_by`` for a person who approved the tried plan, naming every change made after it."""
    text = f"human: {who} approved the tried plan (sha256 {str(plan_sha256)[:12]}; pauses.plan={MODE})"
    if changes:
        text += (f"; changes FI's checks made after the trial, not seen by {who}: "
                 + "; ".join(f"{c['change']} ({c['by']})" for c in changes[:12])
                 + (f"; and {len(changes) - 12} more" if len(changes) > 12 else ""))
    return text


# --- what a person sees ---------------------------------------------------------------------------------------------------


def resume_command(quest_root: Path | str, quest_id: str) -> str:
    return (f"python launch.py --config {(Path(quest_root) / 'config.yaml').as_posix()} --resume {quest_id} "
            "--approve-as <you>")


def stop_card(record: dict[str, Any], quest_root: Path | str, quest_id: str) -> tuple[str, list[str], str, list[str]]:
    """``(headline, steps, recommended, alternatives)`` of the stop after a trial."""
    word = {FEASIBLE: "it looks feasible", NOT_FEASIBLE: "it does not look feasible",
            UNSURE: "FI is not sure it is feasible"}.get(str(record.get("verdict")), "see the record")
    headline = f"FI tried the plan on a small case: {word}"
    steps = [f"What FI found: {record.get('line')}",
             *[f"{c['id']}: {c['says']}" for c in record.get("checks") or [] if isinstance(c, dict)],
             f"The record is `{RECORD.as_posix()}` in the quest folder ({Path(quest_root).as_posix()}); QUALITY.md has the "
             "same in its Trial section.",
             f"To run the tried study: `{resume_command(quest_root, quest_id)}` (without `--approve-as`, the record of the "
             "frozen plan says nobody approved it).",
             "To change the plan instead: edit `plan.md`, or ask for a change with `--revise-plan \"what to change\"`, "
             "then resume: FI tries the changed plan again before it runs."]
    recommended = ("Run the tried study." if record.get("verdict") == FEASIBLE else
                   "Change the plan (`--revise-plan`), then resume: FI tries it again."
                   if record.get("verdict") == NOT_FEASIBLE else
                   "Read what FI could not show, then run the tried study or change the plan.")
    alternatives = ["Run the tried study as it is.", "Change the plan and let FI try it again."]
    return headline, steps, recommended, alternatives


def summary(quest_root: Path | str) -> dict[str, Any] | None:
    """For the quality report: the verdict, the line, each check's result and sentence, and what changed after the
    trial; ``None`` when the quest made no trial."""
    record = read(quest_root)
    if not record:
        return None
    return {"verdict": record.get("verdict"), "line": record.get("line"),
            "checks": [{"id": c.get("id"), "result": c.get("result"), "says": c.get("says")}
                       for c in record.get("checks") or [] if isinstance(c, dict)],
            "goals": _dict(record.get("goals")).get("state"),
            "small_changes": list(record.get("small_changes") or []),
            "changed_after_trial": record.get("changed_after_trial")}


def report_lines(summary_: dict[str, Any]) -> list[str]:
    """QUALITY.md's Trial section: plain sentences."""
    lines = ["Before the study, FI tried the plan on a small case (`pauses.plan: after_trial`).", "",
             str(summary_.get("line") or ""), ""]
    word = {PASS: "passed", FAIL: "failed", NOT_SHOWN: "not shown", NOT_APPLICABLE: "does not apply"}
    lines += [f"- {c.get('id')} ({word.get(str(c.get('result')), c.get('result'))}): {c.get('says')}"
              for c in summary_.get("checks") or []]
    if summary_.get("small_changes"):
        lines += ["", "Changes FI made itself: " + "; ".join(summary_["small_changes"]) + "."]
    changed = summary_.get("changed_after_trial")
    if isinstance(changed, list):
        lines += ["", "Changes FI's checks made to the plan after the trial: "
                  + ("; ".join(f"{c.get('change')} ({c.get('by')})" for c in changed[:12]) + "." if changed else "none.")]
    return lines
