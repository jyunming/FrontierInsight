"""One record of what a study costs, kept apart term by term (``<quest_root>/.fi/cost_model.json``).

FI times a few real settings before a study runs (``core/run_estimate.py``) and used to keep only their total. A total
cannot be reused: summed trial time is a median 0.39 of a grid study's wall time, because each setting also pays a fixed
cost (the process, the loading), the first run pays more, some work is paid once and shared, and settings that run at the
same time slow each other. This module keeps those terms apart, each with where it came from (``basis``), and holds the
one rule that turns them into a search size:

* :func:`keep_measured` stores what the sizing measured (a pure addition: nothing reads it back yet);
* :func:`from_legacy_record` builds the same record for a quest that was run before this file existed, from its other
  files, so earlier quests can be scored against the time they took with no re-run;
* :func:`actuals_from_log` reads the times a quest really took from its own ``run.log``;
* :func:`search_plan` turns a time allowed into the number of evaluations a search may use, the deadline it must stop at
  and the time kept for FI's check at finer settings. With no measured cost per evaluation it says so (``evaluations`` is
  ``None``); it never guesses one.
"""
from __future__ import annotations

import datetime as _dt
import json
import hashlib
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RECORD = ".fi/cost_model.json"
VERSION = 1

#: Where a term came from. Only ``measured`` terms are timings FI took before the study ran.
MEASURED, CLAIMED, DERIVED, DEFAULT, UNKNOWN = "measured", "claimed by the plan", "derived from actuals", "default", "not recorded"

#: Time kept for the analysis run after a study: the 90th percentile of 80 recorded analysis runs (median 13 s).
ANALYSIS_RESERVE_S = 71.0


@dataclass
class Term:
    """One cost: seconds (``None`` when not known), where it came from, and whether it is only a lower bound."""

    seconds: float | None = None
    basis: str = UNKNOWN
    at_least: bool = False


@dataclass
class CostModel:
    source: str = "live"  # "live" (kept by the sizing) or "legacy" (rebuilt from an earlier quest's files)
    version: int = VERSION
    key: str = ""
    limit_s: float | None = None
    #: per_trial, fixed, first_run, paid_once (grid) ; per_evaluation, finer_check_unit (search) ; scan, analysis
    terms: dict[str, Term] = field(default_factory=dict)
    at_once: int = 1
    #: How many times longer a setting takes beside ``at_once - 1`` others (``None``: not measured).
    slowdown: float | None = None
    cells: int = 0
    trials: int = 0
    probes: list[dict[str, Any]] = field(default_factory=list)
    #: What the plan asked of a search: ``{"evaluations": N, "check_runs": N, "scan": N}``.
    plan: dict[str, Any] = field(default_factory=dict)
    #: The design (the plan's search block and the simulation) the search terms were measured on; terms of another one are not used.
    search_key: str = ""
    #: Every timing of one new design of a search (seconds): one design timed twice has differed by 3.8x between runs.
    evaluation_timings: list[float] = field(default_factory=list)
    #: The estimate FI printed (padded) for the last attempt, seconds.
    printed_s: float | None = None
    actuals: list[dict[str, Any]] = field(default_factory=list)

    def term(self, name: str) -> Term:
        return self.terms.get(name) or Term()

    def seconds(self, name: str) -> float | None:
        """The term's seconds; ``None`` when it was not measured or the measurement is not a finite number."""
        value = self.term(name).seconds
        return value if value is not None and math.isfinite(value) else None


# ---- the record -------------------------------------------------------------------------------------------------------


def _from_dict(raw: dict[str, Any]) -> CostModel:
    terms = {k: Term(**{f: v.get(f) for f in ("seconds", "basis", "at_least") if f in v}) for k, v in (raw.get("terms") or {}).items()
             if isinstance(v, dict)}
    model = CostModel(**{k: v for k, v in raw.items() if k in CostModel.__dataclass_fields__ and k != "terms"})
    model.terms = terms
    return model


def write(quest_root: Path | str, model: CostModel) -> bool:
    try:
        path = Path(quest_root) / RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(asdict(model), indent=1, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)  # a kill in the middle leaves the earlier record, never half of one
        return True
    except OSError:
        return False


def read(quest_root: Path | str) -> CostModel | None:
    try:
        raw = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
        return _from_dict(raw) if isinstance(raw, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _probe_row(p: Any) -> dict[str, Any]:
    return {"axis": None if p.axis is None else str(p.axis), "cell": {str(k): v for k, v in dict(p.cell).items()},
            "seconds": p.seconds, "at_least": bool(p.at_least), "fixed_s": p.fixed_s, "per_trial_s": p.per_trial_s,
            "first_s": p.first_s, "timed_trials": p.timed_trials}


_SEARCH_TERMS = ("per_evaluation", "check_unit", "finer_check_unit", "scan", "wall_per_evaluation")


def design_key(block: Any, simulate: Path | str | None = None) -> str:
    """Which design a search timing belongs to: the plan's search block and every file the simulation may read
    (:func:`core.code_layout.simulation_inputs`: ``simulate`` and the files beside and below it, a parameter file or a
    note among them, the analysis FI writes aside, line endings ignored). A redesign, or a change to one of them,
    changes it."""
    try:
        h = hashlib.sha256(json.dumps(block, sort_keys=True, default=str).encode("utf-8"))
        if simulate:
            from . import code_layout as _code_layout

            sim = Path(simulate)
            quest = sim.parent.parent if sim.parent.name == "code" else sim.parent
            files = {f: _code_layout.input_label(f, sim.parent) if f.is_relative_to(sim.parent)
                     else _code_layout.input_label(f, quest) for f in _code_layout.simulation_inputs(quest, sim.parent)}
            if sim not in files:  # the simulation itself is always part of its design
                files[sim] = sim.name
            for f, label in sorted(files.items(), key=lambda kv: kv[1]):
                h.update(label.encode("utf-8") + b"\0")
                h.update(f.read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()[:16]
    except Exception:  # noqa: BLE001 -- a key that cannot be made leaves the terms unkeyed, never changes a run
        return ""


def keep_measured(quest_root: Path | str, *, probes: list[Any] | None = None, est: Any = None, slowdown: float | None = None,
                  at_once: int | None = None, limit_s: float | None = None, key: str = "", per_evaluation_s: float | None = None,
                  per_evaluation_at_least: bool = False, check_unit_s: float | None = None, check_unit_basis: str = MEASURED,
                  finer_unit_s: float | None = None, scan_s: float | None = None, plan: dict[str, Any] | None = None,
                  design: str = "", finer_unit_basis: str = MEASURED, finer_unit_at_least: bool = False) -> bool:
    """Add what the sizing measured to the quest's record (made if there is none). Adds only; never raises, since a
    missing record must not change a run. ``design`` (:func:`design_key`) ties the search terms to the design they were
    measured on: terms kept for another design are dropped first. ``probes`` are :class:`core.run_estimate.Probe`; ``est`` an ``Estimate``."""
    try:
        model = read(quest_root) or CostModel()
        if design and model.search_key != design:
            for name in _SEARCH_TERMS:
                model.terms.pop(name, None)
            model.evaluation_timings, model.plan, model.search_key = [], {}, design
        if probes:
            model.probes = [_probe_row(p) for p in probes]
            base = next((p for p in probes if p.axis is None), None)
            if base is not None:
                if base.fixed_s is not None and base.per_trial_s is not None:
                    model.terms["fixed"] = Term(base.fixed_s, MEASURED)
                    model.terms["per_trial"] = Term(base.per_trial_s, MEASURED)
                if base.first_s is not None:
                    model.terms["first_run"] = Term(base.first_s, MEASURED)
        if est is not None:
            model.terms["paid_once"] = Term(float(est.once_s), MEASURED)
            model.cells, model.trials = int(est.cells), int(est.trials)
            model.at_once = int(est.at_once)
        if at_once:
            model.at_once = int(at_once)
        if slowdown is not None:
            model.slowdown = float(slowdown)
        if limit_s is not None:
            model.limit_s = float(limit_s)
        if key:
            model.key = key
        if per_evaluation_s:
            model.evaluation_timings.append(float(per_evaluation_s))
            model.terms["per_evaluation"] = Term(max(model.evaluation_timings), MEASURED, per_evaluation_at_least)
        if check_unit_s:
            model.terms["check_unit"] = Term(float(check_unit_s), check_unit_basis, check_unit_basis != MEASURED)
        if finer_unit_s:
            model.terms["finer_check_unit"] = Term(float(finer_unit_s), finer_unit_basis, finer_unit_at_least)
        if scan_s is not None:
            model.terms["scan"] = Term(float(scan_s), MEASURED)
        if plan:
            model.plan.update(plan)
        return write(quest_root, model)
    except Exception:  # noqa: BLE001 -- keeping the record must never change a run
        return False


# ---- what a quest really took -----------------------------------------------------------------------------------------

_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \[(\w+)\] (.*)$")
_EST = re.compile(r"FI estimates (?:the )?(experiment|search for the best design)(?: takes| would take) (about|at least) "
                  r"(\d+) (seconds?|minutes|h|days)(?: (\d+) (min|h))?")
_UNIT = {"second": 1, "seconds": 1, "minutes": 60, "h": 3600, "days": 86400}


def _when(text: str) -> float:
    return _dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").timestamp()


def actuals_from_log(quest_root: Path | str) -> list[dict[str, Any]]:
    """One row per execute of the quest, from ``.fi/run.log``: the estimate FI printed (padded, seconds), the study's
    time, the analysis run's, the execute's total and its return code; for a search also the scan's, the search's and the
    check's time, the evaluations done and allowed, how many ran at the same time and how long the check's first run took.
    Times are log timestamps (whole seconds)."""
    try:
        lines = (Path(quest_root) / ".fi" / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    cur: dict[str, Any] = {}
    for line in lines:
        m = _TS.match(line)
        if not m:
            continue
        ts, _lvl, msg = m.groups()
        if "[execute]" in msg:
            e = _EST.search(msg)
            if e:
                n, u, n2, u2 = e.group(3), e.group(4), e.group(5), e.group(6)
                secs = int(n) * _UNIT[u] + (int(n2) * {"min": 60, "h": 3600}[u2] if n2 else 0)
                if cur.get("kind") != "search" or e.group(1) != "experiment":  # a search's own estimate is the one kept
                    cur.update(kind="grid" if e.group(1) == "experiment" else "search", qualifier=e.group(2),
                               printed_s=float(secs), printed_at=_when(ts))
            r = re.search(r"FI ran (\d+) trial\(s\) in (\d+) cell", msg)
            if r:
                cur.update(study_end=_when(ts), trials=int(r.group(1)), cells=int(r.group(2)))
            c = re.search(r"rc=(-?\d+) duration=([\d.]+)s", msg)
            if c:
                end = _when(ts)
                cur.update(rc=int(c.group(1)), exec_s=float(c.group(2)), ended=end)
                cur["started"] = end - cur["exec_s"]
                if cur.get("search_end") is not None:
                    # a search's analysis starts when its check ends (no check logged: when the search ends); a check that
                    # started and never finished leaves the analysis time unknown
                    after = cur.get("check_end")
                    if after is None and cur.get("check_start") is None:
                        after = cur["search_end"]
                    if after is not None:
                        cur["analysis_s"] = round(end - after, 1)
                elif cur.get("study_end") is not None:
                    cur["analysis_s"] = round(end - cur["study_end"], 1)
                    cur["study_s"] = round(cur["study_end"] - (cur.get("printed_at") or cur["started"]), 1)
                rows.append(cur)
                cur = {}
        elif "[optimise]" in msg:
            s = re.search(r"the coarse scan ran (\d+) setting", msg)
            if s:
                cur.update(scan_end=_when(ts), scan_settings=int(s.group(1)))
            d = re.search(r"(\d+) evaluation\(s\) of the (\d+) allowed", msg)
            if d and "still searching" not in msg:
                cur.update(search_end=_when(ts), evaluations=int(d.group(1)), allowed=int(d.group(2)))
                cur["study_end"] = cur["search_end"]
            w = re.search(r"still searching: .*\((\d+) at the same time\)", msg)
            if w:
                cur["search_at_once"] = int(w.group(1))
            if "checking the best design at finer" in msg:
                cur["check_start"] = _when(ts)
                p = re.search(r"at most (\d+) runs", msg)
                cur["check_planned"] = int(p.group(1)) if p else None
            f = re.search(r"checking at finer settings: 1 of at most (\d+) run", msg)
            if f and cur.get("check_start") is not None:
                cur["finer_first_run_s"] = round(_when(ts) - cur["check_start"], 1)
            k = re.search(r"the check ran the simulation (\d+) time", msg)
            if k:
                cur["check_runs_done"] = int(k.group(1))
                cur["check_end"] = _when(ts)
    if cur.get("search_end") is not None:  # a search whose execute never logged its end (the process was killed)
        # when it started is known only from the estimate line before it; without one the scan's time is not known either
        cur.update(rc=None, ended=cur.get("check_start") or cur["search_end"], partial=True, started=cur.get("printed_at"))
        cur["exec_s"] = cur["ended"] - cur["started"] if cur["started"] else None
        rows.append(cur)
    for row in rows:
        if row.get("search_end") is not None:
            if row.get("scan_end") is not None:
                if row.get("started") is not None:
                    row["scan_s"] = round(row["scan_end"] - row["started"], 1)
                row["search_s"] = round(row["search_end"] - row["scan_end"], 1)
                # a search the scan stopped after one evaluation has no rate
                row["search_rate_per_s"] = round(row["evaluations"] / row["search_s"], 4)                     if row["search_s"] > 0 and row["evaluations"] >= 50 else None
            row["kind"] = "search"
            row["check_s"] = round(row["ended"] - row["search_end"], 1)
    return rows


def _ledger_split(quest_root: Path) -> tuple[float, int, int]:
    """Summed trial seconds, trials and settings in ``raw/ledger.jsonl`` (what the trials themselves took)."""
    total, n, cells = 0.0, 0, set()
    try:
        for line in (quest_root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event") == "trial" and isinstance(d.get("duration_s"), (int, float)):
                total, n = total + float(d["duration_s"]), n + 1
                cells.add(d.get("cell"))
    except OSError:
        pass
    return total, n, len(cells)


def from_legacy_record(quest_root: Path | str) -> CostModel | None:
    """The record for a quest run before this file existed, from ``.fi/run_estimate.json``, ``.fi/run.log`` and
    ``raw/ledger.jsonl``. Terms the files do not hold are left ``not recorded`` (never a default that looks measured); the
    per-trial share of a grid is ``derived from actuals`` (what the trials took, not what the sizing measured)."""
    root = Path(quest_root)
    try:
        rec = json.loads((root / ".fi" / "run_estimate.json").read_text(encoding="utf-8"))
        rec = rec if isinstance(rec, dict) else {}
    except (OSError, ValueError):
        rec = {}
    actuals = actuals_from_log(root)
    if not rec and not actuals:
        return None
    cut = rec.get("truncated") if isinstance(rec.get("truncated"), dict) else {}
    model = CostModel(source="legacy", key=str(rec.get("key") or ""), limit_s=rec.get("limit_s"), at_once=int(rec.get("at_once") or 1),
                      cells=int(rec.get("cells") or cut.get("settings") or 0), trials=int(rec.get("trials") or 0),
                      printed_s=rec.get("estimate_s") or cut.get("estimate_s"), actuals=actuals)
    if rec.get("once_s") is not None:
        model.terms["paid_once"] = Term(float(rec["once_s"]), MEASURED)
    if cut:
        if cut.get("one_s"):
            model.terms["per_evaluation"] = Term(float(cut["one_s"]), MEASURED, bool(cut.get("at_least")))
        if cut.get("check_reserve_s"):
            model.terms["check_unit"] = Term(None, UNKNOWN)
            model.plan["check_reserve_s"] = float(cut["check_reserve_s"])
        model.plan.update({k: cut[k] for k in ("scan", "evaluations", "check_runs") if k in cut})
    total, n, cells = _ledger_split(root)
    study = next((a["study_s"] for a in reversed(actuals) if a.get("study_s") and a.get("kind") == "grid"), None)
    if n and study:
        model.terms["per_trial"] = Term(total / n, DERIVED)
        model.terms["fixed"] = Term(max(0.0, study - total) / max(1, cells or model.cells), DERIVED)
    seen = sorted(float(a["analysis_s"]) for a in actuals if a.get("analysis_s") is not None)
    if seen:  # the 90th percentile of the analysis runs seen, not the last one
        model.terms["analysis"] = Term(seen[max(0, math.ceil(0.9 * len(seen)) - 1)], DERIVED)
    for a in reversed(actuals):
        if a.get("search_rate_per_s"):
            at_once = int(a.get("search_at_once") or model.at_once or 1)
            model.at_once = max(model.at_once, at_once)
            # wall seconds per evaluation as the search really ran (already includes the slowdown and the at-once share)
            model.terms["wall_per_evaluation"] = Term(1.0 / a["search_rate_per_s"], DERIVED)
            if a.get("scan_s") is not None:
                model.terms["scan"] = Term(a["scan_s"], DERIVED)
            break
    for a in reversed(actuals):
        if a.get("finer_first_run_s"):
            model.terms["finer_check_unit"] = Term(a["finer_first_run_s"], DERIVED)
            break
    return model


def verdict(model: CostModel, limit_s: float | None = None) -> str:
    """What the record says of a study against the time allowed (``limit_s``, else the record's own): ``fits`` (the padded
    estimate is within it), ``run_to_limit`` (over it, but one setting fits: FI runs the plan's order up to the limit and
    says so) or ``stop`` (not even one setting fits). ``unknown`` when the record holds no estimate."""
    limit = float(limit_s if limit_s is not None else (model.limit_s or 0.0))
    total = model.printed_s
    if not total or limit <= 0:
        return "unknown"
    if total <= limit:
        return "fits"
    fixed, per_trial = model.seconds("fixed"), model.seconds("per_trial")
    one = fixed + per_trial if fixed is not None and per_trial is not None else (total / model.cells if model.cells else None)
    return "stop" if one is not None and one > limit else "run_to_limit"


# ---- what a settings-per-process change would give ---------------------------------------------------------------------


def seconds_with_settings_per_process(model: CostModel, settings: int, per_process: int) -> float | None:
    """The time of ``settings`` settings of one trial each when ``per_process`` of them share one process: every setting
    pays its per-trial time, and the fixed cost (process, loading) is paid once per process. ``None`` when the two terms
    are not known. Settings that run at the same time are not shared out here: the result is one after another."""
    fixed, per_trial = model.seconds("fixed"), model.seconds("per_trial")
    if fixed is None or per_trial is None or settings <= 0:
        return None
    processes = math.ceil(settings / max(1, int(per_process)))
    return settings * per_trial + processes * fixed


# ---- the size of a search ---------------------------------------------------------------------------------------------


@dataclass
class SearchPlan:
    #: How many evaluations fit; ``None`` when no cost per evaluation was measured (then ``why`` says so).
    evaluations: int | None
    #: Seconds from the start of the study at which the search must stop, so that the check and the analysis keep their time.
    deadline_s: float | None
    check_reserve_s: float
    check_runs: int
    analysis_reserve_s: float
    scan_s: float
    wall_per_evaluation_s: float | None
    basis: str
    why: str = ""


def check_wall_s(per_design_s: float, batches: list[int], *, at_once: int, slowdown: float | None) -> float:
    """Seconds FI's check at finer settings takes: ``batches`` are the designs of each request it makes (they run together,
    up to ``at_once`` at a time, in whole waves: a request that ends with a part-filled wave still takes a whole one), each
    ``per_design_s`` alone and as much slower beside the others as ``slowdown`` measured. A slowdown that was not measured is
    not assumed: the designs are then counted one after another. The one rule both the run's estimate
    (core/run_checks.py) and :func:`search_plan` price the check by."""
    designs = [max(0, int(n)) for n in batches]
    together = max(1, int(at_once or 1))
    if together > 1 and slowdown is not None:
        waves = sum(-(-n // together) for n in designs)
        return waves * per_design_s * max(1.0, float(slowdown))
    return sum(designs) * per_design_s


def search_plan(model: CostModel, limit_s: float, *, planned_evaluations: int | None = None, check_runs: int | None = None,
                scan_s: float | None = None, analysis_s: float = ANALYSIS_RESERVE_S, design: str = "",
                margin: float = 1.0) -> SearchPlan:
    """The evaluations a search may use in ``limit_s`` seconds:

    ``(limit - scan - check reserve - analysis reserve) / wall seconds per evaluation``

    where one evaluation costs what it took alone (``per_evaluation``) times the measured slowdown divided by how many run
    at once (a rate observed on the search itself, ``wall_per_evaluation``, wins), and the check reserve is the check's runs
    times one measured design at the finer settings (``finer_check_unit``; the search-design cost, flagged a lower bound,
    when that was not measured), at most half the time allowed. A slowdown that was not measured is not assumed: the check
    is then priced one run after another. Never more than ``planned_evaluations``. ``evaluations`` is ``None`` (and ``why``
    says what is missing) when nothing per-evaluation was measured, when the scan the plan has was not timed, or when the
    timings are of another ``design`` than the one asked for."""
    runs = int(check_runs if check_runs is not None else model.plan.get("check_runs") or 0)
    scan = scan_s if scan_s is not None else model.seconds("scan")
    scan_planned = int(model.plan.get("scan") or 0)
    wall = model.seconds("wall_per_evaluation")
    basis = "observed on the search itself"
    # Designs share the time only when they run at the same time (as core/run_checks.py `_size_the_scan` prices it) and
    # the slowdown beside each other was measured: an unmeasured slowdown is not assumed, so the check runs one after another.
    at_once = int(model.at_once or 1)
    share = model.slowdown / at_once if model.slowdown is not None and at_once > 1 else 1.0  # per evaluation

    def nothing(why: str) -> SearchPlan:
        return SearchPlan(None, None, 0.0, runs, analysis_s, float(scan or 0.0), None, UNKNOWN, why)

    if design and model.search_key != design:
        return nothing("the timings kept are of an earlier design of this study, so no number of evaluations is given")
    if wall is None:
        alone = model.seconds("per_evaluation")
        if alone is None:
            return nothing("no cost per evaluation was measured, so no number of evaluations is given")
        wall, basis = alone * share, model.term("per_evaluation").basis
    if scan is None:
        if scan_planned > 1:
            return nothing("the time of the coarse scan was not measured, so no number of evaluations is given")
        scan = 0.0
    finer = model.seconds("finer_check_unit")
    # A finer-settings unit is the time of one run of the check (a whole design's time over its ``check_trials`` runs); the
    # check is priced by the one rule the run's estimate uses (:func:`check_wall_s`) over the requests it makes.
    per_setting_runs = max(1, int(model.plan.get("runs_per_evaluation") or 1))
    unit = finer if finer is not None else (model.seconds("check_unit") or model.seconds("per_evaluation") or 0.0) / per_setting_runs
    trials = max(1, int(model.plan.get("check_trials") or 1))
    batches = [int(n) for n in model.plan.get("check_batches") or []]
    if not batches:  # a record from before the requests were kept: the runs as one request of single-run designs
        batches, trials = [runs], 1
    wanted = check_wall_s(unit * trials, batches, at_once=min(at_once, int(model.plan.get("check_at_once") or at_once)),
                          slowdown=model.slowdown) * max(1.0, float(margin))
    reserve = min(wanted, float(limit_s) / 2.0)  # as the run itself keeps for the check: never more than half the limit
    room = float(limit_s) - scan - reserve - analysis_s
    n = max(0, int(room / wall)) if wall > 0 else 0
    if planned_evaluations is not None:
        n = min(n, int(planned_evaluations))
    notes = ["the scan, the check and the analysis already fill the time allowed"] if room <= 0 else []
    if reserve < wanted:
        notes.append("the check would need more than half the time allowed; it is given half")
    timed = [t for t in model.evaluation_timings if t > 0]
    if len(timed) > 1 and max(timed) > 2.0 * min(timed):
        notes.append(f"the same design was timed {len(timed)} times, from {min(timed):.3g} s to {max(timed):.3g} s; the longest time is used")
    if runs and finer is None:
        notes.append("the check at finer settings was not timed; it is counted at the cost of one search design, at least")
    return SearchPlan(n, max(0.0, float(limit_s) - reserve - analysis_s), reserve, runs, analysis_s, float(scan), wall, basis, "; ".join(notes))
