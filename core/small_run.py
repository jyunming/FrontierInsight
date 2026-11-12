"""FI runs the simulation once on a small case before anyone reads its code.

The code review used to read code that had never run: in the iteration study, most of the review-and-repair time of the
kinetics quests went into findings whose tests only showed a crash that one call of ``run_trial`` shows in a tenth of a
second, at its file and line. So, before the second model reads the code, FI runs the simulation itself with its own
trial harness (:mod:`core.trial_runner`) on a few settings of the plan's grid (:func:`settings`: the middle of every axis
and the two corners, every axis at its first and at its last value), a few trials each (:data:`TRIALS`), within a tight
time (:func:`budget`):

* A run that cannot load the simulation, or a setting where more than half of the trials fail
  (:data:`core.trial_checks.FAULT_FRACTION`), is a crash: it goes to the existing repair of a failed run with the setting,
  the reason, the place in the code and the last traceback, until the small run runs or the repairs are spent
  (``RunChecksMixin._small_run_gate``).
* What it gave (each returned value per setting, the failed trials and why) is shown to the code's reader
  (:meth:`SmallRun.for_reader`), who then reads the code with results.
* A repair of a reader's finding is kept only if the small run still runs (``core.finding_repair.Env.small_run``).

The calibration of intervals (:mod:`core.trial_checks`) needs more trials than this and is judged on the study's own run.
The small run writes only under ``.fi/small_run/`` and its record ``.fi/small_run.json``; the study's ``raw/`` is never
touched. Nothing here knows a field of science.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import call_record as _calls
from . import in_run as _in_run
from . import metric_spec as _metric_spec
from . import optimise_search as _search
from . import trial_checks as _checks
from . import trial_runner as _trials

#: Trials per setting (a study without randomness runs each setting once).
TRIALS = 3
#: The time the small run may take: this share of the study's time limit, within the floor and the ceiling (seconds).
BUDGET_FRAC = 0.1
BUDGET_MIN_S = 30
BUDGET_MAX_S = 180
DIR = Path(".fi") / "small_run"
RECORD = Path(".fi") / "small_run.json"
#: The small run's trials in the record format of a study, which FI's analysis is run on before the long run.
TRIALS_PATH = DIR / "raw" / "trials.json"
#: In the record: each fault of the code sent back once, ``"kind:quantity"``, and how many send-backs were made. Kept
#: whatever code version a repair produces (a new small run keeps them), so a fault is sent back once per quest.
SENT = "sent_back"


def budget(timeout_s: Any) -> int:
    try:
        limit = float(timeout_s)
    except (TypeError, ValueError):
        limit = 0.0
    return int(min(BUDGET_MAX_S, max(BUDGET_MIN_S, BUDGET_FRAC * limit)))


def settings(grid: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The settings the small run uses: the middle value of every axis, then every axis at its first value, then every
    axis at its last value (the two corners), each once. A study of separate parts (a grid that knows its ``sub_studies``) has
    no such corners (they are not settings of any part): one real setting of every part is run instead (:func:`core.optimise_search.part_cells`)."""
    parts = _search.part_cells(grid) if getattr(grid, "sub_studies", None) else []
    if parts:
        out = []
        for _name, cell in parts:
            if cell not in out:
                out.append(cell)
        return out
    axes = {str(k): list(v) for k, v in (grid or {}).items() if isinstance(v, (list, tuple)) and len(v)}
    if not axes:
        return [{}]
    out: list[dict[str, Any]] = []
    for pick in (lambda v: v[(len(v) - 1) // 2], lambda v: v[0], lambda v: v[-1]):
        cell = {k: pick(v) for k, v in axes.items()}
        if cell not in out:
            out.append(cell)
    return out


@dataclass
class Setting:
    key: str
    planned: int
    ok: int = 0
    failed: int = 0
    timed_out: bool = False
    reasons: list[tuple[str, int]] = field(default_factory=list)
    values: dict[str, list[Any]] = field(default_factory=dict)
    traceback: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)  # the rows of the trials, for what the code asked of its cell
    cell: dict[str, Any] = field(default_factory=dict)  # the setting the trials ran at
    levels: list[dict[str, Any]] = field(default_factory=list)  # the values each run compares inside it

    @property
    def fault(self) -> bool:
        """More than half of its trials failed, and not because the small run's time ran out."""
        return not self.timed_out and self.planned > 0 and self.failed / self.planned > _checks.FAULT_FRACTION


@dataclass
class SmallRun:
    settings: list[Setting]
    load_error: str = ""
    seconds: float = 0.0
    axes: list[str] = field(default_factory=list)  # the grid's settings that take two values or more: the ones compared
    #: For a search: the search's own evaluation of its baseline design (:func:`baseline_answer`), ``None`` when not made.
    baseline: dict[str, Any] | None = None

    @property
    def faults(self) -> list[Setting]:
        return [s for s in self.settings if s.fault]

    @property
    def crashed(self) -> bool:
        return bool(self.load_error) or bool(self.faults)

    @property
    def timed_out(self) -> bool:
        return any(s.timed_out for s in self.settings)

    @property
    def reads(self) -> dict[str, Any]:
        """Which settings the code asked for over the whole small run (:func:`core.trial_runner.read_facts`). A run that
        crashed, that timed out in a setting, or in which a setting had no trial that ran makes no claim about what was
        never read: only a whole run is positive evidence."""
        facts = _trials.read_facts([r for s in self.settings for r in s.rows], self.axes)
        if facts["never"] and (self.crashed or self.timed_out or any(s.ok < 1 for s in self.settings)):
            return {**facts, "measurable": False, "never": [],
                    "why": "the small run did not run every setting to the end"}
        return facts

    def unread_note(self) -> str:
        """One plain sentence when a setting of the study was never asked for by the code in the small run, else ``""``."""
        never = self.reads["never"]
        if not never:
            return ""
        names = ", ".join(f"`{a}`" for a in never)
        return (f"FI's small run: the simulation never asked for {names}, a setting of the plan's grid; "
                "every run gets the same value of it, whatever the grid gives.")

    def says(self) -> str:
        """One plain line for run.log and the record."""
        if self.load_error:
            return f"the simulation could not be loaded: {self.load_error}"
        n = sum(s.planned for s in self.settings)
        bad = self.faults
        if bad:
            first = bad[0]
            why = first.reasons[0][0] if first.reasons else "no reason given"
            return (f"in {len(bad)} of {len(self.settings)} setting(s) most trials failed ({first.key or 'the one setting'}: "
                    f"{first.failed} of {first.planned}, {why})")
        failed = sum(s.failed for s in self.settings)
        return (f"{n} trial(s) at {len(self.settings)} setting(s) ran" + (f", {failed} failed" if failed else "")
                + (f" (the small run's time ran out in {sum(1 for s in self.settings if s.timed_out)} of them; "
                   "a setting that did not finish is slower than the time it was given)" if self.timed_out else ""))

    def directive(self, level_rule: str = "") -> str:
        """For the repair of a failed run: what failed, where, and the last traceback. ``level_rule``: FI's one rule for a
        result compared inside each run (:func:`core.skeleton.level_rule`), said only when a failure is of that kind."""
        if self.load_error:
            return (f"FI ran the simulation on a small case before the study, and it could not be loaded: {self.load_error}\n\n"
                    "Repair it so that FI can import simulate.py and call its trial function. Keep everything else as it is.")
        lines = []
        for s in self.faults[:4]:
            why = "; ".join(f"{n}x {r}" for r, n in s.reasons[:2])
            lines.append(f"- {s.key or 'the one setting'}: {s.failed} of {s.planned} trials failed - {why}")
        tb = next((s.traceback for s in self.faults if s.traceback), "")
        rule = (" " + level_rule) if level_rule and any(_checks.is_level_fault(r) for s in self.faults for r, _ in s.reasons) else ""
        return ("FI ran the simulation on a small case before the study (a few trials at the middle of the plan's settings "
                "and at its corners), and most trials failed in these settings:\n" + "\n".join(lines)
                + (f"\n\nThe last traceback:\n{tb[-1500:]}" if tb else "")
                + "\n\nRepair the simulation so that a trial at these settings runs to the end and gives its numbers. Never "
                "return NaN, an infinity or a made-up value in place of a trial that failed: FI counts such a trial as "
                "failed." + rule + " Keep everything else as it is.")

    def for_reader(self, limit: int = 2500) -> str:
        """What the code's reader is shown: each setting's returned values (at most a few per name) and its failures."""
        if self.load_error:
            return f"FI could not load the simulation to run it: {self.load_error}"
        parts = [f"FI ran the simulation with its own harness before this reading: {self.says()}."]
        if self.unread_note():
            parts.append(self.unread_note())
        for s in self.settings:
            head = f"Setting {s.key or '(the one setting)'}: {s.ok} of {s.planned} trial(s) ran"
            if s.failed:
                head += "; failed: " + "; ".join(f"{n}x {r}" for r, n in s.reasons[:2])
            shown = []
            for name in sorted(s.values)[:24]:
                vals = [_short(v) for v in s.values[name][:3]]
                shown.append(f"{name}={', '.join(vals)}")
            parts.append(head + ("\n  " + "; ".join(shown) if shown else ""))
        text = "\n".join(parts)
        return text if len(text) <= limit else text[:limit - 3] + "..."

    def record(self) -> dict[str, Any]:
        return {"says": self.says(), "crashed": self.crashed, "load_error": self.load_error, "seconds": round(self.seconds, 2),
                # The names the simulation's entry returned in the settings that ran: FI's record of what the entry itself
                # gives (a headline metric listed here is one the entry returns, not one the analysis derives later).
                "returned": sorted({name for s in self.settings for name in s.values}),
                # Which settings of the grid the code asked for (a trial's ``Cell`` record): ``never`` is a fact only when
                # ``measurable``; ``why`` says what stopped the claim.
                "reads": self.reads,
                "settings": [{"key": s.key, "planned": s.planned, "ok": s.ok, "failed": s.failed, "timed_out": s.timed_out,
                              "reasons": [[r, n] for r, n in s.reasons[:3]],
                              # what each trial that ran returned, name by name, for the checks of the plan's shapes
                              "values": {n: v[:TRIALS] for n, v in s.values.items()}} for s in self.settings],
                **({"baseline": self.baseline} if self.baseline is not None else {})}

    def trials(self, thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        """The small run's trials as FI records a study's (``raw/trials.json``, :func:`core.in_run.summary_cells`): what
        FI's analysis is run on before the long run (:func:`core.pipeline_gate.analysis_gate`)."""
        cells: list[dict[str, Any]] = []
        for s in self.settings:
            if s.rows:
                cells.extend(_in_run.summary_cells(s.cell, s.key, s.planned, s.rows, s.levels))
        return {"schema": "fi.trials/v1", "thresholds": dict(thresholds or {}), "cells": cells}


def _short(v: Any) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        text = json.dumps(v, default=str)
        return text if len(text) <= 40 else text[:37] + "..."
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return f"{v:.4g}"


def key_of(simulate: Path, grid: dict[str, Any], deterministic: bool, thresholds: dict[str, Any] | None,
           in_each_run: dict[str, list[Any]] | None = None, baseline: dict[str, Any] | None = None) -> str:
    """What the small run depends on: the simulation and its package (as FI keys its trials), the settings it uses, the
    thresholds the trials are given, the values each run is given to compare inside it and, for a search, the cell of its
    baseline design (:func:`baseline_answer` evaluates it)."""
    base = _trials._run_key(Path(simulate), {"grid": grid, "thresholds": thresholds or {}}, TRIALS, 0, deterministic)
    inside = f"|{json.dumps(in_each_run, sort_keys=True, default=str)}" if in_each_run else ""
    inside += f"|baseline={json.dumps(baseline, sort_keys=True, default=str)}" if baseline else ""
    return hashlib.sha256(f"{base}|{json.dumps(settings(grid), sort_keys=True, default=str)}{inside}".encode()).hexdigest()[:20]


async def run(executor: Any, python: Path | str, quest_root: Path, module: str, grid: dict[str, Any], *, deterministic: bool,
              timeout_s: int, env: dict[str, str] | None = None, thresholds: dict[str, Any] | None = None,
              paired: bool = False, in_each_run: dict[str, list[Any]] | None = None, base_seed: int = 0) -> SmallRun:
    """Run the small case: each setting of :func:`settings` in its own process, :data:`TRIALS` trials each, within
    ``timeout_s`` in all, through FI's own harness. Writes only under :data:`DIR`."""
    quest_root = Path(quest_root)
    folder = quest_root / DIR
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    (quest_root / _trials.HARNESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    (quest_root / _trials.HARNESS_PATH).write_text(_trials.HARNESS_SOURCE, encoding="utf-8")
    started = time.monotonic()
    out: list[Setting] = []
    load_error = ""
    # From a copy of code/ outside the quest folder (core/outside_run.py), as every cell of the study runs.
    executor = _trials._outside_executor(executor, quest_root)
    chosen = settings(grid)
    axes = [str(k) for k, v in (grid or {}).items() if isinstance(v, (list, tuple)) and len(v) >= 2]
    parts_run = bool(getattr(grid, "sub_studies", None)) and len(chosen) > 1
    total = len(chosen)
    for i, cell in enumerate(chosen):
        sub = folder / f"s{i}"
        sub.mkdir(parents=True, exist_ok=True)
        plan = _trials._plan(quest_root, module, {k: [v] for k, v in cell.items()}, runs_per_setting=TRIALS,
                             base_seed=base_seed, deterministic=deterministic, folder=sub, out_name="cell{index}.out.jsonl",
                             paired=paired, thresholds=thresholds, in_each_run=in_each_run)
        for task in plan:
            setting = Setting(task["key"], len(task["trials"]), cell=dict(task["cell"]), levels=list(task.get("levels") or []))
            out.append(setting)
            left = int(timeout_s - (time.monotonic() - started))
            if parts_run:
                # every part is run: each setting gets an equal share of what is left, so one slow part cannot take the time of the rest
                left = left // max(1, total - len(out) + 1)
            if left <= 0:
                setting.timed_out = True
                continue
            # The small run is a checking run: it notes which functions of the code were called (core/call_record.py).
            runner, calls = _calls.prepare(quest_root, "small_run")
            try:
                result = await executor.execute([str(python), runner, calls.as_posix(), _trials.HARNESS_PATH.as_posix(),
                                                 task["spec"], task["out"]],
                                                cwd=quest_root, timeout_s=max(1, left), env=env)
            finally:
                _calls.absorb(quest_root, quest_root / calls)
            reasons: Counter = Counter()
            seen: set[int] = set()
            try:
                lines = (quest_root / task["out"]).read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            for line in lines:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict) or row.get("nonce") != task["nonce"]:
                    continue
                if row.get("load_error"):
                    load_error = load_error or str(row["load_error"])
                    continue
                trial = row.get("trial")
                if not isinstance(trial, int) or trial in seen:
                    continue
                seen.add(trial)
                setting.rows.append(row)
                if row.get("status") == "ok":
                    setting.ok += 1
                    for name, value in (row.get("values") or {}).items():
                        setting.values.setdefault(str(name), []).append(value)
                else:
                    setting.failed += 1
                    where = _trials.last_frame(str(row.get("traceback") or ""), quest_root)
                    reasons[str(row.get("reason") or "no reason given") + (f" (at {where})" if where else "")] += 1
                    setting.traceback = setting.traceback or str(row.get("traceback") or "")
            missing = setting.planned - setting.ok - setting.failed
            if missing > 0:
                if getattr(result, "timed_out", False):
                    setting.timed_out = True
                else:
                    setting.failed += missing
                    reasons[f"the process stopped (exit code {getattr(result, 'returncode', '?')}) before reporting"
                            + (f": {(getattr(result, 'stderr', '') or '').strip().splitlines()[-1][:200]}"
                               if (getattr(result, "stderr", "") or "").strip() else "")] += missing
            setting.reasons = reasons.most_common()
        if load_error:
            break
    got = SmallRun(out, load_error, time.monotonic() - started, axes)
    try:
        (quest_root / TRIALS_PATH).parent.mkdir(parents=True, exist_ok=True)
        (quest_root / TRIALS_PATH).write_text(json.dumps(got.trials(thresholds), allow_nan=True, default=str),
                                              encoding="utf-8")
    except OSError:
        pass
    return got


def write_record(quest_root: Path | str, key: str, run_: SmallRun) -> None:
    try:
        path = Path(quest_root) / RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        kept = read_record(quest_root).get(SENT)  # the faults sent back stay sent, whatever code this run ran
        path.write_text(json.dumps({"key": key, **run_.record(), "for_reader": run_.for_reader(),
                                    **({SENT: kept} if isinstance(kept, dict) else {})}, indent=1, default=str),
                        encoding="utf-8")
    except OSError:
        pass


def update_record(quest_root: Path | str, **fields: Any) -> None:
    """Set ``fields`` in the record as it is on disk (nothing when there is no record)."""
    record = read_record(quest_root)
    if not record:
        return
    record.update(fields)
    try:
        (Path(quest_root) / RECORD).write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass


def read_record(quest_root: Path | str) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# --- what the small run says about the frozen plan --------------------------------------------------------------------
# Each check reads a fact FI holds (the plan's declarations, the names and shapes the small run returned, the search's own
# judgement of its baseline design) and gives a :class:`Fault`. Only an exact contradiction is sent back to the code
# (``send``), once per fault per quest; anything a small sample can explain is a line for the code's reader. Nothing here
# stops the quest (``RunChecksMixin._small_run_checks``).

SHAPE, OBJECTIVE, CONSTRAINT, MISSING, ANALYSIS = "shape", "objective", "constraint", "missing_output", "analysis"
COUNTING_ROLES = ("result", "reference", "physical_bound")


@dataclass
class Fault:
    kind: str  # SHAPE, OBJECTIVE, CONSTRAINT, MISSING or ANALYSIS
    quantity: str  # the name the fault is about (for ANALYSIS: the function of derive.py)
    why: str  # one plain sentence
    send: bool = True  # an exact contradiction, sent back once; else a line for the reader only
    script: str = "simulate.py"
    body: str = ""  # for ANALYSIS: the function of derive.py to repair
    #: What is wrong, whichever check found it: ``"per_value"`` (one number per value where the plan declares one per run),
    #: ``"absent"`` (no trial returned it); ``""`` for any other fault (then the kind names it).
    fact: str = ""

    @property
    def key(self) -> str:
        """The fault in the ledger of send-backs: the same fact about the same quantity is one fault."""
        return f"{self.fact or self.kind}:{self.quantity}"


def _settings_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in record.get("settings") or [] if isinstance(s, dict)]


def _whole(record: dict[str, Any]) -> bool:
    """Every planned trial of every setting of the small run ran to the end and ended well: only then is a name it never
    returned a fact (a trial that failed might have returned it)."""
    settings = _settings_of(record)
    return bool(settings) and not record.get("crashed") and all(
        not s.get("timed_out") and int(s.get("failed") or 0) == 0 and int(s.get("ok") or 0) >= max(1, int(s.get("planned") or 0))
        for s in settings)


def shape_faults(protocol: dict[str, Any] | None, record: dict[str, Any]) -> list[Fault]:
    """A number the plan declares one per run that a trial of the small run returned per value of what each run compares
    inside it (:func:`core.in_run.shape_problem`): exact whatever the sample, sent back."""
    found: dict[str, str] = {}
    for s in _settings_of(record):
        values = s.get("values") if isinstance(s.get("values"), dict) else {}
        for n in range(max((len(v) for v in values.values() if isinstance(v, list)), default=0)):
            row = {name: v[n] for name, v in values.items() if isinstance(v, list) and n < len(v)}
            for name, why in _in_run.shape_problem(protocol, row).items():
                found.setdefault(name, why)
    return [Fault(SHAPE, name, f"FI's small run: {why}.", fact="per_value") for name, why in sorted(found.items())]


def _returned_forms(record: dict[str, Any]) -> set[str]:
    return {str(n) for n in record.get("returned") or []}


def _is_returned(name: str, returned: set[str]) -> bool:
    return any(form in returned for form in _metric_spec.result_forms(name))


def _setting_names(protocol: dict[str, Any]) -> set[str]:
    """Every name the plan gives a run as a setting or holds as a constant: a derived quantity may read these too."""
    names: set[str] = set()
    for key in ("grid", "constants", "thresholds", "in_each_run"):
        if isinstance(protocol.get(key), dict):
            names |= {str(k) for k in protocol[key]}
    block = protocol.get("optimisation") if isinstance(protocol.get("optimisation"), dict) else {}
    for key in ("grid", "fixed", "in_each_run", "numerical_settings"):
        if isinstance(block.get(key), dict):
            names |= {str(k) for k in block[key]}
    names |= {str(v.get("name")) for v in block.get("design_variables") or [] if isinstance(v, dict)}
    for part in [*(protocol.get("sub_studies") or []), *(block.get("sub_studies") or [])]:
        if isinstance(part, dict) and isinstance(part.get("hold"), dict):
            names |= {str(k) for k in part["hold"]}
    return names


def missing_outputs(protocol: dict[str, Any] | None, record: dict[str, Any]) -> list[Fault]:
    """A result the plan declares each run returns (``returned: each_run``, a counting role), or a name a derived
    quantity is computed from (``derived[].inputs``) that is neither a setting nor computed afterwards, that no trial of
    the small run returned under any of its forms. Decided from the plan's declarations and the names the small run
    returned, never from an exception the analysis raised; only when every setting of the small run ran to the end."""
    if not isinstance(protocol, dict) or not _whole(record):
        return []
    returned = _returned_forms(record)
    metrics = {str(m.get("id")): m for m in protocol.get("metrics") or [] if isinstance(m, dict) and m.get("id")}
    derived = {str(d.get("id")): d for d in protocol.get("derived") or [] if isinstance(d, dict) and d.get("id")}
    out: dict[str, str] = {}
    for name, m in metrics.items():
        if m.get("returned") == "each_run" and str(m.get("role") or "result") in COUNTING_ROLES \
                and not _is_returned(name, returned):
            out[name] = (f"FI's small run: the plan declares `{name}` a result each run returns, and no trial returned it "
                         f"(the trials returned {', '.join(sorted(returned)[:10]) or 'nothing'})")
    settings = _setting_names(protocol)
    for d_id, d in derived.items():
        for name in [str(i) for i in d.get("inputs") or [] if isinstance(i, str)]:
            if name in out or name in derived or name in settings or _is_returned(name, returned):
                continue
            m = metrics.get(name)
            if m is not None and m.get("returned") != "each_run":
                continue  # computed afterwards, by the analysis
            out[name] = (f"FI's small run: `{d_id}` is computed from `{name}`, and no trial returned `{name}` "
                         f"(the trials returned {', '.join(sorted(returned)[:10]) or 'nothing'})")
    return [Fault(MISSING, name, why + ".", fact="absent") for name, why in out.items()]


def judge_baseline(block: dict[str, Any] | None, record: dict[str, Any]) -> list[Fault]:
    """For a search: the search's own judgement (:func:`core.optimise_search.judge`) of its evaluation of the baseline
    design, on the objective and on each constraint separately. A quantity returned per value of what each run compares
    inside it, or returned at none of the designs FI ran (the baseline and every trial of a small run that ran to the end),
    is exact and sent back; one with no finite value at the baseline is a line only (it can be defined at other designs,
    and the search's own rule judges every evaluation)."""
    base = record.get("baseline") if isinstance(record.get("baseline"), dict) else None  # None: not evaluated
    if not isinstance(block, dict) or base is None or not isinstance(base.get("values"), dict):
        return []
    values = base["values"]
    returned = _returned_forms(record)
    quantity = str((block.get("objective") or {}).get("quantity") or "")
    out: list[Fault] = []

    def classify(kind: str, name: str, judged: dict[str, Any]) -> None:
        if judged.get("status") == "ok":
            return
        why = f"FI's small run: the search's own judgement of its baseline design says {judged.get('reason')}"
        if name not in values:
            never = not _is_returned(name, returned) and _whole(record)
            out.append(Fault(kind, name, why + ("; no trial of the small run returned it either." if never else "."),
                             send=never, fact="absent" if never else ""))
        elif isinstance(values[name], dict):
            out.append(Fault(kind, name, why + ".", fact="per_value"))
        else:  # a non-finite value at the baseline, or a limit the plan wrote that cannot be read: not the code's shape
            out.append(Fault(kind, name, why + "; the search judges every design it evaluates by the same rule.",
                             send=False))

    if quantity:
        classify(OBJECTIVE, quantity, _search.judge({**block, "constraints": []}, values))
    for c in block.get("constraints") or []:
        if not isinstance(c, dict) or not c.get("quantity"):
            continue
        name = str(c["quantity"])
        # each constraint on its own: the objective is given a number so that only this constraint is judged
        alone = {**values, quantity: 0.0} if quantity and quantity != name else dict(values)
        classify(CONSTRAINT, name, _search.judge({**block, "constraints": [c]}, alone))
    return out


def analysis_split(gate: Any, *, docker: bool = False) -> Fault | None:
    """FI's analysis run on the small run's trials (:func:`core.pipeline_gate.analysis_gate`), split by evidence: a fault
    of derive.py whatever the data (FI's stub, a return FI refused, an exception that says the code is wrong) is sent back
    to that function; anything a small sample can explain (a fit with too few trials, a value missing at every small
    setting, a ``TypeError`` three trials can cause) is a line for the code's reader, and the study's own record decides;
    a gate that could not run says so. ``None`` when the analysis ran from start to end."""
    from . import pipeline_gate as _gate

    if gate is None or getattr(gate, "ok", False):
        return None
    kind = getattr(gate, "kind", "")
    if kind == _gate.NOT_CHECKED:
        why = ("FI's small run: FI's analysis was not checked on the small run's trials in Docker (the container cannot "
               "see FI's scratch copy)" if docker else
               f"FI's small run: FI's analysis was not checked on the small run's trials ({getattr(gate, 'detail', '')})")
        return Fault(ANALYSIS, "not_checked", why + ".", send=False, script="derive.py")
    body = str(getattr(gate, "body", "") or "")
    said = " ".join(str(gate.says()).split())
    said = said if len(said) <= 400 else said[:397] + "..."
    if kind == _gate.FAULT and getattr(gate, "exception", "") not in _gate._MAYBE_FAULTS:
        return Fault(ANALYSIS, body or str(getattr(gate, "where", "") or "derive.py"),
                     f"FI's small run: FI ran its analysis on the small run's trials, and {said}.", script="derive.py",
                     body=body)
    return Fault(ANALYSIS, body or str(getattr(gate, "stage", "") or "analysis"),
                 f"FI's small run: FI ran its analysis on the small run's trials, and {said}; the small run has at most {TRIALS} "
                 "trial(s) a setting, so the study's own record decides.", send=False, script="derive.py", body=body)


def no_value(missing: list[dict[str, Any]]) -> list[Fault]:
    """A declared result FI's analysis on the small run's trials holds no value of, at any small setting
    (:func:`core.analysis_frame.not_computed`, ``status: missing``): what a few trials can explain (a fit may need more),
    so a line for the code's reader; the study's own record decides."""
    return [Fault(ANALYSIS, f"no_value:{m.get('id')}",
                  f"FI's small run: FI's analysis on the small run's trials gives no value of `{m.get('id')}` at any of its "
                  f"settings ({str(m.get('why') or 'no reason given')[:200]}); the small run has at most {TRIALS} trial(s) a "
                  "setting, so the study's own record decides.", send=False, script="derive.py")
            for m in missing if m.get("id")]


def sent(record: dict[str, Any]) -> dict[str, Any]:
    """The faults sent back so far in this quest: ``{"faults": [key, ...], "count": send-backs made}``."""
    got = record.get(SENT) if isinstance(record.get(SENT), dict) else {}
    return {"faults": [str(k) for k in got.get("faults") or []], "count": int(got.get("count") or 0)}


async def baseline_answer(executor: Any, python: Any, quest_root: Path, module: str, block: dict[str, Any], *,
                          timeout_s: int, env: dict[str, str] | None,
                          thresholds: dict[str, Any] | None) -> dict[str, Any] | None:
    """The search's own evaluation of its baseline design: the function the search calls
    (:func:`core.trial_runner.study_entry`) on the cell the search gives it (:func:`core.optimise_search.cell_of`), once,
    as the search calls it (:func:`core.optimise._evaluate`). ``{"values": ..., "reason": ...}``, or ``None`` when the
    simulation has no such function, the plan has no baseline, or the evaluation was not made (the time ran out, the
    simulation could not be loaded): then nothing is judged on it."""
    from . import optimise as _optimise

    entry = _trials.study_entry(_trials.entries(Path(quest_root) / module))
    baseline = (block.get("baseline") or {}).get("values") if isinstance(block.get("baseline"), dict) else None
    if not entry or not isinstance(baseline, dict):
        return None
    try:
        answer, _stderr, load_error = await _optimise._evaluate(
            executor, python, Path(quest_root), module, entry, _search.cell_of(block, dict(baseline)), runs=1,
            base_seed=0, timeout_s=timeout_s, env=env, thresholds=thresholds, name="small_run_baseline",
            in_each_run=_search.in_each_run(block) or None)
    except (OSError, KeyError, TypeError, ValueError):
        return None
    if load_error or not isinstance(answer, dict) or answer.get("stop"):
        return None
    if isinstance(answer.get("values"), dict):
        return {"values": answer["values"], "reason": ""}
    return {"values": None, "reason": str(answer.get("failed") or "no reason given")}
