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
  (:func:`reader_text`, with the plan's checked values hidden), who then reads the code with results.
* A repair of a reader's finding is kept only if the small run still runs (``core.finding_repair.Env.small_run``).

The calibration of intervals (:mod:`core.trial_checks`) needs more trials than this and is judged on the study's own run.
The small run writes only under ``.fi/small_run/`` and its record ``.fi/small_run.json``; the study's ``raw/`` is never
touched. Nothing here knows a field of science.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import execution as _execution
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
#: Seconds past the budget that the settings run together are given to stop (their process trees killed) before they are
#: cancelled.
CLEANUP_S = 5.0
#: Seconds the cancelled settings are then waited for while their executors stop their process trees (the tree's own
#: wait for a killed process, plus one). The small run returns within the budget plus ``CLEANUP_S`` plus this, even when
#: a stop takes longer; a stop still unfinished then goes on by itself and its setting is recorded as timed out.
STOP_WAIT_S = 11.0
#: The clock of the time budget (a seam for tests: they give it a clock they advance themselves).
_clock = time.monotonic
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
        """What the code's reader is shown (:func:`reader_text` of this run's record), with no value hidden."""
        return reader_text(self.record(), limit=limit)

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


def reader_text(record: dict[str, Any], hidden: list[float] | None = None, limit: int = 2500) -> str:
    """What the code's reader is shown of a small run, from its record (:meth:`SmallRun.record`): each setting's returned
    values (at most a few per name) and its failures. ``hidden``: the plan's values the reader must not see
    (:func:`core.review_pairs.plan_values`). A returned value FI recorded that equals one of them is shown as
    :data:`core.review_pairs.PLACEHOLDER`, decided on the number FI holds, never on the digits it prints: a value 0.10003
    printed as ``0.1`` is hidden, and the run's own estimate 0.10360, printed as ``0.1036``, is shown beside a plan value
    of 0.103556. The text is cut where it would be cut with nothing hidden, and never inside a value. A record written
    before the settings' values were kept holds only its printed text, where FI cannot tell which number is which: with
    values to hide, every number in it is hidden."""
    settings_ = record.get("settings")
    if not isinstance(settings_, list) or any(not isinstance(s, dict) or "values" not in s for s in settings_):
        text = str(record.get("for_reader") or "")
        if not hidden:
            return text
        from . import review_pairs as _pairs

        return _pairs._TOKEN_RE.sub(_pairs.PLACEHOLDER, text)
    if record.get("load_error"):
        return f"FI could not load the simulation to run it: {record['load_error']}"
    pieces: list[tuple[str, str]] = []  # (as printed with nothing hidden, as shown): one per value, one per other text

    def say(text: str) -> None:
        pieces.append((text, text))

    say(f"FI ran the simulation with its own harness before this reading: {record.get('says') or ''}.")
    reads = record.get("reads") if isinstance(record.get("reads"), dict) else {}
    if reads.get("never"):
        names = ", ".join(f"`{a}`" for a in reads["never"])
        say(f"\nFI's small run: the simulation never asked for {names}, a setting of the plan's grid; "
            "every run gets the same value of it, whatever the grid gives.")
    for s in settings_:
        head = f"Setting {s.get('key') or '(the one setting)'}: {s.get('ok', 0)} of {s.get('planned', 0)} trial(s) ran"
        if s.get("failed"):
            head += "; failed: " + "; ".join(f"{n}x {r}" for r, n in (s.get("reasons") or [])[:2])
        say("\n" + head)
        values = s.get("values") if isinstance(s.get("values"), dict) else {}
        for i, name in enumerate(sorted(values)[:24]):
            say(("\n  " if i == 0 else "; ") + f"{name}=")
            for j, v in enumerate((values[name] or [])[:3]):
                if j:
                    say(", ")
                pieces.append((_short(v), _short(v, hidden)))
    if sum(len(plain) for plain, _ in pieces) <= limit:
        return "".join(shown for _, shown in pieces)
    out, room = [], limit - 3
    for plain, shown in pieces:
        if len(plain) > room:
            if plain == shown:
                out.append(plain[:room])
            break
        out.append(shown)
        room -= len(plain)
    return "".join(out) + "..."


def _short(v: Any, hidden: list[float] | None = None) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        text = json.dumps(v, default=str)
        return text if len(text) <= 40 else text[:37] + "..."
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if hidden:
        from . import review_pairs as _pairs

        if _pairs.equals_one_of(float(v), hidden):
            return _pairs.PLACEHOLDER
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


def _read_setting(quest_root: Path, task: dict[str, Any], setting: Setting, result: Any) -> str:
    """Fill ``setting`` from what its process wrote (FI's harness, one line per trial); the simulation's load error when
    one was reported, else ``""``."""
    load_error = ""
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
    return load_error


async def run(executor: Any, python: Path | str, quest_root: Path, module: str, grid: dict[str, Any], *, deterministic: bool,
              timeout_s: int, env: dict[str, str] | None = None, thresholds: dict[str, Any] | None = None,
              paired: bool = False, in_each_run: dict[str, list[Any]] | None = None, base_seed: int = 0,
              at_once: int = 1, log: Any = None) -> SmallRun:
    """Run the small case: each setting of :func:`settings` in its own process, :data:`TRIALS` trials each, within
    ``timeout_s`` in all, through FI's own harness. Writes only under :data:`DIR`.

    ``at_once``: how many settings run at the same time (the study's ``execution.settings_at_once``), each process with its
    share of the cores and fewer at once when the memory would not hold them, as the study's own run starts its settings
    (:class:`core.execution.SettingsGate`). One after another, a study of separate parts gave each part's setting an equal
    share of the time, so a setting that cost more than its share never finished even when the whole time would have held
    it; run together, every setting that has a place may use what is left of the whole time. A part's setting that waits
    for a place gets an equal share of what is left with the others still waiting, so one slow part cannot take the time of
    the rest. A setting that does not finish in its time is timed out, as before; ``timeout_s`` bounds the whole small run."""
    quest_root = Path(quest_root)
    folder = quest_root / DIR
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    (quest_root / _trials.HARNESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    (quest_root / _trials.HARNESS_PATH).write_text(_trials.HARNESS_SOURCE, encoding="utf-8")
    started = _clock()
    # From a copy of code/ outside the quest folder (core/outside_run.py), as every cell of the study runs.
    executor = _trials._outside_executor(executor, quest_root)
    chosen = settings(grid)
    axes = [str(k) for k, v in (grid or {}).items() if isinstance(v, (list, tuple)) and len(v) >= 2]
    parts_run = bool(getattr(grid, "sub_studies", None)) and len(chosen) > 1
    tasks: list[tuple[dict[str, Any], Setting]] = []
    for i, cell in enumerate(chosen):
        sub = folder / f"s{i}"
        sub.mkdir(parents=True, exist_ok=True)
        plan = _trials._plan(quest_root, module, {k: [v] for k, v in cell.items()}, runs_per_setting=TRIALS,
                             base_seed=base_seed, deterministic=deterministic, folder=sub, out_name="cell{index}.out.jsonl",
                             paired=paired, thresholds=thresholds, in_each_run=in_each_run)
        for task in plan:
            tasks.append((task, Setting(task["key"], len(task["trials"]), cell=dict(task["cell"]),
                                        levels=list(task.get("levels") or []))))
    slots = max(1, min(int(at_once or 1), len(tasks)))
    waiting = [len(tasks)]  # the settings not started yet

    async def start(task: dict[str, Any], setting: Setting, run_env: dict[str, str] | None) -> Any:
        """The process of one setting, given what is left of the time (for a study of parts, its share of it); ``None``
        when no time is left (the setting is timed out without starting)."""
        left = int(timeout_s - (_clock() - started))
        if parts_run:
            # an equal share of what is left with those still waiting for a place
            left = left // max(1, math.ceil(waiting[0] / slots))
        waiting[0] -= 1
        if left <= 0:
            setting.timed_out = True
            return None
        return await executor.execute([str(python), _trials.HARNESS_PATH.as_posix(), task["spec"], task["out"]],
                                      cwd=quest_root, timeout_s=max(1, left), env=run_env)

    out: list[Setting] = []
    load_error = ""
    if slots == 1:
        for task, setting in tasks:
            out.append(setting)
            result = await start(task, setting, env)
            if result is not None:
                load_error = _read_setting(quest_root, task, setting, result) or load_error
            if load_error:
                break  # the simulation cannot be loaded: no other setting is started
    else:
        gate = _execution.SettingsGate(slots, log=log)  # fewer at once when the memory would not hold them
        run_env = _execution.threads_beside(env, slots, _execution.quest_cores(executor))  # each its share of the cores
        metered = hasattr(executor, "meter")
        meter_before = getattr(executor, "meter", None)
        if metered:
            executor.meter = gate.note

        first_bad = [len(tasks)]  # the first setting, in the plan's order, that could not load the simulation

        async def one(index: int, task: dict[str, Any], setting: Setting) -> str:
            async with gate.slot():
                if index > first_bad[0]:
                    return ""  # one before it could not load the simulation: as one after another, this is never started
                result = await start(task, setting, run_env)
                error = "" if result is None else _read_setting(quest_root, task, setting, result)
                if error:
                    first_bad[0] = min(first_bad[0], index)  # said before the place is given up: no waiting setting starts
            return error

        jobs = [asyncio.ensure_future(one(i, task, setting)) for i, (task, setting) in enumerate(tasks)]
        position = {job: i for i, job in enumerate(jobs)}
        pending = set(jobs)
        stopping: set[asyncio.Future] = set()  # cancelled because an earlier setting could not load the simulation
        hard_stop = started + timeout_s + CLEANUP_S  # the whole small run, cleanup included, ends by then
        try:
            while pending:
                left = hard_stop - _clock()
                if left <= 0:
                    break
                done, pending = await asyncio.wait(pending, timeout=left, return_when=asyncio.FIRST_COMPLETED)
                for job in done:
                    if job.exception() is not None:
                        raise job.exception()  # type: ignore[misc]  # an executor that raised: the finally stops the rest
                # No setting after the first that could not load the simulation is started or waited for (the one-at-a-time
                # run stopped there); those before it are still run, as they were before it.
                stopping |= {job for job in pending if position[job] > first_bad[0]}
                pending -= stopping
                for job in stopping:
                    job.cancel()
        finally:
            # Whatever is not finished (the time is out, one raised) is cancelled and awaited: the executor then stops its
            # process tree. The wait is bounded (STOP_WAIT_S), so a stop that takes longer does not hold the small run.
            for job in pending:
                job.cancel()
            if pending | stopping:
                await asyncio.wait(pending | stopping, timeout=STOP_WAIT_S)
            if metered:
                executor.meter = meter_before
        stopped = SimpleNamespace(timed_out=True)
        for (task, setting), job in zip(tasks, jobs):
            out.append(setting)
            if job.cancelled() or not job.done():
                _read_setting(quest_root, task, setting, stopped)  # what it wrote; the trials it did not finish: timed out
                continue
            error = job.result()
            if error:
                load_error = error
                break  # as one after another: the settings up to the first that could not load the simulation
    got = SmallRun(out, load_error, _clock() - started, axes)
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


def values_of(record: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
    """What a small run's record holds of each result: ``{setting: {result name: [value of each trial]}}``."""
    return {str(s.get("key") or ""): {str(n): list(v) for n, v in (s.get("values") or {}).items() if isinstance(v, list)}
            for s in _settings_of(record) if isinstance(s, dict)}


def changed_results(before: dict[str, dict[str, list[Any]]], after: dict[str, dict[str, list[Any]]]) -> list[str]:
    """The names (sorted) of the results ``before`` has, in a setting both small runs ran, whose values are not exactly
    what they were in ``after`` (a result taken out of a setting counts as changed; a result only ``after`` has is new and
    is not compared). The small run's seeds are fixed, so the same code gives the same values."""
    def same(a: Any, b: Any) -> bool:
        return a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)  # NaN is NaN

    out: set[str] = set()
    for key, names in before.items():
        if key not in after:
            continue
        for name, values in names.items():
            now = after[key].get(name)
            if now is None or len(now) != len(values) or not all(same(a, b) for a, b in zip(values, now)):
                out.add(name)
    return sorted(out)


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
    shapes = planned_shapes(protocol)
    return [Fault(MISSING, name, f"{why}. {_unheld(name, shapes[name])}", send=False, fact="absent") if name in shapes
            else Fault(MISSING, name, why + ".", fact="absent") for name, why in out.items()]


#: The fact of :func:`unheld_shapes`: a result the plan states is a curve or a map in each run, which FI's record of a trial
#: cannot hold (a trial returns numbers only: core/in_run.py ``read_returned``).
UNHELD = "planned_shape"


def planned_shapes(protocol: dict[str, Any] | None) -> dict[str, str]:
    """``{result id: what the plan states it is in each run}`` (``a curve along `time```, ``a map over `x` and `y```) for each
    declared result the plan states is a curve or a map (``metrics[].shape`` and ``along``, core/contract.py ``Result``); a
    result the plan states no shape of is one number and is not here."""
    out: dict[str, str] = {}
    for m in (protocol or {}).get("metrics") or [] if isinstance(protocol, dict) else []:
        along = m.get("along") if isinstance(m, dict) else None
        if not isinstance(m, dict) or not m.get("id") or not isinstance(along, list):
            continue
        if m.get("shape") == "curve" and len(along) == 1:
            out[str(m["id"])] = f"a curve along `{along[0]}`"
        elif m.get("shape") == "map" and len(along) == 2:
            out[str(m["id"])] = f"a map over `{along[0]}` and `{along[1]}`"
    return out


def _unheld(name: str, shape: str) -> str:
    kind = shape.split()[1]
    return (f"The plan states `{name}` is {shape} in each run, but FI keeps one number per run for a result, so no trial "
            f"can give it the {kind}: it is not sent back, and the study's results hold no {kind} of `{name}`.")


def unheld_shapes(protocol: dict[str, Any] | None, record: dict[str, Any]) -> list[Fault]:
    """A result the plan states is a curve or a map in each run (:func:`planned_shapes`) that a trial of the small run returned
    as one number: the plan's form is not what the study records. FI's record of a trial holds numbers only (a list is
    refused, core/in_run.py ``read_returned``), so a repair asked to return the curve would make every trial fail: a line for
    the code's reader and plan.md, never a send-back. A result no trial returned is :func:`missing_outputs`' (also a line)."""
    shapes = planned_shapes(protocol)
    out: list[Fault] = []
    for name, shape in shapes.items():
        numbers = [v for s in _settings_of(record) for v in ((s.get("values") or {}).get(name) or [])
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numbers:
            out.append(Fault(SHAPE, name, f"FI's small run: the trials returned one number for `{name}`. {_unheld(name, shape)}",
                             send=False, fact=UNHELD))
    return out


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


#: The fact of :func:`never_value`: a result the analysis cannot give a value of, whatever the trials give.
NEVER = "never_value"


def planned_settings(grid: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Every setting of the plan's grid, as the study's run makes them (:func:`core.trial_runner.cells`), or ``None`` when FI
    does not list them (a grid larger than FI runs, or one it cannot read): then nothing is judged on them."""
    try:
        if getattr(grid, "sub_studies", None):
            return _trials.cells(grid)
        # the axes :func:`settings` takes, so a setting the small run ran has the same key here
        return _trials.cells({str(k): list(v) for k, v in (grid or {}).items() if isinstance(v, (list, tuple)) and len(v)})
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def no_trial_values(trials_path: Path | str) -> bool:
    """Whether the small run's record of its trials (:data:`TRIALS_PATH`) holds no value of any trial: none ran to the end."""
    try:
        cells = json.loads(Path(trials_path).read_text(encoding="utf-8")).get("cells") or []
    except (OSError, ValueError, AttributeError):
        return False
    return not any(isinstance(c, dict) and c.get("metrics") for c in cells)


def calls_of(derived: dict[str, Any]) -> list[dict[str, Any]]:
    """What FI's record of an analysis run (``derived.json``) says of each call of a study's function that works out a declared
    quantity of a counting role: ``{"id", "by", "status", "asked", "gave", "why"}``, ``asked`` being every name the call asked
    the table for (refused reads too) and ``gave`` whether it gave a finite value anywhere. Only calls the record holds
    ``asked`` for (a record of an older frame makes no claim)."""
    out: list[dict[str, Any]] = []
    if not isinstance(derived, dict) or derived.get("from_earlier_run"):
        return out  # the run stopped before it wrote its own record: what is there is an earlier run's
    records = derived.get("derived") if isinstance(derived.get("derived"), dict) else {}
    for ident, rec in records.items():
        if not isinstance(rec, dict) or not isinstance(rec.get("asked"), list):
            continue
        by = str(rec.get("by") or "")
        if not by.startswith("derive_") or str(rec.get("role") or "result") not in COUNTING_ROLES:
            continue
        values = [rec.get("value")] + list((rec.get("by_setting") or {}).values()) \
            + list(((rec.get("by_level") or {}).get("values") or {}).values())
        gave = any(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
        why = next((str(w) for w in (rec.get("missing") or {}).values() if w), "") if isinstance(rec.get("missing"), dict) else ""
        out.append({"id": str(ident), "by": by, "status": str(rec.get("status") or ""), "asked": [str(n) for n in rec["asked"]],
                    # whether the call read the record's shape (its settings, parts, compared factors); unknown counts as read
                    "looked": rec.get("looked") is not False, "gave": gave, "why": why})
    return out


def never_value(calls: list[dict[str, Any]], record: dict[str, Any], ledger: dict[str, Any] | None = None) -> list[Fault]:
    """A declared result FI's analysis can give no value of whatever the trials give, from FI's record of one call of its
    function on a table that holds every setting of the plan (:func:`calls_of`): the call ended, gave no value, and read
    nothing of the table at all (not a result, not even its settings: a function that looks at the settings and finds too
    few, or none of the kind it needs, is honest about the plan), or asked only for names no trial of the small run returned (that one only when a trial ran to the
    end, since none returned anything otherwise). Exact, so sent back: first to the simulation (it must return what the
    result is worked out from), then to the function itself, once the simulation was sent back for it (``ledger``,
    :func:`sent`) and its trials now return a name they did not return then (a repair that added nothing leaves nothing new
    for the function to read: a result the plan's settings cannot give, or one that is two others, costs one send-back,
    not two). A call that read a returned name and found too few values, or that raised, is not one of these."""
    returned = _returned_forms(record)
    ran = any(int(s.get("ok") or 0) >= 1 for s in _settings_of(record))
    names = sorted(returned)
    have = (", ".join(f"`{n}`" for n in names[:12]) + (f" and {len(names) - 12} more" if len(names) > 12 else "")) or "nothing"
    led = ledger if isinstance(ledger, dict) else {}
    then = led.get("returned") if isinstance(led.get("returned"), dict) else {}
    out: list[Fault] = []
    for c in calls:
        if c.get("status") != "ok" or c.get("gave"):
            continue
        asked = sorted(set(c.get("asked") or []))
        if not asked and c.get("looked", True):
            continue  # it read the settings and gave no value from them: the plan's settings may be too few for it
        if asked and not (ran and not any(_is_returned(n, returned) for n in asked)):
            continue
        ident, by = c["id"], c["by"]
        what = ("asks the trials for nothing" if not asked else
                "asks only for " + ", ".join(f"`{n}`" for n in asked) + ", which no trial returns")
        said = f" (it says: {c['why'][:200]})" if c.get("why") else ""
        head = f"FI's small run: `{by}` {what}, so `{ident}` can never get a value{said}"
        given = f"the trials return {have}" if ran else "no trial of the small run ran to the end"
        out.append(Fault(NEVER, ident, f"{head}; {given}. Make each trial return what `{ident}` is worked out from.",
                         fact=NEVER))
        key = f"{NEVER}:{ident}"
        if key not in (led.get("faults") or []) or not (returned - set(then.get(key) or [])):
            continue  # the simulation first; then the function only when the trials return something new
        out.append(Fault(ANALYSIS, by, f"{head}; {given}. Work `{ident}` out from what the trials return.",
                         script="derive.py", body=by, fact=NEVER))
    return out


def sent(record: dict[str, Any]) -> dict[str, Any]:
    """The faults sent back so far in this quest: ``{"faults": [key, ...], "count": send-backs made}``, and ``returned``
    when a :func:`never_value` fault was sent to the simulation: ``{key: the names the trials returned then}``."""
    got = record.get(SENT) if isinstance(record.get(SENT), dict) else {}
    out: dict[str, Any] = {"faults": [str(k) for k in got.get("faults") or []], "count": int(got.get("count") or 0)}
    if isinstance(got.get("returned"), dict):
        out["returned"] = {str(k): [str(n) for n in v] for k, v in got["returned"].items() if isinstance(v, list)}
    return out


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
