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

    @property
    def fault(self) -> bool:
        """More than half of its trials failed, and not because the small run's time ran out."""
        return not self.timed_out and self.planned > 0 and self.failed / self.planned > _checks.FAULT_FRACTION


@dataclass
class SmallRun:
    settings: list[Setting]
    load_error: str = ""
    seconds: float = 0.0

    @property
    def faults(self) -> list[Setting]:
        return [s for s in self.settings if s.fault]

    @property
    def crashed(self) -> bool:
        return bool(self.load_error) or bool(self.faults)

    @property
    def timed_out(self) -> bool:
        return any(s.timed_out for s in self.settings)

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

    def directive(self) -> str:
        """For the repair of a failed run: what failed, where, and the last traceback."""
        if self.load_error:
            return (f"FI ran the simulation on a small case before the study, and it could not be loaded: {self.load_error}\n\n"
                    "Repair it so that FI can import simulate.py and call its trial function. Keep everything else as it is.")
        lines = []
        for s in self.faults[:4]:
            why = "; ".join(f"{n}x {r}" for r, n in s.reasons[:2])
            lines.append(f"- {s.key or 'the one setting'}: {s.failed} of {s.planned} trials failed - {why}")
        tb = next((s.traceback for s in self.faults if s.traceback), "")
        return ("FI ran the simulation on a small case before the study (a few trials at the middle of the plan's settings "
                "and at its corners), and most trials failed in these settings:\n" + "\n".join(lines)
                + (f"\n\nThe last traceback:\n{tb[-1500:]}" if tb else "")
                + "\n\nRepair the simulation so that a trial at these settings runs to the end and gives its numbers. Never "
                "return NaN, an infinity or a made-up value in place of a trial that failed: FI counts such a trial as "
                "failed. Keep everything else as it is.")

    def for_reader(self, limit: int = 2500) -> str:
        """What the code's reader is shown: each setting's returned values (at most a few per name) and its failures."""
        if self.load_error:
            return f"FI could not load the simulation to run it: {self.load_error}"
        parts = [f"FI ran the simulation with its own harness before this reading: {self.says()}."]
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
                "settings": [{"key": s.key, "planned": s.planned, "ok": s.ok, "failed": s.failed, "timed_out": s.timed_out,
                              "reasons": [[r, n] for r, n in s.reasons[:3]]} for s in self.settings]}


def _short(v: Any) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        text = json.dumps(v, default=str)
        return text if len(text) <= 40 else text[:37] + "..."
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return f"{v:.4g}"


def key_of(simulate: Path, grid: dict[str, Any], deterministic: bool, thresholds: dict[str, Any] | None,
           in_each_run: dict[str, list[Any]] | None = None) -> str:
    """What the small run depends on: the simulation and its package (as FI keys its trials), the settings it uses, the
    thresholds the trials are given and the values each run is given to compare inside it."""
    base = _trials._run_key(Path(simulate), {"grid": grid, "thresholds": thresholds or {}}, TRIALS, 0, deterministic)
    inside = f"|{json.dumps(in_each_run, sort_keys=True, default=str)}" if in_each_run else ""
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
    parts_run = bool(getattr(grid, "sub_studies", None)) and len(chosen) > 1
    total = len(chosen)
    for i, cell in enumerate(chosen):
        sub = folder / f"s{i}"
        sub.mkdir(parents=True, exist_ok=True)
        plan = _trials._plan(quest_root, module, {k: [v] for k, v in cell.items()}, runs_per_setting=TRIALS,
                             base_seed=base_seed, deterministic=deterministic, folder=sub, out_name="cell{index}.out.jsonl",
                             paired=paired, thresholds=thresholds, in_each_run=in_each_run)
        for task in plan:
            setting = Setting(task["key"], len(task["trials"]))
            out.append(setting)
            left = int(timeout_s - (time.monotonic() - started))
            if parts_run:
                # every part is run: each setting gets an equal share of what is left, so one slow part cannot take the time of the rest
                left = left // max(1, total - len(out) + 1)
            if left <= 0:
                setting.timed_out = True
                continue
            result = await executor.execute([str(python), _trials.HARNESS_PATH.as_posix(), task["spec"], task["out"]],
                                            cwd=quest_root, timeout_s=max(1, left), env=env)
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
    return SmallRun(out, load_error, time.monotonic() - started)


def write_record(quest_root: Path | str, key: str, run_: SmallRun) -> None:
    try:
        path = Path(quest_root) / RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, **run_.record(), "for_reader": run_.for_reader()}, indent=1),
                        encoding="utf-8")
    except OSError:
        pass


def read_record(quest_root: Path | str) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
