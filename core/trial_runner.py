"""FI runs the experiment's trials and keeps their record; the experiment's code only says what one trial does.

The re-audit's P0-2: the script that ran the experiment also wrote the trial ledger that was the evidence it ran, so a
wrong or dishonest script could write both. Now the generated simulation exposes one function,

    run_trial(cell: dict, trial_id: int, seed: int) -> dict[str, float]      (a stochastic study)
    run_cell(cell: dict) -> dict[str, float]                                  (a deterministic one)

and, when the protocol has oracles, ``oracle() -> dict[str, float]`` (the values it computes where the answer is known).
FI owns everything around them:

- the cells: every combination of the frozen protocol's grid (:func:`cells`), in a fixed order;
- the trials: ``runs_per_setting`` per cell in all (one for ``run_cell``), each with its own seed that FI derives from
  the quest's base seed, the cell and the trial number (:func:`trial_seed`), so a trial is reproducible on its own;
- the processes: one child process per cell runs that cell's trials (a crash or a hang costs that cell, and cells do
  not share state), through the quest's own executor, with a small harness FI writes fresh for every run;
- the record: the child returns each trial's values, status, time and warnings in a results file; FI, and only FI,
  writes the ledger (``raw/ledger.jsonl``: a ``planned`` line before a cell starts, then one line per trial) and the
  per-cell summary the analysis script reads (``raw/trials.json``: each metric's values, count and total per cell).

A trial the child never reported (the cell's process crashed or ran out of time) is recorded as failed, with why. The
ledger rows use the run-manifest schema (``cell``, ``trial``, ``status``, ``reason``), so the existing checker
(:mod:`core.run_manifest`) reads them; the counts it checks are now FI's own.

On a cluster (``execution.background_jobs``), the settings run as one job-array task each instead of one local process
each: FI writes the harness, one spec per setting and the task list into ``job/fi/`` (:func:`prepare_cluster`), the
experiment's ``code/submit.py`` submits the array the way the cluster's skill says and reports it pending or done
(:mod:`core.job_watch`), and when it is done FI reads each task's results and writes the ledger and the summary itself
(:func:`collect_cluster`), with the same checks as a local run.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
import math
import re
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

RAW_DIRNAME = "raw"
LEDGER_NAME = "ledger.jsonl"
SUMMARY_NAME = "trials.json"
HARNESS_PATH = Path(".fi") / "trial_harness.py"
#: The environment variable that tells the analysis script where FI's per-cell summary is.
RESULTS_ENV = "FI_TRIALS"
#: On a cluster: the folder FI writes the job array's tasks into, the task list, and the variable naming it.
CLUSTER_DIR = Path("job") / "fi"
TASKS_NAME = "tasks.json"
TASKS_ENV = "FI_TASKS"
CLUSTER_RECORD = Path(".fi") / "trials" / "cluster.json"
SUBMIT_NAME = "submit.py"

# The harness runs in the quest's own Python (venv, shared interpreter or container), which may not have FI installed:
# it is self-contained, standard library only. It loads the simulation from its file, calls the entry function for
# each trial it is given, and writes one JSON line per trial to the results file the parent named. The simulation's own
# prints go to stderr, so nothing it prints can pass for a result line.
HARNESS_SOURCE = r'''
import importlib.util, json, math, sys, time, traceback, warnings

def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)

def main():
    import os
    spec_path = sys.argv[1]
    with open(spec_path, encoding="utf-8") as f:  # closed before it is deleted: Windows refuses to delete an open file
        spec = json.load(f)
    out = open(sys.argv[2], "w", encoding="utf-8")
    nonce = spec.pop("nonce")
    # The frozen protocol's thresholds (what counts as an outbreak, a success): run_trial reads them from here, never
    # from a number of its own.
    os.environ["FI_THRESHOLDS"] = json.dumps(spec.pop("thresholds", None) or {})
    # What names the results file and the nonce goes before the simulation is loaded: the spec file is deleted and argv
    # cleared, so code that looks for them has to dig through this process's memory rather than read a path.
    if not spec.pop("keep_spec", False):  # a cluster's scheduler may run a task again: its spec stays there
        try:
            os.remove(spec_path)
        except OSError:
            pass
    del sys.argv[1:]
    # The simulation's own folder is where its imports are (a helper module beside simulate.py).
    sys.path.insert(0, os.path.dirname(os.path.abspath(spec["module"])))
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        loader = importlib.util.spec_from_file_location("fi_experiment", spec["module"])
        mod = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(mod)
        fn = getattr(mod, spec["entry"], None)
        if not callable(fn):
            raise AttributeError(f"{spec['module']} has no function {spec['entry']}()")
    except BaseException as e:
        out.write(json.dumps({"nonce": nonce, "load_error": f"{type(e).__name__}: {e}"[:500]}) + "\n")
        out.close()
        sys.exit(3)
    for trial in spec["trials"]:
        t0 = time.monotonic()
        row = {"nonce": nonce, "trial": trial["trial"], "seed": trial.get("seed")}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if spec["entry"] == "run_trial":
                    value = fn(dict(spec["cell"]), trial["trial"], trial["seed"])
                elif spec["entry"] == "run_cell":
                    value = fn(dict(spec["cell"]))
                else:
                    value = fn()
                if not isinstance(value, dict) or not value:
                    raise TypeError(f"{spec['entry']}() returned {type(value).__name__}, not a dict of numbers")
                bad = [k for k, v in value.items() if not _number(v)]
                if bad:
                    raise TypeError(f"{spec['entry']}() returned non-numbers for {bad[:5]}")
                row.update(status="ok", values={str(k): float(v) for k, v in value.items()})
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                row.update(status="failed", reason=f"{type(e).__name__}: {e}"[:300],
                           traceback=traceback.format_exc()[-1500:])
        row["duration_s"] = round(time.monotonic() - t0, 4)
        row["warnings"] = [f"{w.category.__name__}: {w.message}"[:300] for w in caught][:20]
        for w in caught:  # printed as Python prints them, so the numeric-warning check reads them from stderr
            sys.stderr.write(f"{w.filename}:{w.lineno}: {w.category.__name__}: {w.message}\n")
        out.write(json.dumps(row, allow_nan=True) + "\n")
        out.flush()
    out.close()
    sys.stdout = real_stdout

main()
'''


def cells(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination of the grid's values, axes in the grid's order, values in theirs. An empty grid is one cell
    with no axes (a study that runs one setting)."""
    axes = list(grid.items())
    if not axes:
        return [{}]
    return [dict(zip((a for a, _ in axes), combo)) for combo in product(*(v for _, v in axes))]


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return repr(value)
    return str(value)


def cell_key(cell: dict[str, Any]) -> str:
    """``"R0=0.9,N=100"``: the run-manifest checker's way of naming a cell."""
    return ",".join(f"{axis}={_fmt(value)}" for axis, value in cell.items())


def trial_seed(base: int, key: str, trial: int) -> int:
    """A 32-bit seed for one trial, fixed by the quest's base seed, the cell and the trial number: the same trial gets
    the same seed on a rerun, and no two trials of a study share one by construction of their inputs. A paired design
    passes ``key=""``: trial ``t`` of every setting then gets the same seed (common random numbers), which is what makes
    joining the settings' trials by their number a pairing."""
    digest = hashlib.sha256(f"{int(base)}|{key}|{int(trial)}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


@dataclass
class CellRun:
    """What one cell's process did."""

    key: str
    cell: dict[str, Any]
    planned: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    returncode: int = 0
    timed_out: bool = False
    stderr: str = ""
    load_error: str = ""


@dataclass
class TrialRun:
    """What a whole study's trials did, as FI recorded them."""

    cells: list[CellRun]
    ledger_path: Path
    summary_path: Path

    @property
    def failed_trials(self) -> int:
        return sum(1 for c in self.cells for r in c.rows if r.get("status") != "ok")

    @property
    def ok_trials(self) -> int:
        return sum(1 for c in self.cells for r in c.rows if r.get("status") == "ok")

    def ledger_rows(self) -> list[dict[str, Any]]:
        """The ledger's trial rows in the run-manifest schema, for :func:`core.run_manifest.manifest_from_ledger`."""
        return [
            {"cell": c.key, "trial": r["trial"], "status": "ok" if r.get("status") == "ok" else "failed",
             **({"reason": r["reason"]} if r.get("reason") else {})}
            for c in self.cells for r in c.rows
        ]

    def stderr(self) -> str:
        """Every cell's stderr (its warnings and prints), for the numeric-warning scan."""
        return "\n".join(c.stderr for c in self.cells if c.stderr)


def _append(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, allow_nan=True, default=str) + "\n")


def _summary(runs: list[CellRun], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per cell, each metric's values over the trials that succeeded, with their count and total: what the analysis
    script reads (``FI_TRIALS``) and what FI's statistics are computed from."""
    out = []
    for run in runs:
        metrics: dict[str, list[float]] = {}
        trials_of: dict[str, list[int]] = {}
        for row in run.rows:
            if row.get("status") != "ok":
                continue
            for name, value in (row.get("values") or {}).items():
                metrics.setdefault(name, []).append(value)
                trials_of.setdefault(name, []).append(row.get("trial"))
        out.append({
            "cell": run.cell,
            "key": run.key,
            "planned": run.planned,
            "ok": sum(1 for r in run.rows if r.get("status") == "ok"),
            "failed": sum(1 for r in run.rows if r.get("status") != "ok"),
            "metrics": {
                name: {
                    "values": values,
                    # The trial each value came from (FI's own number): what a paired design joins on.
                    "trials": trials_of.get(name, []),
                    "count": len(values),
                    "total": math.fsum(v for v in values if math.isfinite(v)),
                    "non_finite": sum(1 for v in values if not math.isfinite(v)),
                }
                for name, values in metrics.items()
            },
        })
    return {"schema": "fi.trials/v1", "thresholds": dict(thresholds or {}), "cells": out}


def _plan(quest_root: Path, module: Path | str, grid: dict[str, list[Any]], *, runs_per_setting: int, base_seed: int,
          deterministic: bool, folder: Path, out_name: str, keep_spec: bool = False,
          paired: bool = False, thresholds: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Write one spec per cell (the simulation file, the entry, the cell, its trials with their seeds, a nonce) into
    ``folder`` and return the plan: per cell its key, cell, trials, nonce, and the spec and results paths relative to
    ``quest_root``."""
    quest_root = Path(quest_root)
    entry = "run_cell" if deterministic else "run_trial"
    per_cell = 1 if deterministic else max(1, int(runs_per_setting))
    plan = []
    for index, cell in enumerate(cells(grid)):
        key = cell_key(cell)
        trials = [{"trial": t, "seed": None if deterministic else trial_seed(base_seed, "" if paired else key, t)}
                  for t in range(per_cell)]
        spec_path = folder / f"cell{index}.json"
        out_path = folder / out_name.format(index=index)
        out_path.unlink(missing_ok=True)
        nonce = hashlib.sha256(f"{time.time_ns()}|{index}|{id(trials)}".encode()).hexdigest()[:24]
        spec_path.write_text(json.dumps({"module": str(module).replace("\\", "/"), "entry": entry, "cell": cell,
                                         "trials": trials, "nonce": nonce, "keep_spec": keep_spec,
                                         "thresholds": dict(thresholds or {})}, default=str),
                             encoding="utf-8")
        plan.append({"index": index, "key": key, "cell": cell, "trials": trials, "nonce": nonce, "entry": entry,
                     "spec": spec_path.relative_to(quest_root).as_posix(),
                     "out": out_path.relative_to(quest_root).as_posix()})
    return plan


def _collect(quest_root: Path, plan: list[dict[str, Any]], results: dict[int, Any], *, run_id: str,
             thresholds: dict[str, Any] | None, not_reported: str) -> TrialRun:
    """Read each cell's results file, keep the rows the harness FI started wrote (the cell's nonce, a trial it was
    given, reported once, with the seed it was given), and write the ledger and the summary. ``results`` holds each
    cell's process result by index (a cluster task has none: ``not_reported`` says why a trial is missing then)."""
    quest_root = Path(quest_root)
    raw = quest_root / RAW_DIRNAME
    raw.mkdir(parents=True, exist_ok=True)
    ledger = raw / LEDGER_NAME
    ledger.write_text("", encoding="utf-8")
    runs: list[CellRun] = []
    for task in plan:
        key, trials, nonce = task["key"], task["trials"], task["nonce"]
        per_cell = len(trials)
        _append(ledger, {"event": "planned", "run_id": run_id, "cell": key, "trials": per_cell,
                         "entry": task["entry"], "at": time.time()})
        result = results.get(task["index"])
        run = CellRun(key=key, cell=task["cell"], planned=per_cell,
                      returncode=getattr(result, "returncode", None) if result is not None else None,
                      timed_out=bool(getattr(result, "timed_out", False)),
                      stderr=(getattr(result, "stderr", "") or "") if result is not None else "")
        reported: dict[int, dict[str, Any]] = {}
        twice: set[int] = set()
        out_path = quest_root / task["out"]
        try:
            lines = out_path.read_text(encoding="utf-8").splitlines() if out_path.is_file() else []
        except OSError:
            lines = []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("nonce") != nonce:
                continue  # not written by the harness FI started for this cell
            if row.get("load_error"):
                run.load_error = str(row["load_error"])
            elif isinstance(row.get("trial"), int) and row["trial"] in range(per_cell):
                if row["trial"] in reported:
                    twice.add(row["trial"])
                reported.setdefault(row["trial"], row)
        if run.load_error:
            why_missing = f"the simulation could not be loaded ({run.load_error})"
        elif result is None:
            why_missing = not_reported
        elif run.timed_out:
            why_missing = "the study's time (execution.timeout_s) ran out before this setting's trial"
        elif result.returncode == 0:
            why_missing = "the cell's process ended normally but never reported this trial"
        else:
            why_missing = f"the cell's process stopped (exit code {result.returncode}) before this trial"
        for t in trials:
            row = reported.get(t["trial"]) or {"trial": t["trial"], "seed": t["seed"], "status": "failed",
                                              "reason": why_missing}
            if row.get("seed") != t["seed"]:
                row = {**row, "status": "failed", "reason": "the reported seed is not the one FI gave this trial"}
            if t["trial"] in twice:
                row = {**row, "status": "failed", "reason": "this trial was reported more than once"}
            run.rows.append(row)
            _append(ledger, {
                "event": "trial", "run_id": run_id, "cell": key, "trial": t["trial"], "seed": t["seed"],
                "status": "ok" if row.get("status") == "ok" else "failed",
                **({"reason": row["reason"]} if row.get("reason") else {}),
                "duration_s": row.get("duration_s"),
                "values_sha256": hashlib.sha256(
                    json.dumps(row.get("values") or {}, sort_keys=True, allow_nan=True).encode("utf-8")
                ).hexdigest(),
                "warnings": len(row.get("warnings") or []),
            })
        runs.append(run)
    summary = raw / SUMMARY_NAME
    summary.write_text(json.dumps(_summary(runs, thresholds), indent=1, allow_nan=True), encoding="utf-8")
    return TrialRun(cells=runs, ledger_path=ledger, summary_path=summary)


async def run_trials(
    executor: Any, python: Path | str, quest_root: Path, module: Path | str, grid: dict[str, list[Any]], *,
    runs_per_setting: int, base_seed: int, deterministic: bool, timeout_s: int, env: dict[str, str] | None = None,
    run_id: str = "", thresholds: dict[str, Any] | None = None, paired: bool = False,
) -> TrialRun:
    """Run every cell of ``grid`` in its own process and record every trial (see the module docstring). ``module`` is
    the simulation file relative to ``quest_root``; ``timeout_s`` bounds the whole study."""
    quest_root = Path(quest_root)
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    harness = quest_root / HARNESS_PATH
    harness.write_text(HARNESS_SOURCE, encoding="utf-8")  # fresh every run: nothing the experiment wrote is run
    (quest_root / RUN_RECORD).unlink(missing_ok=True)  # an earlier run's record never stands beside this run's ledger
    plan = _plan(quest_root, module, grid, runs_per_setting=runs_per_setting, base_seed=base_seed,
                 deterministic=deterministic, folder=work, out_name="cell{index}.out.jsonl", paired=paired,
                 thresholds=thresholds)
    results: dict[int, Any] = {}
    started = time.monotonic()  # timeout_s bounds the whole study, as it bounded one simulation script before
    for task in plan:
        left = int(timeout_s - (time.monotonic() - started))
        if left <= 0:
            results[task["index"]] = type("NotRun", (), {"returncode": -1, "timed_out": True, "stderr": ""})()
            continue
        results[task["index"]] = await executor.execute(
            [str(python), HARNESS_PATH.as_posix(), task["spec"], task["out"]],
            cwd=quest_root, timeout_s=max(1, left), env=env,
        )
    return _collect(quest_root, plan, results, run_id=run_id, thresholds=thresholds, not_reported="")


def prepare_cluster(quest_root: Path, module: Path | str, grid: dict[str, list[Any]], *, runs_per_setting: int,
                    base_seed: int, deterministic: bool, key: str, paired: bool = False,
                    thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """The job array for a cluster: FI's harness, one spec per setting and the task list in ``job/fi/``, and FI's own
    record of the plan (``.fi/trials/cluster.json``). Kept as it is while ``key`` (the simulation and the protocol) is
    the same, so every check of a submitted job sees the tasks that were submitted. Returns the record."""
    quest_root = Path(quest_root)
    record_path = quest_root / CLUSTER_RECORD
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and record.get("key") == key:
        return record
    folder = quest_root / CLUSTER_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*"):
        if old.is_file():
            old.unlink()
    # A job submitted for an earlier plan is not this plan's: submit.py submits again (its state goes), and the new
    # tasks write to files named for this plan, so a task of the old job still running cannot write into them.
    state = quest_root / "job" / "state.json"
    if state.is_file():
        state.replace(state.with_name("state.previous.json"))  # kept, not deleted: it may name a job still running
    (folder / "harness.py").write_text(HARNESS_SOURCE, encoding="utf-8")
    plan = _plan(quest_root, module, grid, runs_per_setting=runs_per_setting, base_seed=base_seed,
                 deterministic=deterministic, folder=folder, out_name="out{index}-" + key[:10] + ".jsonl", keep_spec=True,
                 paired=paired, thresholds=thresholds)
    harness = (CLUSTER_DIR / "harness.py").as_posix()
    tasks = {
        "count": len(plan),
        "how": "Run task i as: <the cluster's python> " + harness + " <spec> <out>, from the quest folder; one task per "
               "setting, all of them independent. Each writes its results to <out>; FI reads them.",
        "tasks": [{"index": t["index"], "setting": t["key"], "trials": len(t["trials"]),
                   "argv": [harness, t["spec"], t["out"]]} for t in plan],
    }
    (folder / TASKS_NAME).write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    from core.attempt_records import script_hashes

    # The code the tasks will import, as it was when they were submitted: compared when their results are collected.
    record = {"key": key, "plan": plan, "code_at_submit": script_hashes(quest_root)}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, default=str), encoding="utf-8")
    return record


def code_changed_while_queued(quest_root: Path, record: dict[str, Any], *, local: tuple[str, ...] = ()) -> list[str]:
    """The files in ``code/`` (paths relative to it) that differ from when the job was submitted, but for ``local``
    ones (the analysis and the submit script, which run here, not in the tasks); records them in
    ``.fi/trials/cluster.json`` so the attempt's context says its code is not known. A record from before the
    submission was hashed has nothing to compare: it is named as unknown."""
    from core.attempt_records import script_hashes

    quest_root = Path(quest_root)
    before = record.get("code_at_submit")
    if not isinstance(before, dict):
        changed = ["(the code at submission was not recorded)"]
    else:
        now = script_hashes(quest_root)
        # A file the submit script or a task wrote next to the code after submission is not code that changed.
        changed = sorted(f for f in before if f not in local and before.get(f) != now.get(f))
    record["code_changed_while_queued"] = changed
    try:
        (quest_root / CLUSTER_RECORD).write_text(json.dumps(record, default=str), encoding="utf-8")
    except OSError:
        pass
    return changed


def _forget_queued_changes(quest_root: Path) -> None:
    path = Path(quest_root) / CLUSTER_RECORD
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(record, dict) and record.pop("code_changed_while_queued", None) is not None:
        try:
            path.write_text(json.dumps(record, default=str), encoding="utf-8")
        except OSError:
            pass


def collect_cluster(quest_root: Path, record: dict[str, Any], *, run_id: str = "",
                    thresholds: dict[str, Any] | None = None) -> TrialRun:
    """The ledger and the summary from the job array's results, checked as a local run's are."""
    return _collect(Path(quest_root), record.get("plan") or [], {}, run_id=run_id, thresholds=thresholds,
                    not_reported="the cluster task for this setting reported no result for this trial (its own log "
                                 "on the cluster says why)")


_ENTRY_RE = re.compile(r"^def\s+(run_trial|run_cell|oracle)\s*\(", re.MULTILINE)


def entries(simulate: Path) -> set[str]:
    """Which of ``run_trial`` / ``run_cell`` / ``oracle`` the simulation defines at its top level (read, not run)."""
    try:
        return set(_ENTRY_RE.findall(Path(simulate).read_text(encoding="utf-8")))
    except OSError:
        return set()


class TrialsRunner:
    """Stands in for the executor when the simulation follows the trial contract, as ``split_run.SplitRunner`` does for
    the older two-script one: running ``experiment.py`` first runs every trial of the protocol (:func:`run_trials`,
    one process per cell, FI's ledger), then the analysis with ``FI_TRIALS`` naming FI's per-cell summary. What it
    returns is the analysis's run, its stderr led by every cell's (so the simulation's warnings reach the numeric
    check). ``failed_script`` says which script to repair; ``last`` keeps the trial run for the checks after it."""

    def __init__(self, executor: Any, *, quest_root: Path, protocol: Any, deterministic: bool,
                 simulate: Path, analysis: Path, log: Any = None, submit: Path | None = None) -> None:
        self.executor = executor
        self.quest_root = Path(quest_root)
        self._protocol = protocol
        self.deterministic = deterministic
        self.simulate = Path(simulate)
        self.analysis = Path(analysis)
        self.log = log
        self.failed_script: str | None = None
        self.last: TrialRun | None = None
        # On a cluster: the experiment's script that submits the job array FI prepared and reports it pending or done.
        self.submit = Path(submit) if submit is not None else None

    async def execute(self, cmd: list[str], *, cwd: Path, timeout_s: int, env: dict[str, str] | None = None) -> Any:
        from core.execution import ExecutionResult

        if len(cmd) != 2 or Path(cmd[1]).name != self.analysis.name:
            return await self.executor.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)
        started = time.monotonic()
        base = int((env or {}).get("FI_REPLICATE_SEED") or 0)
        protocol = (self._protocol() if callable(self._protocol) else self._protocol) or {}
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        runs = int(protocol.get("runs_per_setting") or 1)
        key = _run_key(self.simulate, protocol, runs, base, self.deterministic)
        # A paired metric compares the settings trial by trial: each trial number gets one seed across the settings.
        paired = any(isinstance(m, dict) and m.get("paired") for m in protocol.get("metrics") or [])
        run = _load_run(self.quest_root, key)
        if run is not None:
            if self.log is not None:
                self.log.info("[execute] simulate.py and the protocol are unchanged: the trials FI already ran are used")
        elif self.submit is not None:
            # A cluster: the tasks are FI's, the submission is the experiment's; a pending job returns as it is and
            # the quest waits for it (core/job_watch.py).
            from core import job_watch

            record = prepare_cluster(
                self.quest_root, self.simulate.relative_to(self.quest_root).as_posix(), grid,
                runs_per_setting=runs, base_seed=base, deterministic=self.deterministic, key=key, paired=paired,
                thresholds=protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None,
            )
            submitted = await self.executor.execute(
                [cmd[0], str(self.submit)], cwd=cwd, timeout_s=timeout_s,
                env={**(env or {}), TASKS_ENV: (CLUSTER_DIR / TASKS_NAME).as_posix()},
            )
            job = job_watch.job_of(_last_result_json(submitted.stdout or ""))
            if submitted.returncode != 0 or job is None or job.get("status") == job_watch.FAILED:
                self.failed_script = self.submit.name
                why = ("" if submitted.returncode != 0 else
                       "\nsubmit.py must end with a RESULT_JSON line holding fi_job (pending or done)" if job is None
                       else f"\nthe job failed: {job.get('note') or ''}")
                return ExecutionResult(returncode=submitted.returncode or 1, stdout=submitted.stdout or "",
                                       duration_s=time.monotonic() - started,
                                       stderr=((submitted.stderr or "") + why).strip(), timed_out=submitted.timed_out)
            if job.get("status") == job_watch.PENDING:
                self.failed_script = None
                _note_job(self.quest_root, record, job)
                return submitted
            code_dir = self.quest_root / "code"
            local = tuple(p.resolve().relative_to(code_dir.resolve()).as_posix()
                          for p in (self.analysis, self.submit) if p.resolve().is_relative_to(code_dir.resolve()))
            changed = code_changed_while_queued(self.quest_root, record, local=local)
            if changed and self.log is not None:
                self.log.warning("[execute] the code changed while the cluster job was queued (%s): the tasks may have "
                                 "run other code than FI has now; the attempt's record says so", ", ".join(changed[:10]))
            run = collect_cluster(
                self.quest_root, record,
                thresholds=protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None,
            )
            _save_run(self.quest_root, key, run)
            if self.log is not None:
                self.log.info("[execute] the cluster job is done: FI read every setting's results and wrote the ledger")
        else:
            # Run here, not on a cluster: what an earlier cluster job's code did while queued is not this run's.
            _forget_queued_changes(self.quest_root)
            run = await run_trials(
                self.executor, cmd[0], self.quest_root, self.simulate.relative_to(self.quest_root).as_posix(), grid,
                runs_per_setting=runs, base_seed=base, deterministic=self.deterministic, timeout_s=timeout_s, env=env,
                thresholds=protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None,
                paired=paired,
            )
            _save_run(self.quest_root, key, run)
        self.last = run
        if self.log is not None:
            self.log.info("[execute] FI ran %d trial(s) in %d cell(s): %d ok, %d failed (ledger: %s)",
                          run.ok_trials + run.failed_trials, len(run.cells), run.ok_trials, run.failed_trials,
                          run.ledger_path.relative_to(self.quest_root).as_posix())
        load_errors = sorted({c.load_error for c in run.cells if c.load_error})
        if run.ok_trials == 0:
            self.failed_script = self.simulate.name
            reason = load_errors[0] if load_errors else next(
                (r.get("reason") for c in run.cells for r in c.rows if r.get("reason")), "every trial failed")
            return ExecutionResult(returncode=1, stdout="", duration_s=time.monotonic() - started,
                                   stderr=f"{run.stderr()}\nFI ran no trial successfully: {reason}".strip())
        # Relative to the quest folder the analysis runs in: the same path inside a container (/work) as on the host.
        analysis_env = {**(env or {}), RESULTS_ENV: run.summary_path.relative_to(self.quest_root).as_posix(),
                        "FI_RAW_DIR": run.summary_path.parent.relative_to(self.quest_root).as_posix()}
        result = await self.executor.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=analysis_env)
        self.failed_script = None if result.returncode == 0 else self.analysis.name
        return ExecutionResult(
            returncode=result.returncode, stdout=result.stdout, duration_s=time.monotonic() - started,
            stderr=(run.stderr() + "\n" + (result.stderr or "")).strip(), timed_out=result.timed_out,
        )


def _note_job(quest_root: Path, record: dict[str, Any], job: dict[str, Any]) -> None:
    """A job id FI has not seen for this plan is a new submission (the last one failed or was cancelled): the results
    an earlier job left are removed, so only this job's are read when it is done."""
    job_id = str(job.get("id") or "")
    if not job_id or record.get("job_id") == job_id:
        return
    if record.get("job_id"):
        # Another job than the one FI saw for this plan. The first id FI sees is this plan's first job: its tasks may
        # already be writing, and there is nothing earlier to clear.
        for task in record.get("plan") or []:
            (Path(quest_root) / task["out"]).unlink(missing_ok=True)
    record["job_id"] = job_id
    try:
        (Path(quest_root) / CLUSTER_RECORD).write_text(json.dumps(record, default=str), encoding="utf-8")
    except OSError:
        pass


def _last_result_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT_JSON:"):
            try:
                value = json.loads(line[len("RESULT_JSON:"):].strip())
            except ValueError:
                return None
            return value if isinstance(value, dict) else None
    return None


RUN_RECORD = Path(".fi") / "trials" / "run.json"


def _run_key(simulate: Path, grid: Any, runs: int, base: int, deterministic: bool) -> str:
    """What decides a set of trials: the simulation's text, the protocol (``grid``: the whole of it is passed), the
    count, the base seed and whether each setting runs once (``run_cell``)."""
    try:
        text = Path(simulate).read_bytes()
    except OSError:
        text = b""
    return hashlib.sha256(text + json.dumps([grid, runs, base, deterministic], sort_keys=True, default=str).encode()).hexdigest()


def _file_hashes(quest_root: Path) -> dict[str, str]:
    out = {}
    for name in (LEDGER_NAME, SUMMARY_NAME):
        try:
            out[name] = hashlib.sha256((Path(quest_root) / RAW_DIRNAME / name).read_bytes()).hexdigest()
        except OSError:
            out[name] = ""
    return out


def _save_run(quest_root: Path, key: str, run: TrialRun) -> None:
    record = {"key": key, "files": _file_hashes(quest_root), "cells": [
        {"key": c.key, "cell": c.cell, "planned": c.planned, "rows": c.rows, "returncode": c.returncode,
         "timed_out": c.timed_out, "stderr": c.stderr[-20000:], "load_error": c.load_error} for c in run.cells]}
    try:
        (Path(quest_root) / RUN_RECORD).write_text(json.dumps(record, allow_nan=True, default=str), encoding="utf-8")
    except OSError:
        pass


def _load_run(quest_root: Path, key: str) -> TrialRun | None:
    """The trials already run for ``key``, when their record, FI's ledger and the summary are all still there: a rerun
    of the analysis alone (a review's revision, a repaired experiment.py) does not run the simulation again."""
    root = Path(quest_root)
    ledger, summary = root / RAW_DIRNAME / LEDGER_NAME, root / RAW_DIRNAME / SUMMARY_NAME
    try:
        record = json.loads((root / RUN_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("key") != key or not ledger.is_file() or not summary.is_file():
        return None
    if record.get("files") != _file_hashes(root):
        return None  # the ledger or the summary was changed since FI wrote them: the trials are run again
    return TrialRun(cells=[CellRun(**c) for c in record.get("cells") or []], ledger_path=ledger, summary_path=summary)


async def run_oracle(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *, timeout_s: int,
                     env: dict[str, str] | None = None, thresholds: dict[str, Any] | None = None) -> tuple[dict[str, float] | None, str]:
    """Call the simulation's ``oracle()`` in its own process: ``(values, "")``, or ``(None, why)`` when it could not."""
    quest_root = Path(quest_root)
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    (quest_root / HARNESS_PATH).write_text(HARNESS_SOURCE, encoding="utf-8")
    spec = work / "oracle.json"
    out = work / "oracle.out.jsonl"
    out.unlink(missing_ok=True)
    nonce = hashlib.sha256(f"oracle|{time.time_ns()}".encode()).hexdigest()[:24]
    spec.write_text(json.dumps({"module": str(module).replace(chr(92), "/"), "entry": "oracle", "cell": {},
                                "trials": [{"trial": 0, "seed": None}], "nonce": nonce,
                            "thresholds": dict(thresholds or {})}, default=str), encoding="utf-8")
    result = await executor.execute(
        [str(python), HARNESS_PATH.as_posix(), spec.relative_to(quest_root).as_posix(), out.relative_to(quest_root).as_posix()],
        cwd=quest_root, timeout_s=timeout_s, env=env,
    )
    try:
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        rows = []
    for row in rows:
        if not isinstance(row, dict) or row.get("nonce") != nonce:
            continue  # not written by the harness FI started
        if row.get("load_error"):
            return None, str(row["load_error"])
        if row.get("status") == "ok":
            return dict(row.get("values") or {}), ""
        if row.get("status") == "failed":
            return None, str(row.get("reason") or "oracle() failed")
    return None, ("oracle() ran out of time" if getattr(result, "timed_out", False)
                  else f"oracle() did not report (exit code {result.returncode})")


def read_ledger(quest_root: Path) -> list[dict[str, Any]] | None:
    """The trial rows of FI's own ledger (run-manifest schema), or ``None`` when FI wrote none for this quest."""
    path = Path(quest_root) / RAW_DIRNAME / LEDGER_NAME
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("event") == "trial":
            rows.append({"cell": row.get("cell"), "trial": row.get("trial"), "status": row.get("status"),
                         **({"reason": row["reason"]} if row.get("reason") else {})})
    return rows


def _value_key(v: Any) -> str | None:
    """A number as the ledger and a JSON round trip both keep it (12 significant digits), or ``None`` if not a number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return f"{float(v):.12g}"


def recorded_values(quest_root: Path) -> tuple[dict[str, Counter], list[str]]:
    """Every value FI's trials returned, per name, over all settings (see :func:`recorded_values_by_cell`)."""
    per_cell, problems = recorded_values_by_cell(quest_root)
    return pooled(per_cell), problems


def pooled(per_cell: dict[str, dict[str, Counter]]) -> dict[str, Counter]:
    """The per-setting values of :func:`recorded_values_by_cell`, added together per name."""
    out: dict[str, Counter] = {}
    for by_name in per_cell.values():
        for name, counts in by_name.items():
            out.setdefault(name, Counter()).update(counts)
    return out


def _trial_id(row: dict[str, Any]) -> int | None:
    """A row's trial number, or ``None`` when it has none (never read as trial 0)."""
    t = row.get("trial")
    return t if isinstance(t, int) and not isinstance(t, bool) and t >= 0 else None


def recorded_rows_by_cell(quest_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Each trial FI ran to the end, as a row: its setting (cell key), trial number, seed and the whole dict
    ``run_trial`` returned for it, checked against the hash FI's ledger holds for that trial (a row whose values no
    longer match is left out; :func:`recorded_values_by_cell` says so). What a mean over a subset of the trials is
    joined on: the same trial's membership and value, never two separate pools."""
    root = Path(quest_root)
    try:
        record = json.loads((root / RUN_RECORD).read_text(encoding="utf-8"))
        lines = (root / RAW_DIRNAME / LEDGER_NAME).read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return {}
    hashes: dict[tuple[str, int], str] = {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if (isinstance(row, dict) and row.get("event") == "trial" and row.get("status") == "ok"
                and _trial_id(row) is not None):
            hashes[(str(row.get("cell")), _trial_id(row))] = str(row.get("values_sha256") or "")
    out: dict[str, list[dict[str, Any]]] = {}
    for cell in record.get("cells") or []:
        key = str(cell.get("key"))
        for row in cell.get("rows") or []:
            if not isinstance(row, dict) or row.get("status") != "ok":
                continue
            values = row.get("values") if isinstance(row.get("values"), dict) else {}
            digest = hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=True).encode("utf-8")).hexdigest()
            if _trial_id(row) is None or hashes.get((key, _trial_id(row))) != digest:
                continue
            out.setdefault(key, []).append({"cell": key, "trial": row.get("trial"), "seed": row.get("seed"),
                                            "values": values})
    return out


#: What a mean over a subset of the trials needs from the simulation under research, said the same way by the check,
#: the repair and the implement prompt.
RETURN_MEMBERSHIP = (
    "Under research, a mean over the trials a proportion counts is recomputed by FI from its own trial record, trial "
    "by trial: run_trial must return, under the proportion's own id `{given}`, 1 (the trial is in the subset) or 0 "
    "(it is not) for every trial, and the quantity the mean averages under the mean's own id `{metric}`; a cut-off "
    "that decides it is read from FI_THRESHOLDS, never written into the script."
)


def given_rows_problems(protocol: dict[str, Any] | None, rows_by_cell: dict[str, list[dict[str, Any]]],
                        result_json: Any, *, ok_trials: int = 0) -> tuple[list[str], list[str]]:
    """Under research, a mean over the trials a proportion counts (a metric with ``given``), checked trial by trial:
    FI takes, from its own record, the trials of each reported stratum (or of the run) whose ``given`` is 1, and the
    analysis's count must be how many there are and its values must be those same trials' values of one quantity
    (or that value divided by the trial's own size setting), as a multiset. ``(analysis's to fix, simulation's to
    fix)``: values or a count that are not those trials' are the analysis's; a record with no 0/1 ``given`` is the
    simulation's (:data:`RETURN_MEMBERSHIP`), and so is a record that cannot be read while the ledger says trials ran
    (``ok_trials``): the trials are run again. Every mapping holding the mean's values is checked, under a stratum
    key or not (a ``summary`` or ``results`` wrapper is the whole run); the values are one quantity the trials
    returned, the same one in every stratum, never a quantity that is only 0 and 1."""
    from core import run_manifest as _rm

    if not isinstance(protocol, dict) or result_json is None:
        return [], []
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    given_of = {str(m["id"]): str(m["given"]) for m in protocol.get("metrics") or []
                if isinstance(m, dict) and m.get("kind") == "mean" and m.get("given") and m.get("id")}
    if not given_of:
        return [], []
    if not rows_by_cell:
        if ok_trials > 0:
            return [], [
                f"FI's record of the trials' own values (.fi/trials/run.json) is missing or unreadable, though its ledger "
                f"says {ok_trials} trial(s) ran to the end: a mean over a subset of the trials cannot be checked trial "
                "by trial without it, so the trials are run again"
            ]
        return [], []
    keyed: dict[frozenset, str] = {}
    for key in rows_by_cell:
        canon, _why = _rm._canonicalize_cell(key, grid)
        if canon is not None:
            keyed[canon] = key
    all_cells = sorted(rows_by_cell)
    sizes = _count_axes(grid)
    analysis: list[str] = []
    simulation: list[str] = []

    def membership(row: dict[str, Any], given: str) -> int | None:
        v = row["values"].get(given)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v not in (0, 1):
            return None
        return int(v)

    for metric, given in sorted(given_of.items()):
        rows = [r for key in all_cells for r in rows_by_cell[key]]
        if any(membership(r, given) is None for r in rows):
            simulation.append(
                f"`{metric}` is a mean over the trials `{given}` counts, but FI's trial record does not hold `{given}` "
                f"as 1 or 0 for every trial, so FI cannot tell which trials the mean is over. "
                + RETURN_MEMBERSHIP.format(given=given, metric=metric)
            )
            continue

        if any(not isinstance(r["values"].get(metric), (int, float)) or isinstance(r["values"].get(metric), bool)
               for r in rows if membership(r, given) == 1):
            simulation.append(
                f"`{metric}` is a mean over the trials `{given}` counts, but FI's trial record does not hold the "
                f"quantity it averages, under `{metric}`, as a number for every trial in the subset. "
                + RETURN_MEMBERSHIP.format(given=given, metric=metric)
            )
            continue

        def forms(subset: list[dict[str, Any]]) -> dict[str, list[float | None]]:
            """The subset's values of ``metric`` as the trials returned them, and each divided by a size setting, as
            the list of per-trial values (``None`` where a trial's value is not a finite number)."""
            plain: list[float | None] = []
            divided: dict[str, list[float | None]] = {axis: [] for axis in sizes}
            for r in subset:
                v = r["values"].get(metric)
                x = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
                plain.append(x)
                parsed = _rm._parse_cell(r["cell"]) or {}
                for axis in sizes:
                    try:
                        a = float(parsed.get(axis, "nan"))
                    except ValueError:
                        a = float("nan")
                    divided[axis].append(x / a if x is not None and a and math.isfinite(a) else None)
            return {metric: plain, **{f"{metric}/{axis}": vals for axis, vals in divided.items()}}

        def same_multiset(reported: list[Any], expected: list[float | None]) -> bool:
            if len(reported) != len(expected):
                return False
            got_nulls = sum(1 for x in reported if x is None)
            if got_nulls != sum(1 for x in expected if x is None):
                return False
            counts = Counter(_exact_key(x) for x in expected if x is not None)
            numbers = [x for x in reported if x is not None]
            return _missing_exact(numbers, counts) == 0

        common: set[str] | None = None
        first_at = ""
        for under, mapping in _rm._mappings_with(result_json, f"{metric}_values"):
            where = f"`{under}`" if under is not None else "the top level"
            cells, why = _rm._stratum_cells(under, grid) if under is not None else (None, None)
            if why:
                continue  # shaped like a stratum the grid does not have: the manifest check reports it
            if not cells and any(axis in mapping for axis in grid):  # a record that names its setting as fields
                analysis.append(
                    f"{where} holds a stratum's values with its setting as fields, not as a key: each setting is its own "
                    "stratum, keyed like `R0=1.5,N=100` (its settings, as `name=value`), so its mean is checked "
                    "against that setting's trials and not against the whole run"
                )
                continue
            # A key that is no setting at all (`summary`, `results`, a list of records) holds the whole run's mean.
            own = [keyed[c] for c in cells if c in keyed] if cells else all_cells
            subset = [r for key in own for r in rows_by_cell.get(key, []) if membership(r, given) == 1]
            reported = _rm._values_of(mapping, metric)
            if not subset and not reported:
                continue
            count, count_why = _rm._whole(mapping.get(f"{given}_count"))
            if count_why is None and count is not None and count != len(subset):
                analysis.append(
                    f"at {where}, `{given}_count` says {count:g}, but FI's trials of those settings returned "
                    f"`{given}` = 1 for {len(subset)} of them: the mean is over every trial in the subset, not a "
                    "selection"
                )
            matched = {name for name, vals in forms(subset).items() if same_multiset(reported, vals)}
            if not matched:
                analysis.append(
                    f"at {where}, `{metric}_values` are not the values of `{metric}` for the trials whose `{given}` is "
                    f"1 (FI took those {len(subset)} trial(s) from its own record, trial by trial): a mean over a "
                    "subset lists each of its trials' own value once, and no trial outside it. " + _rm.DERIVED
                )
            elif common is None:
                common, first_at = matched, where
            elif not (common & matched):
                analysis.append(
                    f"at {where}, `{metric}_values` are the subset's values as `{sorted(matched)[0]}`, but at {first_at} "
                    f"as `{sorted(common)[0]}`: one mean averages one quantity, in the same form, in every stratum"
                )
            else:
                common &= matched
    return analysis, simulation


def recorded_values_by_cell(quest_root: Path) -> tuple[dict[str, dict[str, Counter]], list[str]]:
    """Every value FI's trials returned, per setting (its cell key) and per name (the keys of ``run_trial``'s dict),
    counted, and what stood in the way.

    The values are in FI's run record (``.fi/trials/run.json``, each trial's dict as the harness reported it); each
    trial's dict is checked against the hash FI's ledger holds for it, so a record edited after the trials ran is found
    rather than believed. Only trials that ran to the end count."""
    root = Path(quest_root)
    try:
        record = json.loads((root / RUN_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, []
    hashes: dict[tuple[str, int], str] = {}
    try:
        for line in (root / RAW_DIRNAME / LEDGER_NAME).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (isinstance(row, dict) and row.get("event") == "trial" and row.get("status") == "ok"
                    and _trial_id(row) is not None):
                hashes[(str(row.get("cell")), _trial_id(row))] = str(row.get("values_sha256") or "")
    except OSError:
        return {}, []
    out: dict[str, dict[str, Counter]] = {}
    altered = 0
    for cell in record.get("cells") or []:
        for row in cell.get("rows") or []:
            if not isinstance(row, dict) or row.get("status") != "ok":
                continue
            values = row.get("values") or {}
            digest = hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=True).encode("utf-8")).hexdigest()
            if _trial_id(row) is None or hashes.get((str(cell.get("key")), _trial_id(row))) != digest:
                altered += 1
                continue
            for name, value in values.items():
                key = _value_key(value)
                if key is not None:
                    out.setdefault(str(cell.get("key")), {}).setdefault(str(name), Counter())[key] += 1
    problems = []
    if altered:
        problems = [f"{altered} trial(s) in FI's run record (.fi/trials/run.json) no longer match the ledger's hash of their "
                    f"values: the record was changed after the trials ran, so FI runs the trials again"]
        (root / RUN_RECORD).unlink(missing_ok=True)  # the next run cannot reuse it: the trials are run afresh
    return out, problems


#: Grid axis names that are a size (a population, a number of agents): the only axes a trial's value may be divided by.
_SIZE_AXIS = re.compile(
    r"^(n|n_\w+|\w+_n|size|\w+_size|size_\w+|population|pop\w*|agents|n_?agents|particles|n_?particles|"
    r"individuals|nodes|n_?nodes|households)$",
    re.IGNORECASE,
)


def _count_axes(grid: dict[str, list[Any]]) -> list[str]:
    """The grid axes that are sizes: named like one (N, population, agents, ...: never a seed, a replicate or a rate
    such as R0) and every value a whole number above 1. The only axes a trial's value may be divided by to make it a
    fraction."""
    out = []
    for axis, values in grid.items():
        nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if (_SIZE_AXIS.match(str(axis).strip()) and nums and len(nums) == len(values)
                and all(float(v).is_integer() and v > 1 for v in nums)):
            out.append(axis)
    return out


def _exact_key(v: Any) -> str | None:
    """A number to ten significant digits: equal up to floating-point noise, never up to a rounding a script chose."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return f"{float(v):.10g}"


def _derived_counts(per_cell: dict[str, dict[str, Counter]], cells: list[str], name: str,
                    grid: dict[str, list[Any]]) -> list[Counter]:
    """The values of ``name`` in ``cells`` as they are, and each divided by one of the cell's own size axes
    (:func:`_count_axes`): every form a list of that quantity may take (a final size, or a final size divided by N).
    Keyed by :func:`_exact_key`."""
    from core import run_manifest as _rm

    sizes = _count_axes(grid)
    forms: dict[str | None, Counter] = {None: Counter()}
    for key in cells:
        counts = (per_cell.get(key) or {}).get(name)
        if not counts:
            continue
        parsed = _rm._parse_cell(key) or {}
        for value, n in counts.items():
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            k = _exact_key(x)
            if k is not None:
                forms[None][k] += n
            for axis in sizes:
                try:
                    a = float(parsed.get(axis, "nan"))
                except ValueError:
                    continue
                if a and math.isfinite(a):
                    k = _exact_key(x / a)
                    if k is not None:
                        forms.setdefault(axis, Counter())[k] += n
    return list(forms.values())


#: How close a printed value must be to a trial's: a float32 or a float printed in full passes, a rounding does not.
_VALUE_REL_TOL = 1e-6


def _missing_exact(values: list[Any], counts: Counter) -> int:
    """How many of ``values`` are not among ``counts`` (each counted value used once): equal within a relative
    :data:`_VALUE_REL_TOL` (the noise of storing a number in fewer bits), never within a rounding the script chose."""
    import bisect

    pool = sorted(float(k) for k, n in counts.items() for _ in range(max(int(n), 0)))
    missing = 0
    for x in values:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            continue
        v = float(x)
        tol = max(abs(v) * _VALUE_REL_TOL, 1e-12)
        i = bisect.bisect_left(pool, v - tol)
        if i < len(pool) and pool[i] <= v + tol:
            pool.pop(i)
        else:
            missing += 1
    return missing


def given_values_not_run(protocol: dict[str, Any] | None, per_cell: dict[str, dict[str, Counter]],
                         result_json: Any) -> list[str]:
    """The values of each mean over a subset of the trials (a metric with ``given``) that FI's trials never produced in
    the settings they are reported for, one sentence each. Each list under a stratum key (``R0=1.5``) is held to the
    trials of exactly those settings; one that names no stratum, and one with the stratum in its name
    (``<metric>_R0_1_5_values``), to every setting. A value is a trial's own value of one quantity, or that value divided
    by the trial's own size setting (:data:`core.run_manifest.DERIVED`), compared exactly; where the trials return the proportion as 0/1, each stratum's count is held to FI's own count."""
    from core import run_manifest as _rm

    if not isinstance(protocol, dict) or not per_cell or result_json is None:
        return []
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    means = [str(m["id"]) for m in protocol.get("metrics") or []
             if isinstance(m, dict) and m.get("kind") == "mean" and m.get("given") and m.get("id")]
    names = sorted({n for by_name in per_cell.values() for n in by_name})
    all_cells = sorted(per_cell)
    keyed: dict[frozenset, str] = {}
    for key in per_cell:
        canon, _why = _rm._canonicalize_cell(key, grid)
        if canon is not None:
            keyed[canon] = key
    out: list[str] = []

    def check(metric: str, where: str, values: list[Any], cells: list[str]) -> None:
        numbers = [x for x in values if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if not numbers:
            return
        for name in names:
            if any(not _missing_exact(numbers, form) for form in _derived_counts(per_cell, cells, name, grid)):
                return
        out.append(
            f"the analysis reports `{where}` for `{metric}`, a mean over a subset of the trials, with values FI's trials "
            f"never produced in {'those settings' if len(cells) < len(all_cells) else 'the run'}: "
            + _rm.DERIVED
        )

    given_of = {str(m["id"]): str(m["given"]) for m in protocol.get("metrics") or []
                if isinstance(m, dict) and m.get("kind") == "mean" and m.get("given") and m.get("id")}

    def members(proportion: str, cells: list[str]) -> float | None:
        """How many trials of ``cells`` are in the proportion's subset, when the trials returned it as 0/1; ``None``
        when FI cannot tell (the analysis decides membership itself, from a threshold)."""
        total = 0.0
        seen = False
        for key in cells:
            counts = (per_cell.get(key) or {}).get(proportion)
            if not counts:
                continue
            for value, n in counts.items():
                if float(value) not in (0.0, 1.0):
                    return None
                total += float(value) * n
                seen = True
        return total if seen else None

    ok_trials = {key: max((sum(c.values()) for c in by_name.values()), default=0) for key, by_name in per_cell.items()}
    everything = sum(ok_trials.values())
    for metric in means:
        # The strata the mean is reported for: when FI cannot count their subset itself and they hold under half
        # the run's trials, a selection of the trials cannot be told from the whole (see run_manifest.PARTIAL).
        given = given_of.get(metric, "")
        covered: set[str] = set()
        unverifiable = False
        for under, _mapping in _rm._mappings_with(result_json, f"{metric}_values"):
            cells, _why = _rm._stratum_cells(under, grid) if under is not None else (None, None)
            if cells is None:
                continue
            own = [keyed[c] for c in cells if c in keyed]
            covered.update(own)
            if members(given, own) is None:
                unverifiable = True
        reported = sum(ok_trials.get(k, 0) for k in covered)
        if covered and unverifiable and everything and reported < 0.5 * everything:
            out.append(
                f"`{metric}` is reported for settings holding {reported:g} of the run's {everything:g} trials, and FI "
                f"cannot count its subset (`{given}`) itself, so a selection of the trials cannot be told from all of "
                "them. " + _rm.PARTIAL.format(given=given)
            )
        stem = re.compile(rf"^{re.escape(metric)}_.+_values$")

        def walk(node: Any, under: Any, path: str) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    here = f"{path}.{k}" if path else str(k)
                    if k == f"{metric}_values" and isinstance(v, list):
                        cells, _why = _rm._stratum_cells(under, grid) if under is not None else (None, None)
                        own = [keyed[c] for c in (cells or []) if c in keyed] if cells else all_cells
                        check(metric, here, v, own or all_cells)
                        given = given_of.get(metric, "")
                        count = node.get(f"{given}_count")
                        exact = members(given, own or all_cells) if cells else None
                        if exact is not None and isinstance(count, (int, float)) and not isinstance(count, bool) \
                                and float(count) != exact:
                            out.append(
                                f"at `{path or 'the top level'}`, `{given}_count` says {count:g}, but FI's trials of "
                                f"those settings returned `{given}` = 1 for {exact:g} of them: the mean averages "
                                "every trial in the subset, not a selection"
                            )
                    elif isinstance(k, str) and stem.match(k) and isinstance(v, list):
                        check(metric, here, v, all_cells)
                    else:
                        walk(v, k, here)
            elif isinstance(node, list):
                for v in node[:200]:
                    walk(v, under, path)

        walk(result_json, None, "")
    return out


def _half_step(x: Any) -> float:
    """Half the last printed place of ``x``: what rounding to it could have moved a value by. ``1.2346`` -> 0.00005,
    ``0.0`` -> 0.05, ``1e-05`` -> 0.000005, an int -> 0.5."""
    if isinstance(x, int) and not isinstance(x, bool):
        return 0.5
    text = repr(float(x))
    mantissa, _, exponent = text.partition("e")
    places = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return 0.5 * 10 ** (-(places - int(exponent or 0)))


def _not_among(values: list[Any], recorded: Counter) -> list[float]:
    """The numbers of ``values`` that are not trial values, each trial's value used once. A number printed to fewer
    places than the trial returned (``1.2346`` for ``1.23456789``, ``0.0`` for ``1e-05``, ``3`` for ``2.8``) is that
    value: it matches a recorded value within half its last printed place. The most precise numbers are matched first,
    each to the nearest value left, so a coarse one cannot take the value a precise one needed."""
    left = Counter(recorded)
    pending: list[Any] = []
    for x in values:
        key = _value_key(x)
        if key is None:
            continue
        if left[key] > 0:
            left[key] -= 1
        else:
            pending.append(x)
    if not pending:
        return []
    pool = [float(k) for k, n in left.items() for _ in range(max(n, 0))]
    extra = []
    for x in sorted(pending, key=_half_step):
        value, half = float(x), _half_step(x)
        near = [(abs(v - value), i) for i, v in enumerate(pool) if math.isfinite(v) and abs(v - value) <= half * (1 + 1e-9)]
        if near:
            pool.pop(min(near)[1])
        else:
            extra.append(value)
    return extra


def reported_values_not_run(recorded: dict[str, Counter], result_json: Any) -> list[str]:
    """Each ``<name>_values`` list the analysis printed, for a ``<name>`` FI's trials returned, that holds values those
    trials never produced (or more copies of one than they did): one sentence each.

    The run-manifest check counts a metric's values against the trials; a script could still print that many numbers
    of its own. Under the trial contract FI holds every trial's value, so a list the analysis says it computed from is
    checked value by value. A list named for something the analysis derived itself (no trial returned that name) is not
    FI's to check here; a mean over a subset of the trials is checked by :func:`given_values_not_run`."""
    out: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else str(k)
                if isinstance(k, str) and k.endswith("_values") and isinstance(v, list) and k[:-7] in recorded:
                    extra = _not_among(v, recorded[k[:-7]])
                    if extra:
                        out.append(
                            f"the analysis reports `{here}` with {len(extra)} value(s) FI's trials of `{k[:-7]}` never "
                            f"produced (e.g. {extra[0]:g}): experiment.py must print the values FI_TRIALS holds, as they "
                            f"are; a list of something derived from them needs a name of its own"
                        )
                else:
                    walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node[:200]):
                walk(v, f"{path}[{i}]")

    walk(result_json, "")
    return out
