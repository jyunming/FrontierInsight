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
    the same seed on a rerun, and no two trials of a study share one by construction of their inputs."""
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
        for row in run.rows:
            if row.get("status") != "ok":
                continue
            for name, value in (row.get("values") or {}).items():
                metrics.setdefault(name, []).append(value)
        out.append({
            "cell": run.cell,
            "key": run.key,
            "planned": run.planned,
            "ok": sum(1 for r in run.rows if r.get("status") == "ok"),
            "failed": sum(1 for r in run.rows if r.get("status") != "ok"),
            "metrics": {
                name: {
                    "values": values,
                    "count": len(values),
                    "total": math.fsum(v for v in values if math.isfinite(v)),
                    "non_finite": sum(1 for v in values if not math.isfinite(v)),
                }
                for name, values in metrics.items()
            },
        })
    return {"schema": "fi.trials/v1", "thresholds": dict(thresholds or {}), "cells": out}


def _plan(quest_root: Path, module: Path | str, grid: dict[str, list[Any]], *, runs_per_setting: int, base_seed: int,
          deterministic: bool, folder: Path, out_name: str, keep_spec: bool = False) -> list[dict[str, Any]]:
    """Write one spec per cell (the simulation file, the entry, the cell, its trials with their seeds, a nonce) into
    ``folder`` and return the plan: per cell its key, cell, trials, nonce, and the spec and results paths relative to
    ``quest_root``."""
    quest_root = Path(quest_root)
    entry = "run_cell" if deterministic else "run_trial"
    per_cell = 1 if deterministic else max(1, int(runs_per_setting))
    plan = []
    for index, cell in enumerate(cells(grid)):
        key = cell_key(cell)
        trials = [{"trial": t, "seed": None if deterministic else trial_seed(base_seed, key, t)} for t in range(per_cell)]
        spec_path = folder / f"cell{index}.json"
        out_path = folder / out_name.format(index=index)
        out_path.unlink(missing_ok=True)
        nonce = hashlib.sha256(f"{time.time_ns()}|{index}|{id(trials)}".encode()).hexdigest()[:24]
        spec_path.write_text(json.dumps({"module": str(module).replace("\\", "/"), "entry": entry, "cell": cell,
                                         "trials": trials, "nonce": nonce, "keep_spec": keep_spec}, default=str),
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
    run_id: str = "", thresholds: dict[str, Any] | None = None,
) -> TrialRun:
    """Run every cell of ``grid`` in its own process and record every trial (see the module docstring). ``module`` is
    the simulation file relative to ``quest_root``; ``timeout_s`` bounds the whole study."""
    quest_root = Path(quest_root)
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    harness = quest_root / HARNESS_PATH
    harness.write_text(HARNESS_SOURCE, encoding="utf-8")  # fresh every run: nothing the experiment wrote is run
    plan = _plan(quest_root, module, grid, runs_per_setting=runs_per_setting, base_seed=base_seed,
                 deterministic=deterministic, folder=work, out_name="cell{index}.out.jsonl")
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
                    base_seed: int, deterministic: bool, key: str) -> dict[str, Any]:
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
                 deterministic=deterministic, folder=folder, out_name="out{index}-" + key[:10] + ".jsonl", keep_spec=True)
    harness = (CLUSTER_DIR / "harness.py").as_posix()
    tasks = {
        "count": len(plan),
        "how": "Run task i as: <the cluster's python> " + harness + " <spec> <out>, from the quest folder; one task per "
               "setting, all of them independent. Each writes its results to <out>; FI reads them.",
        "tasks": [{"index": t["index"], "setting": t["key"], "trials": len(t["trials"]),
                   "argv": [harness, t["spec"], t["out"]]} for t in plan],
    }
    (folder / TASKS_NAME).write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    record = {"key": key, "plan": plan}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, default=str), encoding="utf-8")
    return record


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
                runs_per_setting=runs, base_seed=base, deterministic=self.deterministic, key=key,
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
            run = collect_cluster(
                self.quest_root, record,
                thresholds=protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None,
            )
            _save_run(self.quest_root, key, run)
            if self.log is not None:
                self.log.info("[execute] the cluster job is done: FI read every setting's results and wrote the ledger")
        else:
            run = await run_trials(
                self.executor, cmd[0], self.quest_root, self.simulate.relative_to(self.quest_root).as_posix(), grid,
                runs_per_setting=runs, base_seed=base, deterministic=self.deterministic, timeout_s=timeout_s, env=env,
                thresholds=protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None,
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
                     env: dict[str, str] | None = None) -> tuple[dict[str, float] | None, str]:
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
                                "trials": [{"trial": 0, "seed": None}], "nonce": nonce}), encoding="utf-8")
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
