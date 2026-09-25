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

# The harness runs in the quest's own Python (venv, shared interpreter or container), which may not have FI installed:
# it is self-contained, standard library only. It loads the simulation from its file, calls the entry function for
# each trial it is given, and writes one JSON line per trial to the results file the parent named. The simulation's own
# prints go to stderr, so nothing it prints can pass for a result line.
HARNESS_SOURCE = r'''
import importlib.util, json, math, sys, time, traceback, warnings

def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)

def main():
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    out = open(sys.argv[2], "w", encoding="utf-8")
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
        out.write(json.dumps({"load_error": f"{type(e).__name__}: {e}"[:500]}) + "\n")
        out.close()
        sys.exit(3)
    for trial in spec["trials"]:
        t0 = time.monotonic()
        row = {"trial": trial["trial"], "seed": trial.get("seed")}
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


def _summary(runs: list[CellRun]) -> dict[str, Any]:
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
    return {"schema": "fi.trials/v1", "cells": out}


async def run_trials(
    executor: Any, python: Path | str, quest_root: Path, module: Path | str, grid: dict[str, list[Any]], *,
    runs_per_setting: int, base_seed: int, deterministic: bool, timeout_s: int, env: dict[str, str] | None = None,
    run_id: str = "",
) -> TrialRun:
    """Run every cell of ``grid`` in its own process and record every trial (see the module docstring). ``module`` is
    the simulation file relative to ``quest_root``; ``timeout_s`` bounds each cell's process."""
    quest_root = Path(quest_root)
    raw = quest_root / RAW_DIRNAME
    raw.mkdir(parents=True, exist_ok=True)
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    harness = quest_root / HARNESS_PATH
    harness.write_text(HARNESS_SOURCE, encoding="utf-8")  # fresh every run: nothing the experiment wrote is run
    ledger = raw / LEDGER_NAME
    ledger.write_text("", encoding="utf-8")
    entry = "run_cell" if deterministic else "run_trial"
    per_cell = 1 if deterministic else max(1, int(runs_per_setting))
    runs: list[CellRun] = []
    for index, cell in enumerate(cells(grid)):
        key = cell_key(cell)
        trials = [{"trial": t, "seed": None if deterministic else trial_seed(base_seed, key, t)} for t in range(per_cell)]
        spec_path = work / f"cell{index}.json"
        out_path = work / f"cell{index}.out.jsonl"
        out_path.unlink(missing_ok=True)
        spec_path.write_text(json.dumps({"module": str(module), "entry": entry, "cell": cell, "trials": trials},
                                        default=str), encoding="utf-8")
        _append(ledger, {"event": "planned", "run_id": run_id, "cell": key, "trials": per_cell, "entry": entry,
                         "at": time.time()})
        result = await executor.execute(
            [str(python), str(HARNESS_PATH), str(spec_path.relative_to(quest_root)),
             str(out_path.relative_to(quest_root))],
            cwd=quest_root, timeout_s=timeout_s, env=env,
        )
        run = CellRun(key=key, cell=cell, planned=per_cell, returncode=result.returncode,
                      timed_out=bool(getattr(result, "timed_out", False)), stderr=result.stderr or "")
        reported: dict[int, dict[str, Any]] = {}
        try:
            lines = out_path.read_text(encoding="utf-8").splitlines() if out_path.is_file() else []
        except OSError:
            lines = []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("load_error"):
                run.load_error = str(row["load_error"])
            elif isinstance(row, dict) and isinstance(row.get("trial"), int) and row["trial"] in range(per_cell):
                reported.setdefault(row["trial"], row)  # a trial reported twice keeps its first report
        why_missing = (
            f"the simulation could not be loaded ({run.load_error})" if run.load_error
            else "the cell's process ran out of time before this trial" if run.timed_out
            else f"the cell's process stopped (exit code {result.returncode}) before this trial"
        )
        for t in trials:
            row = reported.get(t["trial"]) or {"trial": t["trial"], "seed": t["seed"], "status": "failed",
                                              "reason": why_missing}
            if row.get("seed") != t["seed"]:
                row = {**row, "status": "failed", "reason": "the reported seed is not the one FI gave this trial"}
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
    summary.write_text(json.dumps(_summary(runs), indent=1, allow_nan=True), encoding="utf-8")
    return TrialRun(cells=runs, ledger_path=ledger, summary_path=summary)


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

    def __init__(self, executor: Any, *, quest_root: Path, protocol: dict[str, Any], deterministic: bool,
                 simulate: Path, analysis: Path, log: Any = None) -> None:
        self.executor = executor
        self.quest_root = Path(quest_root)
        self.protocol = protocol or {}
        self.deterministic = deterministic
        self.simulate = Path(simulate)
        self.analysis = Path(analysis)
        self.log = log
        self.failed_script: str | None = None
        self.last: TrialRun | None = None

    async def execute(self, cmd: list[str], *, cwd: Path, timeout_s: int, env: dict[str, str] | None = None) -> Any:
        from core.execution import ExecutionResult

        if len(cmd) != 2 or Path(cmd[1]).name != self.analysis.name:
            return await self.executor.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)
        started = time.monotonic()
        base = int((env or {}).get("FI_REPLICATE_SEED") or 0)
        grid = self.protocol.get("grid") if isinstance(self.protocol.get("grid"), dict) else {}
        runs = int(self.protocol.get("runs_per_setting") or 1)
        run = await run_trials(
            self.executor, cmd[0], self.quest_root, self.simulate.relative_to(self.quest_root).as_posix(), grid,
            runs_per_setting=runs, base_seed=base, deterministic=self.deterministic, timeout_s=timeout_s, env=env,
        )
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
        analysis_env = {**(env or {}), RESULTS_ENV: str(run.summary_path),
                        "FI_RAW_DIR": str(run.summary_path.parent)}
        result = await self.executor.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=analysis_env)
        self.failed_script = None if result.returncode == 0 else self.analysis.name
        return ExecutionResult(
            returncode=result.returncode, stdout=result.stdout, duration_s=time.monotonic() - started,
            stderr=(run.stderr() + "\n" + (result.stderr or "")).strip(), timed_out=result.timed_out,
        )


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
    spec.write_text(json.dumps({"module": str(module), "entry": "oracle", "cell": {},
                                "trials": [{"trial": 0, "seed": None}]}), encoding="utf-8")
    result = await executor.execute(
        [str(python), str(HARNESS_PATH), str(spec.relative_to(quest_root)), str(out.relative_to(quest_root))],
        cwd=quest_root, timeout_s=timeout_s, env=env,
    )
    try:
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        rows = []
    for row in rows:
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
