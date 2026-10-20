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

import ast
import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Collection
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import in_run as _in_run
from . import mc_error as _mc
from . import optimise_search as _search
from . import oracle_check as _oracle
from . import oracle_forms as _forms
from . import outside_run as _outside

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
# prints go to stderr, so nothing it prints can pass for a result line. It carries the text of core/in_run.py, so what a
# run returned is read the same way here, in code/run.py and in FI.
HARNESS_BODY = r'''
import importlib.util, json, math, sys, time, traceback, warnings

def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)

def _fi_capture(cfg):
    # What each named function of the model's package is called with when the simulation runs (core/second_impl.py):
    # the first few distinct calls, each argument by its parameter's name, kept only when every argument is plain data
    # (numbers, text, real arrays up to a size). The functions themselves are called as they are.
    import functools, importlib, inspect
    try:
        import numpy as np
    except Exception:
        np = None
    mod = importlib.import_module(cfg["module"])
    limit, most = int(cfg.get("limit") or 3), int(cfg.get("max_elements") or 20000)
    # ``shift``: ``{"name", "scale"}``, the one function whose result is changed by that share before the caller gets it
    # (does a result depend on it?); ``counts``: how often each named function was called; ``shifted``: how many of the
    # shifted function's results held a number that could be changed.
    shift = cfg.get("shift") or {}
    shifted_name, scale = shift.get("name"), float(shift.get("scale") or 0.0)
    seen, counts, shifted = {}, {}, {}

    def conv(x, left):
        if np is not None and isinstance(x, np.ndarray):
            if x.dtype.kind not in "biuf" or x.size > left[0]:
                raise TypeError("not plain data")
            left[0] -= x.size
            return x.tolist()
        if np is not None and isinstance(x, np.generic):
            x = x.item()
        if x is None or isinstance(x, (bool, str)):
            return x
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            if not math.isfinite(x):
                raise TypeError("not finite")
            return x
        if isinstance(x, (list, tuple)):
            left[0] -= len(x)
            if left[0] < 0:
                raise TypeError("too large")
            return [conv(i, left) for i in x]
        raise TypeError("not plain data")

    # Each number is changed by its own share (scale times the next of these, in turn): a share common to all of them
    # would vanish in a result that only uses their ratios (weights of a fit, a normalised distribution).
    pattern, turn = (1.0, -0.7, 0.4, -0.9, 0.6, -0.3, 0.8, -0.5), [0]

    def factors(n):
        out = [1.0 + scale * pattern[(turn[0] + i) % len(pattern)] for i in range(n)]
        turn[0] += n
        return out

    def changed(x, hit):
        # Every real or complex number in what the function returned is changed; whole numbers and flags stay as they
        # are (a count or an index), and so does anything that is not a number, a list, a tuple or a dict of them.
        if np is not None and isinstance(x, np.ndarray):
            if x.dtype.kind in "fc" and x.size:
                hit[0] = hit[0] or bool(np.any(x != 0))
                return x * np.asarray(factors(x.size)).reshape(x.shape)
            return x
        if np is not None and isinstance(x, np.generic):
            if x.dtype.kind in "fc":
                hit[0] = hit[0] or bool(x != 0)
                return x * factors(1)[0]
            return x
        if isinstance(x, bool) or isinstance(x, int):
            return x
        if isinstance(x, (float, complex)):
            hit[0] = hit[0] or x != 0  # a zero stays zero: changing it by a share changes nothing
            return x * factors(1)[0]
        if isinstance(x, dict):
            return {k: changed(v, hit) for k, v in x.items()}
        if isinstance(x, tuple):
            items = [changed(v, hit) for v in x]
            return type(x)(*items) if hasattr(x, "_fields") else tuple(items)
        if isinstance(x, list):
            return [changed(v, hit) for v in x]
        return x

    def wrap(name, fn):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn
        plain = not any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in sig.parameters.values())

        @functools.wraps(fn)
        def recorded(*a, **k):
            counts[name] = counts.get(name, 0) + 1
            cases = seen.setdefault(name, [])
            if plain and len(cases) < limit:
                try:
                    bound = sig.bind(*a, **k)
                    left = [most]
                    case = {p: conv(v, left) for p, v in bound.arguments.items()}
                    if case not in cases:
                        cases.append(case)
                except Exception:
                    pass
            got = fn(*a, **k)
            if name == shifted_name:
                hit = [False]
                got = changed(got, hit)
                shifted[name] = shifted.get(name, 0) + (1 if hit[0] else 0)
            return got
        return recorded

    originals = {}
    for name in cfg.get("names") or []:
        fn = getattr(mod, name, None)
        if callable(fn):
            originals[name] = fn
            setattr(mod, name, wrap(name, fn))
    prefix = str(cfg["module"]).split(".")[0]
    for mname, m in list(sys.modules.items()):  # a module of the package that imported a function by name
        if m is None or not (mname == prefix or mname.startswith(prefix + ".")):
            continue
        for name, fn in originals.items():
            if getattr(m, name, None) is fn:
                setattr(m, name, getattr(mod, name))
    return {"cases": seen, "counts": counts, "shifted": shifted}

def main():
    import os
    clock0 = time.perf_counter()
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
    capture, captured = spec.pop("capture", None), None
    if capture:
        try:
            captured = _fi_capture(capture)
        except Exception:
            captured = None
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        loader = importlib.util.spec_from_file_location("fi_experiment", spec["module"])
        mod = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(mod)
        fn = getattr(mod, spec["entry"], None)
        if not callable(fn):
            raise AttributeError(f"{spec['module']} has no function {spec['entry']}()")
        # A run_cell measured for a check must really run the study's trials: run_trial is wrapped to count its calls.
        trial_calls, trial_sums = [0], {}
        if spec.get("spy_trial") and spec["entry"] == "run_cell" and callable(getattr(mod, "run_trial", None)):
            _real_trial = mod.run_trial
            def _counted(*a, **k):
                trial_calls[0] += 1
                got = _real_trial(*a, **k)
                if isinstance(got, dict):
                    for key, v in got.items():
                        if _number(v):
                            total, n = trial_sums.get(str(key), (0.0, 0))
                            trial_sums[str(key)] = (total + float(v), n + 1)
                return got
            mod.run_trial = _counted
    except BaseException as e:
        out.write(json.dumps({"nonce": nonce, "load_error": f"{type(e).__name__}: {e}"[:500]}) + "\n")
        out.close()
        sys.exit(3)
    # A check's trials stop when the next one would not fit in the time it is given (``budget_s``): the first always runs.
    # A timing run (core/run_estimate.py) stops once the trials after the first have taken ``stop_after_s``: enough of
    # them to measure what one costs, however short a trial is.
    budget, started = spec.get("budget_s"), time.perf_counter()
    stop_after = spec.get("stop_after_s")
    later = 0.0
    for index, trial in enumerate(spec["trials"]):
        if budget is not None and index:
            spent = time.perf_counter() - started
            if spent + spent / index > budget:
                break
        # perf_counter, not monotonic: on Windows monotonic moves in steps of about 16 ms, so a trial of a few
        # milliseconds was recorded as 0 or 0.016 s, and a run timed from one such trial was off by a factor of ten.
        t0 = time.perf_counter()
        # When the trial started (seconds into this process): from one start to the next is what a trial costs with the
        # harness's own writing, which a timing run counts.
        row = {"nonce": nonce, "trial": trial["trial"], "seed": trial.get("seed"), "at_s": round(t0 - clock0, 6)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if spec["entry"] == "run_trial":
                    value = fn(dict(spec["cell"]), trial["trial"], trial["seed"])
                elif spec["entry"] == "run_cell":
                    value = fn(dict(spec["cell"]))
                else:
                    value = fn()
                # A number per run, or (for what each run compares inside it) one per value it was given.
                kept, why = read_returned(value, spec.get("levels") or [], spec["entry"])
                if kept is None:
                    raise TypeError(why)
                row.update(status="ok", values=kept)
                reasons = missing_reasons(value, kept)
                if reasons:
                    row[WHY_MISSING] = reasons
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                row.update(status="failed", reason=f"{type(e).__name__}: {e}"[:300],
                           traceback=traceback.format_exc()[-1500:])
        row["duration_s"] = round(time.perf_counter() - t0, 6)
        if spec.get("spy_trial"):
            row["trial_calls"] = trial_calls[0]
            row["trial_means"] = {k: t / n for k, (t, n) in trial_sums.items() if n}
            trial_calls[0] = 0
            trial_sums.clear()
        row["warnings"] = [f"{w.category.__name__}: {w.message}"[:300] for w in caught][:20]
        for w in caught:  # printed as Python prints them, so the numeric-warning check reads them from stderr
            sys.stderr.write(f"{w.filename}:{w.lineno}: {w.category.__name__}: {w.message}\n")
        out.write(json.dumps(row, allow_nan=True) + "\n")
        out.flush()
        if stop_after and index > 0:
            later += row["duration_s"]
            if later >= float(stop_after):
                break
    out.close()
    if captured is not None:
        try:
            with open(capture["path"], "w", encoding="utf-8") as fh:
                json.dump(captured, fh)
        except Exception:
            pass
    sys.stdout = real_stdout

main()
'''
HARNESS_SOURCE = Path(_in_run.__file__).read_text(encoding="utf-8") + "\n" + HARNESS_BODY


def numerical_axes(protocol: Any) -> set[str]:
    """The names of the numerical settings of a protocol (a measurement's ``numerical_settings``, or a search's under
    ``optimisation``): resolutions FI's check at finer settings judges, never a setting the study compares."""
    out: set[str] = set()
    if not isinstance(protocol, dict):
        return out
    for block in (protocol, protocol.get("optimisation")):
        settings = block.get("numerical_settings") if isinstance(block, dict) else None
        if isinstance(settings, dict):
            out.update(str(k) for k in settings)
    return out


def cells(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination of the grid's values, axes in the grid's order, values in theirs (for a grid with sub-studies, the
    settings of each in turn, none twice: :func:`core.optimise_search.grid_cells`). An empty grid is one cell with no axes
    (a study that runs one setting)."""
    return _search.grid_cells(grid)


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return repr(value)
    if isinstance(value, list):  # an array setting (a design variable that is an array): no comma, a key is split on them
        return "[" + ";".join(_fmt(x) for x in value) + "]"
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
    #: The values each run compared inside it (``protocol.in_each_run``), one dict per combination; empty when none.
    levels: list[dict[str, Any]] = field(default_factory=list)


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

    def record_cells(self) -> list[CellRun]:
        """The settings as FI records them (:func:`record_cells`): each run setting with its numbers per run, then a
        record of its own for each value the runs compared inside them, with the numbers per value. Every row holds
        plain numbers only."""
        return record_cells(self.cells)


def record_cells(runs: list[CellRun]) -> list[CellRun]:
    """Each run setting as :class:`CellRun` with its numbers per run, and for each value its runs compared inside them
    (``levels``) a :class:`CellRun` keyed by that value too (``m=20,procedure=bh``) with its numbers per value; the same
    trials, seeds and statuses (a failed run fails for every value). A study that compares nothing inside its runs gives
    back its settings as they are."""
    out: list[CellRun] = []
    for run in runs:
        if not run.levels:
            out.append(run)
            continue
        out.append(CellRun(key=run.key, cell=dict(run.cell), planned=run.planned, returncode=run.returncode,
                           timed_out=run.timed_out, stderr=run.stderr, load_error=run.load_error,
                           rows=[{**r, "values": _in_run.split(r.get("values"))[0]} if r.get("status") == "ok" else dict(r)
                                 for r in run.rows]))
        if not any(_in_run.split(r.get("values"))[1] for r in run.rows if r.get("status") == "ok"):
            continue  # the run returned no number per value: no value has a record (core/in_run.py summary_cells)
        for level in run.levels:
            lk = _in_run.level_key(level)
            out.append(CellRun(key=_in_run.record_key(run.key, lk), cell={**run.cell, **level}, planned=run.planned,
                               returncode=run.returncode, timed_out=run.timed_out, load_error=run.load_error,
                               rows=[{**r, "values": _in_run.split(r.get("values"))[1].get(lk, {})}
                                     if r.get("status") == "ok" else dict(r) for r in run.rows]))
    return out


def _append(fh: Any, row: dict[str, Any]) -> None:
    # One open file for the whole ledger: opening it again for each of 110,592 trials cost about 26 s on Windows (r13).
    fh.write(json.dumps(row, allow_nan=True, default=str) + "\n")


#: Why a trial the study's time limit cut off has no result: FI's limit, not a fault of the code (core/trial_checks.py
#: counts these apart from the failed trials).
OUT_OF_TIME = "the study's time (execution.timeout_s) ran out before this setting's trial"


def _summary(runs: list[CellRun], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per cell, each metric's values over the trials that succeeded, with their count and total: what the analysis
    script reads (``FI_TRIALS``) and what FI's statistics are computed from. Each metric's ``trials`` is the trial each
    value came from (FI's own number): what a paired design joins on. A setting whose runs compared values inside them
    is followed by a record per value (:func:`core.in_run.summary_cells`)."""
    out: list[dict[str, Any]] = []
    for run in runs:
        out.extend(_in_run.summary_cells(run.cell, run.key, run.planned, run.rows, run.levels))
    return {"schema": "fi.trials/v1", "thresholds": dict(thresholds or {}), "cells": out}


def _plan(quest_root: Path, module: Path | str, grid: dict[str, list[Any]], *, runs_per_setting: int, base_seed: int,
          deterministic: bool, folder: Path, out_name: str, keep_spec: bool = False,
          paired: bool = False, thresholds: dict[str, Any] | None = None,
          in_each_run: dict[str, list[Any]] | None = None) -> list[dict[str, Any]]:
    """Write one spec per cell (the simulation file, the entry, the cell, its trials with their seeds, a nonce) into
    ``folder`` and return the plan: per cell its key, cell, trials, nonce, and the spec and results paths relative to
    ``quest_root``. ``in_each_run``: the values every run goes through together; the cell a run is given holds each such
    list whole, and the run may return a number per value (:func:`core.in_run.read_returned`)."""
    quest_root = Path(quest_root)
    entry = "run_cell" if deterministic else "run_trial"
    per_cell = 1 if deterministic else max(1, int(runs_per_setting))
    inside = {str(k): list(v) for k, v in (in_each_run or {}).items()}
    levels = _in_run.level_cells(inside)
    plan = []
    for index, cell in enumerate(cells(grid)):
        key = cell_key(cell)
        trials = [{"trial": t, "seed": None if deterministic else trial_seed(base_seed, "" if paired else key, t)}
                  for t in range(per_cell)]
        spec_path = folder / f"cell{index}.json"
        out_path = folder / out_name.format(index=index)
        out_path.unlink(missing_ok=True)
        nonce = hashlib.sha256(f"{time.time_ns()}|{index}|{id(trials)}".encode()).hexdigest()[:24]
        spec_path.write_text(json.dumps({"module": str(module).replace("\\", "/"), "entry": entry,
                                         "cell": {**cell, **inside}, "levels": levels,
                                         "trials": trials, "nonce": nonce, "keep_spec": keep_spec,
                                         "thresholds": dict(thresholds or {})}, default=str),
                             encoding="utf-8")
        plan.append({"index": index, "key": key, "cell": cell, "levels": levels, "trials": trials, "nonce": nonce,
                     "entry": entry,
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
    with open(ledger, "w", encoding="utf-8") as fh:
        runs = _collect_cells(quest_root, plan, results, fh, run_id=run_id, not_reported=not_reported)
    summary = raw / SUMMARY_NAME
    summary.write_text(json.dumps(_summary(runs, thresholds), indent=1, allow_nan=True), encoding="utf-8")
    return TrialRun(cells=runs, ledger_path=ledger, summary_path=summary)


def _collect_cells(quest_root: Path, plan: list[dict[str, Any]], results: dict[int, Any], fh: Any, *, run_id: str,
                   not_reported: str) -> list["CellRun"]:
    """Each cell's rows (see :func:`_collect`), with the ledger written to ``fh`` as they are read."""
    runs: list[CellRun] = []
    for task in plan:
        key, trials, nonce = task["key"], task["trials"], task["nonce"]
        per_cell = len(trials)
        _append(fh, {"event": "planned", "run_id": run_id, "cell": key, "trials": per_cell,
                         "entry": task["entry"], "at": time.time()})
        result = results.get(task["index"])
        run = CellRun(key=key, cell=task["cell"], planned=per_cell, levels=list(task.get("levels") or []),
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
            why_missing = OUT_OF_TIME
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
            _append(fh, {
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
    return runs


async def run_trials(
    executor: Any, python: Path | str, quest_root: Path, module: Path | str, grid: dict[str, list[Any]], *,
    runs_per_setting: int, base_seed: int, deterministic: bool, timeout_s: int, env: dict[str, str] | None = None,
    run_id: str = "", thresholds: dict[str, Any] | None = None, paired: bool = False, on_plan: Any = None,
    in_each_run: dict[str, list[Any]] | None = None, log: Any = None, at_once: int = 1, cost: Any = None,
) -> TrialRun:
    """Run every cell of ``grid`` in its own process and record every trial (see the module docstring). ``module`` is
    the simulation file relative to ``quest_root``; ``timeout_s`` bounds the whole study. ``on_plan(plan)`` is told the
    plan before the first cell runs (for a progress line while it does). ``at_once``: how many cells run at the same
    time (:func:`settings_at_once`); ``cost(cell)``: what a cell is expected to cost, when known, so the costliest start
    first and none is left to run alone at the end. The ledger and the summary are the same whatever the two are."""
    quest_root = Path(quest_root)
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    harness = quest_root / HARNESS_PATH
    harness.write_text(HARNESS_SOURCE, encoding="utf-8")  # fresh every run: nothing the experiment wrote is run
    (quest_root / RUN_RECORD).unlink(missing_ok=True)  # an earlier run's record never stands beside this run's ledger
    plan = _plan(quest_root, module, grid, runs_per_setting=runs_per_setting, base_seed=base_seed,
                 deterministic=deterministic, folder=work, out_name="cell{index}.out.jsonl", paired=paired,
                 thresholds=thresholds, in_each_run=in_each_run)
    if on_plan is not None:
        on_plan(plan)
    results: dict[int, Any] = {}
    # Each cell runs from a copy of code/ outside the quest folder (core/outside_run.py): FI's records (needs/, .fi/)
    # are not next to the code; FI's harness, the cell's spec and its results file come and go by FI's own list.
    executor = _outside.OutsideExecutor(executor, quest_root, log=log)
    started = time.monotonic()  # timeout_s bounds the whole study, as it bounded one simulation script before
    gate = asyncio.Semaphore(max(1, int(at_once or 1)))

    async def one(task: dict[str, Any]) -> None:
        async with gate:
            # What is left of the study's time when this cell starts (cells running at once share one deadline).
            left = int(timeout_s - (time.monotonic() - started))
            if left <= 0:
                results[task["index"]] = type("NotRun", (), {"returncode": -1, "timed_out": True, "stderr": ""})()
                return
            results[task["index"]] = await executor.execute(
                [str(python), HARNESS_PATH.as_posix(), task["spec"], task["out"]],
                cwd=quest_root, timeout_s=max(1, left), env=env,
            )

    try:
        await asyncio.gather(*(one(task) for task in _costliest_first(plan, cost)))
    finally:
        await executor.settle()
    return _collect(quest_root, plan, results, run_id=run_id, thresholds=thresholds, not_reported="")


#: ``execution.settings_at_once: auto`` runs half the machine's cores' worth of settings at once (the rest is left for FI
#: and for what the simulations share, such as memory bandwidth), and at most this many.
AUTO_AT_ONCE_MOST = 8


def settings_at_once(value: Any, *, cores: int | None = None, fleet: int = 1) -> int:
    """How many settings of the study run at the same time: ``execution.settings_at_once`` as a number (at least 1), or
    for ``"auto"`` half the cores (at most :data:`AUTO_AT_ONCE_MOST`), shared among the ``fleet`` quests running at once."""
    if isinstance(value, int) and not isinstance(value, bool):
        return max(1, value)
    cores = int(cores or os.cpu_count() or 1)
    return max(1, min(AUTO_AT_ONCE_MOST, cores // 2) // max(1, int(fleet or 1)))


def _costliest_first(plan: list[dict[str, Any]], cost: Any) -> list[dict[str, Any]]:
    """The plan's cells in the order they start: the costliest first when ``cost(cell)`` can tell (a tie keeps the plan's
    order), else the plan's order."""
    if cost is None:
        return list(plan)

    def of(task: dict[str, Any]) -> float:
        try:
            value = float(cost(task["cell"]))
        except Exception:  # noqa: BLE001 -- a cost that cannot be worked out puts the cell where the plan has it
            return 0.0
        return value if math.isfinite(value) else 0.0

    return sorted(plan, key=lambda t: (-of(t), t["index"]))


def prepare_cluster(quest_root: Path, module: Path | str, grid: dict[str, list[Any]], *, runs_per_setting: int,
                    base_seed: int, deterministic: bool, key: str, paired: bool = False,
                    thresholds: dict[str, Any] | None = None,
                    in_each_run: dict[str, list[Any]] | None = None) -> dict[str, Any]:
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
                 paired=paired, thresholds=thresholds, in_each_run=in_each_run)
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


_READS_CELL = re.compile(r"\bcell\s*\[")
_ENTRY_RE = re.compile(r"^def\s+(run_trial|run_cell|oracle)\s*\(", re.MULTILINE)


def entries(simulate: Path) -> set[str]:
    """Which of ``run_trial`` / ``run_cell`` / ``oracle`` the simulation defines at its top level (read, not run)."""
    try:
        return set(_ENTRY_RE.findall(Path(simulate).read_text(encoding="utf-8")))
    except OSError:
        return set()


TRIAL_AGGREGATE_WHY = ("run_trial must run ONE trial of the setting, and it currently returns the result of run_cell, "
                       "which is the setting's number computed over many trials")

_OWN_MODULE = "simulate"


class _Module:
    """What one reading of simulate.py (never run) says about which function reaches which. A module-level ``def``, a
    module-level name bound to an expression (``_run = run_cell``, ``run_trial = lambda c, t, s: run_cell(c)``,
    ``TABLE = {"a": run_cell}``) and an import (``from x import run_cell as y``) are all followed by name. An attribute
    call counts only when its object is this very module (``import simulate``, ``import x as simulate``); ``obj.run_cell()``
    on any other object is that object's own method and is not this file's ``run_cell``."""

    def __init__(self, source: str) -> None:
        self.ok = True
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            self.ok = False
            return
        self.nodes: dict[str, list[ast.AST]] = {}
        self.alias: dict[str, str] = {}      # an imported name -> the name it stands for in this module
        self.modules: set[str] = {_OWN_MODULE}
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.nodes.setdefault(n.name, []).append(n)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)) and getattr(n, "value", None) is not None:
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        self.nodes.setdefault(t.id, []).append(n.value)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.split(".")[-1] == _OWN_MODULE:
                        self.modules.add(a.asname or a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    if a.name in ("run_cell", "run_trial"):
                        self.alias[a.asname or a.name] = a.name
                    elif a.name == _OWN_MODULE:
                        self.modules.add(a.asname or a.name)

    def _names(self, node: ast.AST, *, calls_only: bool) -> set[str]:
        """The module-level names ``node`` refers to. ``calls_only``: only a call, or a function handed to a call as an
        argument (``map(run_trial, ...)``); otherwise any mention (a table of functions, a returned function)."""
        found: set[str] = set()

        # A name the code binds itself (a parameter, an assignment, a loop variable) is that variable, not the module's
        # function of the same name.
        local: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            local |= {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs, *filter(None, (a.vararg, a.kwarg)))}
        local |= {x.id for x in ast.walk(node) if isinstance(x, ast.Name) and not isinstance(x.ctx, ast.Load)}

        def resolve(x: ast.AST) -> str | None:
            if isinstance(x, ast.Name):
                return None if x.id in local else self.alias.get(x.id, x.id)
            if (isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id in self.modules
                    and x.value.id not in local):
                return x.attr
            return None

        for sub in ast.walk(node):
            if calls_only:
                if isinstance(sub, ast.Call):
                    for x in (sub.func, *sub.args, *(k.value for k in sub.keywords)):
                        name = resolve(x)
                        if name:
                            found.add(name)
            elif isinstance(sub, (ast.Name, ast.Attribute)):
                name = resolve(sub)
                if name:
                    found.add(name)
        return found

    def reaches(self, start: str, goal: str, *, calls_only: bool) -> bool:
        """Whether ``goal`` is reached from what ``start`` is, through any number of module-level functions and names."""
        seen: set[str] = set()
        todo = [start]
        while todo:
            current = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            for node in self.nodes.get(current, []):
                names = self._names(node, calls_only=calls_only)
                if goal in names and current != goal:
                    return True
                todo.extend(n for n in names if n in self.nodes and n not in seen)
        return False


def trial_aggregates(source: str) -> str:
    """A plain sentence when ``run_trial`` in ``source`` (read, never run) is not one trial because it reaches ``run_cell``
    (it calls or names it, directly or through any number of the file's own functions and names, an alias such as
    ``_run = run_cell`` or ``run_trial = run_cell``, a table of functions, or an imported ``run_cell``), and ``""``
    otherwise.

    ``run_cell`` is the setting's own number, the mean over many trials for a random study. FI's harness calls
    ``run_trial`` once per trial and keeps each result, so a ``run_trial`` that returns ``run_cell``'s number makes every
    trial as long as a whole cell (the study would not fit its time) and every kept "trial" a mean of many (wrong
    statistics). The converse, a ``run_cell`` that calls ``run_trial``, is how a cell is built and is not flagged. Not
    followed: a function looked up by a computed string (``globals()[name]``), ``getattr``, and code in other files."""
    mod = _Module(source)
    if not mod.ok or "run_trial" not in mod.nodes:
        return ""
    return TRIAL_AGGREGATE_WHY if mod.reaches("run_trial", "run_cell", calls_only=False) else ""


def cell_runs_trials(source: str) -> bool:
    """Whether ``run_cell`` in ``source`` really calls ``run_trial`` (directly or through the file's own functions, also
    handed to ``map`` and the like): the mean it returns then comes from the trials the study itself runs."""
    mod = _Module(source)
    return mod.ok and mod.reaches("run_cell", "run_trial", calls_only=True)


class TrialsRunner:
    """Stands in for the executor when the simulation follows the trial contract, as ``split_run.SplitRunner`` does for
    the older two-script one: running ``experiment.py`` first runs every trial of the protocol (:func:`run_trials`,
    one process per cell, FI's ledger), then the analysis with ``FI_TRIALS`` naming FI's per-cell summary. What it
    returns is the analysis's run, its stderr led by every cell's (so the simulation's warnings reach the numeric
    check). ``failed_script`` says which script to repair; ``last`` keeps the trial run for the checks after it."""

    def __init__(self, executor: Any, *, quest_root: Path, protocol: Any, deterministic: bool,
                 simulate: Path, analysis: Path, log: Any = None, submit: Path | None = None,
                 leave_out: Any = None, repairs_left: Any = None, at_once: int = 1) -> None:
        self.executor = executor
        #: How many settings of the study run at the same time (:func:`settings_at_once`), and what each is expected to cost
        #: (``cost(cell)`` from FI's timing, set by the sizing step; ``None`` until then): the costliest start first.
        self.at_once = max(1, int(at_once or 1))
        self.cost: Any = None
        self.quest_root = Path(quest_root)
        self._protocol = protocol
        self.deterministic = deterministic
        self.simulate = Path(simulate)
        self.analysis = Path(analysis)
        self.log = log
        self.failed_script: str | None = None
        #: Why the last run was sent back as a simulation that does not use its inputs (core/flat_output.py), or ``None``.
        self.flat: str | None = None
        #: Every setting gave the same numbers but nothing in the code ignores the settings (core/flat_output.py): the plain
        #: record of what was observed, or ``None``.
        self.no_difference: str | None = None
        self.last: TrialRun | None = None
        # On a cluster: the experiment's script that submits the job array FI prepared and reports it pending or done.
        self.submit = Path(submit) if submit is not None else None
        #: Quantities the analysis must not be given (the simulation typed their value into its code, core/typed_results.py):
        #: a function returning their names, or ``None``. The ledger keeps every value the simulation returned.
        self._leave_out = leave_out
        #: Whether the quest has a repair of the simulation left (a function, or a bool): FI's checks of the trials
        #: (core/trial_checks.py) send a fault back while one is left, and leave out or stop once they are spent.
        self._repairs_left = repairs_left
        #: What FI's checks of the trials found when they sent the run back or stop it (``{"action", "say", "tried"}``),
        #: else ``None``; the engine keeps it in ``exec_result['trial_checks']``.
        self.trial_fault: dict[str, Any] | None = None
        #: The settings whose numbers the analysis is not given (most of their trials failed, the repairs were spent).
        self.left_out_settings: list[str] = []
        self._plan_seen: list[dict[str, Any]] | None = None
        self._plan_started = 0.0

    def _note_plan(self, plan: list[dict[str, Any]]) -> None:
        self._plan_seen, self._plan_started = plan, time.monotonic()

    def progress_text(self) -> str:
        """For the "still running" line: how many settings are done and how long is left, in plain words; empty when the
        settings are not being run (nothing to say yet, or the trials were already run)."""
        plan = self._plan_seen
        if not plan:
            return ""
        from . import run_estimate as _estimate

        done_trials = 0
        done_cells = 0
        for task in plan:
            try:
                lines = (self.quest_root / task["out"]).read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            seen = sum(1 for ln in lines if '"status"' in ln)
            done_trials += seen
            done_cells += seen >= len(task["trials"])
        return _estimate.progress_line(done_cells, len(plan), done_trials, sum(len(t["trials"]) for t in plan),
                                       time.monotonic() - self._plan_started)

    def _without_left_out(self, summary: Path) -> Path:
        """The per-cell summary the analysis reads: FI's own, or (when quantities are left out) a copy of it without them,
        beside it. FI's summary and the ledger are not changed."""
        names = {str(n) for n in (self._leave_out() if callable(self._leave_out) else self._leave_out or ())}
        settings = set(self.left_out_settings)
        if not names and not settings:
            return summary
        from . import trial_checks as _checks

        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            data["cells"] = [c for c in data.get("cells") or [] if not _checks.of_settings(c, settings)]
            for cell in data["cells"]:
                cell["metrics"] = {k: v for k, v in (cell.get("metrics") or {}).items() if k not in names}
            copy = summary.with_name("trials.analysis.json")
            copy.write_text(json.dumps(data, indent=1, allow_nan=True), encoding="utf-8")
            return copy
        except (OSError, ValueError, AttributeError):
            return summary

    async def execute(self, cmd: list[str], *, cwd: Path, timeout_s: int, env: dict[str, str] | None = None) -> Any:
        from core.execution import ExecutionResult

        if len(cmd) != 2 or Path(cmd[1]).name != self.analysis.name:
            return await self.executor.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)
        started = time.monotonic()
        self.flat = None
        self.trial_fault, self.left_out_settings = None, []
        base = int((env or {}).get("FI_REPLICATE_SEED") or 0)
        protocol = (self._protocol() if callable(self._protocol) else self._protocol) or {}
        grid = _search.study_grid(protocol)
        runs = int(protocol.get("runs_per_setting") or 1)
        key = _run_key(self.simulate, protocol, runs, base, self.deterministic)
        # A paired metric compares the settings trial by trial: each trial number gets one seed across the settings.
        paired = any(isinstance(m, dict) and m.get("paired") for m in protocol.get("metrics") or [])
        inside = protocol.get("in_each_run") if isinstance(protocol.get("in_each_run"), dict) else None
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
                in_each_run=inside,
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
            _keep_run_time(self.quest_root, key, run, None)  # the queue's waiting time is not a run time FI measured
            if self.log is not None:
                self.log.info("[execute] the cluster job is done: FI read every setting's results and wrote the ledger")
        else:
            # Run here, not on a cluster: what an earlier cluster job's code did while queued is not this run's.
            _forget_queued_changes(self.quest_root)
            if not grid and _READS_CELL.search(self.simulate.read_text(encoding="utf-8", errors="replace")):
                self.failed_script = self.simulate.name
                return ExecutionResult(
                    returncode=1, stdout="", duration_s=time.monotonic() - started,
                    stderr=("simulate.py reads the setting it is given (`cell[...]`), but the plan's protocol has no `grid`, so "
                            "FI would give it an empty setting and every trial would fail. Put the settings in the plan's "
                            "protocol as `grid` (one list per parameter: numbers, or names such as [euler, rk4]); no change "
                            "to simulate.py is needed."),
                )
            ran_from = time.monotonic()
            if self.at_once > 1:
                said = f"{self.at_once} settings of the study run at the same time (execution.settings_at_once)"
                print(f"[FI] {said}")
                if self.log is not None:
                    self.log.info("[execute] %s", said)
            run = await run_trials(
                self.executor, cmd[0], self.quest_root, self.simulate.relative_to(self.quest_root).as_posix(), grid,
                runs_per_setting=runs, base_seed=base, deterministic=self.deterministic, timeout_s=timeout_s, env=env,
                thresholds=protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None,
                paired=paired, on_plan=self._note_plan, in_each_run=inside, log=self.log, at_once=self.at_once,
                cost=self.cost,
            )
            self._plan_seen = None
            _save_run(self.quest_root, key, run)
            # How long the experiment took is the time of THIS run of the trials; the analysis run again on stored trials
            # (the branch above) is not the experiment and leaves the record as it is (core/run_facts.py).
            _keep_run_time(self.quest_root, key, run, time.monotonic() - ran_from)
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
        # Every setting gave the very same numbers. That alone is an observation (a conserved quantity, a plateau and a true
        # absence of effect look the same); it is a fault only when the simulation's own code never lets the varied setting
        # reach what it returns. The fault is sent back to be repaired like a simulation that does not run
        # (core/flat_output.py); the observation is kept as a result and recorded as observed (`self.no_difference`).
        from . import degenerate_results as _degenerate
        from . import flat_output as _flat

        self.no_difference = None
        seen = _flat.observed_flat_sweep(run.cells)
        flat = _flat.flat_sweep(
            run.cells, source=self.simulate.read_text(encoding="utf-8", errors="replace"),
            expected=_degenerate.expected_constants(protocol)) if seen is not None else None
        if seen is not None and flat is None:
            self.no_difference = seen.note()
            if self.log is not None:
                self.log.info("[execute] every setting gave the same numbers, and the simulation's code does read the settings: "
                              "kept as an observed absence of difference, not a fault. %s", self.no_difference)
        elif seen is None:
            # Some results moved, but one setting may still have moved none of them (core/flat_output.py).
            unchanged = _flat.unchanged_axes(run.cells, skip=numerical_axes(protocol))
            if unchanged:
                self.no_difference = _flat.unchanged_axes_note(unchanged)
                if self.log is not None:
                    self.log.warning("[execute] %s", self.no_difference)
        if flat is not None:
            self.flat = flat
            self.failed_script = self.simulate.name
            if self.log is not None:
                self.log.warning("[execute] the simulation does not use the settings it is given: %s", flat)
            return ExecutionResult(returncode=1, stdout="", duration_s=time.monotonic() - started,
                                   stderr=f"{run.stderr()}\nFI's run gave a result that is not usable: {flat}".strip())
        sent = self._check_trials(run, protocol, started)
        if sent is not None:
            return sent
        # Relative to the quest folder the analysis runs in: the same path inside a container (/work) as on the host.
        summary_path = self._without_left_out(run.summary_path)
        analysis_env = {**(env or {}), RESULTS_ENV: summary_path.relative_to(self.quest_root).as_posix(),
                        "FI_RAW_DIR": run.summary_path.parent.relative_to(self.quest_root).as_posix()}
        # The analysis runs from a copy outside the quest folder too; what it writes under raw/, figures/, results/,
        # data/ (and FI's figure records) comes back, anything else is said in run.log and dropped.
        result = await _outside.OutsideExecutor(self.executor, self.quest_root, inputs=_outside.ANALYSIS_INPUTS, skip=(),
                                                log=self.log).execute(
            cmd, cwd=cwd, timeout_s=timeout_s, env=analysis_env)
        self.failed_script = None if result.returncode == 0 else self.analysis.name
        if result.returncode != 0 and self.left_out_settings and not self.trial_fault:
            # Settings are left out only once the repairs are spent, so no repair of the analysis is coming: an analysis
            # that cannot run without them (it checks the planned grid) ends the quest with that said plainly.
            last = next((ln.strip() for ln in reversed((result.stderr or "").splitlines()) if ln.strip()), "")
            many = len(self.left_out_settings)
            self.trial_fault = {
                "action": "stop", "partial": False,
                "say": (f"{many} setting(s) were left out because most of their trials failed "
                        f"({', '.join(self.left_out_settings[:4])}), and the analysis could not run without them"
                        + (f" ({last[:300]})" if last else "")),
                "tried": ["ran every trial of the study, sent the settings where most trials failed back to be repaired, "
                          "and left them out once the repairs were spent",
                          "ran the analysis on the remaining settings; it failed"]}
        return ExecutionResult(
            returncode=result.returncode, stdout=result.stdout, duration_s=time.monotonic() - started,
            stderr=(run.stderr() + "\n" + (result.stderr or "")).strip(), timed_out=result.timed_out,
        )


    def _check_trials(self, run: TrialRun, protocol: dict[str, Any], started: float) -> Any:
        """FI's two checks of what the trials gave (core/trial_checks.py): the failed trials of each setting, and the reported
        intervals and standard errors against the true values. A fault is sent back like a failed run of the simulation (the
        directive at the end of its stderr) while a repair is left; once they are spent, a fault in some settings leaves
        them out of what the analysis reads, and any other fault is a failed run the engine stops on. ``None`` to go on."""
        from core.execution import ExecutionResult

        from . import trial_checks as _checks

        try:
            data = json.loads(run.summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        cells = [c for c in data.get("cells") or [] if isinstance(c, dict)]
        thresholds = data.get("thresholds") if isinstance(data.get("thresholds"), dict) else {}
        reasons = _checks.reasons_of_run(run.cells, self.quest_root) or _checks.ledger_reasons(self.quest_root)
        left = self._repairs_left() if callable(self._repairs_left) else bool(self._repairs_left)
        verdict = _checks.judge(cells, protocol, thresholds or None, reasons, repairs_left=bool(left))
        _checks.write_record(self.quest_root, verdict.record())
        stated = [f for f in verdict.fails if f.stated and not f.fault]
        if self.log is not None:
            for f in stated:
                self.log.info("[execute] %s (recorded; the paper states it)", f.says())
            for q in verdict.calibration.quantities:
                if not q.fault and q.below:
                    self.log.info("[execute] %s; below the level in %d setting(s): reported, not a fault", q.says(), q.below)
            if not verdict.calibration.checked:
                self.log.info("[execute] calibration not checked: %s", "; ".join(verdict.calibration.not_checked)[:400])
            for u in verdict.undefined[:8]:
                self.log.info("[execute] %s (not a failed trial)", u.says())
        if verdict.action == _checks.LEAVE_OUT:
            self.left_out_settings = list(verdict.left_out)
            if self.log is not None:
                self.log.warning("[execute] in %d setting(s) more than half of the trials failed and the repairs are spent: "
                                 "their numbers are left out of the results and the paper says so (%s)",
                                 len(verdict.left_out), ", ".join(verdict.left_out[:6]))
            return None
        if verdict.action == _checks.OK:
            return None
        self.failed_script = self.simulate.name
        self.trial_fault = {"action": verdict.action, "say": verdict.say, "tried": list(verdict.tried),
                            "partial": verdict.partial}
        if self.log is not None:
            self.log.warning("[execute] FI's checks of the trials found a fault; %s:\n%s",
                             "sending the simulation back to be repaired" if verdict.action == _checks.REPAIR
                             else "the repairs are spent, the quest stops", verdict.directive[:1500])
        directive = verdict.directive if len(verdict.directive) <= 1800 else verdict.directive[:1797] + "..."
        return ExecutionResult(returncode=1, stdout="", duration_s=time.monotonic() - started,
                               stderr=f"{run.stderr()[-4000:]}\n{directive}".strip())


def _keep_run_time(quest_root: Path, key: str, run: TrialRun, wall_s: float | None) -> None:
    """Keep the wall time of a run of the trials, how many it ran, the sum of the per-trial times and, in a quest run in
    two stages, which stage the run belongs to (exploring or confirming) in ``.fi/run_facts.json``."""
    from . import phased, run_facts

    rows = [r for c in run.cells for r in c.rows]
    times = [r["duration_s"] for r in rows if isinstance(r.get("duration_s"), (int, float)) and not isinstance(r["duration_s"], bool)]
    try:
        stage = phased.stage(quest_root)
    except Exception:  # noqa: BLE001 -- a record that cannot be read leaves the stage unsaid
        stage = None
    run_facts.record_trials(quest_root, key=key, wall_s=wall_s, trials=len(rows), cells=len(run.cells),
                            trial_sum_s=sum(times) if times else None,
                            stage=stage if stage in (phased.EXPLORE, phased.CONFIRM) else None)


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
    """What decides a set of trials: the simulation's text (with its model package), the protocol (``grid``: the whole of it is passed), the
    count, the base seed and whether each setting runs once (``run_cell``)."""
    try:
        text = Path(simulate).read_bytes()
    except OSError:
        text = b""
    # The model's package beside it (core/code_layout.py) is part of the simulation: other equations are other trials.
    from . import code_layout as _code_layout

    package = "".join(f"\n{rel}\n{body}" for rel, body in sorted(_code_layout.package_sources(Path(simulate).parent).items()))
    text += package.encode("utf-8")
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
         "timed_out": c.timed_out, "stderr": c.stderr[-20000:], "load_error": c.load_error, "levels": c.levels}
        for c in run.cells]}
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


def _outside_executor(executor: Any, quest_root: Path | str, log: Any = None) -> Any:
    """``executor`` running from a copy of code/ outside the quest folder (core/outside_run.py), wrapped once: a run of
    the checks wraps its executor at the start, and every batch it starts uses that one."""
    return executor if isinstance(executor, _outside.OutsideExecutor) else _outside.OutsideExecutor(executor, quest_root,
                                                                                                    log=log)


async def _run_batch(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *, entry: str,
                     cell: dict[str, Any], trials: list[dict[str, Any]], timeout_s: int, env: dict[str, str] | None,
                     thresholds: dict[str, Any] | None, spy_trial: bool = False,
                     levels: list[dict[str, Any]] | None = None,
                     budget_s: float | None = None, log: Any = None) -> tuple[list[dict[str, Any]], Any]:
    """Call one entry function for each of ``trials`` (``trial``, ``seed``) in ONE process: ``(the rows FI's harness
    wrote, in order, the process result)``. ``budget_s``: no trial is started that would not end within it (the first
    always runs); ``timeout_s`` stops the process. A row is a ``load_error``, or a trial with ``status`` ``ok`` (its
    ``values``) or ``failed``."""
    quest_root = Path(quest_root)
    work = quest_root / ".fi" / "trials"
    work.mkdir(parents=True, exist_ok=True)
    (quest_root / HARNESS_PATH).write_text(HARNESS_SOURCE, encoding="utf-8")
    spec = work / "oracle.json"
    out = work / "oracle.out.jsonl"
    out.unlink(missing_ok=True)
    nonce = hashlib.sha256(f"oracle|{time.time_ns()}".encode()).hexdigest()[:24]
    spec.write_text(json.dumps({"module": str(module).replace(chr(92), "/"), "entry": entry, "cell": cell,
                                "levels": list(levels or []), "trials": list(trials), "nonce": nonce,
                                "thresholds": dict(thresholds or {}), "spy_trial": bool(spy_trial),
                                **({"budget_s": float(budget_s)} if budget_s is not None else {})}, default=str),
                    encoding="utf-8")
    # From a copy of code/ outside the quest folder (core/outside_run.py), as every cell of the study runs.
    result = await _outside_executor(executor, quest_root, log).execute(
        [str(python), HARNESS_PATH.as_posix(), spec.relative_to(quest_root).as_posix(), out.relative_to(quest_root).as_posix()],
        cwd=quest_root, timeout_s=timeout_s, env=env,
    )
    try:
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        rows = []
    # Only what the harness FI started wrote (its nonce), each trial it was given once.
    given = {int(t["trial"]): t.get("seed") for t in trials}
    kept: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("nonce") != nonce:
            continue
        if row.get("load_error"):
            kept.append(row)
            continue
        number = row.get("trial")
        if not isinstance(number, int) or number not in given or number in seen or row.get("seed") != given[number]:
            continue
        seen.add(number)
        kept.append(row)
    return kept, result


async def _run_one(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *, entry: str, cell: dict[str, Any],
                   seed: int | None, timeout_s: int, env: dict[str, str] | None, thresholds: dict[str, Any] | None,
                   label: str, trial: int = 0, spy_trial: bool = False,
                   spy_out: dict[str, Any] | None = None,
                   levels: list[dict[str, Any]] | None = None, log: Any = None) -> tuple[dict[str, Any] | None, str]:
    """Call one entry function once in its own process: ``(values, "")``, or ``(None, why)`` when it could not.
    ``spy_trial``: count the calls ``run_cell`` makes to ``run_trial`` and the mean of what they returned, per
    name, in ``spy_out`` (``calls``, ``means``)."""
    quest_root = Path(quest_root)
    rows, result = await _run_batch(executor, python, quest_root, module, entry=entry, cell=cell,
                                    trials=[{"trial": int(trial), "seed": seed}], timeout_s=timeout_s, env=env,
                                    thresholds=thresholds, spy_trial=spy_trial, levels=levels, log=log)
    for row in rows:
        if row.get("load_error"):
            return None, str(row["load_error"])
        if row.get("status") == "ok":
            if spy_out is not None:
                spy_out.update(calls=int(row.get("trial_calls") or 0), means=dict(row.get("trial_means") or {}))
            return dict(row.get("values") or {}), ""
        if row.get("status") == "failed":
            where = last_frame(str(row.get("traceback") or ""), quest_root)
            return None, str(row.get("reason") or f"{label} failed") + (f" (at {where})" if where else "")
    return None, (f"{label} ran out of time" if getattr(result, "timed_out", False)
                  else f"{label} did not report (exit code {result.returncode})")


_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')


def last_frame(traceback_text: str, quest_root: Path | str | None = None) -> str:
    """``<file> line <n>`` of the last frame of a Python traceback that is in the quest's own code (``code/...``), the
    place a person opens to fix it; ``""`` when the traceback names none (the error was raised inside a library and
    no frame of the quest's code is in what was kept)."""
    root = str(Path(quest_root)).replace("\\", "/").rstrip("/") + "/" if quest_root else ""
    found = ""
    for match in _FRAME_RE.finditer(traceback_text or ""):
        path = match.group(1).replace("\\", "/")
        if root and path.casefold().startswith(root.casefold()):
            rel = path[len(root):]  # the quest's own folder: what is under it, never the folders above it
        elif "/code/" in path:
            rel = path[path.rfind("/code/") + 1:]
        else:
            rel = path
        # A library's frame (the quest's venv, site-packages) is not the place a person fixes.
        if not rel.startswith("code/") or "site-packages" in rel or "/.venv/" in f"/{rel}":
            continue
        found = f"{rel} line {match.group(2)}"
    return found


def clip_keeping_place(text: str, most: int) -> str:
    """``text`` cut to ``most`` characters, keeping a trailing ``(at <file> line <n>)``: the place is what a person
    opens, so it is the last thing to lose."""
    tail = re.search(r" \(at [^()]+ line \d+\)$", text or "")
    if not tail or len(text) <= most:
        return (text or "")[:most]
    return text[: max(0, most - len(tail.group(0)))] + tail.group(0)


async def run_oracle(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *, timeout_s: int,
                     env: dict[str, str] | None = None, thresholds: dict[str, Any] | None = None,
                     log: Any = None) -> tuple[dict[str, float] | None, str]:
    """Call the simulation's ``oracle()`` in its own process: ``(values, "")``, or ``(None, why)`` when it could not."""
    return await _run_one(executor, python, quest_root, module, entry="oracle", cell={}, seed=None, timeout_s=timeout_s,
                          env=env, thresholds=thresholds, label="oracle()", log=log)


def is_per_trial(oracle: dict[str, Any] | None) -> bool:
    """A check that holds in every single trial: kind ``invariant``, or declared ``per_trial: true``."""
    if oracle is None:
        return False
    flag = oracle.get("per_trial")
    if flag is True or (isinstance(flag, str) and flag.strip().lower() in ("true", "yes", "1")):
        return True
    return _oracle.kind_of(oracle) == "invariant"


def oracle_entry(have: set[str] | frozenset[str], oracle: dict[str, Any] | None = None, *, per_trial: bool = False,
                 cell_runs_trials: bool = False) -> str:
    """Which simulation function FI calls to measure an oracle: ``"run_trial"`` or ``"run_cell"``.

    The two mean different things. ``run_trial`` is ONE trial of one setting (FI's trial harness calls it once per trial
    and keeps each result); ``run_cell`` is the setting's own number, the one a deterministic study computes or the mean
    over many trials of a random one. A simulation that defines only one of them is called through it. When it defines
    both, the study runs ONLY ``run_trial``, so a check must test that code: a check that holds in every single trial
    (:func:`is_per_trial`) is measured on ``run_trial``; any other check is measured on ``run_cell`` only when
    ``cell_runs_trials`` (``run_cell`` really calls ``run_trial``, so its mean comes from the study's own trials), and
    otherwise on ``run_trial`` as before. A ``run_cell`` with a formula of its own would let a bug in ``run_trial`` pass."""
    if "run_trial" not in have:
        return "run_cell"
    if "run_cell" not in have:
        return "run_trial"
    if per_trial or is_per_trial(oracle) or not cell_runs_trials:
        return "run_trial"
    return "run_cell"


def oracle_entry_for(simulate: Path | str, oracle: dict[str, Any] | None = None, *, per_trial: bool = False) -> str:
    """:func:`oracle_entry` for the simulation file at ``simulate`` (read, not run)."""
    path = Path(simulate)
    have = entries(path)
    reaches = False
    if {"run_trial", "run_cell"} <= have:
        try:
            reaches = cell_runs_trials(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            reaches = False
    return oracle_entry(have, oracle, per_trial=per_trial, cell_runs_trials=reaches)


def _built_from_trials(values: dict[str, float], spy: dict[str, Any], oracle: dict[str, Any] | None) -> bool:
    """Whether what ``run_cell`` returned really comes from the trials it ran: it called ``run_trial`` at least once, and
    for every name both return, ``run_cell``'s number agrees (within the check's own limit) with the mean of what the
    trials returned. A ``run_cell`` that runs a trial and drops it for a formula of its own does not agree."""
    if int(spy.get("calls") or 0) <= 0:
        return False
    limit = _oracle.limit_of(oracle)[1] if oracle is not None else None
    means = spy.get("means") or {}
    for key, v in values.items():
        if key in means and isinstance(v, (int, float)):
            allowed = limit if limit is not None else 1e-6 * max(1.0, abs(v))
            if abs(float(v) - float(means[key])) > allowed:
                return False
    return True


async def run_case(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *, cell: dict[str, Any],
                   timeout_s: int, env: dict[str, str] | None = None, thresholds: dict[str, Any] | None = None,
                   trial: int = 0, oracle: dict[str, Any] | None = None,
                   per_trial: bool = False,
                   in_each_run: dict[str, list[Any]] | None = None,
                   paired: bool = False, log: Any = None) -> tuple[dict[str, Any] | None, str]:
    """The ENGINE calls the simulation function itself on one case (``run_trial`` at trial ``trial``, 0 unless asked,
    with the seed FI gives that trial of that cell, or ``run_cell``; :func:`oracle_entry` says which): ``(the metrics it
    returned, "")``, or ``(None, why)``. This is how an oracle is measured without asking the script for the number.

    With values compared inside each run (``in_each_run``), the run is given every one of them, as in the study, with the
    study's seed for the case's setting (and a number the case names that the study does not compare, :func:`case_inside`);
    a number per value is then read at the value the case names, or kept as the list of every value when the case names
    none (:func:`core.in_run.for_case`).

    ``paired``: the study runs trial ``t`` of every setting on one seed (common random numbers), so the case's trial
    gets that seed too (:func:`trial_seed` with ``key=""``), the one the reported trial had."""
    inside = {str(k): list(v) for k, v in (in_each_run or {}).items()}
    if not inside:
        return await _run_case(executor, python, quest_root, module, cell=dict(cell), timeout_s=timeout_s, env=env,
                               thresholds=thresholds, trial=trial, oracle=oracle, per_trial=per_trial,
                               seed_key="" if paired else None, log=log)
    inside = case_inside(inside, cell)
    levels = _in_run.level_cells(inside)
    setting = {k: v for k, v in cell.items() if k not in inside}
    named = {k: v for k, v in cell.items() if k in inside}
    if named and _in_run.level_named(named, levels) is None:
        shown = ", ".join(f"{k}={_fmt(v)}" for k, v in named.items())
        return None, (f"the case sets {shown}; each run compares {', '.join(inside)} inside it, so a case names one of "
                      f"the values each run is given for every one of them ({', '.join(_in_run.level_key(lv) for lv in levels[:6])}), "
                      "or none of them")
    got, why = await _run_case(executor, python, quest_root, module, cell={**setting, **inside}, timeout_s=timeout_s,
                               env=env, thresholds=thresholds, trial=trial, oracle=oracle, per_trial=per_trial,
                               levels=levels, seed_key="" if paired else cell_key(setting), log=log)
    if got is None:
        return None, why
    return _in_run.for_case(got, levels, named), why


async def _run_case(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *, cell: dict[str, Any],
                    timeout_s: int, env: dict[str, str] | None, thresholds: dict[str, Any] | None, trial: int,
                    oracle: dict[str, Any] | None, per_trial: bool, levels: list[dict[str, Any]] | None = None,
                    seed_key: str | None = None, log: Any = None) -> tuple[dict[str, Any] | None, str]:
    """:func:`run_case` on the cell the run is given (``seed_key``: the setting the seed is FI's for, the cell's own key
    when not given)."""
    path = Path(module) if Path(module).is_absolute() else Path(quest_root) / module
    entry = oracle_entry_for(path, oracle, per_trial=per_trial)
    base = int((env or {}).get("FI_REPLICATE_SEED") or 0)
    if entry == "run_cell" and "run_trial" in entries(path):
        # The study runs only run_trial: a run_cell measurement counts only if run_cell really called it (a mention the
        # code never reaches, or a call whose result is dropped for a formula of its own, proves nothing). The AST read
        # is the first filter; this is the decision.
        spy: dict[str, Any] = {}
        values, why = await _run_one(executor, python, quest_root, module, entry="run_cell", cell=dict(cell), seed=None,
                                     timeout_s=timeout_s, env=env, thresholds=thresholds, label="run_cell()",
                                     trial=int(trial), spy_trial=True, spy_out=spy, levels=levels, log=log)
        if values is None or _built_from_trials(values, spy, oracle):
            return values, why
        entry = "run_trial"  # run_cell never ran a trial: the check is measured on the code the study runs
    key = cell_key(cell) if seed_key is None else seed_key
    seed = trial_seed(base, key, int(trial)) if entry == "run_trial" else None
    return await _run_one(executor, python, quest_root, module, entry=entry, cell=dict(cell), seed=seed, timeout_s=timeout_s,
                          env=env, thresholds=thresholds, label=f"{entry}()", trial=int(trial), levels=levels, log=log)


def case_inside(inside: dict[str, list[Any]], cell: dict[str, Any]) -> dict[str, list[Any]]:
    """The values a run on a check's case compares inside it: the study's, with each number the case names that the
    study does not compare added at the end. A factor whose values are numbers can be run at any number, so a check of a
    limiting case (a value near zero, say) runs on the simulation itself; a choice the study does not have is not added
    (the simulation has no such alternative), so it is refused as before."""
    out = {k: list(v) for k, v in inside.items()}
    for k, v in cell.items():
        values = out.get(k)
        if (values and _in_run._number(v) and all(_in_run._number(x) for x in values)
                and not any(_in_run._same_level_value(v, x) for x in values)):
            out[k] = [*values, v]
    return out


async def run_case_trials(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *,
                          cell: dict[str, Any], trials: list[int], timeout_s: int, env: dict[str, str] | None = None,
                          thresholds: dict[str, Any] | None = None, in_each_run: dict[str, list[Any]] | None = None,
                          budget_s: float | None = None, paired: bool = False,
                          store: dict[str, dict[str, dict[str, Any]]] | None = None,
                          ) -> tuple[list[dict[str, Any]], str, list[str], int]:
    """The trials ``trials`` of ``run_trial`` on one case, all in ONE process, each with the seed FI gives that trial of
    that setting (trial 0 gets the seed :func:`run_case` gives it; ``paired``: the seed the paired study gives trial
    ``t`` of every setting, ``key=""``): ``(the values of each trial that finished, in order; why the first trial gave
    none ("" when it did); why each later trial failed; how many trials were started)``. ``budget_s``: no trial is
    started that would not end within it (:func:`_run_batch`). With values compared inside each run (``in_each_run``)
    each trial's values are read as :func:`run_case` reads them.

    ``store``: the trials already run in this one run of the checks (:func:`measure_oracles` makes it), by case, seed
    and settings: a trial in it is read from it instead of being run again, so two checks on the same case share their
    trials. It lives only as long as that one run of the checks: no trial is ever kept from one run of the checks to
    the next, since no key can name everything a trial reads (a data file, an environment variable, a package)."""
    inside = {str(k): list(v) for k, v in (in_each_run or {}).items()}
    levels: list[dict[str, Any]] | None = None
    named: dict[str, Any] = {}
    given, key = dict(cell), cell_key(cell)
    if inside:
        inside = case_inside(inside, cell)
        levels = _in_run.level_cells(inside)
        setting = {k: v for k, v in cell.items() if k not in inside}
        named = {k: v for k, v in cell.items() if k in inside}
        if named and _in_run.level_named(named, levels) is None:
            return [], "the case names a value each run compares that the run is not given", [], 0
        given, key = {**setting, **inside}, cell_key(setting)
    if paired:
        key = ""
    base = int((env or {}).get("FI_REPLICATE_SEED") or 0)
    plan = [{"trial": int(t), "seed": trial_seed(base, key, int(t))} for t in trials]
    if not plan:
        return [], "no trial was asked for", [], 0
    store_key = json.dumps({"module": str(module), "cell": given, "levels": levels, "key": key, "base": base,
                            "thresholds": thresholds or {}, "env": dict(sorted((env or {}).items()))},
                           sort_keys=True, default=str) if store is not None else ""
    stored = dict(store.get(store_key) or {}) if store is not None else {}
    todo = [p for p in plan if str(p["trial"]) not in stored]
    result: Any = None
    if todo:
        rows, result = await _run_batch(executor, python, quest_root, module, entry="run_trial", cell=given,
                                        trials=todo, timeout_s=timeout_s, env=env, thresholds=thresholds,
                                        levels=levels, budget_s=budget_s)
        fresh: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.get("load_error"):
                return [], str(row["load_error"]), [], 0
            if row.get("status") == "ok":
                fresh[str(row["trial"])] = {"status": "ok", "values": dict(row.get("values") or {})}
            else:
                where = last_frame(str(row.get("traceback") or ""), quest_root)
                fresh[str(row["trial"])] = {"status": "failed", "reason": str(row.get("reason") or "run_trial() failed")
                                            + (f" (at {where})" if where else "")}
        stored = {**stored, **fresh}
        if fresh and store is not None:
            store[store_key] = stored
    values: list[dict[str, Any]] = []
    failed: list[str] = []
    started = 0
    for p in plan:
        row = stored.get(str(p["trial"]))
        if row is None:
            continue  # not started (the time ran out first)
        started += 1
        if row.get("status") == "ok":
            got = dict(row.get("values") or {})
            values.append(_in_run.for_case(got, levels, named) if levels else got)
            continue
        if p["trial"] == plan[0]["trial"]:
            # The first trial is the one a single-trial check judges: said as before.
            return [], str(row.get("reason") or "run_trial() failed"), [], started
        failed.append(str(row.get("reason") or "run_trial() failed"))
    if not values:
        return [], ("run_trial() ran out of time" if getattr(result, "timed_out", False)
                    else f"run_trial() did not report (exit code {getattr(result, 'returncode', None)})"), failed, started
    return values, "", failed, started


def _same_value(a: Any, b: Any) -> bool:
    """Two values a trial returned are the same: numbers to within rounding (two NaNs are the same), anything else equal."""
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
    if not numbers:
        return a == b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


async def run_cell_again(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *, timeout_s: int,
                         env: dict[str, str] | None = None, thresholds: dict[str, Any] | None = None,
                         most: int = 3, in_each_run: dict[str, list[Any]] | None = None,
                         log: Any = None) -> tuple[bool | None, str]:
    """Whether ``run_cell`` returns the same numbers when FI calls it again, each call in a new process, on up to
    ``most`` settings that ran to the end (the fastest first, and only while their first runs together took under a
    tenth of ``timeout_s``; always at least one): ``(True, "")``, ``(False, what differed)``, or ``(None, why it could not
    be checked)``.

    This is what lets one run per setting stand for a protocol that asked for several: a simulation whose code shows no
    source of random numbers can still draw some (a library that samples by default, the order of a set), and a second
    call that returns other numbers says so where reading the code cannot. Settings that are not called again are not
    checked: randomness that only one of them has can be missed."""
    try:
        record = json.loads((Path(quest_root) / RUN_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "FI's run record could not be read"
    done: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for cell in record.get("cells") or []:
        row = next((r for r in cell.get("rows") or [] if isinstance(r, dict) and r.get("status") == "ok"), None)
        if row is not None and isinstance(cell.get("cell"), dict):
            seconds = row.get("duration_s")
            done.append((float(seconds) if isinstance(seconds, (int, float)) else float("inf"), cell, row))
    if not done:
        return None, "no setting ran to the end"
    done.sort(key=lambda item: item[0])
    spent = 0.0
    inside = {str(k): list(v) for k, v in (in_each_run or {}).items()}
    levels = _in_run.level_cells(inside)
    for index, (seconds, cell, row) in enumerate(done[:max(1, most)]):
        spent += seconds
        if index and spent > timeout_s / 10:
            break
        # The setting as the study gave it (with every value each run compares), so the numbers compare one for one.
        again, why = None, ""
        for _attempt in range(2):  # a new venv's first start can fail before it reaches the code (see the execute step)
            if not inside:
                again, why = await run_case(executor, python, quest_root, module, cell=dict(cell["cell"]),
                                            timeout_s=timeout_s, env=env, thresholds=thresholds, trial=0, log=log)
            else:
                again, why = await _run_case(executor, python, quest_root, module, cell={**cell["cell"], **inside},
                                             timeout_s=timeout_s, env=env, thresholds=thresholds, trial=0, oracle=None,
                                             per_trial=False, levels=levels, seed_key=cell_key(cell["cell"]), log=log)
            if again is not None or "did not report" not in why:
                break
        if again is None:
            return None, why
        first = {_in_run.record_key(k, n) if k else n: v for k, n, v in _in_run.flat("", row.get("values") or {})}
        second = {_in_run.record_key(k, n) if k else n: v for k, n, v in _in_run.flat("", again)}
        changed = sorted(k for k in set(first) | set(second)
                         if k not in first or k not in second or not _same_value(first[k], second[k]))
        if changed:
            shown = ", ".join(f"{k}: {first.get(k)!r} then {second.get(k)!r}" for k in changed[:3])
            return False, f"the setting {cell.get('key')} returned other numbers ({shown})"
    return True, ""


async def _seeded_once(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *, cell: dict[str, Any],
                       seed: int, timeout_s: int, env: dict[str, str] | None, thresholds: dict[str, Any] | None,
                       log: Any = None) -> tuple[dict[str, float] | None, float, str]:
    """One call of ``run_trial`` in a new process with FI's seed: ``(its numbers by record key, the seconds the call
    took, "")``, or ``(None, 0, why)``."""
    rows, result = await _run_batch(executor, python, quest_root, module, entry="run_trial", cell=dict(cell),
                                    trials=[{"trial": 0, "seed": int(seed)}], timeout_s=timeout_s, env=env,
                                    thresholds=thresholds, log=log)
    for row in rows:
        if row.get("load_error"):
            return None, 0.0, str(row["load_error"])
        if row.get("status") == "ok":
            numbers = {_in_run.record_key(k, n) if k else n: v for k, n, v in _in_run.flat("", row.get("values") or {})}
            seconds = row.get("duration_s")
            return numbers, float(seconds) if isinstance(seconds, (int, float)) else 0.0, ""
        if row.get("status") == "failed":
            return None, 0.0, str(row.get("reason") or "run_trial failed")
    return None, 0.0, ("run_trial ran out of time" if getattr(result, "timed_out", False)
                       else f"run_trial did not report (exit code {result.returncode})")


def _is_run_time(first: tuple[Any, float], again: tuple[Any, float]) -> bool:
    """A number that differs between two same-seed runs is the run's own time when, in both runs, it lies between zero and
    the seconds the call took (FI's harness times the call): a time cannot be longer than the call it was taken in. No
    name is read."""
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= seconds * 1.05 + 1e-3
               for v, seconds in (first, again))


async def run_trial_seed_check(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *,
                               cell: dict[str, Any], seed: int, timeout_s: int, deterministic: bool = False,
                               varies: set[str] | frozenset[str] | None = None, env: dict[str, str] | None = None,
                               thresholds: dict[str, Any] | None = None, log: Any = None) -> tuple[bool | None, str]:
    """Whether ``run_trial`` uses the seed FI gives it: FI calls it on one setting with seed ``s``, with ``s`` again and
    with ``s + 1`` (each call in a new process): ``(True, "")``, ``(False, what is wrong)`` or ``(None, why it could not
    be judged)``.

    * Two calls with the same seed must return the same numbers. A number that differs and lies within the call's own
      duration is a time (it is left out of the next step and named in the answer); any other number that differs means
      the run does not repeat with its seed.
    * The call with ``s + 1`` must change at least one of the numbers that repeated (only those named in ``varies``,
      when the protocol says which numbers have spread), or the run ignores the seed (an own generator, an own fixed
      seed or none).

    A ``deterministic`` study draws nothing and is not checked."""
    if deterministic:
        return None, "the study is deterministic, so no seed is used"
    kw: dict[str, Any] = dict(cell=cell, timeout_s=timeout_s, env=env, thresholds=thresholds, log=log)
    first = await _seeded_once(executor, python, quest_root, module, seed=seed, **kw)
    if first[0] is None:
        return None, first[2]
    again = await _seeded_once(executor, python, quest_root, module, seed=seed, **kw)
    if again[0] is None:
        return None, again[2]
    other = await _seeded_once(executor, python, quest_root, module, seed=seed + 1, **kw)
    if other[0] is None:
        return None, other[2]
    a, b, c = first[0], again[0], other[0]
    names = sorted(set(a) | set(b))
    moved = [n for n in names if n not in a or n not in b or not _same_value(a[n], b[n])]
    timings = [n for n in moved if n in a and n in b and _is_run_time((a[n], first[1]), (b[n], again[1]))]
    unsteady = [n for n in moved if n not in timings]
    if unsteady:
        shown = ", ".join(f"{n}: {a.get(n)!r} then {b.get(n)!r}" for n in unsteady[:3])
        return False, f"the run does not repeat with the same seed ({shown})"
    stable = [n for n in names if n not in timings and (not varies or n in varies)]
    if not stable:
        return None, ("every number the run returned is its own time" if timings else "the run returned no number to compare")
    if all(n in c and _same_value(a[n], c[n]) for n in stable):
        return False, "the run ignores the seed FI gives it (another seed returned the same numbers)"
    return True, ""


def _check_from(name: str, measure: str, values: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The check an oracle's measure gives on what one run returned: ``(check or None, the problem or "")``."""
    if measure in values:  # one returned name, as every oracle was before formulas: read as it is
        return {"name": name, "value": values[measure], "measured_by": "engine", "returned": dict(values)}, ""
    # How the number is computed is the protocol's formula of what the simulation returned, applied here by the
    # engine (core/oracle_forms.py): the script never decides the representation.
    got = _forms.evaluate(measure, values)
    returned = ", ".join(sorted(values)) or "nothing"
    if not got.unreadable and not _forms.names(measure):
        # A number that takes nothing from the simulation (`0`, `pi - pi`) is not a measurement of it.
        return None, (f"the oracle {name!r}: its number is computed as `{measure}`, which takes nothing the "
                      "simulation returns, so it is not a measurement of the simulation")
    if got.unreadable or (got.missing and _forms.is_name(measure)):
        return None, (
            f"the oracle {name!r}: the simulation returned {returned} but the oracle measures {measure!r} (the "
            "simulation function must return it" + (f"; as a formula it cannot be read: {got.unreadable}" if
                                                     got.unreadable and not _forms.is_name(measure) else "") + ")"
        )
    if got.missing:
        return None, (
            f"the oracle {name!r}: its number is computed as `{measure}`, from "
            f"{', '.join(repr(n) for n in got.missing)}, which the simulation did not return (it returned "
            f"{returned}; the simulation function must return every name the formula uses)"
        )
    if got.value is None:
        return ({"name": name, "value": None, "measured_by": "engine", "computed_as": measure,
                 "formula_problem": got.problem, "returned": dict(values)},
                f"the oracle {name!r}: its number is computed as `{measure}`, and on its case {got.problem}")
    return {"name": name, "value": got.value, "measured_by": "engine", "computed_as": measure,
            "returned": dict(values)}, ""


def _scalar(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


async def _measure_mean(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *,
                        oracle: dict[str, Any], case: dict[str, Any], measure: str, timeout_s: int,
                        budget_s: float, env: dict[str, str] | None, thresholds: dict[str, Any] | None,
                        in_each_run: dict[str, list[Any]] | None, means: Collection[str], paired: bool,
                        most_trials: int, store: dict[str, Any] | None = None, proportions: Collection[str] = (),
                        repeated: bool | None = None,
                        ) -> tuple[dict[str, Any] | None, str, bool] | None:
    """An oracle whose number is a mean over trials (:func:`core.mc_error.reads_a_mean`), decided on trials of
    ``run_trial`` (core/mc_error.py): a first batch, then more trials, sized from their spread, until three standard
    errors put the mean inside or outside the tolerance, ``budget_s`` is spent or ``most_trials`` are run (then it is not
    decided). ``(check or None, problem or "", whether the case could not be run at all)``, or ``None`` when its measure
    is not a number per trial of the names ``run_trial`` returns (then it is measured as any other check, which says what
    is wrong). A number that is the same in every trial of a first batch of at least ``MIN_TRIALS`` is judged at once on
    trial 0's number."""
    name = str(oracle["name"]).strip()
    expected, limit, _mode = _oracle.limit_of(oracle)
    most = max(1, int(most_trials))
    started = time.monotonic()
    rows, why, failed, ran = await run_case_trials(
        executor, python, quest_root, module, cell=case, trials=list(range(min(_mc.PILOT_TRIALS, most))),
        timeout_s=timeout_s, env=env, thresholds=thresholds, in_each_run=in_each_run,
        budget_s=budget_s * _mc.PILOT_SHARE, paired=paired, store=store)
    if not rows:
        return None, why, True
    first = rows[0]
    used = _mc.measure_names(measure, first)
    if not used or not all(_scalar(first.get(n)) for n in used):
        return None
    stat = _mc.statistic(rows, measure, means, expected, limit, proportions, repeated=repeated)
    if stat is None:
        return None
    # A number is taken for a constant only when at least MIN_TRIALS trials agree: two or three equal draws of a random
    # number are not a constant (a starved first batch of a 0/1/2 count agreed so, and was judged on trial 0 alone). A
    # 0/1 outcome of a random simulation never is: it is judged on its binomial interval (mc_error.statistic).
    if len(rows) >= _mc.MIN_TRIALS and not stat["varies"] and not stat.get("binary"):
        # The check's own number did not move: judged at once on trial 0's number, as a deterministic check is.
        check, problem = _check_from(name, measure, first)
        verdict, why_failed = _mc.failed_share_verdict(len(failed) + int(stat.get("dropped") or 0), ran, None)
        if check is not None and why_failed:
            check["mc"] = _mc.record(stat, read_as="once", verdict=verdict, why=why_failed,
                                     failed=len(failed) + int(stat.get("dropped") or 0), started=ran,
                                     failed_example=clip_keeping_place(failed[0], 200) if failed else "")
        return check, problem, False
    verdict, why_open = None, ""
    while True:
        verdict = _mc.mean_verdict(stat, expected, limit)
        if verdict is not None or stat.get("value") is None:
            break
        if len(rows) >= most:
            why_open = "most_trials"
            break
        left = budget_s - (time.monotonic() - started)
        if left < 0.5:
            why_open = "time"
            break
        wanted = _mc.trials_wanted(stat, float(expected), float(limit), most=most)  # type: ignore[arg-type]
        more, _why, more_failed, more_ran = await run_case_trials(
            executor, python, quest_root, module, cell=case, trials=list(range(ran, ran + max(1, wanted - len(rows)))),
            timeout_s=max(1, int(math.ceil(left)) + 1), env=env, thresholds=thresholds, in_each_run=in_each_run,
            budget_s=left, paired=paired, store=store)
        rows, failed, ran = rows + more, failed + more_failed, ran + more_ran
        if not more:
            why_open = "time"
            break
        stat = _mc.statistic(rows, measure, means, expected, limit, proportions, repeated=repeated) or stat
    lost = len(failed) + int(stat.get("dropped") or 0)
    verdict, why_failed = _mc.failed_share_verdict(lost, ran, verdict)
    record = _mc.record(stat, read_as="mean", verdict=verdict, why=why_failed or why_open, failed=lost, started=ran,
                        failed_example=clip_keeping_place(failed[0], 200) if failed else "")
    keys = [k for k, v in first.items() if _scalar(v)]
    returned = {k: math.fsum(float(r[k]) for r in rows) / len(rows) for k in keys
                if all(_scalar(r.get(k)) and math.isfinite(r[k]) for r in rows)}
    returned["n_trials"] = float(stat["trials"])
    check: dict[str, Any] = {"name": name, "value": stat.get("value"), "measured_by": "engine",
                             **({} if measure in first else {"computed_as": measure}), "returned": returned,
                             "mc": record}
    if stat.get("value") is None:
        problem = stat.get("problem") or "the formula gives no number"
        check["formula_problem"] = problem
        return check, f"the oracle {name!r}: its number is computed as `{measure}`, and on its case {problem}", False
    return check, "", False


async def _measure_every(executor: Any, python: Path | str, quest_root: Path, module: Path | str, *,
                         oracle: dict[str, Any], case: dict[str, Any], measure: str, timeout_s: int,
                         budget_s: float, env: dict[str, str] | None, thresholds: dict[str, Any] | None,
                         in_each_run: dict[str, list[Any]] | None, paired: bool,
                         most_trials: int, store: dict[str, Any] | None = None,
                         ) -> tuple[dict[str, Any] | None, str, bool] | None:
    """A rule every trial keeps (core/mc_error.py), judged on a first batch of ``run_trial`` trials (trial 0 among them,
    with the seed :func:`run_case` gives it, so it is never weaker than trial 0 alone): the trial furthest from the
    expected value decides. Same return as :func:`_measure_mean`. Trials that all give the same number are judged as
    before, on trial 0's (and carry no ``mc``)."""
    name = str(oracle["name"]).strip()
    expected, limit, _mode = _oracle.limit_of(oracle)
    rows, why, failed, ran = await run_case_trials(
        executor, python, quest_root, module, cell=case, trials=list(range(min(_mc.PILOT_TRIALS, max(1, most_trials)))),
        timeout_s=timeout_s, env=env, thresholds=thresholds, in_each_run=in_each_run, budget_s=budget_s, paired=paired,
        store=store)
    if not rows:
        return None, why, True
    first = rows[0]
    used = _mc.measure_names(measure, first)
    if not used or not all(_scalar(first.get(n)) for n in used):
        return None
    check, problem = _check_from(name, measure, first)
    if check is None or check.get("value") is None or expected is None or limit is None:
        return check, problem, False  # trial 0 gives no number: said as before
    values = _mc.per_trial_values(rows, measure)
    finite = [v for v in values if v is not None]
    lost = len(failed) + (len(values) - len(finite))
    varies = any(abs(v - finite[0]) > 1e-15 + 1e-12 * abs(finite[0]) for v in finite[1:])
    at = _mc.worst(values, float(expected))
    if at is not None and varies:
        check, problem = _check_from(name, measure, rows[at])
    if check is None or check.get("value") is None:
        return check, problem, False
    verdict = "pass" if abs(float(check["value"]) - float(expected)) <= float(limit) else "fail"
    verdict, why_failed = _mc.failed_share_verdict(lost, ran, verdict)
    if varies or why_failed:
        check["mc"] = _mc.record({"trials": len(finite), "dropped": len(values) - len(finite), "varies": varies,
                                  "on": "trials"}, read_as="every_trial", verdict=verdict, why=why_failed,
                                 failed=lost, started=ran,
                                 failed_example=clip_keeping_place(failed[0], 200) if failed else "")
    return check, problem, False


async def measure_oracles(executor: Any, python: Path | str, quest_root: Path, module: Path | str,
                          oracles: list[dict[str, Any]], *, timeout_s: int, env: dict[str, str] | None = None,
                          thresholds: dict[str, Any] | None = None, case_env: dict[str, str] | None = None,
                          in_each_run: dict[str, list[Any]] | None = None, log: Any = None,
                          mean_names: Collection[str] | None = None, most_trials: int | None = None,
                          paired: bool = False, proportion_names: Collection[str] | None = None,
                          repeated: bool | None = None,
                          ) -> tuple[list[dict[str, Any]], list[str], bool]:
    """``(checks, problems, timed_out)`` for the declared oracles. An oracle with a case is measured by the ENGINE: it calls
    the simulation function on that case and reads the measure from what comes back (``measured_by: "engine"``). The rest
    are read from the script's ``oracle()`` (``measured_by: "script"``), called once and only when one needs it.

    On a simulation with ``run_trial`` (core/mc_error.py), a check whose number is a mean over trials is decided on as
    many trials as its tolerance needs, at most ``most_trials``, and a rule every trial keeps is judged on the worst of a
    first batch of trials; such a check carries ``mc``: how it was read, its trials, their error and the verdict they
    gave (``pass``, ``fail`` or ``undecided``). All of these checks share ``timeout_s`` (each is given what is left
    divided by the checks still to measure, so one that is decided quickly leaves its time to the others).
    ``mean_names``: the protocol's metrics that are means over runs (a mean check whose names are all among them is the
    formula on their means). ``paired``: the study gives trial ``t`` of every setting one seed, and so do the checks."""
    checks: list[dict[str, Any]] = []
    problems: list[str] = []
    timed_out = False
    # Every case, every batch of trials and the script's oracle() run from a copy of code/ outside the quest folder
    # (core/outside_run.py), as the study's cells do.
    executor = _outside_executor(executor, quest_root, log)
    # The simulation must not be able to tell it is being checked: a case runs with the environment a trial of the main run
    # has (``case_env``; the engine gives it exactly that), never with the oracle-mode variable ``FI_ORACLE`` set for the script's own oracle().
    cenv = case_env if case_env is not None else {k: v for k, v in (env or {}).items() if k != "FI_ORACLE"}
    path = Path(module) if Path(module).is_absolute() else Path(quest_root) / module
    try:
        has_trial = "run_trial" in entries(path)
    except (OSError, ValueError):
        has_trial = False
    means = frozenset(mean_names or ())
    proportions = frozenset(proportion_names or ())  # judged on their binomial interval (mc_error.statistic)
    most = int(most_trials or _mc.MAX_TRIALS)

    def on_trials(oracle: dict[str, Any]) -> str:
        """``"mean"``, ``"every"`` or ``""`` (measured as before, on one run)."""
        if not has_trial or _oracle.case_of(oracle) is None:
            return ""
        if _mc.reads_a_mean(oracle):
            return "mean"
        try:
            return "every" if oracle_entry_for(path, oracle) == "run_trial" else ""
        except (OSError, ValueError):
            return ""

    left_checks = sum(1 for o in oracles if on_trials(o))
    began = time.monotonic()
    store: dict[str, Any] = {}  # the trials run in this one run of the checks, shared by the checks on one case
    for oracle in oracles:
        own = _oracle.case_of(oracle)
        if own is None:
            continue
        case, measure = own
        name = str(oracle["name"]).strip()
        how = on_trials(oracle)
        if how:
            share = max(1.0, (timeout_s - (time.monotonic() - began)) / max(left_checks, 1))
            left_checks -= 1
            kw = dict(oracle=oracle, case=case, measure=measure, timeout_s=timeout_s, budget_s=share, env=cenv,
                      thresholds=thresholds, in_each_run=in_each_run, paired=paired, most_trials=most,
                      store=store)
            got = (await _measure_mean(executor, python, quest_root, module, means=means, proportions=proportions,
                                       repeated=repeated, **kw)
                   if how == "mean"
                   else await _measure_every(executor, python, quest_root, module, **kw))
            if got is not None:
                check, problem, not_run = got
                if not_run:
                    problems.append(f"the oracle {name!r}: the simulation could not be run on its case "
                                    f"({clip_keeping_place(problem, 300)})")
                    timed_out = timed_out or "ran out of time" in problem
                else:
                    if check is not None:
                        checks.append(check)
                    if problem:
                        problems.append(problem)
                continue
        values, why = await run_case(executor, python, quest_root, module, cell=case, timeout_s=timeout_s, env=cenv,
                                     thresholds=thresholds, oracle=oracle, in_each_run=in_each_run, paired=paired, log=log)
        if values is None:
            problems.append(f"the oracle {name!r}: the simulation could not be run on its case "
                            f"({clip_keeping_place(why, 300)})")
            timed_out = timed_out or "ran out of time" in why
            continue
        check, problem = _check_from(name, measure, values)
        if check is not None:
            checks.append(check)
        if problem:
            problems.append(problem)
    if any(_oracle.case_of(o) is None for o in oracles):
        values, why = await run_oracle(executor, python, quest_root, module, timeout_s=timeout_s, env=env, thresholds=thresholds,
                                       log=log)
        if values is None:
            problems.append(f"simulate.py's oracle() did not give its values: "
                            f"{clip_keeping_place(why.strip(), 300) or 'no reason given'}")
            timed_out = timed_out or "ran out of time" in why
        else:
            checks += [{"name": k, "value": v, "measured_by": "script"} for k, v in values.items()
                       if not any(str(o["name"]).strip().lower() == str(k).strip().lower() for o in oracles if _oracle.case_of(o))]
    return checks, problems, timed_out


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
            # A run that compared values inside it is a row of its setting (its numbers per run) and one per value.
            per_run, per_level = _in_run.split(values)
            for where, named in [(key, per_run), *((_in_run.record_key(key, lk), v) for lk, v in per_level.items())]:
                if named or where == key:
                    out.setdefault(where, []).append({"cell": where, "trial": row.get("trial"), "seed": row.get("seed"),
                                                      "values": named})
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
    grid = _search.study_grid(protocol)
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
            # Each value's numbers are its own setting's (core/in_run.py): a number per run is counted once per run.
            for where, name, value in _in_run.flat(str(cell.get("key")), values):
                key = _value_key(value)
                if key is not None:
                    out.setdefault(where, {}).setdefault(str(name), Counter())[key] += 1
    problems = []
    if altered:
        from core import run_manifest as _rm

        problems = [_rm.fabricated(
            f"{altered} trial(s) in FI's run record (.fi/trials/run.json) no longer match the ledger's hash of their "
            f"values: the record was changed after the trials ran, so FI runs the trials again")]
        (root / RUN_RECORD).unlink(missing_ok=True)  # the next run cannot reuse it: the trials are run afresh
    return out, problems


def recorded_series(quest_root: Path, names: set[str] | list[str]) -> dict[str, dict[str, list[float]]]:
    """For each of ``names`` (keys of ``run_trial``'s dict), its values per setting (cell key) in trial order, from FI's run
    record, counting only trials that ran to the end and whose values still match the ledger's hash (as
    :func:`recorded_values_by_cell` checks them; a changed record is left to that function to report). Empty when FI kept
    no record."""
    root = Path(quest_root)
    wanted = {str(n) for n in names}
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
    out: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for cell in record.get("cells") or [] if isinstance(record, dict) else []:
        for row in cell.get("rows") or []:
            if not isinstance(row, dict) or row.get("status") != "ok" or _trial_id(row) is None:
                continue
            values = row.get("values") or {}
            digest = hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=True).encode("utf-8")).hexdigest()
            if hashes.get((str(cell.get("key")), _trial_id(row))) != digest:
                continue
            for where, name, v in _in_run.flat(str(cell.get("key")), values):
                if (name in wanted and isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v)):
                    out.setdefault(name, {}).setdefault(where, []).append((_trial_id(row), float(v)))
    return {name: {key: [v for _, v in sorted(rows)] for key, rows in cells.items()} for name, cells in out.items()}


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


#: Fewest reported values a selection of the trials is looked for in: a single value is the extreme of any list.
_SELECTION_MIN_VALUES = 3

#: Fewest ways to choose the reported values out of a setting's trials for "they are exactly the largest" to be worth
#: saying: an honest subset of 3 of 20 trials is the top 3 once in 570 draws, so a small setting is not judged.
_SELECTION_MIN_CHOICES = 20000


def _extreme_selection(values: list[Any], per_cell: dict[str, dict[str, Counter]], cells: list[str], name: str,
                       grid: dict[str, list[Any]], thresholds: list[float]) -> str | None:
    """``"largest"`` or ``"smallest"`` when ``values`` are exactly the largest (or the smallest) values of ``name`` among the trials of ``cells``, as they are or divided by a size setting, while the trials returned more values than
    that: the signature of the best results picked out of every setting. Not when a threshold the protocol fixes
    separates them from the rest of the trials: that is what a mean over the trials beyond a cut-off looks like."""
    numbers = [float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool)
               and math.isfinite(float(x))]
    k = len(numbers)
    if k < _SELECTION_MIN_VALUES:
        return None
    for form in _derived_counts(per_cell, cells, name, grid):
        pool = sorted(float(v) for v, n in form.items() for _ in range(max(int(n), 0)))
        if len(pool) <= k or math.comb(len(pool), k) < _SELECTION_MIN_CHOICES:
            continue
        for label, part, edge in (("largest", pool[-k:], (pool[-k - 1], pool[-k])),
                                  ("smallest", pool[:k], (pool[k - 1], pool[k]))):
            if _missing_exact(numbers, Counter(_exact_key(v) for v in part)):
                continue
            if any(min(edge) <= t <= max(edge) for t in thresholds):
                continue
            return label
    return None


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
    grid = _search.study_grid(protocol)
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
        out.append(_rm.fabricated(
            f"the analysis reports `{where}` for `{metric}`, a mean over a subset of the trials, with values FI's trials "
            f"never produced in {'those settings' if len(cells) < len(all_cells) else 'the run'}: "
            + _rm.DERIVED
        ))

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

    cutoffs = [float(v) for v in (protocol.get("thresholds") or {}).values()
               if isinstance(v, (int, float)) and not isinstance(v, bool)] if isinstance(protocol.get("thresholds"), dict) else []
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
                        side = (_extreme_selection(v, per_cell, own or all_cells, metric, grid, cutoffs)
                                if cells and exact is None else None)
                        n_given = sum(1 for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)
                                      and math.isfinite(float(x)))
                        if side:
                            out.append(
                                f"at `{path or 'the top level'}`, your {n_given} values of `{metric}` are exactly the "
                                f"{n_given} {side} values of the trials of those settings. That is what the trials beyond "
                                f"a cut-off look like, and also what the best {n_given} picked by hand look like, and FI "
                                f"can tell them apart only if each trial records whether it is in the subset: have "
                                f"run_trial return `{given}` as 1 or 0 for each trial. A cut-off the protocol fixes "
                                "(`thresholds`) that falls between them and the other trials is accepted as it is."
                            )
                        if exact is not None and isinstance(count, (int, float)) and not isinstance(count, bool) \
                                and float(count) != exact:
                            out.append(_rm.fabricated(
                                f"at `{path or 'the top level'}`, `{given}_count` says {count:g}, but FI's trials of "
                                f"those settings returned `{given}` = 1 for {exact:g} of them: the mean averages "
                                "every trial in the subset, not a selection"
                            ))
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


#: A value read through float32 (``np.float32(v).tolist()``) moves by at most this much of itself: it is still the
#: trial's value. A 0 read that way matches a recorded value below float32's smallest normal number.
_FLOAT32 = 2.0 ** -24
_FLOAT32_TINY = 1.1754943508222875e-38


def _not_among(values: list[Any], recorded: Counter) -> list[float]:
    """The numbers of ``values`` that are not trial values, each trial's value used once. A number printed to fewer
    places than the trial returned (``1.2346`` for ``1.23456789``, ``0.0`` for ``1e-05``, ``3`` for ``2.8``) is that
    value: it matches a recorded value within half its last printed place. The most precise numbers are matched first,
    each to the nearest value left, so a coarse one cannot take the value a precise one needed. A value within a relative
    :data:`_FLOAT32` of a recorded one is that value read through float32."""
    import bisect

    base, floats, keys = _prepared(recorded)
    left = dict(base)
    pending: list[Any] = []
    for x in values:
        key = _value_key(x)
        if key is None:
            continue
        n = left.get(key, 0)
        if n > 0:
            left[key] = n - 1
        else:
            pending.append(x)
    if not pending:
        return []
    extra = []
    for x in sorted(pending, key=_half_step):
        value, half = float(x), _half_step(x)
        if not math.isfinite(value):
            extra.append(value)
            continue
        # The nearest value left on each side: the recorded values in order, those used up passed over. A value used up
        # that is already too far means every one past it is too (the distance grows faster than the float32 margin).
        i = bisect.bisect_left(floats, value)
        best: tuple[float, int] | None = None
        for step, start in ((-1, i - 1), (1, i)):
            j = start
            while 0 <= j < len(floats):
                near = _near(floats[j], value, half)
                if left[keys[j]] > 0:
                    if near and (best is None or abs(floats[j] - value) < best[0]):
                        best = (abs(floats[j] - value), j)
                    break
                if not near:
                    break
                j += step
        if best is not None:
            left[keys[best[1]]] -= 1  # the nearest value left: the same choice as a search of every value
        else:
            extra.append(value)
    return extra


def _prepared(recorded: Counter) -> tuple[dict[str, int], list[float], list[str]]:
    """``recorded`` read once for :func:`_not_among`: its counts by value key, and its finite values in order with their
    keys. Kept on the Counter while it is unchanged: one check matches many lists against the same record (each list
    against each recorded name), and reading a record of 25,000 values again for every list was nearly all of its time."""
    kept = getattr(recorded, "_fi_prepared", None)
    if kept is not None and dict.__eq__(kept[0], recorded):
        return kept[1]
    base: dict[str, int] = {}
    for k, n in recorded.items():
        key = k if isinstance(k, str) else _value_key(k)
        if key is not None:
            base[key] = base.get(key, 0) + n
    finite = sorted((f, key) for key in base if base[key] > 0 and math.isfinite(f := _as_float(key)))
    prepared = (base, [f for f, _key in finite], [key for _f, key in finite])
    try:
        recorded._fi_prepared = (dict(recorded), prepared)  # type: ignore[attr-defined]
    except AttributeError:  # a plain dict, not a Counter: read each time
        pass
    return prepared


def _near(recorded: float, value: float, half: float) -> bool:
    """Whether ``value``, printed to within ``half`` of its last place, is the ``recorded`` value (:func:`_not_among`)."""
    return abs(recorded - value) <= max(half * (1 + 1e-9), _FLOAT32 * max(abs(recorded), abs(value)),
                                        _FLOAT32_TINY if value == 0 else 0.0)


def drawn_from(values: list[Any], recorded: Counter) -> bool:
    """Whether every number of ``values`` is one of the recorded values, matched as :func:`_not_among` matches one but
    with the copies not counted: a list drawn from the trials (a bootstrap resample, a subset given twice), not the
    trials themselves. Such a list is derived from FI's record: it is not numbers no trial produced, and it never counts
    as the trials (``core.run_manifest._attribute``)."""
    import bisect

    keys = {k if isinstance(k, str) else _value_key(k) for k, n in recorded.items() if n > 0} - {None}
    pool = sorted({v for v in map(_as_float, keys) if math.isfinite(v)})
    for x in values:
        key = _value_key(x)
        if key is None or key in keys:
            continue
        value = float(x)
        if not math.isfinite(value):
            return False
        i = bisect.bisect_left(pool, value)
        if not any(_near(pool[j], value, _half_step(x)) for j in (i - 1, i) if 0 <= j < len(pool)):
            return False
    return True


def _as_float(key: Any) -> float:
    try:
        return float(key)
    except (TypeError, ValueError):
        return math.nan


def held_as(name: str, values: list[Any], recorded: dict[str, Counter], keys: list[Any] | tuple[Any, ...] = ()) -> str:
    """The name of FI's record a ``<name>_values`` list is held to over all settings: ``name`` when every value is one of
    its trials'; else a recorded name a key on the list's path gives (``x1_power/power_values``) when every value is one of
    that name's; else the one other recorded name whose trials hold every value (a list printed under a name of its own);
    else ``name``. A list held to another name is judged setting by setting in :mod:`core.run_manifest`
    (``_attribute``), which says it in run.log or calls it fabricated."""
    if name not in recorded or not _not_among(values, recorded[name]):
        return name
    named = [k for k in keys if isinstance(k, str) and k != name and k in recorded]
    if named and not _not_among(values, recorded[named[-1]]):
        return named[-1]
    others = [o for o in recorded if o != name and not _not_among(values, recorded[o])]
    return others[0] if len(others) == 1 else name


def mean_beside(mapping: dict[str, Any], name: str, values: list[Any]) -> str | None:
    """A sentence when a number printed beside ``<name>_values`` as what they give (``<name>``, ``<name>_mean`` or
    ``<name>: {"mean": ...}``) is neither the mean of that list's finite values nor their standard deviation (a metric
    may be the spread of the per-trial values, as ``SD(Ea_hat)/Ea_true`` beside each trial's ``Ea_hat/Ea_true``); ``None``
    when it is one of them, within what rounding the printed numbers could have moved it (the number's last printed place,
    each value's, and float32), or when there is none."""
    import statistics

    finite = [float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    if not finite:
        return None
    mean = math.fsum(finite) / len(finite)
    spreads = [statistics.pstdev(finite)] + ([statistics.stdev(finite)] if len(finite) > 1 else [])
    slack = max(_half_step(x) for x in finite) + 2 ** -24 * max(abs(x) for x in finite)
    given = []
    for key in (name, f"{name}_mean"):
        v = mapping.get(key)
        if isinstance(v, dict):
            v = v.get("mean")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            given.append((key, v))
    for key, v in given:
        tol = (_half_step(v) + slack) * (1 + 1e-9) + 1e-9 * max(1.0, abs(mean))
        if all(abs(v - s) > tol for s in [mean, *spreads]):
            return (f"`{key}` is printed as {v:g} beside `{name}_values`, whose {len(finite)} value(s) have a mean of "
                    f"{mean:.6g} (and a standard deviation of {spreads[-1]:.6g})")
    return None


def reported_values_not_run(recorded: dict[str, Counter], result_json: Any) -> list[str]:
    """Each ``<name>_values`` list the analysis printed, for a ``<name>`` FI's trials returned, that holds values those
    trials never produced: one sentence each. A list that holds more copies of some trials' values than the trials gave,
    every value one of theirs (:func:`drawn_from`: a bootstrap resample), is derived from them, not made up; it never
    counts as the trials (``core.run_manifest``), so one standing in for them leaves them short.

    The run-manifest check counts a metric's values against the trials; a script could still print that many numbers
    of its own. Under the trial contract FI holds every trial's value, so a list the analysis says it computed from is
    checked value by value. A list named for something the analysis derived itself (no trial returned that name) is not
    FI's to check here; a mean over a subset of the trials is checked by :func:`given_values_not_run`."""
    from core import run_manifest as _rm

    out: list[str] = []

    def walk(node: Any, path: str, keys: tuple[Any, ...] = ()) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else str(k)
                if isinstance(k, str) and k.endswith("_values") and isinstance(v, list) and k[:-7] in recorded:
                    # the mean printed beside a list is that list's mean (a made-up mean beside honest values is not)
                    off = mean_beside(node, k[:-7], v)
                    if off:
                        out.append(_rm.fabricated(
                            f"the analysis reports at `{path or 'the top level'}` a mean FI's trials do not give: {off}; "
                            "a number printed beside its values must be computed from them"))
                    # a list of another recorded name's values (by its path, or the only one that holds them all) is
                    # judged setting by setting in core/run_manifest.py
                    if held_as(k[:-7], v, recorded, keys) != k[:-7]:
                        continue
                    extra = _not_among(v, recorded[k[:-7]])
                    if extra and not drawn_from(v, recorded[k[:-7]]):
                        out.append(_rm.fabricated(
                            f"the analysis reports `{here}` with {len(extra)} value(s) FI's trials of `{k[:-7]}` never "
                            f"produced (e.g. {extra[0]:g}): experiment.py must print the values FI_TRIALS holds, as they "
                            f"are; a list of something derived from them needs a name of its own"
                        ))
                else:
                    walk(v, here, keys + (k,))
        elif isinstance(node, list):
            for i, v in enumerate(node[:200]):
                walk(v, f"{path}[{i}]", keys)

    walk(result_json, "")
    return out
