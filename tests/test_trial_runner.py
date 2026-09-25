"""FI runs the trials and writes their record (core/trial_runner.py): the experiment's code only says what a trial does."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from core import run_manifest, trial_runner
from core.execution import SharedInterpreterExecutor

SIM = '''
import random, warnings, sys

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    print("noise the simulation prints")               # must never pass for a result
    if cell["R0"] == 2.0 and trial_id == 1:
        raise ValueError("diverged")
    if trial_id == 2:
        warnings.warn("step size too large", RuntimeWarning)
    return {"infected": cell["R0"] * 100 + rng.random(), "peak_day": 10.0 + trial_id}
'''

CRASH = '''
import os

def run_trial(cell, trial_id, seed):
    if trial_id == 1:
        os._exit(7)                                       # the process dies part-way through the cell
    return {"x": 1.0}
'''

DET = '''
def run_cell(cell):
    return {"energy_drift": cell["dt"] * 2}
'''


def _run(tmp_path: Path, source: str, grid: dict, *, runs: int = 3, deterministic: bool = False, timeout: int = 60):
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return root, asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"),
        sys.executable, root, "code/simulate.py", grid, runs_per_setting=runs, base_seed=0,
        deterministic=deterministic, timeout_s=timeout, run_id="r1",
    ))


def test_cells_and_seeds_are_fixed_by_the_protocol() -> None:
    grid = {"R0": [1.5, 2.0], "N": [100]}
    assert trial_runner.cells(grid) == [{"R0": 1.5, "N": 100}, {"R0": 2.0, "N": 100}]
    assert trial_runner.cell_key({"R0": 1.5, "N": 100}) == "R0=1.5,N=100"
    a = trial_runner.trial_seed(0, "R0=1.5,N=100", 0)
    assert a == trial_runner.trial_seed(0, "R0=1.5,N=100", 0)
    assert len({trial_runner.trial_seed(0, k, t) for k in ("R0=1.5,N=100", "R0=2.0,N=100") for t in range(50)}) == 100


def test_fi_writes_the_ledger_and_the_per_cell_summary(tmp_path: Path) -> None:
    root, run = _run(tmp_path, SIM, {"R0": [1.5, 2.0]})
    assert run.ok_trials == 5 and run.failed_trials == 1
    rows = trial_runner.read_ledger(root)
    assert len(rows) == 6 and {r["status"] for r in rows} == {"ok", "failed"}
    failed = next(r for r in rows if r["status"] == "failed")
    assert failed["cell"] == "R0=2.0" and "diverged" in failed["reason"]
    # The existing checker reads FI's rows, and finds the grid and counts complete.
    manifest, row_problems = run_manifest.manifest_from_ledger({"grid": {"R0": [1.5, 2.0]}, "runs_per_setting": 3}, rows, {})
    assert not row_problems and len(manifest["failed_trials"]) == 1
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    first = summary["cells"][0]
    assert first["key"] == "R0=1.5" and first["ok"] == 3 and first["metrics"]["infected"]["count"] == 3
    assert 150 < first["metrics"]["infected"]["total"] / 3 < 151
    assert "noise the simulation prints" in run.stderr() and "step size too large" not in json.dumps(summary)
    events = [json.loads(line) for line in run.ledger_path.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events][:1] == ["planned"] and events[-1]["warnings"] in (0, 1)


def test_trials_a_crashed_process_never_reported_are_failed_with_why(tmp_path: Path) -> None:
    root, run = _run(tmp_path, CRASH, {"a": [1]})
    statuses = [r.get("status") for r in run.cells[0].rows]
    assert statuses == ["ok", "failed", "failed"]
    assert "exit code 7" in run.cells[0].rows[1]["reason"]


def test_a_simulation_without_the_function_fails_every_trial_and_says_so(tmp_path: Path) -> None:
    root, run = _run(tmp_path, "x = 1\n", {"a": [1]}, runs=2)
    assert run.failed_trials == 2 and "has no function run_trial()" in run.cells[0].rows[0]["reason"]


def test_a_deterministic_study_runs_each_cell_once(tmp_path: Path) -> None:
    root, run = _run(tmp_path, DET, {"dt": [0.1, 0.01]}, deterministic=True)
    assert [len(c.rows) for c in run.cells] == [1, 1] and run.ok_trials == 2
    assert run.cells[1].rows[0]["values"]["energy_drift"] == 0.02


def test_the_harness_is_written_fresh_so_an_edited_one_never_runs(tmp_path: Path) -> None:
    root, _ = _run(tmp_path, SIM, {"R0": [1.5]})
    harness = root / trial_runner.HARNESS_PATH
    harness.write_text("print('tampered')", encoding="utf-8")
    asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", {"R0": [1.5]},
        runs_per_setting=1, base_seed=0, deterministic=False, timeout_s=60,
    ))
    assert "tampered" not in harness.read_text(encoding="utf-8")
