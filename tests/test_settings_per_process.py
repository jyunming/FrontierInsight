"""Several settings of a study share one process (core/trial_runner.py ``per_process``, ``measure_grouping``; the size is
computed in core/cost_model.py). Neutral made-up studies only."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

from core import cost_model, trial_runner
from core.execution import SharedInterpreterExecutor

MARK = '''
import os
with open(os.environ["FI_TEST_MARK"], "a") as _f:
    _f.write("start\\n")                       # one line per process that loads the study
'''

PURE = MARK + '''
def run_cell(cell):
    return {"level": cell["rate"] * 3.0 + cell["size"], "ratio": cell["rate"] / cell["size"]}
'''

STOCHASTIC = MARK + '''
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    return {"draw": cell["rate"] + rng.random()}
'''

# keeps a count between calls: a setting's number depends on how many ran before it in the same process
COUNTER = MARK + '''
_calls = 0

def run_cell(cell):
    global _calls
    _calls += 1
    return {"level": cell["rate"] + _calls}
'''

# one setting kills its process
CRASHES = MARK + '''
import os

def run_cell(cell):
    if cell["rate"] == 3:
        os._exit(9)
    return {"level": cell["rate"] * 2.0}
'''

GRID = {"rate": [1, 2, 3, 4, 5, 6], "size": [2]}


def _quest(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return root


def _executor() -> SharedInterpreterExecutor:
    return SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")


def _run(root: Path, tmp_path: Path, *, per_process: int, deterministic: bool = True, runs: int = 3, grid=None):
    mark = tmp_path / f"marks{per_process}.txt"
    mark.unlink(missing_ok=True)
    run = asyncio.run(trial_runner.run_trials(
        _executor(), sys.executable, root, "code/simulate.py", grid or GRID, runs_per_setting=runs, base_seed=7,
        deterministic=deterministic, timeout_s=120, env={"FI_TEST_MARK": str(mark)}, run_id="r1", per_process=per_process))
    starts = len(mark.read_text().splitlines()) if mark.exists() else 0
    return run, starts


def _rows(run) -> list[tuple]:
    return [(c.key, r["trial"], r.get("seed"), r["status"], json.dumps(r.get("values"), sort_keys=True))
            for c in run.cells for r in c.rows]


def test_group_size_follows_start_and_trial_time() -> None:
    # a process costs 2 s, a setting 0.01 s: share; a setting of 30 s: do not
    many = cost_model.settings_per_process(680, 0.01, 2.0, 16, 3.0)
    assert many > 1
    assert cost_model.settings_per_process(22, 30.0, 2.0, 8, 2.4) == 1
    assert cost_model.settings_per_process(1, 0.01, 2.0, 1) == 1
    # sharing is used only when it saves enough: a small start against slow settings stays at 1
    assert cost_model.settings_per_process(100, 1.0, 0.05, 1) == 1
    one = cost_model.grouped_seconds(680, 0.01, 2.0, 16, 1, 3.0)
    assert cost_model.grouped_seconds(680, 0.01, 2.0, 16, many, 3.0) < 0.8 * one


def test_start_and_trial_from_a_sample() -> None:
    start, trial = cost_model.start_and_trial([2.5, 2.2, 2.4], [0.3, 0.31, 0.29], 1)
    assert math.isclose(trial, 0.3) and math.isclose(start, 1.9)


def test_fewer_processes_than_at_once_are_not_priced_at_the_full_slowdown() -> None:
    # one process runs alone: no slowdown from others beside it
    assert math.isclose(cost_model.grouped_seconds(55, 0.22, 1.8, 16, 55, 4.5), 1.8 + 55 * 0.22)
    # the full number of processes at once: the full slowdown
    assert math.isclose(cost_model.grouped_seconds(16, 1.0, 2.0, 16, 1, 3.0), 3.0 * 3.0)


def test_shared_processes_give_the_very_same_rows_with_fewer_processes(tmp_path: Path) -> None:
    root = _quest(tmp_path, PURE)
    alone, alone_starts = _run(root, tmp_path, per_process=1)
    shared, shared_starts = _run(root, tmp_path, per_process=3)
    assert alone_starts == 6 and shared_starts == 2
    assert _rows(alone) == _rows(shared)
    assert shared.ok_trials == 6 and shared.failed_trials == 0
    ledger = [json.loads(line) for line in shared.ledger_path.read_text(encoding="utf-8").splitlines()]
    assert sorted({d["cell"] for d in ledger}) == sorted(trial_runner.cell_key(c) for c in trial_runner.cells(GRID))


def test_shared_processes_keep_the_seeds_of_a_stochastic_study(tmp_path: Path) -> None:
    root = _quest(tmp_path, STOCHASTIC)
    alone, _ = _run(root, tmp_path, per_process=1, deterministic=False)
    shared, starts = _run(root, tmp_path, per_process=4, deterministic=False)
    assert starts == 2
    assert _rows(alone) == _rows(shared)
    assert all(r["seed"] is not None for c in shared.cells for r in c.rows)


def test_a_setting_that_kills_its_process_costs_only_itself(tmp_path: Path) -> None:
    root = _quest(tmp_path, CRASHES)
    alone, _ = _run(root, tmp_path, per_process=1)
    shared, _ = _run(root, tmp_path, per_process=6)
    assert [(c.key, [r["status"] for r in c.rows]) for c in alone.cells] == [(c.key, [r["status"] for r in c.rows]) for c in shared.cells]
    assert shared.ok_trials == 5 and shared.failed_trials == 1
    assert [json.dumps(r.get("values")) for c in alone.cells for r in c.rows] == [json.dumps(r.get("values")) for c in shared.cells for r in c.rows]


def _measure(root: Path, tmp_path: Path, grid=None):
    return asyncio.run(trial_runner.measure_grouping(
        _executor(), sys.executable, root, "code/simulate.py", grid or GRID, runs_per_setting=1, base_seed=7,
        deterministic=True, timeout_s=120, env={"FI_TEST_MARK": str(tmp_path / "check_marks.txt")}))


def test_the_sample_shows_a_pure_study_gives_the_same_numbers(tmp_path: Path) -> None:
    root = _quest(tmp_path, PURE)
    grid = {"rate": list(range(1, 41)), "size": [2]}
    got = _measure(root, tmp_path, grid)
    assert got.same and got.differences == [] and got.start_s is not None
    assert got.per_process >= 1
    assert not (root / ".fi" / "trials" / "group_check").exists()  # nothing of the sample stays in the quest


def test_state_kept_between_settings_fails_the_sample_and_stays_one_per_process(tmp_path: Path) -> None:
    root = _quest(tmp_path, COUNTER)
    got = _measure(root, tmp_path)
    assert not got.same and got.per_process == 1
    assert got.differences and "level" in got.differences[0]
    assert "one per process" in got.says and "different numbers" in got.says
    # the planted study really does change its numbers when settings share a process
    alone, _ = _run(root, tmp_path, per_process=1)
    shared, _ = _run(root, tmp_path, per_process=3)
    assert _rows(alone) != _rows(shared)


def test_slow_trials_keep_one_setting_per_process(tmp_path: Path) -> None:
    slow = MARK + '''
import time

def run_cell(cell):
    time.sleep(3.0)
    return {"level": cell["rate"] * 1.0}
'''
    root = _quest(tmp_path, slow)
    got = _measure(root, tmp_path, {"rate": [1, 2, 3], "size": [2]})
    assert got.same and got.per_process == 1 and "one per process" in got.says
