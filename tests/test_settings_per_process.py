"""Several settings of a study share one process (core/trial_runner.py ``per_process``, ``measure_grouping``; the size is
computed in core/cost_model.py). Neutral made-up studies only."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

from core import cost_model, run_checks, trial_runner
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
        deterministic=deterministic, timeout_s=120, env={"FI_TEST_MARK": str(mark)}, run_id="r1", per_process=per_process,
        cost=lambda cell: 5.0))
    starts = len(mark.read_text().splitlines()) if mark.exists() else 0
    return run, starts


def _rows(run) -> list[tuple]:
    return [(c.key, r["trial"], r.get("seed"), r["status"], json.dumps(r.get("values"), sort_keys=True))
            for c in run.cells for r in c.rows]


def test_group_size_comes_from_the_estimate_less_the_starts() -> None:
    # 680 settings of 2.0 s of which 1.95 s is the start: sharing saves almost all of it
    work = 680 * 2.0
    many = cost_model.settings_per_process(work, 680, 1.95, 16, 3.0)
    assert many > 1
    alone = cost_model.grouped_wall(work, 680, 680, 1.95, 16, 3.0)
    assert cost_model.grouped_wall(work, 680, math.ceil(680 / many), 1.95, 16, 3.0) < 0.8 * alone
    # the same settings when the settings themselves take the time: nothing to save, so 1
    assert cost_model.settings_per_process(680 * 60.0, 680, 1.95, 16, 3.0) == 1
    assert cost_model.settings_per_process(24.0, 1, 1.95, 1) == 1
    assert cost_model.settings_per_process(22 * 30.0, 22, 2.0, 8, 2.4) == 1


def test_cost_that_grows_along_an_axis_is_not_averaged() -> None:
    # twelve settings: eleven cost 0.5 s, one costs 100 s; the estimate knows it (a median of three settings would not)
    work = 11 * 0.5 + 100.0
    assert cost_model.grouped_wall(work, 12, 3, 0.2, 4) > 100.0 / 4  # the costly setting is still paid in full
    assert cost_model.settings_per_process(work, 12, 0.2, 4) == 1


def test_fewer_processes_than_at_once_are_not_priced_at_the_full_slowdown() -> None:
    one = cost_model.grouped_wall(100.0, 50, 1, 1.0, 16, 4.5)
    assert math.isclose(one, 100.0 - 49 * 1.0)  # one process runs alone: no slowdown
    full = cost_model.grouped_wall(160.0, 16, 16, 1.0, 16, 3.0)
    assert math.isclose(full, 160.0 / 16 * 3.0)  # all sixteen at once: the full slowdown


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


def _measure(root: Path, tmp_path: Path, grid=None, per_process: int = 3):
    return asyncio.run(trial_runner.measure_grouping(
        _executor(), sys.executable, root, "code/simulate.py", grid or GRID, runs_per_setting=1, base_seed=7,
        deterministic=True, timeout_s=120, per_process=per_process, cost=lambda cell: 5.0,
        env={"FI_TEST_MARK": str(tmp_path / "check_marks.txt")}))


def test_a_pure_study_gives_the_same_numbers_on_every_setting(tmp_path: Path) -> None:
    root = _quest(tmp_path, PURE)
    got = _measure(root, tmp_path, {"rate": list(range(1, 41)), "size": [2]}, per_process=10)
    assert got.same and got.differences == [] and got.settings == 40 and got.alone == 6
    assert not (root / ".fi" / "trials" / "group_check").exists()  # nothing of the check stays in the quest


def test_state_kept_between_settings_fails_the_check_and_stays_one_per_process(tmp_path: Path) -> None:
    root = _quest(tmp_path, COUNTER)
    got = _measure(root, tmp_path)
    assert not got.same
    assert got.differences and "level" in got.differences[0]
    assert "one per process" in got.says and "differed" in got.says
    # the planted study really does change its numbers when settings share a process
    alone, _ = _run(root, tmp_path, per_process=1)
    shared, _ = _run(root, tmp_path, per_process=3)
    assert _rows(alone) != _rows(shared)


# a memo keyed on part of the setting: settings with different rates agree, later ones with a rate seen before do not
MEMO = MARK + '''
_memo = {}

def run_cell(cell):
    if cell["rate"] not in _memo:
        _memo[cell["rate"]] = cell["rate"] * 10.0 + cell["size"]
    return {"level": _memo[cell["rate"]]}
'''


def test_state_that_shows_only_between_later_settings_is_found(tmp_path: Path) -> None:
    root = _quest(tmp_path, MEMO)
    grid = {"rate": list(range(1, 11)), "size": [1, 2, 3]}
    got = _measure(root, tmp_path, grid, per_process=30)  # a check on three spread settings shares no rate and sees nothing
    assert not got.same and got.differences
    alone, _ = _run(root, tmp_path, per_process=1, grid=grid)
    shared, _ = _run(root, tmp_path, per_process=30, grid=grid)
    assert _rows(alone) != _rows(shared)


# a number changes from the fourth call on in one process
FLAG = MARK + '''
_seen = []

def run_cell(cell):
    _seen.append(cell["rate"])
    return {"level": cell["rate"] * 1.0 + (0.5 if len(_seen) > 3 else 0.0)}
'''


def test_state_that_appears_after_the_third_setting_is_found(tmp_path: Path) -> None:
    root = _quest(tmp_path, FLAG)
    got = _measure(root, tmp_path, {"rate": list(range(1, 9)), "size": [2]}, per_process=8)
    assert not got.same


def test_a_hung_setting_costs_only_itself(tmp_path: Path) -> None:
    hang = MARK + '''
import time

def run_cell(cell):
    if cell["rate"] == 2:
        time.sleep(300)
    return {"level": cell["rate"] * 2.0}
'''
    root = _quest(tmp_path, hang)
    mark = tmp_path / "m.txt"

    def go(per_process: int):
        return asyncio.run(trial_runner.run_trials(
            _executor(), sys.executable, root, "code/simulate.py", {"rate": [1, 2, 3, 4, 5, 6], "size": [2]},
            runs_per_setting=1, base_seed=7, deterministic=True, timeout_s=30, env={"FI_TEST_MARK": str(mark)},
            run_id="r1", per_process=per_process, at_once=6, cost=lambda cell: 1.0))

    alone, shared = go(1), go(3)

    def failed(run) -> list[str]:
        return sorted(c.key for c in run.cells for r in c.rows if r["status"] != "ok")

    assert failed(alone) == ["rate=2,size=2"] == failed(shared)
    # a setting that finished is never marked out of time
    assert [c.timed_out for c in shared.cells if c.key != "rate=2,size=2"] == [False] * 5
    assert _rows(alone) == _rows(shared)


def test_each_setting_keeps_its_own_warnings(tmp_path: Path) -> None:
    warn = MARK + '''
import warnings

def run_cell(cell):
    warnings.warn("careful %s" % cell["rate"], RuntimeWarning)
    return {"level": cell["rate"] * 2.0}
'''
    root = _quest(tmp_path, warn)
    shared, _ = _run(root, tmp_path, per_process=6)
    for c in shared.cells:
        assert f"careful {c.cell['rate']}" in c.stderr and c.stderr.count("careful") == 1 and "[fi-setting]" not in c.stderr


# ---- the over-limit step in the sizing (core/run_checks.py) -------------------------------------------------------------


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg, *args) -> None:
        self.lines.append(msg % args if args else msg)

    warning = info


def _sizing(tmp_path: Path, source: str, monkeypatch, *, work_s: float, settings: int, fixed_s: float, limit_s: float):
    from types import SimpleNamespace

    from core import run_estimate

    root = _quest(tmp_path, source)
    started = {"n": 0}

    async def fake_start(*args, **kwargs):
        started["n"] += 1
        return fixed_s

    monkeypatch.setattr(trial_runner, "measure_start", fake_start)
    me = SimpleNamespace(
        _log=_Log(), quest_root=root, executor=_executor(), _pilot_spent=0.0, _protocol_block=lambda state: {},
        config=SimpleNamespace(execution=SimpleNamespace(timeout_s=120)))
    runner = SimpleNamespace(simulate=root / "code" / "simulate.py", deterministic=True, per_process=1, cost=lambda cell: 2.0)
    est = run_estimate.Estimate(seconds=work_s / 4, at_least=False, complete=True, cells=settings, trials=settings,
                                at_once=4, slowdown=1.0, work_seconds=work_s)
    got = asyncio.run(run_checks.RunChecksMixin._group_settings(
        me, None, runner, sys.executable, {"FI_TEST_MARK": str(tmp_path / "s.txt")},
        {"rate": list(range(1, settings + 1)), "size": [2]}, 1, {}, {}, est, limit_s, budget_s=120))
    return got, runner, me


def test_sizing_shares_processes_when_starts_are_the_time(tmp_path: Path, monkeypatch) -> None:
    got, runner, me = _sizing(tmp_path, PURE, monkeypatch, work_s=12 * 2.0, settings=12, fixed_s=1.9, limit_s=10.0)
    assert got is not None and runner.per_process > 1
    said, seconds, est = got
    assert "several settings together" in said and "process" not in said and seconds * 1.5 <= 10.0
    assert est.seconds == seconds and me._pilot_spent > 0


def test_sizing_leaves_a_study_whose_cost_grows_to_the_cut(tmp_path: Path, monkeypatch) -> None:
    # the estimate knows one setting costs 100 s: starts are a small part, so the study is still over and nothing is shared
    got, runner, me = _sizing(tmp_path, PURE, monkeypatch, work_s=11 * 0.5 + 100.0, settings=12, fixed_s=0.2, limit_s=10.0)
    assert got is None and runner.per_process == 1
    assert any("one per process" in line for line in me._log.lines)


def test_sizing_does_not_share_a_study_that_keeps_state(tmp_path: Path, monkeypatch) -> None:
    got, runner, me = _sizing(tmp_path, COUNTER, monkeypatch, work_s=12 * 2.0, settings=12, fixed_s=1.9, limit_s=10.0)
    assert got is None and runner.per_process == 1
    assert any("differed" in line for line in me._log.lines)


def test_the_one_setting_worker_is_the_text_it_was_before_sharing_existed() -> None:
    import hashlib

    # sha256 of HARNESS_SOURCE on 894a9265 (it is part of the keys of FI's search records)
    assert hashlib.sha256(trial_runner.HARNESS_SOURCE.encode("utf-8")).hexdigest() == (
        "a0cea43ced006871470b56c4f502ad392df7697413175b224db8b880c748dfc1")
    assert trial_runner.GROUP_HARNESS_SOURCE != trial_runner.HARNESS_SOURCE and "members" in trial_runner.GROUP_HARNESS_SOURCE


def test_a_group_is_given_the_time_of_its_settings_not_half_of_what_is_left(tmp_path: Path) -> None:
    slow = MARK + '''
import time

def run_cell(cell):
    time.sleep(2.0)
    return {"level": cell["rate"] * 2.0}
'''
    root = _quest(tmp_path, slow)
    run = asyncio.run(trial_runner.run_trials(
        _executor(), sys.executable, root, "code/simulate.py", {"rate": list(range(1, 9)), "size": [2]},
        runs_per_setting=1, base_seed=7, deterministic=True, timeout_s=14, env={"FI_TEST_MARK": str(tmp_path / "m.txt")},
        run_id="r1", per_process=4, at_once=2, cost=lambda cell: 2.0))
    assert run.ok_trials == 8 and run.failed_trials == 0


def test_what_a_setting_prints_without_a_newline_is_not_joined_to_the_marker(tmp_path: Path) -> None:
    chatty = MARK + '''
import sys

def run_cell(cell):
    sys.stderr.write("working on %s" % cell["rate"])
    return {"level": cell["rate"] * 2.0}
'''
    root = _quest(tmp_path, chatty)
    shared, _ = _run(root, tmp_path, per_process=4)
    for c in shared.cells:
        assert c.stderr.strip() == f"working on {c.cell['rate']}"
    assert "[fi-setting]" not in shared.stderr()


def test_a_study_of_one_second_settings_is_refused_before_any_shared_run(tmp_path: Path, monkeypatch) -> None:
    # twenty settings of one second each, a process starts in 0.6 s: the estimate is 26 s, the limit 10 s. The time is
    # the settings', not their starts, so sharing cannot fit it and no shared run is made to find out.
    called = []

    async def never(*args, **kwargs):
        called.append(1)
        raise AssertionError("a shared run was made")

    monkeypatch.setattr(trial_runner, "measure_grouping", never)
    got, runner, me = _sizing(tmp_path, PURE, monkeypatch, work_s=26.0, settings=20, fixed_s=0.6, limit_s=10.0)
    assert got is None and runner.per_process == 1 and not called
    assert any("settings themselves take" in line for line in me._log.lines)
