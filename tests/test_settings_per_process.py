"""Several settings of a study share one process (core/trial_runner.py ``per_process``, ``groups``, ``measure_grouping``; the
decision in core/run_checks.py ``_group_settings`` is made from the measured time of the grouped run). Neutral made-up
studies only."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import run_checks, trial_runner
from core.execution import SharedInterpreterExecutor

MARK = '''
import os, uuid
with open(os.environ["FI_TEST_MARK"] + "." + uuid.uuid4().hex, "w") as _f:   # one file per process that loads the study
    _f.write("start")
'''


def _starts(mark: Path) -> int:
    """How many processes loaded the study: one mark file each (one shared file lost lines to concurrent appends)."""
    return len(list(mark.parent.glob(mark.name + ".*")))


def _clear(mark: Path) -> None:
    for old in mark.parent.glob(mark.name + ".*"):
        old.unlink()


def _virtual_time(monkeypatch: pytest.MonkeyPatch, cost=lambda cell: 0.0, per_process_s: float = 0.1) -> dict:
    """The runner's clock is a clock of the test's own: each process started moves it by a fixed ``per_process_s`` plus
    ``cost(cell)`` for every trial of every setting in it, whatever the machine is doing. The timings the sizing and the check
    decide on are then the same on every run."""
    clock = {"now": 0.0}
    monkeypatch.setattr(trial_runner, "time", SimpleNamespace(
        monotonic=lambda: clock["now"], time=time.time, time_ns=time.time_ns, perf_counter=time.perf_counter,
        sleep=time.sleep))
    started = SharedInterpreterExecutor.execute

    def spent(cmd, cwd) -> float:
        try:
            path = Path(cmd[2])
            spec = json.loads((path if path.is_absolute() else Path(cwd) / path).read_text(encoding="utf-8"))
            members = spec.get("members") or [spec]
            return sum(cost(m["cell"]) * len(m["trials"]) for m in members)
        except Exception:  # noqa: BLE001 -- a process whose spec cannot be read costs only its start
            return 0.0

    async def start(self, cmd, **kw):  # type: ignore[no-untyped-def]
        took = per_process_s + spent(cmd, kw.get("cwd"))
        clock["now"] += took
        if took > kw.get("timeout_s", float("inf")):  # the process would have been stopped for its time
            from core.execution import ExecutionResult

            return ExecutionResult(returncode=-9, stdout="", stderr="", duration_s=float(kw["timeout_s"]), timed_out=True)
        kw["timeout_s"] = 600  # the virtual time decided the stop above; the real process gets a limit no load reaches
        return await started(self, cmd, **kw)

    monkeypatch.setattr(SharedInterpreterExecutor, "execute", start)
    return clock

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
    _clear(mark)
    run = asyncio.run(trial_runner.run_trials(
        _executor(), sys.executable, root, "code/simulate.py", grid or GRID, runs_per_setting=runs, base_seed=7,
        deterministic=deterministic, timeout_s=120, env={"FI_TEST_MARK": str(mark)}, run_id="r1", per_process=per_process,
        cost=lambda cell: 5.0))
    starts = _starts(mark)
    return run, starts


def _rows(run) -> list[tuple]:
    return [(c.key, r["trial"], r.get("seed"), r["status"], json.dumps(r.get("values"), sort_keys=True))
            for c in run.cells for r in c.rows]


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


def test_a_group_cut_by_its_time_marks_no_setting_out_of_time_that_finishes_alone(tmp_path: Path) -> None:
    # a slow machine: the group is given 1 s, each of its 6 settings takes 0.5 s, so the group is cut mid-way; every setting
    # finishes alone, so none is out of time (an equal share of the group's time, 0.17 s, used to mark the running one)
    slow = MARK + '''
import time

def run_cell(cell):
    time.sleep(0.5)
    return {"level": cell["rate"] * 2.0}
'''
    root = _quest(tmp_path, slow)
    grid = {"rate": [1, 2, 3, 4, 5, 6], "size": [2]}
    keys = [trial_runner.cell_key(c) for c in trial_runner.cells(grid)]
    run = asyncio.run(trial_runner.run_trials(
        _executor(), sys.executable, root, "code/simulate.py", grid, runs_per_setting=1, base_seed=7, deterministic=True,
        timeout_s=60, env={"FI_TEST_MARK": str(tmp_path / "m.txt")}, run_id="r1", per_process=6, groups=[keys],
        group_allowance_s=1.0, at_once=2))
    assert [c.timed_out for c in run.cells] == [False] * 6
    assert run.ok_trials == 6 and run.failed_trials == 0


def test_a_slow_opposite_order_stops_the_check(tmp_path: Path) -> None:
    # the first setting is quick when it comes first in its process and slow when it comes later: the reverse run takes far
    # more than twice the forward run
    order = MARK + '''
import time
_calls = 0

def run_cell(cell):
    global _calls
    time.sleep(7.0 if cell["rate"] == 1 and _calls > 0 else 0.05)
    _calls += 1
    return {"level": cell["rate"] * 2.0}
'''
    root = _quest(tmp_path, order)
    grid = {"rate": [1, 2, 3, 4, 5, 6], "size": [2]}
    grouping = asyncio.run(trial_runner.measure_grouping(
        _executor(), sys.executable, root, "code/simulate.py", grid, runs_per_setting=1, base_seed=7, deterministic=True,
        timeout_s=120, per_process=3, cost=None, env={"FI_TEST_MARK": str(tmp_path / "m.txt")}, at_once=2))
    assert grouping.too_long and not grouping.same and "twice as long" in grouping.says


def test_the_check_times_what_the_run_does_so_the_run_is_not_slower_than_the_price(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch)
    # many quick settings: the plan, the groups and the rows are a large part of the time; the price covers them
    root = _quest(tmp_path, PURE)
    grid = {"rate": list(range(1, 61)), "size": [2, 3, 4, 5]}
    env = {"FI_TEST_MARK": str(tmp_path / "m.txt")}
    grouping = asyncio.run(trial_runner.measure_grouping(
        _executor(), sys.executable, root, "code/simulate.py", grid, runs_per_setting=1, base_seed=7, deterministic=True,
        timeout_s=120, per_process=15, cost=None, env=env, at_once=16))
    assert grouping.same and len(grouping.groups) == 16
    began = trial_runner.time.monotonic()
    run = asyncio.run(trial_runner.run_trials(
        _executor(), sys.executable, root, "code/simulate.py", grid, runs_per_setting=1, base_seed=7, deterministic=True,
        timeout_s=120, env=env, run_id="r1", per_process=15, groups=grouping.groups, group_allowance_s=60, at_once=16))
    took = trial_runner.time.monotonic() - began
    assert run.ok_trials == 240
    assert grouping.forward_s * 1.5 >= took, (grouping.forward_s, took)


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


def _sizing(tmp_path: Path, source: str, *, settings: int, at_once: int, limit_s: float, deterministic: bool = True,
            runs: int = 1, est_seconds: float = 9999.0):
    from types import SimpleNamespace

    from core import run_estimate

    root = _quest(tmp_path, source)
    me = SimpleNamespace(
        _log=_Log(), quest_root=root, executor=_executor(), _pilot_spent=0.0, _protocol_block=lambda state: {},
        config=SimpleNamespace(execution=SimpleNamespace(timeout_s=120)))
    runner = SimpleNamespace(simulate=root / "code" / "simulate.py", deterministic=deterministic, per_process=1,
                             shared_groups=None, group_allowance_s=None, cost=lambda cell: 2.0)
    # the estimate's own terms are made wrong on purpose (work_seconds 100x too high): nothing here may read them
    est = run_estimate.Estimate(seconds=est_seconds, at_least=False, complete=True, cells=settings, trials=settings,
                                at_once=at_once, slowdown=4.5, work_seconds=99999.0)
    got = asyncio.run(run_checks.RunChecksMixin._group_settings(
        me, None, runner, sys.executable, {"FI_TEST_MARK": str(tmp_path / "s.txt")},
        {"rate": list(range(1, settings + 1)), "size": [2]}, runs, {}, {}, est, limit_s, budget_s=120))
    return got, runner, me


def test_sizing_shares_processes_by_the_measured_time_of_the_grouped_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch)
    got, runner, me = _sizing(tmp_path, PURE, settings=12, at_once=4, limit_s=60.0)
    assert got is not None
    said, seconds, est = got
    assert runner.per_process == 3 and len(runner.shared_groups) == 4  # ceil(12 / 4) to a process, 4 processes at once
    assert sorted(k for g in runner.shared_groups for k in g) == sorted(
        trial_runner.cell_key(c) for c in trial_runner.cells({"rate": list(range(1, 13)), "size": [2]}))
    assert "several settings together" in said and "process" not in said
    assert 0 < seconds * 1.5 <= 60.0 and runner.group_allowance_s == seconds * 1.5 and est.seconds == seconds
    assert me._pilot_spent > 0
    assert runner.group_reuse is not None  # the check's first run is kept for the run


def test_the_settings_are_spread_over_the_slots_the_run_uses_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch)
    got, runner, _ = _sizing(tmp_path, PURE, settings=20, at_once=8, limit_s=60.0)
    assert got is not None and runner.per_process == 3 and len(runner.shared_groups) == 7  # ceil(20 / 8) = 3 to a process, 7 processes: no more than the 8 at once
    assert max(len(g) for g in runner.shared_groups) == 3
    # a run with a slot to every setting has nothing to share
    none, runner, _ = _sizing(tmp_path / "b", PURE, settings=4, at_once=8, limit_s=60.0)
    assert none is None and runner.per_process == 1 and runner.shared_groups is None


def test_the_time_is_scaled_to_the_runs_per_setting_and_padded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch, cost=lambda cell: 0.2)
    # a study with random draws and 30 runs per setting: the check runs 3 and the time is scaled ten times, so the
    # padded time passes the limit although the check itself was short
    got, runner, me = _sizing(tmp_path, STOCHASTIC, settings=12, at_once=4, limit_s=20.0, deterministic=False, runs=30)
    assert got is None and runner.per_process == 1 and runner.shared_groups is None
    assert any("over the limit" in line for line in me._log.lines)


def test_a_grouped_run_that_would_fit_once_padded_is_not_stopped_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch, cost=lambda cell: 16.0 if cell['rate'] == 6 else 0.05)  # not one of the sample's settings
    # the first run takes more than a quarter of the limit but less than the limit less the margin: it would fit, so the
    # check lets it finish and the settings share processes
    heavy = MARK + '''
def run_cell(cell):
    return {"level": cell["rate"] * 2.0}
'''
    got, runner, me = _sizing(tmp_path, heavy, settings=12, at_once=4, limit_s=60.0)
    assert got is not None and runner.per_process == 3, me._log.lines
    assert 15.0 < got[1] * 1.5 <= 60.0


def test_a_check_never_runs_longer_than_the_study_it_hopes_to_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch, cost=lambda cell: 6.0 if cell['rate'] == 12 else 0.05)
    # the check's trials are not reused, so its time is extra: it stops where the grouped run would take as long as the
    # study's own estimate without sharing (here 4 s), well inside the limit less the margin
    heavy = MARK + '''
def run_cell(cell):
    return {"level": cell["rate"] * 2.0}
'''
    got, runner, me = _sizing(tmp_path, heavy, settings=12, at_once=4, limit_s=60.0, est_seconds=4.0)
    assert got is None and runner.per_process == 1
    assert any("had not finished" in line for line in me._log.lines)


def test_processes_that_really_are_slow_are_not_shared_and_the_reason_is_stated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch, cost=lambda cell: 0.6)
    # the known positive: a group that cannot finish within the limit less the margin keeps one process per setting
    root = _quest(tmp_path, PURE)
    grid = {"rate": list(range(1, 9)), "size": [2]}
    grouping = asyncio.run(trial_runner.measure_grouping(
        _executor(), sys.executable, root, "code/simulate.py", grid, runs_per_setting=1, base_seed=7, deterministic=True,
        timeout_s=120, per_process=4, cost=None, max_pass_s=2.0, env={"FI_TEST_MARK": str(tmp_path / "m.txt")}, at_once=2))
    assert grouping.too_long and not grouping.same
    assert grouping.says.startswith("Settings run one per process") and "longer than the check allows" in grouping.says


def test_a_group_that_passes_the_limit_less_the_margin_stops_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch, cost=lambda cell: 6.0 if cell['rate'] == 12 else 0.05)
    # one setting is far costlier than the rest (a cost that grows along an axis): the group holding it passes the bound
    grows = MARK + '''
def run_cell(cell):
    return {"level": cell["rate"] * 2.0}
'''
    got, runner, me = _sizing(tmp_path, grows, settings=12, at_once=4, limit_s=8.0)
    assert got is None and runner.per_process == 1 and runner.shared_groups is None
    assert any("had not finished" in line for line in me._log.lines)
    assert not (runner.simulate.parents[1] / ".fi" / "trials" / "group_check").exists()


def test_sizing_does_not_share_a_study_that_keeps_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch)
    got, runner, me = _sizing(tmp_path, COUNTER, settings=12, at_once=4, limit_s=60.0)
    assert got is None and runner.per_process == 1 and runner.shared_groups is None
    assert any("differed" in line for line in me._log.lines)


def test_the_time_of_a_check_that_fails_is_still_counted(tmp_path: Path, monkeypatch) -> None:
    async def broken(*args, **kwargs):
        await asyncio.sleep(0.3)
        raise RuntimeError("the check could not run")

    monkeypatch.setattr(trial_runner, "measure_grouping", broken)
    got, runner, me = _sizing(tmp_path, PURE, settings=12, at_once=4, limit_s=60.0)
    assert got is None and runner.per_process == 1 and me._pilot_spent >= 0.25  # the timer's resolution: a 0.3 s sleep can read 0.297 s
    assert any("could not be run" in line for line in me._log.lines)


def test_a_resumed_run_shares_processes_again_from_the_kept_groups_without_a_cost(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch)  # the sizing's timing does not depend on the machine's load
    got, runner, _ = _sizing(tmp_path, PURE, settings=12, at_once=4, limit_s=60.0)
    assert got is not None
    kept = json.loads(json.dumps({"per_process": runner.per_process, "shared_groups": runner.shared_groups,
                                  "group_allowance_s": runner.group_allowance_s}))  # as run_estimate.json keeps them
    root = runner.simulate.parents[1]
    mark = tmp_path / "resume_marks.txt"
    grid = {"rate": list(range(1, 13)), "size": [2]}

    def go(**extra):
        _clear(mark)
        run = asyncio.run(trial_runner.run_trials(
            _executor(), sys.executable, root, "code/simulate.py", grid, runs_per_setting=1, base_seed=7, deterministic=True,
            timeout_s=120, env={"FI_TEST_MARK": str(mark)}, run_id="r1", at_once=4, cost=None, **extra))
        return run, _starts(mark)

    alone, alone_starts = go()
    resumed, starts = go(per_process=kept["per_process"], groups=kept["shared_groups"],
                         group_allowance_s=kept["group_allowance_s"])
    assert alone_starts == 12 and starts == 4
    assert _rows(alone) == _rows(resumed) and resumed.ok_trials == 12


def test_the_one_setting_worker_is_the_text_it_was_before_sharing_existed() -> None:
    import hashlib

    # sha256 of HARNESS_SOURCE (it is part of the keys of FI's search records); it was a0cea43c... on 894a9265. The text
    # includes core/in_run.py whole; commits 2a36c0db and 92494b76 only ADD helpers there (level_first, settle_missing,
    # per_run_names, unplaced; no line removed or changed), so the worker's own text is unchanged and the pin moves on purpose. It moves again for `LevelsRefused` (the type every refusal of a per-value result is raised as, which a case's retry and the repair rule read in place of the message), so the search records keyed on the old text are not matched.
    # It moves again for `shape_problem` (a number the plan declares one per run that a run returned per value: one rule for
    # FI and its harness) and for `per_run_names` reading what each run compares wherever the plan wrote it (`of_study`).
    # It moves again for `search_one_number_names` and `one_number_rule` (a search's objective and limits are one number
    # per call), added there and nothing else changed (ce17183f...).
    # It moves again for the recording of the files a run opened (the audit hook between the ``#<audit-begin>`` and
    # ``#<audit-end>`` lines of the body), and only for that: the text without those lines is still ce17183f...
    # (``LEGACY_HARNESS_SOURCE``, pinned in test_reuse_key.py), which a frozen study's search record of an older FI is
    # read with.
    assert hashlib.sha256(trial_runner.HARNESS_SOURCE.encode("utf-8")).hexdigest() == (
        "ae98b6e471a6a04c9338787d60dbf1f43004f2f1e1b2a4b874a5bb75414649ef")
    assert hashlib.sha256(trial_runner._without_audit(trial_runner.HARNESS_SOURCE).encode("utf-8")).hexdigest() == (
        "ce17183f5369dcf349c270de63a8c05a82efa53a9f0114cdf9c90703d90a4bb9")
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


# --- the check's first run stands for the real run (core/trial_runner.py ``GroupedRun``) -------------------------------------


class _Lines:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg, *a):  # type: ignore[no-untyped-def]
        self.lines.append(msg % a if a else msg)

    warning = info


def _check_and_run(root: Path, tmp_path: Path, *, deterministic: bool, base_seed: int = 7, edit=None, runs: int = 2):
    """The check, then the real run given the check's first run; returns (grouping, run, log, processes started after the check)."""
    mark = tmp_path / "marks.txt"
    _clear(mark)
    env = {"FI_TEST_MARK": str(mark)}
    grouping = asyncio.run(trial_runner.measure_grouping(
        _executor(), sys.executable, root, "code/simulate.py", GRID, runs_per_setting=runs, base_seed=7,
        deterministic=deterministic, timeout_s=120, per_process=3, cost=None, env=env, at_once=4, sample=2))
    during_check = _starts(mark)
    if edit is not None:
        edit(root)
    log = _Lines()
    run = asyncio.run(trial_runner.run_trials(
        _executor(), sys.executable, root, "code/simulate.py", GRID, runs_per_setting=runs, base_seed=base_seed,
        deterministic=deterministic, timeout_s=120, env=env, run_id="r1", per_process=3, groups=grouping.groups,
        at_once=4, reuse=grouping.reuse, log=log))
    return grouping, run, log, _starts(mark) - during_check


@pytest.mark.parametrize("source,deterministic", [(PURE, True), (STOCHASTIC, False)])
def test_the_first_run_of_the_check_is_the_runs_results_and_the_record_is_the_same(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, deterministic: bool) -> None:
    _virtual_time(monkeypatch)
    root = _quest(tmp_path, source)
    grouping, run, log, started = _check_and_run(root, tmp_path, deterministic=deterministic)
    assert grouping.same and grouping.reuse is not None
    assert started == 0 and any("is not run again" in line for line in log.lines)
    plain, _ = _run(root, tmp_path, per_process=1, deterministic=deterministic, runs=2)
    assert _rows(run) == _rows(plain)  # values, seeds, statuses: those of a run of the same study
    reused_ledger = [json.loads(x) for x in run.ledger_path.read_text(encoding="utf-8").splitlines()]
    plain_ledger = [json.loads(x) for x in plain.ledger_path.read_text(encoding="utf-8").splitlines()]
    key = lambda ds: [(d["event"], d["cell"], d.get("trial"), d.get("seed"), d.get("values_sha256")) for d in ds]  # noqa: E731
    assert key(reused_ledger) == key(plain_ledger)
    assert {d["run_id"] for d in reused_ledger} == {"r1"}
    assert run.per_process == 3 and run.wall_s is not None and run.wall_s > 0
    assert not (root / ".fi" / "trials" / "group_check").exists()


def test_a_changed_code_or_other_seeds_fall_back_to_running_the_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch)
    root = _quest(tmp_path, PURE)

    def edit(r: Path) -> None:
        (r / "code" / "simulate.py").write_text(PURE + "\n# edited\n", encoding="utf-8")

    grouping, run, log, started = _check_and_run(root, tmp_path, deterministic=True, edit=edit)
    assert grouping.same and started == 2 and run.ok_trials == 6 and run.wall_s is None
    assert any("not used as the run (the code changed since it ran)" in line for line in log.lines)
    root = _quest(tmp_path / "b", STOCHASTIC)
    grouping, run, log, started = _check_and_run(root, tmp_path, deterministic=False, base_seed=8)
    assert started == 2 and run.wall_s is None and any("base seed" in line for line in log.lines)
    plain, _ = _run(root, tmp_path, per_process=1, deterministic=False, runs=2)
    assert [r["seed"] for c in run.cells for r in c.rows] != [r["seed"] for c in plain.cells for r in c.rows]


def test_more_runs_than_the_check_ran_fall_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch)
    root = _quest(tmp_path, STOCHASTIC)
    grouping, run, log, started = _check_and_run(root, tmp_path, deterministic=False, runs=5)
    assert grouping.same and started == 2 and run.wall_s is None
    assert any("it ran 3 trial(s) of each setting and the real run runs 5" in line for line in log.lines)


def test_a_check_that_did_not_finish_or_differed_keeps_nothing_and_the_run_is_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _virtual_time(monkeypatch)
    for name, source in (("counter", COUNTER), ("crash", CRASHES)):
        root = _quest(tmp_path / name, source)
        mark = tmp_path / "m.txt"
        got = asyncio.run(trial_runner.measure_grouping(
            _executor(), sys.executable, root, "code/simulate.py", GRID, runs_per_setting=1, base_seed=7,
            deterministic=True, timeout_s=120, per_process=3, cost=None, env={"FI_TEST_MARK": str(mark)}, at_once=2))
        assert not got.same and got.reuse is None
        left = [p.name for p in (root / ".fi" / "trials").glob("*") if p.name.startswith(("cell", "group"))]
        assert left == [], left  # none of the first run's files stay
