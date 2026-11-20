"""FI runs the simulation on a small case before the code is read (core/small_run.py, RunChecksMixin._small_run_gate): a
crash goes to the repair of a failed run with its place and traceback, the reader is shown what the run gave, and a run of
the same code is not made again. Live on the code r12 delivered (tests/fixtures/replay/r12_code/): one call at the noisy
corner shows the solver failure at ``model.py`` line 121 in seconds, where the quest spent its review rounds on tests that
only showed this crash."""

from __future__ import annotations

import asyncio
import shutil
import string
import sys
from pathlib import Path
from typing import Any

import pytest

from core import optimise_search as search
from core import small_run
from core.execution import SharedInterpreterExecutor

R12 = Path(__file__).parent / "fixtures" / "replay" / "r12_code"
GRID = {"noise_std_rel": [0.005, 0.01, 0.02, 0.04, 0.08], "n_timepoints": [5, 8, 12, 20, 32]}


def test_the_settings_are_the_middle_and_the_two_corners() -> None:
    assert small_run.settings(GRID) == [{"noise_std_rel": 0.02, "n_timepoints": 12},
                                        {"noise_std_rel": 0.005, "n_timepoints": 5},
                                        {"noise_std_rel": 0.08, "n_timepoints": 32}]
    assert small_run.settings({"method": ["a", "b"]}) == [{"method": "a"}, {"method": "b"}]
    assert small_run.settings({}) == [{}]
    assert small_run.budget(3600) == 180 and small_run.budget(100) == 30 and small_run.budget(None) == 30


def _quest(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    shutil.copytree(R12, root / "code")
    return root


def test_replay_r12_the_small_run_shows_the_crash_and_its_place(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    root = _quest(tmp_path)
    got = asyncio.run(small_run.run(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root,
                                    "code/simulate.py", GRID, deterministic=False, timeout_s=120, paired=True))
    assert got.crashed and [s.key for s in got.faults] == ["noise_std_rel=0.08,n_timepoints=32"]
    assert "model.py line 121" in got.faults[0].reasons[0][0]
    message = got.directive()
    assert "did not satisfy required optimality" in message and "Traceback" in message and "line 121" in message
    shown = got.for_reader()
    assert "Setting noise_std_rel=0.02,n_timepoints=12: 3 of 3 trial(s) ran" in shown and "weighted_Ea_hat=" in shown
    assert not (root / "raw").exists(), "the study's raw/ is never touched"


class _Counting(SharedInterpreterExecutor):
    calls = 0

    async def execute(self, cmd, **kw):  # noqa: ANN001, ANN003
        type(self).calls += 1
        return await super().execute(cmd, **kw)


@pytest.mark.usefixtures("this_python")
def test_the_gate_sends_a_crash_to_repair_and_reads_back_an_unchanged_run(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    from tests.test_disclosure import _engine

    root = _quest(tmp_path)
    eng = _engine(root)
    eng.executor = _Counting(python_version="3.11")  # type: ignore[assignment]
    eng._trial_mode, eng._trial_entries = True, {"run_trial"}  # type: ignore[attr-defined]
    eng._protocol_block = lambda state: {"grid": GRID, "metrics": [{"id": "weighted_Ea_hat", "paired": True}]}  # type: ignore[assignment]
    state: dict[str, Any] = {"exec_reflect_iter": 0}
    sent = asyncio.run(eng._small_run_gate(state, sys.executable, {}))
    assert sent is not None and sent["exec_result"]["failed_script"] == "simulate.py"
    assert "model.py line 121" in sent["exec_result"]["stderr_tail"]
    ran = _Counting.calls
    assert asyncio.run(eng._small_run_gate(state, sys.executable, {})) is not None
    assert _Counting.calls == ran, "the same code is not run again"
    # no repair left: the study's own run decides
    assert asyncio.run(eng._small_run_gate({"exec_reflect_iter": 99}, sys.executable, {})) is None
    # the crash repaired (the solver's tolerance relaxed): the small run passes and the reader is shown what it gave
    model = root / "code" / "kinetics_parameter_recovery" / "model.py"
    model.write_text(model.read_text(encoding="utf-8").replace("abs(float(gradient[0])) >= solver_gtol",
                                                               "abs(float(gradient[0])) >= 1e-6"), encoding="utf-8")
    assert asyncio.run(eng._small_run_gate(state, sys.executable, {})) is None
    assert "Setting noise_std_rel=0.08,n_timepoints=32: 3 of 3 trial(s) ran" in eng._small_run_note  # type: ignore[attr-defined]
    assert asyncio.run(eng._small_run_again()) == ""


def test_the_reader_is_shown_the_small_run() -> None:
    text = (Path(__file__).parent.parent / "agents" / "code_review.md").read_text(encoding="utf-8")
    filled = string.Template(text).substitute(topic="t", requirements="[]", plan="{}", open_items="", code="c", previous="",
                                              small_run="Setting a=1: 3 of 3 trial(s) ran")
    assert "What the code gave when FI ran it" in filled and "Setting a=1: 3 of 3" in filled


class _Hangs:
    """An executor whose every process runs out of the time it is given; it records that time."""

    def __init__(self) -> None:
        self.given: list[int] = []

    async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
        from core.execution import ExecutionResult

        self.given.append(int(timeout_s))
        return ExecutionResult(returncode=-1, stdout="", stderr="", duration_s=float(timeout_s), timed_out=True)


def test_a_study_of_parts_runs_a_setting_of_every_part_each_with_its_share_of_the_time(tmp_path: Path) -> None:
    grid = search.study_grid({"grid": {"kind": ["a", "b"], "n": [1, 2, 3]},
                              "sub_studies": [{"name": "as", "vary": ["n"], "hold": {"kind": "a"}},
                                              {"name": "bs", "vary": ["n"], "hold": {"kind": "b"}}]})
    assert small_run.settings(grid) == [{"kind": "a", "n": 2}, {"kind": "b", "n": 2}]  # not the corners of the whole grid
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("def run_cell(cell):" + chr(10) + "    return {'y': 1.0}" + chr(10), encoding="utf-8")
    ex = _Hangs()
    got = asyncio.run(small_run.run(ex, sys.executable, root, "code/simulate.py", grid, deterministic=True, timeout_s=60))
    assert len(ex.given) == 2 and ex.given[0] <= 30, "one part that hangs cannot take the time of the next"
    assert got.timed_out and "in 2 of them" in got.says()
    # a plan without parts keeps the middle and the two corners, and the whole time for the first
    plain = {"n": [1, 2, 3, 4, 5]}
    assert small_run.settings(plain) == [{"n": 3}, {"n": 1}, {"n": 5}]
    ex2 = _Hangs()
    asyncio.run(small_run.run(ex2, sys.executable, root, "code/simulate.py", plain, deterministic=True, timeout_s=60))
    # the whole time less what setting up took (a second on a busy machine), not a share of it
    assert ex2.given[0] >= 55, ex2.given


class _VirtualSelector:
    """A selector that never waits: a wait for the next timer moves the loop's own clock forward instead."""

    def __init__(self, real: Any, clock: list[float]) -> None:
        self._real, self._clock = real, clock

    def select(self, timeout: float | None = None) -> Any:
        if timeout is None:
            return self._real.select(None)  # no timer: only a real event (a thread's end) can wake the loop
        if timeout > 0:
            self._clock[0] += timeout
        return self._real.select(0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _virtual(monkeypatch: pytest.MonkeyPatch, make: Any) -> Any:
    """Run ``make()`` (a coroutine) on an event loop whose seconds pass without waiting, with the small run's clock and the
    memory gate set to match: no test here sleeps real seconds, so a slow runner cannot change what it shows."""
    from core import execution

    clock = [1000.0]
    loop = asyncio.SelectorEventLoop()
    loop._selector = _VirtualSelector(loop._selector, clock)  # type: ignore[attr-defined]
    loop.time = lambda: clock[0]  # type: ignore[method-assign]
    monkeypatch.setattr(small_run, "_clock", loop.time)
    # the copy of code/ outside the quest is made in threads, in real time: the fake executor is run as it is
    monkeypatch.setattr(small_run._trials, "_outside_executor", lambda executor, quest_root, log=None: executor)
    monkeypatch.setattr(execution, "_free_mb", lambda: None)  # memory cannot be told: the gate starts settings at once
    try:
        return loop.run_until_complete(make())
    finally:
        loop.close()


class _Costs:
    """An executor that plays a known cost per small-run setting (``costs[i]`` seconds for the setting in ``s<i>``, ``None``
    for one that never ends, ``"load"`` for one that reports that the simulation cannot be loaded, ``"raise"`` for one
    whose process cannot start): a setting that fits its time writes an ``ok`` row per trial, one that does not is timed
    out. ``ignore_timeout``: a process that is still there after its time (the executor's own wait for the pipes of a
    killed process). It records the time each process was given, the order they started in, how many ran at the same time,
    which were cancelled and, for each, whether the shared ``meter`` was still set then."""

    def __init__(self, costs: list[Any], *, ignore_timeout: bool = False) -> None:
        self.costs = costs
        self.ignore_timeout = ignore_timeout
        self.given: dict[int, int] = {}
        self.started: list[int] = []
        self.running = 0
        self.most = 0
        self.cancelled: list[int] = []
        self.meter_at_cancel: dict[int, Any] = {}
        self.meter: Any = "original"

    async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
        import json

        from core.execution import ExecutionResult

        spec_path, out_path = Path(cwd) / cmd[2], Path(cwd) / cmd[3]
        index = int(next(p for p in Path(cmd[2]).parts if p.startswith("s") and p[1:].isdigit())[1:])
        self.given[index] = int(timeout_s)
        self.started.append(index)
        cost = self.costs[index]
        self.running += 1
        self.most = max(self.most, self.running)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if cost == "raise":
                await asyncio.sleep(1)
                raise OSError("the process could not start")
            if cost == "load":
                await asyncio.sleep(0.1)
                out_path.write_text(json.dumps({"nonce": spec["nonce"], "load_error": "No module named 'helper'"}) + "\n",
                                    encoding="utf-8")
                return ExecutionResult(returncode=1, stdout="", stderr="", duration_s=0.1)
            if cost is None or cost > timeout_s:
                await asyncio.sleep(float(timeout_s) + (30.0 if self.ignore_timeout else 0.0))
                return ExecutionResult(returncode=-1, stdout="", stderr="", duration_s=float(timeout_s), timed_out=True)
            await asyncio.sleep(cost)
        except asyncio.CancelledError:
            self.cancelled.append(index)
            self.meter_at_cancel[index] = self.meter
            raise
        finally:
            self.running -= 1
        rows = [json.dumps({"nonce": spec["nonce"], "trial": t["trial"], "seed": t.get("seed"), "status": "ok",
                            "values": {"y": 1.0}}) for t in spec["trials"]]
        out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=float(cost))


def _parts_quest(tmp_path: Path) -> tuple[Path, Any]:
    grid = search.study_grid({"grid": {"kind": ["a", "b", "c"], "n": [1, 2, 3]},
                              "sub_studies": [{"name": "as", "vary": ["n"], "hold": {"kind": "a"}},
                                              {"name": "bs", "vary": ["n"], "hold": {"kind": "b"}},
                                              {"name": "cs", "vary": ["n"], "hold": {"kind": "c"}}]})
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {'y': 1.0}\n", encoding="utf-8")
    return root, grid


def _go(monkeypatch: pytest.MonkeyPatch, ex: Any, root: Path, grid: Any, timeout_s: int, at_once: int) -> Any:
    return _virtual(monkeypatch, lambda: small_run.run(ex, sys.executable, root, "code/simulate.py", grid,
                                                       deterministic=True, timeout_s=timeout_s, at_once=at_once))


def test_settings_run_at_once_and_a_costly_part_may_use_the_whole_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # One after another, each part's setting had a third of the time: the second part, costing half of the whole time,
    # never finished. Run together, each may use the whole time, and the small run still ends within it.
    root, grid = _parts_quest(tmp_path)
    one_by_one = _Costs([0.1, 3.0, 0.1])
    got = _go(monkeypatch, one_by_one, root, grid, 6, 1)
    assert [s.timed_out for s in got.settings] == [False, True, False], "one at a time: each part keeps its share"
    assert one_by_one.given[1] == 2 and one_by_one.most == 1, one_by_one.given
    together = _Costs([1.0, 3.0, 1.0])
    got = _go(monkeypatch, together, root, grid, 6, 3)
    assert [s.ok for s in got.settings] == [1, 1, 1] and not got.timed_out, got.says()
    assert together.started == [0, 1, 2] and together.most == 3
    assert together.given == {0: 6, 1: 6, 2: 6}, "each gets the whole time, not a share"
    assert got.seconds == 3.0 and "ran out" not in got.says()


def test_a_setting_that_never_ends_still_times_out_when_settings_run_at_once(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    root, grid = _parts_quest(tmp_path)
    ex = _Costs([0.1, None, 0.1])
    got = _go(monkeypatch, ex, root, grid, 6, 3)
    assert [s.timed_out for s in got.settings] == [False, True, False]
    assert [s.ok for s in got.settings] == [1, 0, 1]
    assert got.seconds == 6.0, "the small run's whole time bounds it"
    assert "ran out in 1 of them" in got.says()


def test_with_fewer_places_than_settings_a_waiting_part_gets_its_share_of_what_is_left(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two places for three parts that never end: the first to start gets half of the time (two rounds are left), and
    # the third starts when it ends, with what is left; every part starts and the whole small run ends within its time.
    root, grid = _parts_quest(tmp_path)
    ex = _Costs([None, None, None])
    got = _go(monkeypatch, ex, root, grid, 8, 2)
    assert ex.started == [0, 1, 2] and ex.most == 2, "every part started, two at a time"
    assert ex.given == {0: 4, 1: 8, 2: 4}, ex.given
    assert all(s.timed_out for s in got.settings) and got.seconds == 8.0


def test_a_setting_that_cannot_load_the_simulation_stops_the_run_at_once(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    # One after another, the first setting's load failure stopped the run; run together, the others kept running to the end
    # of the time and the record showed a run that had never been made.
    root, grid = _parts_quest(tmp_path)
    expected = _go(monkeypatch, _Costs(["load", None, None]), root, grid, 60, 1)
    ex = _Costs(["load", None, None])
    got = _go(monkeypatch, ex, root, grid, 60, 2)
    assert ex.started == [0, 1], "the setting waiting for a place is never started"
    assert ex.cancelled == [1] and ex.running == 0, "the running one is stopped"
    assert got.seconds < 1 and got.load_error and got.crashed
    a, b = expected.record(), got.record()
    a.pop("seconds"), b.pop("seconds")
    assert a == b and len(b["settings"]) == 1, "the record the one-at-a-time run gave"


def test_the_whole_time_is_a_hard_limit_even_when_a_process_does_not_stop(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    # A timed-out process whose output pipe stays open is waited for 30 s by the executor: the small run is not.
    root, grid = _parts_quest(tmp_path)
    ex = _Costs([0.1, None, None], ignore_timeout=True)
    got = _go(monkeypatch, ex, root, grid, 6, 3)
    assert sorted(ex.cancelled) == [1, 2], "the settings still running are cancelled, which stops their process trees"
    assert got.seconds == 6 + small_run.CLEANUP_S, got.seconds
    assert [s.timed_out for s in got.settings] == [False, True, True] and [s.ok for s in got.settings] == [1, 0, 0]


def test_a_process_that_cannot_start_stops_the_others_before_the_shared_state_is_given_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, grid = _parts_quest(tmp_path)
    ex = _Costs([None, "raise", None])
    with pytest.raises(OSError, match="could not start"):
        _go(monkeypatch, ex, root, grid, 60, 3)
    assert sorted(ex.cancelled) == [0, 2] and ex.running == 0, "no setting runs on after the small run failed"
    assert all(m != "original" for m in ex.meter_at_cancel.values()), "they stopped while the meter was still set"
    assert ex.meter == "original", "the executor is given back as it was"


_CHEAP = "def run_cell(cell):\n    return {'y': 2.0 * float(cell['n']), 'z': float(cell['m'])}\n"


@pytest.mark.parametrize("code", [_CHEAP, "raise ImportError('no such module: helper')\n"])
def test_a_cheap_study_and_one_that_cannot_load_give_the_same_record_at_once(tmp_path: Path, code: str) -> None:
    records = []
    for at_once in (1, 3):
        root = tmp_path / f"quest{at_once}"
        (root / "code").mkdir(parents=True)
        (root / "code" / "simulate.py").write_text(code, encoding="utf-8")
        got = asyncio.run(small_run.run(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root,
                                        "code/simulate.py", {"n": [1, 2, 3], "m": [5, 6]}, deterministic=True,
                                        timeout_s=60, at_once=at_once))
        record = got.record()
        record.pop("seconds")
        records.append(record)
    assert records[0] == records[1]
    if code == _CHEAP:
        assert len(records[0]["settings"]) == 3 and all(s["ok"] == 1 for s in records[0]["settings"])
    else:
        assert records[0]["load_error"] and len(records[0]["settings"]) == 1, "no setting after the first is kept"


@pytest.mark.usefixtures("this_python")
def test_the_engine_runs_the_small_run_with_its_settings_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_disclosure import _engine

    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(_CHEAP, encoding="utf-8")
    eng = _engine(root)
    eng._trial_mode, eng._trial_entries = True, {"run_cell"}  # type: ignore[attr-defined]
    eng._protocol_block = lambda state: {"grid": {"n": [1, 2, 3], "m": [5, 6]}}  # type: ignore[assignment]
    monkeypatch.setattr(eng, "_settings_at_once", lambda: 5)
    seen: dict[str, Any] = {}
    real = small_run.run

    async def spy(*args: Any, **kw: Any) -> Any:
        seen.update(kw)
        return await real(*args, **kw)

    monkeypatch.setattr(small_run, "run", spy)
    record = asyncio.run(eng._small_run({}, sys.executable, {}))
    assert seen.get("at_once") == 5 and record and not record.get("crashed")
    assert [s["ok"] for s in record["settings"]] == [1, 1, 1]


#: A search whose simulation reads a value it compares inside each run as one number: every evaluation fails.
_SEARCH_BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"}],
    "baseline": {"values": {"x": 0.0}, "source": "the design in use"},
    "evaluation_budget": {"starts": 2, "per_start": 10},
    "in_each_run": {"depth": [1.0, 2.0, 4.0]},
}
_READS_ONE = "def run_cell(cell):\n    return {'f': float(cell['x']) ** 2 + float(cell['depth'])}\n"
# (it returns the objective too: code that returned only its numbers per value would fail every evaluation of the search,
# which the small run now names, core/small_run.py ``judge_baseline``)
_READS_EACH = ("def run_cell(cell):\n"
               "    out = {f'f_depth{d:g}': float(cell['x']) ** 2 + d for d in cell['depth']}\n"
               "    return {**out, 'f': sum(out.values())}\n")


@pytest.mark.usefixtures("this_python")
@pytest.mark.parametrize("code, crashed", [(_READS_ONE, True), (_READS_EACH, False)])
def test_a_search_gets_the_small_run_on_its_scan_settings(tmp_path: Path, code: str, crashed: bool) -> None:
    """A live search's simulation read a value compared inside each run as one number and crashed at every evaluation;
    a search had no small run, so the known-answer checks met the crash first and spent one of their two repairs on it.
    The small run now runs a search on its coarse scan's settings, with the baseline design and the values compared
    inside each run, as the search calls the simulation; code that reads them as the search gives them still passes."""
    from tests.test_disclosure import _engine

    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(code, encoding="utf-8")
    eng = _engine(root)
    eng.executor = SharedInterpreterExecutor(python_version="3.11")  # type: ignore[assignment]
    eng._trial_mode, eng._trial_entries = True, {"run_cell"}  # type: ignore[attr-defined]
    eng._protocol_block = lambda state: {"optimisation": _SEARCH_BLOCK}  # type: ignore[assignment]
    sent = asyncio.run(eng._small_run_gate({"exec_reflect_iter": 0}, sys.executable, {}))
    if crashed:
        assert sent is not None and sent["exec_result"]["failed_script"] == "simulate.py"
        assert "float() argument must be" in sent["exec_result"]["stderr_tail"]
    else:
        assert sent is None
        assert "ran" in eng._small_run_note  # type: ignore[attr-defined]


_PER_VALUE = ("def run_cell(cell):\n"
              "    return {'f': {f'depth={d:g}': float(cell['x']) ** 2 + d for d in cell['depth']}}\n")


@pytest.mark.usefixtures("this_python")
def test_a_search_objective_returned_per_value_is_sent_back_once_and_never_stops(tmp_path: Path) -> None:
    """A search whose objective came back as one number per value compared inside each run failed every evaluation, found
    only after the long run had started. The small run's evaluation of the baseline design, judged as the search judges it,
    sends it back once (on its own counter: a spent crash counter does not block it); the same fault after that is a line in
    plan.md, and the quest goes on."""
    from tests.test_disclosure import _engine

    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(_PER_VALUE, encoding="utf-8")
    (root / "plan.md").write_text("# Plan\n", encoding="utf-8")
    eng = _engine(root)
    eng.executor = SharedInterpreterExecutor(python_version="3.11")  # type: ignore[assignment]
    eng._trial_mode, eng._trial_entries = True, {"run_cell"}  # type: ignore[attr-defined]
    eng._protocol_block = lambda state: {"optimisation": _SEARCH_BLOCK}  # type: ignore[assignment]
    state = {"exec_reflect_iter": eng.config.engine.exec_reflect_max_iterations}
    sent = asyncio.run(eng._small_run_gate(state, sys.executable, {}))
    assert sent is not None and sent["exec_result"]["failed_script"] == "simulate.py"
    assert sent["exec_result"]["gate_repair"] is True
    assert "the objective `f` is returned per value of depth" in sent["exec_result"]["stderr_tail"]
    assert asyncio.run(eng._small_run_gate(state, sys.executable, {})) is None
    assert "sent back to be repaired once" in (root / "plan.md").read_text(encoding="utf-8")
