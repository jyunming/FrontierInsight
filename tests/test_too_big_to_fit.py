"""A study FI knows will not fit is sized before it is cut, and a cut reaches every value of every setting early.

Three parts (core/run_checks.py ``_size_the_run`` / ``_cut_grid``, core/optimise_search.py ``iter_spread_cells``,
core/trial_runner.py ``run_trials(order=...)``):

* a grid FI cannot list whole (more than ``MAX_LISTED_SETTINGS``) gets every speed-up that can apply to it before it is cut:
  the code speed-up (as before) and fewer runs per setting (it lists nothing); sharing processes cannot be checked on it;
* when FI's own timing says fewer than one in ten of the plan's settings fit, the quest stops with what FI measured, once,
  so the evidence gate asks the plan for a study that fits; after that one ask the study is cut as before;
* a cut grid runs its settings spread over the plan's values (every value of every setting early), lazily for a grid FI
  cannot list; a study that fits runs in the plan's order as always.

The topic of these tests is a made-up damped spring.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from core import disclosure, optimise_search, plan, run_estimate, trial_runner
from core.execution import ExecutionResult, SharedInterpreterExecutor
from tests import exact_timing
from tests.exact_timing import VIRTUAL_CLOCK


@pytest.fixture(autouse=True)
def _exact_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    exact_timing.install(monkeypatch)
    from core.engine import Engine

    async def off(self: Any, state: Any) -> None:
        return None

    # the fake plan model takes its first call as the plan's draft (as in tests/test_run_checks.py, tests/conftest.py)
    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)


# ---- (a) the spread-out order -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [(34, 6), (6, 34), (4, 6, 9), (12, 8, 6, 4), (5, 1, 3), (1, 7), (2, 2, 2), (3,), ()])
def test_the_spread_order_is_a_permutation_that_reaches_every_value_of_every_setting_first(sizes: tuple[int, ...]) -> None:
    grid = {f"s{i}": [10 * i + k for k in range(n)] for i, n in enumerate(sizes)}
    spread = list(optimise_search.iter_spread_cells(grid))
    plan_order = list(optimise_search.iter_grid_cells(grid))
    key = trial_runner.cell_key
    assert len(spread) == len(plan_order) and sorted(map(key, spread)) == sorted(map(key, plan_order))
    assert len({key(c) for c in spread}) == len(spread), "each setting once"
    first = spread[:max(sizes or (1,))]
    for axis, values in grid.items():
        assert {c[axis] for c in first} == set(values), f"every value of {axis} within the first {len(first)} settings"
    assert spread == list(optimise_search.iter_spread_cells(grid)), "the same order every time (no randomness)"


def test_the_spread_order_of_a_study_in_parts_is_a_permutation_and_starts_every_part_early() -> None:
    grid = optimise_search.study_grid({"grid": {"mass": [1, 2, 3, 4, 5], "damping": [0.1, 0.2, 0.3, 0.4], "drive": [0, 1, 2]},
                                       "sub_studies": [{"vary": ["mass", "damping"], "hold": {"drive": 1}},
                                                       {"vary": ["damping", "drive"], "hold": {"mass": 5}},
                                                       {"vary": ["mass"], "hold": {}}]})
    spread = list(optimise_search.iter_spread_cells(grid))
    plan_order = list(optimise_search.iter_grid_cells(grid))
    key = trial_runner.cell_key
    assert sorted(map(key, spread)) == sorted(map(key, plan_order)) and len(set(map(key, spread))) == len(spread)
    # the parts take turns: each reaches every value it varies within its own first max(len) settings
    first = spread[:3 * 5]
    assert {c["mass"] for c in first} == {1, 2, 3, 4, 5} and {c["damping"] for c in first} == {0.1, 0.2, 0.3, 0.4}
    assert {c["drive"] for c in first} == {0, 1, 2}
    assert {trial_runner.cell_key(c) for c in spread[:3]} == {trial_runner.cell_key(next(iter(optimise_search.iter_spread_cells(
        optimise_search.study_grid({"grid": dict(grid), "sub_studies": [sub]}))))) for sub in grid.sub_studies}


class _Recording(SharedInterpreterExecutor):
    """Runs nothing: records the setting each process was started for. The settings gate lets waiting settings race for a
    free slot, so the order they start in is only near the order they are queued in; the tests read the queue
    (:func:`_queued`)."""

    def __init__(self) -> None:
        super().__init__(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")
        self.started: list[dict[str, Any]] = []

    async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001
        spec = json.loads((Path(cwd) / cmd[2]).read_text(encoding="utf-8"))
        self.started.append(spec["cell"])
        return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.0)


def _queued(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The settings in the order run_trials queues them for the settings gate (each batch of what ``_units`` is given)."""
    seen: list[dict[str, Any]] = []
    real = trial_runner._units

    def spy(quest_root: Any, part: list[dict[str, Any]], *a: Any, **k: Any) -> Any:
        seen.extend(task["cell"] for task in part)
        return real(quest_root, part, *a, **k)

    monkeypatch.setattr(trial_runner, "_units", spy)
    return seen


def _planned(root: Path) -> list[str]:
    rows = [json.loads(line) for line in (root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return [r["cell"] for r in rows if r.get("event") == "planned"]


def _quest(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("def run_trial(cell, trial, seed):\n    return {'y': 1.0}\n", encoding="utf-8")
    return root


@pytest.mark.asyncio
async def test_a_cut_grid_starts_its_settings_spread_out_and_its_ledger_keeps_the_plan_s_order(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path)
    grid = {"mass": [1, 2, 3, 4, 5, 6], "damping": [0.1, 0.2, 0.3, 0.4]}
    queued = _queued(monkeypatch)
    ex = _Recording()
    await trial_runner.run_trials(ex, sys.executable, root, "code/simulate.py", grid, runs_per_setting=1, base_seed=1,
                                  deterministic=False, timeout_s=60, order=trial_runner.ORDER_SPREAD)
    assert queued == list(optimise_search.iter_spread_cells(grid))
    assert {c["damping"] for c in queued[:6]} == {0.1, 0.2, 0.3, 0.4} and {c["mass"] for c in queued[:6]} == set(grid["mass"])
    assert len(ex.started) == len(queued)
    planned = _planned(root)
    assert planned == [trial_runner.cell_key(c) for c in optimise_search.iter_grid_cells(grid)], "the ledger in the plan's order"


# ---- (b) lazily, on a grid FI cannot list ---------------------------------------------------------------------------------

def test_the_spread_order_is_lazy_on_a_grid_far_larger_than_fi_lists() -> None:
    grid = {"mass": list(range(1000)), "damping": list(range(1000)), "drive": list(range(1000)), "phase": list(range(1000))}
    began = time.monotonic()
    first = list(itertools.islice(optimise_search.iter_spread_cells(grid), 1000))  # of 10^12 settings
    assert time.monotonic() - began < 5.0
    assert all(len({c[a] for c in first}) == 1000 for a in grid)


@pytest.mark.asyncio
async def test_a_cut_grid_of_more_settings_than_fi_lists_plans_the_spread_order_lazily(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the whole grid was listed")

    monkeypatch.setattr(optimise_search, "grid_cells", never)
    root = _quest(tmp_path)
    grid = {"mass": list(range(800)), "damping": list(range(800))}  # 640,000 settings
    queued = _queued(monkeypatch)
    ex = _Recording()
    await trial_runner.run_trials(ex, sys.executable, root, "code/simulate.py", grid, runs_per_setting=1, base_seed=1,
                                  deterministic=False, timeout_s=60, first_cells=100, order=trial_runner.ORDER_SPREAD)
    assert queued == list(itertools.islice(optimise_search.iter_spread_cells(grid), 100))
    assert len({c["mass"] for c in queued}) == 100 and len({c["damping"] for c in queued}) == 100
    planned = _planned(root)
    in_plan = sorted(queued, key=lambda c: (c["mass"], c["damping"]))
    assert planned == [trial_runner.cell_key(c) for c in in_plan], "the ledger lists what ran in the plan's order"


# ---- (c), (d), (e): the sizing --------------------------------------------------------------------------------------------

SPRING_SIM = VIRTUAL_CLOCK + '''
def run_trial(cell, trial, seed):
    time.sleep(0.05 * cell["n"])
    return {"y": float(cell["n"]) * 2.0 + cell["k"] + (seed % 3)}
'''


async def _sizing(tmp_path: Path, *, timeout_s: int, grid: dict[str, list[Any]], runs: int = 40, source: str = SPRING_SIM,
                  **protocol: Any) -> tuple[Any, Any, Any]:
    from tests.test_run_checks import _sizing as sizing

    eng, model, runner, ex = await sizing(tmp_path, timeout_s=timeout_s, grid=grid, runs=runs, source=source, **protocol)
    eng.config.engine.improve_rounds = 0  # the code speed-up has its own tests; (f) below spies on it
    return eng, model, runner


@pytest.mark.asyncio
async def test_fewer_than_one_in_ten_fitting_goes_back_to_the_plan_once_and_then_the_study_is_cut(tmp_path: Path) -> None:
    grid = {"n": [1, 2, 4], "k": list(range(20))}  # 60 settings; about one fits the time allowed
    eng, model, runner = await _sizing(tmp_path, timeout_s=8, grid=grid)
    state: dict[str, Any] = {"iteration": 0}
    patch = await eng._size_the_run(state, runner, sys.executable, None)
    assert patch is not None, "fewer than one in ten fit and the plan was not asked yet: the quest stops"
    fit = patch["exec_result"]["too_long_fit"]
    assert fit["settings"] == 60 and fit["settings_that_fit"] / fit["settings"] < run_estimate.FEW_FIT_SHARE
    text = patch["exec_result"]["too_long"]
    assert "fewer than one in ten" in text and "asks the plan once for a study that fits" in text
    assert "truncated" not in run_estimate.read_record(eng.quest_root), "nothing was cut"
    verdict = eng._no_results_verdict({"exec_result": patch["exec_result"], "result_json": {}, "iteration": 0,
                                       "exec_give_up_reason": text})
    assert verdict is not None and verdict["redesign"] is True and verdict.get("fit_request"), "the gate asks the plan once"
    # The plan was asked (it came back with the same study, so the same key): the same quest is now cut, not stopped again.
    asked = {"iteration": 0, "evidence_fit_redesigns": 1}
    assert await eng._size_the_run(asked, runner, sys.executable, None) is None
    cut = run_estimate.read_record(eng.quest_root)["truncated"]
    assert cut["kind"] == "grid" and cut["order"] == "spread" and runner.order == trial_runner.ORDER_SPREAD
    assert model.asked == []


@pytest.mark.asyncio
async def test_one_in_ten_or_more_fitting_is_cut_with_no_stop_and_runs_spread_out(tmp_path: Path) -> None:
    eng, model, runner = await _sizing(tmp_path, timeout_s=8, grid={"n": [1, 2, 4], "k": [0, 1]})  # 6 settings, 1 fits
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    cut = run_estimate.read_record(eng.quest_root)["truncated"]
    assert cut["fit"]["settings_that_fit"] / cut["settings"] >= run_estimate.FEW_FIT_SHARE
    assert cut["order"] == "spread" and runner.order == trial_runner.ORDER_SPREAD
    said = disclosure.cut_short_text(eng.quest_root)
    assert "in an order spread over the plan's values instead of the plan's order" in said
    assert "in the plan's order until" not in said
    # a resume keeps the cut's order
    runner.order = trial_runner.ORDER_PLAN
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert runner.order == trial_runner.ORDER_SPREAD


@pytest.mark.asyncio
async def test_a_study_that_fits_keeps_the_plan_s_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, model, runner = await _sizing(tmp_path, timeout_s=3600, grid={"n": [1, 2, 4], "k": [0, 1]})
    runner.order = trial_runner.ORDER_SPREAD  # left from an earlier sizing: a new sizing starts from the plan's order
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert run_estimate.read_record(eng.quest_root)["fits"] is True and runner.order == trial_runner.ORDER_PLAN
    root = _quest(tmp_path / "direct")
    grid = {"mass": [1, 2, 3], "damping": [0.1, 0.2]}
    queued = _queued(monkeypatch)
    await trial_runner.run_trials(_Recording(), sys.executable, root, "code/simulate.py", grid, runs_per_setting=1,
                                  base_seed=1, deterministic=False, timeout_s=60)
    assert queued == list(optimise_search.iter_grid_cells(grid)), "run_trials' own order is the plan's"


# ---- (f) every speed-up that can apply to a grid FI cannot list is tried before the cut ----------------------------------

TINY_SIM = VIRTUAL_CLOCK + '''
def run_trial(cell, trial, seed):
    time.sleep(0.00001)
    return {"y": float(cell["n"] + cell["m"]) + (seed % 3)}
'''


@pytest.mark.asyncio
async def test_a_grid_fi_cannot_list_is_sped_up_and_given_fewer_runs_before_it_is_cut(tmp_path: Path) -> None:
    big = {"n": list(range(800)), "m": list(range(800))}  # 640,000 settings
    eng, model, runner = await _sizing(tmp_path, timeout_s=120, grid=big, source=TINY_SIM, min_runs_per_setting=8)
    calls: list[str] = []
    speed_up, lower, cut = eng._speed_up_if_long, eng._lower_the_runs, eng._run_up_to_the_limit

    async def spy_speed_up(*a: Any, **k: Any) -> Any:
        calls.append("speed_up")
        return await speed_up(*a, **k)

    async def spy_group(*a: Any, **k: Any) -> Any:
        calls.append("share_processes")
        return None

    def spy_lower(*a: Any, **k: Any) -> Any:
        calls.append("fewer_runs")
        return lower(*a, **k)

    def spy_cut(*a: Any, **k: Any) -> Any:
        calls.append("cut")
        return cut(*a, **k)

    eng._speed_up_if_long = spy_speed_up  # type: ignore[method-assign]
    eng._group_settings = spy_group  # type: ignore[method-assign]
    eng._lower_the_runs = spy_lower  # type: ignore[method-assign]
    eng._run_up_to_the_limit = spy_cut  # type: ignore[method-assign]
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert calls == ["speed_up", "fewer_runs", "cut"], calls
    runs = plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"]
    assert 8 <= runs < 40, "every setting kept, fewer runs"
    record = run_estimate.read_record(eng.quest_root)
    assert record["reduced"]["to"]["runs_per_setting"] == runs
    assert record["truncated"]["planned_settings"] == optimise_search.MAX_LISTED_SETTINGS, "cut only at what FI lists"
    assert record["truncated"]["order"] == "spread" and runner.first_cells == optimise_search.MAX_LISTED_SETTINGS


@pytest.mark.asyncio
async def test_a_grid_fi_cannot_list_whose_time_fits_keeps_its_runs_and_is_cut_only_at_what_fi_lists(tmp_path: Path) -> None:
    big = {"n": list(range(800)), "m": list(range(800))}  # 640,000 settings; their time fits an hour
    eng, model, runner = await _sizing(tmp_path, timeout_s=3600, grid=big, source=TINY_SIM, min_runs_per_setting=8)
    lowered: list[Any] = []
    eng._lower_the_runs = lambda *a, **k: lowered.append(a)  # type: ignore[method-assign]
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert lowered == [], "a study whose time fits is not made smaller"
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == 40
    record = run_estimate.read_record(eng.quest_root)
    assert not record.get("reduced") and record["truncated"]["planned_settings"] == optimise_search.MAX_LISTED_SETTINGS


# ---- a rerun of a cut study finds the settings it planned ----------------------------------------------------------------

def test_a_rerun_of_a_spread_out_cut_finds_its_planned_settings_beyond_what_fi_lists_in_the_plan_s_order(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A grid of 640,000 settings cut in the spread-out order planned settings far past the first 500,000 of the plan's
    order (its second setting is the plan's last); the rerun of the delivered study must find them all."""
    import asyncio

    from core import rerun_check

    grid = {"a": list(range(800)), "b": list(range(800))}
    protocol = {"grid": grid, "runs_per_setting": 1}
    root = _quest(tmp_path)
    # the first three settings of the spread-out order (both ends, then the middle); the second is the plan's last
    planned = [{"a": 0, "b": 0}, {"a": 799, "b": 799}, {"a": 399, "b": 399}]
    rows = []
    for cell in planned:
        key = trial_runner.cell_key(cell)
        rows.append({"event": "planned", "run_id": "r1", "cell": key, "trials": 1})
        rows.append({"event": "trial", "run_id": "r1", "cell": key, "trial": 0, "status": "ok",
                     "seed": trial_runner.trial_seed(0, key, 0)})
    (root / "raw").mkdir()
    (root / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (root / ".fi").mkdir()
    (root / ".fi" / "run_estimate.json").write_text(json.dumps({"fits": False, "truncated": {
        "kind": "grid", "settings": 640000, "planned_settings": 3, "order": "spread", "run_id": "r1"}}), encoding="utf-8")
    asked: list[Any] = []

    class Ran(Exception):
        pass

    async def spy(*_a: Any, **k: Any) -> Any:
        asked.append(k.get("cell_list"))
        raise Ran

    monkeypatch.setattr(trial_runner, "run_trials", spy)
    copy = rerun_check.copy_code(root, tmp_path / "repaired")
    with pytest.raises(Ran):
        asyncio.run(rerun_check._run_like_main(
            _Recording(), sys.executable, root, root, copy, protocol, 60, None, tmp_path / "work", 1, [],
            say=None, finding="", info={}, began=time.monotonic()))
    assert asked == [planned], "the rerun runs the settings the delivered run planned, found in the order it ran them"


def _rerun_cell_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grid: dict[str, list[Any]],
                     ledger: list[tuple[dict[str, Any], str]], order: str) -> list[Any]:
    """The settings a rerun of the delivered study queues (``cell_list``), from a ledger of ``(setting, ok | late)`` rows
    and a cut recorded in ``order``."""
    import asyncio

    from core import rerun_check

    root = _quest(tmp_path)
    rows = []
    for cell, what in ledger:
        key = trial_runner.cell_key(cell)
        rows.append({"event": "planned", "run_id": "r1", "cell": key, "trials": 1})
        rows.append({"event": "trial", "run_id": "r1", "cell": key, "trial": 0, "seed": trial_runner.trial_seed(0, key, 0),
                     "status": "ok" if what == "ok" else "failed",
                     **({"reason": trial_runner.OUT_OF_TIME} if what == "late" else {})})
    (root / "raw").mkdir()
    (root / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (root / ".fi").mkdir()
    (root / ".fi" / "run_estimate.json").write_text(json.dumps({"fits": False, "truncated": {
        "kind": "grid", "settings": len(ledger), "order": order, "run_id": "r1"}}), encoding="utf-8")
    asked: list[Any] = []

    class Ran(Exception):
        pass

    async def spy(*_a: Any, **k: Any) -> Any:
        asked.append(k.get("cell_list"))
        raise Ran

    monkeypatch.setattr(trial_runner, "run_trials", spy)
    copy = rerun_check.copy_code(root, tmp_path / "repaired")
    with pytest.raises(Ran):
        asyncio.run(rerun_check._run_like_main(
            _Recording(), sys.executable, root, root, copy, {"grid": grid, "runs_per_setting": 1}, 60, None,
            tmp_path / "work", 1, [], say=None, finding="", info={}, began=time.monotonic()))
    return asked[0]


def test_a_rerun_of_a_spread_out_cut_queues_its_settings_in_the_order_the_delivered_run_queued_them(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ledger lists a cut study's settings in the plan's order; the delivered run queued them 0, 3, 1, 2 (both ends,
    then the middle), and its time limit cut x=2 off. The rerun queues 0, 3, 1, not the ledger's 0, 1, 3."""
    grid = {"x": [0, 1, 2, 3]}
    ledger = [({"x": 0}, "ok"), ({"x": 1}, "ok"), ({"x": 2}, "late"), ({"x": 3}, "ok")]
    queued = _rerun_cell_list(tmp_path, monkeypatch, grid, ledger, "spread")
    assert queued == [{"x": 0}, {"x": 3}, {"x": 1}]
    # known positive: a cut in the plan's order is rerun in the plan's order
    assert _rerun_cell_list(tmp_path / "plan", monkeypatch, grid, ledger, "plan") == [{"x": 0}, {"x": 1}, {"x": 3}]


def test_the_cut_sentence_says_only_the_order_and_what_the_ledger_counts(tmp_path: Path) -> None:
    """Two of a hundred settings ran: the paper may say the order was spread over the plan's values, never that every
    value was reached."""
    root = tmp_path / "quest"
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "run_estimate.json").write_text(json.dumps({"fits": False, "truncated": {
        "kind": "grid", "estimate_s": 64800.0, "at_least": False, "limit_s": 28800.0, "settings": 100, "order": "spread",
        "run_id": "r1"}}), encoding="utf-8")
    rows = []
    for v in range(100):
        rows.append({"event": "planned", "run_id": "r1", "cell": f"x={v}", "trials": 1})
        rows.append({"event": "trial", "run_id": "r1", "cell": f"x={v}", "trial": 0, "seed": 1,
                     **({"status": "ok"} if v in (0, 99) else {"status": "failed", "reason": trial_runner.OUT_OF_TIME})})
    (root / "raw").mkdir()
    (root / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    said = disclosure.cut_short_text(root)
    assert "in an order spread over the plan's values instead of the plan's order" in said
    assert "of 100 settings, 2 ran all their trials, 0 ran some and 98 ran none" in said
    assert "every value" not in said and "reached" not in said


# ---- the trials of a cut are reused only for that very cut ---------------------------------------------------------------

COUNT_SIM = """
def run_trial(cell, trial_id, seed):
    return {"y": float(cell["x"]) + trial_id}
"""
COUNT_ANALYSIS = """
import json, os
data = json.load(open(os.environ["FI_TRIALS"]))
print("RESULT_JSON: " + json.dumps({"settings": len(data["cells"])}))
"""


class _Counting(SharedInterpreterExecutor):
    def __init__(self) -> None:
        super().__init__(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")
        self.settings_run = 0

    async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001
        if len(cmd) > 2 and "trial_harness" in str(cmd[1]):
            self.settings_run += 1
        return await super().execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)


@pytest.mark.asyncio
async def test_trials_of_a_run_in_another_order_or_of_another_cut_are_not_reused_as_a_cut(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(COUNT_SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(COUNT_ANALYSIS, encoding="utf-8")
    ex = _Counting()
    runner = trial_runner.TrialsRunner(ex, quest_root=root, protocol={"grid": {"x": [0, 1, 2, 3]}, "runs_per_setting": 1},
                                       deterministic=False, simulate=root / "code" / "simulate.py",
                                       analysis=root / "code" / "experiment.py")
    cmd = [sys.executable, str(root / "code" / "experiment.py")]

    async def run() -> int:
        before = ex.settings_run
        result = await runner.execute(cmd, cwd=root, timeout_s=120, env={})
        assert result.returncode == 0, result.stderr
        return ex.settings_run - before

    assert await run() == 4, "the study, in the plan's order"
    assert await run() == 0, "the same study again: its trials are reused (the known positive)"
    runner.run_id, runner.order = "cut1", trial_runner.ORDER_SPREAD  # the sizing now cuts it
    assert await run() == 4, "a run in the plan's order is not a spread-out cut's"
    assert await run() == 0, "the same cut (a resume) reuses its own trials"
    runner.run_id = "cut2"
    assert await run() == 4, "another cut is not reused"
