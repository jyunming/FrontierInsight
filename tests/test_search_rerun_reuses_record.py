"""A run that comes back to execute after its search already ran on the same simulation (a repair of the analysis) uses
FI's record of that search: the scan is not timed and the simulation not sped up again, and the search is given the time
it ran for. A first run, and a run after a real change to the simulation, are sized as before.

A neutral made-up search (tests/test_cost_wiring.py): a 1 h limit and a search too long for it, so the first sizing cuts it."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import optimise, run_estimate
from tests.test_cost_wiring import _engine
from tests.test_run_checks import _log

FRAME = "# >>> FI analysis frame, version 1: written by FI from the plan; a repair does not change it\n"


def _count_timings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    async def timed(*a: Any, name: str = "pilot", **k: Any) -> Any:
        calls.append(name)
        return run_estimate.Probe(None, None, {}, 15.0), ""

    monkeypatch.setattr(run_estimate, "_time_one", timed)
    return calls


def _search_ran(eng: Any, runner: Any, block: dict[str, Any], *, stopped: str = "time") -> dict[str, Any]:
    """Write FI's record of a search that ran on the code as it is now, under the cut the sizing recorded, as
    ``optimise.run_search`` writes it."""
    root = eng.quest_root
    cut = run_estimate.read_record(root).get("truncated") or {}
    norm = optimise._plan.normalize(block)[0]
    key = optimise._run_key(root, runner.simulate, norm, 0, "run_cell", {}, [], True, True, True)
    search_s = int(3600 - runner.check_reserve_s)
    if runner.search_deadline_s is not None:
        search_s = min(search_s, int(runner.search_deadline_s))
    ledger = json.dumps({"event": "evaluation", "status": "ok", "design": {"x": 0.0}, "objective": 0.0}) + "\n"
    best = {"evaluations": {"stopped_because": stopped}, "time": {"seconds": 10.0, "limit_seconds": search_s},
            "run_id": cut.get("run_id", "")}
    optimise._save(root, key, ledger, json.dumps(best), "", False,
                   optimise._study_key(root, runner.simulate, norm, "run_cell", {}))
    return {"search_s": search_s, "cut": cut, "key": key}


def _fresh_runner_fields(runner: Any) -> None:
    """A resumed process: the runner holds nothing of the earlier sizing."""
    runner.check_reserve_s, runner.search_deadline_s, runner.run_id = 0.0, None, ""


@pytest.mark.asyncio
async def test_a_search_that_already_ran_on_this_simulation_is_not_sized_or_sped_up_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, block = await _engine(tmp_path)
    calls = _count_timings(monkeypatch)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    first = len(calls)
    assert first > 0, "a first run is timed"
    ran = _search_ran(eng, runner, block)
    _fresh_runner_fields(runner)
    # the analysis is repaired (FI's derive.py): the search's record is still the simulation's
    (eng.quest_root / "code" / "derive.py").write_text(FRAME + "def figure_a(fig, table):\n    return None\n",
                                                        encoding="utf-8")
    speed_ups: list[Any] = []

    async def speed_up(*a: Any, **k: Any) -> dict[str, Any]:
        speed_ups.append(1)
        return {}

    monkeypatch.setattr(type(eng), "_speed_up", speed_up)
    assert await eng._size_the_run({"iteration": 1}, runner, sys.executable, None) is None
    assert len(calls) == first, "the scan was timed again"
    assert speed_ups == []
    assert "already ran on this simulation" in _log(eng)
    # the search is given the time it ran for, so the run uses its record (core/optimise.py run_search)
    search_s = int(3600 - runner.check_reserve_s)
    if runner.search_deadline_s is not None:
        search_s = min(search_s, int(runner.search_deadline_s))
    assert search_s <= ran["search_s"]
    assert runner.run_id == ran["cut"]["run_id"]
    assert run_estimate.read_record(eng.quest_root).get("truncated") == ran["cut"]
    text = (eng.quest_root / optimise.RECORD).read_text(encoding="utf-8")
    assert optimise._kept(eng.quest_root, json.loads(text), text, ran["key"], frozen=False, timeout_s=search_s) is not None


@pytest.mark.asyncio
async def test_a_real_change_to_the_simulation_is_sized_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, block = await _engine(tmp_path)
    calls = _count_timings(monkeypatch)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    _search_ran(eng, runner, block)
    _fresh_runner_fields(runner)
    sim = eng.quest_root / "code" / "simulate.py"
    sim.write_text(sim.read_text(encoding="utf-8").replace("** 2", "** 2 + 1"), encoding="utf-8")
    first = len(calls)
    assert await eng._size_the_run({"iteration": 1}, runner, sys.executable, None) is None
    assert len(calls) > first
    assert "already ran on this simulation" not in _log(eng)


@pytest.mark.asyncio
async def test_a_change_to_a_module_beside_the_simulation_is_sized_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, block = await _engine(tmp_path)
    calls = _count_timings(monkeypatch)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    _search_ran(eng, runner, block)
    _fresh_runner_fields(runner)
    (eng.quest_root / "code" / "helpers.py").write_text("SCALE = 2\n", encoding="utf-8")
    first = len(calls)
    assert await eng._size_the_run({"iteration": 1}, runner, sys.executable, None) is None
    assert len(calls) > first


@pytest.mark.asyncio
async def test_a_longer_time_limit_is_sized_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, block = await _engine(tmp_path)
    calls = _count_timings(monkeypatch)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    _search_ran(eng, runner, block)
    _fresh_runner_fields(runner)
    eng.config.execution.timeout_s = 7200
    first = len(calls)
    assert await eng._size_the_run({"iteration": 1}, runner, sys.executable, None) is None
    assert len(calls) > first


@pytest.mark.asyncio
async def test_a_first_run_with_no_record_is_sized_as_before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, block = await _engine(tmp_path)
    calls = _count_timings(monkeypatch)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert calls
    assert "already ran on this simulation" not in _log(eng)


def test_a_repair_of_the_analysis_is_not_another_search(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    sim = code / "simulate.py"
    sim.write_text("def run_cell(cell):\n    return {'f': 1.0}\n", encoding="utf-8")
    block = {"objective": {"quantity": "f"}}

    def key() -> str:
        return optimise._run_key(tmp_path, sim, block, 0, "run_cell", {})

    before = key()
    (code / "derive.py").write_text(FRAME + "def figure_a(fig, table):\n    return None\n", encoding="utf-8")
    assert key() == before
    (code / "derive.py").write_text(FRAME + "def figure_a(fig, table):\n    return 1\n", encoding="utf-8")
    assert key() == before
    (code / "helpers.py").write_text("SCALE = 2\n", encoding="utf-8")
    assert key() != before
    with_helper = key()
    sim.write_text("def run_cell(cell):\n    return {'f': 2.0}\n", encoding="utf-8")
    assert key() != with_helper


def _record(tmp_path: Path, stopped: str) -> tuple[Path, dict[str, Any]]:
    """A search record written while every stop was armed (a repair was left), as run_search writes it."""
    code = tmp_path / "code"
    code.mkdir()
    sim = code / "simulate.py"
    sim.write_text("def run_cell(cell):\n    return {'f': cell['x'] ** 2}\n", encoding="utf-8")
    block = {"objective": {"quantity": "f", "direction": "minimise"},
             "design_variables": [{"name": "x", "low": -1.0, "high": 1.0, "kind": "continuous"}],
             "baseline": {"values": {"x": 0.0}, "source": "the design in use"},
             "evaluation_budget": {"starts": 1, "per_start": 4}}
    protocol = {"optimisation": block, "thresholds": {}}
    norm = optimise._plan.normalize(block)[0]
    key = optimise._run_key(tmp_path, sim, norm, 0, "run_cell", {}, [], True, True, True)
    ledger = json.dumps({"event": "evaluation", "status": "ok", "design": {"x": 0.0}, "objective": 0.0}) + "\n"
    optimise._save(tmp_path, key, ledger, json.dumps({"evaluations": {"stopped_because": stopped}}), "", False)
    return sim, protocol


def test_a_search_made_while_a_repair_was_left_is_kept_once_the_repairs_are_spent(tmp_path: Path) -> None:
    _sim, protocol = _record(tmp_path, "budget")
    assert optimise.kept_search(tmp_path, "code/simulate.py", protocol, base_seed=0) is not None
    # the repairs are spent: the stops that send the simulation back are off, and none of them fired in that search
    assert optimise.kept_search(tmp_path, "code/simulate.py", protocol, base_seed=0, stop_stepped=False,
                                stop_unchanged=False, stop_flat=False) is not None


def test_a_search_that_stopped_on_a_stop_that_is_off_now_is_not_kept(tmp_path: Path) -> None:
    _sim, protocol = _record(tmp_path, "stepped_objective")
    assert optimise.kept_search(tmp_path, "code/simulate.py", protocol, base_seed=0, stop_stepped=False) is None
