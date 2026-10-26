"""The search's own evaluations are given the values each run compares inside it, and keep a number per value."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from core import optimise, optimise_search
from core.execution import SharedInterpreterExecutor

_SIM = (
    "def run_cell(cell):\n"
    "    return {'f': cell['x'] ** 2, 'curve': {f'p={v}': cell['x'] + v for v in cell['p']}}\n"
)
_BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"}],
    "in_each_run": {"p": [1, 2, 3]},
}


def _quest(tmp_path: Path, source: str = _SIM) -> Path:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return tmp_path


async def _one(root: Path, **kw: Any) -> dict[str, Any]:
    cell = optimise_search.cell_of(_BLOCK, {"x": 2.0})
    got, _err, _load = await optimise._evaluate(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", "run_cell", cell,
        runs=1, base_seed=0, timeout_s=120, env=None, thresholds=None, **kw)
    return got


@pytest.mark.asyncio
async def test_an_evaluation_given_the_values_compared_inside_each_run_keeps_a_number_per_value(tmp_path: Path) -> None:
    got = await _one(_quest(tmp_path), in_each_run=optimise_search.in_each_run(_BLOCK))
    assert got["values"]["f"] == 4.0
    assert got["values"]["curve"] == {"p=1": 3.0, "p=2": 4.0, "p=3": 5.0}


@pytest.mark.asyncio
async def test_without_the_values_the_same_simulation_is_still_refused(tmp_path: Path) -> None:
    got = await _one(_quest(tmp_path))
    assert "non-numbers" in got["failed"]


@pytest.mark.asyncio
async def test_a_key_the_run_was_not_given_still_fails(tmp_path: Path) -> None:
    root = _quest(tmp_path, "def run_cell(cell):\n    return {'f': 1.0, 'curve': {'p=1': 1.0, 'p=9': 2.0}}\n")
    got = await _one(root, in_each_run=optimise_search.in_each_run(_BLOCK))
    assert "not one of the values the run was given" in got["failed"]


@pytest.mark.asyncio
async def test_a_batch_of_evaluations_is_given_the_values_too(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    cells = [optimise_search.cell_of(_BLOCK, {"x": float(x)}) for x in (1, 2)]
    for at_once in (1, 2):
        got = await optimise.evaluate_many(
            SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", "run_cell", cells,
            at_once=at_once, runs=1, base_seed=0, deadline=__import__("time").monotonic() + 120, env=None,
            thresholds=None, in_each_run=optimise_search.in_each_run(_BLOCK))
        assert [a["values"]["curve"]["p=1"] for a, _e, _l in got] == [2.0, 3.0]


def test_a_per_value_quantity_is_averaged_per_key_over_the_runs_and_a_number_as_before() -> None:
    rows = [{"values": {"f": 1.0, "curve": {"p=1": 2.0, "p=2": 4.0}}}, {"values": {"f": 3.0, "curve": {"p=1": 4.0, "p=2": 8.0}}}]
    assert optimise._mean_values(rows) == {"curve": {"p=1": 3.0, "p=2": 6.0}, "f": 2.0}


def test_an_objective_returned_per_value_fails_with_a_plain_message_and_picks_no_value() -> None:
    judged = optimise_search.judge(_BLOCK, {"f": {"p=1": 1.0, "p=2": 2.0}})
    assert judged["status"] == "failed" and judged["objective"] is None
    assert judged["reason"] == ("the objective `f` is returned per value of p, but a search needs one number per design "
                                "(the plan's objective must name one number)")
    assert optimise_search.judge(_BLOCK, {"f": 2.0})["status"] == "ok"


def test_a_small_run_is_not_reused_once_the_values_compared_inside_each_run_change(tmp_path: Path) -> None:
    from core import small_run

    sim = _quest(tmp_path) / "code" / "simulate.py"
    grid = {"x": [0.0, 1.0]}
    plain = small_run.key_of(sim, grid, True, None)
    assert small_run.key_of(sim, grid, True, None, None) == plain  # a study with none keeps its key
    assert small_run.key_of(sim, grid, True, None, {"p": [1, 2, 3]}) != plain
    assert small_run.key_of(sim, grid, True, None, {"p": [1, 2]}) != small_run.key_of(sim, grid, True, None, {"p": [1, 2, 3]})


@pytest.mark.asyncio
async def test_the_engines_small_run_is_made_again_when_only_in_each_run_is_added(tmp_path: Path) -> None:
    from tests.test_run_checks import _search_engine

    without = {k: v for k, v in _BLOCK.items() if k != "in_each_run"}
    eng, _runner = await _search_engine(tmp_path, without)
    eng._trial_mode, eng._trial_entries = True, {"run_cell"}
    (eng.quest_root / "code" / "simulate.py").write_text(
        "def run_cell(cell):\n    p = cell.get('p')\n    return {'f': 1.0} if p is None else "
        "{'f': {f'p={v}': float(v) for v in p}}\n", encoding="utf-8")
    block = {**without, "grid": {"x": [0.0]}, "baseline": {"values": {"x": 0.0}, "source": "s"}, "fixed": {}}
    eng._protocol_block = lambda state: {"optimisation": block}  # type: ignore[method-assign]
    first = await eng._small_run({}, sys.executable, {})
    block = {**block, "in_each_run": {"p": [1, 2]}}
    second = await eng._small_run({}, sys.executable, {})
    assert first["key"] != second["key"]


def test_the_coarse_scan_gives_a_per_value_objective_to_the_judge_as_an_evaluation_does() -> None:
    block = {**_BLOCK, "fixed": {}, "baseline": {"values": {"x": 0}, "source": "s"}, "grid": {"x": [0, 1]},
             "in_each_run": {"pitch_nm": [74, 80]}}
    summary = {"cells": [
        {"cell": {"x": 0}, "key": "x=0", "planned": 1, "ok": 1, "failed": 0, "metrics": {}},
        {"cell": {"x": 0, "pitch_nm": 74}, "key": "x=0,pitch_nm=74", "level": {"pitch_nm": 74}, "planned": 1, "ok": 1,
         "failed": 0, "metrics": {"f": {"values": [1.0]}}},
        {"cell": {"x": 0, "pitch_nm": 80}, "key": "x=0,pitch_nm=80", "level": {"pitch_nm": 80}, "planned": 1, "ok": 1,
         "failed": 0, "metrics": {"f": {"values": [2.0]}}},
    ]}
    rows = optimise_search.scan_rows(summary, block)
    assert rows[0]["values"] == {"f": {"pitch_nm=74": 1.0, "pitch_nm=80": 2.0}}
    judged = optimise_search.judge(block, rows[0]["values"])
    assert "is returned per value of pitch_nm" in judged["reason"]


def test_a_limit_on_a_quantity_returned_per_value_gets_the_same_plain_message_in_the_scan_and_an_evaluation() -> None:
    block = {**_BLOCK, "fixed": {}, "baseline": {"values": {"x": 0}, "source": "s"}, "grid": {"x": [0, 1]},
             "constraints": [{"quantity": "g", "limit": "<= 3"}]}
    summary = {"cells": [
        {"cell": {"x": 0}, "key": "x=0", "planned": 1, "ok": 1, "failed": 0, "metrics": {"f": {"values": [1.0]}}},
        {"cell": {"x": 0, "p": 1}, "key": "x=0,p=1", "level": {"p": 1}, "planned": 1, "ok": 1, "failed": 0,
         "metrics": {"g": {"values": [2.0]}}},
        {"cell": {"x": 0, "p": 2}, "key": "x=0,p=2", "level": {"p": 2}, "planned": 1, "ok": 1, "failed": 0,
         "metrics": {"g": {"values": [4.0]}}},
    ]}
    values = optimise_search.scan_rows(summary, block)[0]["values"]
    assert values == {"f": 1.0, "g": {"p=1": 2.0, "p=2": 4.0}}
    want = ("the constraint `g` is returned per value of p, but a search needs one number per design "
            "(the plan's constraint must name one number)")
    assert optimise_search.judge(block, values)["reason"] == want
    assert optimise_search.judge(block, {"f": 1.0, "g": 2.0})["status"] == "ok"


@pytest.mark.asyncio
async def test_the_repair_check_judges_a_search_whose_results_are_given_per_value(tmp_path: Path) -> None:
    # The check after a repair runs the search's own evaluation with the values compared inside each run, so a repaired
    # simulation that lost the objective is caught, and one that kept it is accepted (not left unjudged).
    async def check(source: str) -> str:
        (tmp_path / str(len(source))).mkdir()
        root = _quest(tmp_path / str(len(source)), source)
        return await optimise.search_entry_returns_objective(
            SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", _BLOCK,
            optimise_search.cell_of(_BLOCK, {"x": 2.0}), timeout_s=120, env=None, thresholds=None)

    lost = await check("def run_cell(cell):\n    return {'curve': {f'p={v}': cell['x'] + v for v in cell['p']}}\n")
    assert "not the objective `f`" in lost
    assert await check(_SIM) == ""
