"""Every cell FI gives the simulation says what it is (``cell["fi_kind"]``): a design the search evaluates, a row of the scan
or the comparison grid, a case of one of the plan's checks. Neutral toy study."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import grid_keys, in_run, optimise, optimise_search, trial_runner
from core.execution import SharedInterpreterExecutor

# A condition the plan holds fixed (``mode``) differs from the value a guess from values expects, so a rule over values
# takes the search design for something else and never reads the design.
_BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"}],
    "fixed": {"mode": "fast"},
}
_GUESSING = (
    "def run_cell(cell):\n"
    "    if cell.get('mode') in (None, 'base'):\n"
    "        return {'f': (cell['x'] - 1.0) ** 2}\n"
    "    return {'f': 7.0}\n"
)
_DISPATCHING = (
    "def run_cell(cell):\n"
    "    if cell.get('fi_kind') == 'search':\n"
    "        return {'f': (cell['x'] - 1.0) ** 2}\n"
    "    return {'f': 7.0 + (cell.get('fi_kind') == 'check')}\n"
)


def _quest(tmp_path: Path, source: str) -> Path:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return tmp_path


async def _search_objective(root: Path, x: float) -> float:
    cell = optimise_search.cell_of(_BLOCK, {"x": x})
    got, _err, _load = await optimise._evaluate(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", "run_cell", cell,
        runs=1, base_seed=0, timeout_s=120, env=None, thresholds=None)
    return got["values"]["f"]


@pytest.mark.asyncio
async def test_code_dispatching_on_the_kind_reads_the_design_of_a_search_evaluation(tmp_path: Path) -> None:
    root = _quest(tmp_path, _DISPATCHING)
    assert await _search_objective(root, 1.0) == 0.0
    assert await _search_objective(root, 3.0) == 4.0


@pytest.mark.asyncio
async def test_code_guessing_from_the_values_ignores_the_design_of_the_search_cell(tmp_path: Path) -> None:
    # The known positive: the fault (every design gives one output) is real for a guess, and the kind is what removes it.
    root = _quest(tmp_path, _GUESSING)
    assert await _search_objective(root, 1.0) == await _search_objective(root, 3.0) == 7.0


def test_a_scan_row_is_planned_as_a_scan_and_its_name_and_seed_do_not_carry_the_kind(tmp_path: Path) -> None:
    root = _quest(tmp_path, _DISPATCHING)
    folder = root / ".fi" / "trials"
    folder.mkdir(parents=True)
    plan = trial_runner._plan(root, "code/simulate.py", {"x": [1.0, 2.0]}, runs_per_setting=1, base_seed=0,
                              deterministic=True, folder=folder, out_name="c{index}.out.jsonl")
    spec = json.loads((root / plan[0]["spec"]).read_text(encoding="utf-8"))
    assert spec["cell"] == {"x": 1.0, "fi_kind": "scan"}
    assert plan[0]["key"] == "x=1.0" and plan[0]["cell"] == {"x": 1.0}


def test_a_cell_that_already_says_what_it_is_keeps_it_and_the_key_is_never_part_of_a_name() -> None:
    tagged = in_run.with_kind({"x": 1.0, "fi_kind": "search"}, "scan")
    assert tagged["fi_kind"] == "search"
    assert in_run.with_kind({"x": 1.0, "fi_kind": "other"}, "scan")["fi_kind"] == "scan"
    assert trial_runner.cell_key(tagged) == trial_runner.cell_key({"x": 1.0}) == "x=1.0"


@pytest.mark.asyncio
async def test_a_case_of_a_check_is_called_as_a_check(tmp_path: Path) -> None:
    root = _quest(tmp_path, _DISPATCHING)
    got, why = await trial_runner.run_case(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root,
                                           "code/simulate.py", cell={"x": 1.0}, timeout_s=120)
    assert why == "" and got == {"f": 8.0}


def test_the_static_check_accepts_a_read_of_the_kind_as_supplied_by_fi() -> None:
    protocol = {"grid": {"x": [1, 2]}}
    src = "def run_cell(cell):\n    return {'f': cell['x'] if cell['fi_kind'] == 'search' else 0.0}\n"
    assert grid_keys.problems({"simulate.py": src}, protocol) == []
    bad = src.replace("fi_kind", "fi_mode")
    assert grid_keys.problems({"simulate.py": bad}, protocol)
