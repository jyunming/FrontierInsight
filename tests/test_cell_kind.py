"""FI tells the code what it is asking for (``FI_CELL_KIND``: a design the search evaluates, a row of the scan or the
comparison grid, a case of one of the plan's checks) and leaves the cell exactly the study's settings. Neutral toy study."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import engine, oracle_check, optimise, optimise_search, trial_runner
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
    "import os\n"
    "def run_cell(cell):\n"
    "    kind = os.environ.get('FI_CELL_KIND', 'check')\n"
    "    if kind == 'search':\n"
    "        return {'f': (cell['x'] - 1.0) ** 2}\n"
    "    return {'f': {'scan': 7.0, 'check': 8.0}[kind]}\n"
)
_EXPANDING = (
    "def solve(x, mode):\n"
    "    return (x - 1.0) ** 2\n"
    "def run_cell(cell):\n"
    "    return {'f': solve(**cell)}\n"
)


def _quest(tmp_path: Path, source: str) -> Path:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return tmp_path


def _executor() -> SharedInterpreterExecutor:
    return SharedInterpreterExecutor(python_version="3.11")


async def _search_objective(root: Path, x: float) -> float:
    cell = optimise_search.cell_of(_BLOCK, {"x": x})
    got, _err, _load = await optimise._evaluate(
        _executor(), sys.executable, root, "code/simulate.py", "run_cell", cell,
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


@pytest.mark.asyncio
async def test_code_that_expands_the_cell_into_a_functions_arguments_still_runs(tmp_path: Path) -> None:
    # The cell holds only the study's settings, so an added key cannot break ``solve(**cell)``.
    root = _quest(tmp_path, _EXPANDING)
    assert await _search_objective(root, 3.0) == 4.0


def test_a_scan_row_is_planned_as_a_scan_and_the_cell_the_code_gets_is_only_the_settings(tmp_path: Path) -> None:
    root = _quest(tmp_path, _DISPATCHING)
    folder = root / ".fi" / "trials"
    folder.mkdir(parents=True)
    plan = trial_runner._plan(root, "code/simulate.py", {"x": [1.0, 2.0]}, runs_per_setting=1, base_seed=0,
                              deterministic=True, folder=folder, out_name="c{index}.out.jsonl")
    spec = json.loads((root / plan[0]["spec"]).read_text(encoding="utf-8"))
    assert spec["cell"] == {"x": 1.0} and spec["kind"] == "scan"
    assert plan[0]["key"] == "x=1.0" and plan[0]["cell"] == {"x": 1.0}


def test_a_protocol_setting_of_any_name_is_only_a_setting_and_never_changes_what_the_call_is_for(tmp_path: Path) -> None:
    root = _quest(tmp_path, _DISPATCHING)
    folder = root / ".fi" / "trials"
    folder.mkdir(parents=True)
    plan = trial_runner._plan(root, "code/simulate.py", {"x": [2.0], "__fi_kind__": ["search"]}, runs_per_setting=1,
                              base_seed=0, deterministic=True, folder=folder, out_name="c{index}.out.jsonl")
    spec = json.loads((root / plan[0]["spec"]).read_text(encoding="utf-8"))
    assert spec["kind"] == "scan" and spec["cell"] == {"x": 2.0, "__fi_kind__": "search"}
    assert plan[0]["cell"] == {"x": 2.0, "__fi_kind__": "search"}


@pytest.mark.asyncio
@pytest.mark.parametrize("written", [None, "bogus"])
async def test_the_harness_never_takes_the_kind_from_the_callers_environment(tmp_path: Path, written: str | None) -> None:
    root = _quest(tmp_path, _DISPATCHING)
    folder = root / ".fi" / "trials"
    folder.mkdir(parents=True)
    plan = trial_runner._plan(root, "code/simulate.py", {"x": [2.0]}, runs_per_setting=1, base_seed=0,
                              deterministic=True, folder=folder, out_name="c{index}.out.jsonl")
    spec_path = root / plan[0]["spec"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec.pop("kind")
    if written is not None:
        spec["kind"] = written
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    (root / trial_runner.HARNESS_PATH).write_text(trial_runner.HARNESS_SOURCE, encoding="utf-8")
    await _executor().execute([sys.executable, trial_runner.HARNESS_PATH.as_posix(), plan[0]["spec"], plan[0]["out"]],
                              cwd=root, timeout_s=120, env={"FI_CELL_KIND": "search"})
    row = json.loads((root / plan[0]["out"]).read_text(encoding="utf-8").splitlines()[0])
    assert row["values"] == {"f": 8.0}  # a check, not the search the caller's environment said


def test_what_a_cell_is_for_is_kept_beside_the_cell_in_the_capture_record_and_the_cell_has_no_mark(tmp_path: Path) -> None:
    root = tmp_path
    (root / "code" / "pkg").mkdir(parents=True)
    (root / "code" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "code" / "pkg" / "model.py").write_text("def f(gap):\n    return gap\n", encoding="utf-8")
    (root / "code" / "simulate.py").write_text(
        "from pkg import model\ndef run_cell(cell):\n    return {'y': model.f(cell['gap'])}\n", encoding="utf-8")

    async def run(argv: list[str]) -> Any:
        import subprocess
        done = subprocess.run([sys.executable, "-B", *argv], cwd=root, capture_output=True, text=True, timeout=120)
        return done.returncode, done.stdout, done.stderr

    import asyncio

    from core import second_impl
    asyncio.run(second_impl.capture_inputs(run, root, "pkg", ["f"], [{"gap": 3.0}, {"gap": 4.0}], deterministic=True,
                                           kinds=["search", "scan"]))
    record = second_impl.read_captured(root)
    assert record["cells"] == [{"gap": 3.0}, {"gap": 4.0}] and record["kinds"] == ["search", "scan"]


@pytest.mark.asyncio
async def test_a_case_of_a_check_is_called_as_a_check(tmp_path: Path) -> None:
    root = _quest(tmp_path, _DISPATCHING)
    got, why = await trial_runner.run_case(_executor(), sys.executable, root, "code/simulate.py", cell={"x": 1.0},
                                           timeout_s=120)
    assert why == "" and got == {"f": 8.0}


@pytest.mark.asyncio
async def test_a_replay_of_a_scan_row_is_a_scan_so_a_deterministic_scan_is_found_repeatable(tmp_path: Path) -> None:
    root = _quest(tmp_path, _DISPATCHING)
    record = root / trial_runner.RUN_RECORD
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"cells": [{"cell": {"x": 2.0}, "key": "x=2.0", "rows": [
        {"status": "ok", "duration_s": 0.1, "values": {"f": 7.0}}]}]}), encoding="utf-8")
    same, why = await trial_runner.run_cell_again(_executor(), sys.executable, root, "code/simulate.py", timeout_s=120)
    assert (same, why) == (True, "")


def test_every_contract_the_code_is_written_under_names_the_variable() -> None:
    assert "FI_CELL_KIND" in engine._SPLIT_PROTOCOL and "FI_CELL_KIND" in engine._SEARCH_PROTOCOL
    assert "FI_CELL_KIND" in oracle_check._TRIAL_CONTRACT
