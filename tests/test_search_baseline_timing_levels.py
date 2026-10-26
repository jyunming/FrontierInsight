"""The search's baseline is timed with the values each run compares inside it, as the scan's timing is."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.test_run_checks import _SCAN_BLOCK, _search_engine

_SIMULATION = (
    "def run_cell(cell):\n"
    "    return {'f': {f'p={v}': cell['x'] ** 2 + v for v in cell['p']}}\n"
)


def _block() -> dict[str, Any]:
    block = copy.deepcopy(_SCAN_BLOCK)
    block["grid"] = {"axis0": [0, 1]}
    block["in_each_run"] = {"p": [1, 2, 3]}
    block["evaluation_budget"]["seconds_per_evaluation"] = 20
    return block


@pytest.mark.asyncio
async def test_the_baseline_is_timed_with_the_values_compared_inside_each_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                              caplog: pytest.LogCaptureFixture) -> None:
    eng, runner = await _search_engine(tmp_path, _block())
    (eng.quest_root / "code" / "simulate.py").write_text(_SIMULATION, encoding="utf-8")
    from core import run_estimate

    seen: list[Any] = []
    real = run_estimate._time_one

    async def spy(*a: Any, **k: Any) -> Any:
        seen.append(k.get("in_each_run"))
        return await real(*a, **k)

    monkeypatch.setattr(run_estimate, "_time_one", spy)
    caplog.set_level("WARNING")
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert "timed the search's baseline" not in caplog.text, caplog.text
    assert patch is None or "returned non-numbers" not in str(patch)
    assert {"p": [1, 2, 3]} in seen, seen


@pytest.mark.asyncio
async def test_a_mapping_keyed_by_a_value_the_run_was_not_given_still_fails(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    eng, runner = await _search_engine(tmp_path, _block())
    (eng.quest_root / "code" / "simulate.py").write_text(
        "def run_cell(cell):\n    return {'f': {'p=1': 1.0, 'p=2': 2.0, 'p=99': 3.0}}\n", encoding="utf-8")
    caplog.set_level("WARNING")
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert "timed the search's baseline" in caplog.text or patch is not None
    assert "not one of the values the run was given" in caplog.text + str(patch)
