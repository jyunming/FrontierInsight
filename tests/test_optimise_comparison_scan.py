"""A search for the best design can also compare the study across settings of a condition it holds fixed.

The block's own scan (``optimisation.grid``) takes a design variable or a condition listed in ``optimisation.fixed``. The
scan measures the study at every value of the compared condition and keeps all of it in the trial record the analysis
plots; only the cells at the value the plan holds fixed are designs evaluated under the search's own conditions, so only
those can start the search. Neutral toy simulation (tests/test_optimise_runner.py)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import optimisation_plan as op
from core import optimise_search as osearch
from tests.test_optimise_runner import BLOCK, _block, _ledger, _protocol, _quest, _run


def test_a_scan_may_compare_a_condition_the_plan_holds_fixed_and_must_say_the_value_it_is_held_at() -> None:
    fixed, why = op.normalize({**BLOCK, "grid": {"scale": [0.5, 1.0, 2.0], "x": [-1.0, 1.0]}})
    assert fixed is not None, why
    _, why = op.normalize({**BLOCK, "grid": {"unlisted": [1.0, 2.0]}})
    assert why and "unlisted" in why and "optimisation.fixed" in why
    _, why = op.normalize({**BLOCK, "grid": {"scale": [1.0, "wide"]}})
    assert why and "all numbers or all names" in why
    _, why = op.normalize({**BLOCK, "grid": {"x": [99.0]}})
    assert why and "inside that variable's range" in why, "a design variable's scan stays inside its range"


def test_the_scan_grid_crosses_the_compared_condition_and_only_the_cells_at_the_held_value_are_designs() -> None:
    block = _block(grid={"scale": [0.5, 1.0, 2.0], "x": [-1.0, 1.0]})
    grid = osearch.scan_grid(block)
    assert grid["scale"] == [0.5, 1.0, 2.0] and grid["x"] == [-1.0, 1.0] and grid["y"] == [0.0]
    cells = [{"cell": {"scale": s, "x": x, "y": 0.0, "mesh": 0.5}, "ok": True, "failed": 0, "metrics": {"f": {"values": [s * x]}}}
             for s in (0.5, 1.0, 2.0) for x in (-1.0, 1.0)]
    rows = osearch.scan_rows({"cells": cells}, block)
    assert [r["design"] for r in rows] == [{"x": -1.0, "y": 0.0}, {"x": 1.0, "y": 0.0}]
    assert all(r["values"]["f"] == r["design"]["x"] for r in rows), "the cells at the held value (1.0), no other"


def test_a_scan_that_leaves_the_condition_alone_is_as_before() -> None:
    block = _block(grid={"x": [-1.0, 1.0]})
    assert osearch.scan_grid(block)["scale"] == [1.0]
    cells = [{"cell": {"scale": 1.0, "x": x, "y": 0.0}, "ok": True, "failed": 0, "metrics": {"f": {"values": [x]}}} for x in (-1.0, 1.0)]
    assert len(osearch.scan_rows({"cells": cells}, block)) == 2


@pytest.mark.asyncio
async def test_the_comparison_is_measured_and_kept_but_never_starts_the_search(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    run = await _run(root, _protocol(grid={"scale": [1.0, 3.0], "x": [-4.0, 0.0, 4.0]}, search_method="global_then_local",
                                     evaluation_budget={"starts": 2, "per_start": 10}))
    trials: dict[str, Any] = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    scales = sorted({c["cell"]["scale"] for c in trials["cells"]})
    assert scales == [1.0, 3.0] and len(trials["cells"]) == 6, "the study is measured at every compared value"
    scan = [r for r in _ledger(root) if r.get("event") == "evaluation" and r["stage"] == "scan"]
    # three cells at the held value, one of them the baseline (evaluated first, never twice): the cells at 3.0 are no design
    assert [r["design"]["x"] for r in scan] == [-4.0, 4.0], "only the cells at the held value are designs of the search"
    assert run.record["evaluations"]["scan"] == 2
