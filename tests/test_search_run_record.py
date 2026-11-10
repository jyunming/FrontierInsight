"""A study run as a search for the best design keeps the same run record a measurement keeps (``.fi/run_facts.json``).

Without it a repair of a finding could not be run again to see whether it changes a result ("FI has no record of how long
the study took"), and the paper lacked the run time and the settings count. Neutral toy simulation
(tests/test_optimise_runner.py); the search runs through ``run_search`` with the real executor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import paper_facts, rerun_check, run_facts
from core import trial_runner as trials
from tests.test_optimise_runner import _ledger, _protocol, _quest, _run

SCAN = {"x": [-4.0, 0.0, 4.0]}
SMALL = {"starts": 2, "per_start": 6}


def _best(root: Path) -> dict[str, Any]:
    return json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_search_with_a_scan_writes_the_run_record_with_what_fi_measured(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    await _run(root, _protocol(grid=SCAN, search_method="global_then_local", evaluation_budget=SMALL), at_once=2)
    run = run_facts.read(root)["trials_run"]
    summary = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    evaluations = [r for r in _ledger(root) if r.get("event") == "evaluation" and r.get("stage") != "scan"]
    # the scan: the settings it ran, their trials, their wall time, the most that ran together
    assert run["cells"] == len(summary["cells"]) > 0 and run["trials"] >= run["cells"]
    # (the trials' own times are kept to 0.01 s, and the toy's trials can add up to less than that on a fast machine)
    assert run["wall_s"] > 0 and run["settings_s"] > 0 and run["at_once"] >= 1
    assert isinstance(run["trial_sum_s"], float) and run["trial_sum_s"] >= 0
    # the search: how many designs it evaluated and their own times, from FI's ledger
    assert run["evaluations"] == len(evaluations) > 0
    assert run["evaluations_s"] == pytest.approx(sum(r["elapsed_s"] for r in evaluations), abs=0.05)
    # no value is made up: the search has no stage and no settings sharing a process
    assert "stage" not in run and "per_process" not in run
    # the search's own time no longer holds the scan, which the run record counts
    assert _best(root)["time"]["scan_seconds"] == pytest.approx(run["wall_s"], abs=0.01)


@pytest.mark.asyncio
async def test_a_rerun_of_a_search_study_has_a_bound(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    assert rerun_check.time_allowed(root) is None
    await _run(root, _protocol(grid=SCAN, search_method="global_then_local", evaluation_budget=SMALL))
    run = run_facts.read(root)["trials_run"]
    allowed = rerun_check.time_allowed(root)
    assert allowed is not None
    serial = rerun_check.SERIAL_FACTOR * (run["settings_s"] + run["evaluations_s"])
    whole = run_facts.run_seconds(root)
    assert allowed == pytest.approx(max(rerun_check.FLOOR_S, rerun_check.SLACK * max(whole, serial)))
    assert rerun_check.time_allowed(root, 5) == 5.0


@pytest.mark.asyncio
async def test_the_paper_states_the_search_studys_run_time_and_settings_once(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    await _run(root, _protocol(grid=SCAN, search_method="global_then_local", evaluation_budget=SMALL))
    run = run_facts.read(root)["trials_run"]
    wall, least = paper_facts.measured_run_time(root)
    # the study is the scan and the search together: the search's time already holds the scan, so it is counted once
    assert wall == pytest.approx(_best(root)["time"]["seconds"], abs=0.02)
    assert least is not None and least <= wall + 0.02
    said = paper_facts.facts(root)["run_time"].text
    assert paper_facts._plain(wall) in said
    assert paper_facts._cells_run(root) == f"{run['cells']:,}"


@pytest.mark.asyncio
async def test_a_search_without_a_scan_keeps_its_evaluations_and_not_an_earlier_runs_trials(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    run_facts.record_trials(root, key="old", wall_s=900.0, trials=50, cells=10, trial_sum_s=800.0, settings_s=850.0)
    await _run(root, _protocol(search_method="global_then_local", evaluation_budget=SMALL))
    run = run_facts.read(root)["trials_run"]
    assert "key" not in run and "settings_s" not in run and run["evaluations"] > 0
    assert rerun_check.time_allowed(root) is not None
    assert run_facts.run_seconds(root) == pytest.approx(_best(root)["time"]["seconds"], abs=0.02)


def _cell(key: str, wall: float, durations: list[float]) -> trials.CellRun:
    return trials.CellRun(key=key, cell={"x": 1}, planned=len(durations), rows=[{"duration_s": d, "status": "ok"} for d in durations],
                          wall_s=wall)


def test_a_measurements_run_record_is_what_it_was(tmp_path: Path) -> None:
    run = trials.TrialRun(cells=[_cell("a", 3.0, [1.0, 1.5]), _cell("b", 4.0, [2.0])], ledger_path=tmp_path / "l", summary_path=tmp_path / "s",
                          most_at_once=2)
    trials._keep_run_time(tmp_path, "k", run, 7.5)
    assert run_facts.read(tmp_path) == {"trials_run": {"key": "k", "wall_s": 7.5, "trials": 3, "cells": 2, "trial_sum_s": 4.5,
                                                       "at_once": 2, "settings_s": 7.0}}
    # the same study's runs kept together add up
    other = tmp_path / "two"
    trials.keep_runs_time(other, "k", [run, run], 15.0)
    two = run_facts.read(other)["trials_run"]
    assert (two["trials"], two["cells"], two["trial_sum_s"], two["settings_s"], two["wall_s"]) == (6, 4, 9.0, 14.0, 15.0)


def test_a_continued_search_that_took_its_scan_from_the_earlier_one_counts_its_own_time(tmp_path: Path) -> None:
    """A search that continues an earlier one takes the coarse scan from it (the scan ran before this call's time began).
    Old code: the earlier scan's seconds were taken off this call's time, so a 100 s scan and a 20 s continuation gave a
    study of 100 s; the scan is counted once, by the run record, and the continuation's own 20 s stand."""
    from core import optimise

    run_facts.record_trials(tmp_path, key="k", wall_s=100.0, trials=3, cells=3, trial_sum_s=90.0)
    run_facts.record_evaluations(tmp_path, evaluations=4, evaluations_s=30.0)
    scan_s = optimise.keep_search_time(tmp_path, [], scan_ran=False, scan_s=0.0, scan_replayed=True)
    assert scan_s == 0.0
    assert run_facts.read(tmp_path)["trials_run"]["wall_s"] == 100.0, "the earlier scan's record is kept"
    best = tmp_path / "results" / "best_design.json"
    best.parent.mkdir(parents=True)
    best.write_text(json.dumps({"time": {"seconds": 20.0}}), encoding="utf-8")
    assert run_facts.run_seconds(tmp_path) == pytest.approx(120.0)
    # known positive: a scan this call ran is inside its time and is said apart
    assert optimise.keep_search_time(tmp_path, [], scan_ran=True, scan_s=7.0, scan_replayed=False) == 7.0
