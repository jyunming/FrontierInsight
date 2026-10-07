"""The run is timed on a few real trials before it starts (core/run_estimate.py): the arithmetic, the timing of a toy
simulation whose per-trial time is set by its settings, and the plain words."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core import run_estimate as est
from core import trial_runner
from core.execution import ExecutionResult, SharedInterpreterExecutor


def test_a_length_of_time_is_said_as_a_person_says_it() -> None:
    assert est.plain_duration(1) == "1 second" and est.plain_duration(45) == "45 seconds"
    assert est.plain_duration(600) == "10 minutes" and est.plain_duration(3 * 3600 + 20 * 60) == "3 h 20 min"
    assert est.plain_duration(3600) == "1 h" and est.plain_duration(5400) == "1 h 30 min"
    assert est.plain_duration(13 * 3600) == "13 h" and est.plain_duration(60 * 3600) == "2 days 12 h"


def test_the_settings_timed_are_the_base_and_each_setting_at_its_ends() -> None:
    grid = {"mesh": [8, 16, 32, 64], "damping": [0.1, 0.5], "mass": [2.0]}
    cells = est.probe_cells(grid)
    assert cells[0][0] is None and cells[0][2] == {"mesh": 32, "damping": 0.5, "mass": 2.0}
    timed = [(axis, c) for axis, _i, c in cells[1:]]
    assert ("mesh", {"mesh": 8, "damping": 0.5, "mass": 2.0}) in timed
    assert ("mesh", {"mesh": 64, "damping": 0.5, "mass": 2.0}) in timed
    assert ("damping", {"mesh": 32, "damping": 0.1, "mass": 2.0}) in timed
    assert len(cells) == 1 + 2 + 1, "a setting with one value is not timed; the base is a value of the other"
    assert est.probe_cells({}) == [(None, None, {})]


def test_the_total_is_the_base_times_the_mean_ratio_of_each_setting() -> None:
    grid = {"mesh": [8, 16, 32], "damping": [0.1, 0.5]}
    # Base is mesh 16 / damping 0.5 (the middle value; the upper of two). mesh 8 costs half, mesh 32 four times.
    probes = [est.Probe(None, None, {"mesh": 16, "damping": 0.5}, 10.0),
              est.Probe("mesh", 0, {"mesh": 8, "damping": 0.5}, 5.0),
              est.Probe("mesh", 2, {"mesh": 32, "damping": 0.5}, 40.0),
              est.Probe("damping", 0, {"mesh": 16, "damping": 0.1}, 20.0)]
    e = est.estimate(grid, 5, False, probes)
    assert e is not None and e.complete and e.cells == 6 and e.trials == 30
    # mesh mean ratio (0.5 + 1 + 4) / 3; damping mean ratio (2 + 1) / 2
    assert abs(e.seconds - 6 * 10.0 * ((0.5 + 1 + 4) / 3) * 1.5) < 1e-9
    assert e.ratios["mesh"] == {"8": 0.5, "32": 4.0}


def test_a_value_between_two_timed_ones_is_read_between_them() -> None:
    grid = {"mesh": [1, 2, 3, 4, 5]}
    probes = [est.Probe(None, None, {"mesh": 3}, 4.0), est.Probe("mesh", 0, {"mesh": 1}, 1.0),
              est.Probe("mesh", 4, {"mesh": 5}, 16.0)]
    e = est.estimate(grid, 1, True, probes)
    assert e is not None
    ratios = est._interpolated({0: 0.25, 2: 1.0, 4: 4.0}, 5)
    assert abs(ratios[1] - 0.5) < 1e-9 and abs(ratios[3] - 2.0) < 1e-9, "in proportion, on a logarithmic scale"
    assert e.seconds > 5 * 4.0 * 0.99


def test_an_estimate_needs_the_base_and_fits_with_a_margin() -> None:
    assert est.estimate({"n": [1, 2]}, 1, True, []) is None
    e = est.estimate({"n": [1]}, 4, False, [est.Probe(None, None, {"n": 1}, 10.0)])
    assert e is not None and e.seconds == 10.0
    assert est.fits(e, 15.0) and not est.fits(e, 14.9)
    assert est.known_too_long(e, 14.9) and not est.known_too_long(e, 15.0)


def test_the_progress_line_says_settings_done_and_time_left() -> None:
    assert est.progress_line(3, 96, 30, 960, 600) == "3 of 96 settings done, about 5 h 10 min left"
    assert est.progress_line(0, 96, 0, 960, 600) == "0 of 96 settings done"
    assert est.progress_line(1, 1, 5, 5, 10) == "1 of 1 setting done"
    assert est.progress_line(0, 0, 0, 0, 0) == ""


def test_the_request_to_the_plan_has_the_measured_numbers_and_no_expected_value() -> None:
    grid = {"mesh": [8, 16, 32]}
    probes = [est.Probe(None, None, {"mesh": 16}, 100.0), est.Probe("mesh", 0, {"mesh": 8}, 50.0),
              est.Probe("mesh", 2, {"mesh": 32}, 400.0)]
    e = est.estimate(grid, 1000, False, probes)
    assert e is not None
    text = est.request(e, 3600.0, 1000, False)
    assert "3 settings" in text and "1000 runs" in text and "`mesh` = 32 costs 4 times" in text
    assert "protocol.runs_per_setting" in text and "protocol.grid" in text
    assert "not a check, a threshold, a tolerance" in text and "expected" not in text.lower()


# --- timing a toy simulation ------------------------------------------------------------------------------------------

TOY = '''
import time

def run_trial(cell, trial, seed):
    time.sleep(0.05 * cell["n"])
    return {"y": float(cell["n"]) * 2.0 + (seed % 3)}
'''
TOY_CELL = '''
import time

def run_cell(cell):
    time.sleep(0.2 * cell["n"])
    return {"y": float(cell["n"])}
'''


def _quest(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return root


def _measure(root: Path, grid: dict, runs: int, deterministic: bool, limit_s: float = 600.0, executor=None):  # noqa: ANN001, ANN201
    return asyncio.run(est.measure(
        executor or SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", grid,
        runs=runs, deterministic=deterministic, limit_s=limit_s, budget_s=60.0))


def test_a_stochastic_simulation_is_timed_on_two_trials_and_its_cost_follows_its_settings(tmp_path: Path) -> None:
    root = _quest(tmp_path, TOY)
    grid = {"n": [1, 2, 4]}
    probes, why = _measure(root, grid, 20, False)
    assert why == "" and probes is not None and len(probes) == 3
    by = {p.cell["n"]: p.seconds for p in probes}
    # 20 runs of 0.05 * n seconds each, plus the start of the process: n=4 costs about twice n=2
    assert 3.5 < by[4] < 6.0 and 1.7 < by[2] < 3.0 and by[4] > 1.5 * by[2]
    e = est.estimate(grid, 20, False, probes)
    assert e is not None and e.complete and e.cells == 3 and e.trials == 60
    assert 5.0 < e.seconds < 14.0, "about 20 x 0.05 x (1 + 2 + 4) = 7 s"


def test_nothing_the_timing_ran_is_kept_or_counted_as_a_result(tmp_path: Path) -> None:
    root = _quest(tmp_path, TOY)
    _measure(root, {"n": [1, 2]}, 4, False)
    assert not (root / "raw").exists(), "no ledger, no summary"
    assert not (root / ".fi" / "trials" / "pilot.json").exists() and not (root / ".fi" / "trials" / "pilot.out.jsonl").exists()


def test_a_simulation_run_once_per_setting_is_timed_as_one_call(tmp_path: Path) -> None:
    root = _quest(tmp_path, TOY_CELL)
    probes, why = _measure(root, {"n": [1, 3]}, 1, True)
    assert why == "" and probes is not None
    by = {p.cell["n"]: p.seconds for p in probes}
    assert 0.55 < by[3] < 1.5 and by[3] > 1.8 * by[1]


def test_a_simulation_that_cannot_be_timed_is_left_to_the_real_runs_repair(tmp_path: Path) -> None:
    root = _quest(tmp_path, "def run_trial(cell, trial, seed):\n    raise RuntimeError('broken')\n")
    probes, why = _measure(root, {"n": [1, 2]}, 3, False)
    assert probes is None and why == "a trial failed"
    root2 = _quest(tmp_path / "b", "def not_the_entry():\n    return 1\n")
    probes, why = _measure(root2, {"n": [1]}, 3, False)
    assert probes is None and "could not be loaded" in why


class _NeverFinishes:
    """An executor whose process runs out of its allowance without reporting a trial."""

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
        self.calls += 1
        return ExecutionResult(returncode=-1, stdout="", stderr="", duration_s=float(timeout_s), timed_out=True)


def test_a_trial_that_runs_out_of_its_allowance_is_a_lower_bound_and_the_rest_is_not_timed(tmp_path: Path) -> None:
    root = _quest(tmp_path, TOY)
    ex = _NeverFinishes()
    grid = {"n": [1, 2, 4]}
    probes, why = _measure(root, grid, 1000, False, limit_s=3600.0, executor=ex)
    assert why == "" and probes is not None and len(probes) == 1 and ex.calls == 1
    e = est.estimate(grid, 1000, False, probes)
    assert e is not None and e.at_least and not e.complete
    assert est.known_too_long(e, 3600.0)
    assert est.says(e, 3600.0).startswith("FI estimates the experiment takes at least ")
