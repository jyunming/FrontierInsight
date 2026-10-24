"""The run-time estimate counts what the run will really do (core/run_estimate.py, core/run_checks.py, core/optimise.py):
a spread of zero is a measurement, a sub-study is timed at its costliest-looking setting, and the search's check at finer
settings shares the one time limit."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from core import optimise_search as search
from core import run_estimate as est
from core.optimise import OptimisationRunner  # noqa: F401  (the import is the check that the module loads)

MEAN_PLAN = {"metrics": [{"id": "y", "kind": "mean"}], "precision": {"target_half_width": 0.5, "metric": "y"}}


def test_a_spread_of_zero_is_not_a_missing_spread() -> None:
    """Old code: two timed trials that agreed (spread exactly 0) made ``runs_floor`` return the plan's own runs and
    ``precision_known`` False, so a run only slightly too long was stopped. Now zero is a measured spread: two runs."""
    assert est.runs_floor(MEAN_PLAN, 400, {"y": 0.0}) == 2
    assert est.runs_floor({**MEAN_PLAN, "min_runs_per_setting": 30}, 400, {"y": 0.0}) == 30
    assert est.precision_known(MEAN_PLAN, {"y": 0.0}) is True
    # known positives: a missing spread still forbids any cut, and a real spread still sets the floor
    assert est.runs_floor(MEAN_PLAN, 400, {}) == 400 and est.runs_floor(MEAN_PLAN, 400, None) == 400
    assert est.precision_known(MEAN_PLAN, {}) is False
    assert est.runs_floor(MEAN_PLAN, 400, {"y": 1.0}) == 16


def test_a_zero_spread_run_that_is_a_little_too_long_is_cut_not_stopped() -> None:
    probes = [est.Probe(None, None, {"m": 20}, 0.5 + 1000 * 0.1, fixed_s=0.5, per_trial_s=0.1, timed_trials=50)]
    fit = est.fit_runs({"m": [20]}, 1000, False, probes, MEAN_PLAN, limit_s=30.0, topic_limit=False, spreads={"y": 0.0})
    assert fit is not None and 2 <= fit.runs < 1000
    none = est.fit_runs({"m": [20]}, 1000, False, probes, MEAN_PLAN, limit_s=30.0, topic_limit=False, spreads={})
    assert none is None


PARTS_GRID = {"kind": ["a", "b"], "n": [1, 10, 100], "m": [1, 2]}
PARTS = [{"name": "ns", "vary": ["n"], "hold": {"kind": "a", "m": 1}}, {"name": "ms", "vary": ["m"], "hold": {"kind": "b", "n": 10}}]


def _grid() -> search.StudyGrid:
    return search.study_grid({"grid": PARTS_GRID, "sub_studies": PARTS})


def test_the_costliest_looking_setting_of_each_part_is_the_largest_number() -> None:
    """Old code timed a part only at its middle value (n = 10), so a part whose cost grows with n = 100 was costed cheap."""
    cells = dict(search.costly_part_cells(_grid()))
    assert cells["ns"] == {"kind": "a", "n": 100, "m": 1}
    assert cells["ms"] == {"kind": "b", "n": 10, "m": 2}
    assert dict(search.part_cells(_grid()))["ns"]["n"] == 10
    assert search.costly_part_cells({"kind": ["a", "b"]}) == []


def test_a_part_is_costed_at_the_dearer_of_its_timed_settings() -> None:
    grid = _grid()
    base = est.Probe(None, None, {"kind": "b", "n": 10, "m": 1}, 1.0)
    cheap = est.Probe(est.PART, 0, {"kind": "a", "n": 10, "m": 1}, 1.0)
    dear = est.Probe(est.PART, 0, {"kind": "a", "n": 100, "m": 1}, 50.0)
    other = est.Probe(est.PART, 1, {"kind": "b", "n": 10, "m": 1}, 1.0)
    only_cheap = est.estimate(grid, 1, True, [base, cheap, other])
    both = est.estimate(grid, 1, True, [base, cheap, dear, other])
    assert only_cheap is not None and both is not None
    assert both.seconds >= 50.0 * 3 and both.seconds > only_cheap.seconds * 10
    assert both.parts_timed == 2 and not both.at_least


def test_the_timing_includes_the_costly_setting_of_each_part() -> None:
    timed: list[dict] = []

    async def fake_time_one(*args, **kwargs):  # noqa: ANN001, ANN003
        cell = args[4]
        timed.append(dict(cell))
        return est.Probe(None, None, dict(cell), 1.0 + cell.get("n", 0) / 10.0, fixed_s=1.0, per_trial_s=0.0), ""

    real = est._time_one
    est._time_one = fake_time_one
    try:
        probes, _why = asyncio.run(est.measure(None, "python", __import__("pathlib").Path("."), "sim.py", _grid(), runs=1,
                                               deterministic=True, limit_s=1e6, budget_s=1e6))
    finally:
        est._time_one = real
    assert probes is not None
    assert {"kind": "a", "n": 100, "m": 1} in timed, "the part's largest-number setting was timed"
    assert any(p.axis == est.PART and p.cell.get("n") == 100 for p in probes)


def test_the_finer_check_counts_toward_the_one_limit() -> None:
    """Old code gave the check at finer settings a whole limit of its own after the search; now it gets what is left."""
    import inspect

    from core import optimise

    source = inspect.getsource(optimise.OptimisationRunner.execute)
    assert "check_left" in source and "timeout_s, env)" not in source.split("self._check(")[1].split("\n")[0]
    assert time.monotonic() > 0
