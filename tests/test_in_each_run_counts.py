"""Counting what the analysis lists against FI's record when runs compare values inside them (``protocol.in_each_run``):
a study with no grid at all, a number per value listed under two settings, and the cases the plan refuses."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from core import in_run, plan, run_manifest, trial_runner
from core.execution import SharedInterpreterExecutor

from tests.test_in_each_run import INSIDE, PAIRED, _executor, _run

ONLY_INSIDE = PAIRED.replace('found = rng.random() * cell["m"]', "found = rng.random() * 10")


def _record(tmp_path: Path, grid: dict, source: str = PAIRED):
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    run = asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), sys.executable, root,
        "code/simulate.py", grid, runs_per_setting=3, base_seed=0, deterministic=False, timeout_s=60, run_id="r1",
        in_each_run=INSIDE))
    trial_runner._save_run(root, "k", run)
    per_cell, _ = trial_runner.recorded_values_by_cell(root)
    return root, per_cell, trial_runner.pooled(per_cell)


def _findings(protocol, manifest, result, recorded, per_cell):
    return [s for s, _ in run_manifest._value_count_findings(
        protocol, manifest, result, recorded=recorded, recorded_by_cell=per_cell)]


def test_a_study_with_nothing_but_what_each_run_compares_is_still_counted(tmp_path: Path) -> None:
    _root, per_cell, recorded = _record(tmp_path, {}, ONLY_INSIDE)
    protocol = {"grid": {}, "in_each_run": INSIDE, "runs_per_setting": 3,
                "metrics": [{"id": "power_diff", "kind": "mean"}]}
    manifest = {"successful_per_cell": {"": 3}}
    honest = {"power_diff_values": [0.1, 0.2, 0.3]}
    assert _findings(protocol, manifest, honest, recorded, per_cell) == []
    inflated = {"power_diff_values": [0.1] * 30}
    assert any("power_diff" in s for s in _findings(protocol, manifest, inflated, recorded, per_cell))


def test_a_number_per_value_listed_under_two_settings_is_one_account_in_the_check_and_in_the_difference(
        tmp_path: Path) -> None:
    _root, per_cell, recorded = _record(tmp_path, {"m": [10, 20]})
    protocol = {"grid": {"m": [10, 20]}, "in_each_run": INSIDE, "runs_per_setting": 3,
                "metrics": [{"id": "power", "kind": "mean"}]}
    manifest = {"successful_per_cell": {"m=10": 3, "m=20": 3}}
    honest = {key: {"power_values": [0.1] * 6} for key in ("m=10", "m=20")}
    assert _findings(protocol, manifest, honest, recorded, per_cell) == []
    assert run_manifest.differences(protocol, manifest, honest, recorded=recorded, recorded_by_cell=per_cell) == []
    greedy = {key: {"power_values": [0.1] * 12} for key in ("m=10", "m=20")}
    assert run_manifest.differences(protocol, manifest, greedy, recorded=recorded, recorded_by_cell=per_cell)


def test_a_subset_mean_cannot_be_combined_with_what_each_run_compares() -> None:
    both = {"grid": {"m": [20]}, "in_each_run": INSIDE,
            "metrics": [{"id": "size", "kind": "mean", "given": "major", "estimand": "size of a major outbreak", "unit": "run"},
                {"id": "major", "kind": "proportion", "estimand": "share that are major", "unit": "run"}]}
    ok, why = plan.normalize_protocol(both)
    assert ok is None and "`given`" in why and "in_each_run" in why
    without = {"grid": {"m": [20]}, "in_each_run": INSIDE, "metrics": [{"id": "size", "kind": "mean", "estimand": "mean size", "unit": "run"}]}
    assert plan.normalize_protocol(without)[1] is None
    alone = {"grid": {"m": [20]}, "metrics": [{"id": "size", "kind": "mean", "given": "major", "estimand": "size", "unit": "run"},
                                               {"id": "major", "kind": "proportion", "estimand": "share", "unit": "run"}]}
    assert plan.normalize_protocol(alone)[1] is None


def test_the_same_value_twice_in_what_each_run_compares_is_refused_plainly() -> None:
    why = plan.normalize_protocol({"grid": {"m": [1, 2]}, "in_each_run": {"procedure": ["bh", "bh"]}})[1]
    assert why and "same value twice" in why
    assert "same value twice" in plan.normalize_protocol({"in_each_run": {"T": [1, 1.0]}})[1]


def test_a_list_for_several_things_compared_inside_a_run_asks_for_a_mapping() -> None:
    levels = in_run.level_cells({"procedure": ["a", "b"], "T": [300.0, 310.0]})
    _got, why = in_run.read_returned({"k": [1.0, 2.0, 3.0, 4.0]}, levels, "run_trial")
    assert "only read when one thing is compared" in why and "procedure=a,T=300.0" in why
    assert "non-numbers" not in why
