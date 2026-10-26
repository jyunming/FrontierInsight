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


def _record(tmp_path: Path, grid: dict, source: str = PAIRED, in_each_run: dict = INSIDE):
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    run = asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), sys.executable, root,
        "code/simulate.py", grid, runs_per_setting=3, base_seed=0, deterministic=False, timeout_s=60, run_id="r1",
        in_each_run=in_each_run))
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


# --- a measurement the trials return under every value of a setting -----------------------------------------------------

ONE_DIFFERENCE = '''
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(int(cell["m"]) * 1000 + trial_id)  # the same random numbers for both procedures
    first, second = rng.random(), rng.random()
    own = first if cell["procedure"] == "one" else second
    return {"diff": second - first, "own": own}
'''
TWO_AXES = {"m": [10, 20], "procedure": ["one", "two"]}


def _two_axes(tmp_path: Path, source: str = ONE_DIFFERENCE):
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    run = asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), sys.executable, root,
        "code/simulate.py", TWO_AXES, runs_per_setting=3, base_seed=0, deterministic=False, timeout_s=60, run_id="r1"))
    trial_runner._save_run(root, "k", run)
    per_cell, _ = trial_runner.recorded_values_by_cell(root)
    return root, per_cell, trial_runner.pooled(per_cell)


def test_a_measurement_returned_the_same_trial_by_trial_under_every_value_of_a_setting_is_found_from_the_record(
        tmp_path: Path) -> None:
    root, _per_cell, _recorded = _two_axes(tmp_path)
    assert trial_runner.copied_settings(root, TWO_AXES, ["diff", "own"]) == {"diff": ["procedure"]}, \
        "a number each procedure computes on its own is not a copy"
    other = tmp_path / "other"
    other.mkdir()
    noisy = ONE_DIFFERENCE.replace('"diff": second - first', '"diff": second - first + (0.5 if cell["procedure"] == "two" else 0.0)')
    root2, _p, _r = _two_axes(other, noisy)
    assert trial_runner.copied_settings(root2, TWO_AXES, ["diff"]) == {}, "values that differ at one value are no copy"


def test_a_copied_measurement_is_owed_once_per_setting_and_a_setting_left_out_is_still_found(tmp_path: Path) -> None:
    root, per_cell, recorded = _two_axes(tmp_path)
    copied = trial_runner.copied_settings(root, TWO_AXES, ["diff", "own"])
    protocol = {"grid": TWO_AXES, "runs_per_setting": 3,
                "metrics": [{"id": "diff", "kind": "mean"}, {"id": "own", "kind": "mean"}]}
    manifest = {"successful_per_cell": {f"m={m},procedure={p}": 3 for m in (10, 20) for p in ("one", "two")}}
    series = trial_runner.recorded_series(root, ["diff", "own"])
    once = {f"m={m}": {"diff_values": series["diff"][f"m={m},procedure=one"]} for m in (10, 20)}
    own = {f"m={m},procedure={p}": {"own_values": series["own"][f"m={m},procedure={p}"]} for m in (10, 20) for p in ("one", "two")}
    honest = {**once, **own}

    def found(result, copied_now):
        return [s for s, _ in run_manifest._value_count_findings(
            protocol, manifest, result, recorded=recorded, recorded_by_cell=per_cell, copied=copied_now)]

    assert found(honest, copied) == []
    assert any("`diff`" in s for s in found(honest, {})), "without the record's measurement it was owed once per procedure"
    left_out = {**honest}
    del left_out["m=20"]
    assert any("`diff`" in s and "m=20" in s for s in found(left_out, copied)), "a setting left out is still found"
    own_once = {**once, **{f"m={m}": {**once[f"m={m}"], "own_values": series["own"][f"m={m},procedure=one"]} for m in (10, 20)}}
    assert any("`own`" in s for s in found(own_once, copied)), "a number per procedure listed once is still short"
    assert run_manifest.differences(protocol, manifest, honest, recorded=recorded, recorded_by_cell=per_cell,
                                    copied=copied) == []


INSIDE_COPY = '''
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(int(cell["m"]) * 1000 + trial_id)
    first, second = rng.random(), rng.random()
    return {"d": {"first_rule": second - first, "second_rule": second - first}, "own": {"first_rule": first, "second_rule": second}}
'''
RULES = {"procedure": ["first_rule", "second_rule"]}


def test_a_copy_under_what_each_run_compares_is_found_and_a_list_per_value_is_still_complete(tmp_path: Path) -> None:
    root, per_cell, recorded = _record(tmp_path, {"m": [10, 20]}, INSIDE_COPY, RULES)
    grid = {"m": [10, 20], **RULES}
    copied = trial_runner.copied_settings(root, grid, ["d", "own"])
    assert copied == {"d": ["procedure"]}
    protocol = {"grid": {"m": [10, 20]}, "in_each_run": RULES, "runs_per_setting": 3,
                "metrics": [{"id": "d", "kind": "mean"}]}
    manifest = {"successful_per_cell": {"m=10": 3, "m=20": 3}}
    series = trial_runner.recorded_series(root, ["d"])["d"]

    def found(result):
        return [s for s, _ in run_manifest._value_count_findings(
            protocol, manifest, result, recorded=recorded, recorded_by_cell=per_cell, copied=copied)]

    once = {f"m={m}": {"d_values": series[f"m={m},procedure=second_rule"]} for m in (10, 20)}
    assert found(once) == [], "one copy per setting"
    per_value = {f"m={m},procedure={p}": {"d_values": series[f"m={m},procedure={p}"]} for m in (10, 20) for p in RULES["procedure"]}
    assert found(per_value) == [], "a copy per value, as the record holds it, is complete too"
    assert run_manifest.differences(protocol, manifest, per_value, recorded=recorded, recorded_by_cell=per_cell,
                                    copied=copied) == []
    one_left_out = {k: v for k, v in per_value.items() if not k.startswith("m=20")}
    assert any("`d`" in s and "m=20" in s for s in found(one_left_out)), "a setting left out of the lists per value is found"
    half = {k: v for k, v in per_value.items() if k != "m=20,procedure=second_rule"}
    assert found(half) == [], "a copy of every setting is there"
    tripled = {**per_value, "m=10,procedure=second_rule": {"d_values": series["m=10,procedure=second_rule"] * 3}}
    assert any("`d`" in s for s in found(tripled)), "the record's values listed three times are more than it holds"
    made_up = {**per_value, "m=10,procedure=second_rule": {"d_values": series["m=10,procedure=second_rule"] + [0.123, 0.456]}}
    assert any("`d`" in s for s in found(made_up)), "numbers FI's trials never returned are still found"


def test_a_setting_whose_record_misses_one_of_its_values_is_no_copy(tmp_path: Path) -> None:
    root, _per_cell, _recorded = _two_axes(tmp_path)
    assert trial_runner.copied_settings(root, {"m": [10, 20], "procedure": ["one", "two", "three"]}, ["diff"]) == {}, \
        "a value of the setting with no trials is not shown to be the same"
    assert trial_runner.copied_settings(root, {"m": [10, 20], "procedure": ["one", "two"]}, ["diff"]) == {"diff": ["procedure"]}
