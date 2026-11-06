"""A check's case names one value of a factor each run compares inside it; the run is given every value of it, as in the
study. A run that returns one plain number for the measure gives no way to tell which value the number belongs to: FI says
so (the case could not be run) instead of reading that number as the value the case names. Neutral made-up study; no model
is called."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core import in_run, trial_runner
from core.execution import SharedInterpreterExecutor

INSIDE = {"step": [0.4, 0.2, 0.1]}

# Uses only the first value of the list it is given: the number it returns is the value at step=0.4, whatever the case says.
FIRST_ONLY = '''
def run_trial(cell, trial_id, seed):
    step = cell["step"][0] if isinstance(cell["step"], list) else cell["step"]
    return {"end_value": 1.0 - step, "per_run_total": 3.0}
'''

# The contract: a number per value, keyed by the value.
PER_VALUE = '''
def run_trial(cell, trial_id, seed):
    return {"end_value": {"step=%s" % s: 1.0 - s for s in cell["step"]}, "per_run_total": 3.0}
'''

PROTOCOL = {"in_each_run": INSIDE, "metrics": [
    {"id": "end_value", "depends_on": ["step"]},
    {"id": "per_run_total", "depends_on": ["size"]},
]}


def _root(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return root


def _executor() -> SharedInterpreterExecutor:
    return SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")


def _measure(root: Path, oracle: dict) -> tuple[list, list, dict]:
    unrun: dict = {}
    checks, problems, _ = asyncio.run(trial_runner.measure_oracles(
        _executor(), sys.executable, root, "code/simulate.py", [oracle], timeout_s=60, in_each_run=INSIDE,
        unrun=unrun, protocol=PROTOCOL))
    return checks, problems, unrun


def _oracle(measure: str, expected: float, step: float = 0.2) -> dict:
    return {"name": "known", "kind": "special_case", "case": {"step": step}, "measure": measure, "expected": expected,
            "tolerance": 1e-9, "tolerance_mode": "absolute"}


def test_a_plain_number_from_a_run_given_every_value_is_not_read_as_the_value_the_case_names(tmp_path: Path) -> None:
    root = _root(tmp_path, FIRST_ONLY)
    checks, problems, unrun = _measure(root, _oracle("end_value", 0.8))
    assert "known" in unrun, (checks, problems)
    assert "step=0.2" in unrun["known"] and "end_value" in unrun["known"]
    assert not any(c.get("passed") is False for c in checks)  # never judged on the step=0.4 number


def test_a_number_per_value_is_read_at_the_value_the_case_names(tmp_path: Path) -> None:
    root = _root(tmp_path, PER_VALUE)
    checks, problems, unrun = _measure(root, _oracle("end_value", 0.8))
    assert unrun == {} and problems == [], problems
    assert len(checks) == 1


def test_a_declared_number_that_does_not_vary_with_the_factor_stays_a_plain_number(tmp_path: Path) -> None:
    root = _root(tmp_path, FIRST_ONLY)
    checks, problems, unrun = _measure(root, _oracle("per_run_total", 3.0))
    assert unrun == {} and problems == [], problems
    assert len(checks) == 1


def test_unplaced_names_the_case_value_and_the_way_to_return_it() -> None:
    levels = in_run.level_cells(INSIDE)
    why = in_run.unplaced({"end_value": 0.6}, levels, {"step": 0.2}, ["end_value"])
    assert "step=0.2" in why and "end_value" in why and "one number per value" in why
    assert in_run.unplaced({"end_value": {"step=0.2": 0.8}}, levels, {"step": 0.2}, ["end_value"]) == ""
    assert in_run.unplaced({"end_value": 0.6}, levels, {}, ["end_value"]) == ""  # the case picks no value


# --- the run's numbers given level first, and a measure that names the case's own setting --------------------------------

LEVEL_FIRST = '''
def run_trial(cell, trial_id, seed):
    return {"step=%s" % s: {"end_value": 1.0 - s} for s in cell["step"]}
'''


def test_numbers_given_level_first_are_read_as_a_number_per_value() -> None:
    levels = in_run.level_cells(INSIDE)
    got, why = in_run.read_returned({"step=0.4": {"end_value": 0.6}, "step=0.2": {"end_value": 0.8},
                                     "step=0.1": {"end_value": 0.9}}, levels, "run_trial")
    assert why == "" and got == {"end_value": {"step=0.4": 0.6, "step=0.2": 0.8, "step=0.1": 0.9}}


def test_a_check_reads_a_run_that_gives_its_numbers_level_first(tmp_path: Path) -> None:
    root = _root(tmp_path, LEVEL_FIRST)
    checks, problems, unrun = _measure(root, _oracle("end_value", 0.8))
    assert unrun == {} and problems == [], problems
    assert abs(checks[0]["value"] - 0.8) < 1e-12


def test_a_measure_that_names_the_cases_own_setting_reads_it_from_the_case(tmp_path: Path) -> None:
    root = _root(tmp_path, PER_VALUE)
    checks, problems, unrun = _measure(root, _oracle("abs(end_value - (1 - step))", 0.0))
    assert unrun == {} and problems == [], problems
    assert abs(checks[0]["value"]) < 1e-12


def test_a_measure_that_reads_only_the_cases_own_setting_is_not_a_measurement(tmp_path: Path) -> None:
    root = _root(tmp_path, PER_VALUE)
    checks, problems, _ = _measure(root, _oracle("1 - step", 0.8))
    assert checks == [] and any("takes nothing" in p for p in problems), problems


# --- review findings (Codex, 2026-11-06) --------------------------------------------------------------------------------

def _measure_plain(root: Path, oracle: dict, protocol=None) -> tuple[list, list, dict]:
    unrun: dict = {}
    checks, problems, _ = asyncio.run(trial_runner.measure_oracles(
        _executor(), sys.executable, root, "code/simulate.py", [oracle], timeout_s=60, unrun=unrun, protocol=protocol))
    return checks, problems, unrun


def test_a_case_setting_is_read_by_the_formula_without_values_compared_inside_each_run(tmp_path: Path) -> None:
    root = _root(tmp_path, "def run_trial(cell, trial_id, seed):\n    return {'y': 5.0}\n")
    checks, problems, _ = _measure_plain(root, {"name": "known", "kind": "special_case", "case": {"size": 2},
                                                "measure": "y - size", "expected": 3.0, "tolerance": 1e-9})
    assert problems == [] and abs(checks[0]["value"] - 3.0) < 1e-12, problems


def test_a_name_the_run_returns_keeps_the_runs_number_over_the_cases(tmp_path: Path) -> None:
    root = _root(tmp_path, "def run_trial(cell, trial_id, seed):\n    return {'size': 7.0}\n")
    checks, problems, _ = _measure_plain(root, {"name": "known", "kind": "special_case", "case": {"size": 2},
                                                "measure": "size", "expected": 7.0, "tolerance": 1e-9})
    assert problems == [] and checks[0]["value"] == 7.0, problems


def test_a_mean_of_the_cases_own_setting_takes_nothing_from_the_simulation() -> None:
    bound = trial_runner._bind_case("mean(size)", {"size": 2}, {"y": 1.0})
    assert bound == "mean((2.0))"
    assert trial_runner._bind_case("mean(size)", {"size": 2}, {"size": 1.0}) == "mean(size)"


def test_a_function_name_equal_to_a_case_setting_is_never_bound() -> None:
    assert trial_runner._bind_case("abs (y)", {"abs": 2}, {"y": -3.0}) == "abs (y)"
    # a setting named like a function the formula language knows is never bound: the formula reads it as the function
    assert trial_runner._bind_case("abs(y) + abs", {"abs": 2}, {"y": -3.0}) == "abs(y) + abs"
    assert trial_runner._bind_case("y - size", {"size": 2}, {"y": 5.0}) == "y - (2.0)"


MIXED = '''
def run_trial(cell, trial_id, seed):
    return {"y": 1.0} if trial_id == 0 else {"y": 1.0, "size": 100.0}
'''


def test_each_trial_reads_its_own_returned_name_before_the_cases(tmp_path: Path) -> None:
    rows = [{"y": 1.0}, {"y": 1.0, "size": 100.0}]
    got = trial_runner._rows_with_case(rows, {"size": 2})
    assert got == [{"size": 2.0, "y": 1.0}, {"y": 1.0, "size": 100.0}]


def test_a_mean_that_reads_only_the_cases_setting_in_every_trial_is_refused(tmp_path: Path) -> None:
    assert trial_runner._reads_only_case("mean(size)", {"size": 2}, {"y"})
    assert not trial_runner._reads_only_case("mean(y + size)", {"size": 2}, {"y"})
    assert not trial_runner._reads_only_case("mean(size)", {"size": 2}, {"y", "size"})


def test_a_case_setting_returned_as_one_plain_number_from_every_value_is_not_read_as_the_case(tmp_path: Path) -> None:
    src = ("def run_trial(cell, trial_id, seed):\n"
           "    return {'end_value': {'step=%s' % s: 1.0 - s for s in cell['step']}, 'step': cell['step'][0]}\n")
    root = _root(tmp_path, src)
    checks, problems, unrun = _measure(root, _oracle("end_value + step", 1.0))
    assert "known" in unrun and "`step`" in unrun["known"], (checks, problems)


def test_a_name_given_both_ways_is_never_overwritten() -> None:
    levels = in_run.level_cells({"h": [1, 2]})
    value = {"y": {"h=1": 11.0, "h=2": 22.0}, "h=1": {"y": 99.0}, "h=2": {"y": 88.0}}
    assert in_run.level_first(value, levels) is value
    got, why = in_run.read_returned(value, levels, "run_trial")
    assert got is None or got.get("y") == {"h=1": 11.0, "h=2": 22.0}, got
