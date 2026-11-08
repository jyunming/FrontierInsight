"""A check against a known answer whose case fixes one value of a factor the plan sweeps inside each run
(``protocol.in_each_run``) is run on that one value: FI narrows the list of that factor to the case's value, so the
simulation's plain number can only be the case's. The model is not asked to pick the value out of a list."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core import trial_runner
from core.execution import SharedInterpreterExecutor

INSIDE = {"angle_deg": [0, 10, 20], "mode": ["a", "b"]}

# A simulation that reads the first listed value of each factor and returns plain numbers (no number per value).
SIM = '''
def run_trial(cell, trial_id, seed):
    angles, modes = cell["angle_deg"], cell["mode"]
    return {"gain": float(angles[0]) * 2.0 + len(modes), "level_count": float(len(angles))}
'''


def _executor() -> SharedInterpreterExecutor:
    return SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    return root


def _case(root: Path, cell: dict, inside: dict | None = INSIDE):
    return asyncio.run(trial_runner.run_case(_executor(), sys.executable, root, "code/simulate.py", cell=cell, timeout_s=60,
                                             in_each_run=inside, per_value=("gain",)))


def test_a_case_naming_one_value_of_a_factor_compared_in_each_run_is_judged(tmp_path: Path) -> None:
    got, why = _case(_root(tmp_path), {"angle_deg": 20})
    assert why == "" and got is not None, why
    assert got["gain"] == 20 * 2.0 + 2           # the angle is the case's, the unnamed factor keeps its whole list
    assert got["level_count"] == 1.0             # the run was given the one angle the case names


def test_the_trials_of_a_case_are_given_the_one_value_too(tmp_path: Path) -> None:
    rows, why, _failed, _started = asyncio.run(trial_runner.run_case_trials(
        _executor(), sys.executable, _root(tmp_path), "code/simulate.py", cell={"angle_deg": 10, "mode": "b"}, trials=[0, 1],
        timeout_s=60, in_each_run=INSIDE, per_value=("gain",)))
    assert why == "" and [r["gain"] for r in rows] == [20.0 + 1, 20.0 + 1]


def test_a_case_naming_no_factor_compared_in_each_run_is_given_every_value(tmp_path: Path) -> None:
    root = _root(tmp_path)
    for cell in ({"width": 5}, {}):
        got, why = _case(root, cell)
        assert why == "" and got is not None and got["level_count"] == 3.0   # unchanged: nothing is narrowed


def test_a_case_naming_a_value_the_run_is_not_given_is_still_refused(tmp_path: Path) -> None:
    got, why = _case(_root(tmp_path), {"mode": "c"})
    assert got is None and "each run compares" in why


# A simulation that writes out the values of the factor itself (instead of reading the list it is given) returns a number
# per value for all three: it cannot be read on the one-value list, so FI runs the case again with the whole list.
HARD_CODED = '''
def run_trial(cell, trial_id, seed):
    return {"gain": {"a": 1.0, "b": 2.0, "c": 3.0}}
'''


def test_a_simulation_that_cannot_read_the_one_value_list_is_run_again_with_the_whole_list(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_text(HARD_CODED, encoding="utf-8")
    inside = {"mode": ["a", "b", "c"]}
    got, why = asyncio.run(trial_runner.run_case(_executor(), sys.executable, root, "code/simulate.py", cell={"mode": "b"},
                                                 timeout_s=60, in_each_run=inside))
    assert why == "" and got == {"gain": 2.0}, why
    rows, why, _failed, _started = asyncio.run(trial_runner.run_case_trials(
        _executor(), sys.executable, root, "code/simulate.py", cell={"mode": "c"}, trials=[0], timeout_s=60,
        in_each_run=inside))
    assert why == "" and rows == [{"gain": 3.0}], why
