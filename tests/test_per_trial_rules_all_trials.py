"""A rule that must hold in every trial is judged on every trial the check runs, not on the first batch alone.

The real failure: ``_measure_every`` (core/trial_runner.py) judged a ``per_trial: true`` rule on the first
``PILOT_TRIALS`` (20) trials only, so a simulation that broke the rule only in trial 25 of 40 passed. The first batch
stays an early stop (a fault seen there ends the check at once); a pass now needs every trial to be judged.
Everything runs through the real trial runner in its own process; no model is called.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from core import mc_error as mc, trial_runner
from core.execution import SharedInterpreterExecutor

SIM = '''
BROKEN = {broken}

def run_trial(cell, trial_id, seed):
    # a number that moves a little from trial to trial but stays inside the tolerance, except in the broken trials
    return {{"gap": 5.0 if trial_id in BROKEN else 0.1 * (trial_id % 3)}}
'''
RULE = {"name": "gap_small", "kind": "invariant", "per_trial": True, "expected": 0.0, "tolerance": 0.5,
        "case": {"n": 1}, "measure": "gap", "check": "the gap is small in every trial"}


def _measure(tmp_path: Path, broken: set[int], most: int) -> dict[str, Any]:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM.format(broken=sorted(broken)), encoding="utf-8")
    checks, problems, _ = asyncio.run(trial_runner.measure_oracles(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", [RULE],
        timeout_s=60, mean_names=frozenset(), most_trials=most))
    assert not problems, problems
    (check,) = checks
    return check


def test_a_rule_broken_only_in_trial_25_of_40_is_not_passed(tmp_path: Path) -> None:
    """Fails on the old code: only trials 0-19 were judged, the worst of them is 0.2, so the rule passed."""
    check = _measure(tmp_path, {25}, most=40)
    assert check["mc"]["verdict"] == "fail", check["mc"]
    assert check["value"] == 5.0 and check["mc"]["trials"] > mc.PILOT_TRIALS


def test_a_rule_that_holds_in_all_40_trials_passes(tmp_path: Path) -> None:
    """Known positive: the extra trials do not turn an honest rule into a failure or an undecided check."""
    check = _measure(tmp_path, set(), most=40)
    assert check["mc"]["verdict"] == "pass" and check["mc"]["trials"] == 40, check["mc"]
    assert abs(check["value"]) <= 0.5


def test_a_rule_broken_in_trial_3_still_stops_at_the_first_batch(tmp_path: Path) -> None:
    """The first batch is still an early stop: a fault in it fails the check without running the other trials."""
    check = _measure(tmp_path, {3}, most=40)
    assert check["mc"]["verdict"] == "fail" and check["mc"]["trials"] == mc.PILOT_TRIALS, check["mc"]


def test_an_undecided_check_says_how_many_trials_failed_and_that_it_was_not_decided() -> None:
    record = mc.record({"trials": 17, "dropped": 0, "varies": True, "on": "trials"}, read_as="every_trial",
                       verdict="undecided", why="failed_trials", failed=3, started=20, trials_asked=40)
    said = mc.undecided_words(record, 0.5)
    assert "3 of 20 trials" in said and "not decided" in said and "not a pass" in said
    assert "3 of 20 trials" in mc.words(record)


def test_a_pass_on_fewer_trials_than_asked_stays_a_pass_and_names_the_gap() -> None:
    record = mc.record({"trials": 25, "dropped": 0, "varies": True, "on": "trials"}, read_as="every_trial",
                       verdict="pass", why="time", failed=0, started=25, trials_asked=40)
    assert record["verdict"] == "pass" and "judged on 25 of 40 trials" in mc.words(record)
