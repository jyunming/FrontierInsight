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

from core import mc_error as mc, oracle_check, trial_runner
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


def test_time_running_out_gives_the_verdict_partial_with_counts(tmp_path: Path, monkeypatch: Any) -> None:
    """Fails on the old code (verdict stayed "pass"): the first 20 of 40 trials hold, no time is left for the rest."""
    monkeypatch.setattr(mc, "PILOT_SHARE", 0.0)  # no time left after the first batch
    check = _measure(tmp_path, set(), most=40)
    record = check["mc"]
    assert record["verdict"] == "partial", record
    assert record["judged"] == mc.PILOT_TRIALS and record["planned"] == 40
    assert mc.is_partial(record) and "passed on 20 of 40 trials" in mc.words(record)
    assert "not judged in the time allowed" in mc.partial_words(record)


def test_a_full_run_is_not_partial() -> None:
    record = mc.record({"trials": 40, "dropped": 0, "varies": True, "on": "trials"}, read_as="every_trial",
                       verdict="pass", why="", failed=0, started=40, trials_asked=40)
    assert not mc.is_partial(record) and mc.partial_words(record) == ""


def test_partial_is_a_pass_for_stop_and_repair_and_a_gap_for_the_record() -> None:
    """Fails on the old code: mc_of dropped an unknown verdict, and nothing marked the check as partly judged."""
    oracle = {"name": "gap_small", "kind": "invariant", "expected": 0.0, "tolerance": 0.5}
    record = mc.record({"trials": 20, "dropped": 0, "varies": True, "on": "trials"}, read_as="every_trial",
                       verdict="partial", why="time", failed=0, started=20, trials_asked=40, judged=20, planned=40)
    reported = {"engine_measured": True,
                "checks": [{"name": "gap_small", "value": 0.2, "passed": True, "measured_by": "engine", "mc": record}]}
    (j,) = oracle_check.judged([oracle], reported)
    # stop / repair: a pass, nothing to fix, no problem raised
    assert j["passed_by_engine"] is True and not j.get("undecided")
    assert oracle_check.problems([oracle], reported, 0) == []
    assert oracle_check.not_passed([j]) == []
    # the record of what was checked: not fully checked
    assert oracle_check.partial_checks([j]) == [j]
    said = oracle_check.partial_gap_words(j)
    assert "passed on 20 of 40 trials" in said and "not judged in the time allowed" in said
    assert "passed on 20 of 40 trials" in oracle_check.analysis_note([j])
    # a full pass carries no such gap
    full = {**reported, "checks": [{**reported["checks"][0], "mc": {**record, "verdict": "pass"}}]}
    (k,) = oracle_check.judged([oracle], full)
    assert oracle_check.partial_checks([k]) == []
