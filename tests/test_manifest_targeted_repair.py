"""A run-manifest difference in the analysis: FI counts the concrete difference itself (which metric, which settings, how many
values listed against how many measured), sends it to the repair of the analysis code only, tries up to three times, and when
the check still fails the stop opens with a plain sentence. No model is called."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import run_manifest as rm
from core.engine import _manifest_repair_budget
from tests.test_cannot_repair_stops import _engine
from tests.test_run_manifest_nested_path import MANIFEST, PROTOCOL, RUNS, _nested, _recorded, _recorded_by_cell


def _differences(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    return rm.differences(PROTOCOL, MANIFEST, analysis, recorded=_recorded(), recorded_by_cell=_recorded_by_cell())


def test_fi_counts_which_setting_lists_more_than_its_trials_returned() -> None:
    (found,) = _differences(_nested(extra=(20, 0.2)))
    assert found["metric"] == "err" and found["measured"] == (20 * RUNS, 0)
    assert [(s["setting"], s["listed"], s["recorded"]) for s in found["settings"]] == [
        ("n_pts=20,nsr=0.2", (RUNS + 1, 0), (RUNS, 0))]
    assert _differences(_nested()) == [], "a report that matches the trials has no difference"


def test_the_same_values_under_several_settings_are_named_with_their_counts() -> None:
    analysis = _nested()
    for a, b in ((5, 0.01), (10, 0.02), (20, 0.05)):  # three settings list twice what they measured
        analysis["by_n_pts"][str(a)]["by_nsr"][str(b)]["err_values"] = [0.5] * (2 * RUNS)
        analysis["by_n_pts"][str(a)]["by_nsr"][str(b)]["n_runs"] = 2 * RUNS
    (found,) = _differences(analysis)
    assert {s["setting"] for s in found["settings"]} == {"n_pts=5,nsr=0.01", "n_pts=10,nsr=0.02", "n_pts=20,nsr=0.05"}
    request = rm.difference_request([found])
    assert "`err`" in request and "n_pts=5,nsr=0.01" in request and f"lists {2 * RUNS:g} value(s)" in request
    assert "FI_TRIALS" in request and "never one list copied under several settings" in request
    directive = rm.analysis_directive(["x"], [found])
    assert "n_pts=10,nsr=0.02" in directive and "Rewrite experiment.py (not simulate.py)" in directive
    assert "What FI found by counting" not in rm.analysis_directive(["x"]), "no difference counted: the request is as before"


def test_the_plain_sentence_comes_first_and_says_it_in_a_scientists_words() -> None:
    (found,) = _differences(_nested(extra=(20, 0.2)))
    plain = rm.plain_difference([found])
    assert plain.startswith("The analysis lists more values under one setting than the trials of those settings returned")
    assert "so the paper could not be tied to what was measured" in plain and "`n_pts=20,nsr=0.2`" in plain
    for word in ("FI's record", "trial(s)", "RESULT_JSON", "_values"):
        assert word not in plain, word
    many = {**found, "settings": [{"setting": f"s{i}", "listed": (2, 0), "recorded": (1, 0)} for i in range(9)]}
    assert "under several settings" in rm.plain_difference([many]) and "and 5 more" in rm.plain_difference([many])
    pooled = {"metric": "err", "measured": (10, 0), "listed": (30, 0), "settings": []}
    assert "(30 listed, 10 measured)" in rm.plain_difference([pooled])
    assert rm.plain_difference([]) == ""


@pytest.mark.asyncio
async def test_the_stop_opens_with_the_plain_sentence_and_keeps_the_numbers_after_it(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    found = [s for s, _ in rm._value_count_findings(PROTOCOL, MANIFEST, _nested(extra=(20, 0.2)), trial_mode=True,
                                                    recorded=_recorded(), recorded_by_cell=_recorded_by_cell())]
    assert found
    eng._manifest_analysis_problems = list(found)  # type: ignore[attr-defined]
    eng._manifest_differences = _differences(_nested(extra=(20, 0.2)))  # type: ignore[attr-defined]
    patch = eng._manifest_stop(found, 3)
    text = json.dumps(patch["exec_result"]) if "exec_result" in patch else json.dumps(patch)
    stop = patch["exec_result"]["cannot_repair"]
    assert stop.startswith("The analysis lists more values under one setting"), stop
    assert " Details: the metric `err` is measured" in stop and "the analysis reports more than that" in stop
    assert "sent experiment.py back to be repaired 3 time(s)" in text


def test_an_analysis_difference_is_sent_back_up_to_three_times_and_nothing_else_gains_tries() -> None:
    assert _manifest_repair_budget(1, True) == 3
    assert _manifest_repair_budget(2, True) == 3
    assert _manifest_repair_budget(1, False) == 1, "a difference in the simulation keeps the budget it had"
    assert _manifest_repair_budget(0, True) == 0, "no repairs asked for, none made"


def test_a_repair_that_gave_up_spends_the_analysis_budget_so_the_stop_follows(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.config.engine.run_manifest_check = "block"
    state = {"run_manifest_failures": 1, "exec_reflect_iter": 0, "run_manifest_analysis_only": True}
    patch = eng._manifest_repair_gave_up(state)
    assert patch is not None and patch["run_manifest_failures"] >= 3 and patch["exec_reflect_iter"] == 0
    patch = eng._manifest_repair_gave_up({**state, "run_manifest_analysis_only": False})
    assert patch is not None and patch["run_manifest_failures"] == eng.config.engine.run_manifest_repair_attempts
    assert patch["exec_reflect_iter"] == 1
