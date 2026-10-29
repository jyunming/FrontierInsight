"""The cost record (core/cost_model.py): the terms FI measured before a study are kept apart, a quest's real times are read
from its own log, a record is rebuilt for a quest run before the file existed, and a search is sized from a time allowed.
Neutral made-up studies; the log lines are the ones the engine writes."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import cost_model as cm
from core import run_estimate
from tests.test_run_checks import _SCAN_BLOCK, _exact_timing, _search_engine  # noqa: F401 -- a fixture

# ---- the record ---------------------------------------------------------------------------------------------------------


def test_the_record_keeps_every_term_with_where_it_came_from(tmp_path: Path) -> None:
    model = cm.CostModel(limit_s=600.0, at_once=4, slowdown=1.5, terms={
        "fixed": cm.Term(2.0, cm.MEASURED), "per_trial": cm.Term(0.1, cm.DERIVED), "paid_once": cm.Term(30.0, cm.MEASURED, True)})
    assert cm.write(tmp_path, model)
    back = cm.read(tmp_path)
    assert back is not None and back.at_once == 4 and back.slowdown == 1.5
    assert back.term("fixed") == cm.Term(2.0, cm.MEASURED) and back.term("per_trial").basis == cm.DERIVED
    assert back.term("paid_once").at_least is True
    assert back.term("per_evaluation").seconds is None and back.term("per_evaluation").basis == cm.UNKNOWN
    assert cm.read(tmp_path / "nowhere") is None


def test_probes_are_kept_term_by_term_and_a_later_write_adds_to_them(tmp_path: Path) -> None:
    """The total used to be all that was kept: a probe's fixed cost, per-trial cost and first-run cost were thrown away."""
    base = run_estimate.Probe(None, None, {"a": 1}, seconds=7.0, fixed_s=5.0, per_trial_s=0.5, timed_trials=4, first_s=9.0)
    end = run_estimate.Probe("a", 2, {"a": 3}, seconds=9.0)
    est = run_estimate.estimate({"a": [1, 2, 3]}, 4, False, [base, end])
    assert cm.keep_measured(tmp_path, probes=[base, end], est=est, limit_s=100.0, key="key-a")
    assert cm.keep_measured(tmp_path, per_evaluation_s=3.0, plan={"check_runs": 6})
    model = cm.read(tmp_path)
    assert model is not None and model.key == "key-a" and model.limit_s == 100.0
    assert (model.seconds("fixed"), model.seconds("per_trial"), model.seconds("first_run")) == (5.0, 0.5, 9.0)
    assert model.term("fixed").basis == cm.MEASURED and len(model.probes) == 2 and model.probes[1]["axis"] == "a"
    assert model.seconds("per_evaluation") == 3.0 and model.plan["check_runs"] == 6 and model.cells == 3


def test_keeping_the_record_never_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert cm.keep_measured(blocker / "sub", per_evaluation_s=1.0) is False  # a root that cannot hold a folder
    assert cm.keep_measured(tmp_path, probes=[object()]) is False  # not a probe


# ---- what a quest really took -------------------------------------------------------------------------------------------

SEARCH_LOG = """\
2026-03-01 10:00:00,000 [INFO] [design] something earlier
2026-03-01 10:10:00,000 [WARNING] [execute] FI estimates the search for the best design would take at least 3 h, more than the 2 h allowed; it will run
2026-03-01 11:00:00,000 [INFO] [optimise] the coarse scan ran 40 setting(s), not counted in the search's budget: 1 of them
2026-03-01 11:00:05,000 [INFO] [optimise] still searching: 300 evaluation(s) so far (4 at the same time)
2026-03-01 12:30:00,000 [INFO] [optimise] 9000 evaluation(s) of the 50000 allowed, method m; stopped because the time limit was reached; 0 failed
2026-03-01 12:30:10,000 [INFO] [optimise] checking the best design at finer numerical settings (at most 20 runs of the simulation, apart from the search's budget)
2026-03-01 12:31:00,000 [INFO] [optimise] checking at finer settings: 1 of at most 20 run(s) done
2026-03-01 12:32:00,000 [INFO] [optimise]   the check ran the simulation 3 time(s) (at most 20 planned), apart from the search's budget
2026-03-01 12:32:30,000 [INFO] [execute] rc=1 duration=9000.0s figures=0 result_json=False
2026-03-01 13:00:00,000 [INFO] [execute] FI estimates the experiment takes about 2 minutes
2026-03-01 13:02:00,000 [INFO] [execute] FI ran 100 trial(s) in 10 cell(s)
2026-03-01 13:02:20,000 [INFO] [execute] rc=0 duration=200.0s figures=1 result_json=True
"""


def _quest(tmp_path: Path, log: str, estimate: dict[str, Any] | None = None, ledger: list[dict[str, Any]] | None = None) -> Path:
    root = tmp_path / "quest"
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "run.log").write_text(log, encoding="utf-8")
    if estimate is not None:
        (root / ".fi" / "run_estimate.json").write_text(json.dumps(estimate), encoding="utf-8")
    if ledger is not None:
        (root / "raw").mkdir()
        (root / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in ledger) + "\n", encoding="utf-8")
    return root


def test_the_times_a_search_and_a_study_really_took_are_read_from_the_log(tmp_path: Path) -> None:
    rows = cm.actuals_from_log(_quest(tmp_path, SEARCH_LOG))
    search, grid = rows
    assert search["kind"] == "search" and search["qualifier"] == "at least" and search["printed_s"] == 3 * 3600
    assert search["exec_s"] == 9000.0 and search["rc"] == 1
    # the scan ran from the start of the execute (12:32:30 less 9000 s = 10:02:30) to 11:00:00, the search to 12:30:00
    assert search["scan_s"] == 3450.0 and search["search_s"] == 5400.0 and search["check_s"] == 150.0
    assert search["evaluations"] == 9000 and search["allowed"] == 50000 and search["search_at_once"] == 4
    assert search["search_rate_per_s"] == pytest.approx(9000 / 5400, abs=1e-3)
    assert search["finer_first_run_s"] == 50.0 and search["check_runs_done"] == 3 and search["check_planned"] == 20
    assert grid["kind"] == "grid" and grid["printed_s"] == 120.0 and grid["study_s"] == 120.0 and grid["analysis_s"] == 20.0
    assert grid["trials"] == 100 and grid["cells"] == 10


def test_a_log_that_is_missing_or_has_no_execute_gives_nothing(tmp_path: Path) -> None:
    assert cm.actuals_from_log(tmp_path / "none") == []
    assert cm.actuals_from_log(_quest(tmp_path, "2026-03-01 10:00:00,000 [INFO] [design] nothing run\n")) == []


# ---- a quest run before the record existed --------------------------------------------------------------------------------


def test_a_record_is_rebuilt_for_an_earlier_quest_and_says_what_it_could_not_find(tmp_path: Path) -> None:
    ledger = [{"event": "trial", "cell": f"c{i % 10}", "duration_s": 1.0} for i in range(100)]
    root = _quest(tmp_path, SEARCH_LOG.split("2026-03-01 13:00:00")[0] + SEARCH_LOG.split("rc=1 duration=9000.0s figures=0 result_json=False\n")[1],
                  {"limit_s": 7200.0, "cells": 10, "trials": 100, "estimate_s": 180.0, "at_once": 4, "once_s": 0.0}, ledger)
    model = cm.from_legacy_record(root)
    assert model is not None and model.source == "legacy" and model.limit_s == 7200.0 and model.cells == 10
    # what the trials took (100 s of the 120 s study) is derived from actuals, never labelled as a measurement; the other
    # 20 s over 10 settings is each setting's fixed cost
    assert model.term("per_trial") == cm.Term(1.0, cm.DERIVED)
    assert model.seconds("fixed") == pytest.approx(2.0)
    assert model.term("fixed").basis == cm.DERIVED and model.term("paid_once").basis == cm.MEASURED
    # the search's rate and the check's first finer run come from the log
    assert model.term("finer_check_unit") == cm.Term(50.0, cm.DERIVED)
    assert model.seconds("wall_per_evaluation") == pytest.approx(5400 / 9000, abs=1e-3)
    assert model.term("check_unit").basis == cm.UNKNOWN and model.term("per_evaluation").seconds is None
    assert cm.from_legacy_record(tmp_path / "none") is None


def test_a_cut_search_record_gives_the_measured_design_cost_and_the_check_reserve_it_had(tmp_path: Path) -> None:
    cut = {"kind": "search", "estimate_s": 9e4, "at_least": True, "one_s": 7.2, "check_reserve_s": 295.5, "scan": 100,
           "evaluations": 5000, "check_runs": 72}
    model = cm.from_legacy_record(_quest(tmp_path, SEARCH_LOG, {"limit_s": 7200.0, "truncated": cut}))
    assert model is not None
    assert model.term("per_evaluation") == cm.Term(7.2, cm.MEASURED, True)
    assert model.plan["check_reserve_s"] == 295.5 and model.plan["check_runs"] == 72 and model.plan["evaluations"] == 5000


# ---- the size of a search ------------------------------------------------------------------------------------------------


def _search_model(**terms: float) -> cm.CostModel:
    return cm.CostModel(terms={k: cm.Term(v, cm.MEASURED) for k, v in terms.items()}, plan={"check_runs": 10})


def test_a_search_is_sized_from_the_time_left_after_the_scan_the_check_and_the_analysis() -> None:
    model = _search_model(per_evaluation=2.0, finer_check_unit=30.0, scan=500.0)
    got = cm.search_plan(model, 10_000.0, planned_evaluations=1_000_000)
    # (10,000 - 500 scan - 10 x 30 check - 71 analysis) / 2 s = 4,564 evaluations; the search stops at 10,000 - 300 - 71
    assert got.check_reserve_s == 300.0 and got.evaluations == 4564 and got.deadline_s == pytest.approx(9629.0)
    assert got.basis == cm.MEASURED and got.why == ""


def test_a_search_is_never_given_more_than_the_plan_asked() -> None:
    assert cm.search_plan(_search_model(per_evaluation=1.0, finer_check_unit=1.0), 1e6, planned_evaluations=500).evaluations == 500


def test_with_no_measured_cost_per_evaluation_no_number_is_given() -> None:
    got = cm.search_plan(_search_model(finer_check_unit=30.0), 10_000.0, planned_evaluations=1000)
    assert got.evaluations is None and got.deadline_s is None and "no cost per evaluation was measured" in got.why


def test_a_check_not_measured_at_finer_settings_is_priced_at_the_design_cost_and_said_to_be_a_lower_bound() -> None:
    got = cm.search_plan(_search_model(per_evaluation=2.0, scan=0.0), 1000.0)
    assert got.check_reserve_s == 20.0 and "at least" in got.why
    exact = cm.search_plan(_search_model(per_evaluation=2.0, finer_check_unit=47.0), 1000.0)
    assert exact.check_reserve_s == 470.0 and "at least" not in exact.why


def test_settings_run_at_once_share_the_cost_only_when_the_slowdown_was_measured() -> None:
    model = _search_model(per_evaluation=8.0, finer_check_unit=40.0)
    model.at_once = 8
    alone = cm.search_plan(model, 10_000.0)
    assert alone.check_reserve_s == 400.0 and alone.wall_per_evaluation_s == 8.0  # slowdown unknown: one after another
    model.slowdown = 2.0  # each takes twice as long beside seven others, eight run together: a quarter of the time
    shared = cm.search_plan(model, 10_000.0)
    assert shared.check_reserve_s == 100.0 and shared.wall_per_evaluation_s == 2.0
    assert shared.evaluations > alone.evaluations


def test_a_rate_seen_on_the_search_itself_wins_over_the_timing_of_one_design() -> None:
    model = _search_model(per_evaluation=0.1, finer_check_unit=10.0, wall_per_evaluation=1.0)
    assert cm.search_plan(model, 1000.0).wall_per_evaluation_s == 1.0


def test_the_check_keeps_its_time_whatever_the_search_rate_turns_out_to_be() -> None:
    """The same plan ran at 3.25, 1.34 and 0.82 evaluations a second. The reserve and the deadline do not depend on the
    rate; only the count does, and a search that ends at the deadline leaves the check its time at any rate."""
    for rate in (3.25, 1.34, 0.82):
        model = _search_model(wall_per_evaluation=1.0 / rate, finer_check_unit=50.0, scan=1000.0)
        got = cm.search_plan(model, 28_800.0)
        assert got.check_reserve_s == 500.0 and got.deadline_s == 28_800.0 - 500.0 - cm.ANALYSIS_RESERVE_S
        assert got.evaluations * got.wall_per_evaluation_s <= got.deadline_s - 1000.0 + 1e-6


def test_a_scan_and_a_check_that_fill_the_time_leave_no_evaluations() -> None:
    got = cm.search_plan(_search_model(per_evaluation=1.0, finer_check_unit=100.0, scan=500.0), 1000.0)
    assert got.evaluations == 0 and "already fill the time allowed" in got.why


def test_settings_that_share_a_process_pay_its_fixed_cost_once() -> None:
    model = cm.CostModel(terms={"fixed": cm.Term(2.0, cm.MEASURED), "per_trial": cm.Term(0.1, cm.MEASURED)})
    assert cm.seconds_with_settings_per_process(model, 100, 1) == pytest.approx(210.0)
    assert cm.seconds_with_settings_per_process(model, 100, 50) == pytest.approx(14.0)
    assert cm.seconds_with_settings_per_process(cm.CostModel(), 100, 1) is None


# ---- the sizing keeps its timings ----------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_sizing_of_a_search_keeps_what_it_timed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real sizing of a search (tests/test_run_checks.py) writes the cost of one design, the plan's counts and the probes."""
    block = copy.deepcopy(_SCAN_BLOCK)
    block["grid"] = {"axis0": [0, 1, 2]}
    block["evaluation_budget"] = {"starts": 8, "per_start": 2500, "seconds_per_evaluation": 0.3}
    eng, runner = await _search_engine(tmp_path, block)

    async def fifteen(*a: Any, **k: Any) -> Any:
        return run_estimate.Probe(None, None, {}, 15.0, fixed_s=3.0, per_trial_s=12.0), ""

    monkeypatch.setattr(run_estimate, "_time_one", fifteen)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    model = cm.read(eng.quest_root)
    assert model is not None
    assert model.seconds("per_evaluation") == 15.0 and model.seconds("check_unit") == 15.0
    assert model.term("check_unit").basis != cm.MEASURED, "the check is still priced at the search-design cost"
    assert model.plan["evaluations"] == 20000 and model.plan["check_runs"] == 10 and model.plan["scan"] == 3
    assert model.probes and model.limit_s == eng.config.execution.timeout_s
    # the record is a pure addition: the sizing's own record is untouched
    assert run_estimate.read_record(eng.quest_root)["truncated"]["evaluations"] == 20000


# ---- the verdict ---------------------------------------------------------------------------------------------------------


def test_a_study_is_stopped_only_when_not_even_one_setting_fits() -> None:
    fits = cm.CostModel(limit_s=60.0, printed_s=50.0, cells=10)
    over = cm.CostModel(limit_s=60.0, printed_s=130.0, cells=10)  # 13 s a setting: runs on up to the limit
    none = cm.CostModel(limit_s=10.0, printed_s=130.0, cells=10)  # 13 s a setting, 10 s allowed
    assert (cm.verdict(fits), cm.verdict(over), cm.verdict(none)) == ("fits", "run_to_limit", "stop")
    assert cm.verdict(cm.CostModel(limit_s=60.0)) == "unknown"


def test_one_design_timed_twice_with_a_big_difference_is_sized_at_the_larger_and_says_so(tmp_path: Path) -> None:
    for seconds in (1.9, 7.2):
        assert cm.keep_measured(tmp_path, per_evaluation_s=seconds)
    model = cm.read(tmp_path)
    assert model is not None and model.evaluation_timings == [1.9, 7.2] and model.seconds("per_evaluation") == 7.2
    got = cm.search_plan(model, 7200.0)
    assert got.wall_per_evaluation_s == 7.2 and "timed 2 times" in got.why and "longest time" in got.why


def test_an_experiment_estimate_after_a_search_estimate_does_not_replace_it(tmp_path: Path) -> None:
    extra = "2026-03-01 10:50:00,000 [INFO] [execute] FI estimates the experiment takes about 2 minutes"
    log = SEARCH_LOG.split("2026-03-01 13:00:00")[0].replace("2026-03-01 11:00:00,000 [INFO] [optimise] the coarse scan",
                                                              extra + chr(10) + "2026-03-01 11:00:00,000 [INFO] [optimise] the coarse scan")
    search = cm.actuals_from_log(_quest(tmp_path, log))[0]
    assert search["kind"] == "search" and search["printed_s"] == 3 * 3600


def test_a_search_whose_execute_never_logged_its_end_is_still_a_row(tmp_path: Path) -> None:
    kept = [ln for ln in SEARCH_LOG.splitlines() if "[execute] rc=1" not in ln and not ln.startswith("2026-03-01 13:")]
    rows = cm.actuals_from_log(_quest(tmp_path, chr(10).join(kept) + chr(10)))
    assert len(rows) == 1 and rows[0]["partial"] is True and rows[0]["evaluations"] == 9000 and rows[0]["rc"] is None
    assert rows[0]["search_s"] == 5400.0


def test_a_search_with_no_estimate_line_and_no_end_has_a_rate_but_no_scan_time(tmp_path: Path) -> None:
    kept = [ln for ln in SEARCH_LOG.splitlines() if "FI estimates the search" not in ln and "[execute] rc=1" not in ln
            and not ln.startswith("2026-03-01 13:")]
    row = cm.actuals_from_log(_quest(tmp_path, chr(10).join(kept) + chr(10)))[0]
    assert row["search_s"] == 5400.0 and row["search_rate_per_s"] > 1.0 and "scan_s" not in row and row["exec_s"] is None


# ---- fixes of the first review -------------------------------------------------------------------------------------------


def test_the_check_reserve_divides_a_whole_setting_by_its_runs_as_the_run_does() -> None:
    """A setting of 5 runs takes 10 s; the check's 20 runs then cost 20 x 10 / 5 = 40 s, not 20 x 10 = 200 s."""
    model = _search_model(per_evaluation=10.0, scan=0.0)
    model.plan.update(check_runs=20, runs_per_evaluation=5)
    got = cm.search_plan(model, 10_000.0)
    assert got.check_reserve_s == pytest.approx(40.0)
    model.plan["runs_per_evaluation"] = 1
    assert cm.search_plan(model, 10_000.0).check_reserve_s == pytest.approx(200.0)


def test_a_scan_the_plan_has_but_nobody_timed_gives_no_number() -> None:
    model = _search_model(per_evaluation=2.0, finer_check_unit=5.0)
    model.plan["scan"] = 100
    got = cm.search_plan(model, 10_000.0)
    assert got.evaluations is None and "coarse scan was not measured" in got.why
    assert cm.search_plan(model, 10_000.0, scan_s=300.0).evaluations is not None
    model.plan["scan"] = 1  # no scan to time
    assert cm.search_plan(model, 10_000.0).evaluations is not None


@pytest.mark.asyncio
async def test_the_sizing_keeps_the_scan_time_and_the_runs_per_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block = copy.deepcopy(_SCAN_BLOCK)
    block["grid"] = {"axis0": [0, 1, 2]}
    block["evaluation_budget"] = {"starts": 8, "per_start": 2500, "seconds_per_evaluation": 0.3}
    eng, runner = await _search_engine(tmp_path, block)

    async def fifteen(*a: Any, **k: Any) -> Any:
        return run_estimate.Probe(None, None, {}, 15.0, fixed_s=3.0, per_trial_s=12.0), ""

    monkeypatch.setattr(run_estimate, "_time_one", fifteen)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    model = cm.read(eng.quest_root)
    assert model is not None and model.seconds("scan") is not None and model.seconds("scan") > 0
    assert model.plan["runs_per_evaluation"] >= 1 and model.search_key


def test_reserves_larger_than_the_limit_never_make_a_negative_deadline() -> None:
    model = _search_model(per_evaluation=1.0, finer_check_unit=100.0, scan=500.0)  # 10 runs x 100 s of check, 1000 s allowed
    got = cm.search_plan(model, 1000.0)
    assert got.check_reserve_s == 500.0 and got.deadline_s is not None and got.deadline_s >= 0.0
    assert got.evaluations == 0 and "half the time allowed" in got.why and "already fill the time allowed" in got.why
    tiny = cm.search_plan(model, 50.0)  # the analysis reserve alone is over the limit
    assert tiny.deadline_s == 0.0 and tiny.evaluations == 0


def test_timings_of_an_earlier_design_are_dropped_and_not_used(tmp_path: Path) -> None:
    assert cm.keep_measured(tmp_path, per_evaluation_s=90.0, check_unit_s=90.0, plan={"check_runs": 10}, design="old")
    assert cm.keep_measured(tmp_path, per_evaluation_s=2.0, plan={"check_runs": 10}, design="new")
    model = cm.read(tmp_path)
    assert model is not None and model.evaluation_timings == [2.0] and model.seconds("per_evaluation") == 2.0
    assert model.term("check_unit").seconds is None and model.search_key == "new"
    # a record kept for one design says nothing of another
    stale = cm.search_plan(model, 10_000.0, design="newer")
    assert stale.evaluations is None and "earlier design" in stale.why
    assert cm.search_plan(model, 10_000.0, design="new").evaluations is not None


def test_a_design_key_follows_the_block_and_the_code(tmp_path: Path) -> None:
    sim = tmp_path / "simulate.py"
    sim.write_text("a = 1\n", encoding="utf-8")
    one = cm.design_key({"grid": {"x": [1, 2]}}, sim)
    assert one == cm.design_key({"grid": {"x": [1, 2]}}, sim) and one != cm.design_key({"grid": {"x": [1, 3]}}, sim)
    sim.write_text("a = 2\n", encoding="utf-8")
    assert one != cm.design_key({"grid": {"x": [1, 2]}}, sim)


def test_the_analysis_of_a_search_is_timed_from_the_end_of_its_check(tmp_path: Path) -> None:
    log = "\n".join([
        "2026-03-01 10:00:00,000 [INFO] [execute] FI estimates the search for the best design takes about 1 h",
        "2026-03-01 10:01:00,000 [INFO] [optimise] the coarse scan ran 10 setting(s), not counted in the search's budget",
        "2026-03-01 11:00:00,000 [INFO] [optimise] 500 evaluation(s) of the 1000 allowed, method m; stopped because the time limit (execution.timeout_s) was reached",
        "2026-03-01 11:00:05,000 [INFO] [optimise] checking the best design at finer numerical settings (at most 6 runs of the simulation, apart from the search's budget)",
        "2026-03-01 11:05:00,000 [INFO] [optimise]   the check ran the simulation 6 time(s) (at most 6 planned), apart from the search's budget",
        "2026-03-01 11:05:12,000 [INFO] [execute] rc=0 duration=3912.0s figures=1 result_json=True"])
    root = _quest(tmp_path, log)
    row = cm.actuals_from_log(root)[0]
    assert row["check_s"] == 312.0 and row["analysis_s"] == 12.0  # not 312: the check is not analysis
    model = cm.from_legacy_record(root)
    assert model is not None and model.seconds("analysis") == 12.0
