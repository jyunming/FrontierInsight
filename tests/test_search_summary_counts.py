"""The run.log sentence about a finished search names the counts its own budget rule used.

An alternative the plan compares is searched with the whole budget of its own, so a search over several alternatives
holds several budgets: the sentence says, for each way they ended, how many evaluations of how many allowed were used,
and the coarse scan's designs (not counted in the budget) are not counted in the failed / broke-a-limit numbers."""
from __future__ import annotations

from typing import Any

from core import optimise
from core import optimise_search as osearch
from tests.test_optimise_runner import _block, _in_process


def _record(levels: list[tuple[int, int, str]], **ev: Any) -> dict[str, Any]:
    evaluations = {"search": sum(u for u, _, _ in levels), "budget": sum(b for _, b, _ in levels), "scan": 0, "failed": 0,
                   "infeasible": 0, "stopped_because": "budget", **ev}
    return {"objective": {"quantity": "f"}, "method": {"used": "global_then_local"}, "evaluations": evaluations,
            "baseline": {"design": {}, "objective": 1.0, "feasible": True}, "best": None,
            "compared": {"levels": [{"level": {"mode": f"m{i}"}, "evaluations": u, "budget": b, "stopped_because": why}
                                    for i, (u, b, why) in enumerate(levels)]}}


def test_one_alternative_that_failed_does_not_turn_the_other_budgets_into_one_that_ran_out() -> None:
    text = optimise.summary_lines(_record([(180, 180, "budget")] * 4 + [(10, 180, "failures")]))[0]
    assert "730 evaluation(s) in all over 5 alternatives" in text
    assert "4 alternatives used 720 of the 720 allowed and stopped because the evaluation budget ran out" in text
    assert "1 alternative used 10 of the 180 allowed and gave no result" in text
    assert "first evaluations all failed" not in text, "an alternative that gave no result does not name the search's end"
    only = optimise.summary_lines(_record([(10, 120, "failures")] * 2))[0]
    assert "2 alternatives used 20 of the 240 allowed and stopped because the first evaluations all failed" in only
    assert "of the 900 allowed" not in text


def test_every_alternative_using_its_whole_budget_still_says_the_budget_ran_out() -> None:
    text = optimise.summary_lines(_record([(120, 120, "budget")] * 3))[0]
    assert "3 alternatives used 360 of the 360 allowed and stopped because the evaluation budget ran out" in text


def test_a_single_search_keeps_its_sentence() -> None:
    record = _record([(60, 60, "budget")])
    record["compared"] = None
    text = optimise.summary_lines(record)[0]
    assert text.startswith("[optimise] 60 evaluation(s) of the 60 allowed, method global_then_local; stopped because "
                           "the evaluation budget ran out; 0 failed")


def test_the_scans_designs_are_not_counted_among_the_failed_of_the_evaluations_named() -> None:
    record = _record([(10, 120, "failures")], scan=80, failed=130, infeasible=0, stopped_because="failures",
                     scan_failed=80, scan_infeasible=0)
    text = optimise.summary_lines(record)[0]
    assert "after a coarse scan of 80 design(s) (not counted in the budget; 80 failed)" in text
    assert "; 50 failed, 0 broke a limit" in text
    old = optimise.summary_lines({**record, "evaluations": {k: v for k, v in record["evaluations"].items()
                                                              if not k.startswith("scan_")}})[0]
    assert "130 failed" in old and "the coarse scan's designs included" in old, "an older record says what it counts"


def test_the_search_records_how_many_of_the_scans_designs_failed() -> None:
    block = _block(grid={"scale": [0.5, 1.0, 2.0], "x": [-4.0, 4.0]}, search_method="global_then_local",
                   evaluation_budget={"starts": 1, "per_start": 8})
    assert _in_process(block)["scan_failed"] == 0
    out = _in_process(block, fail=lambda cell: cell["x"] == -4.0)
    scan_rows = [r for r in out["rows"] if not r["counted"]]
    assert scan_rows and out["scan_failed"] == sum(1 for r in scan_rows if r["status"] != "ok") > 0
    assert out["failed"] >= out["scan_failed"], "``failed`` still counts the scan's designs; ``scan_failed`` is its share"
    record = osearch.best_design(out, block, seed=0)
    assert record["evaluations"]["scan_failed"] == out["scan_failed"]


def test_alternatives_that_ended_on_their_starts_shares_are_not_reported_as_a_budget_that_ran_out() -> None:
    text = optimise.summary_lines(_record([(401, 1600, "share")] * 8 + [(1625, 1600, "budget")]))[0]
    assert "8 alternatives used 3208 of the 12800 allowed and stopped because each starting point converged or used its share" in text
    assert "1 alternative used 1625 of the 1600 allowed and stopped because the evaluation budget ran out" in text
    assert "of the 14400 allowed" not in text


def test_an_alternative_that_was_not_searched_is_not_said_to_have_used_a_budget() -> None:
    record = _record([(180, 180, "budget"), (1, 1, "not_searched"), (3, 3, "same_as_first")])
    text = optimise.summary_lines(record)[0]
    assert "1 alternative used 180 of the 180 allowed and stopped because the evaluation budget ran out" in text
    assert "1 alternative not searched (1 evaluation(s) to tell)" in text and "used 1 of the 1" not in text
    assert "1 alternative not searched (3 evaluation(s) to tell)" in text and "used 3 of the 3" not in text
