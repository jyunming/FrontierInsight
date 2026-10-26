"""A search whose objective never moves stops early (core/optimise_search.py, ``flat_objective``).

Live searches spent their whole budget (4000 and 6000 evaluations) on an objective that came back the same at every
design, although the design moved other results: nothing to climb. In FI's records of real searches, every search whose
objective moved at all did so at the first design after the baseline. Now the search stops once its first designs,
which between them moved every number of the design, all gave the baseline's objective; it is not sent back as a fault.
Neutral toy topic only (a heat sink).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from core import best_design_report, optimise, optimum_check
from core import optimise_search as search
from tests.test_flat_output import _quest
from tests.test_optimise_runner import LocalExecutor
from tests.test_search_no_effect import _block as _local_block, _drive


def _block(**changes: Any) -> dict[str, Any]:
    """The toy search, sampled across its range first (the global stage) and then refined locally."""
    return _local_block(**{"search_method": "global_then_local", "evaluation_budget": {"starts": 2, "per_start": 100},
                           **changes})


def _drive_with_scan(block: dict[str, Any], answer: Any, scan: list[dict[str, Any]]) -> dict[str, Any]:
    gen = search.search(block, seed=0, results=["mass"])
    request = next(gen)
    try:
        while True:
            if request["kind"] == "scan":
                reply: Any = [{"design": d, "values": answer({**d})} for d in scan]
            else:
                reply = {"values": answer(request["cell"])}
            request = gen.send(reply)
    except StopIteration as done:
        return done.value


def _flat(cell: dict[str, Any]) -> dict[str, Any]:
    return {"peak": 0.0, "mass": 2.0 + cell["fin"]}


def test_an_objective_that_never_moves_stops_the_search_after_its_first_designs() -> None:
    outcome = _drive(_block(), _flat, ["mass"])
    assert outcome["stopped_because"] == "flat_objective"
    assert outcome["evaluations"] == 1 + search.NO_EFFECT_DESIGNS, "the baseline and the designs that showed it"
    assert outcome["flat_objective"] == {"designs": search.NO_EFFECT_DESIGNS, "objective": 0.0, "feasible": True,
                                         "stayed": [], "moved": ["mass"], "stopped": True}
    assert outcome["best"]["objective"] == 0.0


def _parts(cell: dict[str, Any]) -> dict[str, Any]:
    return {"peak": 0.0, "part_a": 3.0, "part_b": cell["fin"]}


def test_the_flat_fact_names_the_results_that_stayed_and_those_that_moved() -> None:
    outcome = _drive(_block(), _parts, ["part_a", "part_b"])
    fact = outcome["flat_objective"]
    assert fact["stayed"] == ["part_a"] and fact["moved"] == ["part_b"]
    assert fact["stopped"] is True and fact["feasible"] is True


def test_after_the_revision_a_flat_search_runs_its_whole_budget_and_still_records_the_fact() -> None:
    gen = search.search(_block(), seed=0, results=["mass"], stop_flat=False)
    request = next(gen)
    try:
        while True:
            request = gen.send([] if request["kind"] == "scan" else {"values": _flat(request["cell"])})
    except StopIteration as done:
        outcome = done.value
    assert outcome["stopped_because"] != "flat_objective"
    # It goes on as a search whose objective moves does (here its local steps converge before the budget is used).
    assert outcome["evaluations"] > 1 + search.NO_EFFECT_DESIGNS
    assert outcome["flat_objective"]["stopped"] is False
    assert outcome["flat_objective"]["designs"] >= search.NO_EFFECT_DESIGNS


def test_an_objective_that_moves_after_a_plateau_shorter_than_the_watch_is_searched_to_its_budget() -> None:
    calls = {"n": 0}

    def late(cell: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        return {"peak": 40.0 if calls["n"] <= 31 else 30.0 + (cell["fin"] - 3.0) ** 2, "mass": cell["fin"]}

    outcome = _drive(_block(), late, ["mass"])
    assert outcome["stopped_because"] != "flat_objective" and "flat_objective" not in outcome
    assert outcome["evaluations"] > 1 + search.NO_EFFECT_DESIGNS


def test_an_objective_that_moves_at_once_is_searched_as_before() -> None:
    outcome = _drive(_block(), lambda cell: {"peak": 30.0 + (cell["fin"] - 3.0) ** 2, "mass": cell["fin"]}, ["mass"])
    assert "flat_objective" not in outcome and outcome["evaluations"] > 1 + search.NO_EFFECT_DESIGNS


# The objective is computed from the design, but it is 0 over the whole range the search may try (a window that never
# opens): the code reads the design, so this is no fault of the simulation.
FLAT_OBJECTIVE = '''\
def run_cell(cell):
    return {"peak": max(0.0, cell["fin"] + cell["gap"] - 100.0), "mass": cell["fin"] + cell["gap"]}
'''


@pytest.mark.asyncio
async def test_a_search_with_an_objective_that_never_moves_ends_as_a_result_not_a_fault(tmp_path: Path) -> None:
    root = _quest(tmp_path, FLAT_OBJECTIVE)
    protocol = {"optimisation": _block(),
                "metrics": [{"id": "mass", "kind": "mean", "label": "fin mass", "estimand": "the mass of the fins",
                             "unit": "one design"}]}
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol=protocol,
                                         simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=300)
    # The search handed its fact back at once (the plan is revised for it): no check and no analysis ran.
    assert result.returncode == 1 and runner.flat is None and runner.flat_objective, result.stderr[-1500:]
    assert runner.flat_objective["designs"] == search.NO_EFFECT_DESIGNS
    evaluations = (optimise.read(root) or {}).get("evaluations") or {}
    assert evaluations.get("stopped_because") == "flat_objective"
    assert evaluations.get("search") == 1 + search.NO_EFFECT_DESIGNS
    assert any("no direction to follow" in line for line in optimise.summary_lines(optimise.read(root)))


def test_a_local_search_near_the_baseline_is_not_taken_for_a_flat_objective() -> None:
    """Steps near the baseline show nothing of the rest of the range: a local search ends as it did before."""
    outcome = _drive(_local_block(), _flat, ["mass"])
    assert outcome["stopped_because"] != "flat_objective" and "flat_objective" not in outcome


def test_a_coarse_scan_whose_objective_moves_ends_the_watch() -> None:
    """The scan's settings under the search's conditions are designs too: one that gives another objective shows the
    objective moves, whatever the first designs of the search give."""
    def window(cell: dict[str, Any]) -> dict[str, Any]:
        return {"peak": -5.0 if cell["fin"] >= 5.5 and cell["gap"] >= 5.5 else 0.0, "mass": cell["fin"]}

    block = _block(grid={"fin": [1.0, 6.0], "gap": [1.0, 6.0]})
    outcome = _drive_with_scan(block, window, [{"fin": 6.0, "gap": 6.0}, {"fin": 1.0, "gap": 6.0}])
    assert outcome["stopped_because"] != "flat_objective" and "flat_objective" not in outcome
    assert outcome["best"]["objective"] == -5.0


def test_a_flat_coarse_scan_counts_toward_the_watch() -> None:
    """A scan of 64 settings across the range, all at the baseline's objective: the search stops after it, with only
    the baseline counted against its budget (the scan is not)."""
    values = [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6]
    block = _block(grid={"fin": values, "gap": values})
    outcome = _drive_with_scan(block, _flat, [{"fin": f, "gap": g} for f in values for g in values])
    assert outcome["stopped_because"] == "flat_objective"
    assert outcome["evaluations"] == 1 and outcome["flat_objective"]["designs"] >= search.NO_EFFECT_DESIGNS


def test_designs_that_meet_the_limits_differently_from_the_baseline_end_the_watch() -> None:
    """The same objective is not the same design when the baseline breaks a limit the others meet: the best design is
    then not the baseline."""
    block = _block(constraints=[{"quantity": "mass", "limit": "<= 3", "unit": "g"}])
    outcome = _drive(block, lambda cell: {"peak": 0.0, "mass": 4.0 if cell["fin"] == 1.0 and cell["gap"] == 1.0
                                          else cell["fin"] * 0.4}, ["mass"])
    assert outcome["stopped_because"] != "flat_objective" and "flat_objective" not in outcome


def test_the_paper_and_the_budget_check_name_the_flat_objective_in_words() -> None:
    assert "all gave the baseline's objective" in best_design_report._STOPPED["flat_objective"]
    import inspect
    assert inspect.getsource(optimum_check).count("all gave the baseline's objective") == 1, "the budget check's words"


def test_a_flat_scan_is_decided_on_all_its_settings_never_cut_short() -> None:
    """The 64th setting of the scan is better: every setting is recorded before the stop is decided, and that one ends
    the watch, so the better design is found, not lost behind an early stop."""
    values = [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6]

    def last_better(cell: dict[str, Any]) -> dict[str, Any]:
        return {"peak": -9.0 if cell["fin"] == 5.6 and cell["gap"] == 5.6 else 0.0, "mass": cell["fin"]}

    block = _block(grid={"fin": values, "gap": values})
    outcome = _drive_with_scan(block, last_better, [{"fin": f, "gap": g} for f in values for g in values])
    assert outcome["stopped_because"] != "flat_objective" and "flat_objective" not in outcome
    assert outcome["best"]["objective"] == -9.0


def test_a_flat_search_that_went_on_is_said_in_the_report_and_the_log() -> None:
    fact = {"designs": 50, "objective": 0.0, "feasible": True, "stayed": [], "moved": [], "stopped": False}
    ev = {"search": 200, "budget": 200, "stopped_because": "budget", "flat_objective": fact}
    said = best_design_report.flat_went_on(ev)
    assert "did not move" in said and "first 50 designs" in said and "went on to its budget" in said
    assert any("did not move" in line for line in optimise.summary_lines({"evaluations": ev}))
    # A search that stopped for it says so through its stop reason, not twice.
    assert best_design_report.flat_went_on({**ev, "flat_objective": {**fact, "stopped": True}}) == ""
    assert best_design_report.flat_went_on({"stopped_because": "budget"}) == ""


# The objective moves, but only between a few levels (a quantity counted in whole steps of a setting): the search has
# no direction to follow either, and stops the same way, with the levels in its record.
def _levels(cell: dict[str, Any]) -> dict[str, Any]:
    return {"peak": -10.0 * int(cell["fin"] // 2.5), "mass": cell["fin"]}


def test_an_objective_that_moves_only_between_a_few_levels_stops_the_search() -> None:
    outcome = _drive(_block(), _levels, ["mass"])
    assert outcome["stopped_because"] == "flat_objective"
    fact = outcome["flat_objective"]
    assert fact["designs"] == search.NO_EFFECT_DESIGNS and fact["start"] == 0.0 and fact["stopped"] is True
    assert sorted(level[0] for level in fact["values"]) == [-20.0, -10.0, 0.0]
    assert sum(level[1] for level in fact["values"]) == search.NO_EFFECT_DESIGNS + 1, "the designs and the baseline"
    assert fact["values"][0][1] == max(level[1] for level in fact["values"]), "the most common level first"


def test_an_objective_with_more_levels_than_a_few_is_searched_as_before() -> None:
    outcome = _drive(_block(), lambda cell: {"peak": -float(int(cell["fin"] // 1.5)), "mass": cell["fin"]}, ["mass"])
    assert "flat_objective" not in outcome and outcome["evaluations"] > 1 + search.NO_EFFECT_DESIGNS


def test_a_count_over_a_wide_range_is_searched_as_before() -> None:
    """A whole-number objective is not a few levels when it takes many values across the range."""
    outcome = _drive(_block(), lambda cell: {"peak": float(round(20 * cell["fin"] + 3 * cell["gap"])),
                                             "mass": cell["fin"]}, ["mass"])
    assert "flat_objective" not in outcome and outcome["evaluations"] > 1 + search.NO_EFFECT_DESIGNS


def test_the_plan_is_told_the_levels_in_words() -> None:
    from core import engine
    fact = {"designs": 50, "objective": 0.0, "values": [[0.0, 46, True], [40.0, 4, True], [20.0, 1, False]],
            "start": 20.0, "stayed": [], "moved": [], "stopped": True}
    said = engine._flat_fact_sentence(fact)
    assert "only 3 different values: 0 (46 of them), 40 (4 of them), 20 (1 of them, not meeting every limit)" in said
    assert "the starting design gave 20" in said
    asked = engine._flat_request(fact)
    assert "jumps between a few levels" in asked and "moved only between a few values." in asked
    assert "returns no result besides the objective" not in asked
    went_on = best_design_report.flat_went_on({"flat_objective": {**fact, "stopped": False}})
    assert "only between a few values" in went_on and "0, 40, 20" in went_on


def test_a_narrow_window_touched_by_single_designs_is_searched_as_before() -> None:
    """Most designs give the common value and two give values of their own: the objective may move smoothly inside a
    region the sampled designs only touched, so it is not taken for steps."""
    def window(cell: dict[str, Any]) -> dict[str, Any]:
        inside = 5.7 <= cell["fin"] and 5.4 <= cell["gap"]
        return {"peak": -(cell["fin"] + cell["gap"]) if inside else 0.0, "mass": cell["fin"]}

    outcome = _drive(_block(), window, ["mass"])
    assert "flat_objective" not in outcome
