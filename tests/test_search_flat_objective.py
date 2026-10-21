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
    assert outcome["flat_objective"] == {"designs": search.NO_EFFECT_DESIGNS, "objective": 0.0}
    assert outcome["best"]["objective"] == 0.0


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
    assert result.returncode == 0 and runner.flat is None, result.stderr[-1500:]
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
