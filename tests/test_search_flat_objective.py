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

from core import optimise
from core import optimise_search as search
from tests.test_flat_output import _quest
from tests.test_optimise_runner import LocalExecutor
from tests.test_search_no_effect import _block, _drive


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
