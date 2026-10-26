"""A search whose objective is counted in whole steps of one of its own numerical settings stops early
(core/optimise_search.py, ``stepped_objective``), and the simulation is sent back once to read it between the steps.

A live search spent its whole budget on an objective that was a length counted as the number of scan points inside a
window times their spacing: it moved only in whole steps of that spacing, most designs tied, and the search could not
tell them apart. FI holds the value and the unit of every numerical setting of the search, so it sees when the
objective's values sit on one setting's step. Neutral toy topic only (a heat sink).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from core import best_design_report, optimise
from core import optimise_search as search
from tests.test_flat_output import _quest
from tests.test_optimise_runner import LocalExecutor
from tests.test_search_no_effect import _block as _local_block, _drive

OBJECTIVE = {"quantity": "gap", "direction": "maximise", "unit": "mm", "meaning": "the widest gap that stays cool"}


def _block(unit: Any = "mm", **changes: Any) -> dict[str, Any]:
    """The toy search with a step setting of 0.25 mm, sampled across its range first."""
    setting = {"search": 0.25, "check": [0.125], "finer": "smaller", **({"unit": unit} if unit else {})}
    return _local_block(**{"search_method": "global_then_local", "evaluation_budget": {"starts": 2, "per_start": 100},
                           "objective": OBJECTIVE, "numerical_settings": {"step": setting}, **changes})


def _counted(cell: dict[str, Any]) -> dict[str, Any]:
    """A length counted in whole steps of 0.25 mm: many levels, all on the step."""
    return {"gap": 0.25 * int(4 * (cell["fin"] + cell["gap"])), "mass": cell["fin"]}


def _smooth(cell: dict[str, Any]) -> dict[str, Any]:
    return {"gap": 1.0 + cell["fin"] * 0.37 + cell["gap"] ** 1.5 / 7.0, "mass": cell["fin"]}


def test_an_objective_on_the_step_of_a_setting_stops_the_search_and_names_the_setting() -> None:
    outcome = _drive(_block(), _counted, ["mass"])
    assert outcome["stopped_because"] == "stepped_objective"
    fact = outcome["stepped_objective"]
    assert fact["setting"] == "step" and fact["step"] == 0.25 and fact["unit"] == "mm" and fact["stopped"] is True
    assert fact["designs"] == search.NO_EFFECT_DESIGNS and fact["on_step"] == fact["nonzero"] >= search.STEPPED_MIN_VALUES
    assert "flat_objective" not in outcome, "the few-values watch is not reached: the step is named first"


def test_a_smooth_objective_is_searched_as_before() -> None:
    outcome = _drive(_block(), _smooth, ["mass"])
    assert "stepped_objective" not in outcome and outcome["evaluations"] > 1 + search.NO_EFFECT_DESIGNS


def test_a_setting_without_the_objectives_unit_is_not_a_step_of_it() -> None:
    """A whole-number result is a multiple of any setting of value 1 or 0.25: only a setting in the objective's own unit
    (written in the plan) can be the step it is counted in; with none written, nothing is decided."""
    for unit in ("s", None):
        outcome = _drive(_block(unit=unit), _counted, ["mass"])
        assert "stepped_objective" not in outcome, unit


def test_a_setting_that_is_finer_when_larger_is_not_a_step() -> None:
    block = _block(numerical_settings={"step": {"search": 0.25, "check": [0.5], "finer": "larger", "unit": "mm"}})
    assert "stepped_objective" not in _drive(block, _counted, ["mass"])


def test_after_the_repair_the_fact_is_recorded_and_the_search_runs_on() -> None:
    gen = search.search(_block(), seed=0, results=["mass"], stop_stepped=False)
    request = next(gen)
    try:
        while True:
            request = gen.send([] if request["kind"] == "scan" else {"values": _counted(request["cell"])})
    except StopIteration as done:
        outcome = done.value
    assert outcome["stopped_because"] != "stepped_objective"
    assert outcome["stepped_objective"]["stopped"] is False
    said = best_design_report.stepped_went_on({"stepped_objective": outcome["stepped_objective"]})
    assert "whole steps of step (0.25 mm)" in said and "went on to its budget" in said
    assert best_design_report.stepped_went_on({"stepped_objective": {**outcome["stepped_objective"], "stopped": True}}) == ""


COUNTED = '''\
def run_cell(cell):
    return {"gap": 0.25 * int(4 * (cell["fin"] + cell["gap"])), "mass": cell["fin"] + cell["gap"]}
'''


@pytest.mark.asyncio
async def test_the_simulation_is_sent_back_once_with_the_setting_named(tmp_path: Path) -> None:
    root = _quest(tmp_path, COUNTED)
    protocol = {"optimisation": _block(),
                "metrics": [{"id": "mass", "kind": "mean", "label": "fin mass", "estimand": "the mass of the fins",
                             "unit": "one design"}]}
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol=protocol,
                                         simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
                                         repairs_left=True)
    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    result = await runner.execute(cmd, cwd=root, timeout_s=300)
    assert result.returncode == 1 and runner.failed_script == "simulate.py", result.stderr[-1500:]
    assert "whole multiple of step (0.25 mm" in result.stderr and "read between the points" in result.stderr
    assert (root / optimise.STEPPED_SENT).is_file()
    # Once per quest: the next search (the same code here) only records the fact and goes on.
    again = await runner.execute(cmd, cwd=root, timeout_s=300)
    evaluations = (optimise.read(root) or {}).get("evaluations") or {}
    assert evaluations.get("stopped_because") != "stepped_objective", again.stderr[-1500:]
    assert evaluations.get("stepped_objective", {}).get("stopped") is False


@pytest.mark.asyncio
async def test_no_repair_left_means_no_stop(tmp_path: Path) -> None:
    root = _quest(tmp_path, COUNTED)
    protocol = {"optimisation": _block(),
                "metrics": [{"id": "mass", "kind": "mean", "label": "fin mass", "estimand": "the mass of the fins",
                             "unit": "one design"}]}
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol=protocol,
                                         simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
                                         repairs_left=False)
    await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=300)
    evaluations = (optimise.read(root) or {}).get("evaluations") or {}
    assert evaluations.get("stopped_because") != "stepped_objective"
    assert not (root / optimise.STEPPED_SENT).is_file()
