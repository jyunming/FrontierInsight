"""A search whose design changes none of its results stops early (core/optimise_search.py, ``no_effect``).

A live search spent its whole budget of 4000 evaluations on a simulation that, at the setting the search held, never
used the design: every result the plan names came back the same at every design. FI's own check found it only after
the budget was spent, and the reading of the code that its repair waits for saw the design read (in a branch the held
setting never took), so the run went on. Now the search stops once its first designs, which between them moved every
number of the design, all gave the baseline's results, and the simulation is sent back to be repaired.
Neutral toy topic only (a heat sink).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from core import optimisation_plan as op
from core import optimise
from core import optimise_search as search
from tests.test_flat_output import _quest
from tests.test_optimise_runner import LocalExecutor


def _block(**changes: Any) -> dict[str, Any]:
    block = {
        "objective": {"quantity": "peak", "direction": "minimise", "unit": "K", "meaning": "the peak temperature"},
        "design_variables": [{"name": "fin", "low": 0.0, "high": 6.0, "kind": "continuous", "unit": "mm"},
                             {"name": "gap", "low": 0.0, "high": 6.0, "kind": "continuous", "unit": "mm"}],
        "baseline": {"values": {"fin": 1.0, "gap": 1.0}, "source": "the design in use"},
        "numerical_settings": {"mesh": {"search": 0.4}},
        "evaluation_budget": {"starts": 2, "per_start": 40},
        "search_method": "bounded_local",
    }
    block.update(copy.deepcopy(changes))
    fixed, why = op.normalize(block)
    assert why is None, why
    return fixed


def _drive(block: dict[str, Any], answer: Callable[[dict[str, Any]], dict[str, Any]],
           results: list[str] | None) -> dict[str, Any]:
    gen = search.search(block, seed=0, results=results)
    request = next(gen)
    try:
        while True:
            reply: Any = [] if request["kind"] == "scan" else {"values": answer(request["cell"])}
            request = gen.send(reply)
    except StopIteration as done:
        return done.value


def _still(_cell: dict[str, Any]) -> dict[str, Any]:
    return {"peak": 40.0, "mass": 2.0}


def _moving(cell: dict[str, Any]) -> dict[str, Any]:
    return {"peak": 40.0, "mass": 2.0 + cell["fin"]}


def test_a_design_that_changes_no_result_stops_the_search_after_its_first_designs() -> None:
    outcome = _drive(_block(), _still, ["mass"])
    assert outcome["stopped_because"] == "no_effect"
    assert outcome["evaluations"] == 1 + search.NO_EFFECT_DESIGNS, "the baseline and the designs that showed it"
    assert outcome["no_effect"] == {"designs": search.NO_EFFECT_DESIGNS, "results": ["mass", "peak"]}


def test_a_plateau_of_the_objective_with_a_result_that_moves_is_not_a_design_that_changes_nothing() -> None:
    """The design acts (the mass moves): not ``no_effect``. The objective that never moves stops the search for that
    reason instead (tests/test_search_flat_objective.py)."""
    outcome = _drive(_block(search_method="global_then_local", evaluation_budget={"starts": 2, "per_start": 100}), _moving, ["mass"])
    assert outcome["stopped_because"] == "flat_objective" and "no_effect" not in outcome


def test_without_a_result_beside_the_objective_and_the_limits_the_search_is_not_stopped() -> None:
    """The objective alone can sit on a plateau (a live search's objective was the same for its first 18 designs and then
    moved): with nothing else to compare, nothing is decided early."""
    for results in (None, [], ["peak"]):
        outcome = _drive(_block(), _still, results)
        assert outcome["stopped_because"] != "no_effect", results
    outcome = _drive(_block(), lambda cell: {"peak": 40.0}, ["mass"])
    assert outcome["stopped_because"] != "no_effect", "a named result the simulation does not return is not compared"


def test_the_stop_waits_until_the_designs_have_moved_every_number_of_the_design() -> None:
    """Designs that each move one number prove nothing about the numbers not yet moved: with more numbers than the designs
    needed to show it, the watch does not stop until every one has moved. Driven on the watch itself: the search's own
    first designs move many numbers at once (the coarse scan), so a whole search never shows the case."""
    names = [f"w{i}" for i in range(80)]
    many = [{"name": n, "low": 0.0, "high": 6.0, "kind": "continuous", "unit": "mm"} for n in names]
    block = _block(design_variables=many, baseline={"values": {n: 1.0 for n in names}, "source": "in use"},
                   evaluation_budget={"starts": 1, "per_start": 200})
    s = search._Search(block, 0, "bounded_local", results=["mass"])
    base = dict(s.baseline)
    values = {"peak": 40.0, "mass": 2.0}
    s._watch_effect(base, s.space.key(base), values)
    for i, name in enumerate(names):
        design = {**base, name: 2.0}  # one number moved, the rest still at the baseline
        if i < len(names) - 1:
            s._watch_effect(design, s.space.key(design), values)
            assert s.stopped is None, f"stopped after {i + 1} designs, with {len(names) - 1 - i} numbers never moved"
        else:
            with pytest.raises(search._Stop):
                s._watch_effect(design, s.space.key(design), values)
    assert s.stopped == "no_effect" and s.no_effect["designs"] == len(names)


TWO_MODES = '''\
def run_cell(cell):
    if cell["mode"] == "detailed":
        return {"peak": 30.0 + (cell["fin"] - 3.0) ** 2, "mass": cell["fin"] + cell["gap"]}
    return {"peak": 40.0, "mass": 2.0}
'''


@pytest.mark.asyncio
async def test_a_simulation_that_leaves_the_design_out_at_the_held_setting_is_sent_back_early(tmp_path: Path) -> None:
    """The code reads the design (in a branch the held setting never takes), so a reading of the code does not show it
    unused: what FI measured decides."""
    root = _quest(tmp_path, TWO_MODES)
    protocol = {"optimisation": _block(fixed={"mode": "simple"}),
                "metrics": [{"id": "mass", "kind": "mean", "label": "fin mass", "estimand": "the mass of the fins", "unit": "one design"}]}
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol=protocol,
                                         simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=300)
    assert result.returncode == 1 and runner.failed_script == "simulate.py", result.stderr[-1500:]
    assert runner.flat and "at the settings the search holds (mode = simple)" in runner.flat
    assert f"the first {search.NO_EFFECT_DESIGNS} designs tried" in runner.flat
    record = optimise.read(root) or {}
    assert (record.get("evaluations") or {}).get("search") == 1 + search.NO_EFFECT_DESIGNS
    assert not (root / "needs" / "OPTIMUM_CHECK.json").exists(), "no check was spent on a result that is not usable"


@pytest.mark.asyncio
async def test_the_same_simulation_at_the_setting_that_uses_the_design_searches_as_before(tmp_path: Path) -> None:
    root = _quest(tmp_path, TWO_MODES)
    protocol = {"optimisation": _block(fixed={"mode": "detailed"}),
                "metrics": [{"id": "mass", "kind": "mean", "label": "fin mass", "estimand": "the mass of the fins", "unit": "one design"}]}
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol=protocol,
                                         simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=300)
    assert result.returncode == 0, result.stderr[-1500:]
    assert runner.flat is None
    assert ((optimise.read(root) or {}).get("evaluations") or {}).get("stopped_because") != "no_effect"


@pytest.mark.asyncio
async def test_a_result_the_plan_expects_to_stay_put_is_not_watched_for_the_designs_effect(tmp_path: Path) -> None:
    """A result the plan itself expects to be constant (a declared bound's quantity, an invariant) says nothing about
    the design when it stays put: watched, it stopped a search whose design does act on the objective."""
    root = _quest(tmp_path, TWO_MODES)
    protocol = {"optimisation": _block(fixed={"mode": "simple"}, search_method="global_then_local", evaluation_budget={"starts": 2, "per_start": 100}),
                "metrics": [{"id": "mass", "kind": "mean", "label": "fin mass", "estimand": "the mass of the fins", "unit": "one design"}],
                "bounds": {"mass": [0.0, 10.0]}}
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol=protocol,
                                         simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")
    await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=300)
    evaluations = (optimise.read(root) or {}).get("evaluations") or {}
    assert "no_effect" not in evaluations
    # The objective never moves either (the simple mode returns a constant): that stop, not the design's effect.
    assert evaluations.get("stopped_because") == "flat_objective"
