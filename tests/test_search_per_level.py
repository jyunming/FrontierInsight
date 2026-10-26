"""A study that compares named alternatives (a family of designs, a mode) and also searches for the best design.

FI owns the comparison: one search for each alternative, each with the plan's own variables, ranges and budget, every
evaluation recorded under its alternative, and the comparison made between the best designs. Before, only the first
alternative the plan named was searched; the others were only measured in the coarse scan. Neutral toy simulation: three
"cooling cups" whose best size and best value differ, the first named one the worst.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import optimisation_plan as op
from core import optimise
from core import optimise_search as osearch
from core import optimum_check
from tests.test_optimise_runner import LocalExecutor

CENTRE = {"cup": (-3.0, 4.0), "mug": (1.0, 2.0), "bowl": (3.0, 1.0)}  # (best x, best value of f)

BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"}],
    "fixed": {"shape": "cup"},
    "grid": {"shape": ["cup", "mug", "bowl"], "x": [-4.0, 0.0, 4.0]},
    "baseline": {"values": {"x": 0.0}, "source": "the design in use"},
    "numerical_settings": {"mesh": {"search": 0.5}},
    "evaluation_budget": {"starts": 2, "per_start": 30},
    "search_method": "bounded_local",
}

SIMULATE = '''
CENTRE = {"cup": (-3.0, 4.0), "mug": (1.0, 2.0), "bowl": (3.0, 1.0)}


def run_cell(cell):
    x0, best = CENTRE[cell["shape"]]
    return {"f": best + (cell["x"] - x0) ** 2, "mesh_seen": cell["mesh"]}
'''


def _block(**changes: Any) -> dict[str, Any]:
    block = copy.deepcopy(BLOCK)
    block.update(changes)
    fixed, why = op.normalize(block)
    assert why is None, why
    return fixed


def _f(cell: dict[str, Any]) -> dict[str, float]:
    x0, best = CENTRE[cell["shape"]]
    return {"f": best + (cell["x"] - x0) ** 2}


def test_the_alternatives_are_the_names_on_the_scan_of_a_condition_the_plan_holds_fixed() -> None:
    assert osearch.compared_levels(_block()) == [{"shape": "cup"}, {"shape": "mug"}, {"shape": "bowl"}]
    # the one the plan holds comes first, whatever the order of the scan
    held = _block(fixed={"shape": "mug"})
    assert osearch.compared_levels(held)[0] == {"shape": "mug"} and osearch.level_count(held) == 3
    # numbers compared across a condition stay a comparison the scan measures (not a search for each)
    numeric = _block(fixed={"scale": 1.0}, grid={"scale": [0.5, 1.0, 2.0], "x": [-1.0, 1.0]})
    assert osearch.compared_levels(numeric) == [] and osearch.level_count(numeric) == 1
    assert osearch.compared_levels(_block(grid={"x": [-1.0, 1.0]})) == []


def test_the_plans_evaluation_count_is_one_budget_for_each_alternative() -> None:
    assert op.evaluations(_block())[0] == 2 * 30 * 3
    assert op.evaluations(_block(grid={"x": [-1.0, 1.0]}))[0] == 2 * 30


def _drive(block: dict[str, Any], _unused: Any = None) -> dict[str, Any]:
    return osearch.run_sync(block, lambda cell: _f(cell), seed=0)


def test_each_alternative_is_searched_and_the_best_of_them_wins_though_it_is_not_the_first_named() -> None:
    block = _block()
    outcome = _drive(block, None)
    assert [p["level"]["shape"] for p in outcome["levels"]] == ["cup", "mug", "bowl"]
    for p in outcome["levels"]:
        want_x, want_f = CENTRE[p["level"]["shape"]]
        assert p["best"] is not None and abs(p["best"]["design"]["x"] - want_x) < 0.05, p
        assert abs(p["best"]["objective"] - want_f) < 5e-3
    assert outcome["best"]["level"] == {"shape": "bowl"} and abs(outcome["best"]["objective"] - 1.0) < 5e-3
    # the baseline to beat is the winner's own (the same design at the same conditions)
    assert outcome["baseline"]["objective"] == pytest.approx(1.0 + 9.0)
    assert outcome["budget"] == 180 and outcome["evaluations"] <= 180
    # every evaluation says its alternative and ran under it
    assert {tuple(r["level"].items()) for r in outcome["rows"]} == {(("shape", s),) for s in CENTRE}
    assert [r["n"] for r in outcome["rows"]] == list(range(1, len(outcome["rows"]) + 1))


def test_only_the_first_alternative_was_searched_before() -> None:
    """The same plan driven through the single search (what FI ran before): the first name wins by default."""
    block = _block()
    gen = osearch.search(block, seed=0)
    request = next(gen)
    answer = None
    try:
        while True:
            if request["kind"] == "scan":
                answer = [{"design": {"x": x}, "values": _f({"shape": "cup", "x": x})} for x in (-4.0, 0.0, 4.0)]
            else:
                answer = {"values": _f(request["cell"])}
            request = gen.send(answer)
    except StopIteration as done:
        old = done.value
    assert old["best"]["objective"] == pytest.approx(4.0, abs=5e-3), "the cup, not the best of the three"
    assert "levels" not in old


def test_the_record_names_the_alternatives_the_winner_and_the_conditions_it_won_under() -> None:
    block = _block()
    outcome = _drive(block, None)
    record = osearch.best_design(outcome, block, seed=0)
    assert record["fixed"]["shape"] == "bowl"
    assert [p["level"]["shape"] for p in record["compared"]["levels"]] == ["cup", "mug", "bowl"]
    assert "best of 3 alternatives" in record["says"] and "the winner is shape = bowl" in record["says"]
    head = json.loads(osearch.ledger_lines(outcome, block)[0])
    assert head["compared"] == [{"shape": "cup"}, {"shape": "mug"}, {"shape": "bowl"}]


def test_a_search_over_no_named_alternative_is_exactly_the_search_it_was() -> None:
    block = _block(grid={"x": [-4.0, 0.0, 4.0]}, fixed={"shape": "mug"})
    outcome = _drive(block, None)
    assert "levels" not in outcome and "level" not in outcome["rows"][0]
    assert abs(outcome["best"]["objective"] - 2.0) < 5e-3
    assert "compared" not in osearch.best_design(outcome, block, seed=0)


def test_a_design_evaluated_under_one_alternative_is_not_taken_for_the_same_design_under_another() -> None:
    space = osearch.Space(BLOCK["design_variables"])
    a = optimise._replay_key(space, {"x": 1.0}, {"shape": "cup"})
    b = optimise._replay_key(space, {"x": 1.0}, {"shape": "mug"})
    assert a != b and a == optimise._replay_key(space, {"x": 1.0}, {"shape": "cup"})
    assert optimise._replay_key(space, {"x": 1.0}, None) != a


def test_the_check_at_finer_settings_is_made_under_the_winning_alternative() -> None:
    block = _block()
    outcome = _drive(block, None)
    record = osearch.best_design(outcome, block, seed=0)
    asked: list[dict[str, Any]] = []
    gen = optimum_check.check(block, record, outcome["rows"])
    request = next(gen)
    try:
        while True:
            reqs = request["requests"] if request["kind"] == "evaluate_many" else [request]
            for one in reqs:
                asked.append(one["cell"])
            answer = ([{"values": _f(r["cell"])} for r in reqs] if request["kind"] == "evaluate_many"
                      else {"values": _f(request["cell"])})
            request = gen.send(answer)
    except StopIteration:
        pass
    assert asked and all(c["shape"] == "bowl" for c in asked), {c["shape"] for c in asked}


@pytest.mark.asyncio
async def test_fi_runs_one_search_per_alternative_measures_the_scan_once_and_records_each(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    block = _block(evaluation_budget={"starts": 1, "per_start": 20})
    run = await optimise.run_search(LocalExecutor(), sys.executable, root, "code/simulate.py", {"optimisation": block},
                                    base_seed=0, timeout_s=300)
    record = run.record
    assert record["best"]["level"] == {"shape": "bowl"} and abs(record["best"]["objective"] - 1.0) < 0.05
    assert record["fixed"]["shape"] == "bowl"
    assert [p["level"]["shape"] for p in record["compared"]["levels"]] == ["cup", "mug", "bowl"]
    trials = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    assert sorted({c["cell"]["shape"] for c in trials["cells"]}) == ["bowl", "cup", "mug"]
    assert len(trials["cells"]) == 9, "the scan ran once, every alternative at every value"
    ledger = [json.loads(line) for line in (root / "raw" / "optimisation_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in ledger if r.get("event") == "evaluation"]
    for shape in CENTRE:
        mine = [r for r in rows if r["level"] == {"shape": shape}]
        assert mine[0]["stage"] == "baseline" and any(r["stage"] == "scan" for r in mine), shape
        assert any(r["stage"] not in ("baseline", "scan") for r in mine), f"{shape} was searched, not only scanned"
    best = json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))
    assert best == record


@pytest.mark.asyncio
async def test_a_search_stopped_part_way_goes_on_without_mixing_the_alternatives(tmp_path: Path) -> None:
    """The journal keeps each answer under its alternative: the same design under another one is evaluated again."""
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    block = _block(evaluation_budget={"starts": 1, "per_start": 8}, grid={"shape": ["cup", "mug"]})
    space = osearch.Space(block["design_variables"])
    path = root / ".fi" / "journal.jsonl"
    path.write_text(json.dumps({"key": "k"}) + "\n", encoding="utf-8")
    optimise._journal(path, {"x": 1.0}, {"values": {"f": 123.0}}, {"shape": "cup"})
    answers, _s = optimise._read_journal(path, "k", space)
    assert list(answers) == [optimise._replay_key(space, {"x": 1.0}, {"shape": "cup"})]
    assert optimise._replay_key(space, {"x": 1.0}, {"shape": "mug"}) not in answers
