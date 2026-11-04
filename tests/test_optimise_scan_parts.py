"""A search for the best design whose coarse scan is a measurement of its own: the scan divided into parts
(``optimisation.sub_studies``, each varying only some of its settings, as a measurement's sub-studies do), and values
every evaluation of a design goes through together (``optimisation.in_each_run``).

Before, a search's scan was every combination of its settings, and nothing went through every evaluation together: a
topic that compares alternatives with parameters of their own and also asks for the best design could not be planned as a
search at all (the plan was refused for leaving the comparison out, or written as a measurement with the search hidden in
the simulation). Neutral toy simulation (tests/test_optimise_runner.py)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core import optimisation_plan as op
from core import optimise_search as osearch
from core import plan as plan_mod
from core import topic_requirements as tr
from tests.test_optimise_runner import BLOCK, _ledger, _quest, _run

# Two kinds of load the design is compared under, each with a parameter of its own, plus the condition the search holds.
PARTS: dict[str, Any] = {
    "fixed": {"scale": 1.0, "kind": "spread", "spread": 0.0, "offset": 0.0},
    "grid": {"kind": ["spread", "offset"], "spread": [0.0, 1.0, 2.0], "offset": [0.0, 0.5]},
    "sub_studies": [
        {"name": "spread", "vary": ["spread"], "hold": {"kind": "spread", "offset": 0.0}},
        {"name": "offset", "vary": ["offset"], "hold": {"kind": "offset", "spread": 0.0}},
    ],
}


def _parts(**changes: Any) -> dict[str, Any]:
    block = copy.deepcopy({**BLOCK, **PARTS})
    block.update(changes)
    return block


def test_a_scan_in_parts_runs_the_settings_of_its_parts_not_every_combination() -> None:
    block, why = op.normalize(_parts())
    assert why is None, why
    grid = osearch.scan_grid(block)
    assert isinstance(grid, osearch.StudyGrid)
    cells = osearch.grid_cells(grid)
    # 3 spreads of one kind + 2 offsets of another: 5 settings, where every combination would be 2 x 3 x 2 = 12.
    assert len(cells) == osearch.grid_size(grid) == op.scan_size(block) == 5
    assert op.evaluations(block)[1] == 5
    assert {(c["kind"], c["spread"], c["offset"]) for c in cells} == {
        ("spread", 0.0, 0.0), ("spread", 1.0, 0.0), ("spread", 2.0, 0.0), ("offset", 0.0, 0.0), ("offset", 0.0, 0.5)}
    assert all(c["x"] == 0.0 and c["y"] == 0.0 and c["mesh"] == 0.5 for c in cells), "the baseline design, search settings"


def test_a_scan_without_parts_is_every_combination_as_before() -> None:
    block, why = op.normalize(_parts(sub_studies=None))
    block = {k: v for k, v in block.items() if k != "sub_studies"} if block else block
    assert why is None and not isinstance(osearch.scan_grid(block), osearch.StudyGrid)
    assert op.scan_size(block) == 12


def test_parts_that_do_not_read_are_refused_with_the_search_s_own_names() -> None:
    _, why = op.normalize(_parts(sub_studies=[{"name": "bad", "vary": ["unlisted"], "hold": {}}]))
    assert why and "`protocol.optimisation.sub_studies`" in why and "`protocol.optimisation.grid`" in why
    assert "`protocol.grid`" not in why and "`protocol.sub_studies`" not in why
    # A setting with several values that no part varies or holds at every value would leave some never run.
    _, why = op.normalize(_parts(sub_studies=[{"name": "spread", "vary": ["spread"], "hold": {"kind": "spread", "offset": 0.0}}]))
    assert why and "would never be run" in why
    _, why = op.normalize({**BLOCK, "sub_studies": [{"name": "a", "vary": ["x"], "hold": {}}]})
    assert why and "which is missing" in why, "parts divide a scan: without one there is nothing to divide"


def test_values_every_evaluation_goes_through_together_reach_each_design_and_are_not_scan_settings() -> None:
    block, why = op.normalize({**BLOCK, "in_each_run": {"case": [1, 2, 3]}, "grid": {"x": [-1.0, 1.0]}})
    assert why is None, why
    assert osearch.cell_of(block, {"x": 1.0, "y": 0.0})["case"] == [1, 2, 3], "each evaluation gets the whole list"
    assert "case" not in osearch.scan_grid(block), "the trial runner gives the scan's cells the list, not as a setting"
    for bad, words in (({"case": [1]}, "two or more values"), ({"x": [1, 2]}, "it is one of them"),
                       ({"scale": [1, 2]}, "it is one of them"), ("all", "must map")):
        _, why = op.normalize({**BLOCK, "in_each_run": bad})
        assert why and words in why, (bad, why)


def test_the_plan_moves_parts_and_values_written_beside_the_block_into_it() -> None:
    raw = {"oracles": [{"name": "o"}], "optimisation": _parts(sub_studies=None),
           "sub_studies": copy.deepcopy(PARTS["sub_studies"]), "in_each_run": {"case": [1, 2]}}
    raw["optimisation"].pop("sub_studies", None)
    fixed, why = plan_mod.normalize_protocol(raw)
    assert fixed is None and why and "cannot stand beside `protocol.optimisation`" in why
    repaired, notes = plan_mod.repair_protocol(raw)
    assert repaired is not None, notes
    assert "sub_studies" not in repaired and "in_each_run" not in repaired
    assert repaired["optimisation"]["sub_studies"][0]["vary"] == ["spread"]
    assert repaired["optimisation"]["in_each_run"] == {"case": [1, 2]}
    assert any("moved into `protocol.optimisation.sub_studies`" in n for n in notes), notes


def test_the_topic_check_sees_what_every_evaluation_goes_through() -> None:
    view = tr.view({"protocol": {"optimisation": {**BLOCK, "in_each_run": {"case": [1, 2, 3]}}}})
    assert view["axes"]["case"] == [1, 2, 3]


def test_a_scan_in_parts_that_is_too_large_says_each_part() -> None:
    big = _parts(grid={"kind": ["spread", "offset"], "spread": list(range(400)), "offset": [0.0, 0.5]},
                 sub_studies=[{"name": "spread", "vary": ["spread"], "hold": {"kind": "spread", "offset": 0.0}},
                              {"name": "offset", "vary": ["offset"], "hold": {"kind": "offset", "spread": 0}}],
                 fixed={"scale": 1.0, "kind": "spread", "spread": 0, "offset": 0.0},
                 evaluation_budget={"starts": 1, "per_start": 5, "seconds_per_evaluation": 10})
    sentence = op.scan_problem({"protocol": {"optimisation": big}}, limit_s=600)
    assert "it is 402 settings" in sentence and "`spread` 400" in sentence and "sub_studies" in sentence
    assert "each multiplies every other" not in sentence


SIMULATE_CASES = '''\
def run_cell(cell):
    x, y = cell["x"], cell["y"]
    cases = cell["case"]
    # one design judged over every case at once: the worst case is the objective
    worst = max(c * ((x - 2) ** 2 + (y + 1) ** 2) + c for c in cases)
    bump = {"spread": cell["spread"], "offset": cell["offset"]}[cell["kind"]]
    return {"f": cell["scale"] * worst + bump, "cases_seen": float(len(cases))}
'''


@pytest.mark.asyncio
async def test_the_scan_runs_its_parts_through_the_trial_runner_and_the_search_starts_from_the_held_part(tmp_path: Path) -> None:
    root = _quest(tmp_path, SIMULATE_CASES)
    block, why = op.normalize(_parts(in_each_run={"case": [1.0, 2.0]}, search_method="global_then_local",
                                     evaluation_budget={"starts": 2, "per_start": 15}))
    assert why is None, why
    run = await _run(root, {"optimisation": block})
    trials: dict[str, Any] = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    kinds = sorted((c["cell"]["kind"], c["cell"]["spread"], c["cell"]["offset"]) for c in trials["cells"])
    assert kinds == sorted([("spread", 0.0, 0.0), ("spread", 1.0, 0.0), ("spread", 2.0, 0.0),
                            ("offset", 0.0, 0.0), ("offset", 0.0, 0.5)]), "the settings of the parts, each once"
    evaluations = [r for r in _ledger(root) if r.get("event") == "evaluation"]
    assert evaluations and all(r["values"].get("cases_seen") == 2.0 for r in evaluations if r.get("values")), \
        "every design is evaluated over every case together"
    best = json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))
    assert best["best"]["objective"] < best["baseline"]["objective"]
    assert run.record["evaluations"]["scan"] == 0, "no cell of the scan is the held part at another design"
