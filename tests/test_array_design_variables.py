"""An array design variable: a design that is a set of numbers (the pixels of a free-form source) is ONE variable with a
``shape``. Strict in the plan, searched element by element, carried back in its shape. No model, no simulation of anyone's."""
from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from core import optimisation_plan as op
from core import optimise_search as search
from core import optimum_check as check


def _block(**changes: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "objective": {"quantity": "q", "direction": "minimise"},
        "design_variables": [
            {"name": "src", "shape": [2, 3], "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "w", "kind": "integer", "low": 1, "high": 3},
        ],
        "baseline": {"values": {"src": [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], "w": 2}, "source": "the topic"},
        "evaluation_budget": {"starts": 2, "per_start": 40},
    }
    block.update(changes)
    return block


def _why(block: dict[str, Any]) -> str:
    fixed, why = op.normalize(block)
    assert fixed is None and why, "the block should have been refused"
    return why


def _src(**changes: Any) -> dict[str, Any]:
    block = _block()
    block["design_variables"][0].update(changes)
    return block


# --- the plan: strict ---------------------------------------------------------------------------------------------------


def test_a_five_by_five_source_has_a_form() -> None:
    """The DUV case: a free-form pixelated source of 5 x 5 had no valid form and the whole block was dropped."""
    values = [[0.1 * ((i + j) % 3) for j in range(5)] for i in range(5)]
    block = _block()
    block["design_variables"][0] = {"name": "source", "shape": [5, 5], "kind": "continuous", "low": 0, "high": 1, "unit": "a.u."}
    block["baseline"]["values"] = {"source": values, "w": 2}
    fixed, why = op.normalize(block)
    assert why is None and fixed is not None
    assert fixed["design_variables"][0]["shape"] == [5, 5] and op.element_count(fixed["design_variables"][0]) == 25
    assert op.is_array(fixed["design_variables"][0]) and not op.is_array(fixed["design_variables"][1])


def test_per_element_bounds_are_read() -> None:
    low = [[0, 0, 0], [0.2, 0.2, 0.2]]
    high = [[1, 1, 1], [0.8, 0.8, 0.8]]
    fixed, why = op.normalize(_src(low=low, high=high))
    assert why is None and fixed is not None
    assert op.bound_at(fixed["design_variables"][0]["low"], (1, 2)) == 0.2
    assert "its own range for each element" in op._range(fixed["design_variables"][0])
    # a baseline element outside ITS range is refused, though inside the widest one
    bad = _src(low=low, high=high)
    bad["baseline"]["values"]["src"][1][0] = 0.1
    assert "[1, 0]" in _why(bad) and "outside its range" in _why(bad)


@pytest.mark.parametrize("shape, fragment", [
    ([], "`shape` must be a list of 1 to 3 whole numbers"),
    ([0, 3], "`shape` must be a list of 1 to 3 whole numbers"),
    ([2, -1], "`shape` must be a list of 1 to 3 whole numbers"),
    ([2.5, 2], "`shape` must be a list of 1 to 3 whole numbers"),
    ([True, 2], "`shape` must be a list of 1 to 3 whole numbers"),
    ([2, 2, 2, 2], "`shape` must be a list of 1 to 3 whole numbers"),
    ("5x5", "`shape` must be a list of 1 to 3 whole numbers"),
    ([33, 33], "has 1089 elements; at most 1024"),
])
def test_a_bad_shape_is_refused_with_its_reason(shape: Any, fragment: str) -> None:
    block = _src(shape=shape)
    assert fragment in _why(block)


def test_the_largest_array_is_accepted_and_one_more_is_not() -> None:
    block = _block()
    block["design_variables"][0] = {"name": "src", "shape": [32, 32], "low": 0, "high": 1}
    block["baseline"]["values"]["src"] = [[0.5] * 32 for _ in range(32)]
    assert op.normalize(block)[1] is None
    block["design_variables"][0]["shape"] = [1025]
    assert "1025 elements" in _why(block)


def test_a_list_without_a_declared_shape_names_the_missing_shape() -> None:
    block = _block()
    block["design_variables"][0] = {"name": "src", "kind": "continuous", "low": [0, 0], "high": [1, 1]}
    why = _why(block)
    assert "declares no `shape`" in why and "`src`" in why


@pytest.mark.parametrize("changes, fragment", [
    ({"low": [[0, 0], [0, 0]]}, "`low` must be a number, or a list of numbers of shape [2, 3]"),
    ({"low": [[0, 0, 0], [0, 0, "x"]]}, "`low` must be a number"),
    ({"high": float("nan")}, "`high` must be a number"),
    ({"high": float("inf")}, "`high` must be a number"),
    ({"low": 1.0, "high": 1.0}, "must be below"),
    ({"low": [[0, 0, 0], [0, 2, 0]]}, "`low` (2) must be below `high` (1.0) at [1, 1]"),
    ({"kind": "integer", "low": 0, "high": 1.5}, "whole-number"),
    ({"kind": "choice"}, "`continuous` or `integer`"),
    ({"values": [1, 2]}, "not a list of `values`"),
    ({"kind": "continuous 8x8 grid of intensities"}, "array of values"),
])
def test_bad_bounds_and_kinds_are_refused_like_a_bad_scalar(changes: dict[str, Any], fragment: str) -> None:
    assert fragment in _why(_src(**changes))


@pytest.mark.parametrize("value, fragment", [
    ([[0.5, 0.5, 0.5]], "must be a list of numbers of shape [2, 3]"),            # wrong shape
    ([[0.5, 0.5], [0.5, 0.5]], "must be a list of numbers of shape [2, 3]"),
    (0.5, "must be a list of numbers of shape [2, 3]"),                           # a scalar for an array
    ([[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]], "outside its range"),
    ([[0.5, 0.5, 0.5], [0.5, 0.5, float("nan")]], "must be a list of numbers"),
    ([[0.5, 0.5, 0.5], [0.5, 0.5, True]], "must be a list of numbers"),
    ([[0.5, 0.5, 0.5], [0.5, 0.5, "0.5"]], "must be a list of numbers"),
])
def test_a_baseline_that_is_not_the_arrays_shape_or_range_is_refused(value: Any, fragment: str) -> None:
    block = _block()
    block["baseline"]["values"]["src"] = value
    why = _why(block)
    assert "`protocol.optimisation.baseline.values` `src`" in why and fragment in why


def test_an_array_is_not_a_scan_axis_and_a_missing_baseline_is_refused() -> None:
    block = _block(grid={"src": [0.1, 0.2]}, fixed={"n": 1})
    assert "array variable" in _why(block)
    block = _block()
    del block["baseline"]["values"]["src"]
    assert "no value for `src`" in _why(block)


def test_an_integer_array_needs_whole_numbers_everywhere() -> None:
    block = _src(kind="integer", low=0, high=1)
    block["baseline"]["values"]["src"] = [[0, 1, 0], [1, 0, 1]]
    assert op.normalize(block)[1] is None
    block["baseline"]["values"]["src"][0][0] = 0.5
    assert "not a whole number" in _why(block)


def test_a_bad_array_is_dropped_alone_by_the_repair_and_the_rest_of_the_block_stays() -> None:
    block = _src(shape=[0])
    fixed, notes = op.repair(block)
    assert fixed is not None and [v["name"] for v in fixed["design_variables"]] == ["w"]
    assert any("src" in n for n in notes)


def test_the_note_about_arrays_is_in_the_request_to_the_plans_model() -> None:
    assert "shape" in op.SCALAR_NOTE and "1024" in op.SCALAR_NOTE
    assert "shape" in op.ask_request("t", ["x"], [])


def test_plan_md_says_the_shape_and_the_size() -> None:
    fixed, _ = op.normalize(_block())
    design = {"study_type": "find_best_design", "protocol": {"optimisation": fixed}}
    text = "\n".join(op.plan_lines(design))
    assert "an array of shape [2, 3] (6 numbers)" in text
    assert "src = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]" in text
    short = copy.deepcopy(fixed)
    short["evaluation_budget"]["per_start"] = 3  # fewer than the 7 a local search needs to start
    assert "needs 8 evaluations just to start" in "\n".join(op.plan_lines({**design, "protocol": {"optimisation": short}}))
    assert "just to start" not in text


# --- the search -------------------------------------------------------------------------------------------------------


def test_the_space_moves_one_coordinate_per_element_and_gives_the_array_back_in_its_shape() -> None:
    fixed, _ = op.normalize(_block())
    space = search.Space(fixed["design_variables"])
    assert space.dims == 7 and space.names == ["src", "w"]
    design = space.design([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5])
    assert design["src"] == [[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]] and design["w"] == 2
    assert space.canonical(design) == design and space.inside(design)
    assert space.unit(design) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5])
    assert space.fold(space.flat(design)) == design
    assert not space.inside({**design, "src": [[0.0, 0.2], [0.6, 0.8]]})  # not its shape
    assert not space.inside({**design, "src": [[0.0, 0.2, 7.0], [0.6, 0.8, 1.0]]})  # outside its range
    assert json.loads(space.key(design))[0] == design["src"]
    with pytest.raises(ValueError):
        space.canonical({**design, "src": [[0.0, 0.2], [0.6, 0.8]]})


def test_scalar_names_of_elements_and_the_plan_agree() -> None:
    fixed, _ = op.normalize(_block())
    variable = fixed["design_variables"][0]
    assert [v["name"] for v in search.scalar_variables([variable])] == [
        op.element_name("src", i) for i in op.indices(op.shape_of(variable))]
    assert op.element_name("src", (1, 2)) == "src[1,2]"


def test_integer_arrays_count_and_enumerate_all_their_designs() -> None:
    block = _src(shape=[2, 2], kind="integer", low=0, high=1)
    block["baseline"]["values"]["src"] = [[0, 0], [0, 0]]
    fixed, _ = op.normalize(block)
    space = search.Space(fixed["design_variables"])
    assert space.count() == 2 ** 4 * 3
    combos = space.combos()
    assert combos is not None and len(combos) == 48 and {json.dumps(c["src"]) for c in combos}.__len__() == 16
    assert op.effective_method(fixed) == ("exhaustive", "rule") or op.evaluations(fixed)[0] < 48
    assert search.effective_method({**fixed, "evaluation_budget": {"starts": 8, "per_start": 8}}) == ("exhaustive", "rule")


def test_exhaustive_does_not_build_a_space_it_cannot_afford() -> None:
    """33 million combinations must not be built to find out they do not fit the budget."""
    block = _src(shape=[5, 5], kind="integer", low=0, high=1, )
    block["baseline"]["values"]["src"] = [[0] * 5 for _ in range(5)]
    block["design_variables"] = block["design_variables"][:1]
    block["baseline"]["values"] = {"src": [[0] * 5 for _ in range(5)]}
    block["search_method"] = "exhaustive"
    block["evaluation_budget"] = {"starts": 1, "per_start": 20}
    fixed, why = op.normalize(block)
    assert why is None and fixed is not None
    out = search.run_sync(fixed, lambda cell: {"q": float(sum(sum(r) for r in cell["src"]))}, seed=1)
    assert out["method_used"] != "exhaustive" and "combinations within the budget" in out["method_why"]


def test_a_search_over_an_array_evaluates_whole_arrays_and_records_them_in_shape() -> None:
    fixed, _ = op.normalize(_block(evaluation_budget={"starts": 2, "per_start": 60}))
    seen: list[dict[str, Any]] = []

    def simulation(cell: dict[str, Any]) -> dict[str, float]:
        seen.append(copy.deepcopy(cell))
        # the objective: the sum of squares of each pixel's distance from 0.2, plus the width
        return {"q": sum((x - 0.2) ** 2 for row in cell["src"] for x in row) + cell["w"]}

    outcome = search.run_sync(fixed, simulation, seed=3)
    assert seen and all(len(c["src"]) == 2 and all(len(r) == 3 for r in c["src"]) for c in seen), "the simulation gets the array"
    assert not any("src[" in k for c in seen for k in c), "never its elements as separate settings"
    base = outcome["baseline"]["design"]
    assert base["src"] == [[0.5] * 3] * 2
    best = outcome["best"]
    assert best["objective"] < outcome["baseline"]["objective"]
    assert all(isinstance(r["design"]["src"], list) and len(r["design"]["src"]) == 2 for r in outcome["rows"])
    record = search.best_design(outcome, fixed, seed=3)
    text = json.dumps(record)
    assert record["best"]["design"]["src"] == best["design"]["src"] and "src[" not in text
    assert record["best"]["design"]["src"] != base["src"]
    assert "src = [[" in record["says"]


def test_a_search_is_the_same_search_for_the_same_seed() -> None:
    fixed, _ = op.normalize(_block())
    run = lambda: search.run_sync(fixed, lambda cell: {"q": sum(x for r in cell["src"] for x in r) + cell["w"]}, seed=5)  # noqa: E731
    a, b = run(), run()
    assert [r["design"] for r in a["rows"]] == [r["design"] for r in b["rows"]]


def test_a_search_note_says_when_the_budget_cannot_start_a_local_search() -> None:
    fixed, _ = op.normalize(_block(evaluation_budget={"starts": 1, "per_start": 4}))
    outcome = search.run_sync(fixed, lambda cell: {"q": sum(x for r in cell["src"] for x in r) + cell["w"]}, seed=5)
    assert any("needs 8 evaluations just to start" in n for n in outcome["notes"])


def test_a_library_method_is_given_one_coordinate_per_element_and_its_answer_is_folded() -> None:
    fixed, _ = op.normalize(_block())
    s = search._Search(fixed, 1, "optuna:tpe")
    spec = s._drive_spec([], None)
    assert [v["name"] for v in spec["variables"]][:2] == ["src[0,0]", "src[0,1]"] and len(spec["variables"]) == 7
    assert spec["baseline"]["src[1,2]"] == 0.5 and spec["baseline"]["w"] == 2
    design = s.space.element_design({**{v["name"]: 0.25 for v in spec["variables"]}, "w": 3})
    assert design == {"src": [[0.25] * 3] * 2, "w": 3}


# --- the check around the best design -----------------------------------------------------------------------------------


def test_the_check_nudges_every_number_of_a_small_array_and_an_even_spread_of_a_large_one() -> None:
    small, _ = op.normalize(_block())
    space = search.Space(small["design_variables"])
    assert check._probed(space) == list(range(7))
    big = _block()
    big["design_variables"][0] = {"name": "src", "shape": [8, 8], "low": 0, "high": 1}
    big["baseline"]["values"]["src"] = [[0.5] * 8 for _ in range(8)]
    fixed, _ = op.normalize(big)
    space = search.Space(fixed["design_variables"])
    picked = check._probed(space)
    assert len([k for k in picked if k < 64]) == op.PROBE_ELEMENTS == 32 and 64 in picked
    assert op.budget(fixed)["continuous"] == 32 + 1


def test_a_nudge_changes_one_element_and_names_it() -> None:
    fixed, _ = op.normalize(_block())
    space = search.Space(fixed["design_variables"])
    chosen = {"design": space.canonical(fixed["baseline"]["values"])}
    asked: list[dict[str, Any]] = []

    def evaluate(design: dict[str, Any], level: int, role: str):
        asked.append(copy.deepcopy(design))
        return {"status": "ok", "objective": 1.0}
        yield  # pragma: no cover -- makes it a generator

    out: dict[str, Any] = {}
    for _ in check._probe(space, chosen, 0, evaluate, out):
        pass
    assert "src[1,2]" in out and "w" in out and out["src[1,2]"]["value"] == 0.5
    assert set(out["src[0,0]"]["sides"]) == {"down", "up"}
    nudged = [d for d in asked if d["src"] != [[0.5] * 3] * 2]
    assert nudged and all(sum(1 for r, b in zip(d["src"], [[0.5] * 3] * 2) for x, y in zip(r, b) if x != y) == 1 for d in nudged)
    assert math.isclose(out["src[0,0]"]["sides"]["up"]["value"], 0.52)


# --- the settings of a trial, the report ------------------------------------------------------------------------------


def test_a_cell_with_an_array_is_named_without_commas_the_same_in_both_copies() -> None:
    from core import code_project
    from core import trial_runner

    cell = {"src": [[0.1, 0.2], [0.3, 1.0]], "w": 2}
    key = trial_runner.cell_key(cell)
    assert key == "src=[[0.1;0.2];[0.3;1.0]],w=2" and key.count(",") == 1
    source = code_project.RUN_SOURCE  # the copy that runs in the quest's own folder: the seeds come from this name
    start = source.index("def fmt(")
    scope: dict[str, Any] = {}
    exec(source[start:source.index("def trial_seed(")], scope)  # noqa: S102
    assert scope["cell_key"](cell) == key


def test_the_report_shows_the_range_of_an_array() -> None:
    from core import best_design_report as report

    fixed, _ = op.normalize(_block())
    rows = report._variables_table(fixed, {"src": [[0.5] * 3] * 2, "w": 2}, {"src": [[0.2] * 3] * 2, "w": 1})
    assert any("an array, 2x3; each element 0 to 1" in r and "[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]" not in r for r in rows)
    assert report._fmt([0.5, 0.25]) == "[0.5, 0.25]"


def test_the_prompts_tell_the_model_the_form_exists() -> None:
    from pathlib import Path

    from core import engine

    assert '"shape": [' in engine._STUDY_TYPE_DIRECTIVE and "1024" in engine._STUDY_TYPE_DIRECTIVE
    root = Path(__file__).resolve().parent.parent / "agents"
    assert "`shape`" in (root / "design_self_critique.md").read_text(encoding="utf-8")
    assert "`shape`" in (root / "plan_revise.md").read_text(encoding="utf-8")
