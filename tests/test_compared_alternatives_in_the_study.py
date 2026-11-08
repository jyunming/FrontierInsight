"""The alternatives a study compares are the combinations its own sub-studies have.

FI searches each named alternative of a compared condition (core/optimise_search.py ``compare``). When the plan has
``sub_studies``, a setting one sub-study varies is held by the others (it only takes other values where the plan says),
so the study has only some of the combinations of the conditions' names: the coarse scan has exactly those. The
alternatives were every combination of the names, so a setting that matters in one sub-study only multiplied every other
alternative. Neutral toy plan: cups of three shapes in two sizes, and a coating that matters only for one shape.
"""
from __future__ import annotations

import copy
from typing import Any

from core import optimisation_plan as op
from core import optimise_search as osearch

BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"}],
    "fixed": {"shape": "cup", "coating": "plain"},
    "grid": {"shape": ["cup", "mug", "bowl"], "coating": ["plain", "glazed", "waxed", "lacquered"], "x": [-4.0, 0.0, 4.0]},
    "baseline": {"values": {"x": 0.0}, "source": "the design in use"},
    "numerical_settings": {"mesh": {"search": 0.5}},
    "evaluation_budget": {"starts": 2, "per_start": 30},
    "search_method": "bounded_local",
}
SUB_STUDIES = [
    {"name": "shapes", "vary": ["shape"], "hold": {"coating": "plain", "x": 0.0}},
    {"name": "coatings_of_a_bowl", "vary": ["coating"], "hold": {"shape": "bowl", "x": 0.0}},
    {"name": "designs", "vary": ["x"], "hold": {"shape": "cup", "coating": "plain"}},
]


def _block(**changes: Any) -> dict[str, Any]:
    block = copy.deepcopy(BLOCK)
    block.update(changes)
    fixed, why = op.normalize(block)
    assert why is None, why
    return fixed


def test_without_sub_studies_every_combination_of_the_names_is_an_alternative() -> None:
    block = _block()
    levels = osearch.compared_levels(block)
    assert len(levels) == 12 == osearch.level_count(block)
    assert levels[0] == {"shape": "cup", "coating": "plain"}


def test_the_alternatives_are_the_combinations_the_sub_studies_have() -> None:
    block = _block(sub_studies=copy.deepcopy(SUB_STUDIES))
    levels = osearch.compared_levels(block)
    # the held combination first; then the other shapes with the plain coating, and the bowl with the other coatings
    assert levels == [{"shape": "cup", "coating": "plain"}, {"shape": "mug", "coating": "plain"},
                      {"shape": "bowl", "coating": "plain"}, {"shape": "bowl", "coating": "glazed"},
                      {"shape": "bowl", "coating": "waxed"}, {"shape": "bowl", "coating": "lacquered"}]
    assert osearch.level_count(block) == 6
    budget, scan = op.evaluations(block)
    assert budget == 2 * 30 * 6, "the search is counted for the alternatives the study has, not for every combination"
    # the scan measures exactly these combinations of the two conditions
    cells = {(c["shape"], c["coating"]) for c in osearch.grid_cells(osearch.scan_grid(block))}
    assert {(lv["shape"], lv["coating"]) for lv in levels} - {("cup", "plain")} <= cells


def test_a_sub_study_that_varies_both_conditions_brings_every_combination_back() -> None:
    both = SUB_STUDIES + [{"name": "all_of_them", "vary": ["shape", "coating"], "hold": {"x": 0.0}}]
    block = _block(sub_studies=both)
    assert osearch.level_count(block) == 12
