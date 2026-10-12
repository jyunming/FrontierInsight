"""Fixes from an external review of the array / manifest / topic batch: the flat-output finding judged on the numbers the check
really nudged; the topic amendment that adds a design variable brings its baseline value. No model of anyone's."""
from __future__ import annotations

import copy
from typing import Any

from core import flat_output as flat
from core import optimisation_plan as op
from core import optimise_search as search
from core import optimum_check as check
from core import plan
from core import topic_requirements as tr
from tests.test_array_design_variables import _block

IGNORES = 'def run_cell(cell):\n    return {"peak": 0.0}\n'
READS = 'def run_cell(cell):\n    return {"peak": sum(map(sum, cell["src"])) * 0.0 + cell["w"] * 0.0}\n'


def _big() -> dict[str, Any]:
    block = _block()
    block["design_variables"][0] = {"name": "src", "shape": [8, 8], "low": 0, "high": 1}
    block["baseline"]["values"]["src"] = [[0.5] * 8 for _ in range(8)]
    fixed, why = op.normalize(block)
    assert why is None and fixed is not None
    return fixed


def _probed(block: dict[str, Any]) -> list[str]:
    space = search.Space(block["design_variables"])
    return [space.scalars[k]["name"] for k in check._probed(space)]


def _hood(names: list[str], probed: list[str] | None) -> dict[str, Any]:
    hood: dict[str, Any] = {"status": "failed", "no_effect": names, "sensitivity": {n: 0.0 for n in names}}
    if probed is not None:
        hood["probed"] = probed
    return {"finished": True, "checks": {"neighbourhood": hood}}


def test_a_flat_objective_ignoring_a_large_array_is_flagged_on_the_elements_that_were_nudged() -> None:
    block = _big()
    probed = _probed(block)
    assert len(probed) == 33 and "src[7,7]" not in probed
    why = flat.flat_search_check(_hood(probed, probed), block, source=IGNORES)
    assert why is not None and "src[0,0]" in why and "27 more" in why


def test_a_simulation_that_reads_the_array_is_not_flagged_and_an_unnudged_variable_says_nothing() -> None:
    block = _big()
    probed = _probed(block)
    assert flat.flat_search_check(_hood(probed, probed), block, source=READS) is None  # reads `src` as a whole
    assert flat.flat_search_check(_hood(["w"], ["w"]), block, source=IGNORES) is None  # no element of `src` was nudged
    assert flat.flat_search_check(_hood(probed[:-1], probed), block, source=IGNORES) is None  # one nudge had an effect


def test_the_neighbourhood_record_names_what_was_nudged() -> None:
    import inspect

    assert '"probed": list(probes)' in inspect.getsource(check)


def _search_design() -> dict[str, Any]:
    block = {"objective": {"quantity": "q", "direction": "minimise"},
             "design_variables": [{"name": "a", "low": 0, "high": 1}],
             "baseline": {"values": {"a": 0.5}, "source": "x"}, "evaluation_budget": {"starts": 1, "per_start": 5}}
    return {"hypothesis": "h", "study_type": "find_best_design", "protocol": {"optimisation": block}}


def test_a_variable_added_for_the_topic_brings_its_baseline_value() -> None:
    old = _search_design()
    new = copy.deepcopy(old)
    new["protocol"]["optimisation"]["design_variables"].append({"name": "b", "low": 0, "high": 2})
    new["protocol"]["optimisation"]["baseline"]["values"]["b"] = 1.0
    merged = tr.merge(old, new, [])
    assert merged is not None
    assert merged["protocol"]["optimisation"]["baseline"]["values"] == {"a": 0.5, "b": 1.0}
    fixed, why = plan.normalize_protocol(merged["protocol"])
    assert why is None and fixed is not None
    # the design to beat keeps what it had: an answer that changes `a` does not change it
    new["protocol"]["optimisation"]["baseline"]["values"]["a"] = 0.9
    assert tr.merge(old, new, [])["protocol"]["optimisation"]["baseline"]["values"]["a"] == 0.5
