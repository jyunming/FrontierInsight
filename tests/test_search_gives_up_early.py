"""A search whose every evaluation fails stops after a few, not after its first batch or its last alternative.

Neutral toy simulation. Before: the first batch of a global search (hundreds of designs) was evaluated whole before any
answer was read, and each compared alternative repeated it, although the simulation's answer never had the objective.
"""
from __future__ import annotations

import copy
from typing import Any

from core import optimisation_plan as op
from core import optimise_search as osearch

BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"},
                         {"name": "y", "low": -5.0, "high": 5.0, "kind": "continuous"}],
    "fixed": {"shape": "cup"},
    "grid": {"shape": ["cup", "mug", "bowl"], "x": [-4.0, 0.0, 4.0]},
    "baseline": {"values": {"x": 0.0, "y": 0.0}, "source": "the design in use"},
    "numerical_settings": {"mesh": {"search": 0.5}},
    "evaluation_budget": {"starts": 4, "per_start": 60},
    "search_method": "global_then_local",
}


def _block() -> dict[str, Any]:
    block, why = op.normalize(copy.deepcopy(BLOCK))
    assert block is not None, why
    return block


def _drive(gen, answer):
    """Run a search generator; returns (outcome, evaluations asked, sizes of the batches)."""
    asked, batches = 0, []
    reply = None
    try:
        request = next(gen)
        while True:
            if request["kind"] == "evaluate":
                asked += 1
                reply = answer(request)
            elif request["kind"] == "evaluate_many":
                asked += len(request["requests"])
                batches.append(len(request["requests"]))
                reply = [answer(r) for r in request["requests"]]
            else:
                reply = {"cells": [], "levels": []}
            request = gen.send(reply)
    except StopIteration as done:
        return done.value, asked, batches


def _no_objective(request):
    return {"values": {"other": 1.0}}


def _healthy(request):
    d = request["design"]
    shift = {"cup": 0.0, "mug": 1.0, "bowl": 2.0}[request["cell"]["shape"]]
    return {"values": {"f": (d["x"] - shift) ** 2 + d["y"] ** 2}}


def test_a_reply_without_the_objective_stops_after_ten_and_skips_the_other_alternatives() -> None:
    block = _block()
    assert osearch.level_count(block) == 3
    outcome, asked, _ = _drive(osearch.compare(block, seed=0, batch=True), _no_objective)
    assert asked == 10
    assert outcome["stopped_because"] == "failures"
    assert [p["evaluations"] for p in outcome["levels"]] == [10, 0, 0]
    assert outcome["level_count"] == 3


def test_one_search_alone_does_not_pay_for_its_whole_first_batch() -> None:
    _, asked, batches = _drive(osearch.search(_block(), seed=0, batch=True), _no_objective)
    assert asked == 10 and all(b < 10 for b in batches)


def test_a_failure_that_may_depend_on_the_alternative_does_not_skip_the_others() -> None:
    def broken_for_the_first(request):
        if request["cell"]["shape"] == "cup":
            return {"failed": "crashed"}
        return _healthy(request)

    outcome, _, _ = _drive(osearch.compare(_block(), seed=0, batch=True), broken_for_the_first)
    by_level = {p["level"]["shape"]: p for p in outcome["levels"]}
    assert by_level["cup"]["stopped_because"] == "failures"
    assert by_level["mug"]["evaluations"] > 10 and by_level["mug"]["best"] is not None


def test_a_healthy_search_asks_for_the_same_designs_in_the_same_order_and_batches() -> None:
    first, asked_a, batches_a = _drive(osearch.compare(_block(), seed=0, batch=True), _healthy)
    assert first["best"] is not None and batches_a and max(batches_a) > 10
    plain, asked_b, _ = _drive(osearch.compare(_block(), seed=0, batch=False), _healthy)
    assert asked_a == asked_b
    assert [r["design"] for r in first["rows"]] == [r["design"] for r in plain["rows"]]


def test_a_late_failure_after_a_working_answer_is_not_a_reason_to_stop() -> None:
    seen = {"n": 0}

    def first_works_then_fails(request):
        seen["n"] += 1
        return _healthy(request) if seen["n"] == 1 else {"values": {"other": 1.0}}

    outcome, asked, _ = _drive(osearch.search(_block(), seed=0, batch=True), first_works_then_fails)
    assert asked > 10 and outcome["stopped_because"] != "failures"
