"""A search whose every evaluation fails stops after a few, not after its first batch or its last alternative.

Neutral toy simulation. Before: the first batch of a global search (hundreds of designs) was evaluated whole before any
answer was read, and each compared alternative repeated it, although the simulation's answer never had the objective.
"""
from __future__ import annotations

import copy
import json
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


def test_each_later_alternative_is_asked_for_its_baseline_when_the_first_had_no_objective_anywhere() -> None:
    block = _block()
    assert osearch.level_count(block) == 3
    outcome, asked, _ = _drive(osearch.compare(block, seed=0, batch=True), _no_objective)
    assert asked == 12  # ten for the first alternative, then one baseline for each of the other two
    assert outcome["stopped_because"] == "failures"
    assert [p["evaluations"] for p in outcome["levels"]] == [10, 1, 1]
    assert [p.get("status") for p in outcome["levels"]] == [None, "not_searched", "not_searched"]
    assert all(p["not_searched_because"] == osearch.NOT_SEARCHED_BECAUSE for p in outcome["levels"][1:])
    assert outcome["level_count"] == 3
    own = [r for r in outcome["rows"] if r["level"] == {"shape": "mug"}]
    assert len(own) == 1 and own[0]["stage"] == "baseline" and own[0]["failure"] == osearch.NO_OBJECTIVE


def test_an_alternative_whose_first_answer_has_the_objective_is_searched_when_another_had_none() -> None:
    def only_the_first_lacks_it(request):
        return _no_objective(request) if request["cell"]["shape"] == "cup" else _healthy(request)

    outcome, asked, _ = _drive(osearch.compare(_block(), seed=0, batch=True), only_the_first_lacks_it)
    by_level = {p["level"]["shape"]: p for p in outcome["levels"]}
    assert by_level["cup"]["evaluations"] == 10
    assert by_level["mug"]["evaluations"] > 10 and by_level["mug"].get("status") is None and by_level["mug"]["best"]
    assert by_level["bowl"]["evaluations"] > 10 and by_level["bowl"].get("status") is None and by_level["bowl"]["best"]


def test_the_report_and_the_log_say_plainly_which_alternative_was_not_searched_and_why(tmp_path) -> None:
    import json
    from core import best_design_report, optimise

    def mug_only(request):
        return _healthy(request) if request["cell"]["shape"] == "mug" else _no_objective(request)

    block = _block()
    outcome, _, _ = _drive(osearch.compare(block, seed=0, batch=True), mug_only)
    by_level = {p["level"]["shape"]: p for p in outcome["levels"]}
    assert by_level["cup"]["stopped_because"] == "failures" and by_level["bowl"]["status"] == "not_searched"
    record = osearch.best_design(outcome, block, seed=0)
    path = tmp_path / optimise.BEST_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    text = best_design_report.section(tmp_path, block)
    assert "bowl" in text and "not searched, because its first evaluation also returned no objective" in text
    assert "Each of the searched ones of the 3 alternatives" in text
    assert "Each of the 3 alternatives was searched on its own" not in text
    assert "bowl was not searched: its first evaluation also returned no objective" in json.dumps(record)
    assert "bowl was not searched" in osearch.not_searched_says(by_level["bowl"])


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


def _drive_streamed(gen, answer, at_once=8):
    """Run a streamed search with a pool of ``at_once``: returns (outcome, designs answered)."""
    import collections
    queue, answered = collections.deque(), 0
    try:
        request = next(gen)
        while True:
            if request["kind"] == "evaluate_stream":
                queue.extend(request["requests"])
                out = []
                while queue and len(out) < at_once:
                    one = queue.popleft()
                    out.append((one["id"], answer(one)))
                answered += len(out)
                reply = out
            elif request["kind"] == "evaluate":
                answered += 1
                reply = answer(request)
            elif request["kind"] == "evaluate_many":
                answered += len(request["requests"])
                reply = [answer(r) for r in request["requests"]]
            else:
                reply = {"cells": [], "levels": []}
            request = gen.send(reply)
    except StopIteration as done:
        return done.value, answered


def test_streamed_starts_do_not_send_more_designs_than_it_takes_to_give_up() -> None:
    from tests.test_search_stream import _block as stream_block
    block = stream_block(starts=4, per_start=60, method="bounded_local")
    outcome, answered = _drive_streamed(osearch.search(block, seed=0, batch=True, stream=True), _no_objective)
    assert outcome["stopped_because"] == "failures"
    assert answered == 10 and not outcome.get("run_not_used")


def test_streamed_healthy_search_is_unchanged_by_the_rule() -> None:
    from tests.test_search_stream import _block as stream_block, _bowl
    block = stream_block(starts=4, per_start=40, method="bounded_local")

    def bowl(request):
        return {"values": {"f": _bowl(request["design"])}}

    streamed, _ = _drive_streamed(osearch.search(block, seed=0, batch=True, stream=True), bowl)
    plain, _, _ = _drive(osearch.search(block, seed=0, batch=False), bowl)
    assert [r["design"] for r in streamed["rows"]] == [r["design"] for r in plain["rows"]]


def _first_lacks_objective(request):
    return _no_objective(request) if request["cell"]["shape"] == "cup" else _healthy(request)


def test_a_later_alternatives_first_evaluation_is_its_searchs_baseline_and_is_not_asked_twice() -> None:
    outcome, asked, _ = _drive(osearch.compare(_block(), seed=0, batch=True), _first_lacks_objective)
    assert asked == len(outcome["rows"]), "every design asked for is a row of the record, and the other way round"
    for shape in ("mug", "bowl"):
        own = [r for r in outcome["rows"] if r["level"] == {"shape": shape}]
        assert [r["stage"] for r in own].count("baseline") == 1 and own[0]["stage"] == "baseline"


def test_the_searchs_end_is_that_of_the_alternatives_that_gave_results() -> None:
    block = _block()
    outcome, _, _ = _drive(osearch.compare(block, seed=0, batch=True), _first_lacks_objective)
    by_level = {p["level"]["shape"]: p["stopped_because"] for p in outcome["levels"]}
    assert by_level["cup"] == "failures" and by_level["mug"] != "failures"
    assert outcome["stopped_because"] == by_level["mug"] != "failures"
    # An alternative that spent its whole budget lets the search go on in an extra round (it needs "budget").
    merged = osearch.merge_outcomes(
        [{**_stub(osearch, outcome, "failures")}, {**_stub(osearch, outcome, "budget")}], [{"shape": "cup"}, {"shape": "mug"}], block)
    assert merged["stopped_because"] == "budget"


def _stub(osearch_module, outcome, stopped):
    """A one-evaluation outcome of the given end, built from a real outcome's fields."""
    base = {k: v for k, v in outcome.items() if k not in ("levels", "level_count", "rows")}
    return {**base, "rows": [], "evaluations": 1, "budget": 1, "stopped_because": stopped, "best": None, "failed": 0,
            "infeasible": 0, "baseline": {"design": {}, "status": "ok", "objective": 1.0, "constraints": {},
                                          "feasible": True}, "start_log": [], "rounds": [], "notes": [],
            "scan_evaluations": 0}


def test_the_log_does_not_say_the_first_evaluations_failed_when_a_winner_exists() -> None:
    from core import optimise
    block = _block()
    outcome, _, _ = _drive(osearch.compare(block, seed=0, batch=True), _first_lacks_objective)
    record = osearch.best_design(outcome, block, seed=0)
    assert record["evaluations"]["stopped_because"] != "failures"
    assert not any("first evaluations all failed" in line for line in optimise.summary_lines(record))


def test_without_a_winner_the_summary_and_the_log_still_name_each_unsearched_alternative_and_why() -> None:
    from core import optimise
    block = _block()
    outcome, _, _ = _drive(osearch.compare(block, seed=0, batch=True), _no_objective)
    assert outcome["best"] is None
    record = osearch.best_design(outcome, block, seed=0)
    says = json.dumps(record)
    for shape in ("mug", "bowl"):
        assert f"shape = {shape} was not searched: its first evaluation also returned no objective" in says
        assert any(f"shape = {shape} was not searched: its first evaluation also returned no objective" in line
                   for line in optimise.summary_lines(record))
