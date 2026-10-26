"""A compared condition the simulation ignores: the alternatives after the first are not searched again.

FI searches each named alternative of a compared condition (core/optimise_search.py ``compare``). When the plan's design
variable makes the simulation ignore that condition, every alternative gives the same answer to every design, and a
search of each is the same search repeated. Before a later alternative is searched, a few designs the first one
evaluated are asked at it; when every answer is exactly the first's, it is not searched and the record says so. Neutral
toy simulations: "cups" of three shapes.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from core import optimisation_plan as op
from core import optimise_search as osearch

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
SHAPES = ("cup", "mug", "bowl")
CUP = {"shape": "cup"}


def _block(**changes: Any) -> dict[str, Any]:
    block = copy.deepcopy(BLOCK)
    block.update(changes)
    fixed, why = op.normalize(block)
    assert why is None, why
    return fixed


class Sim:
    """A simulation that counts its calls by shape. ``reads``: what it takes from the shape."""

    def __init__(self, reads: str) -> None:
        self.reads = reads
        self.calls: dict[str, int] = {s: 0 for s in SHAPES}

    def __call__(self, cell: dict[str, Any]) -> dict[str, float]:
        self.calls[cell["shape"]] += 1
        shift = {"cup": 0.0, "mug": 1.0, "bowl": 2.0}[cell["shape"]]
        x = cell["x"]
        out = {"f": 2.0 + (x + 3.0) ** 2, "mass": abs(x) * 2.0, "elapsed": 0.0}
        if self.reads == "objective":
            out["f"] += shift
        elif self.reads == "result":
            out["mass"] += shift  # a result the plan names, not the objective
        return out


def _run(block: dict[str, Any], sim: Sim, results: list[str] | None = None) -> dict[str, Any]:
    return osearch.run_sync(block, sim, seed=0, results=results)


def _other_rows(outcome: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in outcome["rows"] if r["level"] != CUP]


def test_alternatives_that_give_the_first_ones_results_are_not_searched() -> None:
    block = _block()
    sim = Sim("nothing")
    outcome = _run(block, sim, ["mass"])
    first, *later = outcome["levels"]
    assert "same_as" not in first and len(later) == 2
    for p in later:
        assert p["same_as"] == CUP and p["stopped_because"] == "same_as_first"
        assert p["probes"] == osearch.PROBE_DESIGNS and p["evaluations"] == p["probes"]
    # only the first alternative was searched: the others were called for the probe, and for the three scan points
    # (the scan measures every alternative together)
    assert sim.calls["cup"] > 20 and sim.calls["mug"] == sim.calls["bowl"] == osearch.PROBE_DESIGNS + 3
    assert outcome["best"]["level"] == CUP and outcome["best"]["evaluation"] == first["best"]["evaluation"]
    # evaluations and rows are not counted twice: the first alternative's own, plus each probe
    own = sum(1 for r in outcome["rows"] if r["level"] == CUP and r["counted"])
    assert outcome["evaluations"] == own + 2 * osearch.PROBE_DESIGNS
    assert [r["n"] for r in outcome["rows"]] == list(range(1, len(outcome["rows"]) + 1))
    assert {r["stage"] for r in _other_rows(outcome)} == {"probe"} and len(_other_rows(outcome)) == 16


def test_the_record_says_in_plain_words_which_alternatives_were_not_searched() -> None:
    block = _block()
    outcome = _run(block, Sim("nothing"), ["mass"])
    record = osearch.best_design(outcome, block, seed=0)
    n = osearch.PROBE_DESIGNS
    assert (f"shape = mug gave the same results as shape = cup for every one of the {n} designs tried at both, "
            "so the search was not run again for it") in record["says"]
    assert "shape = bowl gave the same results as shape = cup" in record["says"]
    assert "each searched on its own" not in record["says"] and "the winner is shape = cup" in record["says"]
    assert record["compared"]["levels"][1]["same_as"] == CUP
    lines = osearch.ledger_lines(outcome, block)
    assert json.loads(lines[0])["compared"] == [{"shape": s} for s in SHAPES]
    assert sum(1 for line in lines[1:] if json.loads(line).get("stage") == "probe") == 16


def test_alternatives_with_different_results_are_each_searched_as_before() -> None:
    block = _block()
    outcome = _run(block, Sim("objective"), ["mass"])
    assert all("same_as" not in p and p["stopped_because"] != "same_as_first" for p in outcome["levels"])
    assert all(p["evaluations"] > 20 for p in outcome["levels"])
    assert all(r["stage"] != "probe" for r in outcome["rows"])  # the probe is outside the budget and the rows
    assert outcome["budget"] == 180 and outcome["best"]["level"] == CUP
    assert "each searched on its own" in osearch.best_design(outcome, block, seed=0)["says"]


def test_a_condition_that_changes_only_one_named_result_is_still_searched() -> None:
    block = _block()
    named = _run(block, Sim("result"), ["mass"])
    assert all("same_as" not in p for p in named["levels"]), "it changes a result the plan names"
    # the same simulation, when the plan names no such result, gives the same objective at every alternative
    unnamed = _run(block, Sim("result"), None)
    assert [bool(p.get("same_as")) for p in unnamed["levels"]] == [False, True, True]


def test_a_simulation_whose_answers_vary_between_calls_is_searched() -> None:
    block = _block()
    ticks = iter(range(10 ** 6))

    def noisy(cell: dict[str, Any]) -> dict[str, float]:
        return {"f": 2.0 + (cell["x"] + 3.0) ** 2 + next(ticks) * 1e-9}

    outcome = osearch.run_sync(block, noisy, seed=0)
    assert all("same_as" not in p for p in outcome["levels"])


def test_the_probe_is_deterministic_and_spread_over_the_first_run() -> None:
    block = _block()
    one, two = _run(block, Sim("nothing"), ["mass"]), _run(block, Sim("nothing"), ["mass"])
    dump = lambda o: json.dumps(o["rows"], sort_keys=True, default=str)  # noqa: E731
    assert dump(one) == dump(two)
    mug = [r["design"]["x"] for r in one["rows"] if r["level"] == {"shape": "mug"}]
    cup = [r for r in one["rows"] if r["level"] == CUP and r["counted"]]
    assert len(mug) == len(set(mug)) == osearch.PROBE_DESIGNS
    assert mug[0] == one["levels"][0]["baseline"]["design"]["x"]  # the baseline first
    assert one["levels"][0]["best"]["design"]["x"] in mug  # the best design of the first alternative
    assert cup[0]["design"]["x"] in mug and cup[-1]["design"]["x"] in mug  # spread from the first to the last


def test_a_driver_that_answers_evaluate_many_reaches_the_same_outcome() -> None:
    """FI's runner answers the probe as one ``evaluate_many`` (the designs at the same time)."""
    block = _block()
    sim = Sim("nothing")
    gen = osearch.compare(block, seed=0, results=["mass"])
    request = next(gen)
    try:
        while True:
            if request["kind"] == "scan":
                reply = [{"design": {"x": c["x"]}, "values": sim(c)} for c in osearch.grid_cells(request["grid"])]
            elif request["kind"] == "evaluate_many":
                assert all(r["level"] == request["level"] != CUP for r in request["requests"])
                reply = [{"values": sim(r["cell"]), "elapsed_s": 0.5} for r in request["requests"]]
            else:
                assert request["kind"] == "evaluate" and request["level"] == CUP
                reply = {"values": sim(request["cell"])}
            request = gen.send(reply)
    except StopIteration as done:
        outcome = done.value
    assert [bool(p.get("same_as")) for p in outcome["levels"]] == [False, True, True]
    assert outcome["best"]["level"] == CUP
    assert _other_rows(outcome)[0]["elapsed_s"] == 0.5


def test_run_py_compares_the_same_named_results_as_fis_search(tmp_path) -> None:
    # code/run.py repeats FI's search: it must judge an alternative on the same results FI's search did, or it skips
    # an alternative that FI searched (one that changes only a result the plan names).
    from core import code_project

    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("def run_trial(setting, seed):\n    return {}\n", encoding="utf-8")
    protocol = {"optimisation": BLOCK, "metrics": [{"id": "mass", "kind": "mean", "estimand": "the mass of the part", "unit": "one simulated part"}]}
    study = code_project.study_of(code, protocol)
    assert study is not None and study["results"] == ["mass"]
    assert 'results=study.get("results")' in code_project.RUN_SOURCE


class _Clocked(Sim):
    """A simulation that also reports how long it took: a result of the plan that changes from call to call."""

    def __init__(self, reads: str) -> None:
        super().__init__(reads)
        self.ticks = 0

    def __call__(self, cell: dict[str, Any]) -> dict[str, float]:
        self.ticks += 1
        return {**super().__call__(cell), "clock": float(self.ticks)}


def test_a_named_result_that_changes_by_itself_between_calls_does_not_keep_alternatives_apart() -> None:
    """A live plan named the run time among its results; no two calls gave the same time, so no alternative could ever be
    "the same as the first" and every one was searched. One probe design is asked again at the first alternative: what is
    not the same in the two calls is left out of the comparison."""
    block = _block()
    sim = _Clocked("nothing")
    outcome = _run(block, sim, ["mass", "clock"])
    assert all(p.get("same_as") == CUP for p in outcome["levels"][1:]), "the alternatives give the same other results"
    assert all(r["stage"] == "probe" for r in _other_rows(outcome)) and len(_other_rows(outcome)) == 16
    assert sim.calls["cup"] > 20 and sim.calls["mug"] == sim.calls["bowl"] == osearch.PROBE_DESIGNS + 3


def test_a_result_that_changes_between_calls_never_hides_a_difference_in_another_result() -> None:
    block = _block()
    outcome = _run(block, _Clocked("result"), ["mass", "clock"])
    assert all("same_as" not in p for p in outcome["levels"]), "the shape changes the mass"


def test_the_design_asked_again_is_not_a_row_and_a_varying_objective_is_still_searched() -> None:
    block = _block()
    ticks = iter(range(10 ** 6))

    def noisy(cell: dict[str, Any]) -> dict[str, float]:
        return {"f": 2.0 + (cell["x"] + 3.0) ** 2 + next(ticks) * 1e-9, "mass": 1.0}

    outcome = osearch.run_sync(block, noisy, seed=0, results=["mass"])
    assert all("same_as" not in p for p in outcome["levels"])
    assert all(r["stage"] != "probe" for r in outcome["rows"])


def test_the_answers_a_size_estimate_asks_for_give_the_count_the_run_would_search() -> None:
    block = _block()
    levels = osearch.compared_levels(block)
    designs = osearch.sizing_designs(block)
    assert len(designs) == osearch.PROBE_DESIGNS and designs[0] == {"x": 0.0}
    names = osearch.compared_names(block, ["mass", "clock"])

    def asked(sim: Sim) -> tuple[list[list[dict[str, float]]], dict[str, float]]:
        replies = [[sim({**lv, **d}) for d in designs] for lv in levels]
        return replies, sim({**levels[0], **designs[0]})

    same, again = asked(_Clocked("nothing"))
    assert osearch.alternatives_searched(block, levels, same, again, names) == 1
    differ, again = asked(_Clocked("result"))
    assert osearch.alternatives_searched(block, levels, differ, again, names) == 3, "the known positive: all three differ"
    # an alternative that did not answer is searched
    same[1][2] = None  # type: ignore[assignment]
    _, again = asked(_Clocked("nothing"))
    assert osearch.alternatives_searched(block, levels, same, again, names) == 2
