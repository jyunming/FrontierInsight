"""A compared condition the simulation ignores: the alternatives after the first are not searched again.

FI searches each named alternative of a compared condition (core/optimise_search.py ``compare``). When the plan's design
variable makes the simulation ignore that condition, every alternative gives the same objective and limits at every
design, and a search of each is the same search repeated. Before a later alternative is searched, designs the first one
evaluated (its baseline, its best, ones along its run) and the spread designs the time estimate asks are asked at it; when
every objective and limit is exactly the first's, it is not searched, the record says so, and no best design is claimed
for it (a finite probe shows those designs agree, not that it has none better). Results the plan names besides the
objective and the limits are not read by the search, so they do not keep two alternatives apart. Neutral toy simulations:
"cups" of three shapes.
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
    probes = later[0]["probes"]
    assert osearch.PROBE_DESIGNS < probes <= 2 * osearch.PROBE_DESIGNS - 1, "the path designs and the spread designs"
    for p in later:
        assert p["same_as"] == CUP and p["stopped_because"] == "same_as_first"
        assert p["probes"] == probes and p["evaluations"] == p["probes"]
        assert p["best"] is None, "no best design is claimed for an alternative that was not searched"
    # only the first alternative was searched: the others were called for the probe, and for the three scan points
    # (the scan measures every alternative together)
    assert sim.calls["cup"] > 20 and sim.calls["mug"] == sim.calls["bowl"] == probes + 3
    assert outcome["best"]["level"] == CUP and outcome["best"]["evaluation"] == first["best"]["evaluation"]
    # evaluations and rows are not counted twice: the first alternative's own, plus each probe
    own = sum(1 for r in outcome["rows"] if r["level"] == CUP and r["counted"])
    assert outcome["evaluations"] == own + 2 * probes
    assert [r["n"] for r in outcome["rows"]] == list(range(1, len(outcome["rows"]) + 1))
    assert {r["stage"] for r in _other_rows(outcome)} == {"probe"} and len(_other_rows(outcome)) == 2 * probes


def test_the_record_says_in_plain_words_which_alternatives_were_not_searched() -> None:
    block = _block()
    outcome = _run(block, Sim("nothing"), ["mass"])
    record = osearch.best_design(outcome, block, seed=0)
    n = outcome["levels"][1]["probes"]
    assert (f"shape = mug was not searched on its own: at each of the {n} designs tried at both, it gave shape = cup's "
            "objective and limits (its other results at those designs are its own)") in record["says"]
    assert "shape = bowl was not searched on its own" in record["says"]
    assert "each searched on its own" not in record["says"] and "the winner is" not in record["says"]
    assert "FI names no best alternative" in record["says"]
    assert record["compared"]["levels"][1]["same_as"] == CUP
    lines = osearch.ledger_lines(outcome, block)
    assert json.loads(lines[0])["compared"] == [{"shape": s} for s in SHAPES]
    assert sum(1 for line in lines[1:] if json.loads(line).get("stage") == "probe") == 2 * n


def test_alternatives_with_different_results_are_each_searched_as_before() -> None:
    block = _block()
    outcome = _run(block, Sim("objective"), ["mass"])
    assert all("same_as" not in p and p["stopped_because"] != "same_as_first" for p in outcome["levels"])
    assert all(p["evaluations"] > 20 for p in outcome["levels"])
    assert all(r["stage"] != "probe" for r in outcome["rows"])  # the probe is outside the budget and the rows
    assert outcome["budget"] == 180 and outcome["best"]["level"] == CUP
    says = osearch.best_design(outcome, block, seed=0)["says"]
    assert "It is the best design found over 3 alternatives" in says and "the winner is" not in says
    assert outcome["comparison"]["same_setup"] is True


def test_a_condition_that_changes_only_a_result_the_search_does_not_read_is_not_searched() -> None:
    block = _block()
    named = _run(block, Sim("result"), ["mass"])
    # a result the plan names besides the objective and the limits is not read by the search: two alternatives that
    # differ only in it are searched no differently (the record says its other results are the alternative's own)
    assert [bool(p.get("same_as")) for p in named["levels"]] == [False, True, True]
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
    # elapsed_s is the wall time of a call (0.0 on a coarse clock, 0.0001 on a fine one): a measurement, not a choice of the probe
    dump = lambda o: json.dumps([{k: v for k, v in r.items() if k != "elapsed_s"} for r in o["rows"]], sort_keys=True, default=str)  # noqa: E731
    assert dump(one) == dump(two)
    mug = [r["design"]["x"] for r in one["rows"] if r["level"] == {"shape": "mug"}]
    cup = [r for r in one["rows"] if r["level"] == CUP and r["counted"]]
    assert len(mug) == len(set(mug)) > osearch.PROBE_DESIGNS  # the path designs and the spread designs, each once
    assert mug[0] == one["levels"][0]["baseline"]["design"]["x"]  # the baseline first
    assert mug[1] == one["levels"][0]["best"]["design"]["x"]  # then the best design of the first alternative
    assert cup[0]["design"]["x"] in mug and cup[-1]["design"]["x"] in mug  # spread from the first to the last
    spread = [d["x"] for d in osearch.sizing_designs(block)[1:]]
    assert all(x in mug for x in spread), "the spread designs the time estimate asks"


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
                # the first alternative's answers at the spread designs, then each later alternative's probe
                assert all(r["level"] == request["level"] for r in request["requests"])
                reply = [{"values": sim(r["cell"]), "elapsed_s": 0.5} for r in request["requests"]]
            else:
                assert request["kind"] == "evaluate" and request["level"] == CUP
                reply = {"values": sim(request["cell"])}
            request = gen.send(reply)
    except StopIteration as done:
        outcome = done.value
    assert [bool(p.get("same_as")) for p in outcome["levels"]] == [False, True, True]
    assert outcome["best"]["level"] == CUP
    assert [r["elapsed_s"] for r in _other_rows(outcome)][0] == 0.5


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
    probes = outcome["levels"][1]["probes"]
    assert all(r["stage"] == "probe" for r in _other_rows(outcome)) and len(_other_rows(outcome)) == 2 * probes
    assert sim.calls["cup"] > 20 and sim.calls["mug"] == sim.calls["bowl"] == probes + 3


def test_a_result_that_changes_between_calls_never_hides_a_difference_in_the_objective() -> None:
    block = _block()
    outcome = _run(block, _Clocked("objective"), ["mass", "clock"])
    assert all("same_as" not in p for p in outcome["levels"]), "the shape changes the objective"


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
    only_a_result, again = asked(_Clocked("result"))
    assert osearch.alternatives_searched(block, levels, only_a_result, again, names) == 1, "the search does not read mass"
    differ, again = asked(_Clocked("objective"))
    assert osearch.alternatives_searched(block, levels, differ, again, names) == 3, "the known positive: all three differ"
    # an alternative that did not answer is searched
    same[1][2] = None  # type: ignore[assignment]
    _, again = asked(_Clocked("nothing"))
    assert osearch.alternatives_searched(block, levels, same, again, names) == 2


class _Unprinted(Sim):
    """The objective is null (nothing printed) for part of the range, the same for every shape."""

    def __call__(self, cell: dict[str, Any]) -> dict[str, Any]:
        out = super().__call__(cell)
        if cell["x"] > 1.0:
            out["f"] = None
        return out


def test_a_design_neither_alternative_can_judge_does_not_keep_two_alternatives_apart() -> None:
    block = _block()
    sim = _Unprinted("nothing")
    outcome = _run(block, sim, ["mass"])
    assert [bool(p.get("same_as")) for p in outcome["levels"]] == [False, True, True], "equal null answers agree"
    levels = osearch.compared_levels(block)
    designs = osearch.sizing_designs(block)
    assert any(d["x"] > 1.0 for d in designs), "the spread reaches the part of the range with no objective"
    names = osearch.compared_names(block, ["mass"])
    replies = [[sim({**lv, **d}) for d in designs] for lv in levels]
    assert osearch.alternatives_searched(block, levels, replies, sim({**levels[0], **designs[0]}), names) == 1
    # a difference where the first gave no objective is still a difference
    empty = next(i for i, v in enumerate(replies[0]) if v["f"] is None)
    replies[1][empty] = {**replies[1][empty], "f": 5.0}
    assert osearch.alternatives_searched(block, levels, replies, sim({**levels[0], **designs[0]}), names) == 2
