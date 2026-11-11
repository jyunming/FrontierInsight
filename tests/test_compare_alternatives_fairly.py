"""A comparison of alternatives is made fairly, searches are not repeated, and the winner sentence is earned.

Four parts of one change (core/optimise_search.py ``compare``, ``build_comparison``; core/optimum_check.py): (1) every
alternative that is searched starts from the same designs and has the same setup; (2) an alternative whose objective and
limits agree with the first's at every probe design is not searched, and is never said to share the first's best; (3) the
search says "ahead at the search's settings" only when that is earned, and the check says "still ahead at finer settings"
only when it holds against every other searched alternative; (4) a design number that changes nothing in the search's
first round is reported from that round only, without changing the search. Neutral toy simulations: "cups" of three
shapes."""
from __future__ import annotations

import copy
from typing import Any

from core import optimisation_plan as op
from core import optimise_search as osearch
from core import optimum_check as check

BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"}],
    "fixed": {"shape": "cup"},
    "grid": {"shape": ["cup", "mug", "bowl"], "x": [-4.0, -2.0, 0.0, 2.0, 4.0]},
    "baseline": {"values": {"x": 0.0}, "source": "the design in use"},
    "numerical_settings": {"mesh": {"search": 0.5}},
    "evaluation_budget": {"starts": 3, "per_start": 40},
    "search_method": "bounded_local",
}
SHAPES = ("cup", "mug", "bowl")
SHIFT = {"cup": 0.0, "mug": 1.0, "bowl": 2.0}


def _block(**changes: Any) -> dict[str, Any]:
    block = copy.deepcopy(BLOCK)
    block.update(changes)
    fixed, why = op.normalize(block)
    assert why is None, why
    return fixed


def sim(reads: str = "objective"):
    def run(cell: dict[str, Any]) -> dict[str, float]:
        out = {"f": 2.0 + (cell["x"] + 3.0) ** 2, "mass": abs(cell["x"]) * 2.0}
        if reads == "objective":
            out["f"] += SHIFT[cell["shape"]]
        elif reads == "result":
            out["mass"] += SHIFT[cell["shape"]]
        return out

    return run


def _scan_only_at_the_first(simulation):
    """The layout that gave one alternative many scan cells and the others one: the scan's cells at the held alternative
    only (the case in which each search used to start from different designs)."""
    def scan(grid, block=None, pool=False):
        rows = []
        mine = (block or {}).get("fixed", {}).get("shape", "cup")
        for cell in osearch.grid_cells(grid):
            if not pool and (cell["shape"] != "cup" or mine != "cup"):
                continue
            rows.append({"design": {"x": cell["x"]}, "values": simulation(cell), "reason": ""})
        return rows

    return scan


def _starts_by_alternative(outcome: dict[str, Any]) -> dict[str, dict[int, Any]]:
    out: dict[str, dict[int, Any]] = {}
    for entry in outcome["start_log"]:
        out.setdefault(osearch.level_key(entry["level"]), {})[entry["start"]] = entry["from"]
    return out


# --- part 1: the same starting designs -------------------------------------------------------------------------------


def test_every_searched_alternative_starts_from_the_same_designs_whatever_the_scan_laid_down() -> None:
    block = _block()
    simulation = sim("objective")
    outcome = osearch.run_sync(block, simulation, seed=0, scan_step=_scan_only_at_the_first(simulation))
    by = _starts_by_alternative(outcome)
    assert len(by) == 3
    first = by[osearch.level_key({"shape": "cup"})]
    assert len(first) >= 3 and all(v == first for v in by.values()), "one list of starting designs for all"
    assert outcome["comparison"]["same_setup"] is True


def test_starts_come_from_the_scan_at_any_alternative_not_only_the_held_one() -> None:
    block = _block()
    summary = {"cells": [{"cell": {"shape": s, "x": x, "mesh": 0.5}, "ok": True, "failed": 0,
                          "metrics": {"f": {"values": [2.0 + (x + 3.0) ** 2 + SHIFT[s]]}}}
                         for s in SHAPES for x in (-4.0, -2.0, 0.0, 2.0, 4.0)]}
    own = osearch.scan_rows(summary, block)
    pool = osearch.scan_candidates(summary, block)
    assert {r["design"]["x"] for r in own} == {-4.0, -2.0, 0.0, 2.0, 4.0} and len(own) == 5
    assert len(pool) == 15, "the cells at every compared alternative are candidates"
    ranked = osearch.shared_starts(block, pool)
    assert {d["x"] for d in ranked[:2]} == {-2.0, -4.0}, "best measured objective first, each design once"
    assert len(ranked) == 5


def test_a_start_measured_at_another_alternative_is_evaluated_under_this_one_and_counted() -> None:
    block = _block()
    simulation = sim("objective")
    outcome = osearch.run_sync(block, simulation, seed=0, scan_step=_scan_only_at_the_first(simulation))
    mug = [r for r in outcome["rows"] if r["level"] == {"shape": "mug"} and r["counted"]]
    assert mug and all(r["stage"] != "scan" for r in mug), "the mug has no scan rows of its own"
    assert any(r["design"]["x"] == -2.0 and r["stage"] == "local" for r in mug),         "the start the cup's scan measured is evaluated at the mug, inside its budget"
    assert outcome["levels"][1]["evaluations"] == len(mug)


# --- part 2: no repeated identical searches --------------------------------------------------------------------------


def test_an_alternative_that_differs_only_in_a_result_the_search_does_not_read_is_not_searched() -> None:
    block = _block()
    outcome = osearch.run_sync(block, sim("result"), seed=0, results=["mass"])
    first, mug, bowl = outcome["levels"]
    assert "same_as" not in first
    assert mug["same_as"] == {"shape": "cup"} and bowl["same_as"] == {"shape": "cup"}
    assert mug["best"] is None and bowl["best"] is None, "no best design is claimed for an alternative not searched"
    record = osearch.best_design(outcome, block, seed=0)
    assert "2 alternatives gave the same objective and limits as shape = cup at the" in record["says"]
    assert "were not searched on their own: shape = mug, shape = bowl (their other results at those designs are their own)" in record["says"]
    assert record["says"].count("were not searched on their own:") == 1, "named in one sentence, not one line each"
    assert "the plan allowed" in record["says"]
    assert outcome["comparison"]["statement_1"] is False
    assert "FI names no best alternative" in record["says"]


def test_the_probe_includes_the_spread_designs_so_the_run_searches_what_the_estimate_counted() -> None:
    """A difference that only the spread designs (asked before any search, for the time estimate) can show: the alternative
    is searched, and the estimate's count agrees."""
    block = _block()
    spread = osearch.sizing_designs(block)[3]["x"]

    def run(cell):
        out = sim("nothing")(cell)
        if cell["shape"] == "mug" and abs(cell["x"] - spread) < 1e-6:
            out["f"] += 1.0
        return out

    outcome = osearch.run_sync(block, run, seed=0)
    assert "same_as" not in outcome["levels"][1], "it differs at one spread design"
    assert outcome["levels"][2]["same_as"] == {"shape": "cup"}
    levels = osearch.compared_levels(block)
    designs = osearch.sizing_designs(block)
    names = osearch.compared_names(block, [])
    replies = [[run({**lv, **d}) for d in designs] for lv in levels]
    assert osearch.alternatives_searched(block, levels, replies, run({**levels[0], **designs[0]}), names) == 2


def test_the_probe_always_includes_the_first_alternatives_best_design() -> None:
    block = _block()
    outcome = osearch.run_sync(block, sim("nothing"), seed=0)
    best_x = outcome["levels"][0]["best"]["design"]["x"]
    probed = [r["design"]["x"] for r in outcome["rows"] if r["stage"] == "probe" and r["level"] == {"shape": "mug"}]
    assert best_x in probed


# --- part 3: two earned statements ------------------------------------------------------------------------------------


def _fake(best: float | None, stop: str = "converged", **more: Any) -> dict[str, Any]:
    rows = [{"event": "evaluation", "counted": True, "status": "ok", "stage": "local"} for _ in range(5)]
    return {"rows": rows, "evaluations": 5, "scan_evaluations": 0, "budget": 120, "starts": 3, "per_start": 40,
            "stopped_because": stop, "method_used": "bounded_local", "start_designs": [{"x": 0.0}, {"x": 1.0}],
            "start_log": [{"start": 0}, {"start": 1}], "best": ({"objective": best, "design": {"x": 0.0}, "evaluation": 1}
                                                                  if best is not None else None), **more}


def _comparison(values: list[float | None], block: dict[str, Any] | None = None, **stops: Any) -> dict[str, Any]:
    block = block or _block()
    levels = [{"shape": s} for s in SHAPES[: len(values)]]
    outcomes = [_fake(v, stops.get(s, "converged")) for v, s in zip(values, SHAPES)]
    per = [{"level": lv} for lv in levels]
    return osearch.build_comparison(outcomes, per, block)


def test_a_clear_gap_beyond_the_plans_tolerance_earns_the_first_statement() -> None:
    block = _block(improvement_tolerance={"value": 0.5, "mode": "absolute"})
    c = _comparison([8.0, 10.0, 12.0], block)
    assert c["statement_1"] is True and c["leader"] == {"shape": "cup"}
    assert "ahead of every other alternative by more than the plan's tolerance" in c["says_1"]
    assert "next shape = mug" in c["says_1"]


def test_a_tie_is_not_named_a_winner() -> None:
    c = _comparison([10.0, 10.0, 12.0])
    assert c["statement_1"] is False and c["leader"] is None
    assert "both reach 10 J; no alternative is shown better" in c["says_1"]


def test_a_gap_within_the_tolerance_is_not_ahead_and_a_relative_tolerance_is_of_the_other_value() -> None:
    c = _comparison([10.0, 10.2, 12.0], _block(improvement_tolerance={"value": 0.5, "mode": "absolute"}))
    assert c["statement_1"] is False and "not more than the plan's tolerance (0.5 J)" in c["says_1"]
    relative = _block(improvement_tolerance={"value": 0.1, "mode": "relative"})
    assert _comparison([10.0, 10.9, 12.0], relative)["statement_1"] is False  # 10% of 10.9 is 1.09
    assert _comparison([10.0, 11.2, 12.0], relative)["statement_1"] is True
    assert _comparison([10.0, 10.0001, 12.0])["statement_1"] is True, "no tolerance: only an exact tie is not ahead"


def test_a_search_cut_short_or_a_missing_search_keeps_the_statement_unmade() -> None:
    assert _comparison([8.0, 10.0, 12.0], mug="time")["statement_1"] is False
    assert _comparison([8.0, 10.0, 12.0], mug="failures")["statement_1"] is False
    assert _comparison([8.0, None, 12.0])["statement_1"] is False


def test_different_starting_designs_keep_the_statement_unmade() -> None:
    block = _block()
    outcomes = [_fake(8.0), _fake(10.0), _fake(12.0)]
    outcomes[1]["start_designs"] = [{"x": 0.0}, {"x": 2.0}]
    c = osearch.build_comparison(outcomes, [{"level": {"shape": s}} for s in SHAPES], block)
    assert c["same_setup"] is False and c["statement_1"] is False
    assert "not searched from the same starting designs" in c["says_1"]


def test_the_counts_are_listed_beside_the_statement() -> None:
    c = _comparison([8.0, 10.0, 12.0])
    a = c["alternatives"][0]
    assert (a["starts"], a["evaluations"], a["answered"], a["failed"], a["stopped_because"]) == (2, 5, 5, 0, "converged")


def test_only_a_finer_check_that_holds_against_every_alternative_writes_the_second_statement() -> None:
    comparison = {"statement_1": True, "leader": {"shape": "cup"}, "unit": "J"}
    cup, mug, bowl = ({"shape": s} for s in SHAPES)
    both = {"checked": True, "against": [{"level": mug, "ahead": True}, {"level": bowl, "ahead": True}]}
    assert osearch.second_statement(comparison, both) == "It is still ahead of every other alternative at finer settings."
    plant = {"checked": True, "against": [{"level": mug, "ahead": True},
                                          {"level": bowl, "ahead": False, "better_at_finer": True, "value": 8.0}]}
    said = osearch.second_statement(comparison, plant)
    assert "still ahead" not in said and "shape = bowl reaches 8 J there, better than shape = cup" in said
    assert osearch.second_statement(comparison, None).startswith("Not checked at finer settings (")
    cut = {"checked": False, "why": "the check's time limit ran out"}
    assert "Not checked at finer settings (the check's time limit ran out)" in osearch.second_statement(comparison, cut)
    assert osearch.second_statement({"statement_1": False}, both) == ""


# --- part 3: the finer check recomputes every searched alternative -------------------------------------------------------


def _finer_sim(bowl_drop: float = 0.0, cup_error: float = 0.0):
    """The alternatives' values, and what a finer mesh does to them (the bowl drops by ``bowl_drop``; the cup's value
    moves with the mesh by ``cup_error`` times its size)."""
    def run(cell: dict[str, Any]) -> dict[str, float]:
        finer = cell["mesh"] < 0.5
        f = 2.0 + (cell["x"] + 3.0) ** 2 + {"cup": 0.0, "mug": 3.0, "bowl": 6.0}[cell["shape"]]
        if finer and cell["shape"] == "bowl":
            f -= bowl_drop
        if finer and cell["shape"] == "cup":
            f += cup_error * cell["mesh"]
        return {"f": f}

    return run


def _search_and_check(simulation, **plan) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    block = _block(**plan)
    outcome = osearch.run_sync(block, simulation, seed=0)
    record = osearch.best_design(outcome, block, seed=0)
    rows = [r for r in outcome["rows"] if r.get("event", "evaluation") == "evaluation"]
    return block, record, rows


def test_a_comparison_whose_runner_up_wins_at_finer_settings_never_says_still_ahead() -> None:
    simulation = _finer_sim(bowl_drop=10.0)  # at the search's settings cup 2, mug 5, bowl 8; finer, the bowl reaches -2
    block, record, rows = _search_and_check(simulation)
    assert record["compared"]["comparison"]["statement_1"] is True
    out = check.check_sync(block, record, rows, simulation)
    finer = out["alternatives"]
    assert finer["checked"] is True and finer["still_ahead"] is False
    bowl = next(a for a in finer["against"] if a["level"] == {"shape": "bowl"})
    assert bowl["ahead"] is False and bowl["better_at_finer"] is True
    said = osearch.second_statement({**record["compared"]["comparison"], "statement_1": True}, finer)
    assert "still ahead" not in said and "better than" in said
    mug = next(a for a in finer["against"] if a["level"] == {"shape": "mug"})
    assert mug["ahead"] is True, "the runner-up was rechecked too, not only the one that won"


def test_the_leader_that_stays_ahead_at_finer_settings_earns_the_second_statement() -> None:
    simulation = _finer_sim()
    block, record, rows = _search_and_check(simulation)
    out = check.check_sync(block, record, rows, simulation)
    assert out["alternatives"]["still_ahead"] is True and len(out["alternatives"]["against"]) == 2
    assert osearch.second_statement({"statement_1": True, "leader": {"shape": "cup"}, "unit": "J"},
                                    out["alternatives"]) == "It is still ahead of every other alternative at finer settings."


def test_a_gap_below_the_numerical_error_at_finer_settings_is_not_written_as_ahead() -> None:
    # the cup's value swings with the mesh (2, 6, 4): its numerical error is far larger than the mug's lead over it
    simulation = _finer_sim(cup_error=16.0)
    block, record, rows = _search_and_check(simulation)
    out = check.check_sync(block, record, rows, simulation)
    finer = out["alternatives"]
    mug = next(a for a in finer["against"] if a["level"] == {"shape": "mug"})
    assert mug["ahead"] is False and finer["still_ahead"] is False


def test_the_checks_records_keep_two_alternatives_with_the_same_design_numbers_apart() -> None:
    """The alternatives' best designs have the same numbers here (their values differ by a constant) but different
    answers: each is asked under its own conditions and keeps its own entry (requests, cache and records are keyed by
    alternative + design + level)."""
    simulation = _finer_sim()
    block, record, rows = _search_and_check(simulation)
    asked: list[tuple[str, float, float]] = []

    def watch(cell):
        asked.append((cell["shape"], cell["x"], cell["mesh"]))
        return simulation(cell)

    out = check.check_sync(block, record, rows, watch)
    finest = min(m for _s, _x, m in asked)
    assert {s for s, _x, m in asked if m == finest} == {"cup", "mug", "bowl"}
    finest_values = {tuple(a["level"].values())[0]: a["objective"]["check"][-1] for a in out["alternatives"]["against"]}
    assert finest_values["mug"] != finest_values["bowl"], "two alternatives never share an answer"
    xs = {a["design"]["x"] for a in out["alternatives"]["against"]}
    assert len(xs) == 1, "the two designs have the same numbers, and still two entries"


def test_no_second_statement_is_asked_for_when_the_first_does_not_hold() -> None:
    simulation = sim("nothing")
    block, record, rows = _search_and_check(simulation)
    assert record["compared"]["comparison"]["statement_1"] is False
    out = check.check_sync(block, record, rows, simulation)
    assert "alternatives" not in out, "nothing was claimed, so no alternative is recomputed"


# --- part 4: the first round names a number that changes nothing, and changes nothing else ------------------------------


def _array_block(per_start: int = 40) -> dict[str, Any]:
    block = {
        "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
        "design_variables": [{"name": "kind", "kind": "choice", "values": ["a", "b"]},
                             {"name": "w", "kind": "continuous", "shape": [3], "low": 0.0, "high": 1.0},
                             {"name": "y", "kind": "continuous", "low": -5.0, "high": 5.0}],
        "fixed": {}, "baseline": {"values": {"kind": "a", "w": [0.5, 0.5, 0.5], "y": 1.0}, "source": "x"},
        "numerical_settings": {"mesh": {"search": 0.5}},
        "evaluation_budget": {"starts": 2, "per_start": per_start}, "search_method": "bounded_local",
    }
    fixed, why = op.normalize(block)
    assert why is None, why
    return fixed


def _array_sim(cell: dict[str, Any]) -> dict[str, float]:
    return {"f": (cell["y"] - 2.0) ** 2 + (0.0 if cell["kind"] == "a" else 1.0)}  # the array w is never read


def test_the_first_round_says_which_numbers_changed_nothing_and_the_search_is_the_same(monkeypatch) -> None:
    block = _array_block()
    outcome = osearch.run_sync(block, _array_sim, seed=0)
    record = osearch.best_design(outcome, block, seed=0)
    said = " ".join(record["first_round"])
    assert "each of the 3 numbers of `w` changed neither the objective nor any limited quantity" in said
    assert "a larger or joint change was not tried as such" in said
    assert "`y`" not in said, "a number that changed the answer is not named"
    # the search itself is the one it was without the note: the same designs in the same order
    monkeypatch.setattr(osearch._Search, "_note_build", lambda self, start, simplex: None)
    plain = osearch.run_sync(block, _array_sim, seed=0)
    assert [r["design"] for r in plain["rows"]] == [r["design"] for r in outcome["rows"]]
    assert "first_round" not in osearch.best_design(plain, block, seed=0)


def test_the_start_estimate_names_the_build_only_for_a_simplex() -> None:
    simplex = osearch.start_build(_array_block(per_start=200))
    assert simplex == {"numbers": 5, "evaluations": 6, "per_start": 200}
    many = {"objective": {"quantity": "f", "direction": "minimise"},
            "design_variables": [{"name": "w", "kind": "continuous", "shape": [10], "low": 0.0, "high": 1.0}],
            "baseline": {"values": {"w": [0.5] * 10}, "source": "x"}, "evaluation_budget": {"starts": 2, "per_start": 50},
            "search_method": "bounded_local", "fixed": {}, "numerical_settings": {}}
    fixed, why = op.normalize(many)
    assert why is None, why
    assert osearch.start_build(fixed) is None, "a CMA-ES has no simplex build"


def test_the_build_note_is_for_a_simplex_only() -> None:
    block = _array_block(per_start=3)  # 5 numbers, a simplex needs 6 evaluations to start
    outcome = osearch.run_sync(block, _array_sim, seed=0)
    assert any("evaluations just to start" in n for n in outcome["notes"])


def test_a_design_measured_at_several_alternatives_is_ranked_by_its_best_measurement() -> None:
    block = _block()
    rows = [{"design": {"x": 1.0}, "values": {"f": 100.0}}, {"design": {"x": 1.0}, "values": {"f": 1.0}},
            {"design": {"x": 2.0}, "values": {"f": 2.0}}]
    assert [d["x"] for d in osearch.shared_starts(block, rows)] == [1.0, 2.0], "the worse first sighting is not kept"


def test_a_tie_of_more_than_two_names_every_tied_alternative() -> None:
    c = _comparison([10.0, 10.0, 10.0])
    assert c["statement_1"] is False
    assert "shape = cup, shape = mug and shape = bowl all reach 10 J; no alternative is shown better" in c["says_1"]
