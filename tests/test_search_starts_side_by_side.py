"""The local searches of a search's starting points run side by side (core/optimise_search.py, ``_Search._side_by_side``).

A live search planned 6 starting points of 3000 evaluations each over 64 design numbers. Each local search asks for one
design at a time, and the starting points ran one after another, so nearly every evaluation of the study waited for the
one before it however many cores the machine had. The starting points do not depend on each other's answers: each step,
every start's next design is asked for in one batch, and each start goes on from its own answers. The search is the same
whether the batch runs at the same time or one after another. Neutral toy objective only.
"""
from __future__ import annotations

import math
from typing import Any

from core import optimise_search as search

DIMS = 4
BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": f"x{i}", "low": -2.0, "high": 2.0, "kind": "continuous"} for i in range(DIMS)],
    "baseline": {"values": {f"x{i}": 0.0 for i in range(DIMS)}, "source": "the design in use"},
    "evaluation_budget": {"starts": 6, "per_start": 60},
    "search_method": "global_then_local",
    "grid": {"x0": [-1.5, 0.0, 1.5], "x1": [-1.5, 0.0, 1.5]},
}


def _f(design: dict[str, Any]) -> float:
    xs = [float(design[f"x{i}"]) for i in range(DIMS)]
    return sum((x - 0.3 * (i + 1)) ** 2 for i, x in enumerate(xs)) + 0.3 * math.sin(5 * xs[0])


def _drive(batch: bool) -> tuple[list[list[int | None]], Any, int]:
    """Every request the search makes, answered at once; the starts each batch holds, the outcome, the evaluations."""
    gen = search.search(BLOCK, seed=1, batch=batch)
    batches: list[list[int | None]] = []
    evaluated = 0
    try:
        request = next(gen)
        while True:
            if request["kind"] == "scan":
                rows = []
                for cell in search.grid_cells(request["grid"]):
                    design = {**{f"x{i}": 0.0 for i in range(DIMS)}, **{k: v for k, v in cell.items() if k in BLOCK["grid"]}}
                    rows.append({"design": design, "values": {"f": _f(design)}})
                request = gen.send(rows)
            elif request["kind"] == "evaluate_many":
                batches.append([r.get("start") for r in request["requests"]])
                evaluated += len(request["requests"])
                request = gen.send([{"values": {"f": _f(r["design"])}} for r in request["requests"]])
            else:
                batches.append([request.get("start")])
                evaluated += 1
                request = gen.send({"values": {"f": _f(request["design"])}})
    except StopIteration as done:
        return batches, done.value, evaluated


def test_the_starting_points_are_searched_side_by_side() -> None:
    batches, outcome, evaluated = _drive(batch=True)
    together = [b for b in batches if len(set(b)) > 1]
    assert together, "a batch holds the designs of several starting points"
    assert max(len(set(b)) for b in batches) == 6
    # One wait per step of the slowest start, not one per evaluation.
    assert len(batches) < evaluated / 4, (len(batches), evaluated)
    assert evaluated <= 6 * 60


def test_the_search_is_the_same_run_at_the_same_time_or_one_after_another() -> None:
    _b1, one_by_one, n1 = _drive(batch=False)
    _b2, together, n2 = _drive(batch=True)
    assert n1 == n2
    assert one_by_one == together  # the same designs in the same order, the same best design and record
