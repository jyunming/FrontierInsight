"""The local stage of a search over many continuous design numbers is a CMA-ES (core/optimise_search.py,
``_Search._cma_restarts``).

A live search over 64 continuous design numbers ran Nelder-Mead simplexes from 6 starting points: the simplex asks for
one design at a time and, on test functions with known minima, needed far more evaluations than a CMA-ES for the same
design. With at least ``CMA_FROM_DIMS`` continuous numbers the local stage is one CMA-ES with what is left of the budget,
restarted from the next starting point with twice the population when it converges; fewer numbers keep the simplex.
Neutral toy objective only.
"""
from __future__ import annotations

from typing import Any

import pytest

from core import optimise_search as search

pytest.importorskip("numpy")


def _block(dims: int, per_start: int, kind: str = "continuous") -> dict[str, Any]:
    names = [f"x{i}" for i in range(dims)]
    variables = [{"name": n, "low": 0.0, "high": 1.0, "kind": kind} for n in names]
    if kind == "integer":
        variables = [{"name": n, "low": 0, "high": 10, "kind": "integer"} for n in names]
    return {"objective": {"quantity": "f", "direction": "minimise", "unit": "1"}, "design_variables": variables,
            "baseline": {"values": {n: 0.5 if kind == "continuous" else 5 for n in names}, "source": "s"},
            "evaluation_budget": {"starts": 6, "per_start": per_start}, "search_method": "global_then_local",
            "grid": {"x0": [0.2, 0.5, 0.8] if kind == "continuous" else [2, 5, 8]}}


def _f(design: dict[str, Any]) -> float:
    # A shifted, tilted bowl: the minimum (0) is away from the centre and the numbers are coupled.
    xs = [float(design[f"x{i}"]) for i in range(len(design))]
    target = [0.2 + 0.6 * ((7 * i) % 11) / 10 for i in range(len(xs))]
    d = [x - t for x, t in zip(xs, target)]
    return sum((d[i] + 0.5 * d[i - 1]) ** 2 * (1 + i) for i in range(len(d)))


def _drive(block: dict[str, Any], seed: int = 1, batch: bool = True) -> tuple[dict[str, Any], int, int]:
    gen = search.search(block, seed=seed, batch=batch)
    names = [v["name"] for v in block["design_variables"]]
    base = block["baseline"]["values"]
    waits, largest = 0, 1
    try:
        request = next(gen)
        while True:
            waits += 1
            if request["kind"] == "scan":
                rows = []
                for cell in search.grid_cells(request["grid"]):
                    design = {**base, **{k: v for k, v in cell.items() if k in names}}
                    rows.append({"design": design, "values": {"f": _f(design)}})
                request = gen.send(rows)
            elif request["kind"] == "evaluate_many":
                largest = max(largest, len(request["requests"]))
                request = gen.send([{"values": {"f": _f(r["design"])}} for r in request["requests"]])
            else:
                request = gen.send({"values": {"f": _f(request["design"])}})
    except StopIteration as done:
        return done.value, waits, largest


def test_many_continuous_numbers_are_searched_by_a_cma_es_that_finds_a_better_design(
        monkeypatch: pytest.MonkeyPatch) -> None:
    block = _block(16, 100)
    new, new_waits, population = _drive(block)
    monkeypatch.setattr(search, "CMA_FROM_DIMS", 10 ** 6)  # the simplex, as before
    old, old_waits, _ = _drive(block)
    assert new["best"]["objective"] < old["best"]["objective"] / 2, (new["best"]["objective"], old["best"]["objective"])
    assert new["evaluations"] <= 600 and population >= 4 + 8  # a whole population asked for at a time
    assert new_waits < old_waits


def test_the_cma_es_search_is_the_same_at_the_same_time_or_one_after_another() -> None:
    together, _w1, _p1 = _drive(_block(16, 50), batch=True)
    one_by_one, _w2, _p2 = _drive(_block(16, 50), batch=False)
    assert together == one_by_one


@pytest.mark.parametrize("dims, kind", [(4, "continuous"), (16, "integer")])
def test_few_numbers_or_whole_numbers_keep_the_simplex(dims: int, kind: str) -> None:
    s = search._Search(_block(dims, 40, kind), seed=1, method=None)
    assert s._cma_fits() is False


def test_a_bounded_local_search_over_many_numbers_is_a_cma_es_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real plan asked for ``bounded_local`` over 144 continuous numbers: its starting points ran simplexes one after
    another, one design at a time. Whichever method places the starting points, many continuous numbers are searched by
    the CMA-ES (on test functions with known minima at 144 numbers it found the better design in 20 of 20 cases)."""
    block = {**_block(16, 100), "search_method": "bounded_local"}
    new, new_waits, population = _drive(block)
    monkeypatch.setattr(search, "CMA_FROM_DIMS", 10 ** 6)
    old, old_waits, _ = _drive(block)
    assert new["best"]["objective"] < old["best"]["objective"] / 2, (new["best"]["objective"], old["best"]["objective"])
    assert new["evaluations"] <= 600 and population >= 4 + 8 and new_waits < old_waits

