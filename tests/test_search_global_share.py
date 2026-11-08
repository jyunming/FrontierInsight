"""The global stage of ``global_then_local`` without a coarse scan spends 15% of the evaluation budget (at least
dims + 2 evaluations); the rest goes to the local stage. Neutral toy objective only."""
from __future__ import annotations

from typing import Any

import pytest

from core import optimise_search as search


def _bowl(design: dict[str, Any]) -> float:
    return sum((float(v) - 0.3 - 0.05 * i) ** 2 for i, v in enumerate(design.values()))


def _block(dims: int, starts: int, per_start: int) -> dict[str, Any]:
    names = [f"x{i}" for i in range(dims)]
    return {"objective": {"quantity": "f", "direction": "minimise", "unit": "1"},
            "design_variables": [{"name": n, "low": 0.0, "high": 1.0, "kind": "continuous"} for n in names],
            "baseline": {"values": {n: 0.9 for n in names}, "source": "s"},
            "evaluation_budget": {"starts": starts, "per_start": per_start}, "search_method": "global_then_local"}


def _global_evaluations(block: dict[str, Any]) -> tuple[int, int]:
    gen = search.search(block, seed=1)
    count = 0
    try:
        request = next(gen)
        while True:
            if request.get("stage") == "global":
                count += 1
            request = gen.send({"values": {"f": _bowl(request["design"])}})
    except StopIteration as done:
        return count, done.value["evaluations"]


@pytest.mark.parametrize("dims,starts,per_start", [(3, 4, 100), (2, 2, 20), (6, 5, 60)])
def test_the_global_stage_spends_its_share_of_the_budget(dims: int, starts: int, per_start: int) -> None:
    count, total = _global_evaluations(_block(dims, starts, per_start))
    budget = starts * per_start
    assert total <= budget
    # The baseline is one member of the population and is charged to the same share, so the global rows are one fewer.
    # ... and is never fewer than the designs the watches of a flat objective count (the baseline and NO_EFFECT_DESIGNS
    # others), so a small budget still shows an objective that does not move (tests/test_search_flat_objective.py).
    wanted = max(dims + 2, int(budget * search.GLOBAL_SHARE), search.NO_EFFECT_DESIGNS + 1)
    assert count + 1 == min(wanted, budget), (count, budget)
