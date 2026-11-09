"""FI's check of the best design at finer numerical settings asks for the runs that do not depend on each other as ONE
request (so the harness runs them at the same time), and its time estimate counts the check at what a finest-setting run
costs. A fake evaluator stands in for the simulation: no model, no real run.

Known positive: the record the check writes is the one the one-after-another version wrote (the hashes below were taken
from that version on the same fake answers)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from core import cost_model
from core import optimisation_plan as op
from core import optimise_search as osearch
from core import optimum_check as oc
from tests.test_cost_wiring import _engine, _log, _size

MESH = {"search": 0.4, "check": [0.2, 0.1], "finer": "smaller"}


def _scalar_block() -> dict[str, Any]:
    return {"objective": {"quantity": "f", "direction": "minimise", "unit": "K"},
            "design_variables": [{"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"},
                                 {"name": "y", "low": -5.0, "high": 5.0, "kind": "continuous"}],
            "baseline": {"values": {"x": 0.0, "y": 0.0}, "source": "the design in use"},
            "numerical_settings": {"mesh": MESH}, "evaluation_budget": {"starts": 3, "per_start": 30},
            "search_method": "bounded_local"}


def _array_block(shape: list[int]) -> dict[str, Any]:
    def filled(v: float) -> Any:
        out: Any = v
        for s in reversed(shape):
            out = [copy.deepcopy(out) for _ in range(s)]
        return out

    return {"objective": {"quantity": "f", "direction": "minimise"},
            "design_variables": [{"name": "src", "shape": shape, "kind": "continuous", "low": 0.0, "high": 1.0},
                                 {"name": "w", "kind": "integer", "low": 1, "high": 3}],
            "baseline": {"values": {"src": filled(0.5), "w": 2}, "source": "the topic"},
            "numerical_settings": {"mesh": MESH}, "evaluation_budget": {"starts": 2, "per_start": 40}}


def _flat(v: Any) -> list[float]:
    return [x for r in v for x in _flat(r)] if isinstance(v, list) else [float(v)]


def _sim_scalar(cell: dict[str, Any]) -> dict[str, float]:
    return {"f": (cell["x"] - 2) ** 2 + (cell["y"] + 1) ** 2 + 0.5 * cell["mesh"] ** 2, "g": cell["x"] + cell["y"]}


def _sim_array(cell: dict[str, Any]) -> dict[str, float]:
    return {"f": sum((e - 0.3) ** 2 for e in _flat(cell["src"])) + 0.1 * (cell["w"] - 2) ** 2 + cell["mesh"] ** 2}


SCENARIOS = {
    "scalars": (_scalar_block, _sim_scalar),
    "array_even": (lambda: _array_block([2, 3]), _sim_array),
    "array_odd": (lambda: _array_block([1, 3]), _sim_array),
    "array_one": (lambda: _array_block([1]), _sim_array),
}
# sha256 of the record the one-after-another version wrote for each scenario (same fake answers)
SERIAL_HASH: dict[str, str] = {
    "scalars": "c83cee4e50f7ea004d99ba2b5002329a83416b85fbafd26d0090a90262bcf2a1",
    "array_even": "a2ef77d775211f487e68aaec6d47ba4941ee8e5265a201e5af73dd7bac97c31b",
    "array_odd": "ad21c5c4c6d250d284a3dc76f5c61d75226adaa1fcfaae3e18c2cafdd4d92579",
    "array_one": "beac18c85f5aad89c8e5c8140bf2c871c5865fe4f9bfb41a20d0611588ed573b",
}


def _prepare(name: str):
    make, sim = SCENARIOS[name]
    block, why = op.normalize(make())
    assert why is None, why
    outcome = osearch.run_sync(block, sim, seed=0)
    record = osearch.best_design(outcome, block, seed=0)
    return block, record, outcome["rows"], sim


def _drive(gen, sim, stop_after: int | None = None):
    """Answer the check's requests like the harness would; log each request as (kind, number of designs).
    After ``stop_after`` designs, every one is answered ``{"stop": "time"}``."""
    log: list[tuple[str, int]] = []
    answered = 0

    def reply(cell: dict[str, Any]) -> dict[str, Any]:
        nonlocal answered
        if stop_after is not None and answered >= stop_after:
            return {"stop": "time"}
        answered += 1
        return {"values": sim(cell), "trials": [sim(cell)]}

    try:
        request = next(gen)
        while True:
            if request["kind"] == "evaluate_many":
                log.append(("many", len(request["requests"])))
                request = gen.send([reply(r["cell"]) for r in request["requests"]])
            else:
                log.append(("one", 1))
                request = gen.send(reply(request["cell"]))
    except StopIteration as done:
        return done.value, log, answered


def _digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


def test_the_nudges_arrive_as_one_request() -> None:
    block, record, rows, sim = _prepare("array_even")
    check, log, _ = _drive(oc.check(block, record, rows), sim)
    assert [kind for kind, _n in log] == ["many", "many"], log  # the levels, then every nudge
    assert log[1][1] >= 6  # a nudge both ways of the elements probed and of the whole number
    assert check["finished"] is True


def test_the_levels_arrive_as_one_request() -> None:
    block, record, rows, sim = _prepare("scalars")
    _check, log, _ = _drive(oc.check(block, record, rows), sim)
    designs = len(oc.candidates(block, rows, 3)) + 1  # the best designs and the baseline
    assert log[0] == ("many", designs * len(oc.levels_of(block)["levels"]))


def test_the_runs_are_still_the_planned_runs() -> None:
    for name in SCENARIOS:
        block, record, rows, sim = _prepare(name)
        check, _log, answered = _drive(oc.check(block, record, rows), sim)
        assert answered == check["evaluations"]["check"] <= op.budget(block)["check"]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_the_record_is_the_one_run_one_after_another_wrote(name: str) -> None:
    block, record, rows, sim = _prepare(name)
    check, _log, _ = _drive(oc.check(block, record, rows), sim)
    assert _digest(check) == SERIAL_HASH[name]


def test_the_order_of_the_runs_is_coarse_to_fine_then_the_nudges() -> None:
    block, record, rows, sim = _prepare("array_even")
    gen = oc.check(block, record, rows)
    asked: list[tuple[int, str]] = []
    request = next(gen)
    try:
        while True:
            parts = request["requests"]
            asked += [(r["level"], r["role"]) for r in parts]
            request = gen.send([{"values": sim(r["cell"])} for r in parts])
    except StopIteration:
        pass
    roles = [role for _lv, role in asked]
    first_probe = roles.index("probe")
    levels = [lv for lv, _r in asked[:first_probe]]
    assert roles[0] == "baseline" and levels == sorted(levels)
    assert all(role == "probe" for role in roles[first_probe:])


@pytest.mark.parametrize("stop_after", [0, 3, 5, 8, 11])
def test_a_deadline_keeps_what_was_done_and_marks_the_rest_not_run(stop_after: int) -> None:
    block, record, rows, sim = _prepare("array_even")
    _full, _l, total = _drive(oc.check(block, record, rows), sim)
    assert stop_after < total
    check, _log, _ = _drive(oc.check(block, record, rows), sim, stop_after=stop_after)
    assert check["finished"] is False and check["verdict"] == "unverified"
    assert check["evaluations"]["check"] == stop_after  # what ran before the deadline is counted and kept


def test_a_deadline_during_the_nudges_keeps_the_nudges_before_it() -> None:
    block, record, rows, sim = _prepare("scalars")
    in_levels = (len(oc.candidates(block, rows, 3)) + 1) * len(oc.levels_of(block)["levels"])
    check, _log, _ = _drive(oc.check(block, record, rows), sim, stop_after=in_levels + 3)
    assert check["finished"] is False
    assert check["evaluations"]["check"] == in_levels + 3
    assert check["checks"]["refinement"]["status"] == "passed"  # the levels were all measured


# --- the estimate --------------------------------------------------------------------------------------------------------


async def _estimate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, per_eval: float, finer: Any):
    """A search that fits the hour by itself (16 evaluations); the check's 10 runs are the question."""
    eng, runner, block = await _engine(tmp_path, per_start=2)
    cut, result = await _size(eng, runner, block, monkeypatch, per_eval=per_eval, finer=finer)
    return eng, cut, result


@pytest.mark.asyncio
async def test_a_planned_check_is_counted_at_a_finest_level_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1 s per search design: counted by that, the 10 check runs are 10 s and the study fits. One finest run takes 1000 s:
    # the check alone is ~10,000 s, far past the 1 h allowed.
    eng, cut, result = await _estimate(tmp_path, monkeypatch, per_eval=1.0, finer=1000.0)
    assert cut is not None or result is not None, "a check of 10 runs of 1000 s each fits in an hour?"
    model = cost_model.read(eng.quest_root)
    assert model.seconds("finer_check_unit") == 1000.0


@pytest.mark.asyncio
async def test_a_check_that_fits_leaves_the_study_as_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, cut, result = await _estimate(tmp_path, monkeypatch, per_eval=1.0, finer=2.0)
    assert cut is None and result is None
    assert cost_model.read(eng.quest_root).seconds("finer_check_unit") == 2.0


@pytest.mark.asyncio
async def test_a_finest_run_that_cannot_be_timed_counts_the_check_at_the_search_design(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, cut, result = await _estimate(tmp_path, monkeypatch, per_eval=1.0, finer=None)
    assert cut is None and result is None  # as before: the search design's cost, at the least
    assert cost_model.read(eng.quest_root).seconds("finer_check_unit") is None
    assert "could not time one run of its check" in _log(eng)
