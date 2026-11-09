"""FI's check of the best design at finer numerical settings asks for the runs that do not depend on each other as ONE
request (so the harness runs them at the same time), and its time estimate counts the check at what a finest-setting run
costs. A fake evaluator stands in for the simulation: no model, no real run.

Known positive: the record the check writes is the one the one-after-another version wrote (the hashes below were taken
from that version on the same fake answers)."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
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


def _drive(gen, sim, stop_after: int | None = None, stop_at: dict[int, set[int]] | None = None):
    """Answer the check's requests like the harness would; log each request as (kind, number of designs).
    After ``stop_after`` designs, every one is answered ``{"stop": "time"}``. ``stop_at``: ``{request number: positions in
    it}`` answered ``{"stop": "time"}`` while the others of that request finish (a pool that ends some designs earlier)."""
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
                number = len(log)
                log.append(("many", len(request["requests"])))
                skipped = (stop_at or {}).get(number, set())
                request = gen.send([{"stop": "time"} if k in skipped else reply(r["cell"])
                                    for k, r in enumerate(request["requests"])])
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


# --- a design not run is not run; every design that finished is kept -----------------------------------------------------


def _statuses(check: dict[str, Any]) -> list[list[Any]]:
    return [check["baseline"]["status"], *[c["status"] for c in check["candidates"]]]


@pytest.mark.parametrize("name", ["scalars", "array_even"])
def test_a_design_stopped_in_the_middle_of_the_levels_keeps_the_ones_that_finished_after_it(name: str) -> None:
    block, record, rows, sim = _prepare(name)
    _full, log, _ = _drive(oc.check(block, record, rows), sim)
    many = log[0][1]
    check, _log, answered = _drive(oc.check(block, record, rows), sim, stop_at={0: {many // 2}})
    assert check["finished"] is False and check["verdict"] == "unverified"
    assert answered == many - 1 and check["evaluations"]["check"] == many - 1  # every other design is counted and kept
    flat = [s for row in _statuses(check) for s in row]
    assert flat.count(None) == 1, "exactly the stopped design is marked not run"
    assert _statuses(check)[-1][-1] is not None, "the last design of the request finished and is kept"
    assert check["checks"]["refinement"]["status"] == "not_checked"


def test_designs_stopped_out_of_order_are_each_marked_not_run() -> None:
    block, record, rows, sim = _prepare("scalars")
    _full, log, _ = _drive(oc.check(block, record, rows), sim)
    many = log[0][1]
    check, _log, answered = _drive(oc.check(block, record, rows), sim, stop_at={0: {0, many - 2}})
    assert answered == many - 2
    assert [s for row in _statuses(check) for s in row].count(None) == 2


def test_a_nudge_stopped_in_the_middle_keeps_the_nudges_that_finished_after_it() -> None:
    block, record, rows, sim = _prepare("array_even")
    _full, log, _ = _drive(oc.check(block, record, rows), sim)
    levels_n, nudges_n = log[0][1], log[1][1]
    check, _log, answered = _drive(oc.check(block, record, rows), sim, stop_at={1: {nudges_n // 2}})
    assert check["finished"] is False and check["verdict"] == "unverified"
    assert answered == levels_n + nudges_n - 1 and check["evaluations"]["check"] == levels_n + nudges_n - 1
    assert check["checks"]["refinement"]["status"] == "passed"  # every level was measured
    assert check["checks"]["neighbourhood"]["status"] != "passed"


def test_with_no_stop_nothing_is_marked_not_run() -> None:
    """The known positive of the three above: nobody stopped."""
    for name in SCENARIOS:
        block, record, rows, sim = _prepare(name)
        check, _log, _ = _drive(oc.check(block, record, rows), sim, stop_at={})
        assert [s for row in _statuses(check) for s in row].count(None) == 0
        assert check["finished"] is True


# --- a simulation that cannot be loaded ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_simulation_that_cannot_be_loaded_gives_the_verdict_one_after_another_gave(tmp_path: Path) -> None:
    """Every design of the check fails with the load error (the harness answers each one so), so the verdict, the counts and
    the checks are those of the same answers given one at a time."""
    from core import optimise
    from tests.test_optimise_runner import LocalExecutor
    from tests.test_optimum_check import ARTEFACT_SIM, _block, _runner_quest

    block = _block(design_variables=[{"name": "x", "low": -1, "high": 5, "kind": "integer"}],
                   baseline={"values": {"x": 0}, "source": "s"}, numerical_settings={"mesh": {"search": 0.5}},
                   evaluation_budget={"starts": 1, "per_start": 20}, search_method="exhaustive")
    root = _runner_quest(tmp_path, ARTEFACT_SIM)
    run = await optimise.run_search(LocalExecutor(), sys.executable, root, "code/simulate.py", {"optimisation": block},
                                    base_seed=0, timeout_s=300)
    (root / "code" / "simulate.py").write_text("raise ImportError('cannot be loaded')\n", encoding="utf-8")
    check, _text, _key = await oc.run_check(LocalExecutor(), sys.executable, root, "code/simulate.py",
                                            {"optimisation": block}, run, base_seed=0, timeout_s=300, at_once=4)
    rows = [json.loads(x) for x in (run.files.get(optimise.LEDGER_PATH.as_posix()) or "").splitlines() if x.strip()]
    rows = [r for r in rows if isinstance(r, dict) and r.get("event") == "evaluation"]
    why = "the simulation could not be loaded (ImportError: cannot be loaded)"
    gen = oc.check(block, run.record, rows)
    request = next(gen)
    try:
        while True:
            many = request["requests"] if request["kind"] == "evaluate_many" else [request]
            replies = [{"failed": why} for _r in many]
            request = gen.send(replies if request["kind"] == "evaluate_many" else replies[0])
    except StopIteration as done:
        reference = done.value
    assert check["verdict"] == reference["verdict"]
    assert check["evaluations"]["failed"] == reference["evaluations"]["failed"] > 0
    assert {k: v["status"] for k, v in check["checks"].items()} == {k: v["status"] for k, v in reference["checks"].items()}


# --- the estimate: waves, no assumed parallelism, one whole design ----------------------------------------------------


def test_the_check_is_counted_in_whole_waves_and_one_after_another_without_a_measured_slowdown() -> None:
    from core import cost_model as cm

    assert cm.check_wall_s(100.0, [8, 72], at_once=8, slowdown=None) == 80 * 100.0  # not assumed parallel
    assert cm.check_wall_s(100.0, [8, 72], at_once=8, slowdown=2.0) == (1 + 9) * 100.0 * 2.0
    assert cm.check_wall_s(100.0, [8, 65], at_once=8, slowdown=2.0) == (1 + 9) * 100.0 * 2.0  # a part-filled wave is whole
    assert cm.check_wall_s(100.0, [3, 5], at_once=1, slowdown=2.0) == 800.0  # one at a time: alone
    assert cm.check_wall_s(100.0, [], at_once=8, slowdown=2.0) == 0.0


def test_the_batches_of_a_plan_are_the_requests_of_the_check() -> None:
    block, record, rows, sim = _prepare("array_even")
    _check, log, _ = _drive(oc.check(block, record, rows), sim)
    planned = op.check_batches(block)
    assert len(planned) == len(log) and all(p >= got for p, (_k, got) in zip(planned, log))
    assert sum(planned) * op.budget(block)["check_runs"] == op.budget(block)["check"]


def test_memory_limits_how_many_run_together() -> None:
    from core import execution

    assert execution.at_once_for_memory(16, None) == 16  # nothing measured: the cores decide
    assert execution.at_once_for_memory(16, 4000.0, free_mb=30000.0, total_mb=32000.0) == 5  # (30000-3200)/(1.25*4000)
    assert execution.at_once_for_memory(16, 40000.0, free_mb=30000.0, total_mb=32000.0) == 1  # one always runs
    assert execution.at_once_for_memory(4, 100.0, free_mb=30000.0, total_mb=32000.0) == 4


@pytest.mark.asyncio
async def test_a_study_without_a_measured_slowdown_is_not_priced_as_if_it_ran_in_parallel(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """10 runs of 1000 s with 8 at once and no slowdown measured are counted one after another (10,000 s), not as 1250 s."""
    eng, runner, block = await _engine(tmp_path, per_start=2)
    runner.at_once = 8
    cut, result = await _size(eng, runner, block, monkeypatch, per_eval=1.0, finer=1000.0)
    assert cut is not None or result is not None, "10 runs of 1000 s do not fit one hour"


@pytest.mark.asyncio
async def test_one_whole_design_is_timed_and_its_start_is_counted_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import run_estimate

    eng, runner, block = await _engine(tmp_path, per_start=2)
    block["runs_per_evaluation"] = 4
    block["check_runs"] = 4
    asked: list[int] = []

    async def timed(*a: Any, name: str = "pilot", runs: int = 1, **k: Any) -> Any:
        if name == "pilot_finer":
            asked.append(runs)
            return run_estimate.Probe(None, None, {}, 400.0), ""  # a whole design: its start and its 4 runs
        return run_estimate.Probe(None, None, {}, 1.0), ""

    monkeypatch.setattr(run_estimate, "_time_one", timed)
    eng._protocol_block = lambda state: {"optimisation": block, "thresholds": {}}  # type: ignore[method-assign]
    got = await eng._time_finer_check(block, {}, runner, sys.executable, None, deterministic=False, thresholds={}, inside=None,
                                      left_s=1e6, limit_s=3600.0)
    assert asked == [4], "a design of the check is timed with the runs it has"
    assert got[0] == pytest.approx(100.0), "per run: the start-up of a design is counted once, over its runs"
    assert got[2] is False


@pytest.mark.asyncio
async def test_a_timing_that_hit_its_allowance_stays_a_lower_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import run_estimate

    eng, runner, block = await _engine(tmp_path, per_start=2)

    async def timed(*a: Any, name: str = "pilot", **k: Any) -> Any:
        if name == "pilot_finer":
            return run_estimate.Probe(None, None, {}, 900.0, at_least=True), ""
        return run_estimate.Probe(None, None, {}, 1.0), ""

    monkeypatch.setattr(run_estimate, "_time_one", timed)
    eng._protocol_block = lambda state: {"optimisation": block, "thresholds": {}}  # type: ignore[method-assign]
    got = await eng._time_finer_check(block, {}, runner, sys.executable, None, deterministic=True, thresholds={}, inside=None,
                                      left_s=1e6, limit_s=3600.0)
    assert got[0] == 900.0 and got[2] is True
    assert "did not finish in the time FI gave its timing" in _log(eng)


@pytest.mark.asyncio
async def test_a_failed_timing_is_said_and_the_check_stays_a_lower_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, cut, result = await _estimate(tmp_path, monkeypatch, per_eval=1.0, finer=None)
    assert "counted at the cost of one search design, at the least" in _log(eng)
