"""The starting points of a search are answered as their designs finish, and a start that converges first gives what it
did not use to the others (core/optimise_search.py, ``_Search._stream`` and ``_give_back``; core/optimise.py,
``EvaluationPool``).

A live search planned 6 starting points. Each step of the search waited for the slowest design of the step before any
start went on, and a start that converged early left its share of the budget unspent. Now every start goes on as soon as
its own designs are answered (the other starts' designs keep the machine busy), the rows are put back in the order of the
step-by-step search, and the unused share goes to the starts still running. Neutral toy objective only.
"""
from __future__ import annotations

import asyncio
import random
import time
import zlib
from typing import Any

import pytest

from core import optimise
from core import optimise_search as search

DIMS = 4


def _block(starts: int = 4, per_start: int = 60, method: str = "global_then_local", dims: int = DIMS) -> dict[str, Any]:
    names = [f"x{i}" for i in range(dims)]
    return {
        "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
        "design_variables": [{"name": n, "low": -2.0, "high": 2.0, "kind": "continuous"} for n in names],
        "baseline": {"values": {n: 0.0 for n in names}, "source": "the design in use"},
        "evaluation_budget": {"starts": starts, "per_start": per_start},
        "search_method": method,
        "grid": {"x0": [-1.5, 0.0, 1.5], "x1": [-1.5, 0.0, 1.5]},
    }


def _bowl(design: dict[str, Any]) -> float:
    xs = [float(design[f"x{i}"]) for i in range(len(design))]
    return sum((x - 0.3 * (i + 1)) ** 2 for i, x in enumerate(xs))


def _scan_rows(request: dict[str, Any], block: dict[str, Any], f: Any) -> list[dict[str, Any]]:
    base = block["baseline"]["values"]
    names = {v["name"] for v in block["design_variables"]}
    rows = []
    for cell in search.grid_cells(request["grid"]):
        design = {**base, **{k: v for k, v in cell.items() if k in names}}
        rows.append({"design": design, "values": {"f": f(design)}})
    return rows


def _drive(block: dict[str, Any], f: Any = _bowl, *, stream: bool, rng: random.Random | None = None,
           seed: int = 1, results: list[str] | None = None) -> tuple[dict[str, Any], list[int]]:
    """Run the search. With ``stream`` the designs it asks for finish one at a time in a random order; the second value
    is how many designs were running at each wait."""
    gen = search.search(block, seed=seed, batch=True, stream=stream, results=results)
    vals = lambda d: {"f": f(d), **({"g": 2.0} if results else {})}  # noqa: E731
    running: dict[Any, dict[str, Any]] = {}
    widths: list[int] = []
    try:
        request = next(gen)
        while True:
            if request["kind"] == "scan":
                request = gen.send([dict(r, values=vals(r["design"])) for r in _scan_rows(request, block, f)])
            elif request["kind"] == "evaluate_many":
                request = gen.send([{"values": vals(r["design"])} for r in request["requests"]])
            elif request["kind"] == "evaluate_stream":
                for r in request["requests"]:
                    assert r["id"] not in running, "a design is not asked for twice while it runs"
                    running[r["id"]] = r
                widths.append(len(running))
                assert running, "the search waits for a design only when one is running"
                picks = [(rng or random.Random(0)).choice(sorted(running))]
                request = gen.send([(k, {"values": vals(running.pop(k)["design"])}) for k in picks])
            else:
                request = gen.send({"values": vals(request["design"])})
    except StopIteration as done:
        return done.value, widths


def _rows(outcome: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [(r["n"], r["stage"], r["start"], tuple(sorted(r["design"].items())), r["objective"]) for r in outcome["rows"]]


@pytest.mark.parametrize("method", ["global_then_local", "bounded_local"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_starts_answered_as_designs_finish_give_the_record_of_the_step_by_step_search(method: str, order: int) -> None:
    """The same designs per start, the same rows in the same order, the same best design, whichever design finishes
    first."""
    block = _block(method=method)
    step_by_step, _ = _drive(block, stream=False)
    streamed, widths = _drive(block, stream=True, rng=random.Random(order))
    assert _rows(streamed) == _rows(step_by_step)
    assert streamed["best"] == step_by_step["best"] and streamed["start_log"] == step_by_step["start_log"]
    assert streamed["evaluations"] == step_by_step["evaluations"]
    assert max(widths) > 1, "the starts' designs run together, one start's wait does not hold back the others"


def test_the_stream_asks_for_the_designs_of_several_starts_before_any_is_answered() -> None:
    block = _block(starts=5, method="bounded_local")
    _out, widths = _drive(block, stream=True)
    assert max(widths) >= 5


def test_a_search_not_answered_as_designs_finish_never_asks_for_a_stream() -> None:
    gen = search.search(_block(), seed=1, batch=True)  # ``stream`` is off by default: a driver without a pool is served
    kinds = set()
    try:
        request = next(gen)
        while True:
            kinds.add(request["kind"])
            if request["kind"] == "scan":
                request = gen.send(_scan_rows(request, _block(), _bowl))
            elif request["kind"] == "evaluate_many":
                request = gen.send([{"values": {"f": _bowl(r["design"])}} for r in request["requests"]])
            else:
                request = gen.send({"values": {"f": _bowl(request["design"])}})
    except StopIteration:
        pass
    assert "evaluate_stream" not in kinds and "evaluate_many" in kinds


def test_a_start_that_converges_first_gives_the_budget_it_did_not_use_to_the_others() -> None:
    """One starting point sits on a plateau, so its search converges after a few evaluations; the others keep improving on
    a bowl and ran out of their own share long before. The plan's budget is spent across the starts that were still
    going."""
    def f(design: dict[str, Any]) -> float:
        return 100.0 if design["x0"] < -1.0 else _bowl(design)

    block = _block(starts=4, per_start=70, method="bounded_local")
    block["baseline"]["values"]["x0"] = -1.9  # the baseline is the first starting point: on the plateau
    outcome, _ = _drive(block, f, stream=False)
    by_start = {s["start"]: s for s in outcome["start_log"]}
    assert by_start[0]["stopped_because"] == "converged" and by_start[0]["evaluations"] < 70
    # the others used more than their own share of 70, and together the whole budget of 4 x 70 was spent
    assert max(s["evaluations"] for s in outcome["start_log"]) > 70
    assert outcome["evaluations"] == 4 * 70


def test_the_shares_of_the_budget_are_even_and_leave_nothing_over() -> None:
    """The remainder of an uneven share goes to the first starts, and the baseline's evaluation is part of the first
    start's share: 4 starts and a budget of 4 x 50 are 50 evaluations each, the baseline among the first's."""
    block = _block(starts=4, per_start=50, method="bounded_local")
    outcome, _ = _drive(block, lambda d: -_bowl(d) * 0.0 + sum(abs(float(d[f"x{i}"]) - 0.1 * (i + 1)) for i in range(DIMS)),
                        stream=False)
    assert outcome["evaluations"] == 4 * 50
    assert sorted(s["evaluations"] for s in outcome["start_log"]) == [50, 50, 50, 50]


# --- the pool the real search is answered from -----------------------------------------------------------------------


class _Nothing:
    async def settle(self) -> None:
        return None


class _Pool(optimise.EvaluationPool):
    """The pool with the simulation replaced by a delay: each design takes the seconds its first number says."""

    def __init__(self, *, at_once: int, log: list[tuple[str, Any, float]]):
        self.at_once, self.events, self.t0 = at_once, log, time.monotonic()
        self.gate = asyncio.Semaphore(at_once)
        self.tasks = {}
        self.serial = 0
        self.on_done = None
        self.outside = _Nothing()

    async def _one(self, ident: Any, name: str, cell: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        async with self.gate:
            self.events.append(("start", ident, time.monotonic() - self.t0))
            await asyncio.sleep(cell["seconds"])
            self.events.append(("end", ident, time.monotonic() - self.t0))
            return {"values": {"f": 1.0}}, "", ""


@pytest.mark.asyncio
async def test_a_place_in_the_pool_is_refilled_the_moment_a_design_finishes() -> None:
    """Two at once: a slow design (0.6 s) and a quick one (0.05 s) run first; a third design submitted later starts when
    the quick one ends, not when the slow one does."""
    events: list[tuple[str, Any, float]] = []
    pool = _Pool(at_once=2, log=events)
    pool.submit("slow", {"seconds": 0.6})
    pool.submit("quick", {"seconds": 0.05})
    got = await pool.wait()
    assert [ident for ident, _ in got] == ["quick"]
    pool.submit("next", {"seconds": 0.05})
    got += await pool.wait()
    assert [ident for ident, _ in got] == ["quick", "next"]
    started = {ident: t for kind, ident, t in events if kind == "start"}
    ended = {ident: t for kind, ident, t in events if kind == "end"}
    assert "slow" not in ended, "the third design finished while the slowest of the first two was still running"
    assert started["next"] < 0.3, "the third design did not wait for the slowest of the first two"
    left = await pool.close()
    assert [ident for ident, _ in left] == ["slow"] and not pool.tasks


# --- the global stage asked a generation at a time (off by default) -----------------------------------------------------


def _global_stage_requests(block: dict[str, Any]) -> list[int]:
    """How many designs each request of the global stage holds (a single design is 1)."""
    gen = search.search(block, seed=1, batch=True)
    sizes: list[int] = []
    try:
        request = next(gen)
        while True:
            if request["kind"] == "evaluate_many":
                if request["requests"][0]["stage"] == "global":
                    sizes.append(len(request["requests"]))
                request = gen.send([{"values": {"f": _bowl(r["design"])}} for r in request["requests"]])
            else:
                if request.get("stage") == "global":
                    sizes.append(1)
                request = gen.send({"values": {"f": _bowl(request["design"])}})
    except StopIteration:
        return sizes


def test_the_global_stage_asks_one_trial_at_a_time_unless_a_generation_is_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    block = _block(starts=2, per_start=60)
    block.pop("grid")  # no coarse scan: the global stage runs
    monkeypatch.setattr(search, "DE_GENERATIONAL", False)
    steady = _global_stage_requests(block)
    assert steady[0] > 1 and set(steady[1:]) == {1}, "the population, then one trial after another"
    monkeypatch.setattr(search, "DE_GENERATIONAL", True)
    generational = _global_stage_requests(block)
    assert generational[0] == steady[0] and generational[1] >= generational[0] - 2, "then a whole generation at a time"
    assert len(generational) < len(steady) / 4


# --- the answer to a design does not depend on the order designs finish in ------------------------------------------------


def _flat_block() -> dict[str, Any]:
    block = _block(starts=4, per_start=60, method="bounded_local")
    block["evaluation_budget"] = {"starts": 4, "per_start": 60}
    return block


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_a_stop_that_counts_designs_fires_at_the_same_design_whatever_order_they_finish_in(order: int) -> None:
    """A design that changes none of the results stops the search once enough designs are counted. The designs are counted in the
    order they were asked for, not the order they finish in, so two runs of the same plan with different timing record
    the same rows and stop at the same design."""
    block = _flat_block()
    flat = lambda design: 5.0  # noqa: E731
    reference, _ = _drive(block, flat, stream=True, rng=random.Random(100), results=["f", "g"])
    other, _ = _drive(block, flat, stream=True, rng=random.Random(order), results=["f", "g"])
    assert reference["stopped_because"] == "no_effect", reference["stopped_because"]
    assert _rows(other) == _rows(reference)
    assert other["stopped_because"] == reference["stopped_because"] and other["evaluations"] == reference["evaluations"]


def test_a_count_stop_with_a_moving_objective_is_the_same_in_any_finishing_order() -> None:
    block = _block(starts=4, per_start=40, method="bounded_local")
    outs = [_drive(block, _bowl, stream=True, rng=random.Random(k))[0] for k in range(4)]
    assert all(_rows(o) == _rows(outs[0]) and o["best"] == outs[0]["best"] for o in outs)


# --- a late answer of one alternative is not the answer of the next ----------------------------------------------------------


def test_a_design_still_running_when_one_alternative_ends_is_not_taken_for_the_same_design_of_the_next() -> None:
    block = _block(starts=3, per_start=30, method="bounded_local")
    block["fixed"] = {"shape": "cup"}
    block["grid"] = {"shape": ["cup", "mug"], "x0": [-1.5, 0.0, 1.5], "x1": [-1.5, 0.0, 1.5]}
    gen = search.compare(block, seed=1, batch=True, stream=True)
    running: list[tuple[Any, Any, float]] = []  # (id, level, value) of every design not yet answered
    stale_given = 0
    seen_second: list[float] = []
    stopped_first = False
    try:
        request = next(gen)
        while True:
            level = request.get("level", {}).get("shape")
            if request["kind"] == "scan":
                request = gen.send(_scan_rows(request, block, _bowl))
            elif request["kind"] == "evaluate_stream":
                for r in request["requests"]:
                    running.append((r["id"], level, 999.0 if level == "cup" else _bowl(r["design"])))
                stale = [e for e in running if e[1] != level]
                if stale:  # the pool still holds the first alternative's designs: they finish first
                    reply = [(e[0], {"values": {"f": e[2]}}) for e in stale]
                    stale_given += len(stale)
                    running = [e for e in running if e[1] == level]
                elif level == "cup" and not stopped_first and len(running) > 1:
                    stopped_first = True  # the first alternative ends with designs still running
                    first = running.pop(0)
                    reply = [(first[0], {"stop": "time"})]
                else:
                    first = running.pop(0)
                    reply = [(first[0], {"values": {"f": first[2]}})]
                    if level == "mug":
                        seen_second.append(first[2])
                request = gen.send(reply)
            elif request["kind"] == "evaluate_many":
                request = gen.send([{"values": {"f": _bowl(r["design"])}} for r in request["requests"]])
            else:
                request = gen.send({"values": {"f": _bowl(request["design"])}})
    except StopIteration as done:
        outcome = done.value
    assert stopped_first and stale_given > 0, "the first alternative ended with designs running"
    second = [r for r in outcome["rows"] if r.get("level", {}).get("shape") == "mug"]
    assert second and all(r["objective"] != 999.0 for r in second)


# --- stopping the search stops the designs that are running ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closing_the_pool_for_a_stop_does_not_wait_for_the_designs_that_run() -> None:
    events: list[tuple[str, Any, float]] = []
    pool = _Pool(at_once=2, log=events)
    pool.submit("a", {"seconds": 30})
    pool.submit("b", {"seconds": 30})
    await asyncio.sleep(0.05)
    tasks = list(pool.tasks)
    started = time.monotonic()
    left = await pool.close(discard=True)
    assert time.monotonic() - started < 5 and left == []
    assert all(t.cancelled() for t in tasks) and not pool.tasks


# --- designs that finished when the search stopped ---------------------------------------------------------------------------


def _drive_batch(block: dict[str, Any], f: Any, *, together: bool, results: list[str]) -> tuple[dict[str, Any], int]:
    """The stream driver with the designs it asks for finishing one at a time (``together`` False) or all that run
    finishing at once; the second value is how many designs the stream handed back as run to the end."""
    gen = search.search(block, seed=1, batch=True, stream=True, results=results)
    vals = lambda d: {"f": f(d), "g": 2.0}  # noqa: E731
    running: dict[Any, dict[str, Any]] = {}
    handed = 0
    try:
        request = next(gen)
        while True:
            if request["kind"] == "scan":
                request = gen.send([dict(r, values=vals(r["design"])) for r in _scan_rows(request, block, f)])
            elif request["kind"] == "evaluate_many":
                request = gen.send([{"values": vals(r["design"])} for r in request["requests"]])
            elif request["kind"] == "evaluate_stream":
                for r in request["requests"]:
                    running[r["id"]] = r
                picks = sorted(running) if together else [sorted(running)[0]]
                handed += len(picks)
                request = gen.send([(k, {"values": vals(running.pop(k)["design"])}) for k in picks])
            else:
                request = gen.send({"values": vals(request["design"])})
    except StopIteration as done:
        return done.value, handed


def test_designs_that_finished_when_a_stop_fired_are_listed_apart_and_change_nothing_else() -> None:
    """Several designs finish at once and the stop fires on the first of them in request order: the others ran to the end
    (they are in the journal) but are no part of the rows, the budget or the best, which are those of one design at a time.
    The record lists them apart, so rows plus those equal the designs run."""
    block = _flat_block()
    flat = lambda design: 5.0  # noqa: E731
    one, handed_one = _drive_batch(block, flat, together=False, results=["f", "g"])
    many, handed_many = _drive_batch(block, flat, together=True, results=["f", "g"])
    assert many["stopped_because"] == "no_effect"
    assert _rows(many) == _rows(one) and many["best"] == one["best"] and many["evaluations"] == one["evaluations"]
    unused = many.get("run_not_used") or {}
    assert unused.get("count", 0) >= 2, "the designs that had finished with the stopping one are listed"
    assert unused["count"] == len(unused["designs"])
    stream_rows = sum(1 for r in many["rows"] if r["stage"] == "local")
    assert stream_rows + unused["count"] == handed_many, "every design the stream ran is a row or listed"
    assert sum(1 for r in one["rows"] if r["stage"] == "local") + one["run_not_used"]["count"] == handed_one


@pytest.mark.asyncio
async def test_a_real_fault_of_a_design_that_was_running_when_the_pool_is_discarded_is_logged() -> None:
    class Says:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def warning(self, msg: str, *args: Any) -> None:
            self.lines.append(msg % args)

    class Broken(_Pool):
        async def _one(self, ident: Any, name: str, cell: dict[str, Any]) -> Any:
            if ident == "broken":
                raise RuntimeError("the callback fell over")
            return await super()._one(ident, name, cell)

    says = Says()
    pool = Broken(at_once=2, log=[])
    pool.log = says
    pool.submit("broken", {"seconds": 0})
    pool.submit("slow", {"seconds": 30})
    await asyncio.sleep(0.1)  # the first has failed and nobody has collected it
    await pool.close(discard=True)
    assert len(says.lines) == 1 and "RuntimeError" in says.lines[0] and "the callback fell over" in says.lines[0]
    assert not any("CancelledError" in line for line in says.lines), "the stop itself is not a fault"


# --- a slow design does not hold back the starts that asked after it -----------------------------------------------------


def _drive_timed(block: dict[str, Any], f: Any, seconds: Any, *, at_once: int = 8, results: list[str] | None = None,
                 ) -> tuple[dict[str, Any], list[tuple[float, int, int]], dict[Any, float]]:
    """The stream driver with a clock: ``at_once`` workers, a design takes ``seconds(request)`` and the designs that end at
    the earliest moment are handed back together. Returns the outcome, every design asked for as (time asked, start, number
    of the start's designs asked so far) and the time each design ended."""
    gen = search.search(block, seed=1, batch=True, stream=True, results=results)
    vals = lambda d: {"f": f(d), **({"g": 2.0} if results else {})}  # noqa: E731
    clock, serial = 0.0, 0
    free = [0.0] * at_once
    running: list[tuple[float, int, Any, dict[str, Any]]] = []
    asked: list[tuple[float, int, int]] = []
    ended: dict[Any, float] = {}
    per_start: dict[int, int] = {}
    try:
        request = next(gen)
        while True:
            if request["kind"] == "scan":
                request = gen.send([dict(r, values=vals(r["design"])) for r in _scan_rows(request, block, f)])
            elif request["kind"] == "evaluate_many":
                request = gen.send([{"values": vals(r["design"])} for r in request["requests"]])
            elif request["kind"] == "evaluate_stream":
                for r in request["requests"]:
                    begin = max(clock, min(free))
                    free.remove(min(free))
                    finish = begin + seconds(r)
                    free.append(finish)
                    serial += 1
                    running.append((finish, serial, r["id"], r))
                    per_start[r["start"]] = per_start.get(r["start"], 0) + 1
                    asked.append((clock, r["start"], per_start[r["start"]]))
                running.sort(key=lambda e: (e[0], e[1]))
                clock = running[0][0]
                handed = [e for e in running if e[0] <= clock]
                running = [e for e in running if e[0] > clock]
                for e in handed:
                    ended[e[2]] = e[0]
                request = gen.send([(e[2], {"values": vals(e[3]["design"])}) for e in handed])
            else:
                request = gen.send({"values": vals(request["design"])})
    except StopIteration as done:
        return done.value, asked, ended


def _same_record(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert _rows(a) == _rows(b)
    assert a["best"] == b["best"] and a["start_log"] == b["start_log"] and a["evaluations"] == b["evaluations"]
    assert a["stopped_because"] == b["stopped_because"] and a["budget_ran_out"] == b["budget_ran_out"]


@pytest.mark.parametrize("slow", [0, 2, 3])
@pytest.mark.parametrize("per_start", [40, 12])
def test_a_slow_start_does_not_hold_back_the_others_and_the_record_is_the_same(slow: int, per_start: int) -> None:
    """One start's designs take 1000 times as long. The others ask for their later steps while its design is running
    (they do not wait for it), and the rows, the budget, the stop and the best are those of one design at a time. The
    smaller share makes starts end on their share and hand the rest on."""
    block = _block(starts=4, per_start=per_start, method="bounded_local")
    seconds = lambda r: 1000.0 if r["start"] == slow else 1.0 + (zlib.crc32(repr(r["id"]).encode()) % 7) / 10  # noqa: E731
    step_by_step, _ = _drive(block, stream=False)
    timed, asked, ended = _drive_timed(block, _bowl, seconds)
    _same_record(timed, step_by_step)
    first_slow = min(t for t in ended.values() if t >= 1000.0)
    before = [n for t, start, n in asked if start != slow and t < first_slow]
    if per_start >= 40:
        assert max(before) > 3 * (DIMS + 1), "a start that is not slow went on while the slow design was running"


def test_a_count_stop_fires_at_the_same_design_when_one_start_is_slow() -> None:
    block = _flat_block()
    flat = lambda design: 5.0  # noqa: E731
    reference, _ = _drive(block, flat, stream=False, results=["f", "g"])
    for slow in (0, 3):
        timed, _asked, _ended = _drive_timed(block, flat, lambda r: 1000.0 if r["start"] == slow else 1.0,
                                             results=["f", "g"])
        assert reference["stopped_because"] == "no_effect"
        _same_record(timed, reference)


def test_a_look_ahead_copy_does_not_hold_the_rows_or_the_cache_of_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Many starts over a large coarse scan: each start's look-ahead copy, as made, carries its own small state, not the
    scan's rows or the designs already evaluated (a full copy for each start would hold the scan once per start)."""
    block = _block(starts=8, per_start=10, method="bounded_local")
    axis = [-2.0 + 4.0 * k / 59 for k in range(60)]
    block["grid"] = {"x0": axis, "x1": axis}  # 3600 scan rows
    made: list[int] = []
    original = search._Search._look_ahead

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        kept = (self.run_not_used, self.round_log, self.rounds)
        self.run_not_used, self.round_log, self.rounds = [{}] * 5000, [{}] * 5000, [{}] * 5000  # grown lists
        out = original(self, *args, **kwargs)
        self.run_not_used, self.round_log, self.rounds = kept
        copy_ = out[0]
        assert len(self.rows) > 3000 and copy_.rows is not self.rows and len(copy_.rows) == 0
        assert copy_.cache.maps[1] is self.cache, "the designs already evaluated are read through, not copied"
        big = {k: len(v) for k, v in vars(copy_).items()
               if k not in ("cache", "block", "space") and hasattr(v, "__len__") and not isinstance(v, str) and len(v) > 1000}
        assert not big, big
        assert not (copy_.notes or copy_.run_not_used or copy_.rounds or copy_.round_log)
        made.append(1)
        return out

    monkeypatch.setattr(search._Search, "_look_ahead", spy)
    outcome, _ = _drive(block, stream=True)
    assert len(made) == 8 and len(outcome["rows"]) > 3000
