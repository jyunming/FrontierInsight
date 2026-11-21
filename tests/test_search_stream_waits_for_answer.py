"""A streamed search never asks the pool for nothing while no design is running (core/optimise_search.py,
``_Search._stream``).

A live search had 8 starts side by side and no answer that worked yet. While that is so, the search holds designs back so
a broken simulation costs a few designs, not the whole batch, and it counts the designs sent and answered but not yet
judged as "out". Nine designs had gone out; two of them, answered, were the first two of a four-design step whose other
two were held back, and they could not be judged until the held two went out. The held two waited on the count that the
answered two made. With nothing running, the search sent the pool an empty request, the pool answered at once with
nothing, and the search asked again, for 48 minutes at 100% of one core. Now a search with nothing running sends the next
held design. Neutral toy objective only.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from core import optimise
from core import optimise_search as search

from tests.test_optimise_runner import _protocol, _quest, _run
from tests.test_search_stream import _block, _bowl, _scan_rows

NO_WORK = {"failed": "the toy simulation gave no result"}
CAP = search._FAILURES_BEFORE_GIVING_UP


def _drive_failing(block: dict[str, Any], *, order: int, works_from: int | None = None,
                   empty_limit: int = 200) -> tuple[dict[str, Any], int, int]:
    """Every design fails (with ``works_from``: the design submitted at that number and after it works). The driver
    answers a stream request as the real pool does: it answers at once with what finished, and with nothing at all when
    nothing is running. More than ``empty_limit`` such empty answers in a row is a spin. Returns the outcome, the most
    empty answers in a row and the designs actually submitted to the pool."""
    gen = search.search(block, seed=1, batch=True, stream=True)
    rng = random.Random(order)
    running: dict[Any, dict[str, Any]] = {}
    empty = most = submitted = asked = 0

    def reply(design: dict[str, Any]) -> dict[str, Any]:
        nonlocal asked
        asked += 1
        return {"values": {"f": _bowl(design)}} if works_from is not None and asked >= works_from else dict(NO_WORK)

    try:
        request = next(gen)
        while True:
            if request["kind"] == "scan":
                request = gen.send([dict(r, values={"f": float("nan")}) for r in _scan_rows(request, block, lambda d: 0)])
            elif request["kind"] == "evaluate_many":
                request = gen.send([reply(r["design"]) for r in request["requests"]])
            elif request["kind"] == "evaluate_stream":
                for r in request["requests"]:
                    running[r["id"]] = r
                    submitted += 1
                if not running:
                    empty += 1
                    most = max(most, empty)
                    assert empty <= empty_limit, "the search asks the pool for nothing while nothing runs: a spin"
                    request = gen.send([])
                    continue
                empty = 0
                pick = rng.choice(sorted(running))
                request = gen.send([(pick, reply(running.pop(pick)["design"]))])
            else:
                request = gen.send(reply(request["design"]))
    except StopIteration as done:
        return done.value, most, submitted


@pytest.mark.parametrize("cap", [CAP, 3])
def test_a_stream_with_no_answer_that_worked_yet_never_asks_for_nothing_and_keeps_the_failure_cap(
        cap: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # A cap of 3 puts the first local step (4 designs with 3 variables, 7 with 6) over the cap, as a step of 51 designs is
    # over the real one: a broken simulation costs about the cap in designs, never the whole step.
    monkeypatch.setattr(search, "_FAILURES_BEFORE_GIVING_UP", cap)
    for starts, dims in ((2, 3), (3, 3), (8, 3), (2, 6)):
        for order in range(3):
            block = _block(starts=starts, per_start=60, method="bounded_local", dims=dims)
            outcome, most, submitted = _drive_failing(block, order=order)
            where = f"cap {cap}, {starts} starts, {dims} variables, order {order}"
            assert most == 0, f"the search never asks for nothing while nothing runs ({where})"
            assert outcome["stopped_because"] == "failures", where
            assert 0 < submitted <= cap + 1, f"{submitted} designs went to the pool ({where})"


def test_a_design_that_works_after_a_few_failures_lets_the_search_go_on() -> None:
    for dims in (3, 6):
        outcome, most, submitted = _drive_failing(
            _block(starts=2, per_start=40, method="bounded_local", dims=dims), order=0, works_from=CAP - 3)
        assert most == 0
        assert outcome["stopped_because"] != "failures"
        assert outcome["evaluations"] > CAP


@pytest.mark.asyncio
async def test_an_empty_stream_request_with_no_design_running_is_an_error_not_a_loop(tmp_path: Path,
                                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def asks_for_nothing(*_a: Any, **_k: Any) -> Any:
        yield {"kind": "evaluate_stream", "requests": []}

    monkeypatch.setattr(optimise._search, "compare", asks_for_nothing)
    root = _quest(tmp_path)
    with pytest.raises(RuntimeError, match="asked the pool for nothing"):
        await _run(root, _protocol(evaluation_budget={"starts": 2, "per_start": 6}), at_once=2)
