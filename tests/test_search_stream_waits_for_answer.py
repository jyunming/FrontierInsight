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
from typing import Any

from core import optimise_search as search

from tests.test_search_stream import _block, _scan_rows

NO_WORK = {"failed": "the toy simulation gave no result"}


def _drive_failing(block: dict[str, Any], *, order: int, empty_limit: int = 200) -> tuple[dict[str, Any], int]:
    """Every design fails. The driver answers a stream request as the real pool does: it answers at once with what
    finished, and with nothing at all when nothing is running. More than ``empty_limit`` such empty answers in a row is a
    spin: the search is asking for nothing, again."""
    gen = search.search(block, seed=1, batch=True, stream=True)
    rng = random.Random(order)
    running: dict[Any, dict[str, Any]] = {}
    empty = most = 0
    try:
        request = next(gen)
        while True:
            if request["kind"] == "scan":
                request = gen.send([dict(r, values={"f": float("nan")}) for r in _scan_rows(request, block, lambda d: 0)])
            elif request["kind"] == "evaluate_many":
                request = gen.send([dict(NO_WORK) for _ in request["requests"]])
            elif request["kind"] == "evaluate_stream":
                for r in request["requests"]:
                    running[r["id"]] = r
                if not running:
                    empty += 1
                    most = max(most, empty)
                    assert empty <= empty_limit, "the search asks the pool for nothing while nothing runs: a spin"
                    request = gen.send([])
                    continue
                empty = 0
                pick = rng.choice(sorted(running))
                running.pop(pick)
                request = gen.send([(pick, dict(NO_WORK))])
            else:
                request = gen.send(dict(NO_WORK))
    except StopIteration as done:
        return done.value, most


def test_a_stream_with_no_answer_that_worked_yet_still_sends_its_held_designs_when_nothing_runs() -> None:
    for starts in (2, 3, 8):
        for order in range(3):
            outcome, most = _drive_failing(_block(starts=starts, per_start=30, method="bounded_local", dims=3),
                                           order=order)
            assert most == 0, f"the search never asks for nothing while nothing runs ({starts} starts, order {order})"
            assert outcome["stopped_because"] == "failures"
            # The hold-back still works: a broken simulation costs a few designs, not the whole batch.
            assert outcome["evaluations"] <= 16, outcome["evaluations"]
