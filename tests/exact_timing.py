"""An exact clock for the tests that time a simulation (core/run_estimate.py, core/run_checks.py).

FI times a simulation with two clocks, and both read the machine's real clock: the simulation's own trial time (perf_counter
in the child process) and, for a single deterministic trial, the parent's wall time of the whole process, which includes the
interpreter's start-up. Under load either moves, so a decision made on a timing (whether a setting is "costly", whether a
study fits its time) can flip from one run to the next.

Two parts make the timing exact:

* :data:`VIRTUAL_CLOCK` is put at the top of a test simulation. The child's clock is then the simulation's own: only its
  ``time.sleep`` calls move it, so each trial costs exactly what the simulation says.
* :func:`install` replaces FI's parent clock (``time.monotonic`` of the timing modules) with one that moves by exactly the
  trial seconds of each timing run, and makes a timing run cost what its trials cost, with no start-up.

Use :func:`install` from an autouse fixture in a test module that times simulations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import run_checks, run_estimate

VIRTUAL_CLOCK = '''\
import time

_now = [0.0]
time.perf_counter = lambda: _now[0]  # the simulation's clock: only its own sleeps move it
time.sleep = lambda seconds: _now.__setitem__(0, _now[0] + seconds)
'''


def trial_seconds(out: Path) -> float:
    """What the trials in one timing run's result file cost, in the simulation's own clock."""
    try:
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return 0.0
    return sum(float(r.get("duration_s") or 0.0) for r in rows if isinstance(r, dict))


def install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make FI's timing read the exact clock for the rest of the test."""
    clock = {"now": 0.0}
    run = run_estimate._outside.execute_keeping_only_finished

    async def timed(executor: Any, cmd: list[str], **kw: Any) -> Any:
        result = await run(executor, cmd, **kw)
        if len(cmd) > 3:  # a timing run: [python, harness, spec, results]
            clock["now"] += trial_seconds(Path(kw["cwd"]) / cmd[3])
        return result

    own = SimpleNamespace(monotonic=lambda: clock["now"], time=time.time, time_ns=time.time_ns)
    monkeypatch.setattr(run_estimate, "time", own)
    monkeypatch.setattr(run_checks, "time", own)
    monkeypatch.setattr(run_estimate._outside, "execute_keeping_only_finished", timed)
