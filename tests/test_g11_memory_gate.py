"""The memory gate sizes the settings that run at once from the memory one setting measured, and keeps that size from
one batch of a search to the next (group G11, backlog row 2026-10-24-119).

Before the fix every batch of a search (each ``evaluate_many`` call, and each new pool) built a new ``SettingsGate``. A
new gate has measured nothing, so its first settings were spaced out and the gate logged "holding settings back" even
though memory was not the reason. A real run logged that 133 times with about 20 GB free and ran at about a third of its
speed. The gate's own sizing is right: it allows the configured maximum when memory is plentiful and one setting when it
is short. The fault was that the size was thrown away between batches, and that the spacing was reported as a memory
limit.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from core import optimise
from core.execution import SettingsGate
from tests.test_optimise_runner import LocalExecutor, _quest

# A simulation that takes a moment, so that several settings overlap in time.
SLOW_SIM = '''\
import time


def run_cell(cell):
    time.sleep(1.0)
    return {"f": cell["x"] ** 2}
'''


def _gate_at(at_once: int, free_mb: float) -> SettingsGate:
    # Memory of a 63 GB computer (the reserve is a tenth of it), with ``free_mb`` free, and no ramp-up to wait for.
    gate = SettingsGate(at_once, free_mb=lambda: free_mb, total_mb=63_000.0, poll_s=0.01)
    gate.RAMP_S = 0.0
    return gate


def test_with_20_gb_free_and_1_gb_per_setting_all_the_configured_settings_run_at_once() -> None:
    """With plenty of memory the gate allows the configured maximum (8), each setting sized from its measured 1 GB.

    Passes on the old code too: the old gate sized correctly within one gate. The fault that this file pins is the
    batch-level one, in :func:`test_a_second_batch_of_the_same_search_starts_at_full_size`."""
    gate = _gate_at(8, 20_000.0)
    gate.note(1_000.0)

    async def go() -> int:
        async def one() -> None:
            async with gate.slot():
                await asyncio.sleep(0.2)

        await asyncio.gather(*(one() for _ in range(8)))
        return gate.most_at_once

    assert asyncio.run(go()) == 8


def test_with_1_5_gb_free_and_1_gb_per_setting_only_one_runs_at_a_time(caplog) -> None:  # noqa: ANN001
    """Known positive: when memory is really short the protection still holds: one setting at a time (and the one
    setting always runs, so nothing waits forever)."""
    gate = _gate_at(8, 1_500.0)
    gate.note(1_000.0)
    gate._log = logging.getLogger("g11")

    async def go() -> int:
        async def one() -> None:
            async with gate.slot():
                await asyncio.sleep(0.05)

        await asyncio.gather(*(one() for _ in range(3)))
        return gate.most_at_once

    with caplog.at_level(logging.WARNING, logger="g11"):
        assert asyncio.run(go()) == 1
    assert any("holding settings back" in r.getMessage() for r in caplog.records)


def test_spacing_while_nothing_is_measured_is_not_reported_as_a_memory_hold(caplog) -> None:  # noqa: ANN001
    """Old code: the first settings of a fresh gate wait for the spacing (STAGGER) and logged "holding settings back" on
    the way, although memory was not the reason. The new code logs only a real memory limit."""
    gate = _gate_at(4, 20_000.0)
    gate.RAMP_S = 6.0
    gate.STAGGER = 0.1
    gate._log = logging.getLogger("g11")

    async def go() -> None:
        async def one() -> None:
            async with gate.slot():
                await asyncio.sleep(0.05)

        await asyncio.gather(*(one() for _ in range(3)))

    with caplog.at_level(logging.WARNING, logger="g11"):
        asyncio.run(go())
    assert not any("holding settings back" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]


class _PeakExecutor(LocalExecutor):
    """A local executor that reports each run's largest memory as 1000 MB, the way a measured setting does, and counts
    how many runs are in progress at once."""

    def __init__(self) -> None:
        super().__init__()
        self.in_progress = 0
        self.most_in_progress = 0

    async def execute(self, cmd: list[str], *, cwd: Path, timeout_s: int, env: dict[str, str] | None = None) -> Any:
        self.in_progress += 1
        self.most_in_progress = max(self.most_in_progress, self.in_progress)
        try:
            result = await super().execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)
            return dataclasses.replace(result, peak_mb=1_000.0)
        finally:
            self.in_progress -= 1


@pytest.mark.asyncio
async def test_a_second_batch_of_the_same_search_starts_at_full_size(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    """Two batches of one search, 8 settings each, 20 GB free, 1 GB per setting, the same gate passed to both.

    Old code: ``evaluate_many`` has no ``gate`` argument (the call fails), and a batch built its own fresh gate, so the
    second batch was spaced out and logged a memory hold it did not have. New code: the second batch starts all 8 at
    once, and no "holding settings back" line is written."""
    root = _quest(tmp_path, SLOW_SIM)
    gate = SettingsGate(8, log=logging.getLogger("g11"), free_mb=lambda: 20_000.0, total_mb=63_000.0, poll_s=0.01)
    executor = _PeakExecutor()
    cells = [{"x": k, "y": 0.0, "scale": 1.0} for k in range(8)]

    async def batch() -> None:
        executor.most_in_progress = 0
        await optimise.evaluate_many(executor, sys.executable, root, "code/simulate.py", "run_cell", cells, at_once=8,
                                     runs=1, base_seed=0, deadline=time.monotonic() + 600, env=None, thresholds=None,
                                     log=logging.getLogger("g11"), gate=gate)

    with caplog.at_level(logging.WARNING, logger="g11"):
        await batch()  # measures the size of a setting
        await batch()  # the same search, the next batch: must start at full size

    assert executor.most_in_progress == 8, executor.most_in_progress
    assert not any("holding settings back" in r.getMessage() for r in caplog.records)
