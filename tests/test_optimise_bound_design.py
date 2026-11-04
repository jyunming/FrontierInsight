"""A result the design marks ``may_equal_bound`` is not sent back by the search for the best design when it sits on its
bound at every design the search tries; the same result unmarked still is (the design reaches the search's checks)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from core import optimise
from tests.test_flat_output import FLAT_SIM, _block, _quest
from tests.test_optimise_runner import LocalExecutor

METRICS = [{"id": p, "kind": "proportion", "estimand": "a made-up share", "unit": "one run"} for p in ("peak", "mass")]


def _marked(flag: bool) -> dict:
    return {"result_assertions": [{"path": p, "min": 0, "max": 1, **({"may_equal_bound": True} if flag else {})}
                                  for p in ("peak", "mass")]}


def test_watched_results_leave_out_a_result_the_design_marks() -> None:
    protocol = {"optimisation": _block(), "metrics": METRICS}
    assert optimise.watched_results(protocol, _marked(True)) == []
    assert optimise.watched_results(protocol, _marked(False)) == ["peak", "mass"]
    assert optimise.watched_results(protocol) == ["peak", "mass"], "no design, no mark"


async def _run(tmp_path: Path, design: dict | None) -> tuple[optimise.OptimisationRunner, object]:
    root = _quest(tmp_path, FLAT_SIM)
    runner = optimise.OptimisationRunner(
        LocalExecutor(), quest_root=root, protocol={"optimisation": _block(), "metrics": METRICS},
        simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py", design=lambda: design)
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=300)
    return runner, result


@pytest.mark.asyncio
async def test_a_marked_result_at_its_bound_in_every_design_is_not_sent_back(tmp_path: Path) -> None:
    """Old code: the search was given the protocol only, so the mark was never read and the honest 0 was a flat output."""
    runner, result = await _run(tmp_path, _marked(True))
    assert runner.flat is None and runner.failed_script is None, result.stderr[-800:]
    assert "RESULT_JSON" in result.stdout


@pytest.mark.asyncio
async def test_an_unmarked_result_at_every_design_is_still_sent_back(tmp_path: Path) -> None:
    """Known positive: the same search with no mark is still a simulation that does not use its design."""
    runner, result = await _run(tmp_path, _marked(False))
    assert result.returncode == 1 and runner.failed_script == "simulate.py" and runner.flat
