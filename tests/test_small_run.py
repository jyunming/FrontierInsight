"""FI runs the simulation on a small case before the code is read (core/small_run.py, RunChecksMixin._small_run_gate): a
crash goes to the repair of a failed run with its place and traceback, the reader is shown what the run gave, and a run of
the same code is not made again. Live on the code r12 delivered (tests/fixtures/replay/r12_code/): one call at the noisy
corner shows the solver failure at ``model.py`` line 121 in seconds, where the quest spent its review rounds on tests that
only showed this crash."""

from __future__ import annotations

import asyncio
import shutil
import string
import sys
from pathlib import Path
from typing import Any

import pytest

from core import small_run
from core.execution import SharedInterpreterExecutor

R12 = Path(__file__).parent / "fixtures" / "replay" / "r12_code"
GRID = {"noise_std_rel": [0.005, 0.01, 0.02, 0.04, 0.08], "n_timepoints": [5, 8, 12, 20, 32]}


def test_the_settings_are_the_middle_and_the_two_corners() -> None:
    assert small_run.settings(GRID) == [{"noise_std_rel": 0.02, "n_timepoints": 12},
                                        {"noise_std_rel": 0.005, "n_timepoints": 5},
                                        {"noise_std_rel": 0.08, "n_timepoints": 32}]
    assert small_run.settings({"method": ["a", "b"]}) == [{"method": "a"}, {"method": "b"}]
    assert small_run.settings({}) == [{}]
    assert small_run.budget(3600) == 180 and small_run.budget(100) == 30 and small_run.budget(None) == 30


def _quest(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    shutil.copytree(R12, root / "code")
    return root


def test_replay_r12_the_small_run_shows_the_crash_and_its_place(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    root = _quest(tmp_path)
    got = asyncio.run(small_run.run(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root,
                                    "code/simulate.py", GRID, deterministic=False, timeout_s=120, paired=True))
    assert got.crashed and [s.key for s in got.faults] == ["noise_std_rel=0.08,n_timepoints=32"]
    assert "model.py line 121" in got.faults[0].reasons[0][0]
    message = got.directive()
    assert "did not satisfy required optimality" in message and "Traceback" in message and "line 121" in message
    shown = got.for_reader()
    assert "Setting noise_std_rel=0.02,n_timepoints=12: 3 of 3 trial(s) ran" in shown and "weighted_Ea_hat=" in shown
    assert not (root / "raw").exists(), "the study's raw/ is never touched"


class _Counting(SharedInterpreterExecutor):
    calls = 0

    async def execute(self, cmd, **kw):  # noqa: ANN001, ANN003
        type(self).calls += 1
        return await super().execute(cmd, **kw)


@pytest.mark.usefixtures("this_python")
def test_the_gate_sends_a_crash_to_repair_and_reads_back_an_unchanged_run(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    from tests.test_disclosure import _engine

    root = _quest(tmp_path)
    eng = _engine(root)
    eng.executor = _Counting(python_version="3.11")  # type: ignore[assignment]
    eng._trial_mode, eng._trial_entries = True, {"run_trial"}  # type: ignore[attr-defined]
    eng._protocol_block = lambda state: {"grid": GRID, "metrics": [{"id": "x", "paired": True}]}  # type: ignore[assignment]
    state: dict[str, Any] = {"exec_reflect_iter": 0}
    sent = asyncio.run(eng._small_run_gate(state, sys.executable, {}))
    assert sent is not None and sent["exec_result"]["failed_script"] == "simulate.py"
    assert "model.py line 121" in sent["exec_result"]["stderr_tail"]
    ran = _Counting.calls
    assert asyncio.run(eng._small_run_gate(state, sys.executable, {})) is not None
    assert _Counting.calls == ran, "the same code is not run again"
    # no repair left: the study's own run decides
    assert asyncio.run(eng._small_run_gate({"exec_reflect_iter": 99}, sys.executable, {})) is None
    # the crash repaired (the solver's tolerance relaxed): the small run passes and the reader is shown what it gave
    model = root / "code" / "kinetics_parameter_recovery" / "model.py"
    model.write_text(model.read_text(encoding="utf-8").replace("abs(float(gradient[0])) >= solver_gtol",
                                                               "abs(float(gradient[0])) >= 1e-6"), encoding="utf-8")
    assert asyncio.run(eng._small_run_gate(state, sys.executable, {})) is None
    assert "Setting noise_std_rel=0.08,n_timepoints=32: 3 of 3 trial(s) ran" in eng._small_run_note  # type: ignore[attr-defined]
    assert asyncio.run(eng._small_run_again()) == ""


def test_the_reader_is_shown_the_small_run() -> None:
    text = (Path(__file__).parent.parent / "agents" / "code_review.md").read_text(encoding="utf-8")
    filled = string.Template(text).substitute(topic="t", requirements="[]", plan="{}", code="c", previous="",
                                              small_run="Setting a=1: 3 of 3 trial(s) ran")
    assert "What the code gave when FI ran it" in filled and "Setting a=1: 3 of 3" in filled
