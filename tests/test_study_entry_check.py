"""After a repair of simulate.py FI calls the function it will call for the study, and keeps the repair only when that call
still returns the study's results (core/trial_runner.py::entry_returns_study, Engine._simulation_repair_fault); a
``run_trial`` that delegates to ``run_cell`` is kept when FI called it with two seeds and the numbers were the same
(core/trial_runner.py::settle_aggregate). Neutral made-up simulations, run through FI's real harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import trial_runner as tr
from core.config import Config, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.execution import VenvExecutor

CELL = {"width": 2.0}

# A check branch and a study: the repaired run_trial that sends every setting to the check branch returns no study result.
CHECK_ONLY = '''
def _check(cell):
    return {"check_error": 0.0}

def run_cell(cell):
    return {"yield_rate": cell["width"] * 2.0, "label": "x"}

def run_trial(cell, trial_id=0, seed=0):
    return _check(cell)
'''

DELEGATING = '''
def run_cell(cell):
    return {"yield_rate": cell["width"] * 2.0}

def run_trial(cell, trial_id=0, seed=0):
    return run_cell(cell)
'''

# A run_trial that returns run_cell's number, which is a mean over random trials: its numbers change with the seed.
RANDOM_MEAN = '''
import random

def run_cell(cell):
    return {"yield_rate": sum(random.random() for _ in range(5)) / 5}

def run_trial(cell, trial_id=0, seed=0):
    return run_cell(cell)
'''

KEEPS_OBJECTIVE = '''
def _check(cell):
    return {"check_error": 0.0}

def run_trial(cell, trial_id=0, seed=0):
    if cell.get("mode") == "check":
        return _check(cell)
    return {"Yield_Rate": cell["width"] * 2.0}
'''


def _quest(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return root


async def _entry(root: Path, wanted: set[str]) -> str:
    return await tr.entry_returns_study(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL),
                                        wanted=wanted, timeout_s=60)


def test_the_study_entry_is_one_rule_for_every_caller() -> None:
    assert tr.study_entry({"run_trial", "run_cell"}) == "run_trial"
    assert tr.study_entry({"run_cell"}) == "run_cell"
    assert tr.study_entry(set()) == ""


@pytest.mark.asyncio
async def test_a_run_trial_that_sends_every_setting_to_the_check_returns_no_study_result(tmp_path: Path) -> None:
    root = _quest(tmp_path, CHECK_ONLY)
    why = await _entry(root, {"yield_rate"})
    assert "no longer returns the study's results" in why and "check_error" in why and "yield_rate" in why


@pytest.mark.asyncio
async def test_a_repair_that_keeps_the_study_result_passes_whatever_the_case_of_its_name(tmp_path: Path) -> None:
    # The plan's headline ids are kept lower case without underscores; the code may spell the name as it likes.
    root = _quest(tmp_path, KEEPS_OBJECTIVE)
    assert await _entry(root, {"yieldrate"}) == ""
    assert await _entry(root, {"Yield_Rate"}) == ""


@pytest.mark.asyncio
async def test_nothing_is_judged_when_no_result_is_named_or_the_call_cannot_be_made(tmp_path: Path) -> None:
    root = _quest(tmp_path, CHECK_ONLY)
    assert await _entry(root, set()) == ""
    broken = _quest(tmp_path / "b", "def run_trial(cell, trial_id=0, seed=0):\n    raise RuntimeError('boom')\n")
    assert await _entry(broken, {"yield_rate"}) == "", "a crash is judged by the runs that follow, not by this check"


@pytest.mark.asyncio
async def test_a_run_trial_that_delegates_to_a_deterministic_run_cell_is_kept_once_fi_called_it_twice(tmp_path: Path) -> None:
    root = _quest(tmp_path, DELEGATING)
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert tr.trial_aggregates(source, root), "the reading of the code alone still names it"
    assert await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL), timeout_s=60)
    assert tr.trial_aggregates(source, root) == "", "the measured fact is what decides"
    assert tr.trial_aggregates(source + "# changed\n", root), "another text of the file is measured again"
    record = json.loads((root / tr.ONE_TRIAL_RECORD).read_text(encoding="utf-8"))
    assert record["same"] is True


@pytest.mark.asyncio
async def test_a_run_trial_that_returns_a_random_mean_is_still_named(tmp_path: Path) -> None:
    root = _quest(tmp_path, RANDOM_MEAN)
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert not await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL), timeout_s=60)
    assert "run_trial must run ONE trial" in tr.trial_aggregates(source, root)


def _engine(tmp_path: Path) -> Engine:
    return Engine(Config(
        topic="t", title="entry-check", provider=ProviderConfig(name="openai"), execution=ExecutionConfig(sandbox="venv"),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


PROTOCOL: dict[str, Any] = {"grid": {"width": [1.0, 2.0, 3.0]}, "metrics": [{"id": "yield_rate", "kind": "mean"}]}


@pytest.mark.asyncio
async def test_the_engine_rejects_a_repair_with_no_study_result_and_keeps_one_that_has_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    engine._trial_mode = True
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    (code / "simulate.py").write_text(CHECK_ONLY, encoding="utf-8")
    state: dict[str, Any] = {"design": {"protocol": PROTOCOL}}
    assert "no longer returns the study's results" in await engine._simulation_repair_fault(state, CHECK_ONLY)
    (code / "simulate.py").write_text(KEEPS_OBJECTIVE.replace("Yield_Rate", "yield_rate"), encoding="utf-8")
    assert await engine._simulation_repair_fault(state, (code / "simulate.py").read_text(encoding="utf-8")) == ""
    (code / "simulate.py").write_text(DELEGATING, encoding="utf-8")
    assert await engine._simulation_repair_fault(state, DELEGATING) == "", "a delegating run_trial of a deterministic study"
    (code / "simulate.py").write_text(RANDOM_MEAN, encoding="utf-8")
    assert "run_trial must run ONE trial" in await engine._simulation_repair_fault(state, RANDOM_MEAN)


@pytest.mark.asyncio
async def test_an_oracle_repair_with_no_study_result_is_put_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    engine._trial_mode = True
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    before = DELEGATING
    (code / "simulate.py").write_text(before, encoding="utf-8")
    oracle = {"name": "zero", "kind": "special_case", "check": "c", "expected": 0.0, "tolerance": 0.1,
              "case": {"width": 2.0}, "measure": "check_error"}

    async def answer(prompt: str, **kw: Any) -> str:
        return json.dumps({"code": CHECK_ONLY, "patch_summary": "the check reads the check branch"})

    monkeypatch.setattr(engine, "_chat", answer)
    new, failed, outcome = await engine._repair_script_for_oracle(
        {"design": {"protocol": PROTOCOL}}, code / "simulate.py", [oracle], ["zero is off"], passing=[])
    assert new is None and not failed and outcome == "not_usable"
    assert (code / "simulate.py").read_text(encoding="utf-8") == before
    assert "no longer returns the study's results" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


SLOW_DELEGATING = '''
import time

def run_cell(cell):
    time.sleep(30)
    return {"yield_rate": 1.0}

def run_trial(cell, trial_id=0, seed=0):
    return run_cell(cell)
'''


@pytest.mark.asyncio
async def test_a_call_that_cannot_be_finished_leaves_no_guessed_shape_rule_in_force(tmp_path: Path) -> None:
    root = _quest(tmp_path, SLOW_DELEGATING)
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert tr.trial_aggregates(source, root), "the code's shape alone names it before anything is measured"
    done = await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL), timeout_s=3)
    assert done is None, "not measured is neither the same nor different"
    assert tr.trial_aggregates(source, root) == "", "the shape rule alone does not refuse"


@pytest.mark.asyncio
async def test_a_plan_with_a_run_level_and_an_analysis_derived_metric_passes_on_the_run_level_one(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    engine._trial_mode = True
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    protocol = {"grid": {"width": [1.0, 2.0]}, "metrics": [{"id": "yield_rate", "kind": "mean"},
                                                          {"id": "relative_gain", "kind": "mean"}]}  # derived in analysis
    monkeypatch.setattr(engine, "_protocol_block", lambda state: protocol)
    state: dict[str, Any] = {"design": {"protocol": protocol}}
    keeps = KEEPS_OBJECTIVE.replace("Yield_Rate", "yield_rate")
    (code / "simulate.py").write_text(keeps, encoding="utf-8")
    assert await engine._simulation_repair_fault(state, keeps) == ""
    (code / "simulate.py").write_text(CHECK_ONLY, encoding="utf-8")
    assert "no longer returns the study's results" in await engine._simulation_repair_fault(state, CHECK_ONLY)
