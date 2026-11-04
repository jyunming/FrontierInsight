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

from core import optimise as optimise_mod
from core import run_estimate as estimate
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
    return await estimate.entry_returns_study(VenvExecutor(), sys.executable, root, "code/simulate.py", dict(CELL),
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
async def test_a_repair_that_keeps_the_study_result_passes(tmp_path: Path) -> None:
    # The names are read as the pilot reads them: the plan's id as it is, not a normalised spelling.
    root = _quest(tmp_path, KEEPS_OBJECTIVE)
    assert await _entry(root, {"Yield_Rate"}) == ""
    assert "no longer returns" in await _entry(root, {"yieldrate"})


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


def _gave(engine: Engine, *names: str) -> None:
    """FI's record of what the study's entry returned before the repair (its small run's record)."""
    (engine.quest_root / ".fi").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / ".fi" / "small_run.json").write_text(json.dumps({"returned": list(names)}), encoding="utf-8")


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
    _gave(engine, "yield_rate")
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
    _gave(engine, "yield_rate")
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
    _gave(engine, "yield_rate")  # FI's record: the entry returns yield_rate; relative_gain is the analysis's
    state: dict[str, Any] = {"design": {"protocol": protocol}}
    keeps = KEEPS_OBJECTIVE.replace("Yield_Rate", "yield_rate")
    (code / "simulate.py").write_text(keeps, encoding="utf-8")
    assert await engine._simulation_repair_fault(state, keeps) == ""
    (code / "simulate.py").write_text(CHECK_ONLY, encoding="utf-8")
    assert "no longer returns the study's results" in await engine._simulation_repair_fault(state, CHECK_ONLY)


RUN_CELL_ONLY_LOST = '''
def run_cell(cell):
    return {"check_error": 0.0}
'''

RUN_CELL_ONLY_KEPT = '''
def run_cell(cell):
    return {"yield_rate": cell["width"] * 2.0}
'''

KEYED_BY_PITCH = '''
def run_trial(cell, trial_id=0, seed=0):
    return {"yield_rate": {"pitch=%d" % p: p * 0.01 for p in cell["pitch"]}}
'''


@pytest.mark.asyncio
async def test_a_run_cell_only_simulation_that_loses_its_results_is_caught_without_trial_mode(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    assert not getattr(engine, "_trial_mode", False)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    _gave(engine, "yield_rate")
    state: dict[str, Any] = {"design": {"protocol": PROTOCOL}}
    (code / "simulate.py").write_text(RUN_CELL_ONLY_LOST, encoding="utf-8")
    assert "run_cell() on the plan's first setting returned check_error" in await engine._simulation_repair_fault(
        state, RUN_CELL_ONLY_LOST)
    (code / "simulate.py").write_text(RUN_CELL_ONLY_KEPT, encoding="utf-8")
    assert await engine._simulation_repair_fault(state, RUN_CELL_ONLY_KEPT) == ""


@pytest.mark.asyncio
async def test_a_headline_keyed_by_a_value_of_the_run_is_read_at_that_value(tmp_path: Path) -> None:
    root = _quest(tmp_path, KEYED_BY_PITCH)
    why = await estimate.entry_returns_study(VenvExecutor(), sys.executable, root, "code/simulate.py", {},
                                             wanted={"yield_rate"}, timeout_s=60, in_each_run={"pitch": [74, 90]})
    assert why == "", "yield_rate at pitch=74 and pitch=90 is the headline, read as the pilot reads it"
    lost = _quest(tmp_path / "x", "def run_trial(cell, trial_id=0, seed=0):\n    return {'other': {'pitch=%d' % p: 1.0 for p in cell['pitch']}}\n")
    why = await estimate.entry_returns_study(VenvExecutor(), sys.executable, lost, "code/simulate.py", {},
                                             wanted={"yield_rate"}, timeout_s=60, in_each_run={"pitch": [74, 90]})
    assert "none of yield_rate" in why


SEARCH_KEPT = '''
def run_trial(cell, trial_id=0, seed=0):
    return {"obj": cell["x"] * 2.0, "side": 1.0}
'''

SEARCH_LOST = '''
def run_trial(cell, trial_id=0, seed=0):
    return {"side": 1.0}
'''


@pytest.mark.asyncio
async def test_a_search_is_checked_through_its_own_evaluation_and_reader(tmp_path: Path) -> None:
    block = {"objective": {"quantity": "obj", "direction": "maximise"}}
    for source, expect_fault in ((SEARCH_KEPT, False), (SEARCH_LOST, True)):
        root = _quest(tmp_path / ("a" if not expect_fault else "b"), source)
        why = await optimise_mod.search_entry_returns_objective(
            VenvExecutor(), sys.executable, root, "code/simulate.py", block, {"x": 1.0}, timeout_s=60, env=None,
            thresholds=None)
        assert bool(why) is expect_fault, why
        if expect_fault:
            assert "not the objective `obj`" in why


PACKAGE_SIM = '''
import pk.model as model

def run_cell(cell):
    return {"y": model.value()}

def run_trial(cell, trial_id=0, seed=0):
    return run_cell(cell)
'''


@pytest.mark.asyncio
async def test_a_change_in_the_package_alone_makes_the_measurement_stale(tmp_path: Path) -> None:
    root = _quest(tmp_path, PACKAGE_SIM)
    package = root / "code" / "pk"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "model.py").write_text("def value():\n    return 1.0\n", encoding="utf-8")
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL), timeout_s=60)
    assert tr.trial_aggregates(source, root) == ""
    (package / "model.py").write_text("import random\n\ndef value():\n    return random.random()\n", encoding="utf-8")
    assert tr.trial_aggregates(source, root), "the file is the same, the package is not: measured again"


class _Clock:
    """``time`` for the engine, with a monotonic clock the test moves."""

    def __init__(self) -> None:
        import time as real

        self._real, self.now = real, 1000.0

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@pytest.mark.asyncio
async def test_the_check_of_one_repair_has_one_budget_and_never_a_third_call(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import core.engine as engine_mod

    engine = _engine(tmp_path)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    clock = _Clock()
    monkeypatch.setattr(engine_mod, "time", clock)
    called: list[str] = []

    async def entry(*a: Any, **k: Any) -> str:
        called.append("entry")
        return ""

    monkeypatch.setattr(estimate, "entry_returns_study", entry)
    state: dict[str, Any] = {"design": {"protocol": PROTOCOL}}
    (code / "simulate.py").write_text(DELEGATING, encoding="utf-8")

    async def settled_late(*a: Any, **k: Any) -> bool:
        clock.now += 500.0  # the two seed calls used more than the whole budget
        return True

    monkeypatch.setattr(tr, "settle_aggregate", settled_late)
    assert await engine._simulation_repair_fault(state, DELEGATING) == ""
    assert called == [], "the budget is spent: the study's entry is not called"
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "ran out of its" in log and "checking the repaired code took 500 s" in log

    async def not_measured(*a: Any, **k: Any) -> None:
        return None

    monkeypatch.setattr(tr, "settle_aggregate", not_measured)
    engine._entry_judged_fine = None
    assert await engine._simulation_repair_fault(state, DELEGATING + "# again\n") == ""
    assert called == [], "not judged after the seed calls: no third call"


COUNTS = '''
def run_trial(cell, trial_id=0, seed=0):
    return {"yield_rate_count": 9, "yield_rate_total": 10}
'''

SLOW_TRIAL = '''
import time

def run_trial(cell, trial_id=0, seed=0):
    time.sleep(2.5)
    return {"y": 1.0}
'''


@pytest.mark.asyncio
async def test_a_metric_returned_as_its_counts_is_a_returned_metric(tmp_path: Path) -> None:
    # core/metric_spec.py reads a metric as its id, or `<id>_count` and `<id>_total`, or `<id>_values`.
    root = _quest(tmp_path, COUNTS)
    assert await _entry(root, {"yield_rate"}) == ""


@pytest.mark.asyncio
async def test_the_two_seed_calls_share_the_time_they_are_given(tmp_path: Path) -> None:
    root = _quest(tmp_path, SLOW_TRIAL)
    # each call takes ~2.5 s: with 4 s for both, the second has less than it needs (before: each was given the whole 4 s)
    done = await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL), timeout_s=4)
    assert done is None


def test_the_cache_key_follows_every_file_the_simulation_reads(tmp_path: Path) -> None:
    root = _quest(tmp_path, DELEGATING)
    (root / "code" / "helper.py").write_text("X = 1\n", encoding="utf-8")
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    before = tr.code_inputs_sha(root, source)
    (root / "code" / "helper.py").write_text("X = 2\n", encoding="utf-8")
    assert tr.code_inputs_sha(root, source) != before, "a helper beside simulate.py is part of the key"
    (root / "inputs").mkdir()
    (root / "inputs" / "table.csv").write_text("1,2\n", encoding="utf-8")
    assert tr.code_inputs_sha(root, source) != before


@pytest.mark.asyncio
async def test_the_record_of_a_measurement_is_stale_when_a_helper_changes(tmp_path: Path) -> None:
    root = _quest(tmp_path, "import helper\n\ndef run_cell(cell):\n    return {'yield_rate': helper.value()}\n\n"
                            "def run_trial(cell, trial_id=0, seed=0):\n    return run_cell(cell)\n")
    (root / "code" / "helper.py").write_text("def value():\n    return 1.0\n", encoding="utf-8")
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL), timeout_s=60)
    assert tr.trial_aggregates(source, root) == ""
    (root / "code" / "helper.py").write_text("import random\n\ndef value():\n    return random.random()\n", encoding="utf-8")
    assert tr.trial_aggregates(source, root), "the helper changed: measured again"


@pytest.mark.asyncio
async def test_a_search_whose_evaluation_failed_is_not_judged_and_says_why(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    root = _quest(tmp_path, "def run_trial(cell, trial_id=0, seed=0):\n    raise RuntimeError('boom')\n")
    log = logging.getLogger("entry-check-test")
    with caplog.at_level(logging.WARNING, logger="entry-check-test"):
        why = await optimise_mod.search_entry_returns_objective(
            VenvExecutor(), sys.executable, root, "code/simulate.py", {"objective": {"quantity": "obj"}}, {"x": 1.0},
            timeout_s=60, env=None, thresholds=None, log=log)
    assert why == ""
    assert "not judged" in caplog.text and "boom" in caplog.text


PK_SIM = '''
import pk.model as model

def run_trial(cell, trial_id=0, seed=0):
    return model.run(cell)
'''


def _package_quest(engine: Engine) -> Path:
    code = engine.quest_root / "code"
    (code / "pk").mkdir(parents=True, exist_ok=True)
    (code / "pk" / "__init__.py").write_text("", encoding="utf-8")
    (code / "pk" / "model.py").write_text("def run(cell):\n    return {'yield_rate': cell['width'] * 2.0}\n", encoding="utf-8")
    (code / "simulate.py").write_text(PK_SIM, encoding="utf-8")
    return code


@pytest.mark.asyncio
async def test_a_package_only_oracle_repair_that_removes_the_study_result_is_put_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import code_layout as cl

    engine = _engine(tmp_path)
    engine._trial_mode = True
    code = _package_quest(engine)
    cl.save(engine.quest_root, {"shape": cl.PACKAGE, "package": "pk"})
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    _gave(engine, "yield_rate")
    before = (code / "pk" / "model.py").read_text(encoding="utf-8")
    oracle = {"name": "zero", "kind": "special_case", "check": "c", "expected": 0.0, "tolerance": 0.1,
              "case": {"width": 2.0}, "measure": "check_error"}

    async def answer(prompt: str, **kw: Any) -> str:
        return json.dumps({"code": "", "package_files": {"pk/model.py": "def run(cell):\n    return {'check_error': 0.0}\n"},
                           "patch_summary": "the package returns the check"})

    monkeypatch.setattr(engine, "_chat", answer)
    new, failed, outcome = await engine._repair_script_for_oracle(
        {"design": {"protocol": PROTOCOL}}, code / "simulate.py", [oracle], ["zero is off"], passing=[])
    assert outcome == "not_usable" and new is None
    assert (code / "pk" / "model.py").read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_with_no_record_that_the_entry_returns_a_headline_nothing_is_judged(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    (code / "simulate.py").write_text(CHECK_ONLY, encoding="utf-8")
    state: dict[str, Any] = {"design": {"protocol": PROTOCOL}}
    assert await engine._simulation_repair_fault(state, CHECK_ONLY) == "", "never a refusal on a guess"
    assert "was not judged" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
