"""After a repair of simulate.py FI calls the function it will call for the study, and keeps the repair only when that call
still returns the study's results (core/trial_runner.py::entry_returns_study, Engine._simulation_repair_fault); a
``run_trial`` that runs ``run_cell`` is told from FI's own record of one call of it: ``run_cell`` entered and a random
generator drawn from inside it (core/trial_runner.py::settle_aggregate). Neutral made-up simulations, run through FI's
real harness."""

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

# A study that gives the same results through its own run_trial and does not enter run_cell from it.
ONE_TRIAL = '''
def run_cell(cell):
    return {"yield_rate": cell["width"] * 2.0}

def run_trial(cell, trial_id=0, seed=0):
    return {"yield_rate": cell["width"] * 2.0}
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
async def test_a_run_trial_that_delegates_to_a_run_cell_without_randomness_is_named_too(tmp_path: Path) -> None:
    root = _quest(tmp_path, DELEGATING)
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert tr.trial_aggregates(source), "the reading of the code alone still names it"
    assert tr.trial_aggregates(source, root) == "", "but with a quest only a record decides, and there is none yet"
    assert tr.trial_needs_call_record(source, root)
    assert await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL),
                                     timeout_s=60) is False, "run_cell is the setting's number, random or not"
    assert "run_trial must run ONE trial" in tr.trial_aggregates(source, root), "the call record is what decides"
    assert not tr.trial_needs_call_record(source, root)
    assert tr.trial_needs_call_record(source + "# changed\n", root), "another text of the file is called again"
    record = json.loads((root / tr.ONE_TRIAL_RECORD).read_text(encoding="utf-8"))
    assert record["aggregates"] is True and record["run_cell_entered"] == [1]


@pytest.mark.asyncio
async def test_a_run_trial_that_returns_a_random_mean_is_still_named(tmp_path: Path) -> None:
    root = _quest(tmp_path, RANDOM_MEAN)
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert not await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL), timeout_s=60)
    assert "run_trial must run ONE trial" in tr.trial_aggregates(source, root)
    record = json.loads((root / tr.ONE_TRIAL_RECORD).read_text(encoding="utf-8"))
    assert record["aggregates"] is True and record["run_cell_entered"] == [1]


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
    (code / "simulate.py").write_text(ONE_TRIAL, encoding="utf-8")
    assert await engine._simulation_repair_fault(state, ONE_TRIAL) == "", "a run_trial that does not enter run_cell"
    (code / "simulate.py").write_text(DELEGATING, encoding="utf-8")
    assert "run_trial must run ONE trial" in await engine._simulation_repair_fault(state, DELEGATING)
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
    before = ONE_TRIAL
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


@pytest.mark.asyncio
async def test_an_oracle_repair_is_judged_by_what_the_imported_module_binds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A check with no case is answered by the module's own oracle(): whether the repair binds one is asked of the module,
    imported as FI imports it (an oracle defined in a branch is one), and a repair with none is still refused."""
    engine = _engine(tmp_path)
    engine._trial_mode = True
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    _gave(engine, "yield_rate")
    oracle = {"name": "zero", "kind": "special_case", "check": "c", "expected": 0.0, "tolerance": 0.1}
    in_a_branch = ONE_TRIAL + "\nif True:\n    def oracle():\n        return {'zero': 0.0}\n"
    replies = iter([in_a_branch, ONE_TRIAL])

    async def answer(prompt: str, **kw: Any) -> str:
        return json.dumps({"code": next(replies), "patch_summary": "the check's own function"})

    monkeypatch.setattr(engine, "_chat", answer)
    (code / "simulate.py").write_text(ONE_TRIAL, encoding="utf-8")
    new, failed, outcome = await engine._repair_script_for_oracle(
        {"design": {"protocol": PROTOCOL}}, code / "simulate.py", [oracle], ["zero is off"], passing=[])
    assert outcome == "applied" and new == in_a_branch, outcome
    # known positive: no oracle() in the module, as the import shows it, is refused, and not as a reading of the code
    new, failed, outcome = await engine._repair_script_for_oracle(
        {"design": {"protocol": PROTOCOL}}, code / "simulate.py", [oracle], ["zero is off"], passing=[])
    assert new is None and outcome == "not_usable"
    assert (code / "simulate.py").read_text(encoding="utf-8") == in_a_branch
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "defines no oracle()" in log and "read from the code" not in log


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
    assert tr.trial_aggregates(source), "the code's shape alone names it before anything is measured"
    assert tr.trial_aggregates(source, root) == "", "but it does not refuse"
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
    return {"y": model.value()}
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
    assert tr.trial_needs_call_record(source, root), "the file is the same, the package is not: called again"


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
async def test_a_call_that_does_not_finish_in_the_time_given_is_not_recorded(tmp_path: Path) -> None:
    root = _quest(tmp_path, SLOW_TRIAL)
    # the call takes ~2.5 s and is given 1 s
    done = await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL), timeout_s=1)
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
                            "def run_trial(cell, trial_id=0, seed=0):\n    return {'yield_rate': helper.value()}\n")
    (root / "code" / "helper.py").write_text("def value():\n    return 1.0\n", encoding="utf-8")
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL), timeout_s=60)
    assert tr.trial_aggregates(source, root) == ""
    (root / "code" / "helper.py").write_text("import random\n\ndef value():\n    return random.random()\n", encoding="utf-8")
    assert tr.trial_needs_call_record(source, root), "the helper changed: called again"


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


# ---- the call record decides, not two equal seeds and not the shape of the code (backlog -432, G13) -----------------------

# A discrete outcome that is the same for every seed (every draw is below 2), averaged over 20 draws: two seeds give equal
# numbers, yet the number is a mean over 20 draws.
MEAN_OF_DRAWS = '''
import random

def run_cell(cell):
    return {"yield_rate": sum(random.random() < 2 for _ in range(20)) / 20}

def run_trial(cell, trial_id=0, seed=0):
    random.seed(seed)
    return run_cell(cell)
'''

# The same, reached through a name FI's reading of the code does not follow.
MEAN_THROUGH_GETATTR = '''
import random, types

def run_cell(cell):
    rng = random.Random(7)
    return {"yield_rate": sum(rng.random() < 2 for _ in range(20)) / 20}

def _helper(cell):
    return getattr(types.SimpleNamespace(**globals()), "run_" + "cell")(cell)

def run_trial(cell, trial_id=0, seed=0):
    return _helper(cell)
'''

# One draw per call; run_cell is how a cell is built from trials.
ONE_DRAW = '''
import random

def run_trial(cell, trial_id=0, seed=0):
    return {"yield_rate": float(random.Random(seed).random() < 2)}

def run_cell(cell):
    return {"yield_rate": sum(run_trial(cell, i, i)["yield_rate"] for i in range(20)) / 20}
'''

NUMPY_MEAN = '''
import types
import numpy as np

def run_cell(cell):
    rng = np.random.default_rng(3)
    return {"yield_rate": float((rng.random(20) < 2).mean())}

def run_trial(cell, trial_id=0, seed=0):
    return getattr(types.SimpleNamespace(**globals()), "run_" + "cell")(cell)
'''

DECORATED_MEAN = '''
import functools, random

def _kept(fn):
    @functools.wraps(fn)
    def inner(*a, **k):
        return fn(*a, **k)
    return inner

@_kept
def run_cell(cell):
    rng = random.Random(11)
    return {"yield_rate": sum(rng.random() < 2 for _ in range(20)) / 20}

def run_trial(cell, trial_id=0, seed=0):
    return run_cell(cell)
'''


async def _settled(tmp_path: Path, source: str) -> bool | None:
    root = _quest(tmp_path, source)
    return await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL), timeout_s=60)


@pytest.mark.asyncio
async def test_a_mean_of_many_draws_is_named_although_two_seeds_give_the_same_number(tmp_path: Path) -> None:
    root = _quest(tmp_path, MEAN_OF_DRAWS)
    assert tr.trial_aggregates(MEAN_OF_DRAWS)  # the reading of the code names it too, as a first sight
    assert await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL),
                                     timeout_s=60) is False
    assert "run_trial must run ONE trial" in tr.trial_aggregates(MEAN_OF_DRAWS, root)


@pytest.mark.asyncio
async def test_a_mean_reached_through_a_lookup_by_name_is_named_by_the_record(tmp_path: Path) -> None:
    root = _quest(tmp_path / "a", MEAN_THROUGH_GETATTR)
    assert tr.trial_aggregates(MEAN_THROUGH_GETATTR, root) == "", "the reading of the code does not see it"
    assert tr.trial_needs_call_record(MEAN_THROUGH_GETATTR, root), "but FI has not called it yet"
    assert await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL),
                                     timeout_s=60) is False
    assert "run_trial must run ONE trial" in tr.trial_aggregates(MEAN_THROUGH_GETATTR, root)
    assert not tr.trial_needs_call_record(MEAN_THROUGH_GETATTR, root), "kept: not called again"
    # a generator of numpy and a run_cell kept inside a decorator are seen as well
    pytest.importorskip("numpy")
    assert await _settled(tmp_path / "b", NUMPY_MEAN) is False
    assert await _settled(tmp_path / "c", DECORATED_MEAN) is False


@pytest.mark.asyncio
async def test_one_draw_per_call_is_not_named_and_costs_one_call(tmp_path: Path) -> None:
    root = _quest(tmp_path, ONE_DRAW)
    assert tr.trial_aggregates(ONE_DRAW, root) == ""
    assert tr.trial_needs_call_record(ONE_DRAW, root)  # run_cell is named in the file: FI calls run_trial once to see
    assert await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL),
                                     timeout_s=60) is True
    record = json.loads((root / tr.ONE_TRIAL_RECORD).read_text(encoding="utf-8"))
    assert record["aggregates"] is False and record["run_cell_entered"] == [0]
    assert not tr.trial_needs_call_record(ONE_DRAW, root)
    # a file with no run_trial has nothing to call; any run_trial is called, whatever the file names (the call sees run_cell)
    assert not tr.trial_needs_call_record("def run_cell(cell):\n    return {}\n", root)
    lone = "def run_trial(cell, trial_id=0, seed=0):\n    return {'yield_rate': 1.0}\n"
    assert tr.trial_needs_call_record(lone, root)


@pytest.mark.asyncio
async def test_a_call_record_of_another_text_or_of_the_older_kind_is_not_one(tmp_path: Path) -> None:
    root = _quest(tmp_path, MEAN_OF_DRAWS)
    record = root / tr.ONE_TRIAL_RECORD
    record.parent.mkdir(parents=True)
    # the older record held two seeds' numbers ("same"): it decides nothing
    record.write_text(json.dumps({"sha256": tr.code_inputs_sha(root, MEAN_OF_DRAWS), "same": True, "differing": []}),
                      encoding="utf-8")
    assert tr.trial_needs_call_record(MEAN_OF_DRAWS, root)
    assert tr.trial_aggregates(MEAN_OF_DRAWS, root) == "", "the old record decides nothing, and neither does the reading"


@pytest.mark.asyncio
async def test_the_repair_check_refuses_a_mean_of_many_draws_through_the_real_harness(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    state: dict[str, Any] = {"design": {"protocol": PROTOCOL}}
    (code / "simulate.py").write_text(MEAN_THROUGH_GETATTR, encoding="utf-8")
    assert await engine._simulation_repair_fault(state, MEAN_THROUGH_GETATTR) == tr.TRIAL_AGGREGATE_WHY


# A mean over 20 draws from a generator of the study's own writing, which nothing FI could watch draws from.
OWN_GENERATOR_MEAN = '''
def _next(state):
    return (state * 1103515245 + 12345) % 2147483648

def run_cell(cell):
    state, total = 7, 0.0
    for _ in range(20):
        state = _next(state)
        total += state / 2147483648
    return {"yield_rate": total / 20}

def run_trial(cell, trial_id=0, seed=0):
    return run_cell(cell)
'''


@pytest.mark.asyncio
async def test_a_mean_over_a_random_source_of_the_studys_own_writing_is_named_too(tmp_path: Path) -> None:
    root = _quest(tmp_path, OWN_GENERATOR_MEAN)
    assert await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(CELL),
                                     timeout_s=60) is False
    record = json.loads((root / tr.ONE_TRIAL_RECORD).read_text(encoding="utf-8"))
    assert record["aggregates"] is True and record["run_cell_entered"] == [1]


@pytest.mark.asyncio
async def test_code_the_reading_did_not_name_goes_on_to_the_entry_check_when_the_call_cannot_be_recorded(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    called: list[str] = []

    async def entry(*a: Any, **k: Any) -> str:
        called.append("entry")
        return ""

    async def not_recorded(*a: Any, **k: Any) -> None:
        return None

    monkeypatch.setattr(estimate, "entry_returns_study", entry)
    monkeypatch.setattr(tr, "settle_aggregate", not_recorded)
    _gave(engine, "yield_rate")
    state: dict[str, Any] = {"design": {"protocol": PROTOCOL}}
    (code / "simulate.py").write_text(ONE_DRAW, encoding="utf-8")
    assert not tr.trial_aggregates(ONE_DRAW), "the reading names nothing in this file"
    assert await engine._simulation_repair_fault(state, ONE_DRAW) == ""
    assert called == ["entry"], "an unrecorded call of an unflagged file does not skip the study's entry check"


@pytest.mark.asyncio
async def test_with_no_first_setting_the_reading_of_the_code_does_not_refuse(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    monkeypatch.setattr(engine, "_first_study_cell", lambda protocol: None)  # the plan gives no first setting
    state: dict[str, Any] = {"design": {"protocol": PROTOCOL}}
    (code / "simulate.py").write_text(DELEGATING, encoding="utf-8")
    assert tr.trial_aggregates(DELEGATING), "the reading names it"
    assert await engine._simulation_repair_fault(state, DELEGATING) == ""
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "could not measure whether run_trial runs one trial" in log


# ---- what the call sees, without changing the study's functions (review of backlog -432, G13) ----------------------------

import hashlib  # noqa: E402

BRANCH_BY_MODE = '''
def run_cell(cell):
    return {"yield_rate": sum(range(10)) / 10}

def run_trial(cell, trial_id=0, seed=0):
    if cell["mode"] == "aggregate":
        return run_cell(cell)
    return {"yield_rate": float(seed % 2)}
'''

IMPORTED_ALIAS = '''
from model import run_cell as rc

def run_trial(cell, trial_id=0, seed=0):
    return rc(cell)
'''

PRE_BOUND_DECORATED = '''
def decorate(fn):
    def wrapped(*a, **k):
        return fn(*a, **k)
    return wrapped

@decorate
def run_cell(cell):
    return {"yield_rate": 1.0}

rc = run_cell

def run_trial(cell, trial_id=0, seed=0):
    return rc(cell)
'''

PARTIAL_AND_STATIC = '''
import functools

class Sim:
    @staticmethod
    def run_cell(cell):
        return {"yield_rate": 1.0}

_bound = functools.partial(Sim.run_cell)

def run_trial(cell, trial_id=0, seed=0):
    return _bound(cell)
'''

CLASS_METHOD = '''
class Sim:
    @classmethod
    def run_cell(cls, cell):
        return {"yield_rate": 1.0}

def run_trial(cell, trial_id=0, seed=0):
    return Sim.run_cell(cell)
'''

SPAWNS = '''
import multiprocessing as mp

def run_cell(cell):
    return {"yield_rate": 1.0}

def run_trial(cell, trial_id=0, seed=0):
    with mp.Pool(1) as pool:
        return pool.apply(run_cell, (cell,))
'''

PYTHON_CHILD = '''
import subprocess, sys

def run_cell(cell):
    return {"yield_rate": 1.0}

def run_trial(cell, trial_id=0, seed=0):
    subprocess.run([sys.executable, "-c", "pass"], check=True)
    return {"yield_rate": 1.0}
'''

HELPER_PROGRAM = '''
import platform, subprocess

def run_cell(cell):
    return {"yield_rate": 1.0}

def run_trial(cell, trial_id=0, seed=0):
    platform.uname()
    subprocess.run("echo hi", shell=True, capture_output=True)
    return {"yield_rate": 2.0}
'''

READS_OWN_DEFAULTS = '''
def run_cell(cell, multiplier=2):
    return {"yield_rate": run_cell.__defaults__[0] * cell["width"]}

def run_trial(cell, trial_id=0, seed=0):
    return run_cell(cell)
'''

# A run_cell of another module that is not in the quest's code (a library's) is not this study's.
LIBRARY_RUN_CELL = '''
import json

def run_trial(cell, trial_id=0, seed=0):
    return {"yield_rate": float(len(json.dumps(cell)))}
'''


async def _probe(root: Path, *, more: list[dict[str, Any]] | None = None, cell: dict[str, Any] | None = None) -> bool | None:
    return await tr.settle_aggregate(VenvExecutor(), sys.executable, root, "code/simulate.py", cell=dict(cell or CELL),
                                     more=more, timeout_s=60)


def _record(root: Path) -> dict[str, Any]:
    return json.loads((root / tr.ONE_TRIAL_RECORD).read_text(encoding="utf-8"))


def test_the_probe_settings_cover_every_value_of_every_axis() -> None:
    cells = tr.probe_cells({"a": 1, "b": "x"}, {"a": [1, 2, 3], "b": ["x", "y"], "fixed": 7})
    assert cells[0] == {"a": 1, "b": "x"}
    assert {c["a"] for c in cells} == {1, 2, 3} and {c["b"] for c in cells} == {"x", "y"}
    assert len(cells) <= 3 and all(c["a"] in (1, 2, 3) for c in cells)
    assert tr.probe_cells({"a": 1}, {}) == [{"a": 1}]


def test_the_probe_settings_of_a_grid_with_sub_studies_are_settings_the_study_runs() -> None:
    from core import optimise_search as search

    grid = search.StudyGrid({"a": [1, 2, 3], "b": ["x", "y"], "c": [0, 1]},
                            [{"name": "p", "vary": ["a"], "hold": {"b": "x", "c": 0}},
                             {"name": "q", "vary": ["b", "c"], "hold": {"a": 2}}])
    real = [dict(zip(grid.keys(), (grid[k][i] for k, i in zip(grid.keys(), picked))))
            for picked in search.iter_grid_indexes(grid)]
    cells = tr.probe_cells({"a": 2, "b": "x", "c": 0}, grid)
    assert all(c in real for c in cells)
    assert {c["a"] for c in cells} == {1, 2, 3} and {c["b"] for c in cells} == {"x", "y"} and {c["c"] for c in cells} == {0, 1}


@pytest.mark.asyncio
async def test_a_branch_that_only_another_setting_takes_is_seen_by_probing_every_value(tmp_path: Path) -> None:
    cell = {"mode": "raw"}
    root = _quest(tmp_path / "one", BRANCH_BY_MODE)
    assert await _probe(root, cell=cell) is True, "one setting alone cannot see it"
    root = _quest(tmp_path / "all", BRANCH_BY_MODE)
    cells = tr.probe_cells(cell, {"mode": ["raw", "aggregate"]})
    assert await _probe(root, cell=cells[0], more=cells[1:]) is False
    assert _record(root)["run_cell_entered"] == [0, 1]
    # the record belongs to the settings it was made on
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert not tr.trial_needs_call_record(source, root, tr.probe_key(cells, None))
    assert tr.trial_needs_call_record(source, root, tr.probe_key(cells[:1], None))


@pytest.mark.asyncio
async def test_every_way_of_reaching_a_run_cell_of_the_quests_code_is_seen(tmp_path: Path) -> None:
    sources = {"imported": IMPORTED_ALIAS, "pre_bound_decorated": PRE_BOUND_DECORATED, "partial_static": PARTIAL_AND_STATIC,
               "class_method": CLASS_METHOD, "own_defaults": READS_OWN_DEFAULTS, "lookup": MEAN_THROUGH_GETATTR}
    for name, source in sources.items():
        root = _quest(tmp_path / name, source)
        if name == "imported":
            (root / "code" / "model.py").write_text("def run_cell(cell):\n    return {'yield_rate': 1.0}\n", encoding="utf-8")
        assert await _probe(root) is False, name
        assert _record(root)["aggregates"] is True, name


@pytest.mark.asyncio
async def test_a_run_cell_that_reads_its_own_defaults_still_runs_and_is_named(tmp_path: Path) -> None:
    root = _quest(tmp_path, READS_OWN_DEFAULTS)
    assert await _probe(root) is False
    assert _record(root)["run_cell_entered"] == [1] and _record(root)["status"] == "recorded", "the call was not altered"


@pytest.mark.asyncio
async def test_a_call_that_starts_a_child_process_cannot_be_told_and_counts_as_aggregating(tmp_path: Path) -> None:
    root = _quest(tmp_path, SPAWNS)
    assert await _probe(root) is False
    record = _record(root)
    assert record["aggregates"] is True and record["child_processes"] >= 1 and "child process" in record["why"]


@pytest.mark.asyncio
async def test_a_python_child_process_cannot_be_told_but_a_helper_program_is_not_one(tmp_path: Path) -> None:
    root = _quest(tmp_path / "py", PYTHON_CHILD)
    assert await _probe(root) is False
    assert _record(root)["child_processes"] >= 1
    root = _quest(tmp_path / "helper", HELPER_PROGRAM)
    assert await _probe(root) is True  # run_cell never entered; a non-Python helper holds no copy of the trial
    record = _record(root)
    assert record["child_processes"] == 0 and record["aggregates"] is False


def test_what_the_hook_counts_as_a_child_process_that_runs_python() -> None:
    space: dict[str, Any] = {"sys": sys, "_fi_load_study": lambda path: None, "cell_reads": lambda cell: {}}
    exec(compile(tr.AGGREGATE_SPY_SOURCE, "<spy>", "exec"), space)  # noqa: S102 -- FI's own harness text
    runs = space["_fi_runs_python"]
    assert runs("os.fork", ()) and runs("os.forkpty", ())
    assert runs("subprocess.Popen", (sys.executable, [sys.executable, "-c", "pass"], None, None))
    assert runs("subprocess.Popen", (None, r'C:\Program Files\Python312\python.exe -m tool', None, None))
    assert runs("_winapi.CreateProcess", (None, r'"C:\venv\Scripts\pythonw.exe" -c "import x"'))
    assert runs("os.system", ("python3 job.py",)) and runs("os.posix_spawn", ("/usr/bin/python3.12", ["x"], {}))
    assert not runs("subprocess.Popen", (r"C:\Windows\system32\cmd.exe", "cmd.exe /c ver", None, None))
    assert not runs("os.system", ("echo hi",)) and not runs("subprocess.Popen", (None, ["git", "status"], None, None))
    assert not runs("subprocess.Popen", (None, ["tool", "python_notes.txt"], None, None))


@pytest.mark.asyncio
async def test_a_run_cell_outside_the_quests_code_is_not_counted(tmp_path: Path) -> None:
    root = _quest(tmp_path, LIBRARY_RUN_CELL)
    assert await _probe(root) is True


@pytest.mark.asyncio
async def test_a_call_that_fails_is_not_kept_as_a_final_answer(tmp_path: Path) -> None:
    root = _quest(tmp_path, "def run_trial(cell, trial_id=0, seed=0):\n    raise RuntimeError('boom')\n")
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert await _probe(root) is None
    assert _record(root)["status"] == "unfinished"
    assert tr.trial_needs_call_record(source, root), "tried again next time"
    assert tr.trial_aggregates(source, root) == ""


@pytest.mark.asyncio
async def test_a_record_of_another_harness_or_version_is_not_a_record(tmp_path: Path) -> None:
    root = _quest(tmp_path, ONE_DRAW)
    source = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert await _probe(root) is True
    assert not tr.trial_needs_call_record(source, root)
    good = _record(root)
    path = root / tr.ONE_TRIAL_RECORD
    path.write_text(json.dumps({**good, "harness": hashlib.sha256(b"an older harness").hexdigest()}), encoding="utf-8")
    assert tr.trial_needs_call_record(source, root), "the harness changed: called again"
    path.write_text(json.dumps({**good, "schema": 1}), encoding="utf-8")
    assert tr.trial_needs_call_record(source, root), "another version of the record"
    path.write_text(json.dumps({k: v for k, v in good.items() if k not in ("schema", "harness")}), encoding="utf-8")
    assert tr.trial_needs_call_record(source, root), "a record with neither is unusable"
