"""The two checks the engine makes on the simulation (core/run_checks.py): a result typed into the code is sent back to be
computed and, when it stays, left out of the paper; the whole run is timed on a few real trials before it starts, made
smaller by the plan's model when it would not fit the time allowed, and stops the quest plainly when it still would not.
Fake model, toy simulations (a cooling cup, a damped spring), real subprocesses."""

from __future__ import annotations

import asyncio
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import plan, run_estimate, trial_runner
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.execution import SharedInterpreterExecutor
from tests.test_plan_optimisation import EXTRA, HEAT_SINK, _NO_OBJECTIONS, _engine as _plan_engine

# ---- a result typed into the code ---------------------------------------------------------------------------------------

TYPED_SIM = '''\
import random

def run_trial(cell, trial, seed):
    rng = random.Random(seed)
    temp = 90.0
    for _ in range(int(cell["n"])):
        temp += (20.0 - temp) * 0.01 + rng.gauss(0, 0.1)
    return {"final_temp": temp, "lid_on": 1.0}
'''
COMPUTED_SIM = TYPED_SIM.replace('"lid_on": 1.0', '"lid_on": float(temp < 80.0)')


class _Client:
    def __init__(self, replies: list[str], asked: list[str]) -> None:
        self.replies, self.asked = replies, asked
        self.last_usage = None
        self.last_model = "test-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.asked.append(messages[-1]["content"])
        return self.replies[min(len(self.asked), len(self.replies)) - 1]


def _typed_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> tuple[Engine, list[str]]:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    asked: list[str] = []
    eng = Engine(Config(
        topic="a neutral topic about a cooling cup", title="typed", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng._client = _Client(replies, asked)  # type: ignore[assignment]
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "simulate.py").write_text(TYPED_SIM, encoding="utf-8")
    return eng, asked


@pytest.mark.asyncio
async def test_a_typed_in_result_is_sent_back_and_a_repair_that_computes_it_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": COMPUTED_SIM, "patch_summary": "computed"})])
    assert await eng._results_from_computation({}) is True and len(asked) == 1
    assert "simulate.py line 8: `lid_on` is returned as 1.0, a number typed into the code" in asked[0]
    assert "compute it from the simulation" in asked[0]
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == COMPUTED_SIM
    assert eng._left_out_quantities() == set() and eng._run_checks_note() == ""
    assert not (eng.quest_root / "needs" / "TYPED_RESULTS_CHECK.json").exists()


@pytest.mark.asyncio
async def test_a_typed_in_result_that_stays_is_left_out_with_a_plain_note_and_is_not_asked_about_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": TYPED_SIM, "patch_summary": "no change"})])
    assert await eng._results_from_computation({}) is False
    assert len(asked) == 2, "asked twice, the simulation kept as written"
    assert eng._left_out_quantities() == {"lid_on"}
    record = json.loads((eng.quest_root / "needs" / "TYPED_RESULTS_CHECK.json").read_text(encoding="utf-8"))
    assert record["removed_quantities"] == ["lid_on"] and "typed into the code" in record["note"]
    note = eng._run_checks_note()
    assert "`lid_on` was left out" in note and "limitations" in note and "not computed by the simulation" in note
    log = (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "The paper does not use it and says so in its limitations" in log
    # The quantity is not in the result the analysis hands on.
    assert eng._without_typed_results({"final_temp": 1.0, "lid_on": 1.0, "rows": [{"lid_on": 1}]}) == {
        "final_temp": 1.0, "rows": [{}]}
    # A resume (the same simulation) asks no more, and still leaves it out.
    assert await eng._results_from_computation({}) is False and len(asked) == 2
    assert eng._left_out_quantities() == {"lid_on"}
    # Once the simulation computes it, the record goes.
    (eng.quest_root / "code" / "simulate.py").write_text(COMPUTED_SIM, encoding="utf-8")
    await eng._results_from_computation({})
    assert eng._left_out_quantities() == set() and eng._run_checks_note() == ""


@pytest.mark.asyncio
async def test_a_repair_that_is_not_the_same_simulation_is_not_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for reply in ("{}", json.dumps({"code": "print('hello')\n"}), json.dumps({"code": "def broken(:\n"}),
                  json.dumps({"code": TYPED_SIM})):
        eng, asked = _typed_engine(tmp_path / re.sub(r"\W", "", reply)[:8], monkeypatch, [reply])
        assert await eng._results_from_computation({}) is False
        assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == TYPED_SIM, reply


@pytest.mark.asyncio
async def test_a_simulation_with_nothing_typed_in_is_not_asked_about(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, asked = _typed_engine(tmp_path, monkeypatch, ["{}"])
    (eng.quest_root / "code" / "simulate.py").write_text(COMPUTED_SIM, encoding="utf-8")
    assert await eng._results_from_computation({}) is False and asked == []


@pytest.mark.asyncio
async def test_a_later_text_of_the_simulation_is_read_again_and_an_unchanged_one_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The text a gate's repair leaves: a literal the first reading never saw.
    eng, asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": TYPED_SIM, "patch_summary": "no change"})])
    sim = eng.quest_root / "code" / "simulate.py"
    sim.write_text(COMPUTED_SIM, encoding="utf-8")
    assert await eng._results_from_computation({}) is False and asked == [] and eng._left_out_quantities() == set()
    sim.write_text(TYPED_SIM, encoding="utf-8")  # an oracle repair (or an equation's) typed a result in
    await eng._results_from_computation({})
    assert len(asked) == 2 and eng._left_out_quantities() == {"lid_on"}, "caught, asked twice, then left out"
    # The same text again (the next entry of the node, a retry of the run): nothing is asked, the answer stands.
    await eng._results_from_computation({})
    await eng._results_from_computation({})
    assert len(asked) == 2 and eng._left_out_quantities() == {"lid_on"}
    assert (eng.quest_root / "needs" / "TYPED_RESULTS_CHECK.json").is_file()
    # Another text has its own budget.
    sim.write_text(TYPED_SIM + "\n# repaired elsewhere\n", encoding="utf-8")
    await eng._results_from_computation({})
    assert len(asked) == 4


def test_the_node_reads_the_simulation_again_after_each_gate() -> None:
    src = (Path(__file__).resolve().parent.parent / "core" / "engine.py").read_text(encoding="utf-8")
    node = src[src.index("async def _node_execute("):]
    node = node[:node.index("pilot_run and not self.config.execution.background_jobs")]
    assert "self._gate_pass(state, py, exec_env, seed_path)" in node, "the node makes the checks through one function"
    gates = src[src.index("async def _gate_pass("):src.index("async def _verify_gated_code(")]
    assert gates.count("self._results_from_computation(state)") == 2, "before the gates, and after the equation gate"
    assert "self._typed_results_stable(" in gates, "and until the text the oracle gate leaves has been read"


def _fake_gate(eng: Engine, texts: list[str | None]) -> list[int]:
    """An oracle gate whose repairs write the next text of the simulation (``None``: it leaves the text as it is)."""
    calls: list[int] = []

    async def gate(state: Any, py: Any, env: Any, seed_path: Path) -> str | None:
        calls.append(1)
        text = texts[min(len(calls), len(texts)) - 1]
        if text is None:
            return None
        seed_path.write_text(text, encoding="utf-8")
        return text

    eng._oracle_gate = gate  # type: ignore[method-assign]
    return calls


@pytest.mark.asyncio
async def test_a_second_oracle_repair_that_types_a_result_in_again_is_read_again_until_the_text_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": COMPUTED_SIM, "patch_summary": "computed"})])
    sim = eng.quest_root / "code" / "simulate.py"
    # The first gate left a typed-in result; its repair (the check's) was followed by a second gate repair that typed one in again.
    calls = _fake_gate(eng, [TYPED_SIM + "\n# the gate wrote this\n", None])
    code, stop = await eng._typed_results_stable({}, sys.executable, None, sim, None)
    assert stop is None and len(calls) == 2 and len(asked) == 2, "read, repaired, gated, read, repaired, gated, read"
    assert sim.read_text(encoding="utf-8") == COMPUTED_SIM and code == COMPUTED_SIM
    assert eng._left_out_quantities() == set() and not eng._typed_text_unread(sim)


@pytest.mark.asyncio
async def test_a_simulation_that_never_settles_is_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": COMPUTED_SIM, "patch_summary": "computed"})])
    sim = eng.quest_root / "code" / "simulate.py"
    n = {"i": 0}

    async def gate(state: Any, py: Any, env: Any, seed_path: Path) -> str | None:
        n["i"] += 1
        text = TYPED_SIM + f"\n# rewritten {n['i']}\n"
        seed_path.write_text(text, encoding="utf-8")
        return text

    eng._oracle_gate = gate  # type: ignore[method-assign]
    code, stop = await eng._typed_results_stable({}, sys.executable, None, sim, None)
    assert n["i"] == Engine.TYPED_GATE_ROUNDS and stop is not None
    text = stop["exec_result"]["not_run"]
    assert "the text FI would run was not checked for typed-in results" in text and stop["exec_give_up_reason"] == text
    assert eng._typed_text_unread(sim)
    # The repair step leaves it alone and the gate ends the quest without a second design or a paper.
    assert await eng._node_execute_reflect({"exec_result": stop["exec_result"], "exec_give_up_reason": text}) == {}
    verdict = eng._no_results_verdict({"exec_result": stop["exec_result"], "result_json": {}, "iteration": 0})
    assert verdict is not None and verdict["stuck"] is True and verdict["stuck_reason"] == "not_run"
    stuck = await eng._node_stuck_no_findings({"evidence_assessment": {"stuck_reason": "not_run"},
                                               "exec_result": stop["exec_result"]})
    assert "No paper was written" in stuck["stuck"]["say"] and "not checked for typed-in results" in stuck["stuck"]["say"]


ANALYSIS = '''\
import json, os
data = json.load(open(os.environ["FI_TRIALS"]))
names = sorted({m for c in data["cells"] for m in c["metrics"]})
print("RESULT_JSON: " + json.dumps({"seen": names}))
'''


def test_the_analysis_is_not_given_a_left_out_quantity_but_the_ledger_keeps_every_value(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(TYPED_SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    cmd = [sys.executable, str(root / "code" / "experiment.py")]

    def run(leave_out: Any):  # noqa: ANN202
        runner = trial_runner.TrialsRunner(
            SharedInterpreterExecutor(python_version="3.11"), quest_root=root,
            protocol={"grid": {"n": [3]}, "runs_per_setting": 2}, deterministic=False,
            simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py", leave_out=leave_out)
        return asyncio.run(runner.execute(cmd, cwd=root, timeout_s=60, env={}))

    assert '"lid_on"' in run(None).stdout
    result = run(lambda: {"lid_on"})
    assert result.returncode == 0 and '"seen": ["final_temp"]' in result.stdout
    ledger = (root / "raw" / "ledger.jsonl").read_text(encoding="utf-8")
    assert "values_sha256" in ledger
    summary = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    assert "lid_on" in summary["cells"][0]["metrics"], "FI's own summary is not changed"


# ---- the run fits the time allowed -------------------------------------------------------------------------------------

SLOW_SIM = '''\
import time

def run_trial(cell, trial, seed):
    time.sleep(0.05 * cell["n"])
    return {"y": float(cell["n"]) * 2.0 + (seed % 3)}
'''
TOPIC = "Measure how a damped spring settles"


def _draft(grid: dict[str, list[int]], runs: int, **protocol: Any) -> dict[str, Any]:
    draft = copy.deepcopy(HEAT_SINK)
    draft["study_type"] = "measure"
    draft["protocol"] = {"oracles": copy.deepcopy(HEAT_SINK["protocol"]["oracles"]), "grid": grid,
                         "runs_per_setting": runs, **protocol}
    return draft


MARK = "would take longer than the time it is allowed"


class Model:
    """The plan's model: drafts the plan. A request to make the run smaller (MARK) is recorded: FI must make none."""

    def __init__(self, eng: Engine, draft: dict[str, Any]) -> None:
        self.eng, self.draft = eng, draft
        self.asked: list[str] = []
        self.drafted = False
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if MARK in prompt:
            self.asked.append(prompt)
            raise RuntimeError("FI must not ask the model to make the run smaller")
        if not self.drafted:
            self.drafted = True
            return json.dumps({**self.draft, "plan": EXTRA})
        return _NO_OBJECTIONS


class _Counting(SharedInterpreterExecutor):
    def __init__(self) -> None:
        super().__init__(python_version="3.11")
        self.calls = 0

    async def execute(self, cmd, **kw):  # noqa: ANN001, ANN003
        self.calls += 1
        return await super().execute(cmd, **kw)


async def _sizing(tmp_path: Path, *, timeout_s: int, grid: dict[str, list[int]] | None = None,
                  runs: int = 40, **protocol: Any) -> tuple[Engine, Model, Any, _Counting]:
    eng = _plan_engine(tmp_path, [])
    eng.config.execution.timeout_s = timeout_s
    model = Model(eng, _draft(grid or {"n": [1, 2, 4]}, runs, **protocol))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    code = eng.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text(SLOW_SIM, encoding="utf-8")
    (code / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    eng.executor = _Counting()  # type: ignore[assignment]
    state: dict[str, Any] = {"topic": TOPIC, "iteration": 0}
    runner = trial_runner.TrialsRunner(
        eng.executor, quest_root=eng.quest_root, protocol=lambda: eng._protocol_block(state) or {}, deterministic=False,
        simulate=code / "simulate.py", analysis=code / "experiment.py", log=eng._log)
    return eng, model, runner, eng.executor  # type: ignore[return-value]


def _log(eng: Engine) -> str:
    return (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_run_that_fits_goes_on_after_one_plain_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=3600)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert model.asked == []
    line = re.search(r"FI estimates the experiment takes about (.+?); the limit is 1 h", _log(eng))
    assert line, _log(eng)
    assert "FI estimates the experiment takes about" in capsys.readouterr().out
    assert not (eng.quest_root / "raw").exists(), "the timed trials are not results"
    record = run_estimate.read_record(eng.quest_root)
    assert record["fits"] is True and record["timed"] == 3 and eng._run_checks_note() == ""
    # A resume with the same simulation and plan does not time it again.
    before = ex.calls
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert ex.calls == before and "timed before" in _log(eng)
    # A changed simulation is timed again.
    (eng.quest_root / "code" / "simulate.py").write_text(SLOW_SIM + "\n# changed\n", encoding="utf-8")
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert ex.calls > before


@pytest.mark.asyncio
async def test_a_run_that_is_too_long_has_its_runs_lowered_by_fi_without_asking_and_then_goes_on(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=8, min_runs_per_setting=8)
    before = copy.deepcopy(plan.load_design(eng.quest_root)[0]["protocol"]["oracles"])
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert model.asked == [], "no request to the plan's model"
    protocol = plan.load_design(eng.quest_root)[0]["protocol"]
    assert protocol["grid"] == {"n": [1, 2, 4]} and 8 <= protocol["runs_per_setting"] < 40, "every setting is kept"
    assert protocol["oracles"] == before
    assert eng._design_after_sizing({"design": {"hypothesis": "h"}})["protocol"]["runs_per_setting"] == protocol["runs_per_setting"]
    log = _log(eng)
    assert "FI made the experiment smaller" in log and "the checks, thresholds and tolerances are unchanged" in log
    assert re.search(r"FI estimates the experiment takes about .+; the limit is 8 seconds", log)
    assert "FI's facts about this run" in eng._run_checks_note()
    assert run_estimate.read_record(eng.quest_root)["fits"] is True
    assert not (eng.quest_root / "raw").exists()


@pytest.mark.asyncio
async def test_a_run_that_cannot_fit_at_the_fewest_runs_stops_plainly_and_a_resume_times_nothing_again(
        tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=6)  # the plan states no fewest runs
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and model.asked == []
    text = patch["exec_result"]["too_long"]
    assert re.fullmatch(r"the experiment.s \d+ settings would take about .+, more than the 6 seconds allowed; FI does not run a smaller study "
                        r"than the plan's settings, and they cannot fit even with the fewest runs per setting the plan allows",
                        text), text
    assert patch["exec_give_up_reason"] == text and patch["result_json"] == {} and patch["exec_result"]["returncode"] == 1
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == 40
    # The repair step does not try to fix a script, the quest goes from the run step straight to the stop (no analysis,
    # no second design), and the stop says it plainly.
    assert await eng._node_execute_reflect({"exec_result": patch["exec_result"],
                                            "exec_give_up_reason": text}) == {}
    assert eng._route_after_execute_reflect({"exec_result": patch["exec_result"], "exec_give_up_reason": text}) == "stuck"
    verdict = eng._no_results_verdict({"exec_result": patch["exec_result"], "result_json": {},
                                       "exec_give_up_reason": text, "iteration": 0})
    assert verdict is not None and verdict["stuck"] is True and verdict["stuck_reason"] == "too_long"
    stuck = await eng._node_stuck_no_findings({"evidence_assessment": {"stuck_reason": "too_long"},
                                               "exec_result": patch["exec_result"]})
    assert "more than the 6 seconds allowed" in stuck["stuck"]["say"] and "does not run a smaller study" in stuck["stuck"]["say"]
    assert "No paper was written" in stuck["stuck"]["say"]
    assert (eng.quest_root / "needs" / "STUCK.json").is_file() and not (eng.quest_root / "paper.md").exists()
    calls = ex.calls
    again = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert again is not None and again["exec_result"]["too_long"] == text and ex.calls == calls


@pytest.mark.asyncio
async def test_a_resume_after_the_runs_were_lowered_does_not_lower_them_a_second_time(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=6, min_runs_per_setting=2)
    run_estimate.write_record(eng.quest_root, {"fits": False, "reduced": {"note": "FI lowered the runs per setting"}})
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and model.asked == []
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == 40, "it was not cut again"


@pytest.mark.asyncio
async def test_nothing_is_timed_for_a_later_pass_or_a_cluster_job(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=8)
    assert await eng._size_the_run({"iteration": 1}, runner, sys.executable, None) is None and ex.calls == 0
    eng.config.execution.background_jobs = True
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None and ex.calls == 0


FAILS_AT_END = '''\
import time

def run_trial(cell, trial, seed):
    if cell["n"] == 4:
        raise ValueError("the grid cannot take this size")
    return {"y": float(cell["n"])}
'''
FAILS_TOGETHER = '''\
import time

def run_cell(cell):
    time.sleep(0.05 * max(cell["a"], cell["b"]))
    if cell["a"] == 8 and cell["b"] == 8:
        raise ValueError("both too large")
    return {"y": float(cell["a"])}
'''


@pytest.mark.asyncio
async def test_a_setting_that_fails_when_timed_goes_to_the_repair_of_the_simulation_naming_it(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=3600)
    (eng.quest_root / "code" / "simulate.py").write_text(FAILS_AT_END, encoding="utf-8")
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and patch["exec_result"]["returncode"] == 1
    assert patch["exec_result"]["failed_script"] == "simulate.py" and "exec_give_up_reason" not in patch
    said = patch["exec_result"]["stderr_tail"]
    assert "at n=4: a trial failed: ValueError: the grid cannot take this size" in said and "expected" not in said.lower()
    assert model.asked == [], "the plan is not asked; the simulation is repaired"
    assert eng._route_after_execute_reflect({"exec_result": patch["exec_result"], "exec_reflect_iter": 0}) == "retry"


@pytest.mark.asyncio
async def test_a_corner_that_fails_is_a_failing_setting_too(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=3600, grid={"a": [1, 2, 8], "b": [1, 2, 8]}, runs=1)
    (eng.quest_root / "code" / "simulate.py").write_text(FAILS_TOGETHER, encoding="utf-8")
    runner.deterministic = True
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None
    assert "at a=8,b=8: a trial failed: ValueError: both too large" in patch["exec_result"]["stderr_tail"]
    assert patch["exec_result"]["failed_script"] == "simulate.py"


# ---- "N of M settings done, about T left" -------------------------------------------------------------------------------

PROGRESS_SIM = '''\
import time

def run_trial(cell, trial, seed):
    time.sleep(0.15)
    return {"y": float(cell["n"])}
'''


@pytest.mark.asyncio
async def test_the_still_running_line_says_how_many_settings_are_done_and_how_long_is_left(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(PROGRESS_SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    runner = trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version="3.11"), quest_root=root,
        protocol={"grid": {"n": [1, 2, 3, 4]}, "runs_per_setting": 3}, deterministic=False,
        simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")
    assert runner.progress_text() == "", "nothing to say before the settings are run"
    seen: list[str] = []
    task = asyncio.ensure_future(runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root,
                                                timeout_s=120, env={}))
    while not task.done():
        text = runner.progress_text()
        if text and (not seen or seen[-1] != text):
            seen.append(text)
        await asyncio.sleep(0.05)
    await task
    assert seen and all(re.fullmatch(r"\d of 4 settings done(, about .+ left)?", t) for t in seen), seen
    assert any("about" in t for t in seen), "once a trial has finished, a time left is given"
    assert runner.progress_text() == "", "and nothing once the settings are done"


@pytest.mark.asyncio
async def test_the_heartbeat_puts_that_line_into_the_log(tmp_path: Path) -> None:
    eng = _plan_engine(tmp_path, [])

    async def work() -> int:
        await asyncio.sleep(0.25)
        return 7

    out = await eng._await_with_heartbeat(work(), label="running experiment.py", interval_s=0.05,
                                          progress=lambda: "3 of 96 settings done, about 2 h left")
    assert out == 7
    assert "running experiment.py — still running, 1 second elapsed: 3 of 96 settings done, about 2 h left" in _log(eng)
    # Without a line to add, the heartbeat is the one it always was.
    await eng._await_with_heartbeat(work(), label="running x.py", interval_s=0.05)
    assert re.search(r"running x\.py — still running, \d+s elapsed", _log(eng))


AGGREGATING_SIM = '''\
def run_trial(cell, trial, seed):
    return run_cell(cell)

def run_cell(cell):
    return {"y": sum(float(cell["n"]) * i for i in range(1000)) / 1000}
'''


@pytest.mark.asyncio
async def test_a_run_trial_that_is_a_cell_aggregate_goes_back_to_the_code_not_to_the_plan(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=3600)
    (eng.quest_root / "code" / "simulate.py").write_text(AGGREGATING_SIM, encoding="utf-8")
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and patch["exec_result"]["failed_script"] == "simulate.py"
    assert "run_trial must run ONE trial" in patch["exec_result"]["stderr_tail"]
    assert model.asked == [] and ex.calls == 0, "the plan is not asked to shrink the grid and nothing is timed"
    assert "too_long" not in patch["exec_result"]
    assert "run_trial must run ONE trial" in _log(eng)


@pytest.mark.asyncio
async def test_an_honest_run_trial_that_calls_a_method_named_run_cell_is_not_sent_back(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=3600)
    honest = SLOW_SIM.replace("    return {", "    sim_cell = type('S', (), {'run_cell': staticmethod(lambda s: s)})\n"
                                              "    sim_cell.run_cell(seed)\n    return {")
    (eng.quest_root / "code" / "simulate.py").write_text(honest + "\ndef run_cell(cell):\n    return {'y': 1.0}\n",
                                                         encoding="utf-8")
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert ex.calls > 0, "the experiment was timed as usual"


# ---- a smaller run keeps every setting ---------------------------------------------------------------------------

FDR_SIM = '''\
import time

def run_trial(cell, trial, seed):
    time.sleep(0.04)
    return {"y": float(cell["m"]) * 0.001 + (seed % 3)}
'''
FDR_GRID = {"m": [20, 200, 2000], "pi1": [0.0, 0.05, 0.2], "procedure": ["uncorrected", "bonferroni", "Benjamini-Hochberg"]}


async def _fdr_sizing(tmp_path: Path, *, runs: int = 1600, **protocol: Any) -> tuple[Engine, Model, Any]:
    eng, model, runner, _ex = await _sizing(tmp_path, timeout_s=900, grid=FDR_GRID, runs=runs, **protocol)
    (eng.quest_root / "code" / "simulate.py").write_text(FDR_SIM, encoding="utf-8")
    return eng, model, runner


@pytest.mark.asyncio
async def test_a_run_that_cannot_fit_at_the_plans_fewest_runs_stops_rather_than_run_a_smaller_study(tmp_path: Path) -> None:
    # The real numbers: 27 settings x 1600 runs, about 43 minutes (28 in the real run) against 15 minutes allowed, and the
    # plan's fewest runs (its precision) is 1537: no model is asked and no setting is taken out.
    eng, model, runner = await _fdr_sizing(tmp_path, min_runs_per_setting=1537)
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and model.asked == []
    design = plan.load_design(eng.quest_root)[0]
    assert design["protocol"]["grid"] == FDR_GRID and design["protocol"]["runs_per_setting"] == 1600
    text = patch["exec_result"]["too_long"]
    assert "does not run a smaller study than the plan's settings" in text and "more than the 15 minutes allowed" in text
    assert run_estimate.read_record(eng.quest_root).get("stopped") == text


@pytest.mark.asyncio
async def test_fi_lowers_the_runs_itself_down_to_the_fewest_the_plan_states_without_asking(tmp_path: Path) -> None:
    eng, model, runner = await _fdr_sizing(tmp_path, min_runs_per_setting=200)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert model.asked == [], "no request to the plan's model"
    design = plan.load_design(eng.quest_root)[0]
    runs = design["protocol"]["runs_per_setting"]
    assert design["protocol"]["grid"] == FDR_GRID and 200 <= runs < 1600
    assert eng._design_after_sizing({"design": {"hypothesis": "h"}})["protocol"]["runs_per_setting"] == runs
    assert f"FI lowered the runs per setting from 1600 to {runs} and kept all 27 setting(s)" in eng._run_checks_note()


_K_GRID = {"k": list(range(27))}
# 27 settings, each 1 s to start its process and 0.02 s per run: at 1600 runs, 27 x 33 = 891 s (1336.5 s with the margin).
_K_PROBES = [run_estimate.Probe(None, None, {"k": 13}, 1.0 + 1600 * 0.02, fixed_s=1.0, per_trial_s=0.02)]


def _fit(protocol: dict[str, Any], limit_s: float, topic: bool = False, deterministic: bool = False) -> Any:
    return run_estimate.fit_runs(_K_GRID, 1600, deterministic, _K_PROBES, protocol, limit_s=limit_s, topic_limit=topic)


def test_the_runs_that_fit_need_a_fewest_the_plan_itself_states() -> None:
    assert _fit({}, 900.0) is None, "FI invents no number of runs"
    fit = _fit({"min_runs_per_setting": 100}, 900.0)
    # The most runs whose estimate with its margin fits: 27 x (1 + 0.02 n) x 1.5 <= 900 -> n = 1061 (the process starts do
    # not shrink with the runs, so this is fewer than 1600 x 900 / 1336.5 = 1077).
    assert fit is not None and fit.runs == 1061 and not fit.below_plan and fit.plan_floor == 100
    assert _fit({"min_runs_per_setting": 1537}, 900.0) is None
    assert _fit({"min_runs_per_setting": 1}, 900.0, deterministic=True) is None


def test_the_topics_own_limit_may_take_the_runs_below_the_plans_minimum_and_says_so() -> None:
    fit = _fit({"min_runs_per_setting": 1000}, 60.0, topic=True)
    # 27 x (1 + 0.02 n) x 1.5 <= 60 -> n = 24, below the plan's 1000: allowed for the topic's limit, and marked.
    assert fit is not None and fit.runs == 24 and fit.below_plan and fit.plan_floor == 1000
    assert _fit({}, 60.0, topic=True).runs == 24, "the topic's limit needs no stated minimum"
    # Even 10 runs per setting do not fit 30 s: the process starts alone are 27 s (40.5 s with the margin).
    assert _fit({"min_runs_per_setting": 1000}, 30.0, topic=True) is None
    # The same limit, not the topic's: never below the plan's minimum.
    assert _fit({"min_runs_per_setting": 1000}, 60.0) is None


def test_a_probe_without_its_parts_follows_the_runs_in_proportion() -> None:
    probe = run_estimate.Probe(None, None, {"k": 1}, 100.0)
    assert probe.at(50, 100).seconds == 50.0
    split = run_estimate.Probe(None, None, {"k": 1}, 21.0, fixed_s=1.0, per_trial_s=0.2)
    assert abs(split.at(50, 100).seconds - 11.0) < 1e-9


def test_the_precision_a_smaller_run_gives_is_stated() -> None:
    plan = {"metrics": [{"id": "hit", "kind": "proportion"}], "precision": {"target_half_width": 0.03, "metric": "hit"}}
    got = run_estimate.expected_precision(plan, 24)
    assert got["metric"] == "hit" and got["half_width"] == 0.2 and got["target"] == 0.03 and "near 0.5" in got["how"]
    mean = {"metrics": [{"id": "y", "kind": "mean"}], "precision": {"target_half_width": 0.1, "metric": "y"}}
    assert run_estimate.expected_precision(mean, 100, {"y": 2.0})["half_width"] == 0.392
    assert run_estimate.expected_precision(mean, 100, {}) is None, "a mean's spread is not known: nothing is claimed"
    assert run_estimate.expected_precision({"metrics": [{"id": "hit", "kind": "proportion"}]}, 100)["target"] is None


# ---- the coarse scan of a search is counted before it is listed ---------------------------------------------------------

_SCAN_BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"}],
    "fixed": {f"axis{i}": 0 for i in range(13)}, "baseline": {"values": {"x": 0.0}, "source": "the design in use"},
    "numerical_settings": {"mesh": {"search": 0.5}},
    "evaluation_budget": {"starts": 2, "per_start": 10},
    "grid": {f"axis{i}": list(range(n)) for i, n in enumerate((5, 33, 10, 5, 6, 6, 3, 4, 5, 3, 18, 10, 2))},
}


async def _search_engine(tmp_path: Path, block: dict[str, Any]) -> tuple[Engine, Any]:
    from core import optimise

    eng = _plan_engine(tmp_path, [])
    eng.config.execution.timeout_s = 3600
    code = eng.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text("def run_cell(cell):\n    return {'f': cell['x'] ** 2}\n", encoding="utf-8")
    (code / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    eng.executor = _Counting()  # type: ignore[assignment]
    protocol = {"optimisation": block, "thresholds": {}}
    eng._protocol_block = lambda state: protocol  # type: ignore[method-assign]
    runner = optimise.OptimisationRunner(
        eng.executor, quest_root=eng.quest_root, protocol=lambda: protocol, simulate=code / "simulate.py",
        analysis=code / "experiment.py", log=eng._log)
    return eng, runner


def _nothing_is_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("the settings were listed before the run-size check stopped the run")

    monkeypatch.setattr(trial_runner, "cells", boom)
    monkeypatch.setattr("core.optimise_search.grid_cells", boom)


@pytest.mark.asyncio
async def test_a_scan_of_billions_of_settings_stops_before_any_is_listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block = copy.deepcopy(_SCAN_BLOCK)
    block["evaluation_budget"]["seconds_per_evaluation"] = 20
    eng, runner = await _search_engine(tmp_path, block)
    _nothing_is_listed(monkeypatch)
    started = time.monotonic()
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert time.monotonic() - started < 5
    assert patch is not None and patch["exec_result"]["returncode"] == 1
    text = patch["exec_result"]["too_long"]
    assert re.fullmatch(r"the plan asks for 19,245,600,000 settings .+; FI does not list or run more than 500,000 settings\. "
                        r"Fewer settings or fewer values for each must be planned; FI does not choose which to leave out\.",
                        text), text
    assert eng.executor.calls == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_scan_with_no_stated_time_is_timed_on_one_real_evaluation_and_stops_before_listing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block = copy.deepcopy(_SCAN_BLOCK)
    block["grid"] = {"axis0": list(range(5)), "axis1": list(range(33)), "axis2": list(range(10)), "axis3": list(range(5))}
    eng, runner = await _search_engine(tmp_path, block)
    eng.config.execution.timeout_s = 1
    _nothing_is_listed(monkeypatch)
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and "8,250 settings" in patch["exec_result"]["too_long"]
    assert eng.executor.calls >= 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_small_scan_that_fits_goes_on(tmp_path: Path) -> None:
    block = copy.deepcopy(_SCAN_BLOCK)
    block["grid"] = {"axis0": [0, 1, 2]}
    block["evaluation_budget"]["seconds_per_evaluation"] = 20
    eng, runner = await _search_engine(tmp_path, block)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None


@pytest.mark.asyncio
async def test_a_frozen_protocol_still_gets_the_scan_check_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import frozen_protocol

    block = copy.deepcopy(_SCAN_BLOCK)
    block["evaluation_budget"]["seconds_per_evaluation"] = 20
    eng, runner = await _search_engine(tmp_path, block)
    frozen_protocol.freeze(eng.quest_root, {"optimisation": block, "thresholds": {}}, approved_by="engine", source="test")
    assert frozen_protocol.load(eng.quest_root) is not None
    _nothing_is_listed(monkeypatch)
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and "19,245,600,000 settings" in patch["exec_result"]["too_long"]
    eng.config.execution.background_jobs = True
    patch = await eng._size_the_run({"iteration": 1}, runner, sys.executable, None)
    assert patch is not None and "19,245,600,000 settings" in patch["exec_result"]["too_long"]


@pytest.mark.asyncio
async def test_a_tiny_claimed_time_with_a_huge_grid_still_stops_without_listing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block = copy.deepcopy(_SCAN_BLOCK)
    block["evaluation_budget"]["seconds_per_evaluation"] = 1e-9
    eng, runner = await _search_engine(tmp_path, block)
    _nothing_is_listed(monkeypatch)
    started = time.monotonic()
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert time.monotonic() - started < 5
    assert patch is not None and "19,245,600,000 settings" in patch["exec_result"]["too_long"]
    assert "FI does not list or run more than" in patch["exec_result"]["too_long"]


_MID_GRID = {"axis0": list(range(5)), "axis1": list(range(33)), "axis2": list(range(10)), "axis3": list(range(5))}


@pytest.mark.asyncio
async def test_a_tiny_claimed_time_is_raised_to_the_measured_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under the ceiling, the plan's claim alone does not decide: one real evaluation of the baseline is timed too."""
    block = copy.deepcopy(_SCAN_BLOCK)
    block["grid"] = dict(_MID_GRID)
    block["evaluation_budget"]["seconds_per_evaluation"] = 1e-9
    eng, runner = await _search_engine(tmp_path, block)

    async def slow(*a: Any, **k: Any) -> Any:
        return run_estimate.Probe(None, None, {}, 60.0), ""

    monkeypatch.setattr(run_estimate, "_time_one", slow)
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and "8,250 settings" in patch["exec_result"]["too_long"]


@pytest.mark.asyncio
async def test_a_baseline_that_cannot_be_timed_keeps_the_plans_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block = copy.deepcopy(_SCAN_BLOCK)
    block["grid"] = dict(_MID_GRID)
    block["evaluation_budget"]["seconds_per_evaluation"] = 20
    eng, runner = await _search_engine(tmp_path, block)

    async def broken(*a: Any, **k: Any) -> Any:
        return None, "a trial failed: boom"

    monkeypatch.setattr(run_estimate, "_time_one", broken)
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and "8,250 settings" in patch["exec_result"]["too_long"]
    block["evaluation_budget"].pop("seconds_per_evaluation")
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None  # nothing to judge by


@pytest.mark.asyncio
async def test_the_scan_probe_gets_the_allowance_of_a_measurement_not_the_whole_limit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block = copy.deepcopy(_SCAN_BLOCK)
    block["grid"] = {"axis0": list(range(5)), "axis1": list(range(33))}
    eng, runner = await _search_engine(tmp_path, block)
    seen: list[int] = []

    async def fast(*a: Any, **k: Any) -> Any:
        seen.append(k["timeout_s"])
        return run_estimate.Probe(None, None, {}, 0.01), ""

    monkeypatch.setattr(run_estimate, "_time_one", fast)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert seen and seen[0] < 3600  # max(60, 3 * 3600 / 165) -> 65 s


def test_a_grid_too_large_to_list_ends_the_execute_step_in_the_plain_stop(tmp_path: Path) -> None:
    """The engine turns the listing's own refusal into the quest's stop (no crash, no smaller study)."""
    from core import optimise_search

    eng = _plan_engine(tmp_path, [])
    exc = optimise_search.TooManySettings("the plan asks for 9 settings")
    assert isinstance(exc, ValueError)
    patch = eng._stop_too_long(str(exc))
    assert patch["exec_result"]["too_long"] == "the plan asks for 9 settings" and patch["exec_give_up_reason"]


_TRIALS_CRITERION = {"criteria": [{"name": "se rate", "trials": "x", "direction": "target", "target": 0.5, "tolerance": 0.1},
                                  {"name": "se rate 2", "trials": "y", "direction": "target", "target": 0.5, "tolerance": 0.1}]}


def test_the_topics_cut_never_goes_below_what_the_plans_checks_need_nor_below_ten() -> None:
    # 27 settings: the standard-error criterion needs 32 trials per setting (and 256 in all), so 24 is not allowed
    assert _fit({}, 60.0, topic=True).runs == 24
    assert _fit(_TRIALS_CRITERION, 60.0, topic=True) is None
    assert _fit(_TRIALS_CRITERION, 80.0, topic=True).runs == 48
    assert run_estimate.runs_checks_need(_TRIALS_CRITERION, _K_GRID) == 32
    assert run_estimate.runs_checks_need({}, _K_GRID) == 0
    # a plan whose own minimum is 5: the topic's limit still goes no lower than 10 (7 would fit 47 s)
    assert _fit({"min_runs_per_setting": 5}, 47.0, topic=True) is None
    # FI's own limit keeps the plan's minimum, and what the checks need
    assert _fit({**_TRIALS_CRITERION, "min_runs_per_setting": 5}, 60.0) is None


@pytest.mark.asyncio
async def test_settings_run_at_once_make_a_run_fit_that_one_after_another_would_not(tmp_path: Path) -> None:
    """The estimate uses the same number at once as the run: the settings' work spread over them, each as much slower
    as FI measured on this machine, and never less than the costliest setting."""
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=40, grid={"n": [1, 2, 4], "m": [1, 2, 3, 4]})
    serial_key = eng._sizing_key(runner, {"n": [1, 2, 4], "m": [1, 2, 3, 4]}, 40, 40)
    runner.at_once = 6
    assert eng._sizing_key(runner, {"n": [1, 2, 4], "m": [1, 2, 3, 4]}, 40, 40) == serial_key + "|6"
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    log = _log(eng)
    assert re.search(r"with 6 settings running at the same time, each took [\d.]+ times as long as alone \(timed on a setting that took .+ alone\)", log), log
    line = re.search(r"FI estimates the experiment takes about .+; the limit is 40 seconds \(6 settings run at the same "
                     r"time.*; one after another they would take (.+?), and the run takes at least as long as its "
                     r"costliest setting\)", log)
    assert line, log
    record = run_estimate.read_record(eng.quest_root)
    assert record["fits"] is True and model.asked == []
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == 40, "the runs were not lowered"
    assert runner.cost is not None and runner.cost({"n": 4, "m": 1}) > runner.cost({"n": 1, "m": 1}), \
        "the run starts the costliest settings first"


@pytest.mark.asyncio
async def test_the_slowdown_is_measured_on_the_costliest_setting_that_finished_and_fits_the_time_left(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=3600)
    eng._pilot_spent = 0.0
    timed: list[dict[str, Any]] = []

    async def fake(executor, python, quest_root, module, cell, **kw):  # noqa: ANN001, ANN003
        timed.append(cell)
        return 2.5, ""

    monkeypatch.setattr(run_estimate, "slowdown_at_once", fake)
    probes = [run_estimate.Probe(None, None, {"n": 2}, 10.0),
              run_estimate.Probe("n", 0, {"n": 1}, 1.0),
              run_estimate.Probe("n", 2, {"n": 4}, 500.0, at_least=True),  # did not finish
              run_estimate.Probe(run_estimate.CORNER, None, {"n": 3}, 100.0)]  # finished, too long to time again
    factor = await eng._slowdown_at_once(runner, sys.executable, None, probes, 40, {}, None, 4, budget_s=300.0)
    assert factor == 2.5 and timed == [{"n": 2}], "the costliest that finished and can be timed in the time left"
    timed.clear()
    assert await eng._slowdown_at_once(runner, sys.executable, None, probes[2:3], 40, {}, None, 4, 300.0) is None
    assert timed == [] and "no timed setting finished" in _log(eng)
