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
    assert node.count("self._results_from_computation(state)") == 3, "before the gates, after the equation gate, after the oracle gate"


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
    """The plan's model: drafts the plan, and answers the request to make the run smaller with the next answer."""

    def __init__(self, eng: Engine, draft: dict[str, Any], answers: list[Any]) -> None:
        self.eng, self.draft, self.answers = eng, draft, list(answers)
        self.asked: list[str] = []
        self.drafted = False
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if MARK in prompt:
            self.asked.append(prompt)
            answer = self.answers.pop(0) if self.answers else None
            if answer is None:
                raise RuntimeError("no answer")
            return answer(plan.plan_path(self.eng.quest_root).read_text(encoding="utf-8"))
        if not self.drafted:
            self.drafted = True
            return json.dumps({**self.draft, "plan": EXTRA})
        return _NO_OBJECTIONS


def _smaller(grid: dict[str, list[int]], runs: int, **tamper: Any) -> Any:
    def answer(text: str) -> str:
        out = plan.edit_design_block(text, lambda d: {**d, "protocol": {**d["protocol"], "grid": grid,
                                                                       "runs_per_setting": runs, **tamper}})
        assert out is not None
        return out
    return answer


class _Counting(SharedInterpreterExecutor):
    def __init__(self) -> None:
        super().__init__(python_version="3.11")
        self.calls = 0

    async def execute(self, cmd, **kw):  # noqa: ANN001, ANN003
        self.calls += 1
        return await super().execute(cmd, **kw)


async def _sizing(tmp_path: Path, *, timeout_s: int, answers: list[Any], grid: dict[str, list[int]] | None = None,
                  runs: int = 40, **protocol: Any) -> tuple[Engine, Model, Any, _Counting]:
    eng = _plan_engine(tmp_path, [])
    eng.config.execution.timeout_s = timeout_s
    model = Model(eng, _draft(grid or {"n": [1, 2, 4]}, runs, **protocol), answers)
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
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=3600, answers=[])
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
async def test_a_run_that_is_too_long_is_made_smaller_by_the_plans_model_and_then_goes_on(tmp_path: Path) -> None:
    tolerance_tamper = {"oracles": [{"name": "baseline_energy_balance", "kind": "invariant", "check": "x", "expected": 0,
                                     "tolerance": 5.0, "reference": "derivation: steady state"}]}
    eng, model, runner, ex = await _sizing(
        tmp_path, timeout_s=8, answers=[_smaller({"n": [1, 4]}, 4, **tolerance_tamper)])
    before = copy.deepcopy(plan.load_design(eng.quest_root)[0]["protocol"]["oracles"])
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert len(model.asked) == 1
    ask = model.asked[0]
    assert "3 settings, each setting 40 runs" in ask
    assert "`n` = 4 costs" in ask and "protocol.runs_per_setting" in ask and "protocol.grid" in ask
    asked_part = ask[ask.index(MARK) - 80:]
    assert "not a check, a threshold, a tolerance" in asked_part
    design = plan.load_design(eng.quest_root)[0]
    assert design["protocol"]["grid"] == {"n": [1, 4]} and design["protocol"]["runs_per_setting"] == 4
    assert design["protocol"]["oracles"] == before, "a check the model also changed is put back"
    assert eng._draft_protocol({"iteration": 0})["grid"] == {"n": [1, 4]}
    assert eng._design_after_sizing({"design": {"hypothesis": "h"}})["protocol"]["runs_per_setting"] == 4
    log = _log(eng)
    assert "FI made the experiment smaller" in log and "the checks, thresholds and tolerances are unchanged" in log
    assert re.search(r"FI estimates the experiment takes about .+; the limit is 8 seconds", log)
    note = eng._run_checks_note()
    assert "3 setting(s) with 40 run(s) each to 2 setting(s) with 4 run(s) each" in note and "limitations" in note
    assert run_estimate.read_record(eng.quest_root)["fits"] is True
    assert not (eng.quest_root / "raw").exists()


@pytest.mark.asyncio
async def test_a_run_that_is_still_too_long_after_the_requests_stops_plainly_and_a_resume_asks_no_more(
        tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(
        tmp_path, timeout_s=6, answers=[_smaller({"n": [1, 2, 4]}, 30), _smaller({"n": [1, 2, 4]}, 20), _smaller({"n": [1]}, 1)])
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and len(model.asked) == 2, "asked twice, never a third time"
    text = patch["exec_result"]["too_long"]
    assert re.fullmatch(r"the experiment would take about .+, more than the 6 seconds allowed; "
                        r"FI asked the plan 2 times to make it smaller", text), text
    assert patch["exec_give_up_reason"] == text and patch["result_json"] == {} and patch["exec_result"]["returncode"] == 1
    # The repair step does not try to fix a script, the gate stops without a second design, and the stop says it plainly.
    assert await eng._node_execute_reflect({"exec_result": patch["exec_result"],
                                            "exec_give_up_reason": text}) == {}
    assert eng._route_after_execute_reflect({"exec_result": patch["exec_result"], "exec_give_up_reason": text}) == "proceed"
    verdict = eng._no_results_verdict({"exec_result": patch["exec_result"], "result_json": {},
                                       "exec_give_up_reason": text, "iteration": 0})
    assert verdict is not None and verdict["stuck"] is True and verdict["stuck_reason"] == "too_long"
    stuck = await eng._node_stuck_no_findings({"evidence_assessment": {"stuck_reason": "too_long"},
                                               "exec_result": patch["exec_result"]})
    assert "more than the 6 seconds allowed" in stuck["stuck"]["say"] and "asked the plan 2 times" in stuck["stuck"]["say"]
    assert "No paper was written" in stuck["stuck"]["say"]
    assert (eng.quest_root / "needs" / "STUCK.json").is_file() and not (eng.quest_root / "paper.md").exists()
    # A resume stops again with the same words, asking and timing nothing.
    calls = ex.calls
    again = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert again is not None and again["exec_result"]["too_long"] == text
    assert len(model.asked) == 2 and ex.calls == calls


@pytest.mark.asyncio
async def test_a_plan_that_cannot_be_asked_stops_plainly_without_saying_it_asked(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=8, answers=[])
    eng._client = None  # type: ignore[assignment]
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None
    text = patch["exec_result"]["too_long"]
    assert "FI had no way to ask the plan to make it smaller" in text and "asked the plan" not in text


@pytest.mark.asyncio
async def test_an_answer_that_is_not_a_smaller_run_of_the_same_settings_is_not_used(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    eng, model, runner, ex = await _sizing(
        tmp_path, timeout_s=8,
        answers=[_smaller({"m": [1, 2]}, 4), _smaller({"n": [1, 2, 4, 8]}, 40)])
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and len(model.asked) == 2
    design = plan.load_design(eng.quest_root)[0]
    assert design["protocol"]["grid"] == {"n": [1, 2, 4]} and design["protocol"]["runs_per_setting"] == 40
    assert "complete now" not in capsys.readouterr().out
    assert "plan.md is unchanged" in _log(eng)


def _rejected(log: str) -> str:
    return " ".join(re.findall(r"could not be used: (.+)", log))


@pytest.mark.asyncio
async def test_a_smaller_run_may_not_drop_the_range_a_setting_covers_or_invent_values(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(
        tmp_path, timeout_s=6, answers=[_smaller({"n": [1, 2]}, 4), _smaller({"n": [1, 3, 4]}, 4)])
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is not None
    said = _rejected(_log(eng))
    assert "drops the first or last value of `n` (1 and 4)" in said and "the range the study covers" in said
    assert "gives `n` values it did not have" in said and "`protocol.numerical_axes`" in said
    assert plan.load_design(eng.quest_root)[0]["protocol"]["grid"] == {"n": [1, 2, 4]}
    asked = model.asked[0]
    assert "Keep the first and last value of every setting" in asked and "do not coarsen a numerical resolution" in asked


@pytest.mark.asyncio
async def test_a_setting_the_plan_marks_as_numerical_resolution_may_be_made_coarser(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(
        tmp_path, timeout_s=8, answers=[_smaller({"n": [1, 3, 4]}, 4)], numerical_axes=["n"])
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    assert plan.load_design(eng.quest_root)[0]["protocol"]["grid"] == {"n": [1, 3, 4]}
    assert "only `n` (numerical resolution) may be made coarser" in model.asked[0]


@pytest.mark.asyncio
async def test_a_smaller_run_may_not_cut_the_runs_below_what_the_plans_precision_or_minimum_needs(tmp_path: Path) -> None:
    # A target half-width of 0.1 needs 97 trials; the plan has 40 runs, so none may be cut away.
    eng, model, runner, ex = await _sizing(
        tmp_path, timeout_s=6, answers=[_smaller({"n": [1, 4]}, 10), _smaller({"n": [1, 4]}, 39)],
        precision={"target_half_width": 0.1})
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is not None
    assert "cuts the runs per setting to 10, below the 40 the plan's precision or stated minimum needs" in _rejected(_log(eng))
    assert "keep at least 40 runs per setting" in model.asked[0]
    # A stated minimum is a floor for a run that has more; thinning the settings is still allowed.
    eng2, model2, runner2, _ = await _sizing(
        tmp_path / "b", timeout_s=8, answers=[_smaller({"n": [1, 4]}, 12), _smaller({"n": [1, 4]}, 6)],
        min_runs_per_setting=8)
    assert await eng2._size_the_run({"iteration": 0}, runner2, sys.executable, None) is None
    assert plan.load_design(eng2.quest_root)[0]["protocol"]["runs_per_setting"] == 12
    assert "keep at least 8 runs per setting" in model2.asked[0]


@pytest.mark.asyncio
async def test_nothing_is_timed_for_a_later_pass_or_a_cluster_job(tmp_path: Path) -> None:
    eng, model, runner, ex = await _sizing(tmp_path, timeout_s=8, answers=[])
    assert await eng._size_the_run({"iteration": 1}, runner, sys.executable, None) is None and ex.calls == 0
    eng.config.execution.background_jobs = True
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None and ex.calls == 0


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
