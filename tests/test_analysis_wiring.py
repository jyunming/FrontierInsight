"""The engine with the analysis FI writes (core/analysis_frame.py): with the frame on and a plan in the contract form, the code
step asks the model for the trial and the functions of derive.py only, FI writes experiment.py and the files it uses, and the
run prints exactly the declared ids. A function of derive.py that fails is repaired alone, through its body, and no trial is run
again for it; a fault of FI's own is never sent to the model; a quantity the run did not work out is listed in the paper's
limitations and in QUALITY.md; ``code/run.py`` gives the engine's numbers and leaves the quest's own results as they are. With
the frame off, or a plan in an older form, every text is what it was (sha256 at 811719af). Made-up topic: a colony's growth.
No real model is called."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import analysis_frame as af
from core import paper_values
from core import code_layout, code_project, finding_repair, improve, outside_run, paper_facts, quality_report, skeleton
from core import engine as engine_mod
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

DESIGN: dict[str, Any] = {"hypothesis": "growth rises with the rate", "protocol": {
    "form": "contract", "grid": {"rate": [0.1, 0.2], "dose": [1, 2]}, "runs_per_setting": 3,
    "metrics": [{"id": "growth", "kind": "mean", "role": "result"}],
    "derived": [{"id": "peak", "inputs": ["growth"], "role": "result"},
                {"id": "spread", "inputs": ["growth"], "role": "diagnostic", "formula": "growth / (rate * dose)"}]}}
DECLARED = {"growth", "peak", "spread"}

TRIAL = '''
def trial(cell, ctx):
    out = TrialResult()
    out["growth"] = cell["rate"] * cell["dose"] + ctx.rng.random() * 0.01
    return out
'''
DERIVE = '''
def derive_peak(table):
    return max(table.mean("growth", k) for k in table.keys())
'''


def _reply(derive: str | None = DERIVE) -> str:
    return ("```python\n# file: simulate.py\n" + TRIAL + "```\n\n"
            + ("```python\n# file: derive.py\n" + derive + "```\n\n" if derive is not None else "") + "DEPS:\n")


class _Model:
    """A fake model: the trial request gets the trial, the code request ``code``, a repair request ``repair`` (JSON)."""

    def __init__(self, code: str, repair: dict[str, Any] | None = None) -> None:
        self.code, self.repair, self.prompts = code, repair, []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if prompt.startswith("You are the **Trial Function"):
            return "```python\n" + TRIAL + "```"
        if self.repair is not None and skeleton.DERIVE_REPAIR_NOTE in prompt:
            return json.dumps(self.repair)
        return self.code

    async def aclose(self) -> None:
        return None


def _engine(tmp_path: Path, *, skeleton_on: bool = True) -> Engine:
    cfg = Config(topic="a colony's growth", title="colony", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False, exec_reflect_max_iterations=3, ideate_reflect=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True, code_skeleton=skeleton_on),
                 knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"))
    engine = Engine(cfg)
    engine.quest_root.mkdir(parents=True, exist_ok=True)

    async def no_install(packages):  # noqa: ANN001
        return []

    engine._install_packages = no_install  # type: ignore[method-assign]
    return engine


async def _implement(tmp_path: Path, model: _Model, design: dict[str, Any] = DESIGN, *,
                     skeleton_on: bool = True) -> tuple[Engine, dict[str, Any]]:
    engine = _engine(tmp_path, skeleton_on=skeleton_on)
    await engine.executor.setup(engine.quest_root)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    engine._client = model
    state: dict[str, Any] = {"design": design, "implement_outline": {}}
    state.update(await engine._node_implement(dict(state)))
    return engine, state


def _ids(result: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    for k, v in result.items():
        if k in ("n_failed", "why_missing"):
            continue
        if "=" in k and isinstance(v, dict):
            found |= {n[: -len("_values")] if n.endswith("_values") else n for n in v if n not in ("n_failed", "why_missing")}
        else:
            found.add(k)
    return found


# ---- the whole path: implement, the run, the analysis ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_the_engine_writes_fis_analysis_and_the_run_gives_exactly_the_declared_ids(tmp_path: Path) -> None:
    model = _Model(_reply())
    engine, state = await _implement(tmp_path, model)
    code = engine.quest_root / "code"
    assert af.is_fis_file(code / "experiment.py") and (code / "experiment.py").read_text(encoding="utf-8") == af.EXPERIMENT
    assert "derive_peak" in (code / "derive.py").read_text(encoding="utf-8")
    assert all((code / rel).is_file() for rel in [*af.carried_files(), "fi_table.py", "fi_frame.py"])
    assert not any("# file: experiment.py" in p for p in model.prompts)  # nothing asks the model for experiment.py
    state.update(await engine._node_execute(dict(state)))
    result = state["exec_result"]
    assert result["returncode"] == 0, result.get("stderr_tail")
    assert _ids(state["result_json"]) == DECLARED
    record = af.read_record(engine.quest_root)
    assert record["derived"]["peak"]["status"] == "ok" and record["not_computed"] == []
    peak = record["derived"]["peak"]["value"]
    assert peak == pytest.approx(max(c["growth"] for k, c in state["result_json"].items() if "=" in k))
    answers = json.loads((engine.quest_root / ".fi" / "analysis_answers.json").read_text(encoding="utf-8"))
    assert answers["runs"][0]["values"]["peak"] == peak  # the paper's key for a quantity no trial returns can be filled
    filled, _values = paper_values.fill_values("The peak was {{v:peak}}.", paper_values.records_of(
        engine.quest_root, DESIGN["protocol"]), None)
    assert "fi:v" in filled and f"{peak:.3g}"[:4] in filled, filled
    assert await engine._node_execute_reflect(dict(state)) == {}  # nothing to repair

    # code/run.py gives the engine's numbers through the same files, in its own folder, and leaves the quest's results alone
    kept = (engine.quest_root / af.DERIVED_PATH).read_bytes()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_TRIALS")}
    run = subprocess.run([sys.executable, str(code / "run.py")], cwd=engine.quest_root, env=env, capture_output=True,
                         text=True, timeout=300, creationflags=NO_WINDOW)
    assert run.returncode == 0, run.stderr[-2000:]
    theirs = json.loads((engine.quest_root / "run_output" / af.DERIVED_PATH).read_text(encoding="utf-8"))
    assert theirs["derived"] == record["derived"]
    assert (engine.quest_root / af.DERIVED_PATH).read_bytes() == kept


@pytest.mark.asyncio
async def test_a_failing_body_is_repaired_alone_and_no_trial_runs_again_for_it(tmp_path: Path) -> None:
    broken = DERIVE.replace('table.mean("growth", k)', 'table.mean("growth", k) / 0')
    model = _Model(_reply(broken), repair={"functions": DERIVE, "patch_summary": "no division by zero"})
    engine, state = await _implement(tmp_path, model)
    state.update(await engine._node_execute(dict(state)))
    result = state["exec_result"]
    assert result["returncode"] != 0 and result["failed_script"] == "derive.py"
    version = outside_run.code_version(engine.quest_root)
    calls = len(model.prompts)
    patch = await engine._node_execute_reflect(dict(state))
    assert len(model.prompts) == calls + 1
    prompt = model.prompts[-1]
    assert prompt.startswith(skeleton.DERIVE_REPAIR_NOTE) and "derive_peak" in prompt.split("\n\n", 1)[0] + prompt
    assert af.EXPERIMENT_HEAD not in prompt  # FI's experiment.py is never shown as the code to repair
    code = engine.quest_root / "code"
    assert "/ 0" not in (code / "derive.py").read_text(encoding="utf-8")
    assert (code / "experiment.py").read_text(encoding="utf-8") == af.EXPERIMENT
    assert patch["exec_patch_pending"] and "code" not in patch
    assert outside_run.code_version(engine.quest_root) == version  # the repair of the analysis runs no trial again
    state.update(patch)
    state.update(await engine._node_execute(dict(state)))
    assert state["exec_result"]["returncode"] == 0 and _ids(state["result_json"]) == DECLARED


@pytest.mark.asyncio
async def test_a_formula_fi_cannot_compute_is_left_out_when_the_plan_is_read_and_the_run_goes_on(tmp_path: Path) -> None:
    design = json.loads(json.dumps(DESIGN))
    design["protocol"]["derived"][1]["formula"] = "growth * nothing_here"
    model = _Model(_reply())
    engine, state = await _implement(tmp_path, model, design)
    assert [d["id"] for d in af.plan_of(design)["derived"]] == ["peak"]  # left out, with a gap that says why
    state.update(await engine._node_execute(dict(state)))
    assert state["exec_result"]["returncode"] == 0 and _ids(state["result_json"]) == {"growth", "peak"}
    assert await engine._node_execute_reflect(dict(state)) == {}


@pytest.mark.asyncio
async def test_a_formula_fault_at_run_time_stops_plainly_with_advice_that_works(tmp_path: Path) -> None:
    from core import rerun_from

    engine = _engine(tmp_path)
    _fis_quest(engine)
    record = {"schema": af.DERIVED_SCHEMA, "derived": {"spread": {
        "status": "failed", "fi_fault": True, "plan": True, "by": "formula", "role": "diagnostic",
        "why": "the plan's formula for `spread` cannot be computed: it uses `nothing_here`, which the plan does not declare"}},
        "fi_faults": ["the plan's formula for `spread` cannot be computed"], "figures": []}
    (engine.quest_root / af.DERIVED_PATH).parent.mkdir(parents=True, exist_ok=True)
    (engine.quest_root / af.DERIVED_PATH).write_text(json.dumps(record), encoding="utf-8")
    chat = AsyncMock(return_value="{}")
    engine._client = type("Stub", (), {"chat": chat})()
    state = {"design": DESIGN, "result_json": {"peak": 1.0}, "exec_result": {
        "returncode": af.FI_FAULT_EXIT, "failed_script": "experiment.py", "stdout_tail": "", "stderr_tail": ""}}
    patch = await engine._node_execute_reflect(state)
    chat.assert_not_awaited()  # no model call
    said, advice = patch["exec_result"]["cannot_repair"], patch["exec_result"]["cannot_repair_advice"]
    assert "cannot compute a formula of the plan" in said and "nothing_here" in said
    assert f"--resume {engine.quest_id} --from design --approve-as <your name>" in advice and "plan.md" in advice
    # what the advice says is so: the design step needs the approval, and it reads plan.md again rather than moving it aside
    assert rerun_from.needs_approval("design") and "plan.md" not in rerun_from.outputs_of("design")
    assert "plan.md" in rerun_from.REDOES["design"]
    assert engine._route_after_execute_reflect({**state, **patch}) == "stuck"
    assert (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == af.EXPERIMENT


def _fis_quest(engine: Engine, design: dict[str, Any] = DESIGN) -> Path:
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text(skeleton.frame_source(design, stochastic=True), encoding="utf-8")
    af.ensure_files(code, design)
    return code


@pytest.mark.asyncio
async def test_an_analysis_stopped_for_its_time_in_a_function_goes_to_that_functions_repair(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _fis_quest(engine)
    model = _Model("{}", repair={"functions": DERIVE, "patch_summary": "faster"})
    engine._client = model
    state = {"design": DESIGN, "result_json": {}, "exec_result": {
        "returncode": 1, "timed_out": True, "failed_script": af.DERIVE_FILE, "stdout_tail": "", "analysis_step": "derive_peak",
        "stderr_tail": af.STEP_MARK + "derive_peak"}}
    patch = await engine._node_execute_reflect(state)
    assert "cannot_repair" not in (patch.get("exec_result") or {})  # no stop
    asked = model.prompts[-1]
    assert skeleton.DERIVE_REPAIR_NOTE in asked and "`derive_peak`" in asked


@pytest.mark.asyncio
@pytest.mark.parametrize("repairs_made", [0, 3])  # 3: every repair spent, and the fault still stops (no repair is asked for it)
async def test_an_empty_record_is_fis_fault_said_plainly_with_no_model_call(tmp_path: Path, repairs_made: int) -> None:
    engine = _engine(tmp_path)
    _fis_quest(engine)
    chat = AsyncMock(return_value="{}")
    engine._client = type("Stub", (), {"chat": chat})()
    state = {"design": DESIGN, "result_json": {}, "exec_reflect_iter": repairs_made, "exec_result": {
        "returncode": 2, "failed_script": "experiment.py", "stdout_tail": "",
        "stderr_tail": "FI's analysis cannot start (this is FI's, not the study's code): FI's record of the trials is empty"}}
    patch = await engine._node_execute_reflect(state)
    chat.assert_not_awaited()
    assert "FI's own analysis stopped" in patch["exec_result"]["cannot_repair"]
    assert "record of the trials is empty" in patch["exec_result"]["cannot_repair"]
    assert "fault in FI" in patch["exec_result"]["cannot_repair_advice"]
    assert engine._route_after_execute_reflect({**state, **patch}) == "stuck"
    # the evidence gate names the stop by its kind (``stuck_reason``), and the stop report says it
    said = (await engine._node_stuck_no_findings({**state, **patch, "evidence_assessment": {"stuck_reason": "cannot_repair"}}))["stuck"]
    assert "trials that ran are kept" in said["problem"] and "repairs did not remove it" not in said["problem"]
    assert "FI's own analysis" in said["why_repairs_ended"]


# ---- what the run did not work out is said ---------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_stub_is_sent_to_its_body_before_the_long_run(tmp_path: Path) -> None:
    model = _Model(_reply(None))  # the reply gives no function of derive.py: FI's stub stays
    engine, state = await _implement(tmp_path, model)
    assert "derive_peak" in af.stubs((engine.quest_root / "code" / "derive.py").read_text(encoding="utf-8"))
    state.update(await engine._node_execute(dict(state)))
    result = state["exec_result"]
    assert result.get("directive") and result["failed_script"] == "derive.py" and result["failed_body"] == "derive_peak"
    assert "`peak` is computed by no trial and no function of the study's yet" in result["stderr_tail"]
    assert engine._fis_analysis_route(state, result, 1, []) == {"body": "derive_peak"}


def _run_analysis(quest: Path) -> int:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_TRIALS")}
    return subprocess.run([sys.executable, "experiment.py"], cwd=quest / "code", env=env, capture_output=True, text=True,
                          timeout=300, creationflags=NO_WINDOW).returncode


# ---- a check after a clean run is routed by FI's record of which function made what it flags ------------------------------
FIGURED = {**DESIGN, "protocol": {**DESIGN["protocol"], "figures": [
    {"id": "fig1", "file": "growth.png", "shows": ["growth"], "x": "rate", "series": "dose"}]}}
FIGURE_BODY = '''
def figure_fig1(fig, table):
    for k in table.keys():
        fig.axes[0].axhline(table.mean("growth", k))  # reference lines only: the check of the figures flags it
'''


@pytest.mark.asyncio
async def test_a_result_check_after_a_clean_run_goes_to_the_function_that_made_what_it_flags(tmp_path: Path) -> None:
    from core import result_checks
    from core.execution import ExecutionResult

    model = _Model(_reply(DERIVE + FIGURE_BODY))
    engine, state = await _implement(tmp_path, model, FIGURED)
    state.update(await engine._node_execute(dict(state)))
    assert state["exec_result"]["returncode"] == 0
    record = af.read_record(engine.quest_root)
    assert record["bodies"]["figure_fig1"]["outputs"] == ["figure:fig1"] and record["bodies"]["derive_peak"]["outputs"] == ["peak"]
    assert af.made_by(record, figure="growth.png") == "figure_fig1" and af.made_by(record, quantity="peak") == "derive_peak"
    assert af.made_by(record, quantity="spread") == ""  # FI's formula over the trials alone: no function of the study's
    found = result_checks.Found("degenerate_results", result_checks.ANALYSIS, "redraw it from the run's values",
                                "a figure draws only reference lines", "said", figures=["growth.png"])
    engine._trial_mode = True
    engine._result_findings = lambda st, rj: [found]  # type: ignore[method-assign]
    clean = ExecutionResult(0, "RESULT_JSON: " + json.dumps(state["result_json"]), "", 1.0, False)
    sent, to_fix = await engine._results_match_run(dict(state), clean, None, True)
    assert sent.returncode == 1 and to_fix == af.DERIVE_FILE and engine._analysis_sendback_body == "figure_fig1"
    routed = engine._fis_analysis_route(state, {"returncode": 1, "failed_script": to_fix, "failed_body": "figure_fig1",
                                                "result_check_sent_back": True, "stderr_tail": sent.stderr}, 1, [])
    assert routed == {"body": "figure_fig1"}
    # A finding in what FI's own lines printed (a trial's mean in FI's table) has no function to go to: it is left out of
    # the results and said, the run goes on, and nothing stops.
    own = result_checks.Found("reported_settings", result_checks.ANALYSIS, "d", "a mean is not the trials' mean", "said",
                              keys=["rate=0.1,dose=1.growth"])
    engine._result_findings = lambda st, rj: [own]  # type: ignore[method-assign]
    kept, to_fix = await engine._results_match_run(dict(state), clean, None, True)
    assert kept.returncode == 0 and to_fix is None and engine._analysis_sendback_body == ""
    assert "a mean is not the trials' mean" in (engine.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8")
    # A send-back that names no function, on a run FI's record shows clean, goes where it always went: never a stop.
    assert engine._fis_analysis_route(state, {"returncode": 1, "failed_script": "experiment.py", "directive": True,
                                              "stderr_tail": "[FI] something"}, 1, []) == {"body": None}


def test_problems_of_a_clean_run_go_to_the_function_fi_lays_them_to(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from core import numeric_warnings

    engine = _engine(tmp_path)
    _fis_quest(engine)
    record = {"schema": af.DERIVED_SCHEMA, "figures": [], "derived": {
        "peak": {"status": "ok", "value": 9.0, "by": "derive_peak", "role": "result", "inputs": ["growth"]},
        "spread": {"status": "ok", "value": 1.0, "by": "formula", "role": "diagnostic", "inputs": ["growth"]}},
        "bodies": {"derive_peak": {"status": "ok", "outputs": ["peak"], "seconds": 0.1,
                                   "warnings": ["RuntimeWarning: overflow encountered in exp"]}}}
    (engine.quest_root / af.DERIVED_PATH).parent.mkdir(parents=True, exist_ok=True)
    (engine.quest_root / af.DERIVED_PATH).write_text(json.dumps(record), encoding="utf-8")
    state = {"design": DESIGN, "result_json": {"peak": 9.0}}
    warned = numeric_warnings.scan("x.py:3: RuntimeWarning: overflow encountered in exp\n")
    assert engine._fis_analysis_route(state, {"returncode": 0}, 0, [], warned, []) == {"body": "derive_peak"}
    other = numeric_warnings.scan("simulate.py:9: RuntimeWarning: invalid value encountered in divide\n")
    assert engine._fis_analysis_route(state, {"returncode": 0}, 0, [], other, []) == {"body": None}  # the simulation's
    bound = SimpleNamespace(path="peak", paths=())
    assert engine._fis_analysis_route(state, {"returncode": 0}, 0, [], [], [bound]) == {"body": "derive_peak"}
    trial = SimpleNamespace(path="rate=0.1,dose=1.growth", paths=())
    assert engine._fis_analysis_route(state, {"returncode": 0}, 0, [], [], [trial]) == {"body": None}


@pytest.mark.asyncio
async def test_a_difference_only_in_what_fis_own_analysis_printed_is_recorded_not_sent_back(tmp_path: Path) -> None:
    model = _Model(_reply())
    engine, state = await _implement(tmp_path, model)
    said = "the analysis prints a count of trials for rate=0.1,dose=1 that is not FI's"

    def problems(self, st, split, result, random=None):  # noqa: ANN001
        self._manifest_failed_trials, self._manifest_fabricated, self._manifest_unprinted = 0, [], []
        self._manifest_differences, self._manifest_note, self._manifest_analysis_problems = [], "", [said]
        return "differs", [said]

    engine._run_manifest_problems = problems.__get__(engine)  # type: ignore[method-assign]
    engine.config.engine.run_manifest_check = "block"  # every difference acted on (under warn it is only recorded anyway)
    state.update(await engine._node_execute(dict(state)))
    assert state["exec_result"]["returncode"] == 0 and state["result_json"]
    record = json.loads((engine.quest_root / "needs" / "RUN_MANIFEST_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == "warned" and said in record["problems"] and record["fault"] == "FI's own analysis"


def _record_of_trials() -> dict[str, Any]:
    from core import trial_runner

    cells = []
    for rate in (0.1, 0.2):
        for dose in (1, 2):
            cell = {"rate": rate, "dose": dose}
            cells.append({"cell": cell, "key": trial_runner.cell_key(cell), "planned": 2, "ok": 2, "failed": 0,
                          "metrics": {"growth": {"values": [rate * dose, rate * dose + 0.01], "trials": [0, 1]}}})
    return {"schema": "fi.trials/v1", "cells": cells}


def test_a_quantity_not_worked_out_is_in_the_papers_limitations_and_in_quality_md(tmp_path: Path) -> None:
    design = json.loads(json.dumps(DESIGN))
    design["protocol"]["derived"][0]["role"] = "diagnostic"  # a stub of a diagnostic: the run goes on, and says so
    quest = tmp_path / "quest"
    (quest / "raw").mkdir(parents=True)
    (quest / ".fi").mkdir()
    (quest / "raw" / "trials.json").write_text(json.dumps(_record_of_trials()), encoding="utf-8")
    af.ensure_files(quest / "code", design)
    assert _run_analysis(quest) == 0
    assert [i["id"] for i in af.not_computed(af.read_record(quest))] == ["peak"]
    items = [i for i in paper_facts.required_items(quest) if i.kind == paper_facts.NOT_COMPUTED]
    assert [i.name for i in items] == ["peak"] and "was not written" in items[0].text
    assert "was not written" in paper_facts.limitations_text(quest)
    report = quality_report.build(quest, running=False)
    analysis = next(s for s in report["stages"] if s["stage"] == "analysis")
    rows = [p for p in analysis["problems"]["items"] if "peak was not computed" in p["text"]]
    assert rows and "not written" in rows[0]["text"] and rows[0]["status"] == "left out"
    assert "peak was not computed" in quality_report.render(report)


# ---- the parts each path routes by -----------------------------------------------------------------------------------------
def test_the_failure_is_read_from_fis_record_not_from_the_traceback() -> None:
    record = {"schema": af.DERIVED_SCHEMA, "derived": {
        "spread": {"status": "failed", "role": "diagnostic", "by": "derive_spread", "why": "ValueError: x"},
        "peak": {"status": "refused", "role": "result", "by": "derive_peak", "why": "returned a string"}}}
    assert af.failure(record, "", 1, printed=True)["body"] == "derive_peak"  # a result first, whatever the order
    assert af.failure({}, "", 1, printed=True)["kind"] == "fi"
    crash = 'Traceback (most recent call last):\n  File "C:/q/code/derive.py", line 40\n    def derive_peak(table)\nSyntaxError: x'
    assert af.failure({}, crash, 1, printed=False)["kind"] == "module"
    assert af.failure({}, "Traceback ...\n  File \"experiment.py\", line 3\nKeyError: 'x'", 1, printed=False)["kind"] == "fi"
    fault = {"schema": af.DERIVED_SCHEMA, "derived": {"spread": {"status": "failed", "fi_fault": True, "by": "formula",
                                                                  "why": "the plan's formula for `spread` cannot be computed"}}}
    got = af.failure(fault, "", af.FI_FAULT_EXIT, printed=True)
    assert got["kind"] == "fi" and got["formula"] and "spread" in got["why"]
    assert af.failed_script("C:/nowhere/experiment.py", 1) == "experiment.py"


def test_an_overlap_goes_to_the_figure_body_that_drew_it(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    design = json.loads(json.dumps(DESIGN))
    design["protocol"]["figures"] = [{"id": "fig1", "file": "growth.png", "shows": ["growth"], "x": "rate", "series": "dose"}]
    state = {"design": design}
    said = ['figures/growth.png: the legend is drawn over the data it labels']
    assert engine._fis_analysis_route(state, {}, 0, said) == {"body": "figure_fig1"}
    assert engine._fis_analysis_route(state, {}, 0, ['figures/other.png: x'])["stop"] == {"figure_overlap_repaired": True}
    assert engine._fis_analysis_route(state, {}, 0, []) == {"body": None}  # a run that is wrong: the simulation


def test_compose_keeps_fis_frame_and_a_reply_that_cannot_be_used_leaves_the_stubs() -> None:
    text, note = af.compose(DESIGN, DERIVE)
    assert note == "" and "derive_peak" not in af.stubs(text) and af.derive_intact(af.derive_source(DESIGN), text) == ""
    kept, why = af.compose(DESIGN, "def derive_peak(table:\n")
    assert "not valid Python" in why and "derive_peak" in af.stubs(kept)
    again, _ = af.compose(DESIGN, "", text)
    assert again == text  # the bodies the quest has are kept


def test_the_other_repair_paths_never_edit_fis_files(tmp_path: Path) -> None:
    quest = tmp_path / "q"
    code = quest / "code"
    code.mkdir(parents=True)
    (code / "simulate.py").write_text(skeleton.frame_source(DESIGN, stochastic=True), encoding="utf-8")
    af.ensure_files(code, DESIGN)
    files = improve.editable(quest)
    assert "simulate.py" in files and not {"experiment.py", "derive.py", "fi_table.py", "fi_figures.py"} & set(files)
    assert not any(f.startswith(af.CORE_DIR + "/") for f in files)
    assert finding_repair.target_file({"files": ["experiment.py"]}, code) == "derive.py"
    assert af.failed_script(code / "experiment.py", 1) == "derive.py"
    assert af.failed_script(code / "experiment.py", af.FI_FAULT_EXIT) == "experiment.py"
    names = {p.relative_to(code).as_posix() for p in code_layout.simulation_inputs(quest)}
    assert "simulate.py" in names and not any(n in names for n in ("derive.py", "experiment.py", "fi_table.py", "fi_figures.py"))


# ---- flag off, and a plan in an older form: every text is what it was (sha256 at 811719af) -------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


#: sha256 (first 16 hex digits) of the engine texts this change touched, as the code before it made them (commit 811719af).
PINS_AT_811719AF = {
    "rerun": "34124f75b6227ac4", "extend": "2914c7194f650d8d", "extend_frame": "9f8a6f074628a11a", "reminder": "b4415cb88db73e38",
    "split_reminder": "26239b84a51fbd8a", "cluster_reminder": "b697f4362f85d4a6", "run_none": "dab51e372647d8db",
    "run_plain": "dab51e372647d8db",
    # the frame's own text (written only with the frame on): its RESULTS comment, its trial docstring and the refusal message
    # changed with the one list of names a trial may return (core/skeleton.py returnable_names); it was "ad084c60d61b9dde"
    "run_framed": "5b45d638db4dcd2c",
}


def test_with_the_analysis_not_fis_every_touched_text_is_what_it_was(tmp_path: Path) -> None:
    review = {"blocking": ["the peak is computed from the wrong setting"], "weaknesses": "one seed only", "suggestions": []}
    scripts = {"simulate.py": "def trial(cell, ctx):\n    return {}\n", "experiment.py": "print('RESULT_JSON: {}')\n"}
    package = ("colony", {"colony/model.py": "def growth(r):\n    return r\n"})
    pins = {
        "rerun": _sha(engine_mod._rerun_directive(review, "peak")),
        "extend": _sha(engine_mod._extend_directive(scripts, ["the spread"], "", package, False)),
        "extend_frame": _sha(engine_mod._extend_directive(scripts, ["the spread"], "print('job')\n", package, True)),
        "reminder": _sha(code_layout.reminder("colony")),
        "split_reminder": _sha(engine_mod._SPLIT_REPLY_REMINDER),
        "cluster_reminder": _sha(engine_mod._CLUSTER_REPLY_REMINDER),
        "run_none": _sha(code_project.run_source(None)),
    }
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("print(1)\n", encoding="utf-8")
    (code / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    pins["run_plain"] = _sha(code_project.run_source(code))
    older = {"protocol": {"grid": {"rate": [0.1, 0.2]}, "runs_per_setting": 2, "metrics": [{"id": "growth", "kind": "mean"}]}}
    (code / "simulate.py").write_text(skeleton.frame_source(older, stochastic=True), encoding="utf-8")
    pins["run_framed"] = _sha(code_project.run_source(code))
    assert pins == PINS_AT_811719AF, json.dumps(pins, indent=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("skeleton_on, design", [
    (False, DESIGN),  # the frame off: the model's analysis, as before
    (True, {"hypothesis": "h", "protocol": {k: v for k, v in DESIGN["protocol"].items() if k not in ("form", "derived")}}),
])
async def test_with_the_frame_off_or_an_older_plan_the_model_writes_experiment_py(tmp_path: Path, skeleton_on: bool,
                                                                                   design: dict[str, Any]) -> None:
    analysis = "import json\nprint('RESULT_JSON: ' + json.dumps({}))\n"
    reply = ("```python\n# file: simulate.py\n" + TRIAL + "```\n\n```python\n# file: experiment.py\n" + analysis + "```\n\nDEPS:\n")
    engine, state = await _implement(tmp_path, _Model(reply), design, skeleton_on=skeleton_on)
    code = engine.quest_root / "code"
    assert not af.is_fis_file(code / "experiment.py") and not (code / "derive.py").exists()
    assert (code / "experiment.py").read_text(encoding="utf-8").strip() == analysis.strip()
    assert "FI_ANALYSIS_FILES" not in (code / "run.py").read_text(encoding="utf-8")


# ---- FI's own files are not the study's -----------------------------------------------------------------------------------
def test_fis_formula_calculator_is_no_part_of_the_study(tmp_path: Path) -> None:
    from core import code_review, split_run, trial_runner

    engine = _engine(tmp_path)
    code = _fis_quest(engine)
    root = engine.quest_root
    assert (code / af.CORE_DIR / "__init__.py").is_file()
    assert code_layout.package_dirs(code) == [] and code_layout.package_sources(code) == {}
    assert not any(af.CORE_DIR in p.as_posix() for p in code_layout.simulation_inputs(root))
    assert not any(name.startswith(af.CORE_DIR) for name in code_review._code_files(code))
    before = (outside_run.code_version(root), trial_runner._run_key(code / "simulate.py", {}, 1, 0, False),
              split_run.simulation_sha(code / "simulate.py"))
    calc = code / af.CORE_DIR / "oracle_forms.py"
    calc.write_text(calc.read_text(encoding="utf-8") + "\n# a newer FI\n", encoding="utf-8")  # FI changed between runs
    after = (outside_run.code_version(root), trial_runner._run_key(code / "simulate.py", {}, 1, 0, False),
             split_run.simulation_sha(code / "simulate.py"))
    assert after == before  # a resumed quest reruns no trial for it
    # a package of the study's own is still the study's
    (code / "model").mkdir()
    (code / "model" / "__init__.py").write_text("RATE = 1\n", encoding="utf-8")
    assert [p.name for p in code_layout.package_dirs(code)] == ["model"]
    assert trial_runner._run_key(code / "simulate.py", {}, 1, 0, False) != before[1]


def test_the_gate_judges_a_figure_function_that_names_nothing_a_fault(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from core import pipeline_gate

    engine = _engine(tmp_path)
    code = _fis_quest(engine, FIGURED)
    text, note = af.compose(FIGURED, DERIVE + "\ndef figure_fig1(fig, table):\n    fig.axes[0].plot(rates, [1, 2])\n",
                            (code / "derive.py").read_text(encoding="utf-8"))
    (code / "derive.py").write_text(text, encoding="utf-8")
    (engine.quest_root / "raw").mkdir(exist_ok=True)
    cells = [{"cell": {"rate": r, "dose": d}, "key": f"rate={r},dose={d}", "planned": 2, "ok": 2, "failed": 0,
              "metrics": {"growth": {"values": [r * d, r * d + 0.1], "trials": [0, 1]}}} for r in (0.1, 0.2) for d in (1, 2)]
    (engine.quest_root / "raw" / "trials.json").write_text(json.dumps({"schema": "fi.trials/v1", "cells": cells}),
                                                           encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_TRIALS")}
    r = subprocess.run([sys.executable, "experiment.py"], cwd=code, env=env, capture_output=True, text=True, timeout=300,
                       creationflags=NO_WINDOW)
    assert r.returncode == 1
    got = pipeline_gate._fis_analysis_result(engine.quest_root, SimpleNamespace(returncode=r.returncode, stdout=r.stdout,
                                                                                 stderr=r.stderr), FIGURED["protocol"], {}, [])
    assert got.kind == pipeline_gate.FAULT and got.body == "figure_fig1" and "NameError" in got.detail


def test_a_finding_repair_of_derive_py_says_what_of_the_file_is_fis() -> None:
    derive = af.derive_source(DESIGN)
    assert skeleton.repair_note(derive) == skeleton.DERIVE_FINDING_NOTE and "frame" in skeleton.DERIVE_FINDING_NOTE
    assert skeleton.repair_note("print(1)\n") == ""  # a study without FI's files is asked as before


def test_the_model_reads_only_the_last_step_fis_analysis_said(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    err = "\n".join([af.STEP_MARK + "fi: reading FI's record of the trials", af.STEP_MARK + "derive_peak",
                     "Traceback (most recent call last):", "ZeroDivisionError: division by zero"])
    said = engine._stderr_for_models(err)
    assert af.STEP_MARK + "derive_peak" in said and "reading FI's record" not in said and "ZeroDivisionError" in said


def test_a_run_sent_back_or_stopped_keeps_no_answers_of_its_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _fis_quest(engine)
    path = engine.quest_root / af.DERIVED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": af.DERIVED_SCHEMA, "derived": {"peak": {"status": "ok", "value": 3.0, "by": "derive_peak"}},
                                "from_earlier_run": True}), encoding="utf-8")
    seen: list[Any] = []
    monkeypatch.setattr(paper_values, "write_answers", lambda *a, **k: seen.append(a))
    engine._keep_analysis_answers({"peak": 3.0}, {"design": DESIGN})  # a record of an earlier run
    engine._keep_analysis_answers({}, {"design": DESIGN})  # a run whose RESULT_JSON was taken out (sent back)
    assert seen == []
    path.write_text(json.dumps({"schema": af.DERIVED_SCHEMA, "derived": {"peak": {"status": "ok", "value": 3.0}}}),
                    encoding="utf-8")
    engine._keep_analysis_answers({"peak": 3.0}, {"design": DESIGN})
    assert len(seen) == 1


# ---- what the run needs, and what is read of a run that stopped -----------------------------------------------------------
@pytest.mark.asyncio
async def test_fi_installs_matplotlib_for_the_plans_figures_whatever_the_reply_lists(tmp_path: Path) -> None:
    # FI draws the plan's figures itself (fi_figures.py): the reply's DEPS may leave matplotlib out, and in a clean venv or
    # a container the figures would then fail inside FI's own step, after the long run.
    design = json.loads(json.dumps(DESIGN))
    design["protocol"]["figures"] = [{"id": "fig1", "file": "growth.png", "shows": ["growth"], "x": "rate", "series": "dose"}]
    engine, state = await _implement(tmp_path / "figures", _Model(_reply()), design)  # the reply's DEPS list is empty
    assert "matplotlib" in state["deps"]
    asked: list[list[str]] = []

    async def install(packages):  # noqa: ANN001, ANN202
        asked.append(list(packages))
        return []

    engine._install_packages = install  # type: ignore[method-assign]
    await engine._node_execute(dict(state))
    assert asked and "matplotlib" in asked[0], asked
    _plain, plain = await _implement(tmp_path / "plain", _Model(_reply()))  # no figure planned: nothing added
    assert "matplotlib" not in plain["deps"]


@pytest.mark.asyncio
async def test_the_run_record_names_the_analysis_step_only_when_the_analysis_is_fis(tmp_path: Path) -> None:
    analysis = "raise RuntimeError('the analysis broke')\n"
    reply = ("```python\n# file: simulate.py\n" + TRIAL + "```\n\n```python\n# file: experiment.py\n" + analysis
             + "```\n\nDEPS:\n")
    engine, state = await _implement(tmp_path, _Model(reply), DESIGN, skeleton_on=False)
    state.update(await engine._node_execute(dict(state)))
    assert state["exec_result"]["returncode"] != 0 and "analysis_step" not in state["exec_result"]


def test_a_file_a_killed_run_left_does_not_lay_an_import_failure_to_a_function(tmp_path: Path) -> None:
    # A run killed while derive_peak ran leaves FI's file of what was running; the next run stops while derive.py is
    # imported (before any step is said) and names its own id on stderr: the stop is the module's, not derive_peak's.
    engine = _engine(tmp_path)
    _fis_quest(engine)
    stale = engine.quest_root / af.RUNNING_PATH
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"step": "derive_peak", "run": "a-run-that-was-killed"}), encoding="utf-8")
    stderr = "\n".join([af.RUN_MARK + "this-run", "Traceback (most recent call last):",
                         '  File "C:/q/code/derive.py", line 40, in <module>', "NameError: name 'x' is not defined"])
    for result in ({"returncode": 1, "failed_script": af.DERIVE_FILE, "stdout_tail": "", "stderr_tail": stderr,
                    "analysis_step": af.running_step(stderr, engine.quest_root)},  # as the run record keeps it
                   {"returncode": 1, "failed_script": af.DERIVE_FILE, "stdout_tail": "", "stderr_tail": stderr}):  # an older one
        assert engine._fis_analysis_route({"design": DESIGN, "result_json": {}}, result, 1, []) == {"body": ""}, result
