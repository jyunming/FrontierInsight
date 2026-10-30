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
                {"id": "spread", "inputs": ["growth"], "role": "diagnostic", "formula": "max(growth) - min(growth)"}]}}
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
async def test_a_formula_fi_cannot_compute_stops_plainly_and_is_never_sent_to_the_model(tmp_path: Path) -> None:
    design = json.loads(json.dumps(DESIGN))
    design["protocol"]["derived"][1]["formula"] = "growth * nothing_here"
    model = _Model(_reply())
    engine, state = await _implement(tmp_path, model, design)
    state.update(await engine._node_execute(dict(state)))
    assert state["exec_result"]["returncode"] == af.FI_FAULT_EXIT
    calls = len(model.prompts)
    patch = await engine._node_execute_reflect(dict(state))
    assert len(model.prompts) == calls  # no model call
    said = patch["exec_result"]["cannot_repair"]
    assert "cannot compute a formula of the plan" in said and "nothing_here" in said
    assert "plan.md" in patch["exec_result"]["cannot_repair_advice"]
    assert engine._route_after_execute_reflect({**state, **patch}) == "stuck"
    assert (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == af.EXPERIMENT


def _fis_quest(engine: Engine, design: dict[str, Any] = DESIGN) -> Path:
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text(skeleton.frame_source(design, stochastic=True), encoding="utf-8")
    af.ensure_files(code, design)
    return code


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
    "run_plain": "dab51e372647d8db", "run_framed": "ad084c60d61b9dde",
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
