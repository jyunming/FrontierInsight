"""One step of a commissioned study run on its own (``step:`` in the config, core/rerun_from.py ``ALONE``).

``design`` runs the graph from the start and stops after the design (under a commission the steps before it do no work
of their own); ``compute`` starts a fresh quest from the state a finished design step left, runs the code and its
repairs, and stops before the analysis. Each leaves a receipt (needs/STEP_RECEIPT.json) naming the sha256 of the receipt
before it; a route that would send the study back to an earlier step hands it back instead (``needs_earlier_step``).

Fake model only. The topic is a made-up cooling cup.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from core import audit_log, rerun_from
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig, StepConfig
from tests.test_engine_smoke import _fake_response_for
from tests.test_self_correction_e2e import _BAD_EXPERIMENT, _GOOD_EXPERIMENT

pytestmark = pytest.mark.timeout(1800)  # each compute step builds a real venv


def _commission(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "commission.json"
    path.write_text(json.dumps({
        "question": "How fast does a made-up cup of water cool on a bench?",
        "scope": "one cup, three starting temperatures",
        "limits": {"hours": 1, "model_calls": 80},
        "known_values": [], "no_known_values": True,
        "success": [{"condition": "the cooling curve falls with time", "verify": "read the curve"}],
        "failure": [{"condition": "the curve does not fall", "verify": "read the curve"}],
        "doubts": [],
    }), encoding="utf-8")
    return path


def _config(tmp_path: Path, commission: Path, step: dict[str, Any] | None, name: str) -> Config:
    return Config(
        topic="", title=name, commission=commission, step=StepConfig(**step) if step else None,
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, exec_reflect_max_iterations=3),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / name),
    )


def _fake(bad_first: bool = False) -> Any:
    async def chat(self: Any, messages: list[dict[str, str]], **_kw: Any) -> str:
        prompt = messages[-1]["content"]
        head = prompt.lstrip().splitlines()[0] if prompt.strip() else ""
        if bad_first and "Implementation" in head:
            return json.dumps({"code": _BAD_EXPERIMENT, "deps": []})
        if bad_first and "Execute-Reflect" in head:
            return json.dumps({"code": _GOOD_EXPERIMENT, "deps": ["matplotlib"], "patch_summary": "fixed the index",
                               "give_up_reason": ""})
        return _fake_response_for(prompt)

    return chat


def _trace(root: Path) -> list[dict[str, Any]]:
    return audit_log.read(root / ".fi" / "audit.jsonl")


def _started(root: Path) -> list[str]:
    return [e["node"] for e in _trace(root) if e.get("kind") == "node_started"]


def _receipt(root: Path) -> dict[str, Any]:
    return json.loads((root / rerun_from.RECEIPT).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _design(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from core.engine import Engine

    monkeypatch.setattr("core.engine.LLMClient.chat", _fake())
    eng = Engine(_config(tmp_path, _commission(tmp_path / "order"), {"name": "design"}, "design"))
    await eng.run()
    return eng


@pytest.mark.asyncio
async def test_the_design_step_alone_writes_plan_md_and_a_receipt_and_stops_after_the_design(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = await _design(tmp_path, monkeypatch)
    root = eng.quest_root
    assert _started(root) == list(rerun_from.ALONE["design"]), "only the design step's nodes ran, each once"
    assert await eng._checkpoint_next() == (), "the checkpoint ends where the step ends"
    assert audit_log.verify(root / ".fi" / "audit.jsonl").ok and _trace(root)[-1]["kind"] == "quest_finalized"
    assert (root / ".fi" / "cost.jsonl").is_file() and (root / "plan.md").is_file()
    receipt = _receipt(root)
    assert receipt["step"] == "design" and receipt["ended"] == "done" and receipt["previous_receipt_sha256"] is None
    assert receipt["outputs"]["plan.md"] == _sha(root / "plan.md") and receipt["design_sha256"]
    assert receipt["checks"]["design_audit"]["judged"] == "ran"
    assert receipt["checks"]["known_values"] == {"judged": "not_applicable", "why": "the commission gives no known values"}
    assert not (root / "needs" / "EVIDENCE.json").exists(), "a step alone claims no evidence level"
    from core import quality_report

    assert quality_report.build(root, running=False)["outcome"]["line"].startswith(
        "The design step was run on its own and finished; its receipt")
    receipt_events = [e for e in _trace(root) if e.get("kind") == "step_receipt"]
    assert receipt_events and receipt_events[-1]["sha256"] == _sha(root / rerun_from.RECEIPT)
    rows = [line for line in (root / ".fi" / "model_calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert receipt["model_calls"] == len(rows) > 0, "the receipt counts the step's model calls"
    assert set(receipt["handed_on"]) >= {"plan.md", "needs/receipts/design_audit.json"}
    # A design whose design check did not pass still finished (the caller decides), and its report says which check.
    (root / rerun_from.RECEIPT).write_text(json.dumps({**receipt, "checks": {
        **receipt["checks"], "design_audit": {"judged": "ran", "status": "fail"}}}), encoding="utf-8")
    assert quality_report.build(root, running=False)["outcome"]["line"].startswith(
        "The design step was run on its own and finished with its design check not passed; its receipt")


@pytest.mark.asyncio
async def test_the_compute_step_alone_starts_from_the_design_and_its_repairs_stay_inside_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    design = await _design(tmp_path, monkeypatch)
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(bad_first=True))
    eng = Engine(_config(tmp_path, design.config.commission, {"name": "compute", "after": design.quest_root}, "compute"))
    await eng.run()
    root = eng.quest_root
    # Seeded as if the design had just finished: its router ran on the seed, no node of the design step ran here; the
    # crashed script was repaired inside the step (execute twice) and the step stopped before the analysis.
    assert _started(root) == ["implement_outline", "implement", "execute", "execute_reflect", "execute",
                              "execute_reflect", "improve"]
    assert await eng._checkpoint_next() == () and audit_log.verify(root / ".fi" / "audit.jsonl").ok
    receipt = _receipt(root)
    assert receipt["step"] == "compute" and receipt["ended"] == "done" and receipt["result"] == {"score": 0.5}
    assert receipt["previous_receipt_sha256"] == _sha(design.quest_root / rerun_from.RECEIPT)
    assert receipt["design_sha256"] == _receipt(design.quest_root)["design_sha256"], "the design the receipt named ran"
    assert receipt["inputs"]["plan.md"] == _receipt(design.quest_root)["outputs"]["plan.md"]
    assert "code/experiment.py" in receipt["outputs"] and "needs/FROZEN_PROTOCOL.json" in receipt["outputs"]
    assert receipt["checks"]["checks_before_the_run"]["judged"] == "ran"
    # The reader of the code was the model that wrote it: a check that did not read, not a fact of the study.
    assert receipt["checks"]["code_review"]["judged"] == "not_run" and receipt["checks"]["code_review"]["why"]
    # A second call goes on from the checkpoint: nothing is seeded or run again, and both receipts stay byte for byte.
    count, before, design_before = len(_started(root)), (root / rerun_from.RECEIPT).read_bytes(), _receipt(design.quest_root)
    await eng.run()
    assert len(_started(root)) == count and len([e for e in _trace(root) if e.get("kind") == "step_started"]) == 1
    assert (root / rerun_from.RECEIPT).read_bytes() == before
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake())
    await design.run()
    assert _receipt(design.quest_root) == design_before


@pytest.mark.asyncio
async def test_a_compute_step_stopped_part_way_goes_on_from_where_it_stopped(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    design = await _design(tmp_path, monkeypatch)
    real, calls = Engine._node_execute, {"n": 0}

    async def once(self: Any, state: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the machine went away")
        return await real(self, state)

    monkeypatch.setattr(Engine, "_node_execute", once)
    eng = Engine(_config(tmp_path, design.config.commission, {"name": "compute", "after": design.quest_root}, "compute"))
    with pytest.raises(RuntimeError, match="went away"):
        await eng.run()
    assert not (eng.quest_root / rerun_from.RECEIPT).exists()
    await Engine(eng.config, resume_quest_id=eng.quest_id).run()
    started = _started(eng.quest_root)
    assert started.count("implement_outline") == 1 and started.count("implement") == 1 and started.count("execute") == 2
    assert len([e for e in _trace(eng.quest_root) if e.get("kind") == "step_started"]) == 1
    assert _receipt(eng.quest_root)["ended"] == "done" and _receipt(eng.quest_root)["result"] == {"score": 0.987}


@pytest.mark.asyncio
async def test_a_design_step_held_at_its_plan_says_paused_and_then_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.config import PausesConfig
    from core.engine import Engine

    monkeypatch.setattr("core.engine.LLMClient.chat", _fake())
    cfg = _config(tmp_path, _commission(tmp_path / "order"), {"name": "design"}, "design")
    cfg.pauses = PausesConfig(**{**cfg.pauses.model_dump(), "plan": "ask"})
    eng = Engine(cfg)
    await eng.run()
    assert _receipt(eng.quest_root)["ended"] == "paused" and "design" not in _started(eng.quest_root)
    added = eng.quest_root / "inputs" / "notes.txt"  # a person adds to what the step is given while it waits
    added.write_text("the counter is open at eight\n", encoding="utf-8")
    await Engine(cfg, resume_quest_id=eng.quest_id).run()
    assert _receipt(eng.quest_root)["ended"] == "done" and _started(eng.quest_root)[-1] == "design"
    assert _receipt(eng.quest_root)["inputs"]["inputs/notes.txt"] == _sha(added), "the inputs are read at the step's end"


@pytest.mark.asyncio
async def test_a_script_that_still_fails_after_its_repairs_ends_the_step_with_no_result(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    design = await _design(tmp_path, monkeypatch)

    async def broken(self: Any, messages: list[dict[str, str]], **_kw: Any) -> str:
        head = messages[-1]["content"].lstrip().splitlines()[0]
        if "Implementation" in head or "Execute-Reflect" in head:
            return json.dumps({"code": _BAD_EXPERIMENT, "deps": [], "patch_summary": "tried", "give_up_reason": ""})
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", broken)
    eng = Engine(_config(tmp_path, design.config.commission, {"name": "compute", "after": design.quest_root}, "compute"))
    await eng.run()
    receipt = _receipt(eng.quest_root)
    # Whether to write the code again is the gate's, after the analysis (a later step): this step stopped, and says why.
    assert receipt["result"] is None and receipt["ended"] == "stopped", receipt["ended"]
    assert receipt["stopped_because"].startswith("the run gave no result: the script still failed after its repairs")
    from core import quality_report

    line = quality_report.build(eng.quest_root, running=False)["outcome"]["line"]
    assert line.startswith("The compute step was run on its own and stopped: the run gave no result"), line


@pytest.mark.asyncio
async def test_a_known_value_of_the_commission_is_checked_inside_the_compute_step(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine
    from tests.test_commission import _commission as _with_known_value
    from tests.test_engine_smoke import _FAKE_RESPONSES
    from tests.test_oracle_gate import _HEAD, _TAIL

    # The script answers the check FI writes from the commission's known value (0.25, tolerance 0.01) when asked.
    answers = _HEAD + """if os.environ.get("FI_ORACLE") == "1":
    print("ORACLE_JSON: " + json.dumps({"checks": [{"name": "drain constant", "value": 0.2502}]}))
    raise SystemExit(0)
""" + _TAIL

    async def chat(self: Any, messages: list[dict[str, str]], **_kw: Any) -> str:
        prompt = messages[-1]["content"]
        head = prompt.lstrip().splitlines()[0] if prompt.strip() else ""
        if prompt.startswith("# Compute a Known Value"):
            return "(no program)"  # the known value is left unconfirmed at the plan (tests/test_known_values.py)
        if "Experiment Design" in head:  # a plan with a protocol: FI adds the commission's check to it
            return json.dumps({**json.loads(_FAKE_RESPONSES["design"]), "protocol": {"runs_per_setting": 300}})
        if "Implementation" in head:
            return json.dumps({"code": answers, "deps": ["matplotlib"]})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", chat)
    order = _with_known_value(tmp_path / "order")
    design = Engine(_config(tmp_path, order, {"name": "design"}, "design"))
    await design.run()
    assert _receipt(design.quest_root)["checks"]["known_values"]["judged"] == "ran"
    eng = Engine(_config(tmp_path, order, {"name": "compute", "after": design.quest_root}, "compute"))
    eng.config.engine.execute_replicates, eng.config.engine.pilot_run = 1, False
    await eng.run()
    receipt = _receipt(eng.quest_root)
    oracle = json.loads((eng.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    assert oracle["attempts"][-1]["oracles"] == ["drain constant"], oracle
    assert oracle["status"] == "ok" and receipt["checks"]["known_answer_checks"] == {"judged": "ran", "status": "ok"}
    assert receipt["ended"] == "done" and receipt["previous_receipt_sha256"] == _sha(design.quest_root / rerun_from.RECEIPT)


@pytest.mark.asyncio
async def test_a_script_handed_on_with_the_design_is_run_as_it_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    design = await _design(tmp_path, monkeypatch)
    (design.quest_root / "code" / "experiment.py").write_text(_GOOD_EXPERIMENT, encoding="utf-8")
    eng = Engine(_config(tmp_path, design.config.commission, {"name": "compute", "after": design.quest_root}, "compute"))
    await eng.run()
    assert _started(eng.quest_root)[:2] == ["execute", "execute_reflect"], "no code was written"
    receipt = _receipt(eng.quest_root)
    assert receipt["inputs"]["code/experiment.py"] == _sha(design.quest_root / "code" / "experiment.py")
    assert receipt["result"] == {"score": 0.5}


@pytest.mark.asyncio
async def test_a_route_back_to_the_design_hands_the_study_back_as_needing_an_earlier_step(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    design = await _design(tmp_path, monkeypatch)
    # Planted: the run is read as stopped before the study, and the gate's own verdict asks for the design once more.
    monkeypatch.setattr(Engine, "_route_after_execute_reflect", lambda self, state: "stuck")
    monkeypatch.setattr(Engine, "_no_results_verdict", lambda self, state: {
        "verdict": "insufficient", "rationale": "the run measured nothing; tried once more from the design",
        "gaps": ["nothing measured"], "redesign": True})
    eng = Engine(_config(tmp_path, design.config.commission, {"name": "compute", "after": design.quest_root}, "compute"))
    await eng.run()
    root = eng.quest_root
    assert "design" not in _started(root) and _started(root)[-1] == "evidence_gate"
    stuck = json.loads((root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert stuck["return_reason"] == "needs_earlier_step" and stuck["step"] == "design"
    assert stuck["problem"].startswith("needs an earlier step: design") and stuck["evidence"]["route"] == "redesign"
    receipt = _receipt(root)
    assert receipt["ended"] == "returned" and receipt["return_reason"] == "needs_earlier_step"
    assert audit_log.verify(root / ".fi" / "audit.jsonl").ok


@pytest.mark.asyncio
async def test_a_compute_step_after_a_folder_whose_plan_changed_is_refused_before_any_model_call(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    design = await _design(tmp_path, monkeypatch)
    (design.quest_root / "plan.md").write_text("# a plan edited after its step ended\n", encoding="utf-8")

    async def no_call(*_a: Any, **_k: Any) -> str:
        raise AssertionError("no model call is made")

    monkeypatch.setattr("core.engine.LLMClient.chat", no_call)
    eng = Engine(_config(tmp_path, design.config.commission, {"name": "compute", "after": design.quest_root}, "compute"))
    with pytest.raises(ValueError, match="its plan.md is not the one its receipt names"):
        await eng.run()
    assert not (eng.quest_root / "plan.md").exists()


def test_the_step_key_needs_a_commission_and_compute_needs_the_design_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must name its `commission`"):
        Config(topic="t", step=StepConfig(name="design"))
    with pytest.raises(ValueError, match="`compute` needs it"):
        StepConfig(name="compute")
    with pytest.raises(ValueError):
        StepConfig(name="design", stop="later")  # type: ignore[call-arg]


def test_a_study_run_whole_builds_the_graph_it_always_did(tmp_path: Path) -> None:
    from langgraph.graph import StateGraph

    from core.engine import Engine

    eng = Engine(_config(tmp_path, _commission(tmp_path / "order"), None, "whole"))
    assert type(eng._build_graph()) is StateGraph


def test_a_route_out_of_a_step_is_back_only_when_it_leads_before_the_step() -> None:
    assert rerun_from.step_back_to("compute", "design") == "design"
    assert rerun_from.step_back_to("compute", "literature") == "design"
    assert rerun_from.step_back_to("compute", "analyze") == ""
    assert rerun_from.step_back_to("design", "implement_outline") == ""
    assert set(rerun_from.ALONE["compute"]) | set(rerun_from.ALONE["design"]) <= set(rerun_from.NODES)


@pytest.mark.asyncio
async def test_a_step_quest_goes_on_only_as_the_step_it_was_started_as(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.config import PausesConfig
    from core.engine import Engine

    design = await _design(tmp_path, monkeypatch)
    as_compute = _config(tmp_path, design.config.commission, {"name": "compute", "after": design.quest_root}, "design")
    with pytest.raises(ValueError, match="ran the design step, so it cannot go on as the compute step"):
        await Engine(as_compute, resume_quest_id=design.quest_id).run()
    with pytest.raises(ValueError, match="would go on at analyze, which is not part of the design step"):
        design._step_inside(("analyze",))
    # A whole study held at its plan is not resumed as one of its steps.
    whole = _config(tmp_path, _commission(tmp_path / "order2"), None, "whole")
    whole.pauses = PausesConfig(**{**whole.pauses.model_dump(), "plan": "ask"})
    eng = Engine(whole)
    await eng.run()
    whole_as_step = whole.model_copy(update={"step": StepConfig(name="design")})
    with pytest.raises(ValueError, match="was started as a whole study"):
        await Engine(whole_as_step, resume_quest_id=eng.quest_id).run()


@pytest.mark.asyncio
async def test_a_finished_step_quest_is_not_re_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    design = await _design(tmp_path, monkeypatch)
    before = _started(design.quest_root)
    with pytest.raises(ValueError, match="ran one step on its own and is not re-opened"):
        await Engine(design.config, resume_quest_id=design.quest_id).run(reopen=True)
    assert _started(design.quest_root) == before, "no node ran"


@pytest.mark.asyncio
async def test_a_compute_step_after_a_folder_whose_handed_on_record_changed_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    design = await _design(tmp_path, monkeypatch)
    audit = design.quest_root / "needs" / "receipts" / "design_audit.json"
    audit.write_text(audit.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    eng = Engine(_config(tmp_path, design.config.commission, {"name": "compute", "after": design.quest_root}, "compute"))
    with pytest.raises(ValueError, match="needs/receipts/design_audit.json changed since its receipt"):
        await eng.run()
    assert not _started(eng.quest_root)


@pytest.mark.asyncio
async def test_a_design_with_no_simulation_is_refused_by_the_compute_step_even_with_a_script(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    monkeypatch.setattr("core.engine.LLMClient.chat", _fake())
    cfg = _config(tmp_path, _commission(tmp_path / "order"), {"name": "design"}, "design")
    cfg.engine = cfg.engine.model_copy(update={"no_simulation": True})
    design = Engine(cfg)
    await design.run()
    assert _receipt(design.quest_root)["ended"] == "done"
    (design.quest_root / "code").mkdir(exist_ok=True)
    (design.quest_root / "code" / "experiment.py").write_text(_GOOD_EXPERIMENT, encoding="utf-8")
    eng = Engine(_config(tmp_path, cfg.commission, {"name": "compute", "after": design.quest_root}, "compute"))
    with pytest.raises(ValueError, match="is a study with no simulation"):
        await eng.run()
    assert not _started(eng.quest_root), "the script was not run"


def test_a_check_record_that_says_nothing_is_unreadable_not_ran(tmp_path: Path) -> None:
    rec = tmp_path / "check.json"
    for text in ("{not json", "{}", "[1, 2]", ""):
        rec.write_text(text, encoding="utf-8")
        assert rerun_from.judged(tmp_path, "check.json", None) == {"judged": "unreadable"}, text
    rec.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    assert rerun_from.judged(tmp_path, "check.json", None) == {"judged": "ran", "status": "ok"}


def test_a_redo_from_a_step_moves_the_old_step_receipt_aside(tmp_path: Path) -> None:
    (tmp_path / "needs").mkdir()
    (tmp_path / rerun_from.RECEIPT).write_text("{}", encoding="utf-8")
    (tmp_path / "plan.md").write_text("# plan\n", encoding="utf-8")
    where, moved = rerun_from.back_up(tmp_path, "plan")
    assert rerun_from.RECEIPT in moved and not (tmp_path / rerun_from.RECEIPT).exists()
    assert (where / rerun_from.RECEIPT).is_file()
