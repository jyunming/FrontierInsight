"""A finding's own test that crashes inside the code it called on made-up data (a stub table, a fixture) is a fact about
the TEST, never about the study's run: after a successful run it never sends the quest back to the run step
(``repair_crash``), never ends in ``stuck``, and the results stay. The test is asked for once more as an assertion about
values; when nothing usable comes back the finding is unshown and disclosed. Only the study's own run (its exit code and
results) decides a crash repair. Real tiny files and the real test runner; no model."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from core import code_findings, code_review, finding_repair, paper_facts
from core.engine import Engine
from tests.test_shown_finding_gate import STATE, _engine, _issue, _record

pytestmark = pytest.mark.usefixtures("this_python")

CRASHES = "def dose():\n    return {}['missing']\n"
CALLS = "import experiment\n\n\ndef test_x():\n    assert experiment.dose() != 60.0\n"


def _crashing(tmp_path: Path) -> Engine:
    engine = _engine(tmp_path)
    (engine.quest_root / "code" / "experiment.py").write_text(CRASHES, encoding="utf-8")
    _record(engine, _issue(), test=CALLS)
    return engine


@pytest.mark.asyncio
async def test_a_finding_test_crash_in_the_code_after_a_good_run_is_no_run_crash(tmp_path: Path) -> None:
    engine = _crashing(tmp_path)
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict is None, "no repair_crash, no stuck: the study's run was fine"
    issue = code_review.load(engine.quest_root)["issues"]["F-1"]
    assert issue["gate_error"]["place"] == "program" and issue["gate_error"]["at"] == "code/experiment.py:2"
    assert "crash" not in issue, "nothing is kept for a crash repair"
    # the finding is unshown, and the paper says so
    [item] = [i for i in paper_facts.required_items(engine.quest_root) if i.kind == paper_facts.CODE_FINDING]
    assert "not confirmed that the fault is fixed" in item.sentence


class _Replies(dict):
    """The model's answers to a request for a test, one after the other."""

    def __init__(self, texts: list[str]) -> None:
        super().__init__()
        self.texts = list(texts)

    def __getitem__(self, key: str) -> str:
        return self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]


def _reviewed(tmp_path: Path, replies: list[str]) -> tuple[Path, dict[str, Any], dict[str, Any], Any]:
    """A quest whose code crashes when called, one result-changing finding about it, and the code review's test loop."""
    from tests.test_finding_v2_replay import _env

    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "experiment.py").write_text(CRASHES, encoding="utf-8")
    issue = {"id": "F-9", "changes_result": True, "demonstrated": False, "outcome": "open", "plan_conflict": False,
             "readings": [{"n": 1, "status": "open"}], "severity": "high", "first_seen": 1, "file": "experiment.py",
             "function": "dose", "text": "the dose is a typed-in value", "tests": 0}
    record: dict[str, Any] = {"issues": {"F-9": issue}}
    return root, record, issue, _env(root, record, _Replies(replies))  # type: ignore[arg-type]


FENCED_CALL = "def test_x():\n    import experiment\n    assert experiment.dose() != 60.0\n"


def test_a_review_test_that_raises_inside_the_code_shows_nothing_and_is_no_run_crash(tmp_path: Path) -> None:
    root, record, issue, env = _reviewed(tmp_path, [FENCED_CALL])
    shown = asyncio.run(finding_repair.demonstrate(env, issue, "experiment.py"))
    assert shown is False and issue["demonstrated"] is False and issue["outcome"] == code_findings.NOT_DEMONSTRATED
    assert "crash" not in issue, "nothing is kept for a crash repair"
    assert issue["errors"] and all(e["place"] == "program" for e in issue["errors"])
    assert issue["tests"] == finding_repair.TEST_CALLS, "the bounded tries were all used, asking for another test each time"
    assert not code_findings.shown_open(record), "it stops nothing"
    # no part of the engine takes it for the study's run: the gate on it neither routes nor stops
    code_review.save(root, {**record, "final": True, "version": code_review.version(root / "code")})
    from tests.test_shown_finding_gate import _engine

    engine = _engine(tmp_path / "e")
    engine.quest_root = root
    assert asyncio.run(engine._shown_fault_verdict(STATE)) is None


def test_a_replacement_test_that_fails_by_assertion_is_still_recognised(tmp_path: Path) -> None:
    other = "def test_x():\n    import experiment\n    assert experiment.other() != 60.0\n"
    root, record, issue, env = _reviewed(tmp_path, [FENCED_CALL, other])
    (root / "code" / "experiment.py").write_text(CRASHES + "\n\ndef other():\n    return 60.0\n", encoding="utf-8")
    shown = asyncio.run(finding_repair.demonstrate(env, issue, "experiment.py"))
    assert shown is True and issue["demonstrated"] is True and issue["failure_kind"] == code_findings.ASSERTION
    assert issue["errors"] and issue["errors"][0]["place"] == "program", "the first try's crash is kept, and is no crash route"
    assert "crash" not in issue and code_findings.shown_open(record) == [issue]


def _reviewed_then_changed(tmp_path: Path, replies: list[str], *, problem: dict[str, Any] | None = None,
                           code: str | None = None) -> tuple[Path, dict[str, Any]]:
    """The review's tests run (``replies``), the reading is saved for this code, and then the code is changed."""
    root, record, issue, env = _reviewed(tmp_path, replies)
    if code is not None:
        (root / "code" / "experiment.py").write_text(code, encoding="utf-8")
    asyncio.run(finding_repair.demonstrate(env, issue, "experiment.py"))
    found = {"fid": "F-9", "text": issue["text"], "status": "different", **(problem or {})}
    code_review.save(root, {**record, "final": True, "version": code_review.version(root / "code"), "problems": [found]})
    (root / "code" / "experiment.py").write_text(
        (code or CRASHES) + "\n\ndef changed_after_the_review():\n    return 1\n", encoding="utf-8")
    return root, issue


def _findings(root: Path) -> list[Any]:
    return [i for i in paper_facts.required_items(root) if i.kind == paper_facts.CODE_FINDING]


def test_an_unshown_finding_stays_in_the_paper_when_the_code_changes_after_the_review(tmp_path: Path) -> None:
    root, issue = _reviewed_then_changed(tmp_path, [FENCED_CALL])
    assert issue["errored_only"] is True
    [item] = _findings(root)
    assert "a possible problem with the dose is a typed-in value" in item.sentence
    assert "every test written for it failed to run" in item.sentence and "not checked again" in item.sentence
    assert "can change a reported result" not in item.sentence


def test_a_later_test_that_ran_and_passed_is_not_said_to_have_failed_to_run(tmp_path: Path) -> None:
    passing = "def test_x():\n    import experiment\n    assert experiment.other() == 60.0\n"
    root, issue = _reviewed_then_changed(
        tmp_path, [FENCED_CALL, passing], code=CRASHES + "\n\ndef other():\n    return 60.0\n")
    assert issue["errors"] and issue["errored_only"] is False
    assert _findings(root) == [], "a test that ran and passed is not 'no test could show it'"


def test_a_finding_about_a_function_the_run_never_calls_stays_out(tmp_path: Path) -> None:
    calls_helper = FENCED_CALL.replace("dose()", "helper()")
    root, issue = _reviewed_then_changed(
        tmp_path, [calls_helper], problem={"file_rel": "experiment.py", "function": "helper"},
        code="def helper():\n    return {}['missing']\n\n\ndef dose():\n    return 59.0\n\n\nprint(dose())\n")
    assert issue["errored_only"] is True
    # the same finding about a function the run does call is said: the exclusion is the only difference
    record = code_review.load(root)
    record["problems"][0]["function"] = "dose"
    code_review.save(root, record)
    assert len(_findings(root)) == 1
    record["problems"][0]["function"] = "helper"
    code_review.save(root, record)
    assert _findings(root) == []


def test_a_finding_the_runs_results_contradict_stays_out(tmp_path: Path) -> None:
    root, issue = _reviewed_then_changed(tmp_path, [FENCED_CALL], problem={"contradicted": True})
    assert _findings(root) == []


@pytest.mark.asyncio
async def test_a_later_gate_where_the_test_passes_drops_the_error_and_the_item(tmp_path: Path) -> None:
    engine = _crashing(tmp_path)
    assert await engine._shown_fault_verdict(STATE) is None
    assert code_review.load(engine.quest_root)["issues"]["F-1"].get("gate_error")
    (engine.quest_root / "code" / "experiment.py").write_text("def dose():\n    return 61.0\n", encoding="utf-8")
    assert await engine._shown_fault_verdict(STATE) is None
    assert "gate_error" not in code_review.load(engine.quest_root)["issues"]["F-1"]
    assert [i for i in paper_facts.required_items(engine.quest_root) if i.kind == paper_facts.CODE_FINDING] == []


@pytest.mark.asyncio
async def test_a_fault_that_stops_the_quest_is_not_held_up_by_a_crash_elsewhere(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    (engine.quest_root / "code" / "experiment.py").write_text(
        "def dose():\n    return 60.0\n\n\ndef other():\n    return {}['missing']\n", encoding="utf-8")
    _record(engine, _issue("F-1", first_seen=1), _issue("F-2", first_seen=2, function="other"))
    (engine.quest_root / ".fi" / "finding_tests" / "F-2_1.py").write_text(
        "import experiment\n\n\ndef test_x():\n    assert experiment.other()\n", encoding="utf-8")
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and verdict["stuck_reason"] == "shown_fault_open"


def test_a_real_study_crash_still_goes_to_the_run_repair(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    crashed = {"exec_result": {"returncode": 1, "stderr_tail": "Traceback ... KeyError"}, "result_json": {},
               "exec_reflect_iter": 0}
    assert engine._route_after_execute_reflect(crashed) == "retry"  # type: ignore[arg-type]
    # and the route the gate can still ask for (a search run again) goes back to the run step
    assert engine._route_after_evidence_gate({"evidence_assessment": {"route": "repair_crash"}}) == "repair_crash"
    edges = {(e.source, e.target) for e in engine._build_graph().compile().get_graph().edges}
    assert ("evidence_gate", "execute") in edges


@pytest.mark.asyncio
async def test_a_study_script_that_exits_nonzero_is_repaired_and_run_again(tmp_path: Path) -> None:
    """The study's own run decides a crash repair: execute -> execute_reflect (the repair stubbed) -> the graph retries."""
    import json
    from unittest.mock import AsyncMock

    from core.execution import ExecutionResult
    from tests.test_engine_execute_retry import _count_real_calls, _mk_engine, _mock_execute_router

    eng = _mk_engine(tmp_path)
    (eng.quest_root / "code" / "experiment.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    failed = ExecutionResult(returncode=1, stdout="", stderr="Traceback ...\nKeyError: 'x'", duration_s=2.0)
    eng.executor.install = AsyncMock(return_value=ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1))
    eng.executor.execute = _mock_execute_router(failed)
    state: dict[str, Any] = {"deps": [], "design": {"hypothesis": "h"}, "code": "import sys\nsys.exit(1)\n"}
    state.update(await eng._node_execute(state))
    assert _count_real_calls(eng.executor.execute) == 1
    assert state["exec_result"]["returncode"] == 1 and not state.get("result_json")
    fixed = "print('RESULT_JSON: {}')\n"
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value=json.dumps(
        {"code": fixed, "deps": [], "patch_summary": "fixed the lookup", "give_up_reason": ""}))})()
    state.update(await eng._node_execute_reflect(state))
    assert state["exec_reflect_iter"] == 1 and (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == fixed
    assert eng._route_after_execute_reflect(state) == "retry"  # type: ignore[arg-type]


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.asyncio
async def test_a_study_script_that_exits_nonzero_runs_again_once_repaired_in_a_real_venv(tmp_path: Path) -> None:
    """The same path with the real VenvExecutor: only the model's repair is stubbed; the repaired script's second run is seen."""
    import json
    from unittest.mock import AsyncMock

    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig

    cfg = Config(topic="a small study", title="small", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False, execute_replicates=1),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=120), knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "out"))
    eng = Engine(cfg)
    for sub in ("code", "figures"):
        (eng.quest_root / sub).mkdir(parents=True, exist_ok=True)
    script = eng.quest_root / "code" / "experiment.py"
    script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    state: dict[str, Any] = {"deps": [], "design": {"hypothesis": "h"}, "code": script.read_text(encoding="utf-8")}
    state.update(await eng._node_execute(state))
    assert state["exec_result"]["returncode"] == 1 and not state.get("result_json")
    fixed = "print('RESULT_JSON: {\"x\": 1}')\n"
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value=json.dumps(
        {"code": fixed, "deps": [], "patch_summary": "fixed", "give_up_reason": ""}))})()
    state.update(await eng._node_execute_reflect(state))
    assert eng._route_after_execute_reflect(state) == "retry"  # type: ignore[arg-type]
    assert script.read_text(encoding="utf-8") == fixed
    state.update(await eng._node_execute(state))  # the second execution, of the repaired script
    assert state["exec_result"]["returncode"] == 0 and state["result_json"] == {"x": 1}
