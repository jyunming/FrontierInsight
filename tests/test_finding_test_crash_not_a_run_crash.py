"""A finding's own test that crashes inside the code it called on made-up data (a stub table, a fixture) is a fact about
the TEST, never about the study's run: after a successful run it never sends the quest back to the run step
(``repair_crash``), never ends in ``stuck``, and the results stay. The test is asked for once more as an assertion about
values; when nothing usable comes back the finding is unshown and disclosed. Only the study's own run (its exit code and
results) decides a crash repair. Real tiny files and the real test runner; no model."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import code_review, paper_facts
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


@pytest.mark.asyncio
async def test_the_code_review_never_sends_a_finding_test_crash_to_the_run_repair(tmp_path: Path) -> None:
    engine = _crashing(tmp_path)
    assert not hasattr(engine, "_review_crash_run") and not hasattr(engine, "_gate_crash_repair")


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
