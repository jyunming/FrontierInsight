"""A shown fault's test that crashes inside the program at the evidence gate goes to the same crash repair as before the run
(``RunChecksMixin._gate_crash_repair``, route ``repair_crash`` -> ``execute``, where ``_review_crash_run`` sends it to the
repair of a failed run), within the existing budget (``run_checks.REVIEW_CRASHES`` a quest, and the run's repairs). Only
when that budget is spent is it recorded and not stopped. Real tiny files and the real test runner; no model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core import code_review, paper_facts, run_checks
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
async def test_a_crash_in_the_program_at_the_gate_goes_to_the_crash_repair(tmp_path: Path) -> None:
    engine = _crashing(tmp_path)
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and verdict.get("repair_crash") is True and not verdict.get("stuck")
    assert "code/experiment.py:2" in verdict["rationale"]
    crash = code_review.load(engine.quest_root)["issues"]["F-1"]["crash"]
    assert crash["at"] == "code/experiment.py:2" and not crash.get("sent") and crash["at_gate"] is True
    # back in the run step, the same crash repair as before the run takes it
    sent = await engine._review_crash_run({"exec_reflect_iter": 0})
    assert sent is not None and sent["exec_result"]["returncode"] == 1
    assert "code/experiment.py:2" in sent["exec_result"]["stderr_tail"]
    assert code_review.load(engine.quest_root)["crashes_sent"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("spent", ["crashes", "repairs"])
async def test_with_the_budget_spent_it_is_recorded_and_stops_nothing(tmp_path: Path, spent: str) -> None:
    engine = _crashing(tmp_path)
    state: dict[str, Any] = dict(STATE)
    if spent == "crashes":
        record = code_review.load(engine.quest_root)
        record["crashes_sent"] = run_checks.REVIEW_CRASHES
        code_review.save(engine.quest_root, record)
    else:
        state["exec_reflect_iter"] = engine.config.engine.exec_reflect_max_iterations
    assert await engine._shown_fault_verdict(state) is None
    issue = code_review.load(engine.quest_root)["issues"]["F-1"]
    assert issue["gate_error"]["place"] == "program" and issue["gate_error"]["at"] == "code/experiment.py:2"
    # ... and the paper must say so: nothing shows the fault fixed, and the reading's own list is no help here (it has no
    # record of the version that ran), so the item is read from the issue itself.
    [item] = [i for i in paper_facts.required_items(engine.quest_root) if i.kind == paper_facts.CODE_FINDING]
    assert item.id == "finding-f-1"
    assert "dose_to_size is set to a typed-in value" in item.sentence
    assert "the test no longer runs" in item.sentence and "not confirmed that the fault is fixed" in item.sentence


@pytest.mark.asyncio
async def test_a_later_gate_where_the_test_passes_drops_the_error_and_the_item(tmp_path: Path) -> None:
    engine = _crashing(tmp_path)
    state: dict[str, Any] = {**STATE, "exec_reflect_iter": engine.config.engine.exec_reflect_max_iterations}
    assert await engine._shown_fault_verdict(state) is None
    assert code_review.load(engine.quest_root)["issues"]["F-1"].get("gate_error")
    (engine.quest_root / "code" / "experiment.py").write_text("def dose():\n    return 61.0\n", encoding="utf-8")
    assert await engine._shown_fault_verdict(state) is None
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


def test_the_route_goes_back_to_the_run_step(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    assert engine._route_after_evidence_gate({"evidence_assessment": {"route": "repair_crash"}}) == "repair_crash"
    edges = {(e.source, e.target) for e in engine._build_graph().compile().get_graph().edges}
    assert ("evidence_gate", "execute") in edges
