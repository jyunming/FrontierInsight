"""A design step does not end ``done`` with a search plan that has no search block, and a compute step given such a design
hands it back to the design step (``needs_earlier_step``) instead of stopping. Fake model only (the cooling-cup commission of
tests/test_callable_steps.py); the plan's search is made "lacking its block" by planting what ``_search_problems`` reports."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import audit_log, rerun_from
from core.engine import Engine
from tests.test_callable_steps import _config, _design, _receipt, _started

pytestmark = pytest.mark.timeout(1800)

LACKING = (True, ["its optimisation block could not be read: `protocol.optimisation.objective` is missing"], None)


def test_the_design_step_has_a_search_check() -> None:
    assert "search_ready" in rerun_from.STEP_CHECKS["design"] and "search_ready" not in rerun_from.STEP_CHECKS["compute"]


@pytest.mark.asyncio
async def test_a_design_step_with_a_search_plan_lacking_its_block_ends_stopped(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Before: the step ended `done` and the compute step only found out an hour later."""
    eng = await _design(tmp_path, monkeypatch)
    root = eng.quest_root
    assert _receipt(root)["ended"] == "done" and _receipt(root)["checks"]["search_ready"]["judged"] == "not_applicable"
    assert not (root / "needs" / "STUCK.json").exists()
    monkeypatch.setattr(Engine, "_search_problems", lambda self, state, design: LACKING)
    eng._step_finish({"design": {"study_type": "find_best_design", "protocol": {}}}, False)
    receipt = _receipt(root)
    assert receipt["ended"] == "stopped" and "a search for the best design" in receipt["stopped_because"]
    assert receipt["checks"]["search_ready"] == {"judged": "ran", "status": "not_ready"}
    stuck = json.loads((root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert stuck["kind"] == "no_findings" and stuck["problem"] == receipt["stopped_because"]
    assert audit_log.verify(root / ".fi" / "audit.jsonl").ok


@pytest.mark.asyncio
async def test_a_design_step_whose_search_can_start_still_ends_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Known negative: a search plan that has its block is not stopped."""
    eng = await _design(tmp_path, monkeypatch)
    monkeypatch.setattr(Engine, "_search_problems", lambda self, state, design: (True, [], None))
    eng._step_finish({"design": {"study_type": "find_best_design", "protocol": {}}}, False)
    receipt = _receipt(eng.quest_root)
    assert receipt["ended"] == "done" and receipt["checks"]["search_ready"] == {"judged": "ran", "status": "ready"}
    assert not (eng.quest_root / "needs" / "STUCK.json").exists()


@pytest.mark.asyncio
async def test_a_compute_step_given_a_search_plan_lacking_its_block_is_handed_back_to_the_design_step(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Before: the compute step stopped (`stopped`, no findings) after a minute."""
    design = await _design(tmp_path, monkeypatch)
    monkeypatch.setattr(Engine, "_search_problems", lambda self, state, d: LACKING)
    eng = Engine(_config(tmp_path, design.config.commission, {"name": "compute", "after": design.quest_root}, "compute"))
    await eng.run()
    root = eng.quest_root
    stuck = json.loads((root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert stuck["return_reason"] == "needs_earlier_step" and stuck["step"] == "design"
    assert "design step must write it" in stuck["problem"] and stuck["evidence"]["step"] == "design"
    receipt = _receipt(root)
    assert receipt["ended"] == "returned" and receipt["return_reason"] == "needs_earlier_step"
    assert "implement" in _started(root) and "design" not in _started(root)
