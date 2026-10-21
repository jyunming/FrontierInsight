"""A search whose objective never moves goes back to the plan once (docs/superpowers/specs/2026-10-21-flat-objective-
redesign-design.md).

A live search spent its whole budget on an objective that was the same at every design it tried: the plan's way of
describing the design could not reach the region where the objective moves, and FI never told the plan what it had
measured. Now the search's fact (core/optimise.py) goes to the evidence gate, which has plan.md rewritten from it once;
the design step takes the rewritten plan as a change FI approves itself, made after results were seen. Neutral toy topic
only (a heat sink).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import plan
from core.engine import Engine
from tests.test_plan_optimisation import EXTRA, HEAT_SINK, _NO_OBJECTIONS, _engine as _plan_engine

TOPIC = "Find the fin spacing that minimises the base temperature"

FACT: dict[str, Any] = {"designs": 50, "objective": 0.0, "feasible": True, "stayed": ["peak_margin"],
                        "moved": ["mass"], "unsteady": ["runtime_s"], "stopped": True}


def _revised() -> dict[str, Any]:
    design = copy.deepcopy(HEAT_SINK)
    design["protocol"]["optimisation"]["design_variables"][0]["high"] = 9.5
    return design


class Model:
    """The plan's model: drafts the plan, then answers the request to rewrite it (``revise``: the plan text to return)."""

    def __init__(self, eng: Engine, revise: Any = None) -> None:
        self.eng, self.revise = eng, revise
        self.asked: list[str] = []
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat)})()

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        if "FI ran the search for the best design" in prompt:
            self.asked.append(prompt)
            return self.revise() if callable(self.revise) else str(self.revise)
        if "A measurement, or a search for the best design?" in prompt:
            return json.dumps({**HEAT_SINK, "plan": EXTRA})
        return _NO_OBJECTIONS


async def _planned(tmp_path: Path, revise: Any = None, max_iterations: int = 3) -> tuple[Engine, Model]:
    eng = _plan_engine(tmp_path, [])
    eng.config.engine.max_iterations = max_iterations
    model = Model(eng, revise)
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    return eng, model


def _flat_state(**more: Any) -> dict[str, Any]:
    return {"topic": TOPIC, "iteration": 0, "result_json": {}, "design": copy.deepcopy(HEAT_SINK),
            "exec_result": {"returncode": 1, "duration_s": 1.0, "timed_out": False, "stdout_tail": "",
                            "stderr_tail": "", "failed_script": None, "flat_output": "", "trial_checks": {},
                            "flat_objective": dict(FACT)}, **more}


@pytest.mark.asyncio
async def test_a_flat_search_goes_to_the_gate_and_the_plan_is_rewritten_once_with_the_fact(tmp_path: Path) -> None:
    eng, model = await _planned(tmp_path)
    text = plan.render(TOPIC, EXTRA, plan.normalize_design(_revised())[0])
    model.revise = text
    state = _flat_state()
    # Nothing in the script to repair: the run goes straight to the gate.
    assert await eng._node_execute_reflect(state) == {}
    assert eng._route_after_execute_reflect(state) == "stuck"
    verdict = eng._no_results_verdict(state)
    assert verdict is not None and verdict["redesign"] is True and not verdict.get("stuck")
    request = verdict["flat_request"]
    assert "first 50 designs" in request and "peak_margin" in request and "mass" in request
    assert "runtime_s" in request and "timings" in request
    assert "Keep every case" in request and "do not change what the objective measures" in request
    patch = await eng._node_evidence_gate(state)
    assert patch["evidence_assessment"]["route"] == "redesign"
    assert eng._route_after_evidence_gate({**state, **patch}) == "redesign"
    assert patch["iteration"] == 1 and patch["design_from_plan"] is True
    assert patch["flat_objective_redesigned"] is True and patch["flat_objective_fact"] == FACT
    assert len(model.asked) == 1, "the plan is asked once"
    rewritten = plan.parse(plan.plan_path(eng.quest_root).read_text(encoding="utf-8")).design
    assert rewritten["protocol"]["optimisation"]["design_variables"][0]["high"] == 9.5


@pytest.mark.asyncio
async def test_after_the_one_revision_or_with_no_iteration_left_the_stop_is_off(tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path, max_iterations=2)
    assert eng._flat_redesign_allowed({"iteration": 0}) is True
    assert eng._flat_redesign_allowed({"iteration": 1, "flat_objective_redesigned": True}) is False
    assert eng._flat_redesign_allowed({"iteration": 2}) is False


@pytest.mark.asyncio
async def test_an_unusable_rewrite_runs_the_search_again_with_the_stop_off(tmp_path: Path) -> None:
    eng, model = await _planned(tmp_path, revise="not a plan at all")
    before = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    state = _flat_state()
    patch = await eng._node_evidence_gate(state)
    assert patch["evidence_assessment"]["route"] == "repair_crash"
    assert eng._route_after_evidence_gate({**state, **patch}) == "repair_crash"
    assert patch["flat_objective_redesigned"] is True and "iteration" not in patch
    assert plan.plan_path(eng.quest_root).read_text(encoding="utf-8") == before
    assert not (eng.quest_root / "needs" / "STUCK.json").exists()
    assert model.asked, "the plan was asked"
