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

from core import frozen_protocol, plan
from core.engine import Engine
from tests.plan_parts_helpers import parts_of
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
            reply = self.revise() if callable(self.revise) else str(self.revise)
            # FI asks for the parts of how the design is described and searched, never for the file
            return parts_of(reply, "optimisation") if reply.startswith("# Plan") else reply
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
    verdict = eng._flat_objective_verdict(state)
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


# --- the design step (the change FI approves) ---------------------------------------------------------------------


class Paused(Exception):
    """Stands in for LangGraph's GraphInterrupt."""


def _no_pause(eng: Engine) -> None:
    def fake(**kwargs: Any) -> None:
        raise Paused(kwargs["kind"])

    eng._pause_for_human = fake  # type: ignore[method-assign]


def _rewrite_plan_file(eng: Engine) -> None:
    plan.plan_path(eng.quest_root).write_text(plan.render(TOPIC, EXTRA, plan.normalize_design(_revised())[0]),
                                              encoding="utf-8")


def _after_gate(eng: Engine) -> dict[str, Any]:
    first = plan.load_design(eng.quest_root)[0]
    return {"topic": TOPIC, "iteration": 1, "design": first, "design_from_plan": True, "flat_objective_redesigned": True,
            "flat_objective_fact": dict(FACT), "result_json": {},
            "exec_result": _flat_state()["exec_result"],
            "design_history": [{"revision": 0, "iteration": 0, "post_hoc": False, "after_results": False,
                                "results_seen": False, "reason": "initial design", "hypothesis": first["hypothesis"]}]}


@pytest.mark.asyncio
async def test_the_rewritten_plan_is_a_change_fi_approves_recorded_as_made_after_results(tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path)
    eng._freeze_protocol_if_due({"topic": TOPIC, "iteration": 0})
    assert frozen_protocol.load(eng.quest_root) is not None
    raw = eng._raw_root()
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "optimisation_ledger.jsonl").write_text('{"n": 1}\n', encoding="utf-8")
    state = _after_gate(eng)
    _rewrite_plan_file(eng)
    _no_pause(eng)
    patch = await eng._node_design(state)
    record = json.loads((eng.quest_root / "needs" / "PROTOCOL_AMENDMENT_1.json").read_text(encoding="utf-8"))
    assert record["prespecified"] is False and record["results_seen_before_change"] is True
    assert record["approved_by"].startswith("FI:") and "first 50 designs" in record["reason"]
    now = frozen_protocol.load(eng.quest_root)
    assert now["protocol"]["optimisation"]["design_variables"][0]["high"] == 9.5
    assert patch["design"]["protocol"]["optimisation"]["design_variables"][0]["high"] == 9.5
    assert (eng.quest_root / "archive" / "run_1" / "raw" / "optimisation_ledger.jsonl").is_file(), \
        "the first design's evaluations are kept"
    # The paper is told, and the evidence ladder names the change as a gap.
    told = frozen_protocol.disclosure(eng.quest_root)
    assert "post-hoc" in told and "first 50 designs" in told
    assert frozen_protocol.post_hoc(eng.quest_root)
    entry = patch["design_history"][-1]
    assert entry["results_seen"] is True and "first 50 designs" in entry["reason"]


@pytest.mark.asyncio
async def test_in_the_explore_stage_there_is_no_amendment_but_the_history_says_after_results(tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path)
    assert frozen_protocol.load(eng.quest_root) is None
    state = _after_gate(eng)
    _rewrite_plan_file(eng)
    _no_pause(eng)
    patch = await eng._node_design(state)
    assert not (eng.quest_root / "needs" / "PROTOCOL_AMENDMENT_1.json").exists()
    assert patch["design"]["protocol"]["optimisation"]["design_variables"][0]["high"] == 9.5
    entry = patch["design_history"][-1]
    assert entry["results_seen"] is True and "rewritten once from what FI measured" in entry["reason"]


@pytest.mark.asyncio
async def test_the_rewritten_plan_is_still_held_to_the_topic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, _model = await _planned(tmp_path)
    eng._freeze_protocol_if_due({"topic": TOPIC, "iteration": 0})
    seen: list[Any] = []
    held = copy.deepcopy(_revised())
    held["protocol"]["optimisation"]["design_variables"][0]["high"] = 9.0

    async def hold(state: Any, design: Any, plan_sha: str) -> tuple[Any, str]:
        seen.append(copy.deepcopy(design))
        return held, plan_sha

    monkeypatch.setattr(eng, "_hold_the_design_to_the_topic", hold)
    _rewrite_plan_file(eng)
    _no_pause(eng)
    patch = await eng._node_design(_after_gate(eng))
    assert seen and seen[0]["protocol"]["optimisation"]["design_variables"][0]["high"] == 9.5
    assert patch["design"]["protocol"]["optimisation"]["design_variables"][0]["high"] == 9.0
    assert frozen_protocol.load(eng.quest_root)["protocol"]["optimisation"]["design_variables"][0]["high"] == 9.0, \
        "the protocol FI approves is the one held to the topic"


def test_the_plan_asks_for_the_parts_of_an_objective() -> None:
    from core.engine import _STUDY_TYPE_DIRECTIVE

    assert "returns each part as a result of its own" in _STUDY_TYPE_DIRECTIVE


# --- the rewrite is checked before it is taken (the frozen protocol cannot be repaired after the change) -----------


def _text(design: dict[str, Any]) -> str:
    return plan.render(TOPIC, EXTRA, plan.normalize_design(design)[0])


@pytest.mark.asyncio
async def test_a_rewrite_that_eases_a_check_or_a_limit_has_them_put_back(tmp_path: Path) -> None:
    eased = _revised()
    eased["protocol"]["oracles"][0]["tolerance"] = 0.5
    eased["protocol"]["optimisation"]["constraints"][0]["limit"] = "<= 500"
    eng, _model = await _planned(tmp_path, revise=_text(eased))
    patch = await eng._node_evidence_gate(_flat_state())
    assert patch["evidence_assessment"]["route"] == "redesign"
    now = plan.parse(plan.plan_path(eng.quest_root).read_text(encoding="utf-8")).design["protocol"]
    assert now["oracles"][0]["tolerance"] == 0.01, "the check is held to its tolerance"
    assert now["optimisation"]["constraints"][0]["limit"] == "<= 120", "the limit is held"
    assert now["optimisation"]["design_variables"][0]["high"] == 9.5, "the design itself may change"
    assert "FI put it back as it was" in (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["nothing", "no_search"])
async def test_a_rewrite_that_changes_nothing_or_cannot_be_searched_is_not_taken(tmp_path: Path, change: str) -> None:
    design = copy.deepcopy(HEAT_SINK) if change == "nothing" else _revised()
    if change == "no_search":
        del design["protocol"]["optimisation"]["evaluation_budget"]
    eng, _model = await _planned(tmp_path, revise=lambda: _text(design) if change == "nothing" else
                                 plan.plan_path(eng.quest_root).read_text(encoding="utf-8").replace(
                                     "evaluation_budget", "evaluation_budget_gone"))
    before = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    patch = await eng._node_evidence_gate(_flat_state())
    assert patch["evidence_assessment"]["route"] == "repair_crash" and patch["flat_objective_redesigned"] is True
    assert "first design's search runs again" in patch["evidence_assessment"]["rationale"]
    assert plan.plan_path(eng.quest_root).read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_what_the_topic_asks_for_is_judged_on_the_rewritten_protocol(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    eng, _model = await _planned(tmp_path, revise=lambda: _text(_revised()))
    before = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    judged: list[Any] = []

    async def gaps(req: Any, design: Any, protocol: Any = None) -> tuple[list[dict[str, Any]], list[str]]:
        judged.append(protocol)
        return [{"name": "fin count", "sentence": "the fin count is not varied"}], []

    monkeypatch.setattr(eng, "_topic_requirements_read", lambda: {"items": [{"name": "fin count"}]})
    monkeypatch.setattr(eng, "_topic_gaps", gaps)
    monkeypatch.setattr("core.engine._topic_req.is_empty", lambda req: False)
    patch = await eng._node_evidence_gate(_flat_state())
    assert judged and judged[0]["optimisation"]["design_variables"][0]["high"] == 9.5, "the rewritten one is judged"
    assert patch["evidence_assessment"]["route"] == "repair_crash"
    assert "dropped what the topic asks for" in patch["evidence_assessment"]["rationale"]
    assert plan.plan_path(eng.quest_root).read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_a_flat_search_is_a_run_whose_results_were_seen(tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path)
    state = _after_gate(eng)
    assert eng._run_gave_no_results(state) is False
    changed = {**state["design"], "hypothesis": "a finer description of the fins lets the temperature move"}
    assert eng._keep_hypothesis_after_an_empty_run(state, changed)["hypothesis"] == changed["hypothesis"]


@pytest.mark.asyncio
async def test_with_the_protocol_frozen_the_search_s_own_judges_are_put_back_and_the_plan_shows_them(
        tmp_path: Path) -> None:
    eased = _revised()
    block = eased["protocol"]["optimisation"]
    block["objective"]["direction"] = "maximise" if block["objective"]["direction"] == "minimise" else "minimise"
    block["improvement_tolerance"] = {"value": 99, "mode": "absolute"}
    eased["protocol"]["oracles"][0]["tolerance"] = 0.5
    eng, _model = await _planned(tmp_path, revise=_text(eased))
    eng._freeze_protocol_if_due({"topic": TOPIC, "iteration": 0})
    frozen = frozen_protocol.load(eng.quest_root)["protocol"]
    patch = await eng._node_evidence_gate(_flat_state())
    assert patch["evidence_assessment"]["route"] == "redesign"
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    now = plan.parse(text).design["protocol"]
    assert now["optimisation"]["objective"]["direction"] == frozen["optimisation"]["objective"]["direction"]
    assert now["optimisation"].get("improvement_tolerance") == frozen["optimisation"].get("improvement_tolerance")
    assert now["oracles"][0]["tolerance"] == frozen["oracles"][0]["tolerance"]
    assert now["optimisation"]["design_variables"][0]["high"] == 9.5
    assert plan.refresh_optimisation_section(plan.refresh_model_section(text)) == text, "the plan shows the block"


@pytest.mark.asyncio
async def test_a_topic_check_that_fails_leaves_the_plan_as_it_was(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    eng, _model = await _planned(tmp_path, revise=lambda: _text(_revised()))
    before = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")

    async def broken(*a: Any, **k: Any) -> Any:
        raise RuntimeError("the model did not answer")

    monkeypatch.setattr(eng, "_topic_requirements_read", lambda: {"items": [{"name": "fin count"}]})
    monkeypatch.setattr(eng, "_topic_gaps", broken)
    monkeypatch.setattr("core.engine._topic_req.is_empty", lambda req: False)
    patch = await eng._node_evidence_gate(_flat_state())
    assert patch["evidence_assessment"]["route"] == "repair_crash"
    assert "could not be checked against the topic" in patch["evidence_assessment"]["rationale"]
    assert plan.plan_path(eng.quest_root).read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_the_fact_is_used_once_by_the_design_step(tmp_path: Path) -> None:
    eng, _model = await _planned(tmp_path)
    _rewrite_plan_file(eng)
    _no_pause(eng)
    patch = await eng._node_design(_after_gate(eng))
    assert patch["flat_objective_fact"] == {}
