"""A commissioned study is handed back before it runs when FI's own evidence overturns its plan: a failing known-answer
check whose three numbers (the simulation's measured value, FI's independent recomputation and the expected value)
disagree, each judged by its own error; or a plan that still defines one quantity two ways after its model was asked
once. A study run directly goes on, the check marked unconfirmed, as before.

The topic of these tests is a made-up draining tank; no real model is called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import commission, oracle_challenge as ch, plan
from core.engine import Engine
from tests.test_commission import _commission
from tests.test_oracle_gate import _HEAD, _PROTOCOL, _TAIL, _cfg, _fake, _record

# Measured 0.734 (its trials agree to 0), expected 0.655 within 0.05, and an independent recomputation
# 0.699 +- 0.002: within the tolerance of both sides, yet each pair is many errors apart.
CHECK = {"name": "drain level at the reference time", "kind": "closed_form", "check": "the level at the reference time",
         "expected": 0.655, "tolerance": 0.05, "tolerance_mode": "absolute", "case": {"n": 4}, "measure": "level"}
JUDGED = {"name": CHECK["name"], "value": 0.734, "expected": 0.655, "passed_by_engine": False, "measured_by": "engine",
          "mc": {"verdict": "fail", "se": 0.0, "trials": 40}}
REFEREE = {"kind": "referee", "check": CHECK["name"], "outcome": "unsettled",
           "numeric": {"value": 0.699, "error": 0.002, "how": "a fine-step integration", "model": "model-b"}}


def _with(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**base, **changes}


# --- the rule itself --------------------------------------------------------------------------------------------------

def test_three_numbers_each_beyond_the_others_own_errors_disagree() -> None:
    found = ch.three_disagree(CHECK, JUDGED, REFEREE)
    assert found is not None
    assert found["numbers"]["plan"] == 0.655 and found["numbers"]["reference"]["error"] == 0.002
    assert found["gaps"]["reference_vs_plan"]["standard_errors"] == pytest.approx(22.0)
    assert found["gaps"]["simulation_vs_reference"]["standard_errors"] == pytest.approx(17.5)


@pytest.mark.parametrize("why,judged,referee", [
    ("the referee gave no error", JUDGED, _with(REFEREE, numeric={"value": 0.699, "how": "h"})),
    ("the measurement has no error", _with(JUDGED, mc={"verdict": "fail"}), REFEREE),
    ("the referee sided with the plan", JUDGED, _with(REFEREE, outcome="code")),
    ("the check passed", _with(JUDGED, passed_by_engine=True), REFEREE),
    ("no referee", JUDGED, None),
    ("the referee agrees with the simulation within their errors", JUDGED,
     _with(REFEREE, numeric={"value": 0.729, "error": 0.002})),
    ("a wide referee error covers both", JUDGED, _with(REFEREE, numeric={"value": 0.699, "error": 0.02})),
])
def test_nothing_is_found_when_an_error_is_unknown_or_two_numbers_agree(why: str, judged: dict[str, Any],
                                                                         referee: dict[str, Any] | None) -> None:
    assert ch.three_disagree(CHECK, judged, referee) is None, why


def test_a_check_that_is_zero_by_its_form_or_a_published_value_is_never_found() -> None:
    rule = _with(CHECK, kind="invariant", expected=0.0)
    assert ch.three_disagree(rule, _with(JUDGED, value=0.1), _with(REFEREE, numeric={"value": 0.05, "error": 1e-4})) is None
    assert ch.three_disagree(_with(CHECK, kind="published_value"), JUDGED, REFEREE) is None


def test_the_reason_follows_where_the_expected_value_came_from() -> None:
    given = {"known_values": [{"name": "Drain  Level at the reference time"}]}
    assert commission.mismatch_reason(CHECK, {}) == "plan_contradicts_sources"
    assert commission.mismatch_reason(_with(CHECK, name="drain level at the reference time"), given) == "known_value_mismatch"
    assert commission.mismatch_reason(_with(CHECK, kind="published_value"), {}) == "known_value_mismatch"


# --- the gate on a whole quest ----------------------------------------------------------------------------------------

_MEASURES = _HEAD + """\
if os.environ.get("FI_ORACLE") == "1":
    print("ORACLE_JSON: " + json.dumps({"checks": [{"name": "drain level at the reference time", "value": 0.734,
          "mc": {"verdict": "fail", "se": 0.0, "trials": 40}}]}))
    raise SystemExit(0)
""" + _TAIL


def _quest_fake(calls: list[str], error: str):
    base = _fake(calls, implement=_MEASURES, protocol={**_PROTOCOL, "oracles": [CHECK]}, revise=False)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if prompt.startswith("# Compute a Known Answer Numerically"):
            calls.append("Numeric")
            line = {"value": 0.699, "how": "a fine-step integration", **({"error": float(error)} if error else {})}
            return "```python\nimport json\nprint('NUMERIC_JSON: ' + json.dumps(%r))\n```" % line
        return await base(self, messages, **kw)

    return fake_chat


async def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, commissioned: bool, error: str = "0.002") -> Engine:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _quest_fake(calls, error))
    cfg = _cfg(tmp_path, oracle_repair_attempts=1)
    cfg.provider.model = "model-a"
    cfg.provider.node_models = {"oracle_review": "model-b"}
    if commissioned:
        cfg.commission = _commission(tmp_path / "order")
    engine = Engine(cfg)
    await engine.run()
    assert calls.count("Numeric") == 1, "the referee is asked once"
    return engine


@pytest.mark.asyncio
async def test_a_commissioned_study_whose_three_numbers_disagree_is_handed_back_before_it_runs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import quality_report

    engine = await _run(tmp_path, monkeypatch, commissioned=True)
    stuck = json.loads((engine.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert list(stuck)[:3] == ["return_reason", "kind", "evidence"]
    assert stuck["return_reason"] == "plan_contradicts_sources" and stuck["kind"] == "returned"
    assert stuck["evidence"]["detector"] == "known_answer_three_way"
    assert stuck["evidence"]["numbers"]["simulation"]["value"] == 0.734
    assert stuck["evidence"]["numbers"]["reference"] == {**stuck["evidence"]["numbers"]["reference"], "value": 0.699,
                                                         "error": 0.002}
    assert "0.734" in stuck["problem"] and "0.699" in stuck["problem"] and "0.655" in stuck["problem"]
    assert "FI did not run the study" in stuck["problem"]
    record = _record(engine)
    assert record["status"] == "stopped" and record["returned"] == "plan_contradicts_sources" and "went_on" not in record
    assert not (engine.quest_root / "quest_failed.md").exists()
    assert not (engine.quest_root / "paper.md").exists()
    assert not (engine.quest_root / "figures" / "result.png").exists(), "the study itself never ran"
    outcome = quality_report.build(engine.quest_root, running=False)["outcome"]
    assert outcome["kind"] == "honest_stop" and outcome["return_reason"] == "plan_contradicts_sources"


@pytest.mark.asyncio
async def test_a_referee_with_no_error_hands_nothing_back_and_the_study_goes_on(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = await _run(tmp_path, monkeypatch, commissioned=True, error="")
    record = _record(engine)
    assert record["status"] == "went_on_failing" and "returned" not in record
    assert not (engine.quest_root / "needs" / "STUCK.json").exists()


@pytest.mark.asyncio
async def test_a_study_run_directly_goes_on_with_the_check_marked_unconfirmed_as_before(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = await _run(tmp_path, monkeypatch, commissioned=False)
    record = _record(engine)
    assert record["status"] == "went_on_failing" and "returned" not in record
    assert [e["name"] for e in record["went_on"]] == [CHECK["name"]] and record["went_on"][0]["by"] == "automatic"
    assert not (engine.quest_root / "needs" / "STUCK.json").exists()


# --- the plan that defines one quantity two ways ----------------------------------------------------------------------

METRIC = {"id": "drain_rate", "kind": "mean", "unit": "one trial", "estimand": "(level_start - level_end)/duration"}
TWICE = {"hypothesis": "a made-up hypothesis", "variables": {"independent": ["outlet"],
                                                             "dependent": ["drain_rate = level_end/level_start"]},
         "protocol": {"form": "contract", "metrics": [METRIC]}}


async def _plan_conflict(tmp_path: Path, commissioned: bool, design: dict[str, Any]) -> Engine:
    from tests.test_commission import _engine

    engine = _engine(tmp_path, _commission(tmp_path / "order"))
    if not commissioned:
        engine.config.commission = None
    plan.plan_path(engine.quest_root).parent.mkdir(parents=True, exist_ok=True)
    plan.plan_path(engine.quest_root).write_text(plan.render("a made-up tank", {}, design), encoding="utf-8")
    engine._runs_code = lambda state: True  # type: ignore[method-assign]

    async def asked(*args: Any, **kwargs: Any) -> None:
        return None  # the plan's model leaves the plan as it was

    engine._ask_plan_to_complete = asked  # type: ignore[method-assign]
    record = {"issues": {"F-1": {"id": "F-1", "plan_conflict": True, "outcome": "open", "text": "two definitions",
                                 "why": "the plan's metric and its variable differ", "readings": [{"status": "open"}]}}}
    await engine._route_plan_conflicts({}, record, "code_review")
    return engine


@pytest.mark.asyncio
async def test_a_commissioned_plan_that_still_defines_one_quantity_two_ways_is_handed_back(tmp_path: Path) -> None:
    engine = await _plan_conflict(tmp_path, True, TWICE)
    stop = engine._returned_stop()
    assert stop is not None and engine._returned_stop() is None, "handed back once"
    assert engine._route_after_execute_reflect(stop) == "stuck"
    state = {**stop, "iteration": 0}
    verdict = engine._no_results_verdict(state)
    assert verdict is not None and verdict["stuck_reason"] == "returned"
    await engine._node_stuck_no_findings({**state, "evidence_assessment": verdict})
    stuck = json.loads((engine.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert stuck["return_reason"] == "plan_contradicts_sources" and stuck["evidence"]["detector"] == "defined_twice"
    assert "drain_rate" in stuck["problem"] and "defines one quantity two ways" in stuck["problem"]


@pytest.mark.asyncio
async def test_a_plan_defined_once_or_a_study_run_directly_is_not_handed_back(tmp_path: Path) -> None:
    once = {**TWICE, "variables": {"independent": ["outlet"], "dependent": ["drain_rate"]}}
    assert (await _plan_conflict(tmp_path / "once", True, once))._returned_stop() is None
    assert (await _plan_conflict(tmp_path / "direct", False, TWICE))._returned_stop() is None
