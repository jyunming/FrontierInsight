"""The oracles a protocol declares (core/oracle_check.py) and the gate that holds the quest at them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import oracle_check as oc, plan, protocol_check as pc
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

ORACLE = {"name": "final size closed form", "kind": "closed_form", "check": "small-N final size against the root of the final-size relation", "expected": 1.0, "tolerance": 0.05}


# --- reading an oracle run ---------------------------------------------------------------------------------------------


def _line(checks: list[dict[str, Any]]) -> str:
    return "noise\nORACLE_JSON: " + json.dumps({"checks": checks}) + "\n"


def test_the_declared_oracles_are_read_from_the_protocol() -> None:
    assert oc.declared(None) == [] and oc.declared({"grid": {}}) == []
    assert oc.declared({"oracles": [ORACLE, "conservation of population", {"kind": "x"}, ""]}) == [
        ORACLE, {"name": "conservation of population", "check": "conservation of population"},
    ]


def test_the_last_oracle_line_is_the_answer_and_junk_is_ignored() -> None:
    out = "ORACLE_JSON: {not json}\n" + _line([{"name": "a", "passed": False}]) + _line([{"name": "a", "passed": True}])
    assert oc.parse(out)["checks"][0]["passed"] is True
    assert oc.parse("no such line") is None and oc.parse("") is None
    assert oc.parse('ORACLE_JSON: ["a list"]') is None


def test_a_run_with_every_declared_oracle_passing_has_no_problem() -> None:
    reported = oc.parse(_line([{"name": ORACLE["name"].upper(), "passed": True, "value": 1.0}]))
    assert oc.problems([ORACLE], reported, 0) == []


@pytest.mark.parametrize("reported, returncode, timed_out, expect", [
    (None, 1, False, "printed no `ORACLE_JSON:` line"),
    (None, 0, True, "did not finish the oracle checks in time"),
    ({"checks": []}, 0, False, "was not checked"),
    ({"checks": [{"name": ORACLE["name"], "value": 0.5}]}, 1, False,
     "failed: the script measured 0.5, the protocol expects 1 within 0.05"),
    ({"checks": [{"name": ORACLE["name"], "passed": True, "value": 0.5}]}, 0, False, "failed: the script measured 0.5"),
    ({"checks": [{"name": ORACLE["name"], "passed": True}]}, 0, False, "reported no finite numeric `value`"),
    ({"checks": [{"name": ORACLE["name"], "value": "0.99"}]}, 0, False, "reported no finite numeric `value`"),
    ({"checks": [{"name": ORACLE["name"], "value": float("nan")}]}, 0, False, "reported no finite numeric `value`"),
    ({"checks": [{"name": ORACLE["name"], "value": 0.99, "passed": False}]}, 0, False, "reports the check as failed itself"),
    ({"checks": [{"name": ORACLE["name"], "value": 1.0}]}, 3, False, "exited with code 3"),
])
def test_what_is_wrong_with_an_oracle_run_is_named(reported, returncode, timed_out, expect) -> None:
    found = oc.problems([ORACLE], reported, returncode, timed_out)
    assert len(found) == 1 and expect in found[0], found


def test_a_protocol_with_no_oracle_is_a_problem_of_its_own() -> None:
    found = oc.problems([], {"checks": []}, 0)
    assert len(found) == 1 and "declares no oracle" in found[0]


def test_the_repair_request_carries_the_oracles_the_problems_and_the_contract() -> None:
    text = oc.directive([ORACLE], ["the oracle 'x' failed (value 0.5)"])
    for needle in (ORACLE["name"], "the oracle 'x' failed", "FI_ORACLE=1", "ORACLE_JSON", "Never make a check pass by measuring something else",
                   "the engine judges", "does NOT decide pass or fail"):
        assert needle in text, needle


# --- the plan ------------------------------------------------------------------------------------------------------------


def test_oracles_in_the_protocol_are_checked_for_shape_and_kept() -> None:
    design, error = plan.normalize_design({"hypothesis": "h", "protocol": {"oracles": [ORACLE, "invariant: S+I+R = N"]}})
    assert error is None
    assert design["protocol"]["oracles"][0] == ORACLE
    assert design["protocol"]["oracles"][1] == {"name": "invariant: S+I+R = N", "check": "invariant: S+I+R = N"}
    assert plan.parse(plan.render("t", {}, design)).design["protocol"]["oracles"] == design["protocol"]["oracles"]
    assert plan.normalize_design({"hypothesis": "h", "protocol": {"oracles": ORACLE}})[0]["protocol"]["oracles"] == [ORACLE]


@pytest.mark.parametrize("bad, why", [
    ({"oracles": [{"name": "a", "expected": [1, 2]}]}, "`expected` must be a number"),
    ({"oracles": [{"name": "a", "tolerance": {"x": 1}}]}, "`tolerance` must be a number"),
    ({"oracles": [{"name": "a", "tolerance_mode": "fuzzy"}]}, "`tolerance_mode` must be"),
    ({"oracles": [{"name": "a", "reference": 3}]}, "`reference` must be text"),
    ({"oracles": 3}, "list of checks"),
    ({"oracles": [{"check": "x"}]}, "no `name`"),
    ({"oracles": [{"name": "a", "check": 5}]}, "`check` must be text"),
])
def test_oracles_that_could_not_be_answered_are_refused(bad: dict[str, Any], why: str) -> None:
    design, error = plan.normalize_design({"hypothesis": "h", "protocol": bad})
    assert design is None and why in (error or "")


def test_a_value_is_judged_against_the_protocols_numbers_and_never_against_the_scripts_own() -> None:
    lying = oc.parse(_line([{"name": ORACLE["name"], "passed": True, "value": 0.5, "expected": 0.5, "tolerance": 10.0}]))
    found = oc.problems([ORACLE], lying, 0)
    assert len(found) == 1 and "measured 0.5, the protocol expects 1 within 0.05" in found[0], found
    inside = oc.parse(_line([{"name": ORACLE["name"], "value": 1.04, "expected": 7, "tolerance": 0.001}]))
    assert oc.problems([ORACLE], inside, 0) == []


def test_a_relative_tolerance_scales_with_the_expected_value_and_is_refused_around_zero() -> None:
    relative = {"name": "a", "expected": 200.0, "tolerance": 0.01, "tolerance_mode": "relative"}
    assert oc.limit_of(relative) == (200.0, 2.0, "relative")
    assert oc.problems([relative], {"checks": [{"name": "a", "value": 201.5}]}, 0) == []
    assert "measured 203" in oc.problems([relative], {"checks": [{"name": "a", "value": 203.0}]}, 0)[0]
    around_zero = {"name": "inv", "expected": 0, "tolerance": 0.1, "tolerance_mode": "relative"}
    assert "relative tolerance around an expected value of 0" in oc.unjudgeable([around_zero])[0]
    invariant = {"name": "inv", "kind": "invariant", "expected": 0, "tolerance": 1e-9}
    assert oc.problems([invariant], {"checks": [{"name": "inv", "value": 2e-12}]}, 0) == []
    assert oc.problems([invariant], {"checks": [{"name": "inv", "value": 3e-6}]}, 0)


@pytest.mark.parametrize("oracle", [
    {"name": "a"}, {"name": "a", "check": "x", "tolerance": 0.1}, {"name": "a", "expected": 1.0},
    {"name": "a", "expected": "about one", "tolerance": "small"}, {"name": "a", "expected": 1.0, "tolerance": -0.1},
    {"name": "a", "expected": float("inf"), "tolerance": 0.1},
])
def test_an_oracle_with_no_numbers_to_judge_by_is_not_run_and_is_named(oracle) -> None:
    assert len(oc.unjudgeable([oracle])) == 1 and "cannot judge it" in oc.unjudgeable([oracle])[0]
    assert oc.unjudgeable([ORACLE]) == []


def test_what_the_engine_made_of_each_check_is_recorded() -> None:
    reported = oc.parse(_line([{"name": ORACLE["name"], "passed": True, "value": 0.5}]))
    (row,) = oc.judged([ORACLE], reported)
    assert row["value"] == 0.5 and row["expected"] == 1.0 and row["limit"] == 0.05 and row["passed_by_engine"] is False
    assert row["script_said"] is True
    assert oc.judged([ORACLE], None)[0]["passed_by_engine"] is None


def test_the_plan_says_when_an_oracle_has_no_numbers_the_engine_can_judge_by() -> None:
    notes = pc.oracle_notes({"oracles": [{"name": "closed form", "check": "x"}]})
    assert len(notes) == 1 and "cannot judge it" in notes[0] and "while the plan is a draft" in notes[0]
    assert pc.oracle_notes({"oracles": [ORACLE]}) == []


def test_the_plan_says_when_its_protocol_declares_no_oracle() -> None:
    assert pc.oracle_notes({"oracles": [ORACLE]}) == []
    assert "declares no oracle" in pc.oracle_notes({"grid": {"R0": [1.0]}})[0]
    assert "declares no oracle" in pc.oracle_notes({"oracles": []})[0]


# --- the gate, through the real graph and a real checkpoint -------------------------------------------------------------

_HEAD = """\
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
_TAIL = """\
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1, 2], [0, 1, 4]); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: {"score": 0.987}')
"""
_PASSING = _HEAD + """\
if os.environ.get("FI_ORACLE") == "1":
    print("ORACLE_JSON: " + json.dumps({"checks": [{"name": "final size closed form", "passed": True, "value": 0.99, "expected": 1.0, "tolerance": 0.05}]}))
    raise SystemExit(0)
""" + _TAIL
_FAILING = _HEAD + """\
if os.environ.get("FI_ORACLE") == "1":
    print("ORACLE_JSON: " + json.dumps({"checks": [{"name": "final size closed form", "passed": False, "value": 0.5, "expected": 1.0, "tolerance": 0.05}]}))
    raise SystemExit(1)
""" + _TAIL
_UNAWARE = _HEAD + _TAIL  # never looks at FI_ORACLE: it runs the whole experiment instead
_PROTOCOL = {"runs_per_setting": 300}


def _cfg(tmp_path: Path, **engine: Any) -> Config:
    return Config(
        topic="smoke topic for the oracle gate", title="oracle-smoke", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False, **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def _fake(calls: list[str], *, implement: str, repair: str | None = None, protocol: dict[str, Any] | None = None,
          revise: bool = True):
    protocol = {**_PROTOCOL} if protocol is None else protocol

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = protocol
            return json.dumps(body)
        if kind == "Implementation":
            return json.dumps({"code": implement, "deps": ["matplotlib"]})
        if kind == "ExecuteReflect" and "FI_ORACLE=1 to check its oracles" in prompt:
            calls.append("OracleRepair")
            return json.dumps({"code": repair, "deps": [], "patch_summary": "answered the oracles"}) if repair else "{}"
        if "You are revising the plan" in prompt:
            calls.append("PlanRevise")
            if not revise:
                return "no plan file here"
            current = prompt.split("# The plan as it stands", 1)[1].split("# What the person asked for", 1)[0].strip()
            design = plan.parse(current).design
            design["protocol"] = {**design["protocol"], "oracles": [ORACLE]}
            return plan.render("smoke", {}, design)
        return _fake_response_for(prompt)

    return fake_chat


def _record(engine: Engine) -> dict[str, Any]:
    return json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_declared_oracle_the_script_passes_is_checked_before_the_main_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_PASSING, protocol=protocol))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    record = _record(engine)
    assert record["status"] == "ok" and len(record["attempts"]) == 1 and record["attempts"][0]["oracles"] == [ORACLE["name"]]
    assert "OracleRepair" not in calls and "PlanRevise" not in calls
    assert "[oracle] 1 oracle(s) passed before the main run" in (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_script_that_ignores_the_oracle_request_is_sent_back_and_the_repaired_one_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_UNAWARE, repair=_PASSING, protocol=protocol))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and calls.count("OracleRepair") == 1
    assert "FI_ORACLE" in (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
    record = _record(engine)
    assert record["status"] == "ok" and "printed no `ORACLE_JSON:` line" in record["attempts"][0]["problems"][0]


@pytest.mark.asyncio
async def test_a_failing_oracle_stops_the_quest_before_the_main_run_and_a_fixed_script_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_FAILING, repair=_FAILING, protocol=protocol))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()

    log = (first.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert calls.count("OracleRepair") == 2
    assert "[oracle] paused" in log and "[FI] paused for the oracle checks" in log and "paused for clarify" not in log
    assert not (first.fi_dir / "clarify_questions.json").exists()
    descriptor = json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert descriptor["kind"] == "oracle" and descriptor["interaction"] == "supply"
    text = (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "the oracle 'final size closed form' failed: the script measured 0.5, the protocol expects 1 within 0.05" in text
    assert _record(first)["status"] == "stopped"
    assert not list((first.quest_root / "figures").glob("*.png")), "the main run must not have started"
    assert not (first.quest_root / "paper" / "paper.md").exists()

    (first.quest_root / "code" / "experiment.py").write_text(_PASSING, encoding="utf-8")
    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert _record(second)["status"] == "ok"


@pytest.mark.asyncio
async def test_a_protocol_with_no_oracle_asks_the_plan_for_one_and_then_checks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_PASSING))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and calls.count("PlanRevise") == 1
    assert plan.load_design(engine.quest_root)[0]["protocol"]["oracles"] == [ORACLE]
    assert [r["by"] for r in plan.history(engine.quest_root)] == ["model", "request"]
    record = _record(engine)
    assert record["status"] == "ok" and "declares no oracle" in record["attempts"][0]["problems"][0]


@pytest.mark.asyncio
async def test_when_the_plan_cannot_be_given_an_oracle_the_quest_stops_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_PASSING, revise=False))
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    assert "[oracle] paused" in (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "declares no oracle" in (engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert not (engine.quest_root / "paper" / "paper.md").exists()


@pytest.mark.asyncio
async def test_warn_records_the_problem_and_the_quest_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_FAILING, protocol=protocol))
    engine = Engine(_cfg(tmp_path, oracle_check="warn", oracle_repair_attempts=0))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert _record(engine)["status"] == "warned" and calls.count("OracleRepair") == 0


@pytest.mark.asyncio
async def test_off_and_a_quest_with_no_protocol_are_not_looked_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, extra, protocol in (("off", {"oracle_check": "off"}, _PROTOCOL), ("none", {}, {})):
        calls: list[str] = []
        monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_UNAWARE, protocol=protocol))
        engine = Engine(_cfg(tmp_path / name, **extra))
        artifacts = await engine.run()
        assert artifacts.paper_md is not None
        assert "OracleRepair" not in calls and "PlanRevise" not in calls
        assert not (engine.quest_root / "needs" / "ORACLE_CHECK.json").exists()


# --- a plan edit made while the quest was stopped is heard --------------------------------------------------------------

_BAD_GRID = "R0_LIST = [0.8, 1.2, 3.0]\nNUM_RUNS = 300\n" + _PASSING


@pytest.mark.asyncio
async def test_a_protocol_edited_in_the_plan_while_stopped_is_the_one_the_script_is_held_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The protocol stop tells a person to fix the script or the plan. Fixing the plan must work: the check reads the
    protocol from plan.md, not the copy the design step left in the checkpoint."""
    calls: list[str] = []
    protocol = {"grid": {"R0": [0.9, 1.5, 3.0]}, "runs_per_setting": 300, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_BAD_GRID, repair=_BAD_GRID, protocol=protocol))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()
    assert (first.fi_dir / "protocol_stop.json").is_file()

    path = plan.plan_path(first.quest_root)
    edited = path.read_text(encoding="utf-8").replace("- 0.9\n", "- 0.8\n").replace("- 1.5\n", "- 1.2\n")
    assert edited != path.read_text(encoding="utf-8")
    path.write_text(edited, encoding="utf-8")

    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert "agrees with the plan's protocol; keeping it" in (second.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")


def test_the_dashboard_and_the_quest_page_name_the_oracle_stop() -> None:
    static = Path(__file__).resolve().parent.parent / "web" / "static"
    for page in ("index.html", "quest.html"):
        assert "oracle: 'oracle'" in (static / page).read_text(encoding="utf-8"), page


def test_the_code_writing_prompts_give_the_oracle_contract() -> None:
    agents = Path(__file__).resolve().parent.parent / "agents"
    for name in ("implement.md", "implement_body.md"):
        text = (agents / name).read_text(encoding="utf-8")
        assert "FI_ORACLE" in text and "ORACLE_JSON" in text and "Never make a check pass by measuring something else" in text, name
        assert "do NOT print pass/fail" in text and "the engine judges" in text, name


_LYING = _HEAD + """\
if os.environ.get("FI_ORACLE") == "1":
    print("ORACLE_JSON: " + json.dumps({"checks": [{"name": "final size closed form", "passed": True, "value": 0.5, "expected": 0.5, "tolerance": 10.0}]}))
    raise SystemExit(0)
""" + _TAIL


@pytest.mark.asyncio
async def test_a_script_that_declares_its_own_pass_expected_value_and_tolerance_is_still_judged_by_the_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The script prints passed: true beside its own expected value and a tolerance wide enough to pass anything; the engine
    reads the value only, and holds it to the protocol's numbers."""
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_LYING, repair=_LYING, protocol=protocol))
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    record = _record(engine)
    assert record["status"] == "stopped" and record["judged_by"] == "engine"
    last = record["attempts"][-1]
    assert last["judged"][0]["passed_by_engine"] is False and last["judged"][0]["script_said"] is True
    assert "the script measured 0.5, the protocol expects 1 within 0.05" in last["problems"][0]
    assert not (engine.quest_root / "paper" / "paper.md").exists()


@pytest.mark.asyncio
async def test_an_oracle_with_no_numbers_is_completed_in_the_plan_before_anything_is_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    prompts: list[str] = []
    bare = {"name": "final size closed form", "kind": "closed_form", "check": "small-N final size", "tolerance": "small"}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        prompts.append(prompt)
        if "You are revising the plan" in prompt:
            calls.append("PlanRevise")
            current = prompt.split("# The plan as it stands", 1)[1].split("# What the person asked for", 1)[0].strip()
            design = plan.parse(current).design
            design["protocol"] = {**design["protocol"], "oracles": [ORACLE]}
            return plan.render("smoke", {}, design)
        return await _fake(calls, implement=_PASSING, protocol={**_PROTOCOL, "oracles": [bare]})(self, messages, **kw)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and calls.count("PlanRevise") == 1
    asked = [p for p in prompts if "cannot be judged by the engine" in p]
    assert asked and "numeric `expected`" in asked[0]
    record = _record(engine)
    assert record["status"] == "ok" and "cannot judge it" in record["attempts"][0]["problems"][0]
    assert record["attempts"][0]["judged"] == [], "an oracle the engine cannot judge is not run"
    assert record["attempts"][1]["judged"][0]["passed_by_engine"] is True
