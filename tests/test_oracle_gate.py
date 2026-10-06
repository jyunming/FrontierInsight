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

ORACLE = {"name": "final size closed form", "kind": "closed_form", "check": "small-N final size against the root of the final-size relation", "expected": 1.0, "expected_formula": "1.0", "tolerance": 0.05}


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
        # A quest whose result is for a decision stops at a failed check; an exploration goes on by itself.
        result_use="decision",
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
async def test_a_failing_oracle_is_repaired_then_gone_on_with_marked_unconfirmed_never_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_FAILING, repair=_FAILING, protocol=protocol))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()

    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert calls.count("OracleRepair") == 2, "FI repairs first"
    assert "[oracle] paused" not in log, "whether the check or the script is wrong is never asked"
    record = _record(engine)
    assert record["status"] == "went_on_failing" and record["went_on"][0]["by"] == "automatic"
    last = record["attempts"][-1]
    assert "the script measured 0.5, the protocol expects 1 within 0.05" in last["problems"][0]
    assert last["judged"][0]["passed_by_engine"] is False, "the check stays failed"
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert "FI could not confirm 'final size closed form'" in capsys.readouterr().out


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
    assert [r["by"] for r in plan.history(engine.quest_root)] == ["model", "engine"]  # the engine's rewrite, not a person's
    record = _record(engine)
    assert record["status"] == "ok" and "declares no oracle" in record["attempts"][0]["problems"][0]


@pytest.mark.asyncio
async def test_when_the_plan_cannot_be_given_an_oracle_the_quest_goes_on_and_the_result_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import evidence

    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_PASSING, revise=False))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert "[oracle] paused" not in (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    record = _record(engine)
    assert record["status"] == "went_on_failing" and "declares no oracle" in record["went_on"][0]["unmeasured"]
    assert artifacts.paper_md is not None
    gaps = json.dumps(evidence.read(engine.quest_root) or {})
    assert "no known-answer check could be judged" in gaps
    assert (evidence.read(engine.quest_root) or {}).get("level") not in ("independently_validated", "publication_ready")


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
        # The pause kind stays `oracle`; a person reads the one plain name.
        assert "oracle: 'known-answer check'" in (static / page).read_text(encoding="utf-8"), page


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
    assert record["status"] == "went_on_failing" and record["judged_by"] == "engine", "never passed on its own word"
    last = record["attempts"][-1]
    assert last["judged"][0]["passed_by_engine"] is False and last["judged"][0]["script_said"] is True
    assert "the script measured 0.5, the protocol expects 1 within 0.05" in last["problems"][0]


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


# --- the gate under execution.split_analysis: true (a real quest's own generated code, not a fixture) -----------------

# Every real two-script quest's simulate.py reads FI_RAW_DIR unconditionally at module level -- the split-experiment
# directive's own worked example does exactly this (core/engine.py: ``raw = pathlib.Path(os.environ["FI_RAW_DIR"])``;
# tests/test_split_analysis.py's own SIMULATE fixture follows the same shape). Python runs that line on import
# regardless of which branch the script takes, so it fires during the oracle pre-check too -- discovered on a real
# live-quest campaign (2026-09-23) where 2 of 4 real Kimi-authored quests crashed with a bare KeyError before ever
# reaching their oracle logic, every declared oracle repair attempt exhausted trying to fix "the oracle mode" when
# the actual bug was this module-level read having no value to read during the pre-check.
_SPLIT_SIMULATE_HEAD = """\
import json, os, pathlib
raw = pathlib.Path(os.environ["FI_RAW_DIR"])
raw.mkdir(parents=True, exist_ok=True)
"""
_SPLIT_SIMULATE_PASSING = _SPLIT_SIMULATE_HEAD + """\
if os.environ.get("FI_ORACLE") == "1":
    print("ORACLE_JSON: " + json.dumps({"checks": [{"name": "final size closed form", "passed": True, "value": 0.99, "expected": 1.0, "tolerance": 0.05}]}))
    raise SystemExit(0)
seed = int(os.environ.get("FI_REPLICATE_SEED", "0"))
(raw / "values.json").write_text(json.dumps({"values": [seed + 1.0]}))
"""
_SPLIT_ANALYSIS = """\
import json, os, pathlib
raw = pathlib.Path(os.environ["FI_RAW_DIR"])
data = json.loads((raw / "values.json").read_text())
print("RESULT_JSON: " + json.dumps({"score": sum(data["values"])}))
"""


def _split_reply(sim: str, ana: str) -> str:
    return f"```python\n# file: simulate.py\n{sim}\n```\n```python\n# file: experiment.py\n{ana}\n```\nDEPS: numpy\n"


@pytest.mark.asyncio
async def test_a_two_script_oracle_check_gets_a_real_raw_dir_not_a_keyerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact shape a real quest's own generated simulate.py takes under ``execution.split_analysis: true``: it
    reads ``FI_RAW_DIR`` before it ever checks ``FI_ORACLE``. Without a real value for that variable during the
    oracle pre-check, this is indistinguishable from a script that never implemented its oracles at all -- the
    engine can't tell "crashed on an unrelated KeyError" from "ignored the oracle request," and asks for a repair
    that can never fix the actual problem."""
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = protocol
            return json.dumps(body)
        if kind == "Implementation":
            return _split_reply(_SPLIT_SIMULATE_PASSING, _SPLIT_ANALYSIS)
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = Config(
        topic="split oracle smoke", title="split-oracle-smoke", provider=ProviderConfig(name="openai"),
        # The fixture's simulation writes no run manifest; this test is about the oracle pre-check, so that
        # difference is recorded, not stopped on (a stop is what block does when no repair removes it).
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1, pilot_run=False,
                            run_manifest_check="warn"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)
    artifacts = await engine.run()

    assert artifacts.paper_md is not None
    record = _record(engine)
    assert record["status"] == "ok" and len(record["attempts"]) == 1, record
    assert "OracleRepair" not in calls, "the module-level FI_RAW_DIR read must not look like an unanswered oracle request"


# --- what a repair is told, and what uses one up (two live kimi-k3 quests) ----------------------------------------------

_CRASHING = _HEAD + """\
if os.environ.get("FI_ORACLE") == "1":
    raise TypeError("Put.__init__() got an unexpected keyword argument 'priority'")
""" + _TAIL


@pytest.mark.asyncio
async def test_a_repair_sees_the_crash_and_a_call_that_never_answered_does_not_use_it_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live quest's script crashed inside simpy; the repair was told only "printed no ORACLE_JSON line", and the one
    repair left timed out at the provider and was counted as spent. With one repair allowed: the traceback reaches the
    repair, and the timed-out call is tried again instead of ending the quest."""
    calls: list[str] = []
    repair_prompts: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    inner = _fake(calls, implement=_CRASHING, repair=_PASSING, protocol=protocol)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "ExecuteReflect" and "FI_ORACLE=1 to check its oracles" in prompt:
            repair_prompts.append(prompt)
            if len(repair_prompts) == 1:
                raise TimeoutError("the provider never answered")
        return await inner(self, messages, **kw)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=1))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert len(repair_prompts) == 2
    assert "unexpected keyword argument 'priority'" in repair_prompts[0], "the repair must be shown the crash"
    record = _record(engine)
    assert record["status"] == "ok" and len(record["attempts"]) == 3


@pytest.mark.asyncio
async def test_completing_the_plan_does_not_use_up_the_scripts_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live quest needed its plan completed (an oracle with no numbers), then its FI_ORACLE branch added, and had
    nothing left for the problem after that. With one of each allowed, both happen."""
    calls: list[str] = []
    bare = {"name": "final size closed form", "kind": "closed_form", "check": "small-N final size", "tolerance": "small"}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if "You are revising the plan" in prompt:
            calls.append("PlanRevise")
            current = prompt.split("# The plan as it stands", 1)[1].split("# What the person asked for", 1)[0].strip()
            design = plan.parse(current).design
            design["protocol"] = {**design["protocol"], "oracles": [ORACLE]}
            return plan.render("smoke", {}, design)
        return await _fake(calls, implement=_UNAWARE, repair=_PASSING, protocol={**_PROTOCOL, "oracles": [bare]})(self, messages, **kw)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=1))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert calls.count("PlanRevise") == 1 and calls.count("OracleRepair") == 1
    assert _record(engine)["status"] == "ok"


# --- a repair may say the check itself is wrong; a person decides (four kimi-k3 rounds on one topic) -----------------


def test_a_proposed_oracle_change_is_kept_only_when_it_is_usable() -> None:
    declared = [ORACLE]
    good = {"name": ORACLE["name"].upper(), "expected": 1, "tolerance": 0.2, "tolerance_mode": "relative", "reason": "h^2/8 bound"}
    got = oc.proposals([good, {"name": "not declared", "expected": 1, "tolerance": 1, "reason": "x"},
                        {"name": ORACLE["name"], "expected": 1, "tolerance": -1, "reason": "x"},
                        {"name": ORACLE["name"], "expected": 1, "tolerance": 1}, "junk"], declared)
    assert got == [{"name": ORACLE["name"], "expected": 1.0, "tolerance": 0.2, "tolerance_mode": "relative", "reason": "h^2/8 bound"}]
    assert oc.proposals(None, declared) == [] and oc.proposals({"name": "x"}, declared) == []
    request = oc.proposal_request(got[0])
    assert ORACLE["name"] in request and "tolerance 0.2 (relative)" in request and "h^2/8 bound" in request
    assert "oracle_change" in oc.directive(declared, ["x"]) and "nothing changes without them" in oc.directive(declared, ["x"])


@pytest.mark.asyncio
async def test_a_check_the_repair_calls_wrong_is_recorded_and_never_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live quest's repair said, four rounds running, that the declared tolerance was below the method's own error.
    Its proposal is recorded; it is never applied (the repair saw the run, so its numbers are not independent of the
    measurement), and the quest goes on with the check marked unconfirmed. The plan is not touched."""
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "ExecuteReflect" and "FI_ORACLE=1 to check its oracles" in prompt:
            calls.append("OracleRepair")
            return json.dumps({
                "code": _FAILING, "deps": [], "patch_summary": "the check, not the script",
                "oracle_change": [{"name": ORACLE["name"], "expected": 1.0, "tolerance": 0.6, "reason": "the small case's own error is 0.5"}],
            })
        return await _fake(calls, implement=_FAILING, protocol=protocol)(self, messages, **kw)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=1))
    await engine.run()
    record = _record(engine)
    assert record["status"] == "went_on_failing"
    assert record["proposed_changes"] == [{"name": ORACLE["name"], "expected": 1.0, "tolerance": 0.6, "tolerance_mode": "absolute", "reason": "the small case's own error is 0.5"}]
    plan_after = plan.parse(plan.plan_path(engine.quest_root).read_text(encoding="utf-8")).design["protocol"]["oracles"]
    assert plan_after == [ORACLE], "a repair's proposal is never applied by the engine"
    assert not (engine.fi_dir / "oracle_corrections.json").exists()


# --- a repair that blames the check never rewrites the script for it (a real kimi-k3 quest) --------------------------
# The declared expected value was wrong (1.637e-08 where the true RK4 error is 3.33e-07) and the script measured the true
# value. Each repair said so in `oracle_change` AND returned new code; the code was applied anyway, and two or three
# repairs later the correct script crashed, so the quest produced no result at all.

# What a repair that bends the script to the (wrong) expected value looks like: it reports the declared number.
_BENT = _HEAD + """\
if os.environ.get("FI_ORACLE") == "1":
    print("ORACLE_JSON: " + json.dumps({"checks": [{"name": "final size closed form", "value": 1.0}]}))
    raise SystemExit(0)
""" + _TAIL
_DISPUTE = [{"name": ORACLE["name"], "expected": 0.5, "tolerance": 0.01,
             "reason": "the closed form gives 0.5 on this case; 1.0 is an arithmetic slip in the protocol"}]




def _disputing_fake(calls: list[str], protocol: dict[str, Any], prompts: list[str] | None = None):
    base = _fake(calls, implement=_FAILING, protocol=protocol)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "ExecuteReflect" and "FI_ORACLE=1 to check its oracles" in prompt:
            calls.append("OracleRepair")
            if prompts is not None:
                prompts.append(prompt)
            return json.dumps({"code": _BENT, "deps": [], "patch_summary": "made the check pass",
                               "oracle_change": _DISPUTE})
        return await base(self, messages, **kw)

    return fake_chat


@pytest.mark.asyncio
async def test_a_repair_that_blames_the_check_does_not_rewrite_the_script_and_the_quest_goes_on_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _disputing_fake(calls, protocol))
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=2))
    await engine.run()
    assert (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == _FAILING, \
        "the script the repair called right must be kept as it was"
    assert calls.count("OracleRepair") == 1, "once every failing check is disputed, no more repairs are spent"
    record = _record(engine)
    assert record["status"] == "went_on_failing" and record["disputed"] == [ORACLE["name"]]
    assert record["attempts"][0]["repair"] == "set_aside_disputed"
    assert record["proposed_changes"][0]["expected"] == 0.5
    assert record["explained"]["leaning"] == "check", "the record says FI's look points to the check"
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[oracle] rewrote experiment.py" not in log


@pytest.mark.asyncio
async def test_under_warn_the_quest_goes_on_with_the_original_script_and_says_the_check_is_disputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _disputing_fake(calls, protocol))
    engine = Engine(_cfg(tmp_path, oracle_check="warn", oracle_repair_attempts=2))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == _FAILING
    assert calls.count("OracleRepair") == 1
    record = _record(engine)
    assert record["status"] == "warned" and record["disputed"] == [ORACLE["name"]]
    judged = oc.last_judged(record)
    assert judged[0]["passed_by_engine"] is False and judged[0]["disputed_expected"] == 0.5
    note = oc.analysis_note(judged)
    assert "failed" in note and "the expected value the protocol declares is itself wrong" in note
    assert "nobody has approved" in note and "counts as failed" in note


_OTHER = {"name": "mass conservation", "kind": "invariant", "check": "total mass drift", "expected": 0.0, "tolerance": 0.01}
_TWO = {**_PROTOCOL, "oracles": [ORACLE, _OTHER]}


def _two(a: float, b: float) -> str:
    """A script measuring the two checks: ORACLE (expected 1.0) at ``a`` and _OTHER (expected 0.0) at ``b``."""
    return _HEAD + (
        'if os.environ.get("FI_ORACLE") == "1":\n'
        '    print("ORACLE_JSON: " + json.dumps({"checks": [{"name": "final size closed form", "value": %r}, '
        '{"name": "mass conservation", "value": %r}]}))\n'
        '    raise SystemExit(0)\n' % (a, b)
    ) + _TAIL


def _scripted_repairs(calls: list[str], prompts: list[str], implement: str, replies: list[dict[str, Any]]):
    base = _fake(calls, implement=implement, protocol=_TWO)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "ExecuteReflect" and "FI_ORACLE=1 to check its oracles" in prompt:
            calls.append("OracleRepair")
            prompts.append(prompt)
            reply = replies[min(len(prompts), len(replies)) - 1]
            return json.dumps({"deps": [], **reply})
        return await base(self, messages, **kw)

    return fake_chat


@pytest.mark.asyncio
async def test_with_one_check_disputed_and_one_not_only_the_undisputed_one_is_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two failing checks; the first repair blames check A and returns code (set aside: the code cannot be split per
    check). The next request names only B's failure and says A is not to be touched; its fix for B is applied."""
    original, bent, fixed_b = _two(0.5, 0.3), _two(1.0, 0.0), _two(0.5, 0.0)
    calls: list[str] = []
    prompts: list[str] = []
    replies = [{"code": bent, "patch_summary": "both", "oracle_change": _DISPUTE},
               # repeats the dispute it was told is already recorded: its fix for the other check is still used
               {"code": fixed_b, "patch_summary": "fixed the mass leak", "oracle_change": _DISPUTE}]
    monkeypatch.setattr("core.engine.LLMClient.chat", _scripted_repairs(calls, prompts, original, replies))
    # One repair in the budget: the set-aside answer does not use it up, so the undisputed check is still repaired.
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=1))
    await engine.run()
    assert (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == fixed_b
    assert calls.count("OracleRepair") == 2
    wrong_list = prompts[1].split("What went wrong:", 1)[1].split("The contract:", 1)[0]
    assert "'mass conservation' failed" in wrong_list and "'final size closed form' failed" not in wrong_list
    assert "Do not change the script for these checks" in prompts[1] and "final size closed form" in prompts[1]
    record = _record(engine)
    assert [a.get("repair") for a in record["attempts"]] == ["set_aside_disputed", "applied", None]
    assert record["status"] == "went_on_failing" and record["disputed"] == [ORACLE["name"]]
    assert len(record["problems"]) == 1 and "'final size closed form' failed" in record["problems"][0]


@pytest.mark.asyncio
async def test_a_later_repair_that_makes_a_disputed_check_pass_is_undone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Told to leave the disputed check alone, a repair fixes the other one AND bends the disputed one to the declared
    number. The gate must not end `ok` on that: the script is put back and the other check is asked for again."""
    original, bent, fixed_b = _two(0.5, 0.3), _two(1.0, 0.0), _two(0.5, 0.0)
    calls: list[str] = []
    prompts: list[str] = []
    replies = [{"code": original, "patch_summary": "the check", "oracle_change": _DISPUTE},
               {"code": bent, "patch_summary": "fixed both", "oracle_change": _DISPUTE},
               {"code": fixed_b, "patch_summary": "fixed the mass leak"}]
    monkeypatch.setattr("core.engine.LLMClient.chat", _scripted_repairs(calls, prompts, original, replies))
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=2))
    await engine.run()
    assert (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == fixed_b
    record = _record(engine)
    assert record["status"] == "went_on_failing" and record["disputed"] == [ORACLE["name"]]
    assert [a.get("repair") for a in record["attempts"]] == [
        "set_aside_disputed", "reverted_disputed_changed", None, "applied", None]
    assert "[oracle] put experiment.py back as it was" in (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_disputed_check_whose_measurement_was_dropped_is_not_restored_bent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One step longer than the bend above: a repair drops the disputed check's measurement (then it is sent back to be
    measured again, as the script's own problem), and the next repair restores it bent to the declared number."""
    original, fixed_b = _two(0.5, 0.3), _two(0.5, 0.0)
    dropped_a = _HEAD + (
        'if os.environ.get("FI_ORACLE") == "1":\n'
        '    print("ORACLE_JSON: " + json.dumps({"checks": [{"name": "mass conservation", "value": 0.0}]}))\n'
        '    raise SystemExit(0)\n'
    ) + _TAIL
    bent = _two(1.0, 0.0)
    calls: list[str] = []
    prompts: list[str] = []
    replies = [{"code": original, "patch_summary": "the check", "oracle_change": _DISPUTE},
               {"code": dropped_a, "patch_summary": "fixed the leak", "oracle_change": _DISPUTE},
               {"code": bent, "patch_summary": "measured it again"},
               {"code": fixed_b, "patch_summary": "measured it again, honestly"}]
    monkeypatch.setattr("core.engine.LLMClient.chat", _scripted_repairs(calls, prompts, original, replies))
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=3))
    await engine.run()
    record = _record(engine)
    assert record["status"] == "went_on_failing", "a disputed check must never end up passing through a repair"
    assert "reverted_disputed_changed" in [a.get("repair") for a in record["attempts"]]
    assert (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == fixed_b
    assert "was not checked" in prompts[2] and "restore its honest measurement" in prompts[2]


@pytest.mark.asyncio
async def test_a_dispute_about_a_check_that_passes_does_not_throw_away_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    passes_a, fixed_b = _two(1.0, 0.3), _two(1.0, 0.0)
    calls: list[str] = []
    prompts: list[str] = []
    replies = [{"code": fixed_b, "patch_summary": "fixed the mass leak", "oracle_change": _DISPUTE}]
    monkeypatch.setattr("core.engine.LLMClient.chat", _scripted_repairs(calls, prompts, passes_a, replies))
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=1))
    await engine.run()
    record = _record(engine)
    assert record["status"] == "ok" and record["attempts"][0]["repair"] == "applied"
    assert "disputed" not in record and record["proposed_changes"][0]["name"] == ORACLE["name"]


def test_only_a_disputed_checks_value_outside_tolerance_is_set_aside() -> None:
    """A dispute is about the expected value: a disputed check that was not measured, or whose value is not a number,
    is still the script's to fix, and so is a crash that names no check."""
    name = ORACLE["name"]
    found = [f"the oracle {name!r} failed: the script measured 0.5, the protocol expects 1 within 0.05",
             f"the declared oracle {name!r} was not checked (the script reported: nothing)",
             f"the oracle {name!r} reported no finite numeric `value` (it reported None)",
             "the script printed no `ORACLE_JSON:` line when run with FI_ORACLE=1 (exit code 1)"]
    assert oc.undisputed(found, [name]) == found[1:]
    assert oc.undisputed(found, []) == found
    judged = [{"name": name, "passed_by_engine": None}, {"name": "x", "passed_by_engine": False}]
    assert oc.disputed_failing(judged, [name]) == []  # never measured: nothing was judged against the disputed value
    assert "disputed" not in oc.analysis_note([{**judged[0], "disputed_expected": 0.5}])


def test_the_repair_request_says_a_wrong_check_is_reported_not_coded_around() -> None:
    text = oc.directive([ORACLE], ["the oracle 'final size closed form' failed"])
    assert "do NOT change the code to match it" in text and "leave `code` empty" in text
    assert "Do not change the script for these checks" not in text
    excluded = oc.directive([ORACLE], [], disputed=[ORACLE["name"]])
    assert "Do not change the script for these checks" in excluded and "'final size closed form'" in excluded


def test_a_proposed_change_pasted_into_a_shell_only_ever_carries_text() -> None:
    """The reason and the check are the model's words, shown inside a double-quoted command a person copies."""
    bad = {"name": ORACLE["name"], "expected": 1.0, "tolerance": 0.6, "tolerance_mode": "absolute",
           "check": 'run `whoami` then "quit"\nnext line', "reason": 'error is 0.5"; rm -rf ~; echo "$(id) \\ok! “x” „y‟'}
    request = oc.proposal_request(bad)
    for hazard in ('"', "$", "`", "\\", "\n", "!", "“", "”", "„", "‟"):  # the curly ones end a string in PowerShell
        assert hazard not in request, hazard
    assert "error is 0.5'; rm -rf ~; echo '(id) ok. 'x' 'y' Change nothing else." in request and "run whoami then 'quit' next line" in request


@pytest.mark.asyncio
async def test_a_record_the_gate_did_not_write_this_run_never_reaches_the_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the gate off, an earlier run's ORACLE_CHECK.json would otherwise hand the analysis stale verdicts."""
    prompts: list[str] = []
    inner = _fake([], implement=_UNAWARE, protocol=_PROTOCOL)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompts.append(messages[-1]["content"])
        return await inner(self, messages, **kw)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path, oracle_check="off"))
    stale = engine.quest_root / "needs" / "ORACLE_CHECK.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"status": "ok", "attempts": [{"judged": [
        {"name": "old", "value": 9.87, "expected": 9.87, "limit": 0.1, "passed_by_engine": True}]}]}), encoding="utf-8")
    assert (await engine.run()).paper_md is not None
    assert not stale.exists()
    assert not any("the engine checked the protocol's oracles" in p for p in prompts)


def test_the_analysis_is_told_the_engines_oracle_verdicts_not_the_scripts() -> None:
    """A live quest: the person corrected a tolerance, the engine's check passed under it, and the paper still said the
    oracle failed -- the script's results re-judged it with the old tolerance written into the script."""
    judged = [{"name": "energy drift", "value": 1.25e-05, "expected": 0, "limit": 2e-05, "passed_by_engine": True},
              {"name": "order", "value": None, "expected": 1, "limit": 0.15, "passed_by_engine": None}]
    record = {"status": "ok", "attempts": [{"judged": [{"name": "energy drift", "passed_by_engine": False}]}, {"judged": judged}]}
    assert oc.last_judged(record) == judged
    assert oc.last_judged({**record, "status": "stopped"}) == [] and oc.last_judged(None) == [] and oc.last_judged({"status": "ok"}) == []
    note = oc.analysis_note(judged)
    assert "- energy drift: measured 1.25e-05, expected 0 within 2e-05: passed" in note and "- order: not judged" in note
    assert "do not report an oracle as failed or passed on the script's word" in note
    assert oc.analysis_note([]) == ""


@pytest.mark.asyncio
async def test_the_analyze_prompt_carries_the_engines_verdicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    inner = _fake([], implement=_PASSING, protocol=protocol)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompts.append(messages[-1]["content"])
        return await inner(self, messages, **kw)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path))
    assert (await engine.run()).paper_md is not None
    carrying = [p for p in prompts if "[FI NOTE] Before the main run the engine checked the protocol's oracles" in p]
    assert carrying and "- final size closed form: measured 0.99, expected 1 within 0.05: passed" in carrying[0]


# --- an oracle the engine added after the plan was read is never recorded as a person's approval ---------------------------


def _frozen_record(engine: Engine) -> dict[str, Any] | None:
    path = engine.quest_root / "needs" / "FROZEN_PROTOCOL.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@pytest.mark.asyncio
async def test_an_oracle_the_engine_adds_after_the_plan_was_read_is_said_never_stopped_for_and_never_called_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """With ``pauses.plan: ask`` the person read a plan with no oracle; the gate then had the model add one. Judging a
    check's expected value is not something a person is asked to do (FI checks it itself when it runs), so the quest
    does not stop again for it: one line says so, and the freeze record says nobody approved that oracle, never that
    the person did."""
    from core.config import PausesConfig

    calls: list[str] = []
    # The script has no oracle branch yet: the gate adds the oracle and repairs the script.
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_UNAWARE, repair=_PASSING))
    cfg = _cfg(tmp_path)
    cfg.pauses = PausesConfig(plan="ask")
    first = Engine(cfg)
    await first.run()
    assert "read and edit the plan" in (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")

    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()
    assert calls.count("PlanRevise") == 1 and calls.count("OracleRepair") == 1
    assert artifacts.paper_md is not None, "no second stop for the check FI added"
    assert "Nothing to do: FI checks their expected values itself" in capsys.readouterr().out
    record = _frozen_record(second)
    assert record is not None and record["approved_by"].startswith("auto:")
    assert "'final size closed form' were added by the engine" in record["approved_by"]
    assert "nobody approved them" in record["approved_by"] and "held for the person to read" in record["approved_by"]
    assert record["protocol"]["oracles"] == [ORACLE]
    # The repair is the script the quest carries on with, not the one before it.
    assert "FI_ORACLE" in artifacts.raw_state["code"]


@pytest.mark.asyncio
async def test_without_the_plan_stop_an_oracle_the_engine_added_is_recorded_as_nobodys_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_PASSING))
    engine = Engine(_cfg(tmp_path))
    assert engine.config.pauses.plan == "off"
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and calls.count("PlanRevise") == 1
    record = _frozen_record(engine)
    assert record is not None and record["approved_by"].startswith("auto:")
    assert "added by the engine after the plan was written" in record["approved_by"]
    assert "final size closed form" in record["approved_by"] and "nobody approved" in record["approved_by"]


def _freeze_with(tmp_path: Path, *, plan_mode: str, held: bool, added: dict[str, Any] | None,
                 replaced_by: str | None = None) -> str:
    """What ``approved_by`` the freeze writes for a protocol with ORACLE, given the plan stop and the gate's record."""
    from core import frozen_protocol
    from core.config import PausesConfig

    cfg = _cfg(tmp_path)
    cfg.pauses = PausesConfig(plan=plan_mode)
    engine = Engine(cfg)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    if replaced_by:  # a rerun from the design, approved by a person, after an earlier freeze
        frozen_protocol.freeze(engine.quest_root, _PROTOCOL, approved_by="auto", source="plan.md")
        frozen_protocol.record_replacement(engine.quest_root, approved_by=replaced_by, step="design")
        frozen_protocol.frozen_path(engine.quest_root).unlink()
    if held:
        (engine.fi_dir / "paused_at_plan.flag").write_text("plan", encoding="utf-8")
    if added is not None:
        (engine.fi_dir / "oracles_added.json").write_text(json.dumps(added), encoding="utf-8")
    engine._freeze_protocol_if_due({"design": {"protocol": protocol}, "iteration": 0})
    return str(frozen_protocol.load(engine.quest_root)["approved_by"])


def test_the_freeze_never_says_a_person_approved_what_they_were_not_shown(tmp_path: Path) -> None:
    unseen = {"oracles": [ORACLE["name"]], "shown": False}
    seen = {"oracles": [ORACLE["name"]], "shown": True}
    # A person approved going back to the design before the engine added the oracle: that is said, and so is the oracle.
    replaced = _freeze_with(tmp_path / "r", plan_mode="off", held=False, added=unseen, replaced_by="Jun")
    assert replaced.startswith("human: Jun approved replacing") and "nobody approved them" in replaced
    # The setting says ask, but the quest never stopped at the plan: nobody read it.
    assert _freeze_with(tmp_path / "a", plan_mode="ask", held=False, added=None).startswith("auto:")
    assert _freeze_with(tmp_path / "h", plan_mode="ask", held=True, added=None).startswith("human:")
    # Shown at the stop for the added checks, then the plan stop was turned off: the oracles were still shown.
    shown_then_off = _freeze_with(tmp_path / "s", plan_mode="off", held=True, added=seen)
    assert shown_then_off.startswith("auto:") and "held for the person to read" in shown_then_off
    # A name the person removed at the stop is not named as part of the frozen protocol.
    gone = _freeze_with(tmp_path / "g", plan_mode="off", held=False, added={"oracles": ["removed at the stop"], "shown": False})
    assert "removed at the stop" not in gone and gone.startswith("auto: nobody approved this protocol")


# --- a repair whose code cannot be used does not use one up (a live gemma4 quest, 2026-10-05) ----------------------------


def _repairs_in_turn(calls: list[str], replies: list[str], protocol: dict[str, Any]):
    inner = _fake(calls, implement=_FAILING, repair=_PASSING, protocol=protocol)
    sent: list[int] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "ExecuteReflect" and "FI_ORACLE=1 to check its oracles" in prompt:
            calls.append("OracleRepair")
            reply = replies[min(len(sent), len(replies) - 1)]
            sent.append(1)
            return json.dumps({"code": reply, "deps": [], "patch_summary": "a repair"})
        return await inner(self, messages, **kw)

    return fake_chat


@pytest.mark.asyncio
async def test_a_repair_whose_code_cannot_be_used_does_not_use_one_up_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With one repair allowed: the first repair's code does not even parse (FI keeps the script as it is), so it is not
    counted, and the next one, which passes, is still made."""
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat",
                        _repairs_in_turn(calls, ["def broken(:\n    pass\n", _PASSING], protocol))
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=1))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert calls.count("OracleRepair") == 2
    assert _record(engine)["status"] == "ok"
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "does not count as one of the 1 repairs" in log


@pytest.mark.asyncio
async def test_unusable_repairs_are_exempt_only_once_so_the_loop_stays_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    protocol = {**_PROTOCOL, "oracles": [ORACLE]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _repairs_in_turn(calls, ["def broken(:\n"], protocol))
    engine = Engine(_cfg(tmp_path, oracle_repair_attempts=1))
    await engine.run()
    # One free, then the one allowed: two calls, never more.
    assert calls.count("OracleRepair") == 2
    assert _record(engine)["status"] != "ok"
