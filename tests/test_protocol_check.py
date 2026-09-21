"""The plan's protocol: what the script may not change on its own (core/protocol_check.py) and the gate that holds it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import plan, protocol_check as pc
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_EXPERIMENT_CODE, _FAKE_RESPONSES, _classify, _fake_response_for

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "protocol_runs"
TOPIC_PROTOCOL: dict[str, Any] = {
    "grid": {"R0": [0.9, 1.5, 3.0], "N": [100, 1000, 5000]},
    "runs_per_setting": 300,
    "thresholds": {"major_outbreak_fraction": 0.05},
}


def _real(name: str) -> str:
    return (FIXTURES / f"sir_{name}_experiment.py").read_text(encoding="utf-8")


# --- the three real runs the audit was written about -----------------------------------------------------------------


def test_the_run_that_changed_its_grid_is_caught_and_the_two_that_did_not_are_not() -> None:
    """fr3 swept R0 over {0.8, 1.2, 2.0, 3.0} and N over {100, 500, 1000, 5000} for a topic that asked for
    {0.9, 1.5, 3.0} and {100, 1000, 5000}, and was accepted with the top score. fr1 and fr2 kept the topic's grid (and
    fr2's FI_PILOT branch, which shrinks it for the smoke test, is not read)."""
    assert pc.check(TOPIC_PROTOCOL, {"experiment.py": _real("fr1")}) == []
    assert pc.check(TOPIC_PROTOCOL, {"experiment.py": _real("fr2")}) == []
    found = pc.check(TOPIC_PROTOCOL, {"experiment.py": _real("fr3")})
    assert {(m.kind, m.name) for m in found} == {("grid", "R0"), ("grid", "N")}
    r0 = next(m for m in found if m.name == "R0")
    assert r0.found == [0.8, 1.2, 2.0, 3.0] and "R0_LIST" in r0.where and "line 166" in r0.where
    assert "adds [0.8, 1.2, 2] and leaves out [0.9, 1.5]" in r0.message()
    assert "adds 500" in next(m for m in found if m.name == "N").message()


# --- what counts as a difference --------------------------------------------------------------------------------------


def test_a_script_that_agrees_with_the_protocol_has_no_differences() -> None:
    code = "R0_LIST = [0.9, 1.5, 3.0]\nN_LIST = (100, 1000, 5000)\nNUM_RUNS = 300\nTHRESHOLD = 0.05\n"
    assert pc.check(TOPIC_PROTOCOL, {"experiment.py": code}) == []


def test_the_runs_per_setting_are_read_from_a_count_the_script_names() -> None:
    found = pc.check({"runs_per_setting": 300}, {"experiment.py": "NUM_RUNS = 100\nn_grid = 12\n"})
    assert [(m.kind, m.found) for m in found] == [("runs", [100.0])]
    assert pc.check({"runs_per_setting": 300}, {"experiment.py": "n_samples = 300\n"}) == []
    # A number the script does not name as a count of runs is no evidence either way.
    assert pc.check({"runs_per_setting": 300}, {"experiment.py": "steps = 100\n"}) == []


def test_a_threshold_is_missing_only_when_its_value_appears_nowhere() -> None:
    proto = {"thresholds": {"major": 0.05}}
    assert pc.check(proto, {"experiment.py": "cut = 0.10\n"})[0].kind == "threshold"
    assert pc.check(proto, {"experiment.py": "cut = 0.05\n"}) == []
    assert pc.check(proto, {"experiment.py": "cut_percent = 5\n"}) == [], "5 per cent is 0.05"


def test_ranges_dicts_and_loops_over_literals_are_read_as_the_axis() -> None:
    proto = {"grid": {"dose": [1.0, 2.0, 3.0]}}
    assert pc.check(proto, {"e.py": "dose_values = np.linspace(1, 3, 3)\n"}) == []
    assert pc.check(proto, {"e.py": "for dose in (1.0, 2.0, 3.0):\n    pass\n"}) == []
    assert pc.check(proto, {"e.py": "CONFIG = {'dose': [1.0, 2.0, 3.0]}\n"}) == []
    assert pc.check(proto, {"e.py": "for dose in (1.0, 2.0, 4.0):\n    pass\n"})[0].kind == "grid"
    assert pc.check(proto, {"e.py": "dose_values = list(range(1, 5))\n"})[0].found == [1.0, 2.0, 3.0, 4.0]


def test_a_list_is_matched_by_its_values_when_no_name_says_which_axis_it_is() -> None:
    found = pc.check({"grid": {"R0": [0.9, 1.5, 3.0]}}, {"e.py": "vals = [0.9, 1.5, 2.5]\n"})
    assert found and found[0].found == [0.9, 1.5, 2.5]
    assert pc.check({"grid": {"R0": [0.9, 1.5, 3.0]}}, {"e.py": "vals = [10, 20, 30]\n"})[0].missing == [0.9, 1.5, 3.0]


def test_an_axis_the_script_never_names_is_missing_only_if_its_values_are_absent() -> None:
    assert pc.check({"grid": {"R0": [0.9, 1.5]}}, {"e.py": "def f(r0=0.9):\n    return r0 * 1.5\n"}) == []


def test_the_smoke_branch_is_not_read_but_the_real_branch_is() -> None:
    code = (
        "R0 = [0.9, 1.5, 3.0]\n"
        "if os.environ.get('FI_PILOT') == '1':\n    R0 = [0.9]\n    NUM_RUNS = 5\nelse:\n    NUM_RUNS = 300\n"
    )
    assert pc.check({"grid": {"R0": [0.9, 1.5, 3.0]}, "runs_per_setting": 300}, {"e.py": code}) == []
    inverted = "if os.environ.get('FI_PILOT') != '1':\n    NUM_RUNS = 50\n"
    assert pc.check({"runs_per_setting": 300}, {"e.py": inverted})[0].found == [50.0]


def test_two_scripts_are_read_together_and_a_script_that_does_not_parse_is_not_a_difference() -> None:
    proto = {"grid": {"R0": [0.9, 1.5]}, "runs_per_setting": 300}
    sim = "R0_LIST = [0.9, 1.5]\nNUM_RUNS = 300\n"
    assert pc.check(proto, {"simulate.py": sim, "experiment.py": "print(1)\n"}) == []
    broken = pc.check(proto, {"experiment.py": "def (:\n R0 = 0.9 1.5\n"})
    assert all(m.kind != "runs" for m in broken)


def test_a_protocol_with_nothing_the_check_reads_finds_nothing() -> None:
    assert pc.check(None, {"e.py": "x = 1\n"}) == []
    assert pc.check({"seed_policy": "one stream per run", "acceptance": ["x"]}, {"e.py": "x = 1\n"}) == []
    assert pc.check(TOPIC_PROTOCOL, {}) == []


# --- what the plan says about the topic ---------------------------------------------------------------------------------


def test_the_plan_says_which_numbers_the_topic_sets_and_the_protocol_leaves_out() -> None:
    topic = "For R0 in {0.9, 1.5, 3.0} and N in {100, 1000, 5000}, run 300 stochastic runs per setting."
    assert pc.plan_notes(topic, TOPIC_PROTOCOL) == []
    lacking = {"grid": {"R0": [0.9, 3.0], "N": [100, 1000, 5000]}, "runs_per_setting": 300}
    notes = pc.plan_notes(topic, lacking)
    assert len(notes) == 1 and "1.5" in notes[0] and "does not contain it" in notes[0]
    assert "no protocol block" in pc.plan_notes(topic, None)[0]
    assert pc.plan_notes("Compare two integrators.", None) == []


# --- the protocol block in plan.md ------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad, why", [
    ("not a mapping", "mapping"),
    ({"grid": ["R0"]}, "grid"),
    ({"grid": {"R0": []}}, "non-empty"),
    ({"grid": {"R0": ["a"]}}, "numbers"),
    ({"runs_per_setting": 2.5}, "whole number"),
    ({"runs_per_setting": 0}, "whole number"),
    ({"thresholds": {"t": "high"}}, "thresholds"),
    ({"ci_method": ["x"]}, "ci_method"),
])
def test_a_protocol_that_could_not_be_checked_is_refused(bad: Any, why: str) -> None:
    design, error = plan.normalize_design({"hypothesis": "h", "protocol": bad})
    assert design is None and why in (error or "")


def test_a_protocol_survives_the_plan_file_and_a_single_value_becomes_a_list() -> None:
    design, _ = plan.normalize_design({
        "hypothesis": "h",
        "protocol": {**TOPIC_PROTOCOL, "grid": {"R0": 1.5}, "acceptance": "converges; within 0.02", "extra": 1},
    })
    assert design["protocol"]["grid"] == {"R0": [1.5]}
    assert design["protocol"]["acceptance"] == ["converges", "within 0.02"] and design["protocol"]["extra"] == 1
    assert plan.parse(plan.render("t", {}, design)).design["protocol"] == design["protocol"]


# --- the gate, through the real graph and a real checkpoint -----------------------------------------------------------

_GOOD = "R0_LIST = [0.9, 1.5, 3.0]\nNUM_RUNS = 300\n" + _FAKE_EXPERIMENT_CODE
_BAD = "R0_LIST = [0.8, 1.2, 2.0, 3.0]\nNUM_RUNS = 300\n" + _FAKE_EXPERIMENT_CODE
_PROTOCOL = {"grid": {"R0": [0.9, 1.5, 3.0]}, "runs_per_setting": 300}


def _cfg(tmp_path: Path, **engine: Any) -> Config:
    return Config(
        topic="smoke-test topic for the protocol: R0 in {0.9, 1.5, 3.0}, 300 runs each",
        title="protocol-smoke",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def _fake(calls: list[str], *, implement: str, repair: str | None, protocol: dict[str, Any] | None = _PROTOCOL):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            if protocol is not None:
                body["protocol"] = protocol
            return json.dumps(body)
        if kind == "Implementation":
            return json.dumps({"code": implement, "deps": ["matplotlib"]})
        if kind == "ExecuteReflect" and "experiment protocol that was fixed in the plan" in prompt:
            calls.append("ProtocolRepair")
            return json.dumps({"code": repair, "deps": [], "patch_summary": "used the protocol's grid"}) if repair else "{}"
        return _fake_response_for(prompt)

    return fake_chat


@pytest.mark.asyncio
async def test_a_script_that_changed_the_grid_is_sent_back_and_the_repaired_one_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_BAD, repair=_GOOD))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()

    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert calls.count("ProtocolRepair") == 1
    assert "R0_LIST = [0.9, 1.5, 3.0]" in (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
    record = json.loads((engine.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == "ok" and len(record["attempts"]) == 2
    assert "adds [0.8, 1.2, 2] and leaves out [0.9, 1.5]" in record["attempts"][0]["differences"][0]
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[protocol] the script differs" in log and "holds to the plan's protocol (after 1 repair(s))" in log


@pytest.mark.asyncio
async def test_a_script_that_agrees_is_left_alone_and_costs_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_GOOD, repair=None))
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    assert "ProtocolRepair" not in calls
    assert json.loads((engine.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8"))["status"] == "ok"


@pytest.mark.asyncio
async def test_a_quest_with_no_protocol_or_the_check_off_is_not_looked_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for extra, protocol in (({}, None), ({"protocol_check": "off"}, _PROTOCOL)):
        calls: list[str] = []
        monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_BAD, repair=None, protocol=protocol))
        engine = Engine(_cfg(tmp_path / str(len(extra)), **extra))
        artifacts = await engine.run()
        assert artifacts.paper_md is not None
        assert "ProtocolRepair" not in calls
        assert not (engine.quest_root / "needs" / "PROTOCOL_CHECK.json").exists()


@pytest.mark.asyncio
async def test_warn_records_the_difference_and_the_quest_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_BAD, repair=None))
    engine = Engine(_cfg(tmp_path, protocol_check="warn", protocol_repair_attempts=1))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and calls.count("ProtocolRepair") == 1
    record = json.loads((engine.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == "warned" and record["differences"]


@pytest.mark.asyncio
async def test_a_script_that_still_differs_stops_the_quest_and_a_fixed_one_is_kept_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stop is its own pause (not a clarify), says what differs, and a script the person fixed is used as it is:
    the model is not asked to write it again."""
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_BAD, repair=_BAD))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()

    log = (first.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert calls.count("ProtocolRepair") == 2, "two repairs were asked for before stopping"
    assert "[protocol] paused" in log and "[FI] paused for the protocol" in log and "paused for clarify" not in log
    assert not (first.fi_dir / "clarify_questions.json").exists()
    descriptor = json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert descriptor["kind"] == "protocol" and descriptor["interaction"] == "supply"
    text = (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "adds [0.8, 1.2, 2] and leaves out [0.9, 1.5]" in text and "plan.md" in text
    assert json.loads((first.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8"))["status"] == "stopped"
    assert not (first.quest_root / "paper" / "paper.md").exists()

    (first.quest_root / "code" / "experiment.py").write_text(_GOOD, encoding="utf-8")
    implementations_before = calls.count("Implementation")

    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()

    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert calls.count("Implementation") == implementations_before, "the fixed script was written again"
    assert "agrees with the plan's protocol; keeping it" in (second.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert not (second.fi_dir / "protocol_stop.json").exists()


@pytest.mark.asyncio
async def test_an_unfixed_script_is_written_again_on_resume_and_the_old_stop_is_forgotten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_BAD, repair=_BAD))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()
    assert (first.fi_dir / "protocol_stop.json").is_file()
    before = calls.count("Implementation")

    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_GOOD, repair=None))
    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()

    assert artifacts.paper_md is not None
    assert calls.count("Implementation") == before + 1, "a script that still differed is written again"
    assert not (second.fi_dir / "protocol_stop.json").exists(), "a later pass must not adopt an old script"


@pytest.mark.asyncio
async def test_the_plan_names_the_topic_numbers_its_protocol_leaves_out(tmp_path: Path) -> None:
    calls: list[str] = []
    lacking = {"grid": {"R0": [0.9, 3.0]}, "runs_per_setting": 300}
    engine = Engine(_cfg(tmp_path))
    engine._client = type("Stub", (), {"chat": _fake(calls, implement=_GOOD, repair=None, protocol=lacking)})()
    await engine._node_plan({"topic": engine.config.topic, "iteration": 0})
    text = plan.plan_path(engine.quest_root).read_text(encoding="utf-8")
    assert "The topic sets 1.5" in text and "the protocol does not contain it" in text
    assert plan.parse(text).design["protocol"]["grid"] == {"R0": [0.9, 3.0]}
    prompts = [c for c in calls if c == "Experiment Design"]
    assert len(prompts) == 1


def test_the_review_notes_carry_a_difference_that_appeared_after_the_script_was_written(tmp_path: Path) -> None:
    """A repair after a failed run can change a constant; the review's advisory notes say so."""
    engine = Engine(_cfg(tmp_path))
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "code" / "experiment.py").write_text(_BAD, encoding="utf-8")
    state = {
        "topic": engine.config.topic, "result_json": {"score": 1.0}, "figures": ["a.png"],
        "design": {"hypothesis": "h", "protocol": _PROTOCOL},
    }
    notes = engine._goal_coverage_notes(state)
    assert any("differs from the plan's protocol" in n and "R0" in n for n in notes)
    engine.config.engine.protocol_check = "off"
    assert not any("differs from the plan's protocol" in n for n in engine._goal_coverage_notes(state))


def test_the_dashboard_and_the_quest_page_name_the_protocol_stop() -> None:
    static = Path(__file__).resolve().parent.parent / "web" / "static"
    for page in ("index.html", "quest.html"):
        assert "protocol: 'protocol'" in (static / page).read_text(encoding="utf-8"), page
