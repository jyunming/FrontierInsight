"""A study FI starts from a commission file instead of a topic, and hands back when its run cannot fit.

The rules a commission must meet are refused at the door, before any model call (core/commission.py ``problems``); a
commission is frozen into the quest with its full texts and its hash; a resume with another commission stops before
any step; making ideas and choosing a study are skipped; the plan's model is shown what the commission fixes, and
plan.md says which known value the plan has no check for; a run FI times as too long is handed back
(``needs/STUCK.json`` ``return_reason`` first) instead of the plan being rewritten smaller.

The topic of these tests is a made-up draining tank.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from tests import exact_timing

PASSAGE = "The drain constant of the reference tank is 0.25 per hour under the stated inlet conditions"
CASE = "the reference tank, half full, outlet open"
SOURCE_TEXT = ("A made-up note on a draining tank, used only to test FI. " * 6) + PASSAGE + ". More text follows here."


def _commission(folder: Path, **changes: Any) -> Path:
    """A valid commission in ``folder`` with its one full text, changed by ``changes`` (``None`` removes a key)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tank_note.md").write_text(SOURCE_TEXT, encoding="utf-8")
    data: dict[str, Any] = {
        "question": "How fast does a tank drain through outlets of three sizes?",
        "scope": "three outlet diameters, one fill level",
        "limits": {"hours": 2, "model_calls": 60},
        "known_values": [{"name": "drain constant", "value": 0.25, "tolerance": 0.01, "case": CASE, "passage": PASSAGE,
                          "source": {"title": "A made-up tank note", "doi": "10.0000/tank.1", "file": "tank_note.md"}}],
        "success": [{"condition": "the drain time falls with the outlet size", "verify": "compare the three runs"}],
        "failure": [{"condition": "the drain time does not depend on the outlet", "verify": "compare the three runs"}],
        "doubts": [],
    }
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    path = folder / "commission.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _known(**changes: Any) -> list[dict[str, Any]]:
    kv: dict[str, Any] = {"name": "drain constant", "value": 0.25, "tolerance": 0.01, "case": CASE, "passage": PASSAGE,
                          "source": {"title": "A made-up tank note", "file": "tank_note.md"}}
    for key, value in changes.items():
        if value is None:
            kv.pop(key, None)
        else:
            kv[key] = value
    return [kv]


def _engine(tmp_path: Path, commission: Path | None, **engine: Any) -> Any:
    from core.engine import Engine

    eng = Engine(Config(
        commission=commission, provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=AssertionError("no model call is made")),
                                    "aclose": AsyncMock()})()
    return eng


# ---- refused at the door ---------------------------------------------------------------------------------------------

REFUSALS: list[tuple[str, dict[str, Any], str]] = [
    ("no question", {"question": None}, "`question`"),
    ("no scope", {"scope": None}, "`scope`"),
    ("no limits", {"limits": None}, "`limits`"),
    ("hours not a number", {"limits": {"hours": "two", "model_calls": 60}}, "`limits.hours`"),
    ("no model_calls", {"limits": {"hours": 2}}, "`limits.model_calls`"),
    ("no known_values", {"known_values": None}, "`known_values`"),
    ("empty known_values unsaid", {"known_values": []}, "`no_known_values: true`"),
    ("no tolerance", {"known_values": _known(tolerance=None)}, "no `tolerance` for the known value 'drain constant'"),
    ("tolerance not a number", {"known_values": _known(tolerance="1%")}, "`tolerance` of the known value 'drain constant'"),
    ("value not a number", {"known_values": _known(value="0.25")}, "`value` of the known value 'drain constant'"),
    ("no passage", {"known_values": _known(passage=None)}, "'drain constant' has no `passage`"),
    ("no source", {"known_values": _known(source=None)}, "'drain constant' has no `source`"),
    ("no source file", {"known_values": _known(source={"title": "t"})}, "has no `file`"),
    ("no case", {"known_values": _known(case=None)}, "'drain constant' has no `case`"),
    ("empty case", {"known_values": _known(case="  ")}, "'drain constant' has no `case`"),
    ("value not in its passage", {"known_values": _known(value=0.3)},
     "passage of the known value 'drain constant' does not state its value 0.3"),
    ("no success", {"success": None}, "`success`"),
    ("no failure", {"failure": None}, "`failure`"),
    ("no doubts", {"doubts": None}, "`doubts`"),
    ("missing file", {"known_values": _known(source={"title": "t", "file": "gone.md"})}, "'gone.md'"),
    ("passage not in its file", {"known_values": _known(passage="The drain constant of the reference tank is 0.25 per minute")},
     "passage of the known value 'drain constant' is not in its source file"),
]


@pytest.mark.parametrize("why,changes,named", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_a_commission_that_breaks_a_rule_is_refused_naming_it(tmp_path: Path, why: str, changes: dict[str, Any],
                                                             named: str) -> None:
    from core import commission

    found = commission.problems(_commission(tmp_path, **changes))
    assert found and any(named in p for p in found), found
    with pytest.raises(commission.CommissionRefused, match="no study was started"):
        commission.validate(_commission(tmp_path, **changes))


def test_a_source_file_with_under_200_characters_of_text_is_refused(tmp_path: Path) -> None:
    from core import commission

    path = _commission(tmp_path)
    (tmp_path / "tank_note.md").write_text(PASSAGE, encoding="utf-8")
    assert any("'tank_note.md' has under 200 characters" in p for p in commission.problems(path))


def test_a_valid_commission_has_no_problem(tmp_path: Path) -> None:
    from core import commission

    assert commission.problems(_commission(tmp_path)) == []
    assert commission.problems(_commission(tmp_path, known_values=[], no_known_values=True)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("changes,named", [({"doubts": None}, "`doubts`"),
                                           ({"known_values": _known(passage="The drain constant is 0.25 per day, said nowhere")},
                                            "not in its source file")])
async def test_the_command_line_refuses_it_before_any_model_call_and_exits_nonzero(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        changes: dict[str, Any], named: str) -> None:
    import launch

    monkeypatch.setattr("core.engine.LLMClient.chat", AsyncMock(side_effect=AssertionError("no model call is made")))
    _commission(tmp_path / "order", **changes)
    yaml = tmp_path / "order" / "quest.yaml"
    yaml.write_text(f"commission: commission.json\nprovider:\n  name: ollama\noutput:\n  output_dir: "
                    f"{(tmp_path / 'outputs').as_posix()}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # the commission is read against the YAML's folder, not this one
    assert await launch.main_async(launch.parse_args(["--config", str(yaml), "--no-axon-sidecar"])) == 2
    err = capsys.readouterr().err
    assert named in err and "no study was started" in err and "Traceback" not in err
    assert not (tmp_path / "outputs").exists() or not list((tmp_path / "outputs").iterdir()), "no quest folder"


# ---- the config ------------------------------------------------------------------------------------------------------

def test_the_question_becomes_the_topic_and_only_silent_keys_are_turned_off(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    _commission(tmp_path / "order")
    yaml = tmp_path / "order" / "quest.yaml"
    yaml.write_text("commission: commission.json\npauses:\n  scope: ask\nengine:\n  clarify_mode: auto\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = Config.from_yaml(yaml)
    assert cfg.commission == (tmp_path / "order" / "commission.json").resolve()
    assert cfg.topic == "How fast does a tank drain through outlets of three sizes?"
    assert cfg.pauses.scope == "ask", "a key the config sets is kept"
    assert cfg.pauses.clarify == "auto", "a legacy pause flag the config sets counts as said"
    assert cfg.pauses.papers is False and cfg.knowledge.novelty_check is False
    assert cfg.knowledge.foundational_works is False and cfg.knowledge.requery_on_low_relevance is False
    assert cfg.engine.evidence_gate_max_broaden == 0 and cfg.engine.enable_analyze_reroute is False
    assert cfg.execution.timeout_s == 2 * 3600, "the commission's hours are the run's time limit when the config is silent"
    plain = Config(topic="a draining tank")
    assert plain.commission is None and plain.pauses.scope == "ask" and plain.engine.evidence_gate_max_broaden == 1


def test_a_commission_with_a_missing_key_is_refused_on_load_and_at_the_door(tmp_path: Path) -> None:
    from core import commission

    path = _commission(tmp_path / "order", question=None)
    yaml = tmp_path / "order" / "quest.yaml"
    yaml.write_text("commission: commission.json\n", encoding="utf-8")
    with pytest.raises(commission.CommissionRefused, match="`question`"):
        Config.from_yaml(yaml)
    with pytest.raises(commission.CommissionRefused, match="`question`"):
        commission.door(path, None)  # a config built in code is refused at the door every start path runs


# ---- frozen into the quest; a resume with another stops -----------------------------------------------------------------

class _Stop(Exception):
    pass


@pytest.mark.asyncio
async def test_the_commission_and_its_full_texts_are_copied_into_the_quest_with_its_hash(tmp_path: Path) -> None:
    from core import audit_log, commission

    path = _commission(tmp_path / "order")
    eng = _engine(tmp_path, path)

    async def stop(_root: Any) -> None:
        raise _Stop()

    eng.executor.setup = stop  # the run is stopped just after the inputs are staged
    with pytest.raises(_Stop):
        await eng.run()
    root = eng.quest_root
    assert (root / "inputs" / "commission.json").read_bytes() == path.read_bytes()
    assert (root / "inputs" / "papers" / "tank_note.md").read_text(encoding="utf-8") == SOURCE_TEXT
    events = [e for e in audit_log.read(root / ".fi" / "audit.jsonl") if e.get("kind") == "commission_recorded"]
    assert events and events[-1]["sha256"] == commission.sha256(path) == commission.recorded_sha(root)


def test_a_resume_with_a_changed_commission_stops_and_an_unchanged_one_goes_on(tmp_path: Path) -> None:
    from core import commission

    path = _commission(tmp_path / "order")
    root = tmp_path / "out" / "q1"
    commission.stage(path, root)
    commission.door(path, root)  # the same file: goes on
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "scope": "five outlet diameters"}), encoding="utf-8")
    with pytest.raises(commission.CommissionRefused, match="not the one quest q1 was started from"):
        commission.door(path, root)
    with pytest.raises(commission.CommissionRefused, match="no longer names one"):
        commission.door(None, root)


def test_a_commission_given_to_a_quest_that_already_has_a_plan_is_refused(tmp_path: Path) -> None:
    from core import commission

    root = tmp_path / "out" / "q2"
    root.mkdir(parents=True)
    (root / "plan.md").write_text("# a plan\n", encoding="utf-8")
    with pytest.raises(commission.CommissionRefused, match="already has a plan.md"):
        commission.door(_commission(tmp_path / "order"), root)


@pytest.mark.asyncio
async def test_the_command_line_stops_a_resume_with_a_changed_commission_before_the_quest_config_is_replaced(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import launch
    from core import commission

    monkeypatch.setattr("core.engine.LLMClient.chat", AsyncMock(side_effect=AssertionError("no model call is made")))
    path = _commission(tmp_path / "order")
    root = tmp_path / "outputs" / "q3"
    commission.stage(path, root)
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "state.sqlite").write_bytes(b"")
    own = "commission: commission.json\nprovider:\n  name: ollama\n"
    (root / "config.yaml").write_text(own, encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "scope": "five outlet diameters"}), encoding="utf-8")
    yaml = tmp_path / "order" / "quest.yaml"
    yaml.write_text(own + f"output:\n  output_dir: {(tmp_path / 'outputs').as_posix()}\n", encoding="utf-8")
    rc = await launch.main_async(launch.parse_args(["--config", str(yaml), "--resume", "q3", "--no-axon-sidecar"]))
    assert rc == 2 and "not the one quest q3 was started from" in capsys.readouterr().err
    assert (root / "config.yaml").read_text(encoding="utf-8") == own, "the quest's own config was not replaced"


def test_the_quest_s_own_config_finds_its_frozen_commission(tmp_path: Path) -> None:
    from core import commission

    root = tmp_path / "out" / "q4"
    commission.stage(_commission(tmp_path / "order"), root)
    (root / "config.yaml").write_text("commission: commission.json\n", encoding="utf-8")
    assert Config.from_yaml(root / "config.yaml").commission == commission.staged_path(root).resolve()


# ---- the steps the commission gives -------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ideate_and_literature_do_no_work_of_their_own_and_say_nothing_of_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import commission

    monkeypatch.setattr("core.engine.Knowledge.asearch", AsyncMock(side_effect=AssertionError("nothing is searched")))
    path = _commission(tmp_path / "order")
    eng = _engine(tmp_path, path)
    commission.stage(path, eng.quest_root)
    said: list[str] = []
    eng._progress = said.append  # type: ignore[method-assign]
    for node in ("ideate", "scope", "literature"):
        eng._progress_stage(node)
    assert said == ["[literature] Reading the sources the commission gives."]
    assert await eng._node_ideate({"topic": eng.config.topic}) == {}
    out = await eng._node_literature({"topic": eng.config.topic})
    [entry] = out["literature"]
    assert entry["metadata"]["source"] == "user_supplied" and PASSAGE in entry["content"]
    assert entry["metadata"]["title"] == "A made-up tank note" and entry["metadata"]["doi"] == "10.0000/tank.1"
    assert "literature_novelty" not in out and eng._client.chat.call_count == 0


@pytest.mark.asyncio
async def test_a_rerun_from_a_step_the_commission_gives_says_so_and_changes_nothing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    eng = _engine(tmp_path, _commission(tmp_path / "order"))

    async def no_env(*_a: Any, **_k: Any) -> None:
        return None

    eng.executor.setup = no_env
    eng._record_environment = no_env  # type: ignore[method-assign]
    eng._connect_llm = no_env  # type: ignore[method-assign]
    await eng.run(from_step="ideas")
    assert "the ideas step is not one FI did" in capsys.readouterr().out
    assert eng._client.chat.call_count == 0


# ---- the plan ------------------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_plan_is_shown_what_the_commission_fixes_and_plan_md_names_a_known_value_with_no_check(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import commission
    from core.engine import Engine
    from tests.test_plan_optimisation import EXTRA, _NO_OBJECTIONS
    from tests.test_run_checks import _draft

    async def off(self: Any, state: Any) -> None:
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)
    path = _commission(tmp_path / "order")
    eng = _engine(tmp_path, path)
    commission.stage(path, eng.quest_root)
    lit = await eng._node_literature({"topic": eng.config.topic})
    prompts: list[str] = []

    async def chat(messages: list[dict[str, str]], **_kw: Any) -> str:
        prompts.append(messages[-1]["content"])
        return json.dumps({**_draft({"n": [1, 2, 4]}, 4), "plan": EXTRA}) if len(prompts) == 1 else _NO_OBJECTIONS

    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=chat)})()
    await eng._node_plan({"topic": eng.config.topic, "iteration": 0, **lit})
    assert "WHAT THE COMMISSION FIXES" in prompts[0]
    assert "name 'drain constant': 0.25 with tolerance 0.01, for " + CASE + "; stated in [1]" in prompts[0] and PASSAGE in prompts[0]
    plan_md = (eng.quest_root / "plan.md").read_text(encoding="utf-8")
    assert "known value 'drain constant' (0.25 with tolerance 0.01) has no check in this plan" in plan_md
    assert commission.oracle_gaps(commission.read(path), {"oracles": [{"name": "Drain  Constant", "kind": "published_value"}]}) == []


# ---- handed back -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reason", ["known_value_mismatch", "plan_contradicts_sources", "scope_exceeds_cap"])
def test_the_return_record_leads_with_its_reason_and_quality_json_says_it(tmp_path: Path, reason: str) -> None:
    from core import quality_report

    eng = _engine(tmp_path, _commission(tmp_path / "order"))
    eng._return_to_qc(reason, "the study cannot be done as commissioned", {"seconds_per_setting": 3.0}, ["timed it"])
    text = (eng.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8")
    record = json.loads(text)
    assert list(record)[:3] == ["return_reason", "kind", "evidence"] and text.lstrip("{\n ").startswith('"return_reason"')
    assert record["return_reason"] == reason and record["kind"] == "returned"
    assert record["evidence"] == {"seconds_per_setting": 3.0} and record["problem"] and record["say"]
    outcome = quality_report.build(eng.quest_root, running=False)["outcome"]
    assert outcome["kind"] == "honest_stop" and outcome["return_reason"] == reason
    assert outcome["line"].startswith(f"FI handed the commissioned study back ({reason}): the study cannot be done")


def test_a_reason_that_is_not_one_of_the_three_is_refused(tmp_path: Path) -> None:
    eng = _engine(tmp_path, _commission(tmp_path / "order"))
    with pytest.raises(ValueError, match="scope_exceeds_cap"):
        eng._return_to_qc("missing_field", "p", {}, [])


@pytest.fixture
def _timing(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    exact_timing.install(monkeypatch)

    async def off(self: Any, state: Any) -> None:
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)


async def _too_long(tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    """A study about one of whose 60 settings fits the time allowed: FI's own timing stops it as too long."""
    from tests.test_too_big_to_fit import _sizing

    eng, _model, runner = await _sizing(tmp_path, timeout_s=8, grid={"n": [1, 2, 4], "k": list(range(20))})
    return eng, runner


@pytest.mark.asyncio
async def test_too_long_under_a_commission_is_handed_back_without_the_plan_rewritten(tmp_path: Path, _timing: None) -> None:
    eng, runner = await _too_long(tmp_path)
    eng.config.commission = _commission(tmp_path / "order")
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and "handed back" in patch["exec_result"]["too_long"]
    state = {"exec_result": patch["exec_result"], "result_json": {}, "iteration": 0,
             "exec_give_up_reason": patch["exec_result"]["too_long"]}
    verdict = eng._no_results_verdict(state)
    assert verdict is not None and not verdict.get("redesign") and not verdict.get("fit_request")
    assert verdict["stuck"] is True and verdict["stuck_reason"] == "too_long"
    await eng._node_stuck_no_findings({**state, "evidence_assessment": verdict})
    record = json.loads((eng.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert list(record)[0] == "return_reason" and record["return_reason"] == "scope_exceeds_cap"
    assert record["kind"] == "returned" and record["evidence"]["settings"] == 60
    assert record["evidence"]["seconds_per_setting"] > 0, "FI's measured time of one setting"
    assert "cannot be done in the time allowed" in record["problem"] and "handed back" in record["problem"]


@pytest.mark.asyncio
async def test_too_long_without_a_commission_still_asks_the_plan_once_for_a_study_that_fits(tmp_path: Path,
                                                                                          _timing: None) -> None:
    eng, runner = await _too_long(tmp_path)
    patch = await eng._size_the_run({"iteration": 0}, runner, sys.executable, None)
    assert patch is not None and "asks the plan once for a study that fits" in patch["exec_result"]["too_long"]
    verdict = eng._no_results_verdict({"exec_result": patch["exec_result"], "result_json": {}, "iteration": 0,
                                       "exec_give_up_reason": patch["exec_result"]["too_long"]})
    assert verdict is not None and verdict["redesign"] is True and verdict.get("fit_request")


@pytest.mark.asyncio
async def test_the_summary_carries_the_commission_s_hash(tmp_path: Path) -> None:
    import launch
    from core import commission
    from core.engine import QuestArtifacts

    path = _commission(tmp_path / "order")
    eng = _engine(tmp_path, path)
    commission.stage(path, eng.quest_root)
    eng._return_to_qc("scope_exceeds_cap", "too long", {}, [])
    summary = await launch._finish_outputs(eng.config, QuestArtifacts(quest_id=eng.quest_id, quest_root=eng.quest_root),
                                           supervisor=None)  # type: ignore[arg-type]
    assert summary["commission_sha256"] == commission.sha256(path)


@pytest.mark.parametrize("value,passage", [(0.888, "measured as 0.8880 at rest"), (0.888, "a ratio of .888"),
                                           (1200, "about 1,200 cycles"), (0.001, "an error of 1e-3"),
                                           (0.001, "an error of 1 x 10^-3"), (-0.3, "a change of −0.3 per step"),
                                           (25000.0, "a load of 2.5E4 newtons")])
def test_a_value_is_found_in_its_passage_however_the_number_is_written(value: float, passage: str) -> None:
    from core import commission

    assert commission.number_in_text(value, passage)
    assert not commission.number_in_text(value * 1.01, passage)
    assert not commission.number_in_text(-value, passage), "the sign is part of the number"


def test_the_web_page_refuses_it_before_a_quest_folder_is_made(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from web.server import make_app

    path = _commission(tmp_path / "order", known_values=_known(passage="The drain constant is 0.25 per day, said nowhere"))
    out = tmp_path / "outputs"
    out.mkdir()
    r = TestClient(make_app(out)).post("/api/quests/start", json={"yaml": f"commission: {path.as_posix()}\n"})
    assert r.status_code == 400 and "not in its source file" in r.json()["detail"]
    assert [p for p in out.iterdir() if p.is_dir() and (p / ".fi").exists()] == [], "no quest folder"


async def _handed_back(eng: Any, patch: dict[str, Any] | None) -> dict[str, Any]:
    """The stop FI's sizing returned, carried through the gate and the stop step: the record in needs/STUCK.json."""
    assert patch is not None and patch["exec_result"].get("too_long"), "the study stops instead of being made smaller"
    state = {"exec_result": patch["exec_result"], "result_json": {}, "iteration": 0,
             "exec_give_up_reason": patch["exec_result"]["too_long"]}
    verdict = eng._no_results_verdict(state)
    assert verdict is not None and verdict["stuck"] is True and not verdict.get("fit_request")
    await eng._node_stuck_no_findings({**state, "evidence_assessment": verdict})
    return json.loads((eng.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_commissioned_study_one_in_ten_of_whose_settings_fit_is_handed_back_not_cut(tmp_path: Path,
                                                                                          _timing: None) -> None:
    from core import run_estimate
    from tests.test_too_big_to_fit import _sizing

    eng, _model, runner = await _sizing(tmp_path, timeout_s=8, grid={"n": [1, 2, 4], "k": [0, 1]})  # 6 settings, 1 fits
    eng.config.commission = _commission(tmp_path / "order")
    record = await _handed_back(eng, await eng._size_the_run({"iteration": 0}, runner, sys.executable, None))
    assert "truncated" not in run_estimate.read_record(eng.quest_root), "nothing was run cut at the time limit"
    assert record["return_reason"] == "scope_exceeds_cap" and record["evidence"]["settings"] == 6


@pytest.mark.asyncio
async def test_a_commissioned_study_whose_runs_fi_would_lower_is_handed_back_with_its_runs_kept(tmp_path: Path,
                                                                                              _timing: None) -> None:
    from core import plan, run_estimate
    from tests.test_run_checks import _sizing

    eng, _model, runner, _ex = await _sizing(tmp_path, timeout_s=8, min_runs_per_setting=8)
    eng.config.commission = _commission(tmp_path / "order")
    record = await _handed_back(eng, await eng._size_the_run({"iteration": 0}, runner, sys.executable, None))
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == 40, "the plan's runs are kept"
    assert not run_estimate.read_record(eng.quest_root).get("reduced")
    assert record["return_reason"] == "scope_exceeds_cap" and record["kind"] == "returned"
