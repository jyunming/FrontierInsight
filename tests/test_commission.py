"""A study FI starts from a commission file instead of a topic, and hands back when its run cannot fit.

The rules a commission must meet are refused at the door, before any model call (core/commission.py ``problems``); a
commission is frozen into the quest with its full texts and its hash; a resume with another commission stops before
any step; the literature step reads only the commission's sources; the plan's model is shown what the commission fixes, and
plan.md says which known value the plan has no check for; a run FI times as too long is handed back
(``needs/STUCK.json`` ``return_reason`` first) instead of the plan being rewritten smaller.

The topic of these tests is a made-up draining tank.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core import commission
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
    ("passage too short to find", {"known_values": _known(passage="0.25 per hour")},
     "give the sentence of the source that states it, at least 25 characters"),
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


def test_a_number_json_dumps_writes_with_an_exponent_is_a_number(tmp_path: Path) -> None:
    from core import commission

    path = _commission(tmp_path, known_values=_known(tolerance=0.00005))
    assert "5e-05" in path.read_text(encoding="utf-8"), "how another program's json.dumps writes it"
    assert commission.problems(path) == []
    as_yaml = tmp_path / "commission.yaml"
    as_yaml.write_text("tolerance: 5e-05\nvalue: -2E3\nname: 1e5 cycles\n", encoding="utf-8")
    assert commission.read(as_yaml) == {"tolerance": 5e-05, "value": -2000.0, "name": "1e5 cycles"}


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

def test_the_question_becomes_the_topic_and_the_steps_it_gives_stay_off_whatever_the_config_says(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _commission(tmp_path / "order")
    yaml = tmp_path / "order" / "quest.yaml"
    yaml.write_text("commission: commission.json\npauses:\n  clarify: ask\n  scope: ask\nengine:\n  clarify_mode: auto\n"
                    "knowledge:\n  web_search: false\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("INFO", logger="frontier_insight.config"):
        cfg = Config.from_yaml(yaml)
    assert cfg.commission == (tmp_path / "order" / "commission.json").resolve()
    assert cfg.topic == "How fast does a tank drain through outlets of three sizes?"
    # The setup questions are gone: the keys load, and the log says once for each that it does nothing.
    gone = sorted(r.getMessage() for r in caplog.records if "no longer does anything" in r.getMessage())
    assert gone == ["config: engine.clarify_mode no longer does anything; the line can be removed.",
                    "config: pauses.clarify no longer does anything; the line can be removed.",
                    "config: pauses.scope no longer does anything; the line can be removed."], gone
    assert not [r for r in caplog.records if "are not used" in r.getMessage()], "nothing the commission overrides"
    assert cfg.pauses.papers is False
    assert cfg.engine.enable_analyze_reroute is False
    assert cfg.execution.timeout_s == 2 * 3600, "the commission's hours are the run's time limit when the config is silent"
    plain = Config(topic="a draining tank")
    assert plain.commission is None and plain.engine.enable_analyze_reroute is True


def test_the_commission_s_hours_cap_the_run_s_time_limit_the_config_sets(tmp_path: Path) -> None:
    _commission(tmp_path / "order")
    yaml = tmp_path / "order" / "quest.yaml"
    for given, limit in ((99999, 2 * 3600), (600, 600)):  # the smaller one is kept
        yaml.write_text(f"commission: commission.json\nexecution:\n  timeout_s: {given}\n", encoding="utf-8")
        assert Config.from_yaml(yaml).execution.timeout_s == limit
    in_code = Config(commission=tmp_path / "order" / "commission.json", execution=ExecutionConfig(sandbox="venv"))
    assert in_code.execution.timeout_s == 2 * 3600, "a limit left unset in code is the commission's"


@pytest.mark.parametrize("written", ['"99999"', "99999.0"])
def test_a_time_limit_written_in_another_form_the_config_accepts_is_capped_too(tmp_path: Path, written: str) -> None:
    _commission(tmp_path / "order")
    yaml = tmp_path / "order" / "quest.yaml"
    yaml.write_text(f"commission: commission.json\nexecution:\n  timeout_s: {written}\n", encoding="utf-8")
    assert Config.from_yaml(yaml).execution.timeout_s == 2 * 3600


@pytest.mark.asyncio
async def test_a_watch_checks_the_commission_again_after_its_wait_before_it_resumes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    import launch
    from core import commission, job_watch

    path = _commission(tmp_path / "order")
    cfg = Config(commission=path, provider=ProviderConfig(name="openai"), knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "outputs"))
    root = tmp_path / "outputs" / "q9"
    commission.stage(path, root)
    job_watch.write_pending(root / ".fi", {"status": "pending", "id": "J1"}, "experiment.py")
    data = json.loads(path.read_text(encoding="utf-8"))

    async def wait(*_a: Any, **_k: Any) -> str:  # the commission is changed while the job runs
        path.write_text(json.dumps({**data, "scope": "five outlet diameters"}), encoding="utf-8")
        return "done"

    monkeypatch.setattr(job_watch, "watch", wait)
    monkeypatch.setattr(launch, "run_one", AsyncMock(side_effect=AssertionError("the quest is not resumed")))
    args = argparse.Namespace(watch="q9", watch_every=1, watch_max_hours=0.0, config=tmp_path / "order" / "quest.yaml",
                              auto_accept_on_pass=None)
    assert await launch._watch_quest(cfg, args, supervisor=None) == 2
    assert "not the one quest q9 was started from" in capsys.readouterr().err


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
    with pytest.raises(commission.CommissionRefused, match="q2 was started without a commission file"):
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
    (root / ".fi").mkdir(parents=True, exist_ok=True)
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
async def test_the_skill_picker_and_the_design_are_given_the_commission_s_question_and_scope(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FI makes no research ideas, so the slot that carried the chosen idea was an empty ``{}``: the skill picker never
    saw what the study is limited to. It is given the commission's question and scope, and so are the design's
    prompts (the same slot)."""
    from core.skills import approval
    from core.skills.base import EXTERNAL_SOURCE, Skill

    monkeypatch.delenv("FI_EXTERNAL_SKILLS_DIRS", raising=False)
    monkeypatch.setenv("FI_SKILLS_DIR", str(tmp_path / "own_skills"))
    monkeypatch.setenv("FI_SKILLS_APPROVALS", str(tmp_path / "approvals.json"))
    folder = tmp_path / "agent" / "skills" / "flow-units"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: flow-units\ndescription: units for flow rates\n---\nUse SI.\n",
                                     encoding="utf-8")
    skill = Skill(name="flow-units", path=folder, source=EXTERNAL_SOURCE)
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="me", note="t")
    eng = _engine(tmp_path, _commission(tmp_path / "order"), skills_dirs=[str(folder.parent)],
                  skills_scan_known_dirs=False)
    asked: list[str] = []

    async def chat(prompt: str, node: str | None = None, **_kw: Any) -> str:
        if node == "select_skills":
            asked.append(prompt)
        return json.dumps({"skills": [], "declined": []})

    eng._chat = chat  # type: ignore[method-assign]
    await eng._node_select_skills({"topic": eng.config.topic})
    (prompt,) = asked
    given = ("Question: How fast does a tank drain through outlets of three sizes?\n"
             "Scope: three outlet diameters, one fill level")
    assert f"## The study\n\n{given}\n" in prompt and "flow-units" in prompt
    old = {"chosen_idea": {"title": "an old checkpoint's idea"}}
    assert eng._study_given(old) == given, "the commission fixes the study, whatever a checkpoint holds"
    # A quest with no commission: the direction an earlier FI's checkpoint chose, else nothing.
    eng._commissioned = lambda: False  # type: ignore[method-assign]
    assert eng._study_given({}) == "(none)"
    assert '"title": "an old checkpoint\'s idea"' in eng._study_given(old)


@pytest.mark.asyncio
async def test_the_literature_step_reads_only_the_commission_s_sources_and_says_so(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import commission

    monkeypatch.setattr("core.engine.Knowledge.asearch", AsyncMock(side_effect=AssertionError("nothing is searched")))
    path = _commission(tmp_path / "order")
    eng = _engine(tmp_path, path)
    commission.stage(path, eng.quest_root)
    said: list[str] = []
    eng._progress = said.append  # type: ignore[method-assign]
    eng._progress_stage("literature")
    assert said == ["[literature] Reading the sources the commission gives."]
    out = await eng._node_literature({"topic": eng.config.topic})
    [entry] = out["literature"]
    assert entry["metadata"]["source"] == "user_supplied" and PASSAGE in entry["content"]
    assert entry["metadata"]["title"] == "A made-up tank note" and entry["metadata"]["doi"] == "10.0000/tank.1"
    assert eng._client.chat.call_count == 0


@pytest.mark.asyncio
async def test_a_rerun_from_a_step_the_commission_gives_says_so_and_changes_nothing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    eng = _engine(tmp_path, _commission(tmp_path / "order"))

    async def no_env(*_a: Any, **_k: Any) -> None:
        return None

    eng.executor.setup = no_env
    eng._record_environment = no_env  # type: ignore[method-assign]
    eng._connect_llm = no_env  # type: ignore[method-assign]
    await eng.run(from_step="literature")
    assert "the literature step is not one FI did" in capsys.readouterr().out
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
        if messages[-1]["content"].startswith("# Compute a Known Value"):
            return "(no program)"  # the known value is left unconfirmed (tests/test_known_values.py)
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


# ---- every door that continues a quest; what is frozen stays frozen ------------------------------------------------------

@pytest.mark.asyncio
async def test_a_resume_stops_when_the_quest_s_own_copy_of_the_commission_or_a_source_changed(tmp_path: Path) -> None:
    from core import commission

    path = _commission(tmp_path / "order")
    eng = _engine(tmp_path, path)

    async def stop(_root: Any) -> None:
        raise _Stop()

    eng.executor.setup = stop
    with pytest.raises(_Stop):
        await eng.run()
    root = eng.quest_root
    commission.door(path, root)  # nothing changed: goes on
    for rel in ("inputs/commission.json", "inputs/papers/tank_note.md"):  # the file handed over is unchanged
        before = (root / rel).read_bytes()
        (root / rel).write_bytes(before.replace(b"0.25", b"0.35"))
        with pytest.raises(commission.CommissionRefused, match=f"{rel} changed or missing"):
            commission.door(path, root)
        (root / rel).write_bytes(before)
    (root / "inputs" / "papers" / "tank_note.md").unlink()
    with pytest.raises(commission.CommissionRefused, match="changed or missing"):
        commission.door(path, root)


@pytest.mark.asyncio
async def test_watching_a_quest_runs_the_same_door_before_it_continues_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import launch
    from core import commission

    monkeypatch.setattr(launch, "_watch_quest", AsyncMock(side_effect=AssertionError("the quest is not continued")))
    path = _commission(tmp_path / "order")
    root = tmp_path / "outputs" / "q5"
    commission.stage(path, root)
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    (root / ".fi" / "state.sqlite").write_bytes(b"")
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "scope": "five outlet diameters"}), encoding="utf-8")
    yaml = tmp_path / "order" / "quest.yaml"
    yaml.write_text(f"commission: commission.json\nprovider:\n  name: ollama\noutput:\n  output_dir: "
                    f"{(tmp_path / 'outputs').as_posix()}\n", encoding="utf-8")
    rc = await launch.main_async(launch.parse_args(["--config", str(yaml), "--watch", "q5", "--no-axon-sidecar"]))
    assert rc == 2 and "not the one quest q5 was started from" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["--config", "--fleet"])
async def test_a_refused_commission_starts_no_service_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str], flag: str) -> None:
    import launch
    from core import axon_sidecar

    started: list[int] = []
    monkeypatch.setattr(axon_sidecar, "ensure_axon_up", lambda *a, **k: started.append(1))
    monkeypatch.setattr("core.engine.LLMClient.chat", AsyncMock(side_effect=AssertionError("no model call is made")))
    _commission(tmp_path / "order", doubts=None)
    yaml = tmp_path / "order" / "quest.yaml"
    yaml.write_text(f"commission: commission.json\nprovider:\n  name: ollama\noutput:\n  output_dir: "
                    f"{(tmp_path / 'outputs').as_posix()}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert await launch.main_async(launch.parse_args([flag, str(yaml)])) == 2
    assert started == [], "the knowledge service was not started for a study that is refused"
    assert "`doubts`" in capsys.readouterr().err


def test_a_number_that_is_one_side_of_a_fraction_is_not_the_value() -> None:
    from core import commission

    assert not commission.number_in_text(1, "a ratio of 1/2 at rest")
    assert not commission.number_in_text(2, "a ratio of 1 / 2 at rest")
    assert commission.number_in_text(1, "a ratio of 1 to 2 at rest")
    assert commission.number_in_text(0.25, "a drain constant of 0.25/h at rest"), "a unit written with / is no fraction"
    assert commission.number_in_text(10, "a flow of 10 / min at rest")
    for bracketed in ("a ratio of 1/(2) at rest", "a ratio of (1)/2 at rest", "a ratio of (1)/(2) at rest"):
        assert not commission.number_in_text(1, bracketed) and not commission.number_in_text(2, bracketed)
    for nested in ("a ratio of 1/((2)) at rest", "a ratio of ((1))/2 at rest", "a ratio of [(1)] / [(2)] at rest"):
        assert not commission.number_in_text(1, nested) and not commission.number_in_text(2, nested)
    assert commission.number_in_text(0.5, "a share of (0.5) at rest"), "a bracket alone makes no fraction"


def test_an_exponent_is_read_on_its_own_line_only_and_one_no_float_holds_is_no_number() -> None:
    from core import commission

    header = "Revista, vol. 45, e20230014, 2023\n\ne20230014-4\nThe period of a made-up swing"
    assert commission.number_in_text(2023, header), "the year of a page header is the year, not 2023e20230014"
    assert not commission.number_in_text(1.01741, header)
    assert commission.number_in_text(0.0015, "a drain of 1.5 e-3 per hour") and commission.number_in_text(0.002, "2 x 10^-3")
    assert not commission.number_in_text(2.0, "a drain of 2e400 per hour"), "an exponent no float holds is no number"


def _staged_and_read(tmp_path: Path, files: tuple[str, ...]) -> tuple[dict[str, str], int]:
    """The commission's sources at ``files`` staged into a quest: their staged names, and how many the reader read."""
    import logging

    from core import commission
    from core.engine_base import _ingest_user_dropped_papers

    folder = tmp_path / "order"
    known = [{**_known()[0], "name": f"value {i}", "source": {"title": f"note {i}", "file": rel}}
             for i, rel in enumerate(files)]
    path = _commission(folder, known_values=known)
    for i, rel in enumerate(files):
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_text(f"Source {i}. " + SOURCE_TEXT, encoding="utf-8")
    root = tmp_path / "out" / "q6"
    commission.stage(path, root)
    return commission.staged_names(commission.read(path)), _ingest_user_dropped_papers(
        root, [], set(), logging.getLogger("test"))[1]


def test_two_sources_whose_names_differ_only_in_case_are_both_kept(tmp_path: Path) -> None:
    names, read = _staged_and_read(tmp_path, ("a/Tank.md", "b/tank.md"))
    assert len({n.lower() for n in names.values()}) == 2, names
    assert read == 2, "both sources are read"


def test_a_source_named_as_the_papers_reader_skips_is_staged_under_a_name_it_reads(tmp_path: Path) -> None:
    names, read = _staged_and_read(tmp_path, ("README.md", ".hidden.md"))
    assert not any(n.startswith(".") or n.lower() == "readme.md" for n in names.values()), names
    assert read == 2, "every source the commission gives is read"


@pytest.mark.asyncio
async def test_the_settings_a_commission_overrides_are_named_once_in_the_quest_s_run_log(tmp_path: Path) -> None:
    from core.engine import Engine

    path = _commission(tmp_path / "order")
    eng = Engine(Config(commission=path, provider=ProviderConfig(name="openai"), pauses=PausesConfig(papers=True),
                        engine=EngineConfig(max_iterations=1, review_loop=False),
                        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
                        output=OutputConfig(output_dir=tmp_path / "out")))
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=AssertionError("no model call is made")),
                                    "aclose": AsyncMock()})()

    async def stop(_root: Any) -> None:
        raise _Stop()

    eng.executor.setup = stop
    with pytest.raises(_Stop):
        await eng.run()
    log = (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert log.count("settings of the config are not used: pauses.papers.") == 1, log[-2000:]


@pytest.mark.asyncio
async def test_a_resume_by_a_shortened_id_with_a_changed_commission_stops(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import launch
    from core import commission

    monkeypatch.setattr("core.engine.LLMClient.chat", AsyncMock(side_effect=AssertionError("no model call is made")))
    path = _commission(tmp_path / "order")
    root = tmp_path / "outputs" / "tank-quest-q8"
    commission.stage(path, root)
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    (root / ".fi" / "state.sqlite").write_bytes(b"")
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "scope": "five outlet diameters"}), encoding="utf-8")
    yaml = tmp_path / "order" / "quest.yaml"
    yaml.write_text(f"commission: commission.json\nprovider:\n  name: ollama\noutput:\n  output_dir: "
                    f"{(tmp_path / 'outputs').as_posix()}\n", encoding="utf-8")
    rc = await launch.main_async(launch.parse_args(["--config", str(yaml), "--resume", "tank-quest", "--no-axon-sidecar"]))
    assert rc == 2 and "not the one quest tank-quest-q8 was started from" in capsys.readouterr().err


def test_a_quest_stopped_after_staging_and_before_the_trace_event_still_checks_its_files(tmp_path: Path) -> None:
    from core import commission

    path = _commission(tmp_path / "order")
    root = tmp_path / "out" / "q9"
    files = commission.stage(path, root)  # the process stops here: no commission_recorded event was written
    commission.door(path, root)  # nothing changed: goes on
    assert commission.staged_files(root) == files
    text = next(root / rel for rel in files if rel.startswith("inputs/papers/"))
    before = text.read_bytes()
    text.write_bytes(before.replace(b"0.25", b"0.35"))
    with pytest.raises(commission.CommissionRefused, match="changed or missing"):
        commission.door(path, root)


def test_a_staging_that_stops_in_the_middle_of_the_commission_copy_leaves_no_staged_marker(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import commission

    path = _commission(tmp_path / "order")
    root = tmp_path / "out" / "q10"
    real = commission.shutil.copy2

    def stops(src: Any, dst: Any, **kw: Any) -> Any:
        if Path(src) == path:
            Path(dst).write_bytes(Path(src).read_bytes()[:10])
            raise OSError("stopped")
        return real(src, dst, **kw)

    monkeypatch.setattr(commission.shutil, "copy2", stops)
    with pytest.raises(OSError):
        commission.stage(path, root)
    assert not commission.staged_path(root).exists()
    monkeypatch.setattr(commission.shutil, "copy2", real)
    commission.stage(path, root)
    commission.door(path, root)  # the original file is still the one the quest was started from


def test_a_staged_quest_with_no_trace_event_and_no_manifest_is_refused_but_one_with_the_event_goes_on(tmp_path: Path) -> None:
    from core import commission

    path = _commission(tmp_path / "order")
    root = tmp_path / "out" / "q11"
    files = commission.stage(path, root)
    manifest = root / ".fi" / "commission_files.json"
    for broken in (None, "{not json", "[1]"):
        if broken is None:
            manifest.unlink(missing_ok=True)
        else:
            manifest.write_text(broken, encoding="utf-8")
        with pytest.raises(commission.CommissionRefused, match="record of its files is missing"):
            commission.door(path, root)
    (root / ".fi" / "audit.jsonl").write_text(
        json.dumps({"kind": "commission_recorded", "sha256": files["inputs/commission.json"], "files": files}) + "\n",
        encoding="utf-8")
    commission.door(path, root)  # an older quest: its trace event checks the files, as before


# ---- a study without a commission ------------------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["--config", "--fleet"])
async def test_a_topic_only_config_is_refused_before_any_folder_service_or_model_call(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flag: str) -> None:
    import launch
    from core import axon_sidecar

    started: list[int] = []
    monkeypatch.setattr(axon_sidecar, "ensure_axon_up", lambda *a, **k: started.append(1))
    chat = AsyncMock(side_effect=AssertionError("no model call is made"))
    monkeypatch.setattr("core.engine.LLMClient.chat", chat)
    yaml = tmp_path / "quest.yaml"
    yaml.write_text(f"topic: How fast does a tank drain?\nprovider:\n  name: ollama\noutput:\n  output_dir: "
                    f"{(tmp_path / 'outputs').as_posix()}\n", encoding="utf-8")
    assert await launch.main_async(launch.parse_args([flag, str(yaml)])) == 2
    err = capsys.readouterr().err
    assert "no study was started: every study starts from a commission file" in err and "`fi new`" in err
    assert err.count("\n") == 1 and "Traceback" not in err, "one sentence"
    assert not (tmp_path / "outputs").exists() and started == [] and chat.await_count == 0


def test_the_web_start_refuses_a_topic_only_config_before_a_quest_folder_is_made(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from web.server import make_app

    out = tmp_path / "outputs"
    out.mkdir()
    r = TestClient(make_app(out)).post("/api/quests/start", json={"yaml": "topic: How fast does a tank drain?\n"})
    assert r.status_code == 400 and "every study starts from a commission file" in r.json()["detail"]
    assert list(out.iterdir()) == [], "no quest folder"


def test_an_old_quest_without_a_commission_is_not_resumed(tmp_path: Path) -> None:
    from core import commission

    root = tmp_path / "out" / "q-old"
    (root / ".fi").mkdir(parents=True)
    (root / "config.yaml").write_text("topic: How fast does a tank drain?\n", encoding="utf-8")
    with pytest.raises(commission.CommissionRefused, match="q-old was started without a commission file") as e:
        commission.door(None, root)
    assert "`fi new`" in str(e.value)


@pytest.mark.parametrize("flag", ["--resume", "--rerun", "--watch"])
def test_the_early_check_leaves_a_resumed_quest_to_the_check_that_knows_its_folder(tmp_path: Path, flag: str) -> None:
    """A quest that goes on is judged with its folder (it may have been started from a commission, or not); the check
    before any service starts judges only a new study's config, so it never says "no study was started" for one."""
    import launch

    cfg = tmp_path / "q-old" / "config.yaml"
    cfg.parent.mkdir()
    cfg.write_text("topic: How fast does a tank drain?\n", encoding="utf-8")
    args = launch.parse_args(["--config", str(cfg), flag, "q-old"])
    assert launch._commission_refusal(args) is None
    assert "no study was started" in launch._commission_refusal(launch.parse_args(["--config", str(cfg)]))


@pytest.mark.asyncio
async def test_update_refuses_an_old_quest_without_a_commission_and_approves_nothing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    root = tmp_path / "out" / "q-old"
    (root / ".fi").mkdir(parents=True)
    (root / "config.yaml").write_text("topic: How fast does a tank drain?\nprovider:\n  name: ollama\n", encoding="utf-8")
    rc = await launch._run_update(quest_root=root, vscode_bridge_port=0, interactive=False, supervisor=None)
    assert rc == 2 and "started without a commission file" in capsys.readouterr().err
    assert not (root / ".fi" / "approved_plan.json").exists()


@pytest.mark.parametrize("planted,named", [
    ({"scope": None}, "The commission has no `scope` (a sentence)."),
    ({"known_values": _known(passage="The drain constant of the reference tank is 0.25 per minute")},
     "is not in its source file"),
    ({"known_values": _known(value=0.3)}, "does not state its value 0.3"),
], ids=["missing key", "passage not in its source", "passage without its value"])
def test_fi_check_says_each_problem_in_one_sentence_and_exits_2(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], planted: dict[str, Any], named: str) -> None:
    import launch

    assert launch._check([str(_commission(tmp_path, **planted))]) == 2
    lines = capsys.readouterr().out.splitlines()
    assert any(named in line for line in lines), lines
    assert all(line.endswith(".") for line in lines)


def test_fi_check_passes_a_good_commission_and_one_its_config_names(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    path = _commission(tmp_path)
    assert launch._check([str(path)]) == 0
    assert "can be used: every key is there and each passage is in its source." in capsys.readouterr().out
    config = tmp_path / "study.yaml"
    config.write_text("commission: commission.json\nprovider:\n  name: ollama\n", encoding="utf-8")
    assert launch._check([str(config)]) == 0
    assert launch._check([]) == 2


def test_fi_new_writes_a_template_the_check_refuses_until_it_is_filled_in(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import yaml as _yaml

    import launch
    from core import commission, demo

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(demo, "set_up_providers", lambda: ["ollama"])
    assert launch._new([]) == 0
    assert "check commission.yaml" in capsys.readouterr().out
    template, config = tmp_path / "commission.yaml", tmp_path / "study.yaml"
    assert not config.read_text(encoding="utf-8").startswith("# Auto-generated by"), "never taken for an old interview's"
    usable, lines = commission.check(template)
    assert not usable and any("no `question`" in line for line in lines) and any("`failure`" in line for line in lines)
    assert launch._new([]) == 0 and (tmp_path / "commission-2.yaml").is_file() and (tmp_path / "study-2.yaml").is_file()
    data = _yaml.safe_load(template.read_text(encoding="utf-8"))
    assert set(data) == {"question", "method", "scope", "limits", "known_values", "no_known_values", "success", "failure",
                         "doubts"}
    data.update(question="How fast does a tank drain?", scope="three outlets", no_known_values=True,
                success=[{"condition": "c", "verify": "v"}], failure=[{"condition": "c", "verify": "v"}])
    template.write_text(_yaml.safe_dump(data), encoding="utf-8")
    assert commission.check(template)[0]
    cfg = Config.from_yaml(config)
    assert cfg.commission == template.resolve() and cfg.provider.name == "ollama"
    commission.door(cfg.commission, None)


@pytest.mark.asyncio
async def test_analyze_takes_a_commission_and_refuses_a_bad_one_before_a_quest_folder_is_made(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    monkeypatch.setattr("core.engine.LLMClient.chat", AsyncMock(side_effect=AssertionError("no model call is made")))
    data = tmp_path / "data"
    data.mkdir()
    (data / "runs.csv").write_text("outlet,seconds\n1,30\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        launch.parse_args(["--analyze", str(data)])  # a commission is required
    bad = _commission(tmp_path / "order", doubts=None)
    args = launch.parse_args(["--analyze", str(data), "--commission", str(bad), "--output-root", str(tmp_path / "out"),
                              "--no-axon-sidecar"])
    assert await launch.main_async(args) == 2
    assert "`doubts`" in capsys.readouterr().err and not (tmp_path / "out").exists()


# --- the review's five points: a started quest, the settings baseline, the topic, finite limits ----------------------


def test_a_quest_started_without_a_commission_cannot_take_one_before_its_plan(tmp_path: Path) -> None:
    """A quest stopped before plan.md (in its setup questions, say) began without a commission: its own checkpoint says
    it started, so a commission given on resume is refused, not taken as a new study's."""
    from core import commission

    root = tmp_path / "out" / "q-old"
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "state.sqlite").write_bytes(b"")
    with pytest.raises(commission.CommissionRefused, match="q-old was started without a commission file"):
        commission.door(_commission(tmp_path / "order"), root)
    commission.door(_commission(tmp_path / "order"), tmp_path / "out" / "q-new")  # a folder with no record: a new quest


def test_a_commissioned_quest_records_its_settings_at_the_first_start_and_a_looser_one_asks_for_update(
        tmp_path: Path) -> None:
    from core import plan_settings
    from core.config import Config

    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    first = Config(commission=_commission(tmp_path / "order"), rigor_profile="research")
    assert plan_settings.check(root, root / ".fi", first) == []
    assert (root / ".fi" / plan_settings.NAME).is_file(), "the commissioned quest's settings are its baseline"
    looser = Config(commission=first.commission)
    changed = plan_settings.check(root, root / ".fi", looser)
    assert changed and any("how strictly the quest is checked" in line for line in changed)


@pytest.mark.asyncio
async def test_a_config_topic_that_is_not_the_commissions_question_is_refused(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import launch
    from core import commission

    path = _commission(tmp_path / "order")
    question = commission.read(path)["question"]
    yaml = tmp_path / "quest.yaml"
    yaml.write_text(f"topic: Something else entirely\ncommission: {path.as_posix()}\nprovider:\n  name: ollama\n"
                    f"output:\n  output_dir: {(tmp_path / 'outputs').as_posix()}\n", encoding="utf-8")
    assert await launch.main_async(launch.parse_args(["--config", str(yaml)])) == 2
    err = capsys.readouterr().err
    assert "`topic:` is not the commission's question" in err and err.count("\n") == 1
    assert not (tmp_path / "outputs").exists()
    assert launch._check([str(yaml)]) == 2
    commission.door(path, None, topic=f"  {question}  ")  # the same words are fine
    commission.door(path, None, topic=None)


@pytest.mark.parametrize("hours", [".inf", ".nan", "1.0e+308"])
def test_hours_that_are_not_a_finite_number_fail_the_check_and_the_start_alike(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], hours: str) -> None:
    import launch
    from core import commission

    path = tmp_path / "order" / "commission.yaml"
    path.parent.mkdir()
    path.write_text(f"question: How fast does a tank drain?\nscope: one tank\nlimits: {{hours: {hours}, model_calls: 9}}\n"
                    "known_values: []\nno_known_values: true\nsuccess: [{condition: c, verify: v}]\n"
                    "failure: [{condition: c, verify: v}]\ndoubts: []\n", encoding="utf-8")
    assert launch._check([str(path)]) == 2
    assert "`limits.hours` is missing, is not a number above 0, or is too large to count" in capsys.readouterr().out
    with pytest.raises(commission.CommissionRefused, match="limits.hours"):
        commission.door(path, None)


@pytest.mark.parametrize("hours", ["1.0e+308"])
def test_hours_too_large_to_count_in_seconds_are_refused_when_the_config_is_read(tmp_path: Path, hours: str) -> None:
    """The start counts the time limit in seconds: an hour count whose seconds overflow is refused in one sentence when
    the config is read, never an OverflowError."""
    from core import commission
    from core.config import Config

    order = tmp_path / "order"
    order.mkdir()
    (order / "commission.yaml").write_text(
        f"question: How fast does a tank drain?\nscope: one tank\nlimits: {{hours: {hours}, model_calls: 9}}\n"
        "known_values: []\nno_known_values: true\nsuccess: [{condition: c, verify: v}]\n"
        "failure: [{condition: c, verify: v}]\ndoubts: []\n", encoding="utf-8")
    config = tmp_path / "study.yaml"
    config.write_text("commission: order/commission.yaml\n", encoding="utf-8")
    with pytest.raises(commission.CommissionRefused, match="too large to count"):
        Config.from_yaml(config)
    built = Config(commission=order / "commission.yaml")  # the web start builds it from the text: no crash either
    assert built.execution.timeout_s < 10 ** 12


@pytest.mark.asyncio
async def test_a_first_start_that_stops_at_the_reviewer_check_is_resumed_as_a_commissioned_quest(
        tmp_path: Path) -> None:
    """Research with every reviewer on one model stops before anything runs; the quest it leaves is still the
    commission's (staged before the run's first record), so the resume is not refused as an old one."""
    from core import commission
    from core.config import Config
    from core.engine import Engine

    path = _commission(tmp_path / "order")
    cfg = Config.model_validate({
        "commission": str(path), "rigor_profile": "research", "provider": {"name": "openai", "model": "m-main"},
        "knowledge": {"enabled": False}, "output": {"output_dir": str(tmp_path / "out")},
    })
    engine = Engine(cfg)
    await engine.run()
    assert json.loads((engine.fi_dir / "pause.json").read_text(encoding="utf-8"))["kind"] == "review_models"
    commission.door(path, engine.quest_root, topic=cfg.topic)  # accepted: no CommissionRefused



# ---- method and title: how the study gets its evidence, and what it is called ----------------------------------------


def test_fi_check_refuses_a_method_that_is_not_one_of_the_three_in_one_sentence(tmp_path: Path) -> None:
    path = _commission(tmp_path / "order", method="experiment")
    usable, lines = commission.check(path)
    assert not usable and len(lines) == 1
    assert "`method` is 'experiment'" in lines[0] and "`simulate`, `analyze_data`, `review_literature`" in lines[0]
    for good in ("simulate", "analyze_data", "review_literature"):
        assert commission.check(_commission(tmp_path / good, method=good))[0]
    assert commission.check(_commission(tmp_path / "none"))[0], "no method: every existing commission stays valid"
    assert not commission.check(_commission(tmp_path / "t", title="  "))[0], "an empty title is no title"


def test_the_template_fi_new_writes_names_method_and_title() -> None:
    assert "method: simulate" in commission.TEMPLATE and "# title:" in commission.TEMPLATE


@pytest.mark.parametrize("method,modes", [("simulate", (False, False)), ("analyze_data", (True, False)),
                                          ("review_literature", (True, True))])
def test_the_method_sets_the_two_modes_of_the_config(tmp_path: Path, method: str, modes: tuple[bool, bool]) -> None:
    cfg = Config(commission=_commission(tmp_path / "order", method=method))
    assert (cfg.engine.no_simulation, cfg.engine.survey_mode) == modes
    assert commission.modes(commission.read(cfg.commission)) == modes
    assert commission.modes(commission.read(_commission(tmp_path / "none"))) is None


def _yaml_with(tmp_path: Path, commission_changes: dict[str, Any], engine: str) -> Path:
    _commission(tmp_path / "order", **commission_changes)
    cfg = tmp_path / "order" / "study.yaml"
    cfg.write_text("commission: commission.json\nengine:\n  " + engine + "\n", encoding="utf-8")
    return cfg


@pytest.mark.parametrize("key,method", [("no_simulation", "analyze_data"), ("survey_mode", "review_literature")])
def test_the_door_refuses_a_mode_in_the_config_the_commission_does_not_give_and_accepts_it_when_it_does(
        tmp_path: Path, key: str, method: str) -> None:
    refused = Config.from_yaml(_yaml_with(tmp_path / "a", {}, f"{key}: true"))
    with pytest.raises(commission.CommissionRefused) as e:
        commission.door(refused.commission, None, topic=refused.topic, asked=refused.modes_asked())
    assert f"`engine.{key}: true`" in str(e.value) and f"`method: {method}`" in str(e.value) and "\n" not in str(e.value)
    wrong = Config.from_yaml(_yaml_with(tmp_path / "b", {"method": "simulate"}, f"{key}: true"))
    with pytest.raises(commission.CommissionRefused) as e2:
        commission.door(wrong.commission, None, topic=wrong.topic, asked=wrong.modes_asked())
    assert "`method: simulate`" in str(e2.value) and "the commission decides" in str(e2.value)
    agree = Config.from_yaml(_yaml_with(tmp_path / "c", {"method": method}, f"{key}: true"))
    commission.door(agree.commission, None, topic=agree.topic, asked=agree.modes_asked())  # goes on
    none = Config.from_yaml(_yaml_with(tmp_path / "d", {}, "max_iterations: 1"))
    commission.door(none.commission, None, topic=none.topic, asked=none.modes_asked())  # no mode asked: goes on
    # A quest that has begun carries its modes in its checkpoint: the check is for a new start only.
    root = tmp_path / "out" / "q"
    (root / ".fi").mkdir(parents=True)
    commission.stage(refused.commission, root)
    commission.door(refused.commission, root, topic=refused.topic, asked=refused.modes_asked())


def test_the_early_check_of_launch_says_the_same_before_anything_starts(tmp_path: Path) -> None:
    import launch

    bad = _yaml_with(tmp_path / "a", {}, "no_simulation: true")
    args = launch.parse_args(["--config", str(bad)])
    said = launch._commission_refusal(args)
    assert said and "`method: analyze_data`" in said
    ok = _yaml_with(tmp_path / "b", {"method": "analyze_data"}, "no_simulation: true")
    assert launch._commission_refusal(launch.parse_args(["--config", str(ok)])) is None


@pytest.mark.asyncio
async def test_the_first_node_makes_no_model_call_for_a_commissioned_study(tmp_path: Path) -> None:
    eng = _engine(tmp_path, _commission(tmp_path / "order", method="review_literature"))
    patch = await eng._node_clarify({"topic": eng.config.topic})  # the stub model raises if it is asked
    assert patch == {"clarify_done": True, "no_simulation_resolved": True, "survey_mode_resolved": True}
    eng._client.chat.assert_not_awaited()


def test_a_title_in_the_commission_is_the_studys_title(tmp_path: Path) -> None:
    cfg = Config(commission=_commission(tmp_path / "order", title="Draining a tank"))
    assert cfg.title == "Draining a tank"
    assert _quest_id_has(cfg, "draining-a-tank")


def _quest_id_has(cfg: Config, part: str) -> bool:
    from core.engine import Engine

    cfg = cfg.model_copy(update={"output": OutputConfig(output_dir=cfg.output.output_dir)})
    return part in Engine(cfg).quest_id


# ---- review round: the protocol, the person's own data, the title, an old checkpoint's venue -------------------------------


@pytest.mark.parametrize("method,topic_type,policy", [("review_literature", "survey", "mixed"),
                                                      ("analyze_data", "data_analysis", "user_data"),
                                                      ("simulate", "simulation", None)])
def test_the_protocol_takes_the_kind_of_study_from_the_commissions_method(
        tmp_path: Path, method: str, topic_type: str, policy: str | None) -> None:
    from core.protocol import derive_protocol

    eng = _engine(tmp_path, _commission(tmp_path / "order", method=method))
    out = asyncio.run(eng._node_clarify({"topic": eng.config.topic}))
    protocol = derive_protocol({"topic": eng.config.topic, **out}, eng.config)
    assert protocol.topic_type == topic_type
    if policy:
        assert protocol.source_policy == policy


def test_an_old_checkpoints_topic_shape_still_sets_the_protocol_when_no_mode_says_otherwise() -> None:
    from core.protocol import derive_protocol

    state = {"topic": "a plain question", "no_simulation_resolved": True, "survey_mode_resolved": False,
             "clarify_answers": {"topic_shape": "case_study"}}
    assert derive_protocol(state, None).topic_type == "case_study"


def test_analysing_the_persons_data_needs_the_commission_to_say_so(tmp_path: Path) -> None:
    import launch

    for changes, said in (({}, "no `method`"), ({"method": "simulate"}, "`method: simulate`")):
        order = _commission(tmp_path / ("a" + said[:4]), **changes)
        with pytest.raises(commission.CommissionRefused) as e:
            commission.door(order, None, asked={"analyze_local_first": True})
        assert "`method: analyze_data`" in str(e.value) and said in str(e.value) and "\n" not in str(e.value)
        # ``--analyze <data> --commission <file>`` is refused the same way, before a folder is made or a file copied.
        data = tmp_path / "mydata"
        data.mkdir(exist_ok=True)
        (data / "t.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        code = asyncio.run(launch._run_analyze(data_path=data, commission=order, output_root=tmp_path / "o",
                                               provider_name="openai", vscode_bridge_port=0, supervisor=None))  # type: ignore[arg-type]
        assert code == 2 and not (tmp_path / "o").exists()
    ok = _commission(tmp_path / "ok", method="analyze_data")
    commission.door(ok, None, asked={"analyze_local_first": True})
    # A config that sets ``engine.analyze_local_first: true`` is refused at the door the same way.
    cfg = tmp_path / "cfgs"
    _commission(cfg)
    (cfg / "study.yaml").write_text("commission: commission.json\nengine:\n  analyze_local_first: true\n", encoding="utf-8")
    loaded = Config.from_yaml(cfg / "study.yaml")
    with pytest.raises(commission.CommissionRefused):
        commission.door(loaded.commission, None, topic=loaded.topic, asked=loaded.modes_asked())


@pytest.mark.asyncio
async def test_a_commission_title_is_the_title_the_writer_is_told_to_use(tmp_path: Path) -> None:
    eng = _engine(tmp_path, _commission(tmp_path / "order", title="Draining a tank"))
    out = await eng._node_clarify({"topic": eng.config.topic})
    assert out["title"] == "Draining a tank" and out["title_confirmed"] is True
    plain = _engine(tmp_path / "p", _commission(tmp_path / "porder"))
    out = await plain._node_clarify({"topic": plain.config.topic})
    assert "title_confirmed" not in out, "no title given: the writer proposes one"


def test_an_old_checkpoints_paper_venue_still_picks_the_write_persona_and_the_final_render(tmp_path: Path) -> None:
    import launch

    eng = _engine(tmp_path, _commission(tmp_path / "order"))
    assert eng._write_venue({}) == eng.config.output.paper_format
    assert eng._write_venue({"clarify_answers": {"paper_venue": "Report"}}) == "report"

    class Art:
        raw_state = {"clarify_answers": {"paper_venue": "report"}}

    cfg = Config(commission=_commission(tmp_path / "o2"))
    launch._apply_paper_venue_override(cfg, Art())  # type: ignore[arg-type]
    assert cfg.output.paper_format == "report"
    Art.raw_state = {}
    cfg2 = Config(commission=_commission(tmp_path / "o3"))
    launch._apply_paper_venue_override(cfg2, Art())  # type: ignore[arg-type]
    assert cfg2.output.paper_format == type(cfg2.output).model_fields["paper_format"].default
