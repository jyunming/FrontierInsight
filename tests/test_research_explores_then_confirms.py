"""``rigor_profile: research`` explores first and confirms once by default (``engine.phased``, core/phased.py).

The profile turns it on where the config is silent; a config that says ``phased: false`` keeps it off (and the plan and
run.log say the result is not confirmed on unseen data); the default profile is unchanged; a research quest that began
before this default goes on as it began; a quest that runs no experiment of its own says so in one sentence and holds
nothing back; the interview shows it on for research on all three surfaces.

No real model is called."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from core import evidence as _evidence
from core import phased, plan_settings
from core.config import Config, EngineConfig

REPO = Path(__file__).resolve().parent.parent


def _research(root: Path, **engine: Any) -> dict[str, Any]:
    return {
        "topic": "does the default move", "title": "research-phased", "rigor_profile": "research",
        "provider": {"name": "openai", "node_models": {"review_panel.statistician": "m-other"}},
        "engine": {"max_iterations": 1, **engine},
        "knowledge": {"enabled": False},
        "output": {"output_dir": str(root)},
    }


def _engine_config(**over: Any) -> EngineConfig:
    """An engine section built in code that satisfies the research profile's non-negotiable settings."""
    return EngineConfig(max_iterations=1, protocol_check="block", oracle_check="block", numeric_warnings="block",
                        run_manifest_check="block", cross_check_verify=True, audit_trace=True,
                        review_panel=["methodologist", "statistician", "reproducibility", "devil_advocate"], **over)


# ---- the config ----------------------------------------------------------------------------------------------------


def test_research_turns_explore_then_confirm_on_where_the_config_is_silent(tmp_path: Path) -> None:
    assert Config.model_validate(_research(tmp_path)).engine.phased is True
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in _research(tmp_path).items() if k != "engine"}), encoding="utf-8")
    assert Config.from_yaml(path).engine.phased is True
    # Built in code: a section given without the setting gets it too.
    data = {**_research(tmp_path), "engine": _engine_config()}
    assert Config.model_validate(data).engine.phased is True


def test_an_explicit_off_is_kept_under_research_and_is_not_refused(tmp_path: Path) -> None:
    cfg = Config.model_validate(_research(tmp_path, phased=False))
    assert cfg.engine.phased is False and cfg.rigor_profile == "research"
    assert cfg.engine.oracle_check == "block", "the rest of the profile still applies"
    data = {**_research(tmp_path), "engine": _engine_config(phased=False)}
    assert Config.model_validate(data).engine.phased is False


def test_the_default_profile_is_unchanged(tmp_path: Path) -> None:
    assert EngineConfig().phased is False
    assert Config(topic="t").engine.phased is False
    data = {k: v for k, v in _research(tmp_path).items() if k != "rigor_profile"}
    assert Config.model_validate(data).engine.phased is False


# ---- a quest that began before the default ---------------------------------------------------------------------------


def _quest(tmp_path: Path, *, config_text: str, ran: bool) -> Any:
    from core.engine import Engine

    cfg = Config.model_validate(yaml.safe_load(config_text) | {"output": {"output_dir": str(tmp_path)}})
    engine = Engine(cfg)
    engine.quest_root.mkdir(parents=True, exist_ok=True)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "config.yaml").write_text(config_text, encoding="utf-8")
    if ran:
        engine.audit.append("node_completed", node="clarify")
    return engine, cfg


_OLD_CONFIG = (f"{plan_settings.INTERVIEW_MARK} Frontier Insight\n"
               "topic: an old research quest\ntitle: old-research\nrigor_profile: research\n"
               "provider:\n  name: openai\n  node_models:\n    review_panel.statistician: m-other\n"
               "knowledge:\n  enabled: false\n")


def test_a_research_quest_approved_with_it_off_before_the_default_moved_goes_on_as_it_began(tmp_path: Path) -> None:
    engine, cfg = _quest(tmp_path, config_text=_OLD_CONFIG, ran=True)
    # Approved (and recorded) with explore-then-confirm off: what FI's default was then.
    before = cfg.model_copy(update={"engine": cfg.engine.model_copy(update={"phased": False})})
    plan_settings.record(engine.fi_dir, before, engine.quest_root)
    assert json.loads((engine.fi_dir / plan_settings.NAME).read_text(encoding="utf-8"))["settings"]["engine.phased"] is False

    engine._phased_keep_off_if_began_before()
    assert engine.config.engine.phased is False and not phased.enabled(engine.config)
    assert cfg.engine.phased is True, "the caller's config object is not changed"
    # The approved settings agree with what runs: nothing to stop for, and the record is not rewritten.
    assert plan_settings.check(engine.quest_root, engine.fi_dir, engine.config) == []
    assert json.loads((engine.fi_dir / plan_settings.NAME).read_text(encoding="utf-8"))["settings"]["engine.phased"] is False
    # No confirm edge, no record of the two stages, and one plain line in run.log.
    branches = engine._build_graph().branches["evidence_gate"]
    (spec,) = branches.values()
    assert "confirm" not in dict(spec.ends or {})
    assert not phased.record_path(engine.quest_root).exists()
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "began before research quests explored first and confirmed once by default" in log
    # plan.md says it too.
    lines = engine._plan_confirm_lines({})
    assert lines and "began before" in "\n".join(lines)


def test_a_new_research_quest_or_one_that_asks_for_it_keeps_it_on(tmp_path: Path) -> None:
    engine, _cfg = _quest(tmp_path / "new", config_text=_OLD_CONFIG, ran=False)
    engine._phased_keep_off_if_began_before()
    assert engine.config.engine.phased is True, "a quest that never ran starts with the new default"
    asked = _OLD_CONFIG + "engine:\n  phased: true\n"
    engine, _cfg = _quest(tmp_path / "asked", config_text=asked, ran=True)
    engine._phased_keep_off_if_began_before()
    assert engine.config.engine.phased is True, "a line the person wrote is theirs"
    # A quest that ran with it on and lost its record of the two stages is not taken for one that began before.
    engine, _cfg = _quest(tmp_path / "lost", config_text=_OLD_CONFIG, ran=False)
    engine.audit.append("phased_started")
    engine.audit.append("node_completed", node="clarify")
    engine._phased_keep_off_if_began_before()
    assert engine.config.engine.phased is True


def test_an_older_research_quest_resumed_runs_as_it_began_and_does_not_stop(tmp_path: Path) -> None:
    """The whole start, up to the model connection: an old research quest approved with it off, resumed after the
    default moved, is neither stopped for a changed setting nor given the two stages (no record, no hold-back)."""
    import asyncio

    engine, cfg = _quest(tmp_path, config_text=_OLD_CONFIG, ran=True)
    before = cfg.model_copy(update={"engine": cfg.engine.model_copy(update={"phased": False})})
    plan_settings.record(engine.fi_dir, before, engine.quest_root)
    engine.audit.append("plan_settings_recorded",
                       sha256=__import__("hashlib").sha256((engine.fi_dir / plan_settings.NAME).read_bytes()).hexdigest())
    stops: list[list[str]] = []

    def stop(changed):  # noqa: ANN001
        stops.append(changed)
        raise RuntimeError("stopped")

    engine._stop_for_changed_settings = stop  # type: ignore[method-assign]

    async def no_llm():  # the start stops here: nothing past the settings check is under test
        raise RuntimeError("reached the model connection")

    engine._connect_llm = no_llm  # type: ignore[method-assign]
    engine._review_models_stop = lambda: None  # type: ignore[method-assign]

    async def no_setup(*_a, **_k):  # noqa: ANN002, ANN003
        return None

    engine.executor.setup = no_setup  # type: ignore[method-assign]
    engine._record_environment = no_setup  # type: ignore[method-assign]
    engine._preflight_required_skills = no_setup  # type: ignore[method-assign]
    data = _csv(engine.quest_root / "inputs" / "data" / "d.csv", 100)
    with pytest.raises(RuntimeError, match="reached the model connection"):
        asyncio.run(engine.run())
    assert stops == [], stops
    assert engine.config.engine.phased is False
    assert not phased.record_path(engine.quest_root).exists()
    assert (engine.quest_root / "inputs" / "data" / "d.csv").read_bytes() == data
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "exploration stage" not in log and "began before research quests" in log
    assert "approve that change with `--update`" in log
    from core import audit_log

    assert not any(e.get("kind") == phased.STARTED_EVENT for e in audit_log.read(engine.fi_dir / "audit.jsonl"))


def test_an_update_of_an_older_research_quest_approves_what_runs(tmp_path: Path) -> None:
    """--update of a research quest that began before the default: approved off (as it runs), not "approved on"."""
    from core.interview_update import approve_settings

    engine, cfg = _quest(tmp_path, config_text=_OLD_CONFIG, ran=True)
    before = cfg.model_copy(update={"engine": cfg.engine.model_copy(update={"phased": False})})
    plan_settings.record(engine.fi_dir, before, engine.quest_root)
    said: list[str] = []
    changed = approve_settings(engine.quest_root, cfg, say=said.append)
    assert not any("engine.phased" in line for line in changed + said), (changed, said)
    record = json.loads((engine.fi_dir / plan_settings.NAME).read_text(encoding="utf-8"))
    assert record["settings"]["engine.phased"] is False
    # The next start agrees with that approval.
    engine._phased_keep_off_if_began_before()
    assert plan_settings.check(engine.quest_root, engine.fi_dir, engine.config) == []


# ---- a quest with no experiment of its own ---------------------------------------------------------------------------


def _csv(path: Path, rows: int) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("x,y\n" + "".join(f"{i},{i * i}\n" for i in range(rows))).encode()
    path.write_bytes(data)
    return data


def test_a_quest_with_no_experiment_says_so_in_one_sentence_and_holds_nothing_back(tmp_path: Path) -> None:
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    phased.prepare(tmp_path, "q1")
    assert (tmp_path / "inputs" / "data" / "d.csv").read_bytes() != original, "rows held back at the start"
    lines = phased.mark_not_applicable(tmp_path, "this quest is a literature survey and runs no experiment")
    assert len(lines) == 1 and lines[0].startswith("explore-then-confirm does not apply: this quest is a literature")
    assert "the rows held back at the start are back in inputs/data/" in lines[0]
    assert (tmp_path / "inputs" / "data" / "d.csv").read_bytes() == original, "the analysis sees all of the data"
    record = phased.load(tmp_path)
    assert phased.status(record) == phased.NOT_APPLICABLE and record["files"] == []
    assert phased.mark_not_applicable(tmp_path, "again") == [], "said once"
    # A later start holds nothing back again.
    phased.prepare(tmp_path, "q1")
    assert (tmp_path / "inputs" / "data" / "d.csv").read_bytes() == original
    # No stage gap on the evidence ladder, no stage note in the paper, nothing for the writer.
    settings = phased.evidence_settings(tmp_path)
    assert settings["phased"] == phased.NOT_APPLICABLE and settings["phased_strategy"] == ""
    # Turned off later: nothing is said about data exploration may have seen (there was no exploration stage).
    assert phased.turned_off(tmp_path) == []
    assert not phased.load(tmp_path).get("late_start")
    ready = next(level["gaps"] for level in _evidence.assess(tmp_path, {"result_json": {"a": 1}}, settings=settings)["ladder"]
                 if level["level"] == "publication_ready")
    base = next(level["gaps"] for level in _evidence.assess(tmp_path, {"result_json": {"a": 1}}, settings={})["ladder"]
                if level["level"] == "publication_ready")
    assert ready == base
    paper = "# Title\n\nBody.\n"
    assert phased.mark_paper(paper, record) == paper
    marked = phased.mark_paper(paper, {**record, "not_applicable": ""})  # an earlier note from exploration
    assert "fi:phased" in marked and "fi:phased" not in phased.mark_paper(marked, record)
    assert phased.write_note(tmp_path) == ""
    assert phased.unconfirmable(tmp_path)


def _engine(root: Path, **engine: Any):  # noqa: ANN202
    from core.engine import Engine

    return Engine(Config.model_validate(_research(root, **engine)))


def test_a_research_survey_pinned_in_the_config_holds_nothing_back_and_says_why(tmp_path: Path) -> None:
    engine = _engine(tmp_path, survey_mode=True)
    assert phased.enabled(engine.config)
    data = _csv(engine.quest_root / "inputs" / "data" / "d.csv", 100)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._phased_prepare()
    assert (engine.quest_root / "inputs" / "data" / "d.csv").read_bytes() == data
    assert phased.status(phased.load(engine.quest_root)) == phased.NOT_APPLICABLE
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert log.count("explore-then-confirm does not apply") == 1 and "literature survey" in log
    lines = engine._plan_confirm_lines({"survey_mode_resolved": True, "no_simulation_resolved": True})
    assert any("does not apply" in line for line in lines)


def test_no_simulation_decided_at_clarify_keeps_the_rows_held_back_for_a_data_confirm_run(tmp_path: Path) -> None:
    """Until data quests had a confirm run, a quest found at clarify to analyse data put its held-back rows back and
    could never be confirmed. Now it is a data quest: what was held back stays held back (and its own data folder is
    held back too), so its data-reading step can be confirmed on rows exploration never read."""
    engine = _engine(tmp_path)
    data = _csv(engine.quest_root / "inputs" / "data" / "d.csv", 100)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._phased_prepare()
    assert (engine.quest_root / "inputs" / "data" / "d.csv").read_bytes() != data
    import asyncio

    async def clarify(state):  # noqa: ANN001 -- the clarify step's own answer: this quest collects data
        return {"no_simulation_resolved": True, "survey_mode_resolved": False, "clarify_answers": {}}

    engine._node_clarify = clarify  # type: ignore[method-assign]
    out = asyncio.run(engine._node_clarify_then_phased({}))
    assert out["no_simulation_resolved"] is True
    record = phased.load(engine.quest_root)
    assert record["data_quest"] is True and phased.status(record) == phased.EXPLORE
    # Its data-reading step reads data/ only: a table in inputs/data/ is not what its numbers come from, so it goes
    # back whole and the record says where to put it to have it confirmed.
    assert (engine.quest_root / "inputs" / "data" / "d.csv").read_bytes() == data
    assert "put it in data/" in record["why_no_data"]
    # The same table in data/ is held back.
    engine2 = _engine(tmp_path / "in-data")
    data2 = _csv(engine2.quest_root / "data" / "d.csv", 100)
    engine2.fi_dir.mkdir(parents=True, exist_ok=True)
    engine2._phased_prepare()
    engine2._node_clarify = clarify  # type: ignore[method-assign]
    asyncio.run(engine2._node_clarify_then_phased({}))
    assert (engine2.quest_root / "data" / "d.csv").read_bytes() != data2, "held back"
    # Without a result yet, the route writes from exploration, as before.
    assert engine._phased_route({"no_simulation_resolved": True}, "write") == "write"


def test_a_quest_that_runs_an_experiment_is_not_marked(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._phased_prepare()
    import asyncio

    async def clarify(state):  # noqa: ANN001
        return {**engine._resolve_modes({}), "clarify_answers": {}}

    engine._node_clarify = clarify  # type: ignore[method-assign]
    asyncio.run(engine._node_clarify_then_phased({}))
    assert phased.status(phased.load(engine.quest_root)) == phased.EXPLORE


# ---- plan.md and run.log -------------------------------------------------------------------------------------------


def test_the_plan_says_the_result_will_be_confirmed_once_and_what_it_costs(tmp_path: Path) -> None:
    engine = _engine(tmp_path, execute_replicates=3)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._phased_prepare()
    lines = engine._plan_confirm_lines({})
    text = "\n".join(lines)
    assert lines[0] == f"## {phased.PLAN_HEADING}"
    assert "run once more on new random seeds exploration never used" in text
    assert "one more full run of the frozen design" in text
    assert "as many runs per setting as an exploration run" in text
    assert "[phased] plan: The model may try designs" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    # A record that can no longer be confirmed: the plan does not promise a confirm run.
    phased.mark_compromised(engine.quest_root, "a data file could not be written")
    text = "\n".join(engine._plan_confirm_lines({}))
    assert "cannot be confirmed once more" in text and "a data file could not be written" in text
    assert "costs one more full run" not in text


def test_an_explicit_off_under_research_is_said_in_the_plan_and_run_log(tmp_path: Path) -> None:
    engine = _engine(tmp_path, phased=False)
    text = "\n".join(engine._plan_confirm_lines({}))
    assert "will not be confirmed once more on data or seeds the exploration never saw" in text
    assert "`engine.phased: false`" in text
    # Explicit off on a survey: the plan says the two stages would not apply anyway, not that a design may change.
    survey = _engine(tmp_path / "survey", phased=False, survey_mode=True)
    text = "\n".join(survey._plan_confirm_lines({"survey_mode_resolved": True, "no_simulation_resolved": True}))
    assert "does not apply" in text and "design may be changed" not in text


def test_the_default_profile_plan_says_nothing(tmp_path: Path) -> None:
    from core.engine import Engine

    engine = Engine(Config(topic="t", output={"output_dir": str(tmp_path)}))
    assert engine._plan_confirm_lines({}) == []


def test_the_plan_renders_the_section(tmp_path: Path) -> None:
    from core import plan as _plan

    lines = phased.plan_lines(None, on=True, research=True, runs_code=True)
    text = _plan.render("topic", {}, {"hypothesis": "h"}, confirm=lines)
    assert f"## {phased.PLAN_HEADING}" in text
    assert _plan.parse(text).error is None


# ---- the interview, on all three surfaces --------------------------------------------------------------------------


def test_the_interview_default_is_on_for_research_and_a_decision_off_for_exploring() -> None:
    from core.interview import QUESTIONS, build_smart_defaults, derive_tier3

    (q,) = [q for q in QUESTIONS if q.id == "phased"]
    assert q.default is True and q.mid_quest_editable is False
    for use, want in (("research", True), ("decision", True), ("explore", False), ("", True)):
        assert derive_tier3({"result_use": use})["phased"] is want, use
        assert build_smart_defaults({"result_use": use})["phased"] is want, use
    labels = [c.label for c in q.choices]
    assert any("default for research" in label for label in labels)
    assert not any(label == "No (default)" for label in labels)


def test_the_schema_snapshot_carries_the_new_default() -> None:
    from core.interview import export_schema_json

    snapshot = json.loads((REPO / "core" / "interview_schema.json").read_text(encoding="utf-8"))
    (q,) = [q for q in snapshot["questions"] if q["id"] == "phased"]
    assert q["default"] is True
    assert snapshot == json.loads(json.dumps(export_schema_json())), "regenerate core/interview_schema.json"


def _answers(**over: Any):  # noqa: ANN202
    from core.interview import InterviewAnswers

    base = dict(topic="t", title="t", output_kinds=["paper_md"], paper_format="generic", no_simulation=False,
                study_depth="journal-length", comparative_baseline="", success_metric="", budget="",
                clarify_mode="auto", review_panel=[], knowledge_enabled=False, provider="openai",
                provider_model="gpt-4o", second_reviewer_model="m-other")
    return InterviewAnswers(**{**base, **over})


def test_the_cli_writes_off_under_research_and_nothing_when_unanswered(tmp_path: Path) -> None:
    from core.interview import answers_to_yaml

    def load(text: str) -> Config:
        path = tmp_path / "c.yaml"
        path.write_text(text, encoding="utf-8")
        return Config.from_yaml(path)

    on = answers_to_yaml(_answers(result_use="research", phased=True))
    assert "  phased: true" in on.splitlines() and load(on).engine.phased is True
    off = answers_to_yaml(_answers(result_use="research", phased=False))
    assert "  phased: false" in off.splitlines() and load(off).engine.phased is False
    unset = answers_to_yaml(_answers(result_use="research"))
    assert "phased" not in unset and load(unset).engine.phased is True
    explore_off = answers_to_yaml(_answers(result_use="explore", phased=False))
    assert "phased" not in explore_off and load(explore_off).engine.phased is False


def test_an_update_keeps_a_research_quest_s_unsaid_setting_unsaid(tmp_path: Path) -> None:
    from core.interview import answers_to_yaml
    from core.interview_update import load_current_answers

    root = tmp_path / "q"
    root.mkdir()
    (root / "config.yaml").write_text(answers_to_yaml(_answers(result_use="research")), encoding="utf-8")
    current = load_current_answers(root)[0]
    assert current.phased is None
    assert "phased" not in answers_to_yaml(current), "an update must not write the default back as off"


def test_the_web_page_and_the_vscode_interview_default_it_from_the_result_use() -> None:
    html = (REPO / "web" / "static" / "interview.html").read_text(encoding="utf-8")
    assert "phased: ((tier1 && tier1.result_use) || 'research') !== 'explore'" in html
    ts = (REPO / "vscode-frontier-insight" / "src" / "interview.ts").read_text(encoding="utf-8")
    assert 'phased: resultUse !== "explore"' in ts
    assert "if (phasedWasDefault) a.phased = v.value !== \"explore\";" in ts
    assert "Yes: explore first, then confirm once (default for research)" in ts
    core_ts = (REPO / "vscode-frontier-insight" / "src" / "interview-core.ts").read_text(encoding="utf-8")
    assert 'answers.phased === false && rigorProfile === "research"' in core_ts


def test_the_web_form_writes_off_under_research_and_leaves_a_missing_answer_to_the_profile(tmp_path: Path) -> None:
    from tests.test_web_interview import _client, _ok_answers_payload

    client = _client(tmp_path)
    research = {**_ok_answers_payload(), "result_use": "research", "second_reviewer_model": "m-other"}
    for sent, want in (({"phased": False}, False), ({"phased": True}, True), ({}, True)):
        res = client.post("/api/interview/submit", json={**research, **sent})
        assert res.status_code == 200, res.text
        assert Config.from_yaml(Path(res.json()["yaml_path"])).engine.phased is want, sent
    assert client.post("/api/interview/submit", json={**research, "phased": "no"}).status_code == 400


def test_the_vscode_emitter_writes_off_under_research(tmp_path: Path) -> None:
    from tests.test_vscode_interview_yaml import _emit

    yaml_text, cfg = _emit(tmp_path, result_use="research", second_reviewer_model="m-other", phased=False)
    assert "  phased: false" in yaml_text.splitlines() and cfg.engine.phased is False
    yaml_text, cfg = _emit(tmp_path, result_use="research", second_reviewer_model="m-other", phased=True)
    assert cfg.engine.phased is True
    yaml_text, cfg = _emit(tmp_path, result_use="research", second_reviewer_model="m-other")
    assert "phased" not in yaml_text and cfg.engine.phased is True
