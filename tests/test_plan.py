"""plan.md: the plan step, the design adopting it, the pause, and the rewrite on request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml

from core import plan
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, PausesConfig, ProviderConfig,
)
from core.engine import Engine

DESIGN: dict[str, Any] = {
    "hypothesis": "model-based OPC reduces EPE more than rule-based",
    "variables": {
        "independent": ["correction_strategy"],
        "dependent": ["epe", "cd_error"],
        "controls": ["seed"],
    },
    "method": "compare three strategies on synthetic clips",
    "expected_outcome": "model-based wins on EPE",
    "figures_planned": ["comparison.png"],
    "dependencies": ["numpy", "matplotlib"],
    "result_assertions": [
        {"path": "cd_nm", "min": 0.5, "max": 500, "unit": "nm", "reason": "a CD is positive"},
    ],
}

EXTRA = {
    "in_short": "We compare model-based and rule-based OPC on synthetic clips.",
    "literature": [
        {"source": "Mack 2011", "says": "EPE is the standard OPC figure of merit"},
        {"source": "Cobb 1998", "says": "model-based OPC tunes to a calibrated simulator"},
    ],
    "gap": "No source compares them at low k1 with an independent evaluator.",
    "success_criteria": ["model-based EPE is lower with a 95% CI that excludes zero"],
    "risks": ["the synthetic clips may be easier than real layouts"],
    "out_of_scope": ["real mask data"],
}


def _cfg(tmp_path: Path, **pauses: Any) -> Config:
    return Config(
        topic="OPC under low k1",
        title="opc",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
        pauses=PausesConfig(**pauses),
    )


def _engine(tmp_path: Path, replies: list[str], **pauses: Any) -> Engine:
    eng = Engine(_cfg(tmp_path, **pauses))
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=replies)})()
    return eng


class Paused(Exception):
    """Stands in for LangGraph's GraphInterrupt."""


def _stop_at_pause(eng: Engine) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> None:
        seen.append(kwargs)
        raise Paused(kwargs["kind"])

    eng._pause_for_human = fake  # type: ignore[method-assign]
    return seen


def _plan_reply(design: dict[str, Any] = DESIGN, extra: dict[str, Any] | None = EXTRA) -> str:
    body = dict(design)
    if extra is not None:
        body["plan"] = extra
    return json.dumps(body)


def _prompt(call: Any) -> str:
    """The text the model was sent (the client is given a list of chat messages)."""
    return call.args[0][0]["content"]


def _audit_reply() -> str:
    return json.dumps({"objections_addressed": [
        {"check": "circular_evaluation", "fix": "an independent evaluator"},
    ]})


# --- the file --------------------------------------------------------------------


def test_render_then_parse_gives_back_the_design() -> None:
    text = plan.render("OPC under low k1", EXTRA, DESIGN, ["an independent evaluator"])
    parsed = plan.parse(text)
    assert parsed.error is None
    assert parsed.design == DESIGN
    for heading in ("In short", "What the literature says", "The gap this experiment addresses",
                    "Success criteria", "Risks", "What this quest will not do", "Checks already made"):
        assert f"## {heading}" in text
    assert "**Mack 2011**: EPE is the standard OPC figure of merit" in text


def test_a_design_with_unusual_characters_survives_the_round_trip() -> None:
    design = {**DESIGN, "hypothesis": "θ: a \"quoted\" value, 50% of 1e-3 — with: colons # and hashes",
              "method": "line one\nline two: still text"}
    assert plan.parse(plan.render("t", EXTRA, design)).design == design


def test_the_reader_can_edit_the_block_and_the_edit_is_what_parses() -> None:
    text = plan.render("t", EXTRA, DESIGN)
    edited = text.replace("model-based OPC reduces EPE more than rule-based",
                          "model-based OPC reduces CD error more than rule-based")
    assert plan.parse(edited).design["hypothesis"].endswith("CD error more than rule-based")


@pytest.mark.parametrize("bad, why", [
    ({"method": "x"}, "hypothesis"),
    ({"hypothesis": "h", "variables": ["a"]}, "variables"),
    ({"hypothesis": "h", "result_assertions": [{"min": 1}]}, "path"),
    ({"hypothesis": "h", "result_assertions": [{"path": "a", "min": 5, "max": 1}]}, "above"),
    ({"hypothesis": "h", "result_assertions": [{"path": "a", "min": "low"}]}, "number"),
    ({"hypothesis": "h", "method": ["a"]}, "method"),
])
def test_a_design_that_would_send_the_experiment_elsewhere_is_refused(bad: dict[str, Any], why: str) -> None:
    design, error = plan.normalize_design(bad)
    assert design is None
    assert why in (error or "")


def test_a_dependency_written_as_a_comma_string_is_accepted() -> None:
    design, error = plan.normalize_design({"hypothesis": "h", "dependencies": "numpy; scipy"})
    assert error is None
    assert design["dependencies"] == ["numpy", "scipy"]


def test_unreadable_blocks_say_what_is_wrong() -> None:
    assert "no `## The design" in plan.parse("# Plan\n\nnothing here\n").error
    no_fence = "## The design (used as written)\n\nhypothesis: x\n"
    assert "no fenced" in plan.parse(no_fence).error
    bad_yaml = "## The design (used as written)\n\n```yaml\nhypothesis: [unclosed\n```\n"
    assert "not valid YAML" in plan.parse(bad_yaml).error


def test_a_reply_wrapped_in_a_fence_is_unwrapped() -> None:
    inner = plan.render("t", EXTRA, DESIGN)
    assert plan.strip_outer_fence("```markdown\n" + inner + "```") == inner
    assert plan.strip_outer_fence(inner) == inner


def test_versions_are_kept_and_a_hand_edit_is_its_own_version(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    text = plan.render("t", EXTRA, DESIGN)
    plan.record_version(root, text, by="model")
    assert plan.note_edit(root, text) is None
    edited = text.replace("a CD is positive", "a CD is positive and finite")
    assert plan.note_edit(root, edited)["by"] == "user"
    rows = plan.history(root)
    assert [r["by"] for r in rows] == ["model", "user"]
    assert (root / ".fi" / "plan_versions" / "plan.v2.md").read_text(encoding="utf-8") == edited
    assert plan.note_edit(root, edited) is None


# --- config ------------------------------------------------------------------------


def test_the_plan_pause_is_off_by_default_and_reads_an_unquoted_off() -> None:
    assert PausesConfig().plan == "off"
    assert PausesConfig.model_validate(yaml.safe_load("plan: off")).plan == "off"
    assert PausesConfig(plan="ask").plan == "ask"
    with pytest.raises(Exception):
        PausesConfig(plan="sometimes")  # type: ignore[arg-type]


# --- the plan step -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_plan_step_writes_plan_md_from_one_call_and_one_audit(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [_plan_reply(), _audit_reply()])
    patch = await eng._node_plan({"topic": "OPC", "iteration": 0})

    path = plan.plan_path(eng.quest_root)
    text = path.read_text(encoding="utf-8")
    assert plan.parse(text).design == DESIGN
    assert "We compare model-based and rule-based OPC" in text
    assert "an independent evaluator" in text  # the audit shows under Checks already made
    assert eng._client.chat.await_count == 2  # the design-and-plan call, then the audit: no extra call
    prompt = _prompt(eng._client.chat.await_args_list[0])
    assert "Also write the plan" in prompt
    assert patch["design_objections"][0]["check"] == "circular_evaluation"
    rows = plan.history(eng.quest_root)
    assert [r["by"] for r in rows] == ["model"]


@pytest.mark.asyncio
async def test_the_plan_step_does_not_stop_unless_asked(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [_plan_reply(), _audit_reply()])
    seen = _stop_at_pause(eng)
    await eng._node_plan({"topic": "OPC", "iteration": 0})
    assert seen == []


@pytest.mark.asyncio
async def test_with_the_pause_on_the_quest_stops_once_the_plan_is_written(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [_plan_reply(), _audit_reply()], plan="ask")
    seen = _stop_at_pause(eng)
    with pytest.raises(Paused):
        await eng._node_plan({"topic": "OPC", "iteration": 0})
    assert plan.plan_path(eng.quest_root).is_file()
    assert seen[0]["kind"] == "plan"
    assert seen[0]["interaction"] == "supply"
    assert "plan.md" in " ".join(seen[0]["steps"])
    assert "--revise-plan" in " ".join(seen[0]["steps"])


@pytest.mark.asyncio
async def test_a_resume_reads_the_edited_file_and_does_not_stop_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [_plan_reply(), _audit_reply()], plan="ask")
    _stop_at_pause(eng)
    with pytest.raises(Paused):
        await eng._node_plan({"topic": "OPC", "iteration": 0})
    path = plan.plan_path(eng.quest_root)
    path.write_text(path.read_text(encoding="utf-8").replace("compare three strategies", "compare four strategies"),
                    encoding="utf-8")
    eng._client.chat.reset_mock()
    seen = _stop_at_pause(eng)
    await eng._node_plan({"topic": "OPC", "iteration": 0})  # the resume runs the node again
    assert seen == []
    assert eng._client.chat.await_count == 0  # nothing is written over
    assert "four strategies" in path.read_text(encoding="utf-8")
    assert [r["by"] for r in plan.history(eng.quest_root)] == ["model", "user"]


@pytest.mark.asyncio
async def test_an_unreadable_plan_stops_the_quest_each_time_with_the_reason(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("## The design (used as written)\n\n```yaml\nhypothesis: [unclosed\n```\n", encoding="utf-8")
    for _ in range(2):  # a marker from the first stop must not let the second resume through
        seen = _stop_at_pause(eng)
        with pytest.raises(Paused):
            await eng._node_plan({"topic": "OPC", "iteration": 0})
        assert "not valid YAML" in " ".join(seen[0]["steps"])


@pytest.mark.asyncio
async def test_an_unusable_first_draft_leaves_no_plan_and_lets_design_draft_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps({"method": "no hypothesis at all", "plan": EXTRA}), _audit_reply()])
    patch = await eng._node_plan({"topic": "OPC", "iteration": 0})
    assert patch == {}
    assert not plan.plan_path(eng.quest_root).exists()


@pytest.mark.asyncio
async def test_later_passes_and_analyze_mode_do_not_write_a_plan(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    assert await eng._node_plan({"topic": "OPC", "iteration": 1}) == {}
    assert not plan.plan_path(eng.quest_root).exists()
    eng.config.engine.analyze_local_first = True
    assert await eng._node_plan({"topic": "OPC", "iteration": 0}) == {}
    assert eng._client.chat.await_count == 0


@pytest.mark.asyncio
async def test_a_reply_with_no_plan_prose_still_gives_a_usable_file(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [_plan_reply(extra=None), _audit_reply()])
    await eng._node_plan({"topic": "OPC", "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "(not written)" in text
    assert plan.parse(text).design == DESIGN


# --- the design adopts the plan --------------------------------------------------------


@pytest.mark.asyncio
async def test_design_uses_the_design_block_as_written_and_asks_the_model_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    edited = {**DESIGN, "hypothesis": "the person's own hypothesis", "method": "the person's own method"}
    path.write_text(plan.render("OPC", EXTRA, edited), encoding="utf-8")

    patch = await eng._node_design({"topic": "OPC", "iteration": 0})

    assert patch["design"] == edited
    assert eng._client.chat.await_count == 0
    first = patch["design_history"][0]
    assert first["plan_sha256"] == plan.sha256(path.read_text(encoding="utf-8"))
    assert "plan.md" in first["reason"]


@pytest.mark.asyncio
async def test_design_without_a_plan_drafts_as_before_and_names_no_plan(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps(DESIGN), json.dumps({})])
    patch = await eng._node_design({"topic": "OPC", "iteration": 0})
    assert patch["design"]["hypothesis"] == DESIGN["hypothesis"]
    assert "plan_sha256" not in patch["design_history"][0]
    assert "Also write the plan" not in _prompt(eng._client.chat.await_args_list[0])


@pytest.mark.asyncio
async def test_a_later_pass_redesigns_from_the_review_not_from_the_plan(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps({**DESIGN, "hypothesis": "revised after the results"}), json.dumps({})])
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render("OPC", EXTRA, DESIGN), encoding="utf-8")
    patch = await eng._node_design({"topic": "OPC", "iteration": 1, "review": {"verdict": "revise"}})
    assert patch["design"]["hypothesis"] == "revised after the results"
    assert "plan_sha256" not in patch["design_history"][-1]


@pytest.mark.asyncio
async def test_an_unreadable_plan_at_design_falls_back_to_drafting(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [json.dumps(DESIGN), json.dumps({})])
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a plan", encoding="utf-8")
    patch = await eng._node_design({"topic": "OPC", "iteration": 0})
    assert patch["design"]["hypothesis"] == DESIGN["hypothesis"]


# --- asking for a change --------------------------------------------------------------


def _existing_plan(eng: Engine) -> Path:
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = plan.render("OPC", EXTRA, DESIGN)
    path.write_text(text, encoding="utf-8")
    plan.record_version(eng.quest_root, text, by="model")
    return path


@pytest.mark.asyncio
async def test_a_request_rewrites_the_file_and_keeps_the_old_version(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    path = _existing_plan(eng)
    revised = plan.render("OPC", EXTRA, {**DESIGN, "hypothesis": "model-based OPC reduces CD error"})
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value="```markdown\n" + revised + "```")})()

    out = await eng.revise_plan("use CD error, not EPE")

    assert out["version"] == 2
    assert plan.parse(path.read_text(encoding="utf-8")).design["hypothesis"] == "model-based OPC reduces CD error"
    rows = plan.history(eng.quest_root)
    assert [r["by"] for r in rows] == ["model", "request"]
    assert rows[1]["note"] == "use CD error, not EPE"
    assert "EPE more than rule-based" in (eng.quest_root / ".fi" / "plan_versions" / "plan.v1.md").read_text(encoding="utf-8")
    prompt = _prompt(eng._client.chat.await_args_list[0])
    assert "use CD error, not EPE" in prompt and "In short" in prompt


@pytest.mark.asyncio
async def test_a_request_that_breaks_the_design_block_is_retried_then_refused_and_changes_nothing(
    tmp_path: Path,
) -> None:
    eng = _engine(tmp_path, [])
    path = _existing_plan(eng)
    before = path.read_text(encoding="utf-8")
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value="# Plan\n\nI could not do that.\n")})()
    with pytest.raises(ValueError, match="unchanged"):
        await eng.revise_plan("make it better")
    assert eng._client.chat.await_count == 2
    assert "could not be used" in _prompt(eng._client.chat.await_args_list[1])
    assert path.read_text(encoding="utf-8") == before
    assert len(plan.history(eng.quest_root)) == 1


@pytest.mark.asyncio
async def test_a_second_try_that_works_is_used(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    path = _existing_plan(eng)
    good = plan.render("OPC", EXTRA, {**DESIGN, "method": "a better method"})
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=["oops", good])})()
    await eng.revise_plan("better method")
    assert plan.parse(path.read_text(encoding="utf-8")).design["method"] == "a better method"


@pytest.mark.asyncio
async def test_a_request_needs_words_and_a_plan(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    with pytest.raises(ValueError):
        await eng.revise_plan("   ")
    with pytest.raises(FileNotFoundError):
        await eng.revise_plan("change it")


@pytest.mark.asyncio
async def test_a_hand_edit_before_a_request_is_kept_as_its_own_version(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    path = _existing_plan(eng)
    path.write_text(path.read_text(encoding="utf-8").replace("a CD is positive", "a CD is finite"), encoding="utf-8")
    good = plan.render("OPC", EXTRA, {**DESIGN, "method": "changed"})
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value=good)})()
    await eng.revise_plan("change the method")
    assert [r["by"] for r in plan.history(eng.quest_root)] == ["model", "user", "request"]


@pytest.mark.asyncio
async def test_a_design_already_in_state_is_never_replaced_by_the_plan(tmp_path: Path) -> None:
    """Even at iteration 0: a re-entry after results (the analysis asked for a new experiment) must draft
    from the diagnosis, not re-adopt the plan and run the same design again."""
    eng = _engine(tmp_path, [json.dumps({**DESIGN, "hypothesis": "redesigned after the diagnosis"}), json.dumps({})])
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render("OPC", EXTRA, DESIGN), encoding="utf-8")
    state = {
        "topic": "OPC", "iteration": 0, "design": DESIGN,
        "analysis": {"next_step": "re_experiment", "summary": "units were wrong", "key_findings": []},
    }
    patch = await eng._node_design(state)
    assert patch["design"]["hypothesis"] == "redesigned after the diagnosis"
    assert eng._client.chat.await_count == 2



@pytest.mark.asyncio
async def test_the_one_shot_rewrite_builds_no_knowledge_layer_and_says_what_it_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """``--revise-plan`` is one model call on a markdown file: it must not load the knowledge layer (the embedding
    model and indexes) for it, and it reports the new version and the way on."""
    import launch
    from core.provider import ProxySupervisor

    built: list[bool] = []

    class _Knowledge:
        def __init__(self, cfg: Any) -> None:
            built.append(bool(cfg.enabled))

    monkeypatch.setattr("core.engine.Knowledge", _Knowledge)
    cfg = _cfg(tmp_path)
    cfg.knowledge = cfg.knowledge.model_copy(update={"enabled": True})
    quest_id = "1700000000-opc-abcdef"
    quest = cfg.output.output_dir / quest_id
    quest.mkdir(parents=True)
    text = plan.render("OPC", EXTRA, DESIGN)
    plan.plan_path(quest).write_text(text, encoding="utf-8")
    plan.record_version(quest, text, by="model")
    revised = plan.render("OPC", EXTRA, {**DESIGN, "method": "a changed method"})

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return revised

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    rc = await launch._revise_plan_once(cfg, quest_id, "change the method", supervisor=ProxySupervisor())

    assert rc == 0
    assert built == [False], "the knowledge layer was built for a one-line rewrite"
    assert plan.parse(plan.plan_path(quest).read_text(encoding="utf-8")).design["method"] == "a changed method"
    out = capsys.readouterr().out
    assert "plan revised (version 2)" in out and f"--resume {quest_id}" in out


@pytest.mark.asyncio
async def test_the_one_shot_rewrite_says_why_it_could_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    import launch
    from core.provider import ProxySupervisor

    cfg = _cfg(tmp_path)
    (cfg.output.output_dir / "1700000000-opc-abcdef").mkdir(parents=True)
    rc = await launch._revise_plan_once(cfg, "1700000000-opc-abcdef", "anything", supervisor=ProxySupervisor())
    assert rc == 1
    assert "has no plan.md yet" in capsys.readouterr().err


# --- a real model's protocol is repaired key by key, not refused whole ----------------------------------------------------


def test_a_protocol_key_that_cannot_be_checked_is_left_out_with_a_note_and_the_rest_is_kept() -> None:
    fixed, notes = plan.repair_protocol({
        "grid": {"R0": [1.5, 3.0]}, "runs_per_setting": 300,
        "thresholds": {"major outbreak": "more than 20% of the population"}, "ci_method": "Wilson",
    })
    assert fixed == {"grid": {"R0": [1.5, 3.0]}, "runs_per_setting": 300, "ci_method": "Wilson"}
    assert len(notes) == 1 and "`protocol.thresholds` was left out" in notes[0] and "must map each name to a number" in notes[0]


def test_several_bad_keys_are_each_left_out_and_a_good_protocol_is_untouched() -> None:
    fixed, notes = plan.repair_protocol({"grid": {"R0": []}, "runs_per_setting": 2.5, "seed_policy": "independent"})
    assert fixed == {"seed_policy": "independent"} and len(notes) == 2
    good = {"grid": {"R0": [1.5]}, "runs_per_setting": 10}
    assert plan.repair_protocol(good) == (good, [])
    assert plan.repair_protocol("nope")[0] is None


@pytest.mark.asyncio
async def test_a_plan_whose_protocol_has_one_unusable_key_is_still_written_and_says_which(tmp_path: Path) -> None:
    bad = {**DESIGN, "protocol": {"runs_per_setting": 300, "thresholds": ["a", "b"]}}
    eng = _engine(tmp_path, [_plan_reply(bad), _audit_reply()])
    await eng._node_plan({"topic": "OPC", "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    written = plan.parse(text).design
    assert written["protocol"] == {"runs_per_setting": 300}
    assert "`protocol.thresholds` was left out of the plan" in text
    assert eng._client.chat.await_count == 2, "the design was not drafted again"
