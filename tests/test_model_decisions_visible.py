"""A person can see the model's decisions: the reasons a model already writes reach ``--why`` (and the trace, as the
model's own account), Copilot is asked for Claude's thinking by default, and the model the VS Code chat panel picked
is the one a chat-started quest uses and says it used.

No real model is called: stand-in clients, a mock bridge server, a fake Codex binary, and the extension's TypeScript
compiled and run under node with a fake chat model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import attempt_records as ar
from core import audit_log as al
from core import thinking_capture as tc
from core import why
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, _format_review_for_writer, _take_refine_why
from core.provider import LLMClient, ResolvedEndpoint, _CLI_SPECS, _CliSpec, _extract_codex_reasoning, _run_cli
from tests.test_vscode_bridge import _MockBridgeServer

MESSAGES = [{"role": "user", "content": "hi"}]


def _engine(tmp_path: Path, *, provider: str = "openai", model: str | None = None, save_thinking: bool = True) -> Engine:
    cfg = Config(
        topic="decision trace", title="decisions",
        provider=ProviderConfig(name=provider, model=model),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out", save_thinking=save_thinking),
    )
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    return eng


def _claims(eng: Engine) -> list[dict[str, Any]]:
    return [e for e in al.read(eng.audit.path) if e["kind"] == "model_claim"]


def _called(eng: Engine, *keys: str) -> None:
    """As if ``_chat`` had answered under each key during this run of the step."""
    for key in keys:
        eng._last_chat[key] = {"provider": "openai", "model": "gpt-5", "prompt_hash": "p" * 64, "response_hash": "r" * 64}


def _run_log(eng: Engine) -> str:
    for h in eng._log.handlers:
        h.flush()
    return (eng.fi_dir / "run.log").read_text(encoding="utf-8")


# ---- 2. every reason the model already writes becomes a model_claim ------------------------------------------------


def test_ideate_s_chosen_idea_and_what_the_comparison_and_self_critique_said(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _called(eng, "ideate", "ideate_tournament")
    eng._audit_claims("ideate", {
        "ideas": [{"title": "A"}, {"title": "B"}],
        "chosen_idea": {"title": "B", "rationale": "B is cheaper [tournament] B tests the claim directly"},
        "ideate_tournament": {"winner_idx": 1, "outcome": "swapped", "matches": [
            {"a_idx": 0, "b_idx": 1, "winner": "B", "reason": "B tests the claim directly", "margin": "decisive"}]},
    })
    first, compared = _claims(eng)
    assert first["topic"] == "idea_chosen" and first["claim"] == "B is cheaper"
    assert "first pick" in first["decision"], "the first rationale was for the model's own first pick, not the winner"
    assert first["options"] == ["A", "B"] and first["model"] == "gpt-5"
    assert compared["topic"] == "idea_compared" and compared["claim"] == "B tests the claim directly"
    assert compared["decision"] == "B"

    eng2 = _engine(tmp_path / "2")
    _called(eng2, "ideate", "ideate_reflect")
    eng2._audit_claims("ideate", {
        "ideas": [{"title": "A"}, {"title": "B"}], "chosen_idea": {"title": "B", "rationale": "refined"},
        "ideate_critique": {"swap_to": "B", "refined_rationale": "refined"},
    })
    (only,) = _claims(eng2)
    assert only["topic"] == "idea_reconsidered" and only["decision"] == "switched to B" and only["claim"] == "refined"


def test_skill_choices_are_the_model_s_but_a_required_skill_is_not(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _called(eng, "select_skills")
    eng._audit_claims("select_skills", {"skill_selection": {
        "reasons": {"units": "the design mixes nm and um", "stats": "required by engine.skills_required"},
        "declined": {"figures": "no figure is planned"}, "forced": ["stats"]}})
    claims = _claims(eng)
    assert [(c["topic"], c["decision"]) for c in claims] == [
        ("skill_chosen", "use units"), ("skill_declined", "do not use figures")]
    assert claims[0]["claim"] == "the design mixes nm and um"


def test_how_each_finding_sits_against_the_literature(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _called(eng, "cross_check_verify")
    eng._audit_claims("cross_check", {"cross_check": [{
        "finding": "RK4 drifts less", "summary": "the literature broadly agrees",
        "supporting": [{"index": 2, "why": "reports the same drift ordering"}], "conflicting": [], "neutral": [],
        "candidates": [{"title": "Paper one"}, {"title": "Hairer 2006"}], "first_pass": {"verdict": "supporting"},
    }]})
    verdict, support = _claims(eng)
    assert verdict["topic"] == "literature_verdict" and verdict["decision"] == "supporting"
    assert support["decision"] == "supporting: Hairer 2006" and support["claim"] == "reports the same drift ordering"
    assert support["finding"] == "RK4 drifts less"


def test_a_repair_s_patch_summary_and_a_give_up_reason(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _called(eng, "execute_reflect")
    eng._audit_claims("execute_reflect", {"exec_reflect_iter": 2, "exec_reflect_history": [
        {"iter": 1, "patch_summary": "old fix"}, {"iter": 2, "patch_summary": "cast n to int; range() needs one"}]})
    (repair,) = _claims(eng)
    assert repair["claim"] == "cast n to int; range() needs one" and repair["decision"] == "repaired the script (attempt 2)"

    eng2 = _engine(tmp_path / "2")
    _called(eng2, "execute_reflect")
    eng2._audit_claims("execute_reflect", {"exec_give_up_reason": "needs a GPU", "exec_reflect_iter": 1})
    (gave_up,) = _claims(eng2)
    assert gave_up["decision"] == "gave up" and gave_up["claim"] == "needs a GPU"


def test_the_engine_s_own_placeholders_are_not_the_model_s_words(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _called(eng, "execute_reflect")
    eng._audit_claims("execute_reflect", {"exec_give_up_reason": "(LLM produced no patched code)", "exec_reflect_iter": 1})
    eng._audit_claims("execute_reflect", {"exec_reflect_iter": 1, "exec_reflect_history": [
        {"iter": 1, "patch_summary": "(no summary)"}]})
    assert _claims(eng) == []
    # A give-up the run-record check turned into a stop is only in the history.
    eng._audit_claims("execute_reflect", {"exec_reflect_iter": 1, "exec_reflect_history": [
        {"iter": 1, "patch_summary": "(gave up: the solver needs a licence (FLEXlm))"}]})
    (claim,) = _claims(eng)
    assert claim["claim"] == "the solver needs a licence (FLEXlm)" and claim["decision"] == "gave up"


def test_a_step_s_reasoning_note_does_not_count_another_step_s_calls(tmp_path: Path) -> None:
    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    ar.append(root / ".fi", tc.THINKING_FILE, {"node": "execute_reflect", "thinking": "yyy"})
    _trace(root, [("node_started", "execute", {}), ("node_started", "execute_reflect", {}),
                  ("node_started", "execute", {})])
    assert "none for this run of the step" in why.explain(root, "execute")
    assert "1 call(s)" in why.explain(root, "execute_reflect")


def test_the_setup_answers_the_model_chose_and_why(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _called(eng, "clarify")
    eng._audit_claims("clarify", {
        "clarify_questions": {"simulatability": {"question": "?", "default": "no", "reason": "needs survey data"},
                              "budget": {"question": "?", "default": "minutes"}},
        "clarify_answers": {"simulatability": "yes", "budget": "minutes"},
    })
    (claim,) = _claims(eng)
    assert claim["topic"] == "setup/simulatability" and claim["decision"] == "no"
    assert claim["answer_used"] == "yes", "a person's own answer is shown beside the model's"


def test_the_review_s_one_line_why_is_recorded_and_not_carried_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _called(eng, "review")
    out = eng._audit_claims("review", {"review": {"verdict": "accept", "why": "every claim is grounded",
                                                   "weaknesses": []}})
    assert "why" not in out["review"], "later prompts that read the review do not carry it"
    (verdict,) = _claims(eng)
    assert verdict["claim"] == "accept" and verdict["reason"] == "every claim is grounded"
    assert '"why"' in (Path(__file__).resolve().parent.parent / "agents" / "review.md").read_text(encoding="utf-8")


def test_the_panel_moderator_s_rationale(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _called(eng, "review_moderator")
    eng._audit_claims("review", {"review": {"verdict": "revise", "rationale": "two reviewers found the same gap"}})
    topics = {c["topic"]: c for c in _claims(eng)}
    assert topics["review_moderator"]["claim"] == "two reviewers found the same gap"


@pytest.mark.asyncio
async def test_a_reason_left_in_the_state_by_an_earlier_run_is_not_recorded_again(tmp_path: Path) -> None:
    """Through the real node wrapper: a step that made no call this time records nothing, one that did records it."""
    eng = _engine(tmp_path)
    _called(eng, "select_skills")  # an earlier run's call
    patch = {"skill_selection": {"reasons": {"units": "mixed units"}}}

    async def quiet(state):  # noqa: ANN001
        return patch

    await eng._audited("select_skills", quiet)({"topic": "t"})
    assert _claims(eng) == []

    class _Client:
        last_usage, last_model = None, "m"

        async def chat(self, messages, **kw):  # noqa: ANN001
            return "{}"

    eng._client = _Client()  # type: ignore[assignment]

    async def asks(state):  # noqa: ANN001
        await eng._chat("pick skills", node="select_skills")
        return patch

    await eng._audited("select_skills", asks)({"topic": "t"})
    (claim,) = _claims(eng)
    assert claim["node"] == "select_skills" and claim["claim"] == "mixed units"


# ---- 3. one extra short reason at the route-changing refine answer ------------------------------------------------


def test_the_writer_is_asked_why_its_answer_takes_its_route_and_the_line_leaves_the_paper() -> None:
    state = {"feedback_history": [{"iteration": 1, "text": "add a runtime table"}],
             "human_feedback": {"action": "refine"}}
    assert "REFINE_WHY:" in _format_review_for_writer(state, refine_round=True)
    paper, reason = _take_refine_why("# P\n\nBody.\n\n**REFINE_WHY:** only the text was unclear\n")
    assert reason == "only the text was unclear" and "REFINE_WHY" not in paper and "Body." in paper
    assert _take_refine_why("# P\n\nBody.\n") == ("# P\n\nBody.\n", "")


# ---- 4. --why says whether the reasoning record exists, and how long ----------------------------------------------


def _trace(root: Path, events: list[tuple[str, str, dict]]) -> None:
    log = al.AuditLog(root / ".fi" / "audit.jsonl", root.name)
    for kind, node, fields in events:
        log.append(kind, node=node, provenance=al.MODEL_CLAIM if kind == "model_claim" else al.DETERMINISTIC, **fields)


def test_why_lists_each_step_s_reasons_and_the_reasoning_note(tmp_path: Path) -> None:
    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    _trace(root, [("node_started", "ideate", {}),
                  ("model_claim", "ideate", {"topic": "idea_chosen", "claim": "B is cheaper", "decision": "B"})])
    ar.append(root / ".fi", tc.THINKING_FILE, {"node": "ideate_reflect", "thinking": "x" * 1234, "note": tc.THINKING_NOTE})
    _trace(root, [("node_completed", "ideate", {}), ("node_started", "select_skills", {}),
                  ("model_claim", "select_skills", {"topic": "skill_chosen", "claim": "mixed units", "decision": "use units"}),
                  ("node_completed", "select_skills", {})])

    step = why.explain(root, "ideate")
    assert "    - B is cheaper -> B" in step
    assert "1 call(s) of this run of the step, 1,234 characters in all" in step and "x" * 50 not in step, "the text stays in the file"
    assert "not its hidden reasoning" in step
    assert "none for this run of the step" in why.explain(root, "select_skills")

    everything = why.explain(root, "reasons")
    assert "ideate:" in everything and "select_skills:" in everything and "mixed units" in everything
    assert "not its hidden reasoning" in everything

    (root / ".fi" / tc.THINKING_FILE).unlink()
    assert ".fi/thinking.jsonl does not exist" in why.explain(root, "ideate")


def test_the_launcher_and_the_extension_offer_the_reasons_view() -> None:
    repo = Path(__file__).resolve().parent.parent
    assert "reasons (the reasons the model gave at every step)" in (repo / "launch.py").read_text(encoding="utf-8")
    assert "reasons" in (repo / "vscode-frontier-insight" / "src" / "trace.ts").read_text(encoding="utf-8")
    assert "loadWhy('reasons')" in (repo / "web" / "static" / "quest.html").read_text(encoding="utf-8")


# ---- thinking: a list value, asking over the bridge, a refusal, no reasoning at all --------------------------------


def test_a_reasoning_value_made_of_several_strings_is_kept_whole() -> None:
    assert tc.as_text(["first part. ", "second part."]) == "first part. second part."
    assert tc.as_text([1, 2]) is None
    holder, token = tc.open_holder()
    try:
        tc.note_thinking(["a", "b"])
        tc.add_thinking(["c"])
        assert holder["text"] == "abc"
    finally:
        tc.close_holder(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("save", [True, False])
async def test_the_bridge_request_asks_for_the_reasoning_only_when_it_is_kept(tmp_path: Path, save: bool) -> None:
    server = _MockBridgeServer()

    def handler(msg: dict, w) -> list[dict]:  # noqa: ANN001
        if msg["type"] != "lm_request":
            return []
        return [{"type": "lm_done", "id": msg["id"], "content": "Answer",
                 "thinking": ["weighed A ", "then B"]}]

    port = await server.start(handler)
    client = LLMClient(ResolvedEndpoint(base_url="", model="(VSCode chat default)", api_key="not-needed",
                                        transport="vscode_bridge", vscode_bridge_port=port))
    eng = _engine(tmp_path, provider="vscode_extension", save_thinking=save)
    eng._client = client  # type: ignore[assignment]
    try:
        assert await eng._chat("prompt", node="design") == "Answer"
        await client.chat(MESSAGES, node="paper")  # a generator's call: nothing keeps its reasoning
    finally:
        await client.aclose()
        await server.stop()
    asks = [m.get("ask_thinking") for m in server.received if m.get("type") == "lm_request"]
    assert asks == [save, False]
    if save:
        (line,) = ar.read(eng.fi_dir, tc.THINKING_FILE)
        assert line["thinking"] == "weighed A then B", "a list of strings is joined, not dropped"


@pytest.mark.asyncio
async def test_a_refusal_and_no_reasoning_are_each_said_once_in_run_log(tmp_path: Path) -> None:
    server = _MockBridgeServer()

    def handler(msg: dict, w) -> list[dict]:  # noqa: ANN001
        if msg["type"] != "lm_request":
            return []
        return [{"type": "lm_done", "id": msg["id"], "content": "Answer",
                 "served_model": {"id": "claude-opus-5", "vendor": "copilot"},
                 "thinking_declined": "Error: unknown model option _enableThinking"}]

    port = await server.start(handler)
    client = LLMClient(ResolvedEndpoint(base_url="", model="(VSCode chat default)", api_key="not-needed",
                                        transport="vscode_bridge", vscode_bridge_port=port))
    eng = _engine(tmp_path, provider="vscode_extension")
    eng._client = client  # type: ignore[assignment]
    try:
        for _ in range(3):
            await eng._chat("prompt", node="design")
        await eng._chat("prompt", node="analyze")
    finally:
        await client.aclose()
        await server.stop()
    log = _run_log(eng)
    assert log.count("did not accept FI's request for its reasoning") == 1
    # Through VS Code the missing reasoning is said once per model (with the route that keeps it), not per step.
    assert log.count("returned no reasoning through VS Code") == 1
    assert "this model/connection returned no reasoning for this step" not in log
    assert not (eng.fi_dir / tc.THINKING_FILE).exists()


def test_a_stream_that_failed_after_asking_is_retried_by_fi() -> None:
    from core.provider import _is_bridge_error_transient

    assert _is_bridge_error_transient("the model did not accept the request for its reasoning (400); asking again")


@pytest.mark.asyncio
async def test_codex_s_reasoning_summaries_are_kept(tmp_path: Path) -> None:
    events = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "**Plan** compare both"}},
        {"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "**Plan** compare both"}},
        {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "the answer"}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
    ]
    raw = "\n".join(json.dumps(e) for e in events)
    assert _extract_codex_reasoning(raw) == "**Plan** compare both"
    assert _CLI_SPECS["codex_cli"].reasoning_extractor is not None

    script = tmp_path / "fake_codex.py"
    script.write_text(
        "import json, sys\n"
        "sys.stdin.read()\n"
        "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
        "open(out, 'w', encoding='utf-8').write('the answer')\n"
        f"print({json.dumps(raw)})\n", encoding="utf-8")
    spec = _CliSpec(argv=(sys.executable, str(script)), pass_prompt_via="stdin", output_via="last_message_file",
                    usage_extractor=_CLI_SPECS["codex_cli"].usage_extractor,
                    reasoning_extractor=_CLI_SPECS["codex_cli"].reasoning_extractor)
    holder, token = tc.open_holder()
    try:
        assert await _run_cli(spec, "p", timeout_s=20.0, inactivity_timeout_s=10.0) == "the answer"
        assert holder["text"] == "**Plan** compare both"
    finally:
        tc.close_holder(token)


# ---- the chat panel's model wins, and the model that answered is said --------------------------------------------


def test_the_chat_panel_s_model_replaces_the_config_s_for_a_vscode_quest() -> None:
    import launch

    cfg = Config(topic="t", provider=ProviderConfig(name="vscode_extension", model="gpt-5.6-luna"))
    line = launch._apply_vscode_chat_model(cfg, "claude-opus-5")
    assert cfg.provider.model == "claude-opus-5"
    assert line == "the chat panel's model claude-opus-5 replaces gpt-5.6-luna from config.yaml"
    assert cfg.provider.extra[Engine.CHAT_MODEL_KEY] == {"before": "gpt-5.6-luna", "after": "claude-opus-5"}
    assert launch._apply_vscode_chat_model(cfg, "claude-opus-5") == "", "the same model: nothing to say"

    other = Config(topic="t", provider=ProviderConfig(name="claude_cli", model="opus"))
    assert launch._apply_vscode_chat_model(other, "claude-opus-5") == "" and other.provider.model == "opus"
    from core import plan_settings

    assert plan_settings.settings_of(cfg)["provider.model"] == "gpt-5.6-luna", \
        "the chat panel's model is this run's choice: the approved settings still compare the config's own model"
    older = Config(topic="t", provider=ProviderConfig(name="vscode_extension", model="gemini-3-flash"))
    assert launch._apply_vscode_chat_model(older, "gemini-3-flash-preview", "gemini-3-flash") == ""
    assert older.provider.model == "gemini-3-flash", "an older interview's family name for the same model is no change"
    plain = Config(topic="t", provider=ProviderConfig(name="vscode_extension", model="gpt-5.6-luna"))
    assert launch._apply_vscode_chat_model(plain, "") == "" and plain.provider.model == "gpt-5.6-luna", \
        "a terminal or web run (no chat panel) keeps the config's model"


def test_the_chat_panel_s_model_does_not_stop_a_start_or_resume_at_the_approved_settings(tmp_path: Path) -> None:
    """Resumed from the chat on another model, or started from the chat and resumed from a terminal: neither is a
    change to the approved settings (the approved-settings check stops a quest whose settings changed)."""
    import launch
    from core import plan_settings

    def cfg_from_chat(chat: str) -> Config:
        c = Config(topic="t", provider=ProviderConfig(name="vscode_extension", model="gpt-5.6-luna"))
        launch._apply_vscode_chat_model(c, chat)
        return c

    for first, then in ((Config(topic="t", provider=ProviderConfig(name="vscode_extension", model="gpt-5.6-luna")),
                         cfg_from_chat("claude-opus-5")),
                        (cfg_from_chat("claude-opus-5"),
                         Config(topic="t", provider=ProviderConfig(name="vscode_extension", model="gpt-5.6-luna")))):
        root = tmp_path / str(id(first))
        (root / ".fi").mkdir(parents=True)
        (root / "config.yaml").write_text(plan_settings.INTERVIEW_MARK + " the interview\ntopic: t\n", encoding="utf-8")
        assert plan_settings.check(root, root / ".fi", first) == []  # recorded on the first start
        assert plan_settings.check(root, root / ".fi", then) == []


def test_model_names_match_only_the_same_model() -> None:
    assert Engine.same_model("gpt-5.6-luna", "gpt-5.6-luna-2026-09")
    assert Engine.same_model("claude-opus-5", "claude-opus-5.20260901") is True
    assert Engine.same_model("gemini-3-flash", "gemini-3-flash-preview", "gemini-3-flash")
    assert not Engine.same_model("gpt-5", "gpt-5-mini") and not Engine.same_model("claude-opus-5", "claude-opus-5.5")
    assert not Engine.same_model("gpt-5", "gpt-5.5") and Engine.same_model("o4", "o4-preview")
    assert Engine.same_model("gemini-2.5-pro", "gemini-2.5-pro-preview-05-06")
    assert not Engine.same_model("gpt-5-mini", "gpt-5")


@pytest.mark.asyncio
async def test_a_fallback_or_the_same_model_again_is_no_change(tmp_path: Path) -> None:
    eng = _engine(tmp_path, provider="vscode_extension", model="claude-opus-5")
    ar.append_model_call(eng.fi_dir, eng.quest_id, ar.model_call_row(
        node="design", attempt=1, served={"provider": "vscode_extension", "model": "claude-opus-5", "reported": True},
        requested_model="claude-opus-5", reports_model=False, messages=MESSAGES, response="x"))
    ar.append_model_call(eng.fi_dir, eng.quest_id, ar.model_call_row(
        node="analyze", attempt=1, served={"provider": "openai", "model": "gpt-5", "reported": True, "fallback": True},
        requested_model="claude-opus-5", reports_model=False, messages=MESSAGES, response="x"))
    # Resumed from the chat on the model the quest already ran on, though config.yaml names another.
    eng.config.provider.extra = {Engine.CHAT_MODEL_KEY: {"before": "gpt-5.6-luna", "after": "claude-opus-5"}}
    eng._say_model_change()
    eng._note_served_model("design", {"model": "gpt-5", "reported": True, "fallback": True})
    eng._note_served_model("design", {"model": "claude-opus-5", "reported": True})
    assert [e for e in al.read(eng.audit.path) if e["kind"] == "model_changed"] == []
    log = _run_log(eng)
    assert "earlier results were made by" not in log and "was answered by claude-opus-5" in log


def test_a_new_quest_started_on_the_chat_panel_s_model_is_not_disclosed_as_two_models(tmp_path: Path) -> None:
    from core import plan_settings

    eng = _engine(tmp_path, provider="vscode_extension", model="claude-opus-5")
    eng.config.provider.extra = {Engine.CHAT_MODEL_KEY: {"before": "gpt-5.6-luna", "after": "claude-opus-5"}}
    eng._say_model_change()  # no earlier call: nothing was made on the config's model
    changes = [e for e in al.read(eng.audit.path) if e["kind"] == "model_changed"]
    assert len(changes) == 1 and changes[0]["before_any_step"] is True
    assert plan_settings.model_disclosure(changes) == ""


def test_config_s_change_and_the_chat_panel_s_other_model_are_both_in_the_trace(tmp_path: Path) -> None:
    from core import plan_settings

    # Earlier steps ran on A; config.yaml now says B; this run is resumed from the chat with C picked.
    eng = _engine(tmp_path, provider="vscode_extension", model="C")
    ar.append_model_call(eng.fi_dir, eng.quest_id, ar.model_call_row(
        node="design", attempt=1, served={"provider": "vscode_extension", "model": "A", "reported": True},
        requested_model="A", reports_model=False, messages=MESSAGES, response="x"))
    eng._audit("node_completed", duration_s=1.0)
    eng.config.provider.extra = {Engine.CHAT_MODEL_KEY: {"before": "B", "after": "C"}}
    eng._audit("model_changed", changes=[{"setting": "provider.model", "label": "the model", "from": "A", "to": "B"}],
               before_any_step=False)
    eng.__dict__["_model_change_taken"] = ("A", "B")
    eng._say_model_change()
    eng._note_served_model("design", {"model": "C", "reported": True})
    changes = [e for e in al.read(eng.audit.path) if e["kind"] == "model_changed"]
    assert [(c.get("before"), c.get("after")) for c in changes] == [(None, None), ("A", "C")],         "config.yaml's change and the chat panel's model C, once each"
    disclosed = plan_settings.model_disclosure(changes)
    assert "A for the steps before the change, C after it" in disclosed
    assert "the model from A to B" in al.describe(changes[0])


@pytest.mark.asyncio
async def test_config_s_own_model_change_confirmed_by_the_first_call_is_one_event(tmp_path: Path) -> None:
    eng = _engine(tmp_path, provider="openai", model="B")
    ar.append_model_call(eng.fi_dir, eng.quest_id, ar.model_call_row(
        node="design", attempt=1, served={"provider": "openai", "model": "A", "reported": True},
        requested_model="A", reports_model=False, messages=MESSAGES, response="x"))
    eng._audit("model_changed", changes=[{"setting": "provider.model", "label": "the model", "from": "A", "to": "B"}],
               before_any_step=False)
    eng.__dict__["_model_change_taken"] = ("A", "B")
    eng._say_model_change()
    eng._note_served_model("design", {"model": "B", "reported": True})
    assert len([e for e in al.read(eng.audit.path) if e["kind"] == "model_changed"]) == 1


def test_steps_done_on_auto_count_as_made_for_a_chat_panel_change(tmp_path: Path) -> None:
    eng = _engine(tmp_path, provider="vscode_extension", model="C")
    eng._audit("node_completed", duration_s=1.0)  # a step done on the picker's Auto: no model named
    eng.config.provider.extra = {Engine.CHAT_MODEL_KEY: {"before": "A", "after": "C"}}
    eng._say_model_change()
    (change,) = [e for e in al.read(eng.audit.path) if e["kind"] == "model_changed"]
    assert change["before_any_step"] is False


@pytest.mark.asyncio
async def test_run_log_names_the_model_that_served_and_a_change_is_in_the_trace(tmp_path: Path) -> None:
    server = _MockBridgeServer()

    def handler(msg: dict, w) -> list[dict]:  # noqa: ANN001
        if msg["type"] != "lm_request":
            return []
        return [{"type": "lm_done", "id": msg["id"], "content": "Answer",
                 "served_model": {"id": "claude-opus-5", "vendor": "copilot", "family": "claude-opus-5"}}]

    port = await server.start(handler)
    client = LLMClient(ResolvedEndpoint(base_url="", model="gpt-5.6-luna", api_key="not-needed",
                                        transport="vscode_bridge", vscode_bridge_port=port,
                                        vscode_model_override="gpt-5.6-luna"))
    eng = _engine(tmp_path, provider="vscode_extension", model="gpt-5.6-luna")
    # An earlier run of this quest was answered by another model.
    ar.append_model_call(eng.fi_dir, eng.quest_id, ar.model_call_row(
        node="design", attempt=1, served={"provider": "vscode_extension", "model": "gpt-5.6-luna", "reported": True},
        requested_model="gpt-5.6-luna", reports_model=False, messages=MESSAGES, response="x"))
    eng.config.provider.extra = {Engine.CHAT_MODEL_KEY: {"before": "gpt-5.6-luna", "after": "claude-opus-5"}}
    eng._say_model_change()
    eng._client = client  # type: ignore[assignment]
    try:
        await eng._chat("prompt", node="design")
        await eng._chat("prompt", node="analyze")
    finally:
        await client.aclose()
        await server.stop()
    log = _run_log(eng)
    assert "the chat panel's model claude-opus-5 replaces gpt-5.6-luna from config.yaml" in log
    assert log.count("was answered by claude-opus-5 (vendor: copilot)") == 1, "said once, on the first call"
    assert "the config names gpt-5.6-luna; VS Code served claude-opus-5" in log
    assert "earlier results were made by gpt-5.6-luna and its results from here on by claude-opus-5" in log
    changes = [e for e in al.read(eng.audit.path) if e["kind"] == "model_changed"]
    assert [(c["before"], c["after"]) for c in changes] == [("gpt-5.6-luna", "claude-opus-5")], \
        "one change, recorded once (the chat panel's), not again when the first call confirms it"
    assert "the model changed from gpt-5.6-luna to claude-opus-5" in al.describe(changes[0])
    cost = [json.loads(line) for line in (eng.fi_dir / "cost.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row.get("model") for row in cost} == {"claude-opus-5"}, "cost.jsonl names the model that served each call"


# ---- the extension (compiled TypeScript, run under node) ----------------------------------------------------------


_NODE_ASK = """
const m = require(%s);
const calls = [];
function fakeModel(rejectOption) {
  return { id: "claude-opus-5", sendRequest: async (msgs, options) => {
    calls.push(options);
    if (rejectOption && options && options.modelOptions) throw new Error("400 unknown model option _enableThinking");
    return { stream: [] };
  }};
}
(async () => {
  const out = {};
  const ok = new m.ThinkingRequests();
  const a = await ok.send("claude-opus-5", true, (o) => fakeModel(false).sendRequest([], o));
  out.asked = [a.asked, JSON.stringify(calls[0])];
  calls.length = 0;
  const off = await ok.send("claude-opus-5", false, (o) => fakeModel(false).sendRequest([], o));
  out.off = [off.asked, JSON.stringify(calls[0])];
  calls.length = 0;
  const r = new m.ThinkingRequests();
  const first = await r.send("claude-opus-5", true, (o) => fakeModel(true).sendRequest([], o));
  out.rejected = [first.asked, calls.map((c) => JSON.stringify(c)), r.declinedReason("claude-opus-5")];
  calls.length = 0;
  const again = await r.send("claude-opus-5", true, (o) => fakeModel(true).sendRequest([], o));
  out.notAskedAgain = [again.asked, calls.length];
  const t = new m.ThinkingRequests();
  try { await t.send("x", true, async () => { throw new Error("net::ERR_HTTP2_PROTOCOL_ERROR"); }); out.transient = "no throw"; }
  catch (e) { out.transient = [e.message, t.declinedReason("x") === undefined]; }
  const q = new m.ThinkingRequests();  // a quota error: the retry without the option fails too, so it is no refusal
  try { await q.send("x", true, async () => { throw new Error("You have exceeded your premium request quota"); }); out.quota = "no throw"; }
  catch (e) { out.quota = [e.message, q.declinedReason("x") === undefined]; }
  const c = new m.ThinkingRequests();  // an access error (consent) is never read as a refusal, nor sent twice
  let consentCalls = 0;
  try { await c.send("x", true, async () => { consentCalls++; const e = new Error("no consent"); e.code = "NoPermissions"; throw e; }); }
  catch (e) { out.consent = [consentCalls, c.declinedReason("x") === undefined]; }
  const s = new m.ThinkingRequests();
  const marker = s.streamFailed("y", true, 0, "400 bad request");
  const retried = await s.send("y", true, async (o) => { out.retriedWith = JSON.stringify(o); return "ok"; });
  out.beforeFirstPart = s.declinedReason("y") === undefined;  // accepted is not answered: no refusal yet
  s.firstPart("y", retried.asked);
  const s2 = new m.ThinkingRequests();  // failed after asking, then the stream without the option failed the same way
  s2.streamFailed("v", true, 0, "500 internal");
  const again2 = await s2.send("v", true, async () => "accepted");
  s2.streamFailed("v", again2.asked, 0, "500 internal");
  const next2 = await s2.send("v", true, async (o) => JSON.stringify(o));
  out.askedAgain = next2.asked;
  out.stream = [marker, retried.asked, s.declinedReason("y") !== undefined, s2.declinedReason("v") === undefined,
                s.streamFailed("z", true, 0, "bridge stalled: no part"), s.streamFailed("w", true, 3, "400")];
  out.marker = m.THINKING_DECLINED_MARKER;
  out.text = [m.thinkingText(["a", "b"]), m.thinkingText("c"), m.thinkingText([1]) === undefined];
  class LanguageModelThinkingPart { constructor(v) { this.value = v; } }
  out.kind = [m.partKind({}, new LanguageModelThinkingPart(["a", "b"])), m.partKind({}, { value: ["a"] })];
  // A GPT reasoning part as Copilot reports it with includeEncryptedThinking: the summary is the value, the encrypted
  // state sits in the metadata and is never read.
  const enc = new LanguageModelThinkingPart(["Step one.", "Step two."]);
  enc.metadata = { encrypted_content: "SECRET" };
  const empty = new LanguageModelThinkingPart("");
  empty.metadata = { encrypted_content: "SECRET" };
  out.encrypted = [m.partKind({}, enc), m.thinkingText(enc.value), m.thinkingText(empty.value)];
  out.chat = [m.chatModelArgs({ id: "claude-opus-5", vendor: "copilot" }), m.chatModelArgs({ id: "auto" }), m.chatModelArgs(undefined),
              m.chatModelArgs({ id: "gemini-3-flash-preview", family: "gemini-3-flash" })];
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { process.stderr.write(String(e && e.stack || e)); process.exit(1); });
"""


def _compile_lm_messages(tmp_path: Path) -> Path:
    ext = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
    node = shutil.which("node")
    tsc = ext / "node_modules" / "typescript" / "bin" / "tsc"
    if node is None or not tsc.exists():
        pytest.skip("node or the extension's typescript (npm install in vscode-frontier-insight) is missing")
    outdir = tmp_path / "compiled"
    run = subprocess.run(
        [node, str(tsc), str(ext / "src" / "lm-messages.ts"), "--outDir", str(outdir), "--module", "commonjs",
         "--target", "es2020", "--skipLibCheck", "--types", "node", "--typeRoots", str(ext / "node_modules" / "@types")],
        capture_output=True, text=True, timeout=120, cwd=str(ext))
    assert run.returncode == 0, run.stdout + run.stderr
    return outdir / "lm-messages.js"


def test_the_extension_asks_for_thinking_and_asks_again_without_it_on_a_refusal(tmp_path: Path) -> None:
    compiled = _compile_lm_messages(tmp_path)
    driver = tmp_path / "driver.js"
    driver.write_text(_NODE_ASK % json.dumps(str(compiled).replace("\\", "/")), encoding="utf-8")
    run = subprocess.run([shutil.which("node"), str(driver)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["asked"] == [True, json.dumps({"modelOptions": {"_enableThinking": True}, "includeEncryptedThinking": True}, separators=(",", ":"))]
    assert out["off"] == [False, "{}"]
    asked, sent, reason = out["rejected"]
    assert asked is False and sent == ['{"modelOptions":{"_enableThinking":true},"includeEncryptedThinking":true}', "{}"] and "unknown model option" in reason
    assert out["notAskedAgain"] == [False, 1], "a model that refused is not asked again"
    assert out["transient"] == ["net::ERR_HTTP2_PROTOCOL_ERROR", True], "a connection error is not a refusal"
    assert out["quota"] == ["You have exceeded your premium request quota", True], "a failure the retry repeats is no refusal"
    assert out["consent"] == [1, True], "an access error is raised at once and not taken for a refusal"
    failed, retried_asked, remembered, not_remembered, stalled, late = out["stream"]
    assert failed.startswith(out["marker"]) and retried_asked is False and out["retriedWith"] == "{}"
    assert out["beforeFirstPart"], "a request accepted but not yet answered is no refusal"
    assert remembered, "the call made again without the option answered: the model refused it"
    assert not_remembered and out["askedAgain"] is True, "the call without the option failed the same way: not a refusal"
    assert stalled == "bridge stalled: no part" and late == "400"
    assert out["text"] == ["a\n\nb", "c", True]
    assert out["encrypted"] == ["thinking", "Step one.\n\nStep two.", ""], "the summary only, never the encrypted state"
    assert out["kind"] == ["thinking", "unknown"]
    assert out["chat"] == [["--vscode-chat-model", "claude-opus-5"], [], [],
                           ["--vscode-chat-model", "gemini-3-flash-preview", "--vscode-chat-model-family", "gemini-3-flash"]]


def test_both_bridges_send_through_the_thinking_request_and_the_chat_passes_its_model() -> None:
    import re

    src = Path(__file__).resolve().parent.parent / "vscode-frontier-insight" / "src"
    for name in ("bridge.ts", "persistent-bridge.ts"):
        text = (src / name).read_text(encoding="utf-8")
        assert "thinkingRequests.send(" in text and "thinkingRequests.streamFailed(" in text, name
        assert "ask_thinking" in text and "thinkingText(" in text, name
        assert not re.search(r"sendRequest\([^)]*,\s*\{\}\s*,", text), f"{name} still sends a bare request"
        # A GPT reasoning part carries Copilot's encrypted state in its metadata: the bridges keep the summary only.
        assert ".metadata" not in text, f"{name} must not read a part's metadata (encrypted reasoning state)"
    ext = (src / "extension.ts").read_text(encoding="utf-8")
    # /start, /fleet and /resume pass the chat panel's model; /update and /generate keep the config's.
    assert ext.count("...chatModelArgs(userPickedModel)") == 1
    assert "const args: string[] = [...chatModelArgs(userPickedModel)];" in ext


def test_a_fallback_provider_that_answered_is_not_called_the_vs_code_route(tmp_path: Path) -> None:
    """With provider vscode_extension, an answer from a fallback provider (it names itself, e.g. ollama) is not said
    to have come through VS Code; one through the bridge (it names Copilot's vendor) is."""
    eng = _engine(tmp_path, provider="vscode_extension")
    eng._say_once_about_thinking("design", {"text": ""}, {"provider": "ollama", "model": "g"}, "ok", "")
    assert "through VS Code" not in _run_log(eng)
    eng._say_once_about_thinking("design", {"text": ""}, {"provider": "copilot", "model": "m"}, "ok", "")
    assert _run_log(eng).count("returned no reasoning through VS Code") == 1
