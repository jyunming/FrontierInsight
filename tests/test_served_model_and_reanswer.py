"""Two faults from the first real self-benchmark recording (a Haiku quest on the claude CLI).

1. Its ``ideate`` call was recorded as answered by ``claude-fable-5-1``. The CLI really did switch: Claude Code answers
   with another model when the asked-for one declines a request (its ``model_refusal_fallback``), and the reported cost
   ($0.68 for 7,170 output tokens) is about twelve times Haiku's price. FI now reads the model of the message that
   carries the answer (not the ``modelUsage`` entry that wrote the most), says every switch in run.log, and asks the
   asked-for model once more with the CLI's switching turned off.
2. Its ``design_self_critique`` reply (67,963 characters) held two JSON objects, and the audit was called unreadable.
   The parser now finds the asked-for object among several, and a reply that still cannot be used is asked for once
   more, short; a second unusable answer still stops a research quest, as before.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, _parse_json_lenient
from core.provider import (
    CALL_ATTEMPTS, CLAUDE_NO_REFUSAL_FALLBACK_ENV, LAST_CALL, LLMClient, _CLI_SPECS, _CliSpec, _extract_claude_usage,
    claude_model_family_differs, resolve_endpoint,
)

HAIKU = "claude-haiku-4-5-20251001"
FABLE = "claude-fable-5-1"


def _evt(**kw: object) -> str:
    return json.dumps(kw)


def _turn(model: str, text: str, stop: str) -> list[str]:
    return [
        _evt(type="stream_event", event={"type": "message_start", "message": {"model": model, "role": "assistant"}}),
        _evt(type="stream_event", event={"type": "content_block_delta", "index": 0,
                                         "delta": {"type": "text_delta", "text": text}}),
        _evt(type="stream_event", event={"type": "message_delta", "delta": {"stop_reason": stop}}),
    ]


def _result(text: str, model_usage: dict, cost: float) -> str:
    return _evt(type="result", subtype="success", is_error=False, result=text, total_cost_usd=cost,
                usage={"input_tokens": 10, "cache_creation_input_tokens": 14500, "cache_read_input_tokens": 8688,
                       "output_tokens": 7170},
                modelUsage=model_usage)


#: The recorded incident: Haiku starts, its turn is refused, the CLI switches to Fable, Fable answers.
SWITCHED_STREAM = [
    _evt(type="system", subtype="init", model=HAIKU),
    *_turn(HAIKU, "Here are three ideas", "refusal"),
    _evt(type="system", subtype="model_refusal_fallback", original_model=HAIKU, fallback_model=FABLE,
         trigger="api_refusal", content="Haiku declined; answered by Fable"),
    *_turn(FABLE, '{"ideas": ["fable"]}', "end_turn"),
    _evt(type="assistant", message={"model": FABLE, "content": [{"type": "text", "text": '{"ideas": ["fable"]}'}]}),
    _result('{"ideas": ["fable"]}', {HAIKU: {"outputTokens": 60}, FABLE: {"outputTokens": 7110}}, 0.6843),
]

#: Haiku answered; a side model wrote MORE than it (the shape a "most output" reading gets wrong).
HELPER_STREAM = [
    _evt(type="system", subtype="init", model=HAIKU),
    *_turn(HAIKU, "ok", "end_turn"),
    _evt(type="assistant", message={"model": HAIKU, "content": [{"type": "text", "text": "ok"}]}),
    _result("ok", {HAIKU: {"outputTokens": 41}, FABLE: {"outputTokens": 900}}, 0.02),
]

HAIKU_STREAM = [
    _evt(type="system", subtype="init", model=HAIKU),
    *_turn(HAIKU, '{"ideas": ["haiku"]}', "end_turn"),
    _evt(type="assistant", message={"model": HAIKU, "content": [{"type": "text", "text": '{"ideas": ["haiku"]}'}]}),
    _result('{"ideas": ["haiku"]}', {HAIKU: {"outputTokens": 900}}, 0.05),
]

REFUSED_STREAM = [
    _evt(type="system", subtype="init", model=HAIKU),
    *_turn(HAIKU, "I can't help with that", "refusal"),
    _evt(type="system", subtype="model_refusal_no_fallback", original_model=HAIKU, content="declined"),
    _evt(type="assistant", message={"model": "<synthetic>", "content": [{"type": "text", "text": "declined"}]}),
    _result("I can't help with that", {HAIKU: {"outputTokens": 20}}, 0.01),
]


# ---- reading who answered -----------------------------------------------------------------------------------------


def test_the_answer_s_own_model_is_read_not_a_side_model_that_wrote_more() -> None:
    measured = _extract_claude_usage("\n".join(HELPER_STREAM))
    assert measured["served_model"] == HAIKU
    assert "switched" not in measured and not measured.get("refused")


def test_the_cli_s_switch_to_another_model_is_read_with_both_models() -> None:
    measured = _extract_claude_usage("\n".join(SWITCHED_STREAM))
    assert measured["served_model"] == FABLE, "the CLI really answered with Fable: recorded truthfully"
    assert measured["switched"] == {"from": HAIKU, "to": FABLE, "kind": "model_refusal_fallback", "why": "api_refusal"}
    assert measured["cost_usd_reported"] == pytest.approx(0.6843)
    assert not measured.get("refused"), "the turn that ended the call was Fable's answer, not the refusal"


def test_a_refusal_with_no_other_model_is_read_as_a_refusal() -> None:
    measured = _extract_claude_usage("\n".join(REFUSED_STREAM))
    assert measured["refused"] is True
    assert measured["served_model"] == HAIKU, "the CLI's own <synthetic> message is not a model"


def test_family_names_compare_aliases_with_full_ids() -> None:
    assert not claude_model_family_differs("haiku", HAIKU)
    assert not claude_model_family_differs("opus[1m]", "claude-opus-4-7")
    assert claude_model_family_differs("haiku", FABLE)
    assert not claude_model_family_differs("opusplan", FABLE), "an alias it cannot read is never called a switch"


def test_a_refused_turn_before_the_switch_does_not_make_the_other_model_s_answer_a_refusal() -> None:
    """The other model's turn may come without a turn-end line: the refusal before the switch must not count."""
    stream = [line for line in SWITCHED_STREAM if '"end_turn"' not in line]
    measured = _extract_claude_usage("\n".join(stream))
    assert measured["served_model"] == FABLE and not measured.get("refused")


def test_a_stream_line_that_is_not_an_object_is_ignored() -> None:
    from core.provider import _claude_stream_fact_line

    assert _claude_stream_fact_line(b'["assistant", "model"]') is None
    assert _claude_stream_fact_line(b'["message_delta", "stop_reason"]') is None


# ---- what the claude CLI client does about a switch ----------------------------------------------------------------


def _fake_claude(tmp_path: Path, pinned_mode: str) -> _CliSpec:
    """A stand-in claude CLI: without the no-switch setting it prints the recorded switch to Fable; with it, Haiku
    either answers (``answer``) or declines again (``refuse``). Each run appends the setting it saw to calls.txt."""
    script = tmp_path / "fake_claude.py"
    script.write_text(
        "import os, sys\n"
        "sys.stdin.read()\n"
        f"pinned = os.environ.get({CLAUDE_NO_REFUSAL_FALLBACK_ENV!r}) == '1'\n"
        f"open({str(tmp_path / 'calls.txt')!r}, 'a', encoding='utf-8').write(('pinned' if pinned else 'free') + chr(10))\n"
        f"lines = {json.dumps(SWITCHED_STREAM)!r} if not pinned else "
        f"({json.dumps(HAIKU_STREAM)!r} if {pinned_mode!r} == 'answer' else {json.dumps(REFUSED_STREAM)!r})\n"
        "import json as _j\n"
        "for line in _j.loads(lines):\n"
        "    print(line, flush=True)\n",
        encoding="utf-8",
    )
    real = _CLI_SPECS["claude_cli"]
    return dataclasses.replace(real, argv=(sys.executable, str(script)), model_flag=None)


def _client(tmp_path: Path, pinned_mode: str, run_log: logging.Logger) -> LLMClient:
    endpoint = resolve_endpoint(ProviderConfig(name="claude_cli", model="haiku"))
    endpoint = dataclasses.replace(endpoint, cli_spec=_fake_claude(tmp_path, pinned_mode))
    return LLMClient(endpoint, run_log=run_log)


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def _run_log() -> tuple[logging.Logger, _Lines]:
    logger = logging.Logger("test.run_log", logging.INFO)
    lines = _Lines()
    logger.addHandler(lines)
    return logger, lines


@pytest.mark.asyncio
async def test_a_switched_answer_is_asked_again_on_the_chosen_model_and_its_answer_kept(tmp_path: Path) -> None:
    run_log, lines = _run_log()
    client = _client(tmp_path, "answer", run_log)
    attempts: list[dict] = []
    token = CALL_ATTEMPTS.set(attempts)
    try:
        text = await client._chat_cli([{"role": "user", "content": "ideas please"}], node="ideate")
    finally:
        CALL_ATTEMPTS.reset(token)
    assert text == '{"ideas": ["haiku"]}', "the chosen model's answer is the one kept"
    assert (tmp_path / "calls.txt").read_text(encoding="utf-8").split() == ["free", "pinned"]
    assert LAST_CALL.get()["model"] == HAIKU and "switched_from" not in LAST_CALL.get()
    # The Fable answer that was not used was paid for: recorded as an attempt with its token counts.
    (paid,) = attempts
    assert paid["model"] == FABLE and paid["error"] == "answered_by_other_model"
    assert paid["usage"]["completion_tokens"] == 7170 and paid["usage"]["cost_usd_reported"] == pytest.approx(0.6843)
    said = "\n".join(lines.lines)
    assert f"[model] ideate: haiku declined this request, so the claude tool answered with {FABLE} instead" in said
    assert "haiku answered when asked again; its answer is kept" in said


@pytest.mark.asyncio
async def test_when_the_chosen_model_declines_again_the_other_answer_is_kept_and_said_each_time(tmp_path: Path) -> None:
    run_log, lines = _run_log()
    client = _client(tmp_path, "refuse", run_log)
    attempts: list[dict] = []
    token = CALL_ATTEMPTS.set(attempts)
    try:
        first = await client._chat_cli([{"role": "user", "content": "ideas please"}], node="ideate")
        second = await client._chat_cli([{"role": "user", "content": "more ideas"}], node="ideate_reflect")
    finally:
        CALL_ATTEMPTS.reset(token)
    assert first == second == '{"ideas": ["fable"]}', "a refusal is never handed on as an answer"
    assert LAST_CALL.get()["model"] == FABLE and LAST_CALL.get()["switched_from"] == HAIKU
    assert [a["error"] for a in attempts] == ["refused", "refused"], "each declined second try is recorded"
    said = "\n".join(lines.lines)
    for node in ("ideate", "ideate_reflect"):  # every time, not only on the first call
        assert f"[model] {node}: haiku could not answer (it declined again); the {FABLE} answer is kept" in said
        assert f"this call cost {FABLE}'s price, not haiku's" in said


@pytest.mark.asyncio
async def test_an_answer_from_the_chosen_model_is_not_asked_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    async def fake_run_cli(spec, prompt, *, usage_out=None, **kw):  # noqa: ANN001
        calls.append(kw)
        usage_out.update(_extract_claude_usage("\n".join(HAIKU_STREAM)))
        return "the answer"

    monkeypatch.setattr("core.provider._run_cli", fake_run_cli)
    client = LLMClient(resolve_endpoint(ProviderConfig(name="claude_cli", model="haiku")))
    assert await client._chat_cli([{"role": "user", "content": "q"}], node="clarify") == "the answer"
    assert len(calls) == 1 and not calls[0].get("extra_env")
    assert LAST_CALL.get()["model"] == HAIKU


def test_the_cli_s_short_name_and_its_full_id_are_not_called_two_models(tmp_path: Path) -> None:
    """The recorded run.log warned "the config names haiku; the connection served claude-haiku-4-5-20251001"."""
    for model, served, warned in (("haiku", HAIKU, False), ("opus[1m]", "claude-opus-4-7", False), ("haiku", FABLE, True)):
        cfg = Config(
            topic="t", title="t", provider=ProviderConfig(name="claude_cli", model=model),
            engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
            execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
            output=OutputConfig(output_dir=tmp_path / model.replace("[", "_").replace("]", "_") / served),
        )
        eng = Engine(cfg)
        eng.fi_dir.mkdir(parents=True, exist_ok=True)
        eng._note_served_model("clarify", {"model": served, "reported": True})
        for h in eng._log.handlers:
            h.flush()
        log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
        assert (f"the config names {model}; the connection served {served}" in log) is warned, (model, served)


@pytest.mark.asyncio
async def test_a_switch_for_another_reason_is_said_and_not_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    unavailable = [line.replace("model_refusal_fallback", "model_fallback") for line in SWITCHED_STREAM]
    calls: list[dict] = []

    async def fake_run_cli(spec, prompt, *, usage_out=None, **kw):  # noqa: ANN001
        calls.append(kw)
        usage_out.update(_extract_claude_usage("\n".join(unavailable)))
        return "fable's answer"

    monkeypatch.setattr("core.provider._run_cli", fake_run_cli)
    run_log, lines = _run_log()
    client = LLMClient(resolve_endpoint(ProviderConfig(name="claude_cli", model="haiku")), run_log=run_log)
    assert await client._chat_cli([{"role": "user", "content": "q"}], node="plan") == "fable's answer"
    assert len(calls) == 1, "switching for unavailability cannot be turned off: asking again would only switch again"
    assert LAST_CALL.get()["switched_from"] == HAIKU
    assert f"[model] plan: haiku was not available, so the claude tool answered with {FABLE} instead" in lines.lines[0]


@pytest.mark.asyncio
async def test_a_plain_refusal_stops_for_a_person_with_its_cost_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.provider import ModelAnswerFiltered

    calls: list[dict] = []

    async def fake_run_cli(spec, prompt, *, usage_out=None, **kw):  # noqa: ANN001
        calls.append(kw)
        usage_out.update(_extract_claude_usage("\n".join(REFUSED_STREAM)))
        return "I can't help with that"

    monkeypatch.setattr("core.provider._run_cli", fake_run_cli)
    client = LLMClient(resolve_endpoint(ProviderConfig(name="claude_cli", model="haiku")))
    with pytest.raises(ModelAnswerFiltered) as caught:
        await client._chat_cli([{"role": "user", "content": "q"}], node="ideate")
    assert len(calls) == 1, "the same model declines again: not retried"
    assert caught.value.finish_reason == "refusal" and caught.value.usage["completion_tokens"] == 7170
    assert "switched" not in caught.value.usage and "refused" not in caught.value.usage


# ---- an unreadable answer to a required check ---------------------------------------------------------------------

DRAFT = {
    "hypothesis": "Verlet conserves energy better than Euler",
    "variables": {"independent": ["method"], "dependent": ["energy drift"], "controls": ["h"]},
    "method": "integrate a damped oscillator",
    "expected_outcome": "Verlet drifts less",
    "figures_planned": ["drift.png"],
    "dependencies": ["numpy"],
}
OBJECTION = {"check": "numerical_convergence", "objection": "no step halving", "fix": "halve h and compare"}
GOOD = json.dumps({"objections_addressed": [OBJECTION], "amended_design": {**DRAFT, "method": "halve h"}})
#: The recorded shape: the amended design printed alone, then the asked-for object, in two fences.
TWO_OBJECTS = f"```json\n{json.dumps(DRAFT, indent=2)}\n```\n\n## Objections Addressed\n\n```json\n{GOOD}\n```"
#: Long and unstructured: no object at all (the kind of answer the retry is for).
RAMBLING = "Let me think about this design carefully. " * 1600


def _engine(tmp_path: Path, replies: list[str], *, research: bool = False) -> tuple[Engine, AsyncMock]:
    cfg = Config(
        topic="integrators", title="integrators", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    if research:
        eng.config = eng.config.model_copy(update={"rigor_profile": "research"})
    chat = AsyncMock(side_effect=replies)
    eng._client = type("Stub", (), {"chat": chat})()
    return eng, chat


def test_the_asked_for_object_is_found_among_two() -> None:
    assert _parse_json_lenient(TWO_OBJECTS) is None, "the plain parse still gives up on two objects"
    parsed = _parse_json_lenient(TWO_OBJECTS, want=("objections_addressed",))
    assert parsed["objections_addressed"] == [OBJECTION]


def test_an_object_inside_a_broken_one_is_never_taken_for_the_answer() -> None:
    broken = '{"verdict": "sufficient", "per_source": [{"id": 1, "verdict": "insufficient"}], "rationale": "r",}'
    assert _parse_json_lenient(broken, want=("verdict",)) is None
    cut = '{"objections_addressed": [{"check": "oracle", "objection": "x", "fix": "y"}], "amended_design": {"a": '
    assert _parse_json_lenient(cut + '{"objections_addressed": []}', want=("objections_addressed",)) is None
    prose = 'Note: { opens a set.\n```json\n{"verdict": "broaden", "rationale": "r"}\n```'
    assert _parse_json_lenient(prose, want=("verdict",))["verdict"] == "broaden", "a brace in prose swallows nothing"


@pytest.mark.asyncio
async def test_the_recorded_two_object_reply_is_read_without_asking_again(tmp_path: Path) -> None:
    eng, chat = _engine(tmp_path, [TWO_OBJECTS])
    design, objections = await eng._audit_design({"topic": "t", "iteration": 0}, dict(DRAFT))
    assert objections == [OBJECTION] and design["method"] == "halve h"
    assert chat.await_count == 1


@pytest.mark.asyncio
async def test_a_long_unstructured_reply_is_asked_for_once_more_short(tmp_path: Path) -> None:
    eng, chat = _engine(tmp_path, [RAMBLING, GOOD], research=True)
    design, objections = await eng._audit_design({"topic": "t", "iteration": 0}, dict(DRAFT))
    assert objections == [OBJECTION] and design["method"] == "halve h"
    assert chat.await_count == 2
    retry = chat.await_args_list[1].args[0][0]["content"]
    assert "Your previous reply to the request above could not be read" in retry
    assert "no JSON object in it" in retry and "objections_addressed" in retry
    assert RAMBLING not in retry, "a long earlier reply is not sent back"
    assert "## The draft design (JSON)" in retry, "the request itself is asked again"
    receipt = json.loads((eng.quest_root / "needs" / "receipts" / "design_audit.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "pass"
    assert not (eng.fi_dir / "pause.json").exists()


@pytest.mark.asyncio
async def test_two_unusable_answers_still_stop_a_research_quest(tmp_path: Path) -> None:
    eng, chat = _engine(tmp_path, [RAMBLING, "still not JSON, sorry"], research=True)
    stops: list[dict] = []

    def pause(**kwargs):  # noqa: ANN003
        stops.append(kwargs)
        raise RuntimeError("stopped")

    eng._pause_for_human = pause  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="stopped"):
        await eng._audit_design({"topic": "t", "iteration": 0}, dict(DRAFT))
    assert chat.await_count == 2, "asked once more, never twice"
    assert stops[0]["kind"] == "design_audit_unknown"
    receipt = json.loads((eng.quest_root / "needs" / "receipts" / "design_audit.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "unknown"


@pytest.mark.asyncio
async def test_the_evidence_gate_and_the_claim_check_ask_once_more_too(tmp_path: Path) -> None:
    eng, chat = _engine(tmp_path, ["no verdict here", json.dumps({"verdict": "sufficient", "rationale": "r"})])
    text, parsed = await eng._chat_json("judge", node="evidence_gate", want=("verdict",),
                                        usable=lambda p: p.get("verdict") in ("sufficient", "broaden", "insufficient"))
    assert parsed["verdict"] == "sufficient" and chat.await_count == 2
    short = chat.await_args_list[1].args[0][0]["content"]
    assert "no verdict here" in short, "a short earlier reply is shown back so the same judgement can be restated"

    eng, chat = _engine(tmp_path / "c", [json.dumps({"summary": "x"}), json.dumps({"claims": []})])
    _, parsed = await eng._chat_json("ground", node="claim_check", want=("claims",),
                                     usable=lambda p: isinstance(p.get("claims"), list))
    assert parsed == {"claims": []} and chat.await_count == 2


def test_the_audit_prompt_asks_for_one_object_printed_once() -> None:
    text = (Path(__file__).resolve().parent.parent / "agents" / "design_self_critique.md").read_text(encoding="utf-8")
    assert "Print it once: do not also print the draft or the amended design on its own" in text
