"""An answer the model did not finish is never handed on as whole (the 2026-09-28 R2 re-audit, P1-02).

A reply that ended with ``finish_reason: length`` (cut off at the output limit) used to be returned as a normal answer,
so a partial script or JSON could flow into later steps; ``content_filter`` was not told apart either. Now: a step's
``provider.node_max_tokens`` is sent; a call cut off at a limit it set is asked once more with twice it (capped), and
an answer still cut off raises :class:`ModelAnswerTruncated`; a filtered one raises :class:`ModelAnswerFiltered` (not
retried). Either stops the quest for a person (a ``model_output`` pause naming the step and the setting to change);
resume runs the step again. The reason and the cost of every attempt are kept. No real API is called here.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from core import attempt_records as ar
from core.config import ProviderConfig
from core.provider import (
    CALL_ATTEMPTS,
    LAST_CALL,
    LLMClient,
    ModelAnswerFiltered,
    ModelAnswerTruncated,
    _raise_if_cut_off,
    outcome_of,
    resolve_endpoint,
)

KIMI = dict(name="openai", base_url="https://api.moonshot.ai/v1", api_key_env="MOONSHOT_API_KEY", model="kimi-k3")
PLAIN = dict(name="openai", base_url="https://api.example.com/v1", api_key_env="OPENAI_API_KEY", model="gpt-x")


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    import tenacity

    monkeypatch.setattr("core.provider.wait_random_exponential", lambda **kw: tenacity.wait_none())
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")


def _sse(text: str, finish: str) -> bytes:
    events = [{"choices": [{"delta": {"content": text}, "finish_reason": None}], "model": "kimi-k3"},
              {"choices": [{"delta": {}, "finish_reason": finish}]},
              {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}]
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n").encode("utf-8")


def _plain(text: str, finish: str) -> dict:
    return {"model": "gpt-x-0928", "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}


async def _chat(cfg: dict, handler, **kwargs):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
    try:
        client = LLMClient(resolve_endpoint(ProviderConfig(**cfg)), http=http)
        attempts: list = []
        token = CALL_ATTEMPTS.set(attempts)
        try:
            text = await client.chat([{"role": "user", "content": "hi"}], **kwargs)
            return text, dict(LAST_CALL.get() or {}), attempts
        finally:
            CALL_ATTEMPTS.reset(token)
    finally:
        await http.aclose()


def _streamed(finishes: list[str], seen: list[dict]):
    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_sse("partial {", finishes[min(len(seen), len(finishes)) - 1]))
    return handler


def _whole(finishes: list[str], seen: list[dict]):
    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_plain("partial {", finishes[min(len(seen), len(finishes)) - 1]))
    return handler


@pytest.mark.parametrize("cfg,make", [(KIMI, _streamed), (PLAIN, _whole)], ids=["streamed kimi", "plain http"])
def test_an_answer_cut_off_with_no_limit_set_is_never_returned(cfg, make) -> None:
    seen: list[dict] = []
    with pytest.raises(ModelAnswerTruncated, match="cut off at its output limit"):
        asyncio.run(_chat(cfg, make(["length"], seen)))
    assert len(seen) == 1, "no limit was set, so there is nothing larger to ask for: not retried"


@pytest.mark.parametrize("cfg,make", [(KIMI, _streamed), (PLAIN, _whole)], ids=["streamed kimi", "plain http"])
def test_a_limit_the_call_set_is_raised_once_and_then_the_answer_is_used(cfg, make) -> None:
    seen: list[dict] = []
    text, last, attempts = asyncio.run(_chat(cfg, make(["length", "stop"], seen), max_tokens=1000))
    assert text == "partial {" and [b["max_tokens"] for b in seen] == [1000, 2000]
    assert last["finish_reason"] == "stop"
    assert [a["error"] for a in attempts] == ["truncated"], "the cut-off attempt is its own line in the record"


def test_a_second_cut_off_is_not_retried_again() -> None:
    seen: list[dict] = []
    with pytest.raises(ModelAnswerTruncated, match="2000 tokens"):
        asyncio.run(_chat(PLAIN, _whole(["length", "length", "stop"], seen), max_tokens=1000))
    assert [b["max_tokens"] for b in seen] == [1000, 2000], "bounded: one larger ask, never a third"


def test_the_larger_ask_is_capped() -> None:
    seen: list[dict] = []
    with pytest.raises(ModelAnswerTruncated):
        asyncio.run(_chat(PLAIN, _whole(["length"], seen), max_tokens=65536))
    assert len(seen) == 1, "already at the cap: nothing larger to ask for"


@pytest.mark.parametrize("cfg,make", [(KIMI, _streamed), (PLAIN, _whole)], ids=["streamed kimi", "plain http"])
def test_a_filtered_answer_stops_and_is_not_retried(cfg, make) -> None:
    seen: list[dict] = []
    with pytest.raises(ModelAnswerFiltered, match="content filter"):
        asyncio.run(_chat(cfg, make(["content_filter"], seen), max_tokens=1000))
    assert len(seen) == 1


def test_a_normal_answer_carries_its_reason_and_an_unknown_one_is_taken_as_finished(caplog) -> None:
    import logging

    text, last, _a = asyncio.run(_chat(PLAIN, _whole(["stop"], [])))
    assert text == "partial {" and last["finish_reason"] == "stop"
    with caplog.at_level(logging.INFO, logger="frontier_insight.provider"):
        text, last, _a = asyncio.run(_chat(PLAIN, _whole(["something_new"], [])))
    assert text == "partial {" and last["finish_reason"] == "something_new"
    assert any("something_new" in r.getMessage() for r in caplog.records)


def test_the_record_of_calls_names_the_outcome_and_the_reason() -> None:
    assert outcome_of(ModelAnswerTruncated("x")) == "truncated"
    assert outcome_of(ModelAnswerFiltered("x")) == "content_filtered"
    assert outcome_of(ValueError("x")) == "ValueError"
    row = ar.model_call_row(node="implement", attempt=1, served={"provider": "openai", "model": "m",
                                                                "finish_reason": "length"},
                            requested_model="m", reports_model=True, messages=[], response=None, outcome="truncated")
    assert row["outcome"] == "truncated" and row["finish_reason"] == "length"
    plain = ar.model_call_row(node="write", attempt=1, served={"provider": "p", "model": "m"}, requested_model="m",
                              reports_model=True, messages=[], response="x")
    assert "finish_reason" not in plain


def test_the_engine_records_a_truncated_call_as_truncated(tmp_path) -> None:
    from core.engine import Engine

    eng = object.__new__(Engine)
    eng.fi_dir = tmp_path
    eng.quest_id = "q"
    eng.config = SimpleNamespace(provider=SimpleNamespace(model="m", node_models={}))
    eng._log = SimpleNamespace(debug=lambda *a, **k: None)
    eng._model_for_node = lambda node: None

    async def cut() -> str:
        LAST_CALL.set({"provider": "openai", "model": "m", "reported": True, "finish_reason": "length"})
        raise ModelAnswerTruncated("the model's answer was cut off at its output limit")

    with pytest.raises(ModelAnswerTruncated):
        asyncio.run(eng._recorded_call("implement", [{"role": "user", "content": "x"}], cut))
    (row,) = ar.read(tmp_path, ar.MODEL_CALLS)
    assert row["outcome"] == "truncated"


def test_a_cli_answer_whose_last_turn_stopped_at_the_limit_is_cut_off() -> None:
    spec = SimpleNamespace(argv=["claude"])
    with pytest.raises(ModelAnswerTruncated, match="max_tokens"):
        _raise_if_cut_off([{"text": ["a"], "stop": "max_tokens"}, {"text": ["b"], "stop": "max_tokens"}], spec)
    _raise_if_cut_off([{"text": ["a"], "stop": "max_tokens"}, {"text": ["b"], "stop": "end_turn"}], spec)
    _raise_if_cut_off([], spec)


# --- a per-step output limit; the quest stops for a person and resumes with the new setting ---------------------------


def test_the_output_limit_setting_is_checked_and_reaches_the_endpoint() -> None:
    from core.provider import node_output_limit

    ep = resolve_endpoint(ProviderConfig(name="openai", node_max_tokens={"write": 16000, "review_panel": 4000}))
    assert ep.node_max_tokens == {"write": 16000, "review_panel": 4000}
    assert node_output_limit(ep.node_max_tokens, "write") == 16000
    assert node_output_limit(ep.node_max_tokens, "write.patch") == 16000, "a sub-call gets its step's limit"
    assert node_output_limit(ep.node_max_tokens, "review_panel.statistician") == 4000
    assert node_output_limit(ep.node_max_tokens, "implement") is None, "unset: no limit is sent, as before"
    assert resolve_endpoint(ProviderConfig(name="openai")).node_max_tokens == {}
    for bad in ({"write": 0}, {"write": -5}, {"write": 10.5}, {"write": True}, {"": 100}):
        with pytest.raises(Exception, match="node_max_tokens"):
            ProviderConfig(name="openai", node_max_tokens=bad)


def test_the_step_setting_is_sent_and_doubled_once_when_cut_off() -> None:
    seen: list[dict] = []
    cfg = {**PLAIN, "node_max_tokens": {"write": 3000}}
    text, last, attempts = asyncio.run(_chat(cfg, _whole(["length", "stop"], seen), node="write"))
    assert text == "partial {" and [b["max_tokens"] for b in seen] == [3000, 6000]
    assert attempts[0]["usage"]["completion_tokens"] == 7, "the cut-off attempt's cost is kept"
    assert attempts[0]["finish_reason"] == "length"
    seen.clear()
    asyncio.run(_chat(cfg, _whole(["stop"], seen), node="implement"))
    assert "max_tokens" not in seen[0], "a step with no setting sends no limit"


def test_a_relay_that_says_max_tokens_is_a_cut_off_too() -> None:
    with pytest.raises(ModelAnswerTruncated):
        asyncio.run(_chat(PLAIN, _whole(["max_tokens"], [])))


def test_a_larger_limit_the_model_refuses_is_still_a_cut_off_not_a_raw_400() -> None:
    seen: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(200, json=_plain("partial {", "length"))
        return httpx.Response(400, json={"error": {"message": "max_tokens is too large"}})

    with pytest.raises(ModelAnswerTruncated, match="refused a larger one"):
        asyncio.run(_chat(PLAIN, handler, max_tokens=8192))
    assert [b["max_tokens"] for b in seen] == [8192, 16384]


def test_a_cut_off_or_filtered_answer_does_not_trip_the_providers_circuit() -> None:
    from core.provider import FallbackLLMClient

    class Primary:
        last_usage = None
        last_model = "m"

        def __init__(self, exc: Exception) -> None:
            self.exc, self.calls = exc, 0

        async def chat(self, messages, **kw):  # noqa: ANN001
            self.calls += 1
            raise self.exc

    class Backup:
        last_usage = None
        last_model = "b"

        async def chat(self, messages, **kw):  # noqa: ANN001
            return "ok"

    async def backup() -> Backup:
        return Backup()

    for exc in (ModelAnswerTruncated("cut"), ModelAnswerFiltered("withheld")):
        primary = Primary(exc)
        chain = FallbackLLMClient(primary, [("backup", backup)], breaker_threshold=1)  # type: ignore[arg-type]
        for _ in range(3):
            assert asyncio.run(chain.chat([{"role": "user", "content": "x"}])) == "ok"
        assert primary.calls == 3, "the primary is asked every time: its circuit never opened"


def test_the_engine_writes_a_cost_row_for_a_cut_off_attempt(tmp_path) -> None:
    from core.engine import Engine

    eng = object.__new__(Engine)
    eng.fi_dir = tmp_path
    eng.quest_id = "q"
    eng.config = SimpleNamespace(provider=SimpleNamespace(model="m", node_models={}))
    eng._log = SimpleNamespace(debug=lambda *a, **k: None)
    eng._model_for_node = lambda node: None

    async def cut() -> str:
        raise ModelAnswerTruncated("cut", usage={"prompt_tokens": 5, "completion_tokens": 4096, "total_tokens": 4101},
                                   model="m", finish_reason="length")

    with pytest.raises(ModelAnswerTruncated) as caught:
        asyncio.run(eng._recorded_call("write", [{"role": "user", "content": "x"}], cut))
    assert caught.value.node == "write", "the engine names the step the pause will name"
    rows = [json.loads(line) for line in (tmp_path / "cost.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and rows[-1]["usage"]["completion_tokens"] == 4096
    (call,) = ar.read(tmp_path, ar.MODEL_CALLS)
    assert call["outcome"] == "truncated" and call["finish_reason"] == "length"
    assert call["usage"]["completion_tokens"] == 4096


def _quest_config(tmp_path, title: str, limits: dict | None = None):
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig

    return Config(
        topic=f"a topic for the {title} test", title=title,
        provider=ProviderConfig(name="openai", node_max_tokens=limits),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_cut_off_step_stops_the_quest_and_resumes_with_a_larger_limit(tmp_path, monkeypatch) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _classify, _fake_response_for

    cut_at: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        step = kw.get("node") or ""
        if _classify(prompt) == "Experiment Design" and not self.endpoint.node_max_tokens.get(step):
            cut_at.append(step)
            raise ModelAnswerTruncated("the model's answer was cut off at its output limit", model="m")
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    first = Engine(_quest_config(tmp_path, "cut-off"))
    artifacts = await first.run()
    assert artifacts.paper_md is None and cut_at
    step = cut_at[0]
    card = (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "cut off at its output limit" in card and f"node_max_tokens: {{{step}:" in card, card
    assert f"node_models: {{{step}:" in card
    assert json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))["kind"] == "model_output"
    assert not (first.quest_root / "quest_failed.md").exists(), "a stop for a person, not a crash"

    # The person sets the limit the card names and resumes: the step runs again and the quest finishes.
    second = Engine(_quest_config(tmp_path, "cut-off", {step: 16000}), resume_quest_id=first.quest_id)
    artifacts = await second.run()
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert not (second.quest_root / "NEXT_STEP.md").exists()


@pytest.mark.asyncio
async def test_a_filtered_step_stops_with_what_to_change(tmp_path, monkeypatch) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _classify, _fake_response_for

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Experiment Design":
            raise ModelAnswerFiltered("withheld by the content filter", model="m")
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = Engine(_quest_config(tmp_path, "filtered"))
    await eng.run()
    card = (eng.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "content filter" in card and "node_models" in card, card
    assert json.loads((eng.fi_dir / "pause.json").read_text(encoding="utf-8"))["kind"] == "model_output"
    assert not (eng.quest_root / "quest_failed.md").exists()


def test_a_first_call_the_model_refuses_for_its_output_size_is_named_for_a_person() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "max_tokens is too large: 999999 > 8192"}})

    with pytest.raises(ModelAnswerTruncated, match="refused an output limit of 999999 tokens") as caught:
        asyncio.run(_chat(PLAIN, handler, max_tokens=999999))
    assert caught.value.refused and caught.value.limit == 999999


def test_a_400_that_is_not_about_the_output_size_stays_the_provider_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "invalid api key"}})

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_chat(PLAIN, handler, max_tokens=1000))


def test_the_seal_records_the_output_limits() -> None:
    src = (ar.__file__ and open(ar.__file__, encoding="utf-8").read()) or ""
    assert '"node_max_tokens"' in src


def _card_of(exc, *, provider_name: str = "openai") -> list[str]:
    from core.engine import Engine

    eng = object.__new__(Engine)
    eng.quest_id = "q1"
    eng.config = SimpleNamespace(provider=SimpleNamespace(name=provider_name))
    eng._log = SimpleNamespace(warning=lambda *a, **k: None)
    got: dict = {}
    eng._pause_for_human = lambda **kw: got.update(kw)  # type: ignore[method-assign]
    eng._pause_for_model_output("write", exc)
    return got["steps"] + [got["recommended"]] + got["alternatives"]


def test_the_card_suggests_twice_what_the_cut_off_answer_used_and_says_how_to_approve_a_model_change() -> None:
    exc = ModelAnswerTruncated("cut", node="write", limit=4000, provider="openai",
                               usage={"prompt_tokens": 1, "completion_tokens": 9000, "total_tokens": 9001})
    text = "\n".join(_card_of(exc))
    assert "node_max_tokens: {write: 18000}" in text, text
    assert "python launch.py --update q1" in text
    assert "`plan`" in text and "`review_panel`" in text, "the step names are listed"


def test_the_card_does_not_offer_an_output_limit_to_a_cli_provider() -> None:
    exc = ModelAnswerTruncated("cut", node="write", provider="claude_cli")
    text = "\n".join(_card_of(exc, provider_name="claude_cli"))
    assert "does not reach a CLI provider" in text and "node_max_tokens: {write" not in text
    assert "node_models" in text


def test_the_card_for_ollama_mentions_the_context_window_as_unchecked() -> None:
    text = "\n".join(_card_of(ModelAnswerTruncated("cut", node="write", limit=1000, provider="ollama")))
    assert "num_ctx" in text and "not checked" in text


def test_the_card_for_a_refused_limit_says_so() -> None:
    exc = ModelAnswerTruncated("refused", node="write", limit=99999, provider="openai", refused=True)
    text = "\n".join(_card_of(exc))
    assert "would not take an output limit of 99999" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("head,step", [("Review", "review"), ("ClaimCheck", "claim_check")])
async def test_a_cut_off_review_step_is_a_model_output_pause_not_a_reviewer_outage(tmp_path, monkeypatch, head, step) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _classify, _fake_response_for

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == head:
            raise ModelAnswerTruncated("cut off at its output limit", model="m", node=kw.get("node") or step)
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = Engine(_quest_config(tmp_path, f"cut-{step}"))
    await eng.run()
    pause = json.loads((eng.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert pause["kind"] == "model_output", pause["kind"]
    card = (eng.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert f"node_max_tokens: {{{step}:" in card, card
    assert "usage limit" not in card


@pytest.mark.asyncio
async def test_a_cut_off_panel_reviewer_is_a_model_output_pause_naming_the_panel_key(tmp_path, monkeypatch) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _classify, _fake_response_for

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        node = kw.get("node") or ""
        if node.startswith("review_panel."):
            raise ModelAnswerTruncated("cut off at its output limit", model="m", node=node)
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = _quest_config(tmp_path, "cut-panel")
    cfg.engine.review_panel = ["skeptic", "methodologist"]
    eng = Engine(cfg)
    await eng.run()
    pause = json.loads((eng.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert pause["kind"] == "model_output", pause["kind"]
    assert "node_max_tokens: {review_panel:" in (eng.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
