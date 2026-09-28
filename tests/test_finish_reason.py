"""An answer the model did not finish is never handed on as whole (the 2026-09-28 R2 re-audit, P1-02).

A reply that ended with ``finish_reason: length`` (cut off at the output limit) used to be returned as a normal answer,
so a partial script or JSON could flow into later steps; ``content_filter`` was not told apart either. Now: a call that
set ``max_tokens`` is asked once more with twice the limit (capped), and an answer still cut off raises
:class:`ModelAnswerTruncated`; a filtered one raises :class:`ModelAnswerFiltered` (not retried); the reason is kept in
the call's record. No real API is called here: ``httpx.MockTransport``.
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
    with pytest.raises(ModelAnswerTruncated, match="max_tokens 2000"):
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
