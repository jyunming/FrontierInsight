"""Kimi's chat calls are streamed; every other HTTP provider keeps the plain request.

Long non-streamed requests to Moonshot's API stalled with zero bytes until the read timeout, while the same request
streamed completed (the 2026-09-27 Kimi K3 campaign). No real API is called here: ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.config import ProviderConfig
from core.provider import LAST_CALL, LLMClient, _http_streams, resolve_endpoint

KIMI = dict(name="openai", base_url="https://api.moonshot.ai/v1", api_key_env="MOONSHOT_API_KEY", model="kimi-k3")


def _sse(*events: object, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _chunk(text: str = "", finish: str | None = None, **top: object) -> dict:
    return {"choices": [{"delta": {"content": text} if text else {}, "finish_reason": finish}], **top}


async def _chat(endpoint, handler, **kwargs):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
    try:
        client = LLMClient(endpoint, http=http)
        text = await client.chat([{"role": "user", "content": "hi"}], **kwargs)
        return text, client, dict(LAST_CALL.get() or {})
    finally:
        await http.aclose()


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    import tenacity

    monkeypatch.setattr("core.provider.wait_random_exponential", lambda **kw: tenacity.wait_none())


def test_only_moonshot_hosts_stream() -> None:
    assert _http_streams(resolve_endpoint(ProviderConfig(**KIMI)))
    assert _http_streams(resolve_endpoint(ProviderConfig(**{**KIMI, "base_url": "https://api.moonshot.cn/v1"})))
    for cfg in (dict(name="openai"), dict(name="gemini"), dict(name="ollama"), dict(name="vllm"),
                dict(name="openai", base_url="https://moonshot.example.com/v1")):
        assert not _http_streams(resolve_endpoint(ProviderConfig(**cfg))), cfg


def test_the_stream_is_put_back_together() -> None:
    sent: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(
            _chunk("Hello, ", model="kimi-k3-0927"), _chunk("world"), _chunk(finish="stop"),
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}},
        ))

    text, client, last = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert text == "Hello, world"
    assert sent[0]["stream"] is True and sent[0]["stream_options"] == {"include_usage": True}
    assert client.last_usage["prompt_tokens"] == 12 and client.last_usage["completion_tokens"] == 3
    assert last["model"] == "kimi-k3-0927" and last["reported"] is True


def test_usage_on_the_finishing_choice_and_an_estimate_when_none() -> None:
    async def with_usage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop",
                          "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}]}))

    _t, client, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), with_usage))
    assert client.last_usage["total_tokens"] == 6

    async def without(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(_chunk("ok", finish="stop")))

    _t, client, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), without))
    assert client.last_usage and client.last_usage.get("estimated") is True


def test_an_error_inside_the_stream_is_tried_again() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=_sse({"error": {"message": "server overloaded"}}, done=False))
        return httpx.Response(200, content=_sse(_chunk("fine", finish="stop")))

    text, _c, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert text == "fine" and len(calls) == 2


def test_a_stream_cut_off_before_the_answer_ends_is_tried_again() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=_sse(_chunk("half an ans"), done=False))
        return httpx.Response(200, content=_sse(_chunk("whole", finish="stop")))

    text, _c, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert text == "whole" and len(calls) == 2


class _StallingStream(httpx.AsyncByteStream):
    """A stream that sends one chunk and then stops sending: a real transport raises ReadTimeout there."""

    async def __aiter__(self):
        yield _sse(_chunk("partial"), done=False)
        raise httpx.ReadTimeout("no more bytes")


def test_a_stream_that_stops_sending_fails_like_a_stalled_request_and_is_tried_again() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, stream=_StallingStream())
        return httpx.Response(200, content=_sse(_chunk("recovered", finish="stop")))

    text, _c, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert text == "recovered" and len(calls) == 2


def test_an_error_status_is_not_retried_when_it_is_a_4xx() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert len(calls) == 1


def test_a_streamed_call_can_be_cancelled() -> None:
    entered = asyncio.Event()

    class _Hang(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await asyncio.sleep(60)
            yield b""  # pragma: no cover

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_Hang())

    async def run() -> None:
        task = asyncio.create_task(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(run())


def test_other_providers_still_send_a_plain_request() -> None:
    sent: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "plain"}}], "usage": {}, "model": "gpt-x"})

    text, _c, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(name="openai", model="gpt-x")), handler))
    assert text == "plain" and "stream" not in sent[0] and "stream_options" not in sent[0]
