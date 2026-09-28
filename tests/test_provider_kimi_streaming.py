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


async def _chat(endpoint, handler, *, timeout_s: float = 120.0, **kwargs):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
    try:
        client = LLMClient(endpoint, http=http, timeout_s=timeout_s)
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


# --- review follow-ups ------------------------------------------------------------------------------------------------


class _Forever(httpx.AsyncByteStream):
    """Bytes keep coming (so no read timeout fires) but the answer never ends: keep-alive comments, or a slow trickle."""

    def __init__(self, line: bytes) -> None:
        self.line = line

    async def __aiter__(self):
        while True:
            await asyncio.sleep(0.05)
            yield self.line


@pytest.mark.parametrize("line", [b": keep-alive\n\n", b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'],
                         ids=["keepalive", "trickle"])
def test_a_stream_that_never_finishes_is_bounded_by_the_step_budget_and_tried_again(line: bytes) -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, stream=_Forever(line))
        return httpx.Response(200, content=_sse(_chunk("done", finish="stop")))

    async def run():
        return await asyncio.wait_for(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler, timeout_s=0.5), 10)

    text, _c, _l = asyncio.run(run())
    assert text == "done" and len(calls) == 2


def test_several_data_lines_of_one_event_are_joined() -> None:
    body = (b'data: {"choices":[{"delta":\n'
            b'data: {"content":"kept, LOST-no-more"}}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    text, _c, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert text == "kept, LOST-no-more"


def test_an_event_that_does_not_parse_is_never_skipped() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=b'data: {"choices": [broken\n\n' + _sse(_chunk("x", finish="stop")))
        return httpx.Response(200, content=_sse(_chunk("clean", finish="stop")))

    text, _c, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert text == "clean" and len(calls) == 2, "the broken event failed the attempt instead of being dropped"


@pytest.mark.parametrize("error", [
    {"error": {"type": "content_filter", "message": "The request was rejected because it was considered high risk"}},
    {"error": {"type": "exceeded_current_quota_error", "message": "Your account is suspended, please check your plan"}},
    {"error": {"type": "invalid_request_error", "message": "bad"}},
])
def test_an_error_retrying_cannot_fix_is_not_tried_again(error: dict) -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, content=_sse(error, done=False))

    with pytest.raises(httpx.HTTPStatusError) as e:
        asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert len(calls) == 1 and e.value.response.status_code == 400


def test_an_error_event_without_an_error_key_is_an_error() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=b'event: error\ndata: {"message": "upstream overloaded"}\n\n')
        return httpx.Response(200, content=_sse(_chunk("after", finish="stop")))

    text, _c, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert text == "after" and len(calls) == 2


def test_done_without_a_finish_reason_is_a_cut_off_answer() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=_sse(_chunk("half")))
        return httpx.Response(200, content=_sse(_chunk("whole", finish="stop")))

    text, _c, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert text == "whole" and len(calls) == 2


def test_a_finished_answer_without_done_or_usage_is_kept_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(_chunk("final", finish="stop"), done=False))

    with caplog.at_level(logging.WARNING, logger="frontier_insight.provider"):
        text, _c, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert text == "final"
    assert any("no [DONE]" in r.getMessage() and "no usage" in r.getMessage() for r in caplog.records)


def test_stream_options_from_extra_body_are_kept_and_a_stream_key_is_refused() -> None:
    sent: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(_chunk("ok", finish="stop")))

    kimi = resolve_endpoint(ProviderConfig(**KIMI, extra_body={"stream_options": {"chunk_include_extra": True}}))
    asyncio.run(_chat(kimi, handler))
    assert sent[0]["stream_options"] == {"chunk_include_extra": True, "include_usage": True}
    for cfg in (dict(KIMI, extra_body={"stream": False}), dict(name="openai", extra_body={"stream": True})):
        with pytest.raises(ValueError, match="stream"):
            asyncio.run(_chat(resolve_endpoint(ProviderConfig(**cfg)), handler))


def test_no_content_at_all_is_none_as_a_plain_call_returns() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(_chunk(finish="tool_calls")))

    text, _c, _l = asyncio.run(_chat(resolve_endpoint(ProviderConfig(**KIMI)), handler))
    assert text is None


# --- second review pass: which in-stream errors are tried again ---------------------------------------------------------

TRANSIENT = [
    {"message": "Rate limit reached: per-minute token quota exceeded, retry in 20s"},
    {"message": "request reached max TPM quota, please try again after 1 seconds"},
    {"message": "insufficient capacity, try again"},
    {"message": "Insufficient system resources"},
    {"message": "upstream connect error: load balancer reset"},
    {"message": "bad gateway from authentication proxy"},
    {"type": "engine_overloaded_error", "message": "The engine is currently overloaded"},
    {"type": "rate_limit_reached_error", "message": "Your account reached max RPM"},
    {"type": "server_error", "message": "x"},
    {"code": 500, "message": "x"},
    {"code": "503", "message": "x"},
    {"code": 429, "type": "exceeded_current_quota_error", "message": "slow down"},
]
PERMANENT = [
    {"type": "content_filter", "message": "The request was rejected because it was considered high risk"},
    {"type": "exceeded_current_quota_error", "message": "Your account is suspended, please check your plan"},
    {"type": "invalid_request_error", "message": "bad"},
    {"type": "invalid_authentication_error", "message": "Invalid Authentication"},
    {"code": 401, "message": "x"},
]


@pytest.mark.parametrize("error", TRANSIENT, ids=[str(e)[:40] for e in TRANSIENT])
def test_an_error_that_will_clear_is_tried_again(error: dict) -> None:
    from core.provider import _retry_http_error, _stream_error

    assert _retry_http_error(_stream_error(error, httpx.Request("POST", "http://x"))), error


@pytest.mark.parametrize("error", PERMANENT, ids=[str(e)[:40] for e in PERMANENT])
def test_an_error_retrying_cannot_fix_stops_at_once(error: dict) -> None:
    from core.provider import _retry_http_error, _stream_error

    assert not _retry_http_error(_stream_error(error, httpx.Request("POST", "http://x"))), error


def test_the_permanent_class_is_read_from_type_and_code_not_the_message() -> None:
    from core.provider import _retry_http_error, _stream_error

    err = _stream_error({"type": "unknown_error", "message": "content filter quota billing"}, httpx.Request("POST", "u"))
    assert _retry_http_error(err)


def test_stream_in_extra_body_is_refused_when_the_config_loads_and_in_a_per_call_extra() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="stream"):
        ProviderConfig(**KIMI, extra_body={"stream": True})

    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover -- never reached
        return httpx.Response(200, content=_sse(_chunk("ok", finish="stop")))

    with pytest.raises(ValueError, match="stream"):
        asyncio.run(_chat(resolve_endpoint(ProviderConfig(name="openai")), handler, extra={"stream": True}))


@pytest.mark.parametrize("error", [
    {"type": "invalid_request_error", "message": "please try again with a shorter prompt"},
    {"type": "authentication_error", "message": "temporarily blocked"},
    {"code": "context_length_exceeded", "message": "too long"},
])
def test_a_permanent_type_or_code_is_not_retried_whatever_the_message_hopes(error) -> None:
    import httpx

    from core.provider import _stream_error

    req = httpx.Request("POST", "https://api.moonshot.ai/v1/chat/completions")
    assert isinstance(_stream_error(error, req), httpx.HTTPStatusError)
