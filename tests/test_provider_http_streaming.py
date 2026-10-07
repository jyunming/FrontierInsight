"""Every direct HTTP provider's chat calls are streamed (OpenAI-compatible ``stream: true``), so a reasoning model that is
silent to one plain request for longer than its read timeout still completes. The step's HTTP timeout limits the SILENCE
between two events (a keep-alive comment is not an event), a whole-call budget bounds the call, reasoning deltas count as
activity and are kept for ``.fi/thinking.jsonl``; a server that cannot stream keeps working. In-memory transports only."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from core import thinking_capture as tc
from core.config import ProviderConfig
from core.provider import LLMClient, _http_streams, resolve_endpoint

MESSAGES = [{"role": "user", "content": "hi"}]


def _event(delta: dict[str, Any] | None = None, finish: str | None = None, **top: Any) -> bytes:
    chunk = {"choices": [{"delta": delta or {}, "finish_reason": finish}], **top}
    return f"data: {json.dumps(chunk)}\n\n".encode()


DONE = b"data: [DONE]\n\n"


class Paced(httpx.AsyncByteStream):
    """Sends ``parts`` one at a time, ``gap`` seconds apart, then (with ``hang``) stays silent that long."""

    def __init__(self, parts: list[bytes], gap: float = 0.0, hang: float = 0.0, tail: bytes = b"") -> None:
        self.parts, self.gap, self.hang, self.tail = parts, gap, hang, tail

    async def __aiter__(self):
        for part in self.parts:
            await asyncio.sleep(self.gap)
            yield part
        if self.hang:
            await asyncio.sleep(self.hang)
        if self.tail:
            yield self.tail


async def _chat(provider: dict[str, Any], handler, *, timeout_s: float = 1.0, node: str | None = None):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    holder, token = tc.open_holder()
    try:
        client = LLMClient(resolve_endpoint(ProviderConfig(**provider)), http=http, timeout_s=timeout_s)
        text = await (client.chat(MESSAGES) if node is None else client.chat(MESSAGES, node=node))
        return text, holder
    finally:
        await http.aclose()
        tc.close_holder(token)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    import tenacity

    monkeypatch.setattr("core.provider.wait_random_exponential", lambda **kw: tenacity.wait_none())


@pytest.mark.parametrize("name", ["openai", "codex", "gemini", "ollama", "vllm"])
def test_every_direct_provider_streams_and_a_proxy_does_not(name: str) -> None:
    assert _http_streams(resolve_endpoint(ProviderConfig(name=name)))


def test_a_proxy_provider_and_a_hand_built_endpoint_are_not_streamed() -> None:
    from core.provider import ResolvedEndpoint

    assert not _http_streams(ResolvedEndpoint(base_url="http://x/v1", model="m", api_key="k", provider_name="claude_code"))
    assert not _http_streams(ResolvedEndpoint(base_url="http://x/v1", model="m", api_key="k"))


def test_a_step_provider_built_from_node_providers_streams_too() -> None:
    from core.config import StepProviderConfig
    from core.provider import step_provider_config

    main = ProviderConfig(name="ollama", model="small")
    entry = StepProviderConfig(name="openai", model="big", base_url="https://api.example.com/v1")
    assert _http_streams(resolve_endpoint(step_provider_config(main, entry)))


def test_a_slow_but_steady_stream_completes_although_the_whole_call_is_longer_than_the_silence_limit() -> None:
    """Five events 0.4 s apart take 2 s in all: more than the 1 s silence limit, less than the 4 s budget. A plain request
    that long with no bytes would have timed out."""
    parts = [_event({"reasoning_content": "thinking "}), _event({"reasoning_content": "more "}),
             _event({"content": "the "}), _event({"content": "answer"}), _event({}, "stop"), DONE]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Paced(parts, gap=0.4))

    text, holder = asyncio.run(_chat({"name": "openai", "model": "m"}, handler, timeout_s=1.0))
    assert text == "the answer"
    assert holder["text"] == "thinking more ", "the reasoning deltas are kept for .fi/thinking.jsonl"
    assert any("thinking from" in line for line in holder["timings"]), holder["timings"]


def test_reasoning_deltas_count_as_activity_while_no_answer_text_arrives() -> None:
    parts = [_event({"reasoning_content": "."}) for _ in range(6)] + [_event({"content": "ok"}), _event({}, "stop"), DONE]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Paced(parts, gap=0.3))

    text, holder = asyncio.run(_chat({"name": "openai", "model": "m"}, handler, timeout_s=0.8))
    assert text == "ok" and holder["text"] == "......"


def test_a_stream_that_goes_silent_times_out_and_is_tried_again() -> None:
    calls: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, stream=Paced([_event({"content": "par"})], hang=30))
        return httpx.Response(200, stream=Paced([_event({"content": "whole"}), _event({}, "stop"), DONE]))

    async def run():
        return await asyncio.wait_for(_chat({"name": "openai", "model": "m"}, handler, timeout_s=0.5), 20)

    text, _holder = asyncio.run(run())
    assert text == "whole" and len(calls) == 2


def test_the_silence_error_names_the_limit_and_the_whole_call_is_cut_at_the_budget() -> None:
    from core.provider import _post_streamed

    async def silent(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Paced([_event({"content": "x"})], hang=30))

    async def trickle(request: httpx.Request) -> httpx.Response:
        parts = [_event({"content": "x"}) for _ in range(400)]
        return httpx.Response(200, stream=Paced(parts, gap=0.05))

    async def go(handler, timeout: float, inactivity: float | None = None):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await _post_streamed(http, "http://x/v1/chat/completions", {"model": "m"}, {}, timeout, inactivity)

    with pytest.raises(httpx.ReadTimeout, match="sent nothing for 0.4 s"):
        asyncio.run(asyncio.wait_for(go(silent, 0.4), 10))
    # Events keep arriving (so the silence limit never fires), but the call goes on longer than 4 x 0.25 s: cut.
    with pytest.raises(httpx.ReadTimeout, match="did not finish its answer within 1 s"):
        asyncio.run(asyncio.wait_for(go(trickle, 0.25), 10))


def test_a_keep_alive_comment_is_not_activity() -> None:
    from core.provider import _post_streamed

    class Comments(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.05)
                yield b": keep-alive\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Comments())

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await _post_streamed(http, "http://x/v1/chat/completions", {"model": "m"}, {}, 0.4)

    with pytest.raises(httpx.ReadTimeout, match="sent nothing"):
        asyncio.run(asyncio.wait_for(go(), 10))


def test_a_server_that_ignores_stream_and_answers_with_one_json_reply_still_works() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "whole reply"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}})

    text, _holder = asyncio.run(_chat({"name": "vllm", "model": "m"}, handler))
    assert text == "whole reply"


def test_a_server_that_refuses_stream_is_sent_the_request_whole_and_not_asked_to_stream_again() -> None:
    sent: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        if body.get("stream"):
            return httpx.Response(400, json={"error": {"message": "unknown field `stream_options`"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "plain"}, "finish_reason": "stop"}]})

    async def two_calls():
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            client = LLMClient(resolve_endpoint(ProviderConfig(name="vllm", model="m")), http=http)
            return [await client.chat(MESSAGES), await client.chat(MESSAGES)]
        finally:
            await http.aclose()

    assert asyncio.run(two_calls()) == ["plain", "plain"]
    assert [bool(b.get("stream")) for b in sent] == [True, False, False], "streamed once, then whole"


def test_a_refusal_that_is_not_about_stream_is_not_taken_for_one() -> None:
    sent: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "the prompt is too long"}})

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_chat({"name": "openai", "model": "m"}, handler))
    assert len(sent) == 1 and sent[0]["stream"] is True


def test_an_error_inside_a_stream_that_started_is_never_read_as_a_refusal_of_stream() -> None:
    from core.provider import _refuses_streaming, _stream_error

    err = _stream_error({"type": "invalid_request_error", "message": "stream broke"}, httpx.Request("POST", "http://x"))
    assert isinstance(err, httpx.HTTPStatusError) and not _refuses_streaming(err)


def test_the_node_timeout_is_the_silence_limit_for_that_node() -> None:
    """``provider.node_http_timeout_s`` for a step is its silence limit, as its read timeout was."""
    calls: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, stream=Paced([_event({"reasoning_content": "t"}), _event({"content": "ok"}),
                                                 _event({}, "stop"), DONE], gap=0.3))

    async def run():
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            client = LLMClient(resolve_endpoint(ProviderConfig(name="openai", model="m")), http=http, timeout_s=0.1,
                               node_http_timeout_s={"implement": 1.0})
            return await client.chat(MESSAGES, node="implement")
        finally:
            await http.aclose()

    assert asyncio.run(run()) == "ok" and len(calls) == 1
