"""A provider's server that is down or busy for a few minutes does not end the quest.

Observed 2026-09-30 on a real Kimi quest: Moonshot answered ``HTTP 521`` (Cloudflare: origin down) on the
``execute_reflect`` call; the HTTP transport tried four times with waits of 1 s, 1 s and 4 s, then the quest failed
after ~134k tokens. Now a 5xx, or a 429 rate limit that is not a used-up quota, gets six attempts with jittered waits of
about 10, 20, 40, 60 and 90 s (3-4.5 minutes in all), a sane ``Retry-After`` is honoured, and a 4xx or a used-up quota
still fails at once. No test here sleeps: the waits the policy chooses are recorded and replaced by zero. No real
network, no real model.
"""

from __future__ import annotations

import email.utils
import time
from types import SimpleNamespace

import httpx
import pytest

import core.provider as provider
from core.config import ProviderConfig
from core.provider import LLMClient, resolve_endpoint

PLAIN = dict(name="openai", base_url="https://api.example.com/v1", api_key_env="OPENAI_API_KEY", model="gpt-x")
KIMI = dict(name="openai", base_url="https://api.moonshot.ai/v1", api_key_env="MOONSHOT_API_KEY", model="kimi-k3")
OK = {"model": "gpt-x", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The waits the outage policy chose, in order; the actual sleep is zero."""
    import tenacity

    chosen: list[float] = []
    real = getattr(provider, "_http_outage_wait_s", None)

    def record(attempt_number: int, exc: BaseException | None) -> float:
        chosen.append(real(attempt_number, exc))
        return 0.0

    # raising=False: the same test runs against the old policy (no such function) to show it failing.
    monkeypatch.setattr("core.provider._http_outage_wait_s", record, raising=False)
    monkeypatch.setattr("core.provider.wait_random_exponential", lambda **kw: tenacity.wait_none())
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    return chosen


def _replies(*responses: httpx.Response):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return responses[min(len(calls), len(responses)) - 1]

    return handler, calls


async def _chat(cfg: dict, handler, node: str = "execute_reflect") -> str:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
    try:
        client = LLMClient(resolve_endpoint(ProviderConfig(**cfg)), http=http)
        return await client.chat([{"role": "user", "content": "hi"}], node=node)
    finally:
        await http.aclose()


def _cloudflare(status: int = 521, **headers: str) -> httpx.Response:
    return httpx.Response(status, text="<html>origin is down</html>", headers=headers)


# --- the observed failure --------------------------------------------------------------------------------------------


async def test_a_521_outage_that_outlasts_four_attempts_is_ridden_out(waits: list[float]) -> None:
    """Four 521s then an answer: the old policy (four attempts, ~6 s) failed the quest here; the new one succeeds,
    having waited minutes, not seconds."""
    handler, calls = _replies(*[_cloudflare(521)] * 4, httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(calls) == 5
    assert len(waits) == 4
    assert sum(waits) >= 0.8 * (10 + 20 + 40 + 60), waits


async def test_three_521s_then_an_answer_succeeds(waits: list[float]) -> None:
    handler, calls = _replies(*[_cloudflare(521)] * 3, httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(calls) == 4
    assert all(w >= 8.0 for w in waits), "no near-zero wait on an outage"


async def test_the_streamed_kimi_path_rides_out_a_521_too(waits: list[float]) -> None:
    sse = ('data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
           'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
           '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
           "data: [DONE]\n\n")
    handler, calls = _replies(*[_cloudflare(521)] * 4,
                              httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))
    assert await _chat(KIMI, handler) == "hello"
    assert len(calls) == 5


async def test_an_outage_longer_than_the_budget_still_fails(waits: list[float]) -> None:
    handler, calls = _replies(_cloudflare(503))
    with pytest.raises(httpx.HTTPStatusError):
        await _chat(PLAIN, handler)
    assert len(calls) == 6
    # tenacity asks for the wait after the last attempt too (before it checks the stop), and never sleeps it.
    slept = waits[:5]
    assert 120.0 <= sum(slept) <= 300.0, waits


# --- what is not waited for ------------------------------------------------------------------------------------------


async def test_a_400_is_not_retried(waits: list[float]) -> None:
    handler, calls = _replies(httpx.Response(400, json={"error": {"message": "bad request", "type": "invalid_request"}}))
    with pytest.raises(httpx.HTTPStatusError):
        await _chat(PLAIN, handler)
    assert len(calls) == 1 and waits == []


async def test_a_used_up_quota_fails_at_once(waits: list[float]) -> None:
    body = {"error": {"message": "You exceeded your current quota, please check your plan and billing details.",
                      "type": "insufficient_quota", "code": "insufficient_quota"}}
    handler, calls = _replies(httpx.Response(429, json=body))
    with pytest.raises(httpx.HTTPStatusError):
        await _chat(PLAIN, handler)
    assert len(calls) == 1 and waits == []


async def test_moonshots_used_up_quota_fails_at_once(waits: list[float]) -> None:
    body = {"error": {"message": "Your account is suspended", "type": "exceeded_current_quota_error"}}
    handler, calls = _replies(httpx.Response(429, json=body))
    with pytest.raises(httpx.HTTPStatusError):
        await _chat(PLAIN, handler)
    assert len(calls) == 1


async def test_a_rate_limit_429_is_waited_out(waits: list[float]) -> None:
    body = {"error": {"message": "Rate limit reached for requests", "type": "rate_limit_exceeded"}}
    handler, calls = _replies(*[httpx.Response(429, json=body)] * 4, httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(calls) == 5 and all(w >= 8.0 for w in waits)


async def test_a_dropped_connection_keeps_the_short_four_attempt_budget(waits: list[float]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        handler.n += 1
        raise httpx.ConnectError("connection reset")

    handler.n = 0
    with pytest.raises(httpx.ConnectError):
        await _chat(PLAIN, handler)
    assert handler.n == 4 and waits == [], "a transport error is not a server outage"


# --- Retry-After -----------------------------------------------------------------------------------------------------


async def test_retry_after_seconds_is_honoured(waits: list[float]) -> None:
    handler, calls = _replies(_cloudflare(503, **{"Retry-After": "7"}), httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(waits) == 1 and 7.0 <= waits[0] <= 7.7


async def test_retry_after_as_an_http_date_is_honoured(waits: list[float]) -> None:
    when = email.utils.formatdate(time.time() + 30, usegmt=True)
    handler, calls = _replies(_cloudflare(503, **{"Retry-After": when}), httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(waits) == 1 and 25.0 <= waits[0] <= 33.0


@pytest.mark.parametrize("value", ["3600", "0", "-5", "soon", "nan"])
async def test_an_unreasonable_retry_after_falls_back_to_the_schedule(waits: list[float], value: str) -> None:
    handler, calls = _replies(_cloudflare(503, **{"Retry-After": value}), httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(waits) == 1 and 8.0 <= waits[0] <= 12.0, "the first scheduled wait, 10 s +/- 20 %"


# --- the policy itself and its log line ------------------------------------------------------------------------------


def test_the_schedule_rides_out_two_to_five_minutes() -> None:
    lo = sum(provider._HTTP_OUTAGE_WAITS_S) * 0.8
    hi = sum(provider._HTTP_OUTAGE_WAITS_S) * 1.2
    assert 120.0 <= lo and hi <= 300.0
    assert provider._HTTP_OUTAGE_ATTEMPTS == len(provider._HTTP_OUTAGE_WAITS_S) + 1


def test_the_retry_line_says_the_server_is_down_and_how_long_it_waits() -> None:
    req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    exc = httpx.HTTPStatusError("Server error '521 <unknown>'", request=req, response=httpx.Response(521, request=req))
    rs = SimpleNamespace(attempt_number=2, next_action=SimpleNamespace(sleep=21.4),
                         outcome=SimpleNamespace(exception=lambda: exc))
    line = provider._retry_line("execute_reflect", "kimi over HTTP", rs, 6)
    assert "attempt 2 of 6" in line
    assert "the provider's server is down or busy (HTTP 521)" in line
    assert line.endswith("trying again in 21s")
