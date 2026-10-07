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
    handler, calls = _replies(_cloudflare(429, **{"Retry-After": "7"}), httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(waits) == 1 and 7.0 <= waits[0] <= 7.7


async def test_a_tiny_retry_after_on_a_5xx_does_not_shorten_the_outage_wait(waits: list[float]) -> None:
    """A 5xx's Retry-After only lengthens the wait: ``Retry-After: 1`` on every 503 would otherwise spend the whole
    outage budget in seconds, the failure this policy exists to fix."""
    handler, calls = _replies(_cloudflare(503, **{"Retry-After": "1"}), httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(waits) == 1 and 8.0 <= waits[0] <= 12.0


async def test_a_long_retry_after_on_a_5xx_is_honoured(waits: list[float]) -> None:
    handler, calls = _replies(_cloudflare(503, **{"Retry-After": "100"}), httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(waits) == 1 and 100.0 <= waits[0] <= 110.0


def test_one_call_waits_at_most_five_minutes_in_all() -> None:
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    exc = httpx.HTTPStatusError("x", request=req,
                                response=httpx.Response(429, request=req, headers={"Retry-After": "120"}))
    rs = SimpleNamespace(attempt_number=3, idle_for=290.0, outcome=SimpleNamespace(exception=lambda: exc))
    assert provider._http_retry_wait(rs) <= 10.0 + 1e-9
    rs_done = SimpleNamespace(attempt_number=3, idle_for=300.0, outcome=SimpleNamespace(exception=lambda: exc))
    assert provider._http_retry_stop(rs_done) is True


async def test_retry_after_as_an_http_date_is_honoured(waits: list[float]) -> None:
    when = email.utils.formatdate(time.time() + 30, usegmt=True)
    handler, calls = _replies(_cloudflare(503, **{"Retry-After": when}), httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(waits) == 1 and 25.0 <= waits[0] <= 34.0


@pytest.mark.parametrize("value", ["3600", "0", "-5", "soon", "nan"])
async def test_an_unreasonable_retry_after_falls_back_to_the_schedule(waits: list[float], value: str) -> None:
    handler, calls = _replies(_cloudflare(503, **{"Retry-After": value}), httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    assert len(waits) == 1 and 8.0 <= waits[0] <= 12.0, "the first scheduled wait, 10 s +/- 20 %"


# --- one outage, several symptoms; a fallback provider; the concurrency slot -----------------------------------------


async def test_a_timeout_in_the_middle_of_an_outage_keeps_the_outage_budget(waits: list[float]) -> None:
    """Behind Cloudflare a sick origin answers 5xx, then times out: the same outage. The budget is set by any outage
    status seen in this call, not by the latest error alone."""
    n = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        n["calls"] += 1
        if n["calls"] <= 3:
            return _cloudflare(503)
        raise httpx.ReadTimeout("origin slow")

    with pytest.raises(httpx.ReadTimeout):
        await _chat(PLAIN, handler)
    assert n["calls"] == 6


async def test_with_a_fallback_provider_ready_an_outage_gets_the_short_budget(waits: list[float]) -> None:
    """When another provider can take the call, minutes on a dead one are wasted: four quick attempts, then the
    fallback chain moves on (the probe of a tripped provider is treated the same way)."""
    token = provider.CALL_SLOT.set({"provider": "kimi", "fallback": False, "short_retry": True})
    try:
        handler, calls = _replies(_cloudflare(521))
        with pytest.raises(httpx.HTTPStatusError):
            await _chat(PLAIN, handler)
    finally:
        provider.CALL_SLOT.reset(token)
    assert len(calls) == 4 and waits == []


async def test_the_fallback_chain_marks_the_primary_short_only_when_another_provider_is_ready() -> None:
    seen: list[dict] = []

    class _Fake:
        def __init__(self, ok: bool) -> None:
            self.ok = ok
            self.last_usage = None
            self.last_model = "m"

        async def chat(self, messages, **kw):
            seen.append(dict(provider.CALL_SLOT.get() or {}))
            if not self.ok:
                raise RuntimeError("down")
            return "fine"

    async def factory():
        return _Fake(True)

    chain = provider.FallbackLLMClient(_Fake(False), [("backup", factory)])
    assert await chain.chat([{"role": "user", "content": "x"}]) == "fine"
    assert seen[0]["short_retry"] is True, "the primary, with a backup ready"
    assert seen[1]["short_retry"] is False, "the last provider waits an outage out"


async def test_when_every_later_provider_is_tripped_the_call_waits_the_outage_out() -> None:
    """A chain whose other providers are all tripped is no less patient than no chain at all."""
    seen: list[dict] = []

    class _Fake:
        last_usage = None
        last_model = "m"

        async def chat(self, messages, **kw):
            seen.append(dict(provider.CALL_SLOT.get() or {}))
            return "fine"

    async def factory():
        return _Fake()

    chain = provider.FallbackLLMClient(_Fake(), [("backup", factory)], breaker_cooldown_s=0.0)
    chain._slots[1].tripped = True
    chain._slots[1].tripped_at = 0.0
    assert await chain.chat([{"role": "user", "content": "x"}]) == "fine"
    assert seen[0]["short_retry"] is False


async def test_the_concurrency_slot_is_given_back_while_waiting() -> None:
    import asyncio

    sem = asyncio.Semaphore(1)
    await sem.acquire()
    box = {"sem": sem, "held": True}
    token = provider._HELD_CALL_SLOT.set(box)
    try:
        sleeper = asyncio.create_task(provider._http_retry_sleep(0.2))
        await asyncio.sleep(0.05)
        await asyncio.wait_for(sem.acquire(), timeout=1.0)  # another call gets the slot during the wait
        sem.release()
        await sleeper
    finally:
        provider._HELD_CALL_SLOT.reset(token)
    assert sem.locked() and box["held"], "the waiting call holds its slot again before its next attempt"
    sem.release()


async def test_a_cancel_while_the_slot_is_taken_elsewhere_is_not_held_up_and_leaves_the_count_balanced() -> None:
    import asyncio

    sem = asyncio.Semaphore(1)
    await sem.acquire()
    box = {"sem": sem, "held": True}

    async def waiting_call() -> None:
        provider._HELD_CALL_SLOT.set(box)
        await provider._http_retry_sleep(0.05)

    task = asyncio.create_task(waiting_call())
    await asyncio.sleep(0.01)
    await sem.acquire()  # another call takes the freed slot and keeps it
    await asyncio.sleep(0.1)  # the waiting call is now queued for the slot
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)
    assert box["held"] is False, "the cancelled call does not hold the slot, so it will not release it"
    sem.release()  # the other call finishes
    assert not sem.locked(), "the count is back where it started"


async def test_a_capped_call_through_an_outage_leaves_the_semaphore_balanced(waits: list[float],
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FI_MAX_CONCURRENT_LLM_CALLS", "1")
    handler, calls = _replies(*[_cloudflare(503)] * 2, httpx.Response(200, json=OK))
    assert await _chat(PLAIN, handler) == "hello"
    sem = provider._llm_call_slot()
    assert not sem.locked(), "the one slot is free again after the call"


async def test_a_524_keeps_the_short_budget(waits: list[float]) -> None:
    """524: Cloudflare gave up on a slow request; the same request is as slow next time."""
    handler, calls = _replies(_cloudflare(524))
    with pytest.raises(httpx.HTTPStatusError):
        await _chat(PLAIN, handler)
    assert len(calls) == 4 and waits == []


# --- which 429s are a used-up quota ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("body", "headers", "used_up"), [
    ({"error": {"type": "insufficient_quota", "code": "insufficient_quota"}}, {}, True),
    ({"error": {"type": "exceeded_current_quota_error", "message": "x"}}, {}, True),
    ({"error": {"type": "api_error", "message": "you (me) have reached your monthly usage limit, upgrade"}}, {}, True),
    ({"error": {"type": "too_many_requests_error", "code": "token_quota_exceeded"}}, {}, False),
    ({"error": {"type": "rate_limit_reached_error", "message": "quota"}}, {}, False),
    ({"error": {"type": "engine_overloaded_error"}}, {}, False),
    ({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota"}}, {}, False),
    ({"error": {"type": "insufficient_quota"}}, {"Retry-After": "20"}, False),
    ({"error": "you have reached your monthly usage limit"}, {}, True),
    ({"error": {"message": "requests per minute exceeded; daily limit 1000"}}, {}, False),
])
def test_which_429_is_a_used_up_quota(body: dict, headers: dict, used_up: bool) -> None:
    assert provider._is_exhausted_quota(httpx.Response(429, json=body, headers=headers)) is used_up


def test_a_used_up_quota_opens_the_fallback_circuit_at_once() -> None:
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    quota = httpx.Response(429, request=req, json={"error": {"type": "insufficient_quota"}})
    limit = httpx.Response(429, request=req, json={"error": {"type": "rate_limit_exceeded"}})
    assert provider._is_fatal_provider_error(httpx.HTTPStatusError("q", request=req, response=quota))
    assert not provider._is_fatal_provider_error(httpx.HTTPStatusError("r", request=req, response=limit))


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


# --- a step that times out on every try is not a passing problem -----------------------------------------------------


async def test_every_try_timing_out_is_said_on_the_error_and_sorted_as_a_setup_problem(waits: list[float]) -> None:
    from core import crash_kind

    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(httpx.ReadTimeout) as e:
        await _chat(PLAIN, handler, node="implement")
    assert len(calls) >= 2 and e.value.fi_timeouts["all"] is True and e.value.fi_timeouts["tries"] == len(calls)
    failure = crash_kind.classify(e.value, node="implement", provider="openai", model="gpt-x")
    assert failure.kind == "setup" and "`implement`" in failure.say and "gpt-x" in failure.say
    assert "http_timeout_s" in failure.do


async def test_a_timeout_among_other_failures_stays_a_passing_problem(waits: list[float]) -> None:
    from core import crash_kind

    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return _cloudflare(503)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(httpx.ReadTimeout) as e:
        await _chat(PLAIN, handler, node="implement")
    assert e.value.fi_timeouts["all"] is False
    assert crash_kind.classify(e.value, node="implement", provider="openai", model="gpt-x").kind == "transient"
