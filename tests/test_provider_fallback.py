"""Unit tests for the provider fallback chain + circuit breaker (PR-5).

These exercise :class:`FallbackLLMClient` directly with fake provider clients —
no real endpoints, proxies, or sockets. The wiring into the Engine (config
field ``provider.fallback`` → lazy factories → proxy release) is covered by the
fleet/engine integration tests; here we pin the chain semantics.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from core.provider import (
    FallbackLLMClient,
    _CliTransientError,
    _is_fatal_provider_error,
)


class _FakeClient:
    """Minimal stand-in for LLMClient: records call count, optionally raises."""

    def __init__(self, name: str, *, error: BaseException | None = None) -> None:
        self.name = name
        self._error = error
        self.calls = 0
        self.closed = False
        self.last_model = f"{name}-model"
        self.last_usage = {"provider": name}

    async def chat(self, messages, **kw):  # noqa: ANN001, ANN003
        self.calls += 1
        if self._error is not None:
            raise self._error
        return f"{self.name}-ok"

    async def aclose(self) -> None:
        self.closed = True


def _factory_for(client: _FakeClient, counter: list[int]):
    async def _make():
        counter[0] += 1
        return client

    return _make


async def test_primary_success_never_touches_fallback():
    primary = _FakeClient("primary")
    fb = _FakeClient("fallback")
    built = [0]
    client = FallbackLLMClient(primary, [("fallback", _factory_for(fb, built))])

    out = await client.chat([{"role": "user", "content": "hi"}])

    assert out == "primary-ok"
    assert fb.calls == 0
    assert built[0] == 0, "fallback must be built lazily — not until needed"
    assert client.last_model == "primary-model"


async def test_falls_back_when_primary_fails():
    primary = _FakeClient("primary", error=_CliTransientError("network blip"))
    fb = _FakeClient("fallback")
    built = [0]
    client = FallbackLLMClient(primary, [("fallback", _factory_for(fb, built))])

    out = await client.chat([{"role": "user", "content": "hi"}])

    assert out == "fallback-ok"
    assert primary.calls == 1 and fb.calls == 1
    assert built[0] == 1
    assert client.built_fallback_providers == ["fallback"]
    # Cost fields reflect the provider that actually served the call.
    assert client.last_model == "fallback-model"
    assert client.last_usage == {"provider": "fallback"}


async def test_circuit_breaker_opens_and_skips_dead_primary():
    """After the primary trips (threshold=2 non-fatal failures), later calls
    jump straight to the fallback without re-hitting the dead primary."""
    primary = _FakeClient("primary", error=_CliTransientError("blip"))
    fb = _FakeClient("fallback")
    client = FallbackLLMClient(
        primary, [("fallback", _factory_for(fb, [0]))], breaker_threshold=2,
    )

    # Two calls: each falls back; the second failure trips the primary.
    assert await client.chat([{"role": "user", "content": "1"}]) == "fallback-ok"
    assert await client.chat([{"role": "user", "content": "2"}]) == "fallback-ok"
    assert primary.calls == 2  # tripped after the 2nd failure

    # Third call must NOT touch the primary again — circuit is open.
    assert await client.chat([{"role": "user", "content": "3"}]) == "fallback-ok"
    assert primary.calls == 2, "open circuit must skip the dead primary"
    assert fb.calls == 3


async def test_auth_error_trips_breaker_immediately():
    """An auth/quota error opens the circuit on the FIRST failure (it won't
    clear within the run), even with a threshold above 1."""
    primary = _FakeClient(
        "primary", error=_CliTransientError("Session limit reached"),
    )
    fb = _FakeClient("fallback")
    client = FallbackLLMClient(
        primary, [("fallback", _factory_for(fb, [0]))], breaker_threshold=3,
    )

    assert await client.chat([{"role": "user", "content": "1"}]) == "fallback-ok"
    assert await client.chat([{"role": "user", "content": "2"}]) == "fallback-ok"
    assert primary.calls == 1, "fatal error must trip after a single failure"


async def test_all_providers_fail_raises_last_error_with_note():
    boom_a = _CliTransientError("primary down")
    boom_b = _CliTransientError("fallback down too")
    primary = _FakeClient("primary", error=boom_a)
    fb = _FakeClient("fallback", error=boom_b)
    client = FallbackLLMClient(primary, [("fallback", _factory_for(fb, [0]))])

    with pytest.raises(_CliTransientError) as ei:
        await client.chat([{"role": "user", "content": "x"}])
    # The last provider's error surfaces, annotated with the exhausted chain.
    assert "fallback down too" in str(ei.value)
    notes = getattr(ei.value, "__notes__", [])
    assert any("all providers exhausted" in n for n in notes)


async def test_cancellation_propagates_without_falling_back():
    """CancelledError is not a provider failure — it must propagate and never
    trigger a fallback attempt."""
    primary = _FakeClient("primary", error=asyncio.CancelledError())
    fb = _FakeClient("fallback")
    client = FallbackLLMClient(primary, [("fallback", _factory_for(fb, [0]))])

    with pytest.raises(asyncio.CancelledError):
        await client.chat([{"role": "user", "content": "x"}])
    assert fb.calls == 0, "must not fall back on cancellation"


async def test_aclose_closes_built_clients_only():
    primary = _FakeClient("primary")
    fb = _FakeClient("fallback")
    client = FallbackLLMClient(primary, [("fallback", _factory_for(fb, [0]))])

    # Fallback never built (primary succeeds), so only the primary is closed.
    await client.chat([{"role": "user", "content": "x"}])
    await client.aclose()
    assert primary.closed is True
    assert fb.closed is False  # never materialised → nothing to close


def test_is_fatal_provider_error_classification():
    assert _is_fatal_provider_error(_CliTransientError("out of credits")) is True
    assert _is_fatal_provider_error(_CliTransientError("temporary blip")) is False
    resp = httpx.Response(status_code=401, request=httpx.Request("GET", "http://x"))
    assert _is_fatal_provider_error(
        httpx.HTTPStatusError("unauthorized", request=resp.request, response=resp)
    ) is True
    resp5 = httpx.Response(status_code=503, request=httpx.Request("GET", "http://x"))
    assert _is_fatal_provider_error(
        httpx.HTTPStatusError("busy", request=resp5.request, response=resp5)
    ) is False
