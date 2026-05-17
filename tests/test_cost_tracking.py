"""Tests for the cost-tracking instrumentation.

* :func:`core.provider.estimate_cost_usd` — pricing-table lookup.
* :class:`core.provider.LLMClient.last_usage` — populated by the
  HTTP transport when the upstream returned a ``usage`` block.
* ``Engine._log_chat_cost`` — appends to ``<quest_root>/.fi/cost.jsonl``
  with the expected shape.

No real LLM calls; the HTTP transport is mocked via
``httpx.MockTransport``."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from core.provider import (
    LLMClient, MODEL_PRICING, ResolvedEndpoint, estimate_cost_usd,
)


# ---------------------------------------------------------------------------
# estimate_cost_usd
# ---------------------------------------------------------------------------


def test_estimate_cost_usd_known_model() -> None:
    """gpt-4o is in the pricing table; cost should compute deterministically."""
    cost = estimate_cost_usd("gpt-4o", 1000, 500)
    expected = 1000 * MODEL_PRICING["gpt-4o"]["prompt_per_1k"] / 1000.0 + \
               500 * MODEL_PRICING["gpt-4o"]["completion_per_1k"] / 1000.0
    assert cost == pytest.approx(expected)


def test_estimate_cost_usd_versioned_model_falls_back_to_substring() -> None:
    """LLM models often carry version suffixes (claude-opus-4-7-20251201).
    The estimator substring-matches against the base model so we still
    price these correctly."""
    cost = estimate_cost_usd("claude-opus-4-7-20251201", 1000, 500)
    expected = 1000 * MODEL_PRICING["claude-opus-4-7"]["prompt_per_1k"] / 1000.0 + \
               500 * MODEL_PRICING["claude-opus-4-7"]["completion_per_1k"] / 1000.0
    assert cost == pytest.approx(expected)


def test_estimate_cost_usd_longest_key_wins() -> None:
    """Both ``gpt-4o`` and ``gpt-4o-mini`` are in the table — a
    request for ``gpt-4o-mini`` must NOT silently match ``gpt-4o``.
    Sorted-longest-first iteration guards this."""
    mini = estimate_cost_usd("gpt-4o-mini", 1000, 0)
    full = estimate_cost_usd("gpt-4o", 1000, 0)
    assert mini != full
    assert mini < full  # mini is cheaper


def test_estimate_cost_usd_unknown_returns_none() -> None:
    """An unknown model returns None so the caller can log "no cost
    data" rather than fabricating a zero."""
    assert estimate_cost_usd("totally-made-up-llm", 100, 50) is None


def test_estimate_cost_usd_empty_model_returns_none() -> None:
    """CLI / vscode_bridge transports may set last_model="" — must
    not crash and must not match any pricing row."""
    assert estimate_cost_usd("", 100, 50) is None


def test_estimate_cost_usd_local_models_are_free() -> None:
    """Ollama local models are priced at 0.0."""
    cost = estimate_cost_usd("llama3.1:8b", 10000, 5000)
    assert cost == 0.0


# ---------------------------------------------------------------------------
# LLMClient.last_usage — populated by HTTP transport
# ---------------------------------------------------------------------------


def _make_http_endpoint() -> ResolvedEndpoint:
    return ResolvedEndpoint(
        transport="http",
        base_url="http://fake.local/v1",
        api_key="sk-test",
        model="gpt-4o",
    )


@pytest.mark.asyncio
async def test_http_chat_populates_last_usage() -> None:
    """When the upstream returns a usage block, LLMClient surfaces it
    on last_usage for the Engine to log."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi there"}}],
                "usage": {
                    "prompt_tokens": 42, "completion_tokens": 17,
                    "total_tokens": 59,
                },
                "model": "gpt-4o-2024-08-06",
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(_make_http_endpoint(), http=http)
    out = await client.chat([{"role": "user", "content": "ping"}])
    assert out == "hi there"
    assert client.last_usage == {
        "prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59,
    }
    assert client.last_model == "gpt-4o-2024-08-06"
    await client.aclose()


@pytest.mark.asyncio
async def test_http_chat_with_no_usage_block_leaves_last_usage_none() -> None:
    """Some Ollama versions and a few proxies omit ``usage`` from the
    response body. last_usage must stay None so the cost-logger
    records "tokens unknown" rather than fabricating zeros."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(_make_http_endpoint(), http=http)
    await client.chat([{"role": "user", "content": "ping"}])
    assert client.last_usage is None
    await client.aclose()


@pytest.mark.asyncio
async def test_last_usage_reset_between_calls() -> None:
    """Call 1 returns usage; call 2 doesn't. last_usage must be
    None after call 2, not the stale value from call 1."""
    sequence = iter([
        {
            "choices": [{"message": {"content": "first"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15},
        },
        {
            "choices": [{"message": {"content": "second"}}],
            # no usage
        },
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(sequence))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(_make_http_endpoint(), http=http)
    await client.chat([{"role": "user", "content": "1"}])
    assert client.last_usage is not None
    await client.chat([{"role": "user", "content": "2"}])
    assert client.last_usage is None, (
        "last_usage from call 1 leaked into call 2"
    )
    await client.aclose()
