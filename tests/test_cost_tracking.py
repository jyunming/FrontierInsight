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
async def test_http_chat_with_no_usage_block_estimates_tokens() -> None:
    """Some Ollama versions and a few proxies omit ``usage`` from the
    response body. The cost log used to record null tokens, which made
    the chart useless. Now the client falls back to a char-based
    token estimate and flags the row with ``estimated: True`` so the
    chart can render it differently."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(_make_http_endpoint(), http=http)
    await client.chat([{"role": "user", "content": "ping"}])
    assert client.last_usage is not None
    assert client.last_usage.get("estimated") is True
    assert client.last_usage["prompt_tokens"] >= 1
    assert client.last_usage["completion_tokens"] >= 0
    await client.aclose()


@pytest.mark.asyncio
async def test_last_usage_reset_between_calls() -> None:
    """Call 1 returns usage from the server (no ``estimated`` flag);
    call 2 doesn't (estimated fallback fires). The two rows must NOT
    bleed: call 2's usage is the estimate, never the stale call 1 value."""
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
    assert client.last_usage["prompt_tokens"] == 10
    assert not client.last_usage.get("estimated"), "real usage should not be flagged estimated"
    await client.chat([{"role": "user", "content": "2"}])
    assert client.last_usage is not None
    assert client.last_usage.get("estimated") is True, (
        "call 2 has no server-side usage → must surface the estimated fallback"
    )
    assert client.last_usage["prompt_tokens"] != 10, (
        "estimated tokens for call 2 must not be the stale value from call 1"
    )
    await client.aclose()


def test_aggregate_cost_rows_sums_per_node_and_model() -> None:
    """Roll-up over a small jsonl excerpt: totals, by_node, by_model
    all match the hand-summed values. Estimated rows surface in
    ``estimated_rows`` so the cost tool can mark the bar accordingly."""
    from core.engine import _aggregate_cost_rows
    rows = [
        {"node": "ideate", "model": "gpt-4o",
         "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                   "total_tokens": 150},
         "cost_usd": 0.0008},
        {"node": "analyze", "model": "gpt-4o",
         "usage": {"prompt_tokens": 200, "completion_tokens": 100,
                   "total_tokens": 300, "estimated": True},
         "cost_usd": None},
        {"node": "ideate", "model": "claude-3-5-sonnet",
         "usage": {"prompt_tokens": 80, "completion_tokens": 40,
                   "total_tokens": 120},
         "cost_usd": 0.0006},
    ]
    summary = _aggregate_cost_rows(rows)
    assert summary["total_requests"] == 3
    assert summary["total_prompt_tokens"] == 380
    assert summary["total_completion_tokens"] == 190
    assert summary["total_tokens"] == 570
    assert summary["total_cost_usd"] == pytest.approx(0.0014, abs=1e-6)
    assert summary["estimated_rows"] == 1
    assert summary["by_node"]["ideate"]["requests"] == 2
    assert summary["by_node"]["analyze"]["estimated_rows"] == 1
    assert summary["by_model"]["gpt-4o"]["requests"] == 2


def test_aggregate_cost_rows_skips_ensemble_breadcrumbs() -> None:
    """Ensemble breadcrumb rows mirror per-call rows already in the
    log. Counting them again would double-bill the quest."""
    from core.engine import _aggregate_cost_rows
    rows = [
        {"node": "ideate.ensemble[m1]", "model": "m1",
         "usage": {"prompt_tokens": 50, "completion_tokens": 25,
                   "total_tokens": 75},
         "cost_usd": 0.0001},
        # Breadcrumb shadow row — should be skipped.
        {"node": "ideate.ensemble[m1]", "model": "m1",
         "ensemble": True, "role": "fanout", "ok": True},
    ]
    summary = _aggregate_cost_rows(rows)
    assert summary["total_requests"] == 1
    assert summary["total_tokens"] == 75


def test_aggregate_cost_rows_no_pricing_data_surfaces_null_cost() -> None:
    """Quests routed through CLI/bridge transports have no pricing
    rows. The summary must surface ``None`` instead of a misleading
    ``0.00`` so the UI can distinguish 'free' from 'unknown'."""
    from core.engine import _aggregate_cost_rows
    rows = [
        {"node": "ideate", "model": "vscode_extension",
         "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                   "total_tokens": 150, "estimated": True},
         "cost_usd": None},
    ]
    summary = _aggregate_cost_rows(rows)
    assert summary["total_cost_usd"] is None
    assert summary["by_node"]["ideate"]["cost_usd"] is None
