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


@pytest.mark.parametrize("model", [
    "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.5", "gpt-50",
    "claude-opus-4-70", "gpt-4o.1",
])
def test_estimate_cost_usd_a_newer_model_does_not_borrow_an_older_rate(model: str) -> None:
    """``gpt-5`` is a row; ``gpt-5.6-luna`` merely CONTAINS it. Pricing the newer
    model at the older one's rate put a made-up dollar figure in every cost log and
    made three different models look like they cost the same per token."""
    assert estimate_cost_usd(model, 1000, 1000) is None


@pytest.mark.parametrize("model, key", [
    ("gpt-5", "gpt-5"),
    ("gpt-5-2025-08-07", "gpt-5"),          # a dated snapshot of the same model
    ("openai/gpt-5", "gpt-5"),              # a provider prefix
    ("gpt-5-mini", "gpt-5-mini"),           # its own row, not gpt-5's
    ("claude-opus-4-7-20251201", "claude-opus-4-7"),
    ("gemini-2.5-pro", "gemini-2.5-pro"),   # a "." INSIDE the key still matches
])
def test_estimate_cost_usd_still_prices_the_models_the_table_names(model: str, key: str) -> None:
    rates = MODEL_PRICING[key]
    assert estimate_cost_usd(model, 1000, 1000) == pytest.approx(
        rates["prompt_per_1k"] + rates["completion_per_1k"])


def test_a_call_to_an_unpriced_model_logs_a_null_cost(tmp_path: Path) -> None:
    """The row the cost log gets, and so every total built from it: usage is kept,
    the dollar field is empty."""
    from core.provider import append_cost_row

    append_cost_row(
        tmp_path, node="cross_check", model="gpt-5.6-luna",
        usage={"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000},
    )
    row = json.loads((tmp_path / "cost.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["model"] == "gpt-5.6-luna"
    assert row["usage"]["total_tokens"] == 1000
    assert row["cost_usd"] is None


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


def _priced(node: str, model: str, cost: float) -> dict:
    return {"node": node, "model": model, "cost_usd": cost,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}


def _unpriced(node: str, model: str) -> dict:
    """What ``append_cost_row`` writes for a model the price table does not list:
    the tokens are there, the dollar field is null."""
    return {"node": node, "model": model, "cost_usd": None,
            "usage": {"prompt_tokens": 400, "completion_tokens": 100, "total_tokens": 500}}


def test_aggregate_cost_rows_all_priced_is_not_partial() -> None:
    from core.engine import _aggregate_cost_rows
    summary = _aggregate_cost_rows([
        _priced("ideate", "gpt-4o", 0.0008),
        _priced("write", "gpt-4o", 0.0012),
        _priced("write", "llama3.1:8b", 0.0),  # a free local model is priced at zero
    ])
    assert summary["total_cost_usd"] == pytest.approx(0.002)
    assert summary["total_cost_usd_partial"] is False
    assert summary["unpriced_requests"] == 0
    assert all(b["unpriced_requests"] == 0
               for b in list(summary["by_node"].values()) + list(summary["by_model"].values()))


def test_aggregate_cost_rows_none_priced_is_null_not_partial() -> None:
    """Nothing to be a lower bound of: the total stays null (unknown), as before, and
    every call is counted as unpriced."""
    from core.engine import _aggregate_cost_rows
    summary = _aggregate_cost_rows([
        _unpriced("ideate", "gpt-5.6-luna"),
        _unpriced("write", "gpt-5.6-luna"),
        {"node": "speech", "model": "", "usage": None, "cost_usd": None},
    ])
    assert summary["total_cost_usd"] is None
    assert summary["total_cost_usd_partial"] is False
    assert summary["unpriced_requests"] == summary["total_requests"] == 3
    assert summary["by_node"]["write"]["cost_usd"] is None
    assert summary["by_node"]["write"]["unpriced_requests"] == 1
    assert summary["by_model"]["gpt-5.6-luna"]["unpriced_requests"] == 2


def test_aggregate_cost_rows_mixed_total_is_a_lower_bound_and_says_so() -> None:
    """A quest that mixes priced and unpriced models: the dollar total is the priced
    calls only, so it is flagged partial and the calls it leaves out are counted,
    in total and in each node and model bucket. The token counts stay complete."""
    from core.engine import _aggregate_cost_rows
    rows = [
        _priced("ideate", "gpt-4o", 0.0008),
        _priced("write", "gpt-4o", 0.0012),
        _unpriced("write", "gpt-5.6-luna"),
        _unpriced("review", "gpt-5.6-luna"),
        # A breadcrumb mirrors a call already in the log: neither a request nor unpriced.
        {"node": "ideate.ensemble[m1]", "model": "m1", "ensemble": True,
         "role": "fanout", "ok": True},
    ]
    summary = _aggregate_cost_rows(rows)
    assert summary["total_cost_usd"] == pytest.approx(0.002)   # exactly as before
    assert summary["total_cost_usd_partial"] is True
    assert summary["total_requests"] == 4
    assert summary["unpriced_requests"] == 2
    assert summary["total_tokens"] == 150 + 150 + 500 + 500  # tokens are complete
    by_node = summary["by_node"]
    assert by_node["ideate"]["unpriced_requests"] == 0
    assert by_node["ideate"]["cost_usd"] == pytest.approx(0.0008)
    assert by_node["write"]["unpriced_requests"] == 1
    assert by_node["write"]["cost_usd"] == pytest.approx(0.0012)
    assert by_node["review"]["unpriced_requests"] == 1
    assert by_node["review"]["cost_usd"] == 0.0   # a priced-run node with no priced call
    assert "ideate.ensemble[m1]" not in by_node
    by_model = summary["by_model"]
    assert by_model["gpt-4o"]["unpriced_requests"] == 0
    assert by_model["gpt-5.6-luna"]["unpriced_requests"] == 2
    assert by_model["gpt-5.6-luna"]["requests"] == 2


def test_write_cost_summary_carries_the_partial_flag(tmp_path: Path) -> None:
    """The file quest finalization writes is the same roll-up, flag included."""
    from core.engine import write_cost_summary
    (tmp_path / "cost.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [
            _priced("ideate", "gpt-4o", 0.001),
            _unpriced("write", "gpt-5.6-luna"),
        ]) + "\n",
        encoding="utf-8",
    )
    write_cost_summary(tmp_path)
    on_disk = json.loads((tmp_path / "cost.summary.json").read_text(encoding="utf-8"))
    assert on_disk["total_cost_usd"] == pytest.approx(0.001)
    assert on_disk["total_cost_usd_partial"] is True
    assert on_disk["unpriced_requests"] == 1
    assert on_disk["by_node"]["write"]["unpriced_requests"] == 1
