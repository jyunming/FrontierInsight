"""HTTP hardening: per-node read timeout + prompt-size guard (PR-8)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.provider import LLMClient, ResolvedEndpoint


def _fake_response(payload: dict) -> MagicMock:
    r = MagicMock()
    r.json = MagicMock(return_value=payload)
    r.status_code = 200
    r.raise_for_status = MagicMock()
    return r


def _http_returning(content: str):
    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=_fake_response(
            {"choices": [{"message": {"content": content}}]}
        )
    )
    return fake_http


async def test_per_node_http_timeout_applied_to_post():
    """A heavy node gets its per-node read timeout; an unmapped node falls
    back to the base client timeout."""
    ep = ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k")
    fake_http = _http_returning("ok")
    client = LLMClient(
        ep, http=fake_http, timeout_s=120.0,
        node_http_timeout_s={"implement": 900.0},
    )

    await client.chat([{"role": "user", "content": "hi"}], node="implement")
    assert fake_http.post.call_args.kwargs["timeout"] == 900.0

    await client.chat([{"role": "user", "content": "hi"}], node="clarify")
    assert fake_http.post.call_args.kwargs["timeout"] == 120.0


async def test_prompt_trim_disabled_by_default():
    ep = ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k")
    fake_http = _http_returning("ok")
    client = LLMClient(ep, http=fake_http)  # max_prompt_chars defaults to 0

    huge = "x" * 500_000
    await client.chat([{"role": "user", "content": huge}], node="write")
    sent = fake_http.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert sent == huge, "trim must be off by default — content untouched"


async def test_prompt_trim_caps_oversized_prompt():
    ep = ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k")
    fake_http = _http_returning("ok")
    client = LLMClient(ep, http=fake_http, max_prompt_chars=1000)

    head, tail = "HEAD" * 100, "TAIL" * 100
    huge = head + ("m" * 50_000) + tail
    await client.chat([{"role": "user", "content": huge}], node="analyze")

    sent = fake_http.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert len(sent) <= 1000, f"trimmed prompt still {len(sent)} chars"
    # Head and tail (task framing + trailing instruction) survive; the marker
    # documents the cut.
    assert sent.startswith("HEAD")
    assert sent.endswith("TAIL")
    assert "trimmed to fit" in sent


async def test_prompt_trim_no_op_when_under_cap():
    ep = ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k")
    fake_http = _http_returning("ok")
    client = LLMClient(ep, http=fake_http, max_prompt_chars=10_000)

    small = "just a normal prompt"
    await client.chat([{"role": "user", "content": small}], node="ideate")
    sent = fake_http.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert sent == small
