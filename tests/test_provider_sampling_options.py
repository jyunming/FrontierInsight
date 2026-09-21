"""provider.fixed_temperature and provider.extra_body: models that fix their own sampling (Moonshot's Kimi)."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from core.config import ProviderConfig
from core.provider import LLMClient, ResolvedEndpoint, resolve_endpoint

KIMI = dict(name="openai", base_url="https://api.moonshot.ai/v1", api_key_env="MOONSHOT_API_KEY", model="kimi-k2.6")


def _bodies(endpoint: ResolvedEndpoint, *calls: dict) -> list[dict]:
    """The request bodies an ``LLMClient`` over ``endpoint`` sends for each of ``calls`` (kwargs of ``chat``)."""
    sent: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    async def run() -> None:
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
        try:
            client = LLMClient(endpoint, http=http)
            for kwargs in calls:
                await client.chat([{"role": "user", "content": "hi"}], **kwargs)
        finally:
            await http.aclose()

    import asyncio

    asyncio.run(run())
    return sent


def test_without_the_options_the_request_is_what_it_always_was() -> None:
    (body,) = _bodies(resolve_endpoint(ProviderConfig(**KIMI)), {"temperature": 0.0})
    assert body["temperature"] == 0.0 and "thinking" not in body


def test_a_fixed_temperature_replaces_the_one_each_node_asks_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    endpoint = resolve_endpoint(ProviderConfig(**KIMI, fixed_temperature=0.6))
    assert endpoint.fixed_temperature == 0.6
    bodies = _bodies(endpoint, {"temperature": 0.0}, {"temperature": 0.2}, {})
    assert [b["temperature"] for b in bodies] == [0.6, 0.6, 0.6]


def test_a_fixed_temperature_of_zero_is_sent_and_not_mistaken_for_unset() -> None:
    endpoint = resolve_endpoint(ProviderConfig(**KIMI, fixed_temperature=0.0))
    assert [b["temperature"] for b in _bodies(endpoint, {"temperature": 0.2})] == [0.0]


def test_extra_body_fields_are_merged_into_every_request_and_a_per_call_extra_wins() -> None:
    endpoint = resolve_endpoint(ProviderConfig(**KIMI, extra_body={"thinking": {"type": "disabled"}, "top_p": 0.9}))
    first, second = _bodies(endpoint, {}, {"extra": {"top_p": 0.5}})
    assert first["thinking"] == {"type": "disabled"} and first["top_p"] == 0.9
    assert second["thinking"] == {"type": "disabled"} and second["top_p"] == 0.5


def test_the_configuration_checks_the_temperature_and_keeps_the_body_fields() -> None:
    assert ProviderConfig().fixed_temperature is None and ProviderConfig().extra_body == {}
    for bad in (-0.1, 2.5):
        with pytest.raises(ValidationError):
            ProviderConfig(fixed_temperature=bad)
    cfg = ProviderConfig.model_validate({"name": "openai", "fixed_temperature": 0.6, "extra_body": {"thinking": {"type": "disabled"}}})
    assert cfg.extra_body == {"thinking": {"type": "disabled"}}


def test_a_fallback_provider_does_not_inherit_the_primary_models_sampling(tmp_path) -> None:
    """The fallback factory resets them: they belong to the primary's model."""
    from core.config import Config, KnowledgeConfig, OutputConfig
    from core.engine import Engine

    cfg = Config(
        topic="t", title="t", knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
        provider=ProviderConfig(**KIMI, fixed_temperature=0.6, extra_body={"thinking": {"type": "disabled"}}, fallback=["ollama"]),
    )
    engine = Engine(cfg)
    derived = engine.config.provider.model_copy(update={
        "name": "ollama", "model": None, "base_url": None, "api_key_env": None,
        "node_model_fallbacks": {}, "fallback": [], "fixed_temperature": None, "extra_body": {},
    })
    endpoint = resolve_endpoint(derived)
    assert endpoint.fixed_temperature is None and endpoint.extra_body == {}
    import inspect

    assert '"fixed_temperature": None' in inspect.getsource(engine._make_fallback_factory)
