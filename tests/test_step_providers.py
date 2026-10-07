"""provider.node_providers: a step on another provider (its own address, key, sampling and model), the rest on the main
one. Only fake HTTP (httpx.MockTransport) is used; no real model or service is called."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from core import attempt_records as ar
from core import oracle_review as orv
from core import provider_readiness as pr
from core.config import Config, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from launch import parse_args
from core.provider import (
    LLMClient, missing_step_api_keys, node_provider_for, step_provider_config, step_provider_lines,
)

MAIN_URL = "https://main.example/v1"
STEP_URL = "https://step.example/v1"
KIMI = {"name": "openai", "model": "kimi-k3", "base_url": STEP_URL, "api_key_env": "STEP_TEST_KEY",
        "fixed_temperature": 1.0, "extra_body": {"thinking": {"type": "enabled"}}}


def _provider(**over: Any) -> ProviderConfig:
    base = dict(name="openai", model="main-model", base_url=MAIN_URL, api_key_env="MAIN_TEST_KEY",
                node_providers={"implement": KIMI, "oracle_review": KIMI})
    base.update(over)
    return ProviderConfig.model_validate(base)


def _config(tmp_path: Path, **over: Any) -> Config:
    return Config(topic="step providers", title="sp", knowledge=KnowledgeConfig(enabled=False),
                  output=OutputConfig(output_dir=tmp_path / "out"), provider=_provider(**over))


# --- the configuration --------------------------------------------------------------------------------------------------


def test_a_step_provider_is_read_and_checked_like_the_main_provider_block() -> None:
    p = _provider()
    entry = p.node_providers["implement"]
    assert (entry.name, entry.model, entry.base_url, entry.fixed_temperature) == ("openai", "kimi-k3", STEP_URL, 1.0)
    assert ProviderConfig().node_providers is None  # nothing set: nothing changes


@pytest.mark.parametrize("bad", [
    {"name": "openai"},                                   # a model is required
    {"name": "openai", "model": "  "},
    {"name": "nope", "model": "m"},                      # not a provider FI knows
    {"name": "openai", "model": "m", "fixed_temperature": 3},
    {"name": "openai", "model": "m", "extra_body": {"stream": True}},
    {"name": "openai", "model": "m", "reasoning_effort": "huge"},
    {"name": "openai", "model": "m", "fixed_temprature": 1},   # a typo is an error, not silently ignored
])
def test_a_bad_step_provider_fails_at_load(bad: dict) -> None:
    with pytest.raises(ValidationError):
        _provider(node_providers={"implement": bad})


def test_the_steps_are_matched_by_the_key_rules_of_node_models() -> None:
    steps = {"implement": "A", "review_panel": "B", "review_panel.statistician": "C"}
    assert node_provider_for(steps, "implement") == ("implement", "A")
    assert node_provider_for(steps, "review_panel.methodologist") == ("review_panel", "B")
    assert node_provider_for(steps, "review_panel.statistician") == ("review_panel.statistician", "C")
    assert node_provider_for(steps, "implement_oracle") is None   # no prefix matching without a dot
    assert node_provider_for(steps, None) is None and node_provider_for(None, "implement") is None
    assert node_provider_for({"oracle_review": "R"}, "oracle_review.recompute") == ("oracle_review", "R")


def test_the_derived_provider_keeps_nothing_of_the_main_models_own_settings() -> None:
    main = _provider(fixed_temperature=0.6, extra_body={"top_p": 0.9}, reasoning_effort="high",
                     node_models={"write": "w"}, fallback=["ollama"])
    derived = step_provider_config(main, main.node_providers["implement"])
    assert (derived.name, derived.model, derived.base_url, derived.api_key_env) == ("openai", "kimi-k3", STEP_URL,
                                                                                   "STEP_TEST_KEY")
    assert derived.fixed_temperature == 1.0 and derived.extra_body == {"thinking": {"type": "enabled"}}
    assert derived.reasoning_effort is None and derived.fallback == [] and derived.node_models is None
    assert derived.node_providers is None
    assert derived.http_timeout_s == main.http_timeout_s   # not about a model: the main provider's


def test_run_log_names_each_routed_step_in_plain_words() -> None:
    lines = step_provider_lines(_provider())
    assert lines == [f"the step `implement` uses kimi-k3 on openai ({STEP_URL})",
                     f"the step `oracle_review` uses kimi-k3 on openai ({STEP_URL})"]


# --- readiness ------------------------------------------------------------------------------------------------------------


def test_a_missing_key_for_a_steps_provider_is_said_plainly_and_names_the_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEP_TEST_KEY", raising=False)
    (first, second) = missing_step_api_keys(_provider())
    assert first.startswith("the step `implement` runs on its own provider: ") and "STEP_TEST_KEY" in first
    assert "oracle_review" in second
    monkeypatch.setenv("STEP_TEST_KEY", "k")
    assert missing_step_api_keys(_provider()) == []


def test_the_engine_stops_before_any_call_when_a_steps_key_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEP_TEST_KEY", raising=False)
    monkeypatch.setenv("MAIN_TEST_KEY", "k")
    engine = Engine(_config(tmp_path))
    with pytest.raises(RuntimeError, match="the step `implement` runs on its own provider.*Nothing was started"):
        asyncio.run(engine._connect_llm())


def test_the_readiness_check_covers_each_steps_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEP_TEST_KEY", raising=False)
    found = asyncio.run(pr.check_steps(_provider()))
    assert [(step, r.state, r.blocked) for step, r in found] == [("implement", "no_key", True),
                                                                 ("oracle_review", "no_key", True)]
    assert "STEP_TEST_KEY is not set" in found[0][1].sentence
    monkeypatch.setenv("STEP_TEST_KEY", "k")
    assert [r.state for _s, r in asyncio.run(pr.check_steps(_provider()))] == ["key_present", "key_present"]


@pytest.mark.asyncio
async def test_the_launch_refuses_a_missing_step_key_like_a_missing_main_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import launch

    monkeypatch.delenv("STEP_TEST_KEY", raising=False)
    monkeypatch.setenv("MAIN_TEST_KEY", "k")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("topic: a topic\nprovider:\n  name: openai\n  model: m\n  api_key_env: MAIN_TEST_KEY\n"
                   "  node_providers:\n    implement: {name: openai, model: kimi-k3, api_key_env: STEP_TEST_KEY}\n"
                   "output:\n  output_dir: " + json.dumps(str(tmp_path / "o")) + "\n", encoding="utf-8")
    assert await launch.main_async(parse_args(["--config", str(cfg), "--no-axon-sidecar"])) == 2
    assert "the step `implement` runs on its own provider" in capsys.readouterr().err


# --- the engine ---------------------------------------------------------------------------------------------------------


class _Fake:
    """Two fake HTTP services; every request body each one gets is kept."""

    def __init__(self) -> None:
        self.bodies: dict[str, list[dict]] = {MAIN_URL: [], STEP_URL: []}
        self.keys: dict[str, set[str]] = {MAIN_URL: set(), STEP_URL: set()}

    def handler_for(self, base: str):  # noqa: ANN201
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.bodies[base].append(body)
            self.keys[base].add(request.headers.get("authorization", ""))
            if body.get("stream"):
                text = ("data: " + json.dumps({"model": body["model"], "choices": [
                    {"delta": {"content": f"from {base}"}, "finish_reason": "stop"}]}) + "\n\ndata: [DONE]\n\n")
                return httpx.Response(200, content=text.encode())
            return httpx.Response(200, json={"model": body["model"], "choices": [{"message": {
                "content": f"from {base}"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1,
                                                                                  "completion_tokens": 1}})
        return handler


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _Fake:
    fake = _Fake()
    monkeypatch.setenv("MAIN_TEST_KEY", "main-secret")
    monkeypatch.setenv("STEP_TEST_KEY", "step-secret")
    real = LLMClient

    def factory(endpoint, **kw):  # noqa: ANN001, ANN202
        base = endpoint.base_url.rstrip("/")
        http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler_for(base)), timeout=30.0)
        return real(endpoint, http=http, **{k: v for k, v in kw.items() if k != "http"})

    monkeypatch.setattr("core.engine.LLMClient", factory)
    return fake


async def _connected(cfg: Config) -> Engine:
    engine = Engine(cfg)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    await engine._connect_llm()
    return engine


@pytest.mark.asyncio
async def test_the_routed_step_goes_to_its_provider_and_only_it(tmp_path: Path, fake: _Fake) -> None:
    engine = await _connected(_config(tmp_path))
    try:
        assert await engine._chat("p", node="implement") == f"from {STEP_URL}"
        assert await engine._chat("p", node="implement") == f"from {STEP_URL}"
        assert await engine._chat("p", node="design") == f"from {MAIN_URL}"
        assert await engine._chat("p", node="analyze") == f"from {MAIN_URL}"
        assert await engine._chat("p", node="implement_oracle") == f"from {MAIN_URL}"   # a key, not a prefix
        assert await engine._chat_messages([{"role": "user", "content": "q"}], node="oracle_review") == f"from {STEP_URL}"
    finally:
        await engine._close_step_clients()
        await engine._client.aclose()
    assert [b["model"] for b in fake.bodies[STEP_URL]] == ["kimi-k3"] * 3
    assert [b["model"] for b in fake.bodies[MAIN_URL]] == ["main-model"] * 3
    # each service got only its own key
    assert fake.keys[STEP_URL] == {"Bearer step-secret"} and fake.keys[MAIN_URL] == {"Bearer main-secret"}


@pytest.mark.asyncio
async def test_the_steps_sampling_and_request_fields_go_only_to_its_provider(tmp_path: Path, fake: _Fake) -> None:
    cfg = _config(tmp_path, fixed_temperature=0.6, extra_body={"top_p": 0.9})
    engine = await _connected(cfg)
    try:
        await engine._chat("p", node="implement")
        await engine._chat("p", node="design")
    finally:
        await engine._close_step_clients()
        await engine._client.aclose()
    (step,), (main,) = fake.bodies[STEP_URL], fake.bodies[MAIN_URL]
    assert step["temperature"] == 1.0 and step["thinking"] == {"type": "enabled"} and "top_p" not in step
    assert main["temperature"] != 1.0 and main["top_p"] == 0.9 and "thinking" not in main
    assert main["temperature"] == 0.6   # the main provider's own fixed temperature, unchanged


@pytest.mark.asyncio
async def test_the_record_of_model_calls_names_the_real_provider_and_model_of_each_call(tmp_path: Path, fake: _Fake) -> None:
    cfg = _config(tmp_path)
    cfg.provider.node_providers["implement"] = cfg.provider.node_providers["implement"].model_copy(
        update={"name": "vllm", "api_key_env": None, "base_url": STEP_URL})
    engine = await _connected(cfg)
    try:
        await engine._chat("p", node="implement")
        await engine._chat("p", node="design")
    finally:
        await engine._close_step_clients()
        await engine._client.aclose()
    rows = {r["node"]: r for r in ar.read(engine.fi_dir, ar.MODEL_CALLS)}
    assert (rows["implement"]["provider"], rows["implement"]["served_model"], rows["implement"]["requested_model"]) == (
        "vllm", "kimi-k3", "kimi-k3")
    assert (rows["design"]["provider"], rows["design"]["served_model"], rows["design"]["requested_model"]) == (
        "openai", "main-model", "main-model")
    assert rows["implement"]["reported"] and rows["implement"]["reports_model"]
    prov = engine._chat_provenance("implement")
    assert (prov["provider"], prov["model"]) == ("vllm", "kimi-k3")


@pytest.mark.asyncio
async def test_concurrent_first_calls_build_the_steps_client_once(tmp_path: Path, fake: _Fake) -> None:
    engine = await _connected(_config(tmp_path))
    try:
        await asyncio.gather(*(engine._chat("p", node="implement") for _ in range(4)))
        assert len(engine.__dict__["_step_clients"]) == 1   # one client for the entry, however many calls
    finally:
        await engine._close_step_clients()
        await engine._client.aclose()
    assert len(fake.bodies[STEP_URL]) == 4 and fake.bodies[MAIN_URL] == []
    assert "_step_clients" not in engine.__dict__   # closed with the quest


@pytest.mark.asyncio
async def test_a_step_in_both_blocks_uses_node_providers_and_run_log_says_so_once(tmp_path: Path, fake: _Fake) -> None:
    cfg = _config(tmp_path, node_models={"implement": "other-model", "write": "w"})
    engine = Engine(cfg)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._log = MagicMock()
    await engine._connect_llm()
    try:
        assert engine._model_for_node("implement") == "kimi-k3"   # node_providers wins
        assert engine._model_for_node("write") == "w"
        warned = [c.args[0] % c.args[1:] for c in engine._log.warning.call_args_list]
        assert len([w for w in warned if "`implement` is in provider.node_models and provider.node_providers" in w]) == 1
        infos = [c.args[0] % c.args[1:] for c in engine._log.info.call_args_list]
        assert any(i == f"[provider] the step `implement` uses kimi-k3 on openai ({STEP_URL})" for i in infos)
        engine._say_step_providers()   # once per engine
        assert len(engine._log.warning.call_args_list) == len(warned)
    finally:
        await engine._close_step_clients()
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_a_resumed_quest_uses_the_setting_again(tmp_path: Path, fake: _Fake) -> None:
    cfg = _config(tmp_path)
    first = await _connected(cfg)
    try:
        await first._chat("p", node="implement")
    finally:
        await first._close_step_clients()
        await first._client.aclose()
    # a new engine on the same quest folder (what --resume builds) from the same config
    again = Engine(Config.model_validate(cfg.model_dump(mode="json")))
    again.fi_dir.mkdir(parents=True, exist_ok=True)
    await again._connect_llm()
    try:
        assert again._model_for_node("implement") == "kimi-k3"
        assert await again._chat("p", node="implement") == f"from {STEP_URL}"
    finally:
        await again._close_step_clients()
        await again._client.aclose()
    assert len(fake.bodies[STEP_URL]) == 2 and fake.bodies[MAIN_URL] == []


@pytest.mark.asyncio
async def test_the_step_provider_has_its_own_fallback_chain(tmp_path: Path, fake: _Fake, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tmp_path)
    cfg.provider.node_providers["implement"] = cfg.provider.node_providers["implement"].model_copy(
        update={"fallback": ["vllm"]})
    engine = await _connected(cfg)
    try:
        await engine._chat("p", node="implement")
        client, _reports = engine.__dict__["_step_clients"]["implement"]
        assert type(client).__name__ == "FallbackLLMClient"
        assert client._slots[0].label == "openai" and [s.label for s in client._slots[1:]] == ["vllm"]
    finally:
        await engine._close_step_clients()
        await engine._client.aclose()


# --- the second reading of the checks -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reviewer_on_another_provider_counts_as_a_different_model(tmp_path: Path, fake: _Fake) -> None:
    engine = await _connected(_config(tmp_path))
    try:
        await engine._chat("p", node="plan")
        await engine._chat("p", node="oracle_review")
        await engine._chat("p", node="oracle_review.recompute")
    finally:
        await engine._close_step_clients()
        await engine._client.aclose()
    calls = ar.read(engine.fi_dir, ar.MODEL_CALLS)
    reading = next(r for r in calls if r["node"] == "oracle_review")
    record = {"lines": ["", "- read"], "verdicts": [{"name": "x", "appropriate": True}], "call_id": reading["call_id"]}
    assert [(r["node"], r["served_model"]) for r in calls] == [("plan", "main-model"), ("oracle_review", "kimi-k3"),
                                                              ("oracle_review.recompute", "kimi-k3")]
    assert orv.independence_gaps(record, calls, configured=bool(engine._model_for_node(orv.NODE))) == []
    assert not orv.same_model("kimi-k3", "main-model")
    assert engine._model_for_node("oracle_review.recompute") == "kimi-k3"
    # the same model on the main provider is still the same model
    assert "wrote them and read them again" in orv.independence_gaps(
        record, [{**calls[0], "served_model": "kimi-k3"}, reading], configured=True)[0]


@pytest.mark.asyncio
async def test_the_launch_stops_when_a_steps_server_does_not_answer(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import launch

    monkeypatch.setenv("STEP_TEST_KEY", "k")
    monkeypatch.setenv("MAIN_TEST_KEY", "k")
    seen: list[tuple[str, str]] = []

    async def preflight(provider: str, *, model: str | None = None, base_url: str = "", api_key_env: str = "",
                        timeout_s: float = 5.0) -> pr.Readiness:
        seen.append((provider, base_url))
        return pr.Readiness(provider, "unreachable", f"Nothing answers at {base_url}.", "Check the address.", model or "")

    monkeypatch.setattr(pr, "preflight", preflight)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("topic: a topic\nprovider:\n  name: openai\n  model: m\n  api_key_env: MAIN_TEST_KEY\n"
                   "  node_providers:\n    implement: {name: openai, model: kimi-k3, base_url: " + STEP_URL
                   + ", api_key_env: STEP_TEST_KEY}\noutput:\n  output_dir: " + json.dumps(str(tmp_path / "o")) + "\n",
                   encoding="utf-8")
    assert await launch.main_async(parse_args(["--config", str(cfg), "--no-axon-sidecar"])) == 2
    err = capsys.readouterr().err
    assert "the step `implement` runs on its own provider" in err and f"Nothing answers at {STEP_URL}" in err
    assert seen == [("openai", STEP_URL)]   # the step's own address, not the main provider's


def test_an_interview_update_keeps_the_steps_providers(tmp_path: Path) -> None:
    """The web form, the CLI interview and the VS Code form do not write ``node_providers`` (a config-file setting);
    an update through any of them keeps what the file holds."""
    import yaml

    from core.interview import answers_to_yaml
    from core.interview_update import rewrite_yaml_with_new_answers
    from tests.test_interview_update import _sample

    raw = yaml.safe_load(answers_to_yaml(_sample(), frontend="cli"))
    raw["provider"]["node_providers"] = {"implement": dict(KIMI)}
    out = yaml.safe_load(rewrite_yaml_with_new_answers(raw, _sample(paper_format="neurips")))
    assert out["provider"]["node_providers"] == {"implement": dict(KIMI)}
    assert Config.model_validate(out).provider.node_providers["implement"].model == "kimi-k3"
