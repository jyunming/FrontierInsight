"""A model call that meets a usage limit is no answer: the step that made it is not kept (r4: the plan's model was out of
quota, FI counted the failed call as an ask and stopped the quest as stuck on a plan nobody had been asked to fix)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.config import Config, EngineConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, ModelUnavailable
from core.provider import _CliTransientError, _is_fatal_provider_error, _retry_cli_error

CODEX_LIMIT = ("C:\\Users\\x\\AppData\\Roaming\\npm\\codex.CMD exited rc=1: You\u2019ve hit your usage limit. Upgrade to "
               "Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more "
               "credits or try again at 4:51 AM.")


def test_the_codex_usage_limit_is_a_limit_that_does_not_clear_within_the_run() -> None:
    err = _CliTransientError(CODEX_LIMIT)
    assert _is_fatal_provider_error(err)
    assert not _retry_cli_error(err), "not retried: it cannot clear in the retry window"
    assert not _is_fatal_provider_error(_CliTransientError("codex.CMD exited rc=1: stream disconnected"))


class _Limited:
    async def chat(self, messages: Any, **kw: Any) -> str:
        raise _CliTransientError(CODEX_LIMIT)


class _Fine:
    async def chat(self, messages: Any, **kw: Any) -> str:
        return "an answer"


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: Any) -> Engine:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    eng = Engine(Config(
        topic="a neutral topic about a cooling cup", title="limit", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng._client = client  # type: ignore[assignment]

    async def _client_for(node: Any) -> Any:
        return client, []

    monkeypatch.setattr(eng, "_client_for", _client_for)
    return eng


async def _swallowing_step(eng: Engine) -> dict[str, Any]:
    try:
        await eng._chat("complete the plan", node="plan_revise")
    except Exception:  # noqa: BLE001 -- the "could not ask" path the plan step really has
        return {"decided": "stuck"}
    return {"decided": "went on"}


@pytest.mark.asyncio
async def test_a_step_that_goes_on_without_the_answer_is_not_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch, _Limited())
    step = eng._audited("plan", lambda state: _swallowing_step(eng))
    with pytest.raises(ModelUnavailable) as caught:
        await step({})
    assert "usage limit" in str(caught.value) and "Resume the quest" in str(caught.value)


@pytest.mark.asyncio
async def test_a_step_whose_calls_answer_is_kept_and_an_earlier_limit_does_not_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine(tmp_path, monkeypatch, _Fine())
    eng.__dict__["_provider_unavailable"] = "plan_revise: an earlier step's limit"
    step = eng._audited("plan", lambda state: _swallowing_step(eng))
    assert (await step({}))["decided"] == "went on"


@pytest.mark.asyncio
async def test_a_pause_after_a_call_its_model_could_not_take_is_a_stop_not_a_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langgraph.errors import GraphInterrupt

    eng = _engine(tmp_path, monkeypatch, _Limited())

    async def asks_a_person(state: Any) -> dict[str, Any]:
        await _swallowing_step(eng)
        raise GraphInterrupt(())  # r4 euv: "Waiting for you: the settings to vary could not be read"

    with pytest.raises(ModelUnavailable):
        await eng._audited("plan", asks_a_person)({})


@pytest.mark.asyncio
async def test_a_real_pause_is_still_a_pause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from langgraph.errors import GraphInterrupt

    eng = _engine(tmp_path, monkeypatch, _Fine())

    async def asks_a_person(state: Any) -> dict[str, Any]:
        await _swallowing_step(eng)
        raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        await eng._audited("plan", asks_a_person)({})
