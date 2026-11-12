"""The Claude CLI bills an API account when ANTHROPIC_API_KEY is in its environment (a .env loaded at launch can put it
there) and the signed-in subscription otherwise. FI says which at the start, and a stop for an empty API balance says that,
not "usage limit reached". Real failure: a quest on provider claude_cli stopped at once with a usage-limit message while the
subscription worked; the CLI's stream error was 'Credit balance is too low' (billing_error), the API account's. The key's
value is never printed. Fake keys only; no model is called."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from core import crash_kind, provider
from core.config import Config, EngineConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, ModelUnavailable
from core.provider import ProviderLimitError, ProviderUnusable, _CliTransientError

KEY = "sk-ant-api03-not-a-real-key-DO-NOT-PRINT"
BILLING = "claude.EXE exited rc=1 (stream error: Credit balance is too low) [the CLI named it: billing_error]"
SESSION = "claude.EXE exited rc=1: You've hit your session limit · resets 5pm (UTC)"


def _engine(tmp_path: Path, **provider: Any) -> Engine:
    return Engine(Config(
        topic="a neutral topic about a cooling cup", title="account", provider=ProviderConfig(**provider),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


class _Seen(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


async def _connect(eng: Engine) -> list[logging.LogRecord]:
    seen = _Seen()
    eng._log.addHandler(seen)
    try:
        await eng._connect_llm()
    finally:
        eng._log.removeHandler(seen)
        await eng._client.aclose()
    return [r for r in seen.records if "Claude CLI" in r.getMessage()]


# --- which account the CLI will bill, said at the start ------------------------------------------------------------------

def test_the_account_sentence_names_the_api_account_when_the_key_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    said = provider.claude_cli_account_line()
    assert "bill the API account" in said and "ANTHROPIC_API_KEY is set" in said and KEY not in said


def test_the_account_sentence_names_the_subscription_when_there_is_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    said = provider.claude_cli_account_line()
    assert "your Claude subscription" in said and "API account" not in said
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")  # an empty variable is no key
    assert "your Claude subscription" in provider.claude_cli_account_line()


@pytest.mark.asyncio
async def test_a_quest_on_the_claude_cli_says_which_account_at_the_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    eng = _engine(tmp_path, name="claude_cli")
    lines = await _connect(eng)
    assert len(lines) == 1
    assert "bill the API account" in lines[0].getMessage() and getattr(lines[0], "progress", False), "reaches the console too"
    assert KEY not in (eng.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_without_a_key_the_start_line_names_the_subscription(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    lines = await _connect(_engine(tmp_path, name="claude_cli"))
    assert len(lines) == 1 and "your Claude subscription" in lines[0].getMessage()


@pytest.mark.asyncio
async def test_a_quest_that_uses_the_claude_cli_only_as_a_fallback_says_it_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    lines = await _connect(_engine(tmp_path, name="openai", fallback=["claude_cli"]))
    assert len(lines) == 1 and "bill the API account" in lines[0].getMessage()


@pytest.mark.asyncio
async def test_a_quest_that_never_uses_the_claude_cli_says_nothing_of_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert await _connect(_engine(tmp_path, name="openai")) == []


# --- an empty API balance is not a usage limit ---------------------------------------------------------------------------

def test_a_billing_error_is_told_from_a_usage_limit() -> None:
    assert provider.is_billing_error(ProviderLimitError(BILLING, provider="claude_cli"))
    assert provider.is_billing_error(_CliTransientError("claude.EXE exited rc=1 (stream error: Credit balance is too low): "))
    assert not provider.is_billing_error(ProviderLimitError(SESSION, provider="claude_cli"))
    assert not provider.is_billing_error(_CliTransientError("claude.EXE exited rc=1: stream disconnected"))
    assert not provider.is_billing_error(RuntimeError(BILLING))


def test_the_stop_for_an_empty_api_balance_says_so_and_how_to_use_the_subscription(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    eng = _engine(tmp_path, name="claude_cli")
    said = eng._unavailable_message("", ProviderLimitError(BILLING, provider="claude_cli", model="sonnet"))
    assert "API account" in said and "balance is empty" in said and "ANTHROPIC_API_KEY" in said and "subscription" in said
    assert "usage limit" not in said and KEY not in said
    assert f"--resume {eng.quest_id}" in said and "Nothing the current step decided was kept" in said


def test_the_stop_for_an_empty_balance_without_a_key_does_not_blame_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    eng = _engine(tmp_path, name="claude_cli")
    said = eng._unavailable_message("", ProviderLimitError(BILLING, provider="claude_cli", model="sonnet"))
    assert "no credit left" in said and "usage limit" not in said and "ANTHROPIC_API_KEY" not in said


def test_a_real_usage_limit_still_says_usage_limit_whatever_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    eng = _engine(tmp_path, name="claude_cli")
    said = eng._unavailable_message("", ProviderLimitError(SESSION, provider="claude_cli", model="sonnet", resumes_at="5pm"))
    assert "usage limit was reached" in said and "after 5pm" in said and "API account" not in said


def test_the_failure_card_says_the_same_for_an_empty_api_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    stop = ModelUnavailable("x")
    stop.__cause__ = ProviderLimitError(BILLING, provider="claude_cli", model="sonnet")
    got = crash_kind.classify(stop, provider="claude_cli", model="sonnet")
    assert got.kind == "setup" and "API account" in got.say and "balance is empty" in got.say
    assert "ANTHROPIC_API_KEY" in got.do and "subscription" in got.do and KEY not in got.say + got.do
    stop.__cause__ = ProviderLimitError(SESSION, provider="claude_cli", model="sonnet", resumes_at="5pm")
    assert "used up its allowance" in crash_kind.classify(stop, provider="claude_cli", model="sonnet").say


@pytest.mark.asyncio
async def test_a_step_that_meets_the_billing_error_stops_with_the_api_account_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    eng = _engine(tmp_path, name="claude_cli")

    async def step(state: Any) -> dict[str, Any]:
        raise ProviderUnusable(ProviderLimitError(BILLING, provider="claude_cli", model="sonnet"))

    with pytest.raises(ModelUnavailable) as caught:
        await eng._audited("execute", step)({})
    assert "API account" in str(caught.value) and "usage limit" not in str(caught.value)
