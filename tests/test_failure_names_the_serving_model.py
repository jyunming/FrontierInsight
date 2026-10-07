"""A failure or stop text names the provider and model that really served the step, and its advice names the setting that
changes it: a step that runs on its own provider (``provider.node_providers``) is not the main model. Fake models only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from core import crash_kind as ck
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig, StepProviderConfig,
)
from core.engine import Engine, _close_quest_logger

NOTE = "[FI] provider=openai, transport=http, model=big-model, node=implement"


def _slow(note: str | None = NOTE) -> BaseException:
    try:
        raise httpx.ReadTimeout("")
    except httpx.ReadTimeout as e:
        e.fi_timeouts = {"tries": 4, "all": True, "limit_s": 900.0}  # type: ignore[attr-defined]
        if note:
            e.add_note(note)
        return e


def _engine(tmp_path: Path, request: pytest.FixtureRequest, *, routed: bool = True) -> Engine:
    provider = ProviderConfig(
        name="ollama", model="main-model", http_timeout_s=900,
        node_providers=({"implement": StepProviderConfig(
            name="openai", model="big-model", base_url="https://api.example.com/v1", api_key_env="SOME_KEY")}
            if routed else None),
    )
    eng = Engine(Config(
        topic="a neutral topic", title="who-served", provider=provider,
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs")))
    request.addfinalizer(lambda: _close_quest_logger(eng.quest_id))
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    return eng


def test_the_call_note_says_who_served_the_step() -> None:
    assert ck.served_by(_slow()) == ("openai", "big-model", "implement")
    assert ck.served_by(_slow(note=None)) is None


def test_a_timeout_on_a_routed_step_names_that_model_and_that_steps_settings() -> None:
    f = ck.classify(_slow(), node="implement", provider="ollama", model="main-model", routed="implement")
    assert f.kind == "setup"
    assert "big-model" in f.say and "openai" in f.say and "main-model" not in f.say
    assert "provider.node_http_timeout_s.implement" in f.do and "provider.node_providers.implement" in f.do
    assert "provider.http_timeout_s" not in f.do


def test_a_timeout_on_the_main_model_keeps_the_main_advice() -> None:
    f = ck.classify(_slow(note=None), node="write", provider="ollama", model="main-model")
    assert "main-model" in f.say and "provider.http_timeout_s" in f.do and "node_providers" not in f.do


@pytest.mark.asyncio
async def test_quest_failed_md_names_the_routed_provider_and_model(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    eng = _engine(tmp_path, request)
    await eng._write_quest_failed_diagnostic(_slow(), run_config=None)
    diag = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "The model (big-model on openai) takes longer to answer the step `implement`" in diag
    assert "**Provider:** `openai` / model `big-model`" in diag
    assert "main-model" not in diag.split("## What broke")[0]
    assert "provider.node_http_timeout_s.implement" in diag and "provider.node_providers.implement" in diag


@pytest.mark.asyncio
async def test_without_a_note_the_step_is_still_resolved_to_its_own_provider(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    eng = _engine(tmp_path, request)
    assert eng._provider_for_step("implement") == ("openai", "big-model", "implement")
    assert eng._provider_for_step("implement_oracle")[:2] == ("ollama", "main-model")
    assert eng._provider_for_step("write.patch")[:2] == ("ollama", "main-model")
    eng.config.provider.node_models = {"write": "writer-model"}
    assert eng._provider_for_step("write.patch") == ("ollama", "writer-model", "")
    plain = _engine(tmp_path / "plain", request, routed=False)
    assert plain._provider_for_step("implement") == ("ollama", "main-model", "")


def test_advice_that_names_node_models_is_rewritten_for_a_routed_step(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    eng = _engine(tmp_path, request)
    said = eng._routed_words("Give `implement` another model: `provider.node_models.implement`, or "
                             "`provider: {node_models: {implement: <model>}}`; for `write`: `provider.node_models.write`.")
    assert "`provider.node_providers.implement.model`" in said
    assert "provider: {node_providers: {implement: {name: <provider>, model: <model>}}}" in said
    assert "`provider.node_models.write`" in said, "a step that is not routed keeps its own setting"


@pytest.mark.asyncio
async def test_a_pause_and_the_stuck_line_name_the_setting_that_changes_a_routed_step(
    tmp_path: Path, request: pytest.FixtureRequest, capsys: pytest.CaptureFixture[str],
) -> None:
    eng = _engine(tmp_path, request)
    seen: dict[str, Any] = {}

    class Paused(Exception):
        pass

    def stop(**kw: Any) -> Any:
        seen.update(kw)
        raise Paused()

    eng._write_next_step = stop  # type: ignore[method-assign]
    with pytest.raises(Paused):
        eng._stop_for_unreadable_design("implement", "not json")
    assert "provider.node_providers.implement.model" in seen["recommended"]
    assert not any("provider.node_models.implement" in s for s in seen["steps"])

    out = await eng._node_stuck_no_findings({
        "evidence_assessment": {"stuck_reason": "cannot_repair"},
        "exec_result": {"cannot_repair": "the run's record differs from the protocol", "returncode": 1}})
    assert out["stuck"]
    printed = capsys.readouterr().out
    assert "provider.node_providers.implement.model" in printed and "provider.node_models" not in printed
