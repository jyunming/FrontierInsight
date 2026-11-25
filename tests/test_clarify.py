"""The first node (``clarify``) and the helper that renders an earlier FI's setup answers.

The node asks nothing and calls no model: it sets the study's two run modes (no experiment, survey) from the
commission's ``method``, which ``Config`` has put into ``engine.no_simulation`` / ``engine.survey_mode``. These tests do
NOT call real LLMs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core import commission
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine, _format_clarify


def _mk_cfg(tmp_path: Path, **engine: object) -> Config:
    return Config(
        topic="a draining tank", title="tank",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, **engine),  # type: ignore[arg-type]
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


def _commissioned(tmp_path: Path, **changes: object) -> Config:
    from tests.test_commission import _commission

    path = _commission(tmp_path / "order", **changes)
    return Config(
        commission=path, provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


# --- the helper that renders an earlier FI's setup answers --------------------------------------------------------------


def test_format_clarify_is_the_empty_marker_when_no_answers() -> None:
    """No clarify_answers in state (every new study) -> a marker, so the prompt still parses cleanly."""
    assert _format_clarify({}) == "(none — no setup questions were asked)"
    assert _format_clarify({"clarify_answers": {}}) == "(none — no setup questions were asked)"


def test_format_clarify_renders_bulleted_block_of_a_checkpoint_that_has_answers() -> None:
    state = {
        "clarify_answers": {
            "comparative_baseline": "the textbook case",
            "empirical_vs_theoretical": "empirical",
            "budget": "a few minutes on CPU",
            "output_kinds": ["paper_md", "slides"],
        }
    }
    out = _format_clarify(state)
    assert "- **Comparative baseline**: the textbook case" in out
    assert "- **Time / compute budget**: a few minutes on CPU" in out
    assert "- **Desired output kinds**: paper_md, slides" in out


def test_format_clarify_skips_unknown_slots() -> None:
    out = _format_clarify({"clarify_answers": {"comparative_baseline": "x", "future_slot": "y"}})
    assert "Comparative baseline" in out and "future_slot" not in out


# --- the node ---------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_node_makes_no_model_call_and_a_commissioned_study_simulates_by_default(tmp_path: Path) -> None:
    eng = Engine(_commissioned(tmp_path))
    chat = AsyncMock(side_effect=AssertionError("the first node asks no model"))
    eng._client = type("Stub", (), {"chat": chat})()
    patch = await eng._node_clarify({"topic": "x"})
    assert patch == {"clarify_done": True, "no_simulation_resolved": False, "survey_mode_resolved": False}
    chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method,no_sim,survey", [("simulate", False, False), ("analyze_data", True, False),
                                                  ("review_literature", True, True)])
async def test_the_commission_s_method_sets_the_two_modes(tmp_path: Path, method: str, no_sim: bool,
                                                          survey: bool) -> None:
    eng = Engine(_commissioned(tmp_path, method=method))
    chat = AsyncMock(side_effect=AssertionError("the first node asks no model"))
    eng._client = type("Stub", (), {"chat": chat})()
    patch = await eng._node_clarify({"topic": "x"})
    assert patch == {"clarify_done": True, "no_simulation_resolved": no_sim, "survey_mode_resolved": survey}
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_config_built_in_code_without_a_commission_keeps_its_own_two_keys(tmp_path: Path) -> None:
    """``Engine`` itself refuses nothing (the start paths call ``commission.door``): the keys of the config are the modes."""
    eng = Engine(_mk_cfg(tmp_path, no_simulation=True))
    assert await eng._node_clarify({"topic": "x"}) == {
        "clarify_done": True, "no_simulation_resolved": True, "survey_mode_resolved": False}
    eng = Engine(_mk_cfg(tmp_path / "s", survey_mode=True))
    assert await eng._node_clarify({"topic": "x"}) == {
        "clarify_done": True, "no_simulation_resolved": True, "survey_mode_resolved": True}


@pytest.mark.asyncio
async def test_the_node_passes_through_when_already_done(tmp_path: Path) -> None:
    """A resume after the node finished does not set the modes again (the checkpoint has them)."""
    eng = Engine(_mk_cfg(tmp_path))
    assert await eng._node_clarify({"topic": "x", "clarify_done": True}) == {}


@pytest.mark.asyncio
async def test_a_checkpoint_that_stopped_at_the_old_setup_questions_goes_on_without_answers(tmp_path: Path) -> None:
    """A state an earlier FI left with its questions asked but not answered (``clarify_questions`` set, no
    ``clarify_done``): the node does not ask them, sets the modes and the quest goes on."""
    eng = Engine(_commissioned(tmp_path, method="analyze_data"))
    patch = await eng._node_clarify({"topic": "x", "clarify_questions": {"budget": {"question": "?", "default": "1 h"}}})
    assert patch["clarify_done"] is True and patch["no_simulation_resolved"] is True and "clarify_answers" not in patch


def test_the_commission_s_title_is_the_study_s_title_and_the_config_s_title_wins(tmp_path: Path) -> None:
    from tests.test_commission import _commission

    path = _commission(tmp_path / "order", title="Draining a tank")
    assert Config(commission=path).title == "Draining a tank"
    assert Config(commission=path, title="My own").title == "My own"
    plain = _commission(tmp_path / "plain")
    assert Config(commission=plain).title is None  # the existing fallback: the question
    assert json.loads(Path(path).read_text(encoding="utf-8"))["title"] == "Draining a tank"
    assert commission.method_of(commission.read(plain)) is None
