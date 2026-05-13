"""Direct tests for `core.portfolio`.

The portfolio module is intentionally thin: it collects snapshots via
the helper from `core.digest`, computes deterministic stats, and
dispatches a single LLM call. Tests cover stats math + prompt
rendering + the end-to-end empty / non-empty paths.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config import ProviderConfig
from core.digest import QuestSnapshot
from core.portfolio import (
    _build_portfolio_prompt,
    _format_quest_block,
    _portfolio_id,
    _render_quest_corpus,
    _stats_block,
    generate_portfolio,
)


# Reuse the synthetic-quest-dir helper from the digest test module to
# avoid duplicating the boilerplate. Importing across test modules is
# tolerated when both live under tests/ and ship together.
from tests.test_digest import _make_quest_dir


def _snap(
    qid: str, *, terminal: str = "review",
    title: str = "", topic: str = "", provider: str = "openai",
    created_offset_days: int = 0,
) -> QuestSnapshot:
    now = datetime.now(timezone.utc)
    return QuestSnapshot(
        quest_id=qid, title=title or qid, topic=topic,
        created_at=now - timedelta(days=created_offset_days),
        last_modified=now - timedelta(days=created_offset_days),
        terminal_node=terminal, paper_abstract="An abstract.",
        provider=provider, quest_root=Path("/tmp"),
    )


# ---------- prompt helpers --------------------------------------------------


def test_portfolio_id_uses_iso_date() -> None:
    now = datetime(2026, 5, 13, 14, 30, tzinfo=timezone.utc)
    assert _portfolio_id(now) == "2026-05-13"


def test_format_quest_block_includes_id_title_topic_abstract() -> None:
    q = _snap(
        "1700000700-x-aabbcc",
        title="Toy quest",
        topic="Multi-line topic\nsecond line should be dropped",
        provider="openai",
    )
    block = _format_quest_block(q)
    assert "1700000700-x-aabbcc" in block
    assert "Toy quest" in block
    assert "Multi-line topic" in block
    # The second line should not appear (we strip after the first line).
    assert "second line should be dropped" not in block
    assert "provider: openai" in block
    assert "completed: True" in block
    assert "abstract:" in block


def test_format_quest_block_omits_topic_when_empty() -> None:
    q = _snap("1700000700-x-aabbcc", topic="")
    block = _format_quest_block(q)
    assert "topic:" not in block


def test_render_quest_corpus_orders_most_recent_first() -> None:
    snaps = [
        _snap("1700000001-old-aabbcc", created_offset_days=30),
        _snap("1700000002-new-aabbcc", created_offset_days=1),
        _snap("1700000003-mid-aabbcc", created_offset_days=10),
    ]
    rendered, n_trunc = _render_quest_corpus(snaps)
    assert n_trunc == 0
    # new first, mid, old last
    new_pos = rendered.index("1700000002-new-aabbcc")
    mid_pos = rendered.index("1700000003-mid-aabbcc")
    old_pos = rendered.index("1700000001-old-aabbcc")
    assert new_pos < mid_pos < old_pos


def test_render_quest_corpus_truncates_at_cap_oldest_first(monkeypatch) -> None:
    import core.portfolio as pm
    monkeypatch.setattr(pm, "_MAX_PROMPT_QUESTS", 3)
    snaps = [
        _snap(f"170000000{i}-q-aabbcc", created_offset_days=i)
        for i in range(6)
    ]
    rendered, n_trunc = pm._render_quest_corpus(snaps)
    assert n_trunc == 3
    # Most-recent 3 (offset 0, 1, 2) are kept; offsets 3-5 dropped.
    assert "1700000000-q-aabbcc" in rendered
    assert "1700000001-q-aabbcc" in rendered
    assert "1700000002-q-aabbcc" in rendered
    assert "1700000003-q-aabbcc" not in rendered
    assert "1700000004-q-aabbcc" not in rendered
    assert "1700000005-q-aabbcc" not in rendered


def test_render_quest_corpus_empty_returns_marker() -> None:
    rendered, n_trunc = _render_quest_corpus([])
    assert "(no quests yet)" in rendered
    assert n_trunc == 0


# ---------- stats block -----------------------------------------------------


def test_stats_block_reports_totals_and_split() -> None:
    snaps = [
        _snap("1700000001-a-aabbcc", terminal="review"),
        _snap("1700000002-b-aabbcc", terminal="review"),
        _snap("1700000003-c-aabbcc", terminal="in_progress"),
    ]
    block = _stats_block(snaps)
    assert "Total quests on disk: 3" in block
    assert "Completed: 2" in block
    assert "In progress / unknown: 1" in block


def test_stats_block_provider_breakdown_sorted_by_count() -> None:
    snaps = [
        _snap("1700000001-a-aabbcc", provider="openai"),
        _snap("1700000002-b-aabbcc", provider="openai"),
        _snap("1700000003-c-aabbcc", provider="claude_cli"),
    ]
    block = _stats_block(snaps)
    # openai (2) before claude_cli (1) — count-descending sort.
    openai_pos = block.index("openai: 2")
    claude_pos = block.index("claude_cli: 1")
    assert openai_pos < claude_pos


def test_stats_block_provider_breakdown_tiebreaks_by_name() -> None:
    """Regression: when two providers tie on count, sort order must
    be stable (alphabetical by name) — otherwise the rendered stats
    block reshuffles run-to-run depending on filesystem traversal
    order and breaks deterministic-output expectations."""
    # claude_cli and openai BOTH appear twice — tied on count. The
    # secondary sort by name (ascending) puts claude_cli first.
    snaps = [
        _snap("1700000001-a-aabbcc", provider="openai"),
        _snap("1700000002-b-aabbcc", provider="claude_cli"),
        _snap("1700000003-c-aabbcc", provider="openai"),
        _snap("1700000004-d-aabbcc", provider="claude_cli"),
    ]
    block = _stats_block(snaps)
    claude_pos = block.index("claude_cli: 2")
    openai_pos = block.index("openai: 2")
    assert claude_pos < openai_pos, (
        "tied counts must fall back to name-ascending for stable output"
    )


def test_stats_block_cadence_is_na_with_lt_2_completions() -> None:
    snaps = [_snap("1700000001-only-aabbcc", terminal="review")]
    block = _stats_block(snaps)
    assert "Completion cadence: n/a" in block


def test_stats_block_cadence_computes_median_gap() -> None:
    snaps = [
        _snap("1700000001-a-aabbcc", terminal="review", created_offset_days=20),
        _snap("1700000002-b-aabbcc", terminal="review", created_offset_days=10),
        _snap("1700000003-c-aabbcc", terminal="review", created_offset_days=5),
        _snap("1700000004-d-aabbcc", terminal="review", created_offset_days=0),
    ]
    block = _stats_block(snaps)
    # Sorted offsets: -20, -10, -5, 0 → gaps 10, 5, 5 → sorted 5,5,10 → median 5.
    assert "Completion cadence:" in block
    assert "5.0 days" in block


def test_stats_block_empty_corpus() -> None:
    block = _stats_block([])
    assert "Total quests on disk: 0" in block
    assert "Time span: n/a" in block
    assert "Completion cadence: n/a" in block


# ---------- prompt assembly -------------------------------------------------


def test_build_portfolio_prompt_contains_required_anchors() -> None:
    snaps = [_snap("1700000001-x-aabbcc", title="Quest X")]
    prompt = _build_portfolio_prompt(
        snaps, generated_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    assert "2026-05-13" in prompt
    assert "1700000001-x-aabbcc" in prompt
    assert "Quest X" in prompt
    assert "Total quests on disk: 1" in prompt


# ---------- end-to-end with mocked LLM --------------------------------------


@pytest.mark.asyncio
async def test_generate_portfolio_writes_file_and_returns_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _make_quest_dir(
        outputs, quest_id="1700000700-finished-aabbcc", with_review=True,
        summary_json={"title": "Finished", "provider": "openai"},
    )
    _make_quest_dir(
        outputs, quest_id="1700001700-running-001122", with_review=False,
        summary_json={"title": "Running", "provider": "openai"},
    )

    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        captured["node"] = kw.get("node", "")
        return "# Portfolio\n\n## Overview\nbody.\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    art = await generate_portfolio(
        outputs, provider=ProviderConfig(name="openai"),
        knowledge=None,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )

    assert art.portfolio_path.is_file()
    body = art.portfolio_path.read_text(encoding="utf-8")
    assert body.startswith("# Portfolio")
    assert art.portfolio_id == "2026-05-13"
    assert art.completed_count == 1
    assert art.in_progress_count == 1
    assert art.ingested_to_axon is False
    assert captured["node"] == "portfolio"
    # Both quest IDs appear in the rendered prompt.
    assert "1700000700-finished-aabbcc" in captured["prompt"]
    assert "1700001700-running-001122" in captured["prompt"]
    # raw_state now exposes the quest_id list in prompt order (most-
    # recent first) so callers can re-resolve without re-walking.
    assert "quest_ids" in art.raw_state
    assert set(art.raw_state["quest_ids"]) == {
        "1700000700-finished-aabbcc",
        "1700001700-running-001122",
    }
    # And the count matches.
    assert art.raw_state["quest_count"] == 2


@pytest.mark.asyncio
async def test_generate_portfolio_empty_writes_marker_no_llm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    called = {"n": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        called["n"] += 1
        return "should not be invoked"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    art = await generate_portfolio(
        outputs, provider=ProviderConfig(name="openai"),
        knowledge=None,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    assert art.completed_count == 0
    assert art.in_progress_count == 0
    assert art.portfolio_path.is_file()
    body = art.portfolio_path.read_text(encoding="utf-8")
    assert "No quests on disk" in body
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_generate_portfolio_ingests_to_axon_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _make_quest_dir(
        outputs, quest_id="1700000700-x-aabbcc", with_review=True,
        summary_json={"title": "X", "provider": "openai"},
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return "# Body\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    knowledge = MagicMock()
    knowledge.enabled = True
    knowledge.add_text = MagicMock(return_value=True)

    art = await generate_portfolio(
        outputs, provider=ProviderConfig(name="openai"),
        knowledge=knowledge,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    assert art.ingested_to_axon is True
    knowledge.add_text.assert_called_once()
    kwargs = knowledge.add_text.call_args.kwargs
    assert kwargs["kind"] == "fi_portfolio"
    assert kwargs["metadata"]["portfolio_id"] == "2026-05-13"
