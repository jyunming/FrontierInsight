"""Tests for the ``--proposal-ensemble`` / ``--critique-ensemble`` CLI
flag parsing.

The flags accept a comma-separated model list. The helper
``_parse_ensemble_flag`` converts that triple into a
``NodeEnsembleConfig`` (or None when the flag is omitted) — those
are the contract points the rest of the wiring relies on.
"""
from __future__ import annotations

import pytest

from core.config import NodeEnsembleConfig
from launch import _parse_ensemble_flag


# ---------------------------------------------------------------------------
# Empty / no-flag → None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("models_csv", ["", "  ", "\t\n", None])
def test_parse_empty_returns_none(models_csv) -> None:
    """No --*-ensemble flag → None (single-call default). Whitespace-
    only strings are also treated as 'not provided'."""
    result = _parse_ensemble_flag(models_csv or "", "tournament", "")
    assert result is None


def test_parse_all_commas_returns_none() -> None:
    """Edge case: ``--proposal-ensemble ,,,`` parses to no models →
    None (don't silently kick off an empty fanout)."""
    result = _parse_ensemble_flag(",,,", "tournament", "")
    assert result is None


# ---------------------------------------------------------------------------
# Real values
# ---------------------------------------------------------------------------


def test_parse_three_models_tournament() -> None:
    result = _parse_ensemble_flag(
        "gpt-5.5,claude-opus-4-7,gemini-2.5-pro",
        "tournament", "",
    )
    assert isinstance(result, NodeEnsembleConfig)
    assert result.models == ["gpt-5.5", "claude-opus-4-7", "gemini-2.5-pro"]
    assert result.merge == "tournament"
    assert result.moderator is None  # falls back to models[0] downstream


def test_parse_synthesize_with_moderator() -> None:
    result = _parse_ensemble_flag(
        "m1,m2,m3", "synthesize", "claude-opus-4-7",
    )
    assert result is not None
    assert result.merge == "synthesize"
    assert result.moderator == "claude-opus-4-7"


def test_parse_strips_whitespace() -> None:
    """Comma-separated list tolerates spaces around each name."""
    result = _parse_ensemble_flag("  m1 , m2 , m3  ", "tournament", "")
    assert result is not None
    assert result.models == ["m1", "m2", "m3"]


def test_parse_drops_empty_entries() -> None:
    """``a,,b`` → models = [a, b]. Empty entries silently dropped."""
    result = _parse_ensemble_flag("a,,b", "tournament", "")
    assert result is not None
    assert result.models == ["a", "b"]


def test_parse_single_model_still_works() -> None:
    """N=1 is degenerate but legal — degrades to a single call with
    a moderator merge step that just returns the lone candidate.
    Useful for testing the wiring without spending 3× the calls."""
    result = _parse_ensemble_flag("only-model", "tournament", "")
    assert result is not None
    assert result.models == ["only-model"]
