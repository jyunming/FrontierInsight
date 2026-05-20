"""Analyze-driven re-route + cross-paper check tests.

Validates:
  - `_node_cross_check`: per-finding literature search + LLM classification.
  - `_route_after_cross_check`: publish / re_experiment / broaden_lit routing.
  - Iteration accounting: re_experiment/broaden_lit consume `max_iterations`.
  - `_format_cross_check`: renders the write-prompt block correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine, _format_cross_check
from core.knowledge import RetrievedDoc


def _mk_cfg(
    tmp_path: Path, *,
    enable_analyze_reroute: bool = True,
    per_finding_k: int = 3,
    max_iter: int = 2,
) -> Config:
    return Config(
        topic="cross-check tests",
        title="cc",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=max_iter, review_loop=False,
            enable_analyze_reroute=enable_analyze_reroute,
            cross_check_per_finding_k=per_finding_k,
            ideate_reflect=False,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


# --- _format_cross_check helper --------------------------------------------


def test_format_cross_check_empty_state() -> None:
    assert "none" in _format_cross_check({}).lower()


def test_format_cross_check_renders_supporting_and_conflicting() -> None:
    state = {
        "cross_check": [{
            "finding": "F_SE ≈ 2.3 at 92 eV",
            "supporting": [{"index": 1, "why": "reports F_SE=2.31"}],
            "conflicting": [{"index": 2, "why": "claims F_SE > 4 above 70 eV"}],
            "neutral": [],
            "summary": "literature broadly agrees in the EUV regime",
            "candidates": [
                {"title": "Grenville 2015 MOR", "doi": "10.1117/12.X"},
                {"title": "Hinsberg 2017 imaging", "doi": "10.1117/12.Y"},
            ],
        }],
    }
    out = _format_cross_check(state)
    assert "F_SE" in out
    assert "supporting (1)" in out
    assert "conflicting (1)" in out
    assert "Grenville 2015 MOR" in out
    assert "10.1117/12.X" in out
    assert "broadly agrees" in out


# --- _node_cross_check ------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_check_skipped_when_no_findings(tmp_path: Path) -> None:
    eng = Engine(_mk_cfg(tmp_path))
    chat_mock = AsyncMock(return_value="{}")
    eng._client = type("Stub", (), {"chat": chat_mock})()
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]

    patch = await eng._node_cross_check({"analysis": {"key_findings": []}})

    assert patch == {"cross_check": []}
    chat_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_cross_check_disabled_when_k_is_zero(tmp_path: Path) -> None:
    eng = Engine(_mk_cfg(tmp_path, per_finding_k=0))
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value="{}")})()
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]

    patch = await eng._node_cross_check({"analysis": {"key_findings": ["x"]}})

    assert patch == {"cross_check": []}


@pytest.mark.asyncio
async def test_cross_check_classifies_each_finding(tmp_path: Path) -> None:
    """For each finding, the node searches literature with the FINDING
    TEXT as query (not the topic) and asks the LLM to classify hits.
    Result lands in `state['cross_check']` for the write prompt."""
    eng = Engine(_mk_cfg(tmp_path))

    async def fake_search(query, **kw):  # noqa: ANN001
        return [
            RetrievedDoc(
                content="A 2015 study reports F_SE ≈ 2.3 in the same regime.",
                metadata={"title": "Grenville 2015", "doi": "10.1117/12.X"},
            ),
            RetrievedDoc(
                content="A 2020 study reports F_SE > 4 in adjacent regimes.",
                metadata={"title": "OtherAuthor 2020", "doi": "10.1117/12.Y"},
            ),
        ]
    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]

    chat_mock = AsyncMock(return_value=json.dumps({
        "supporting": [{"index": 1, "why": "reports same F_SE value"}],
        "conflicting": [{"index": 2, "why": "reports F_SE > 4"}],
        "neutral": [],
        "summary": "two-way disagreement",
    }))
    eng._client = type("Stub", (), {"chat": chat_mock})()

    patch = await eng._node_cross_check({
        "topic": "EUV-MOR SE-yield",
        "analysis": {
            "key_findings": ["F_SE ≈ 2.3 at 92 eV", "Dose scales sub-linearly"],
            "next_step": "publish",
        },
    })

    assert "cross_check" in patch
    assert len(patch["cross_check"]) == 2
    first = patch["cross_check"][0]
    assert first["finding"] == "F_SE ≈ 2.3 at 92 eV"
    assert len(first["supporting"]) == 1
    assert len(first["conflicting"]) == 1
    assert first["candidates"][0]["doi"] == "10.1117/12.X"
    # One classify-LLM call per finding.
    assert chat_mock.await_count == 2


@pytest.mark.asyncio
async def test_cross_check_tolerates_search_failure(tmp_path: Path) -> None:
    """If `asearch` raises, the node logs and records an empty per-
    finding entry — never propagates the exception."""
    eng = Engine(_mk_cfg(tmp_path))

    async def boom(*a, **kw):
        raise RuntimeError("network down")
    eng.knowledge.asearch = boom  # type: ignore[method-assign]
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value="{}")})()

    patch = await eng._node_cross_check({
        "analysis": {"key_findings": ["x"]},
    })
    assert len(patch["cross_check"]) == 1
    assert patch["cross_check"][0]["candidates"] == []


@pytest.mark.asyncio
async def test_cross_check_bumps_iteration_on_re_experiment(tmp_path: Path) -> None:
    """When analyze's `next_step` is `re_experiment` AND budget remains,
    the cross_check node bumps the shared iteration counter (mirrors
    review's bump-on-revise behavior). Without this, the re-route
    would loop forever."""
    eng = Engine(_mk_cfg(tmp_path, max_iter=2))
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value="{}")})()
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]

    patch = await eng._node_cross_check({
        "analysis": {"key_findings": ["x"], "next_step": "re_experiment"},
        "iteration": 0,
    })
    assert patch["iteration"] == 1


@pytest.mark.asyncio
async def test_cross_check_does_not_bump_when_publishing(tmp_path: Path) -> None:
    eng = Engine(_mk_cfg(tmp_path, max_iter=2))
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value="{}")})()
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]

    patch = await eng._node_cross_check({
        "analysis": {"key_findings": ["x"], "next_step": "publish"},
        "iteration": 0,
    })
    assert "iteration" not in patch


# --- routing function -------------------------------------------------------


def test_route_after_cross_check_publish_goes_to_write(tmp_path: Path) -> None:
    eng = Engine(_mk_cfg(tmp_path))
    assert eng._route_after_cross_check({
        "analysis": {"next_step": "publish"},
        "iteration": 0,
    }) == "write"


# A minimal non-empty cross_check payload — the route needs *something*
# in `state["cross_check"]` to proceed past the empty-guard (see below).
_CC_NON_EMPTY = [{
    "finding": "x",
    "supporting": [], "conflicting": [], "neutral": [],
    "summary": "", "candidates": [],
}]


def test_route_after_cross_check_re_experiment_routes_to_design(tmp_path: Path) -> None:
    eng = Engine(_mk_cfg(tmp_path, max_iter=2))
    assert eng._route_after_cross_check({
        "cross_check": _CC_NON_EMPTY,
        "analysis": {"next_step": "re_experiment"},
        "iteration": 0,
    }) == "redesign"


def test_route_after_cross_check_broaden_lit_routes_to_literature(
    tmp_path: Path,
) -> None:
    """``broaden_lit`` re-enters the literature node so the next pass
    can fetch fresh evidence. Until this routing existed, the signal
    collapsed onto ``redesign`` and the design node re-ran with the
    SAME literature it already had — defeating the whole point of
    saying "broaden_lit"."""
    eng = Engine(_mk_cfg(tmp_path, max_iter=2))
    assert eng._route_after_cross_check({
        "cross_check": _CC_NON_EMPTY,
        "analysis": {"next_step": "broaden_lit"},
        "iteration": 0,
    }) == "broaden_lit"


def test_route_after_cross_check_broaden_lit_falls_through_when_budget_exhausted(
    tmp_path: Path,
) -> None:
    """Like the re_experiment branch, broaden_lit respects the
    shared ``engine.max_iterations`` budget — at the cap the quest
    publishes instead of looping again."""
    eng = Engine(_mk_cfg(tmp_path, max_iter=2))
    assert eng._route_after_cross_check({
        "cross_check": _CC_NON_EMPTY,
        "analysis": {"next_step": "broaden_lit"},
        "iteration": 2,
    }) == "write"


def test_route_after_cross_check_falls_through_to_write_when_budget_exhausted(
    tmp_path: Path,
) -> None:
    """Even when analyze says `re_experiment`, if `iteration` is at the
    cap we publish anyway — the quest stays bounded."""
    eng = Engine(_mk_cfg(tmp_path, max_iter=2))
    assert eng._route_after_cross_check({
        "cross_check": _CC_NON_EMPTY,
        "analysis": {"next_step": "re_experiment"},
        "iteration": 2,  # already at cap
    }) == "write"


def test_route_after_cross_check_respects_config_disable(tmp_path: Path) -> None:
    """`enable_analyze_reroute=False` makes the route always go to write,
    regardless of `next_step`."""
    eng = Engine(_mk_cfg(tmp_path, enable_analyze_reroute=False))
    assert eng._route_after_cross_check({
        "cross_check": _CC_NON_EMPTY,
        "analysis": {"next_step": "re_experiment"},
        "iteration": 0,
    }) == "write"


def test_route_after_cross_check_empty_cross_check_terminates_loop(
    tmp_path: Path,
) -> None:
    """Audit BLOCK #11: ``_node_cross_check`` returns ``{"cross_check": []}``
    early when ``analysis.key_findings`` is empty (or ``cross_check_per_finding_k
    <= 0``) BEFORE the iteration-bump block runs. If analyze still emits
    ``next_step: "broaden_lit"`` in that case, iteration never increments,
    the cap never fires, and the engine loops literature → design →
    implement → execute → analyze → cross_check → broaden_lit forever —
    unbounded LLM cost. The empty-cross_check guard must terminate the
    loop by routing to ``write`` regardless of analyze's recommendation."""
    eng = Engine(_mk_cfg(tmp_path, max_iter=10))
    assert eng._route_after_cross_check({
        "cross_check": [],
        "analysis": {"next_step": "broaden_lit"},
        "iteration": 0,
    }) == "write"
    # Same for ``re_experiment`` — no findings to re-experiment against either.
    assert eng._route_after_cross_check({
        "cross_check": [],
        "analysis": {"next_step": "re_experiment"},
        "iteration": 0,
    }) == "write"
    # Missing key entirely (defensive — early aborts may not even set it).
    assert eng._route_after_cross_check({
        "analysis": {"next_step": "broaden_lit"},
        "iteration": 0,
    }) == "write"


# --- _node_literature iterative-loop behaviour ------------------------------


@pytest.mark.asyncio
async def test_node_literature_first_pass_sets_iter_to_one(tmp_path: Path) -> None:
    """First entry into the literature node — no prior literature, no
    counter. Result: literature populated, ``literature_iter`` == 1."""
    eng = Engine(_mk_cfg(tmp_path))

    async def fake_search(query, **kw):  # noqa: ANN001
        return [
            RetrievedDoc(content="abstract A", metadata={"doi": "10.1/a", "url": "u1"}),
            RetrievedDoc(content="abstract B", metadata={"doi": "10.1/b", "url": "u2"}),
        ]
    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]

    patch = await eng._node_literature({
        "topic": "first-pass topic",
        "chosen_idea": {"title": "T"},
    })
    assert patch["literature_iter"] == 1
    assert len(patch["literature"]) == 2
    assert patch["literature"][0]["metadata"]["doi"] == "10.1/a"


@pytest.mark.asyncio
async def test_node_literature_broaden_lit_reentry_accumulates_and_dedups(
    tmp_path: Path,
) -> None:
    """When ``broaden_lit`` routes back into ``literature``, the second
    pass MUST keep the previously-retrieved docs and only append new
    ones (DOI- / URL- / content-prefix dedup). Without accumulation
    the design node would lose context every time the loop fires."""
    eng = Engine(_mk_cfg(tmp_path))

    # Second pass returns one overlap (doi 10.1/a) and two genuinely new.
    async def fake_search(query, **kw):  # noqa: ANN001
        return [
            RetrievedDoc(content="abstract A (dup)", metadata={"doi": "10.1/a"}),
            RetrievedDoc(content="abstract C", metadata={"doi": "10.1/c"}),
            RetrievedDoc(content="abstract D", metadata={"url": "uniq-url-d"}),
        ]
    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]

    prior_state = {
        "topic": "re-entry topic",
        "chosen_idea": {"title": "T"},
        "design": {"hypothesis": "specific design hypothesis"},
        "literature_iter": 1,
        "literature": [
            {"content": "abstract A", "metadata": {"doi": "10.1/a", "url": "u1"}},
            {"content": "abstract B", "metadata": {"doi": "10.1/b", "url": "u2"}},
        ],
    }
    patch = await eng._node_literature(prior_state)
    assert patch["literature_iter"] == 2
    # Original 2 + 2 new (C, D) — the duplicate A is dropped.
    assert len(patch["literature"]) == 4
    dois = [e.get("metadata", {}).get("doi") for e in patch["literature"]]
    assert dois.count("10.1/a") == 1, "duplicate DOI must be dropped on re-entry"
    assert "10.1/c" in dois
    urls = [e.get("metadata", {}).get("url") for e in patch["literature"]]
    assert "uniq-url-d" in urls
