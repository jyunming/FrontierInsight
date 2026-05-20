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
    NodeEnsembleConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine, _format_cross_check
from core.knowledge import RetrievedDoc


def _mk_cfg(
    tmp_path: Path, *,
    enable_analyze_reroute: bool = True,
    per_finding_k: int = 3,
    max_iter: int = 2,
    node_ensemble: dict[str, NodeEnsembleConfig] | None = None,
) -> Config:
    return Config(
        topic="cross-check tests",
        title="cc",
        provider=ProviderConfig(name="openai", node_ensemble=node_ensemble),
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


# --- ensemble cross_check + merge_vote integration -------------------------
#
# Audit Wave 2 — Slice 3 HIGH #1: ``merge_vote(raw, key="verdict")`` at
# engine.py:2606 used to be meaningless because the cross_check prompt did
# not emit a ``verdict`` field. Each survivor's JSON tally fell through to
# the token-sniff regex in ensemble.merge_vote which always matched the
# literal word ``"supporting"`` (the first key in the JSON object) — so
# every model "voted" supporting regardless of content, and the engine's
# survivor pick (obj.get("verdict") == majority.get("verdict")) never
# matched, falling back to ``survivors[0].text``. Net: "first model wins"
# + N-1 wasted LLM calls.
#
# The fix is in the PROMPT (agents/cross_check.md adds an explicit
# ``verdict`` field). These tests pin that the engine's vote tally is now
# a real majority, and that the supporting/conflicting/neutral block
# returned downstream comes from a survivor whose verdict agrees with
# the majority (not blindly survivor[0]).


class _ScriptedClient:
    """Minimal LLM client for ensemble cross_check tests — keyed by model."""

    def __init__(self, script: dict[str, str]):
        self.script = script
        self.calls: list[dict] = []

    async def chat(self, messages, *, temperature=0.2, model=None, node=""):
        self.calls.append({"model": model, "node": node})
        return self.script.get(model, "{}")


def _cross_check_payload(verdict: str, supporting_why: str = "") -> str:
    """Build a JSON cross_check response with the new ``verdict`` field
    plus the existing lists. Matches the schema in agents/cross_check.md."""
    return json.dumps({
        "verdict": verdict,
        "supporting": (
            [{"index": 1, "why": supporting_why or f"{verdict} survivor"}]
            if verdict in ("supporting", "mixed") else []
        ),
        "conflicting": (
            [{"index": 2, "why": f"{verdict} survivor — conflicting note"}]
            if verdict in ("conflicting", "mixed") else []
        ),
        "neutral": [] if verdict != "neutral" else [
            {"index": 3, "why": "topically tangential"},
        ],
        "summary": f"on-balance {verdict}",
    })


@pytest.mark.asyncio
async def test_cross_check_ensemble_unanimous_supporting_picks_a_survivor(
    tmp_path: Path,
) -> None:
    """3 models all emit ``verdict='supporting'``. The vote tally is a
    real majority (3/3), and the engine downstream re-parses one of the
    survivors — confirming the supporting block reaches state['cross_check'].

    Pre-fix the survivor-pick loop (obj.get('verdict') == majority['verdict'])
    never matched any JSON because the prompt didn't emit ``verdict``, so
    the engine always fell back to ``survivors[0].text``. Post-fix, the
    supporting block comes from the FIRST survivor whose verdict matches
    the majority (survivors are iterated in ``models`` order in
    ``_ensemble_chat``) — and the tally is meaningful instead of a
    regex coincidence."""
    ensemble = {
        "cross_check": NodeEnsembleConfig(
            models=["m1", "m2", "m3"], merge="vote",
        ),
    }
    eng = Engine(_mk_cfg(tmp_path, node_ensemble=ensemble))
    eng._client = _ScriptedClient({
        "m1": _cross_check_payload("supporting", "m1 supporting"),
        "m2": _cross_check_payload("supporting", "m2 supporting"),
        "m3": _cross_check_payload("supporting", "m3 supporting"),
    })
    eng._log_chat_cost = lambda **_kw: None  # type: ignore[assignment,method-assign]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]

    async def fake_search(query, **kw):  # noqa: ANN001
        return [
            RetrievedDoc(content="abstract A", metadata={"title": "P1", "doi": "10.1/a"}),
        ]
    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]

    patch = await eng._node_cross_check({
        "topic": "T",
        "analysis": {"key_findings": ["F1"], "next_step": "publish"},
    })

    entry = patch["cross_check"][0]
    # The supporting block came from ONE of the 3 survivors (whichever
    # happened to be picked first matching the majority verdict).
    assert len(entry["supporting"]) == 1
    assert entry["supporting"][0]["why"] in {
        "m1 supporting", "m2 supporting", "m3 supporting",
    }
    # All 3 fan-out calls fired — vote is no LLM moderator, pure tally.
    fanout_calls = [c for c in eng._client.calls
                    if c["node"].startswith("cross_check.ensemble[")]
    assert len(fanout_calls) == 3


@pytest.mark.asyncio
async def test_cross_check_ensemble_majority_vote_is_real(tmp_path: Path) -> None:
    """2 models emit ``verdict='supporting'``, 1 emits ``verdict='conflicting'``.

    The vote must tally to ``supporting`` (2/3), and the supporting
    block returned downstream MUST come from one of the supporting
    survivors (m1 or m2) — NOT from m3 even though m3 is a survivor.

    Pre-fix without an explicit ``verdict`` key in the JSON,
    ``merge_vote``'s regex fallback found the literal word ``supporting``
    (the first key in EVERY response's JSON, including m3's) so every
    model voted supporting; majority was spuriously supporting; and
    the survivor-pick loop matched NO survivor (none had an
    ``obj['verdict']`` field) and fell back to ``survivors[0].text``
    — which happened to be m1 here. Post-fix: the tally is genuine,
    and m3 cannot win the supporting-block selection because its
    verdict legitimately disagrees."""
    ensemble = {
        "cross_check": NodeEnsembleConfig(
            models=["m1", "m2", "m3"], merge="vote",
        ),
    }
    eng = Engine(_mk_cfg(tmp_path, node_ensemble=ensemble))
    eng._client = _ScriptedClient({
        "m1": _cross_check_payload("supporting", "m1 said yes"),
        "m2": _cross_check_payload("supporting", "m2 said yes"),
        "m3": _cross_check_payload("conflicting", "m3 disagreed"),
    })
    eng._log_chat_cost = lambda **_kw: None  # type: ignore[assignment,method-assign]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]

    async def fake_search(query, **kw):  # noqa: ANN001
        return [RetrievedDoc(content="abstract", metadata={"title": "P", "doi": "10.1/x"})]
    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]

    patch = await eng._node_cross_check({
        "topic": "T",
        "analysis": {"key_findings": ["F1"], "next_step": "publish"},
    })

    entry = patch["cross_check"][0]
    # Block came from a supporting survivor (m1 or m2), NOT from the
    # conflicting one (m3). Load-bearing assertion: pre-fix this was
    # always ``survivors[0]`` regardless of vote, so it was brittle to
    # ordering — and conceptually wrong even when it accidentally agreed.
    why = entry["supporting"][0]["why"] if entry["supporting"] else ""
    assert why in {"m1 said yes", "m2 said yes"}, (
        f"supporting block must come from a supporting survivor, got why={why!r}"
    )
    # And the conflicting block from m3 must NOT be the source — a
    # supporting-verdict survivor wouldn't have any conflicting entries
    # under our payload generator.
    assert entry["conflicting"] == []


@pytest.mark.asyncio
async def test_cross_check_ensemble_vote_tally_recorded_via_merge_vote(
    tmp_path: Path,
) -> None:
    """Direct sanity check on the merge_vote primitive against the new
    payload shape: 2 supporting + 1 conflicting → winner='supporting',
    tally={'supporting': 2, 'conflicting': 1}, tie=False."""
    from core.ensemble import FanoutResponse, merge_vote

    raw = [
        FanoutResponse(model="m1", text=_cross_check_payload("supporting")),
        FanoutResponse(model="m2", text=_cross_check_payload("supporting")),
        FanoutResponse(model="m3", text=_cross_check_payload("conflicting")),
    ]
    result = merge_vote(raw, key="verdict")
    assert isinstance(result.merged, dict)
    assert result.merged["verdict"] == "supporting"
    assert result.merged["tally"] == {"supporting": 2, "conflicting": 1}
    assert result.merged["tie"] is False


def test_cross_check_ensemble_vote_tie_is_surfaced() -> None:
    """1 supporting / 1 conflicting → genuine tie. ``merge_vote`` must
    surface ``tie=True`` and pick one deterministic winner (the first
    most-common entry in the Counter, which is the first-encountered
    value among the tied tally entries). The downstream engine consumer
    can read ``tie`` from the merged dict and choose to log a warning
    or fall through to a tie-break heuristic; without this assertion,
    a regression in ``merge_vote``'s tie-detection (e.g. always
    returning ``tie=False``) would silently swallow disagreement
    signals on every 2-model ensemble."""
    from core.ensemble import FanoutResponse, merge_vote

    raw = [
        FanoutResponse(model="m1", text=_cross_check_payload("supporting")),
        FanoutResponse(model="m2", text=_cross_check_payload("conflicting")),
    ]
    result = merge_vote(raw, key="verdict")
    assert isinstance(result.merged, dict)
    # First-encountered most-common entry wins on ties — survivors are
    # iterated in input order so m1's "supporting" lands first.
    assert result.merged["verdict"] == "supporting"
    assert result.merged["tally"] == {"supporting": 1, "conflicting": 1}
    assert result.merged["tie"] is True


def test_cross_check_ensemble_vote_mixed_token_in_fallback_path() -> None:
    """The new ``verdict: mixed`` value must be recognised by
    ``merge_vote``'s non-JSON token-sniff fallback at
    ``core/ensemble.py:385``. Without this, a malformed survivor
    response that contains the word "mixed" but no parseable JSON
    would tally as ``unknown`` and silently distort the majority.
    Pins the post-fix token list against future regression."""
    from core.ensemble import FanoutResponse, merge_vote

    # Three survivors whose responses are NOT valid JSON; the token
    # sniff must recognise "mixed" alongside "supporting"/"conflicting"
    # so the tally reflects the model's actual stated verdict.
    raw = [
        FanoutResponse(model="m1", text="My read: the literature is mixed on this finding."),
        FanoutResponse(model="m2", text="On balance, conflicting evidence dominates here."),
        FanoutResponse(model="m3", text="The literature is mixed; both directions are present."),
    ]
    result = merge_vote(raw, key="verdict")
    assert isinstance(result.merged, dict)
    # Two "mixed" + one "conflicting" → majority mixed.
    assert result.merged["verdict"] == "mixed"
    assert result.merged["tally"]["mixed"] == 2


@pytest.mark.asyncio
async def test_cross_check_single_model_path_unchanged(tmp_path: Path) -> None:
    """Regression guard: the non-ensemble path is the default (no
    ``provider.node_ensemble`` configured) and the verdict prompt
    addition must not break the single-call flow. The single response
    is parsed for the existing supporting/conflicting/neutral/summary
    fields exactly as before. The ``verdict`` field is now optionally
    present in the JSON but the engine does not consume it on the
    single-call path — it's purely a signal for the vote merger."""
    eng = Engine(_mk_cfg(tmp_path))  # no ensemble
    chat_mock = AsyncMock(return_value=_cross_check_payload(
        "supporting", "single-model supporting note",
    ))
    eng._client = type("Stub", (), {"chat": chat_mock})()

    async def fake_search(query, **kw):  # noqa: ANN001
        return [RetrievedDoc(content="abstract", metadata={"title": "P", "doi": "10.1/z"})]
    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]

    patch = await eng._node_cross_check({
        "topic": "T",
        "analysis": {"key_findings": ["F1"], "next_step": "publish"},
    })

    entry = patch["cross_check"][0]
    assert entry["supporting"][0]["why"] == "single-model supporting note"
    assert entry["summary"] == "on-balance supporting"
    # Exactly one chat call — single-call path, no fan-out.
    assert chat_mock.await_count == 1
