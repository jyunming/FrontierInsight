"""Unit tests for ``core/ensemble.py``.

The primitive has three merge strategies and two failure modes; we
cover each. Every test uses an in-process async fake ``chat_fn``
(``FakeTransport.chat`` is an ``async def``, exercised under
``pytest.mark.asyncio``) — no network, no bridge, no transport — so
the tests run in milliseconds and stay reliable on CI.
"""
from __future__ import annotations

import json

import pytest

from core.ensemble import (
    EnsembleError,
    EnsembleResult,
    FanoutResponse,
    cost_jsonl_entries,
    fanout_chat,
    merge_synthesize,
    merge_tournament,
    merge_vote,
)


# ---------------------------------------------------------------------------
# Fake transport: records calls, returns scripted responses.
# ---------------------------------------------------------------------------


class FakeTransport:
    """Records every (model, prompt-tail) and returns the next scripted
    text or raises a scripted exception. Lets each test pin both what
    the ensemble asked for AND what it got back."""

    def __init__(self):
        self.calls: list[dict] = []
        # Map model → list of (status, payload) where status is "ok" or "err".
        # Each call to that model pops the next entry; running out raises.
        self.scripts: dict[str, list[tuple[str, str]]] = {}

    def script(self, model: str, responses: list[tuple[str, str]]) -> None:
        self.scripts[model] = list(responses)

    async def chat(self, messages, *, temperature, model, node):
        self.calls.append({"model": model, "node": node, "temperature": temperature,
                           "n_messages": len(messages)})
        seq = self.scripts.get(model)
        if not seq:
            raise RuntimeError(f"no scripted response for model={model!r}")
        status, payload = seq.pop(0)
        if status == "err":
            raise RuntimeError(payload)
        return payload


@pytest.fixture
def fake() -> FakeTransport:
    return FakeTransport()


# ---------------------------------------------------------------------------
# fanout_chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_all_three_succeed(fake: FakeTransport) -> None:
    """Happy path: every model returns text, all three FanoutResponses
    carry ok=True and the recorded model identity matches."""
    fake.script("gpt-5", [("ok", "from gpt-5")])
    fake.script("claude-opus", [("ok", "from claude-opus")])
    fake.script("gemini-pro", [("ok", "from gemini-pro")])

    results = await fanout_chat(
        [{"role": "user", "content": "test"}],
        ["gpt-5", "claude-opus", "gemini-pro"],
        chat_fn=fake.chat, node="ideate",
    )
    assert len(results) == 3
    assert all(r.ok for r in results)
    assert [r.model for r in results] == ["gpt-5", "claude-opus", "gemini-pro"]
    assert [r.text for r in results] == ["from gpt-5", "from claude-opus", "from gemini-pro"]
    # Each call was logged with the ensembled-node tag.
    assert {c["node"] for c in fake.calls} == {
        "ideate.ensemble[gpt-5]", "ideate.ensemble[claude-opus]", "ideate.ensemble[gemini-pro]",
    }


@pytest.mark.asyncio
async def test_fanout_one_fails_two_survive(fake: FakeTransport) -> None:
    """Lenient semantics: a single model error is captured as
    FanoutResponse.error; the other two still return ok responses."""
    fake.script("gpt-5", [("ok", "good")])
    fake.script("claude-opus", [("err", "rate limit")])
    fake.script("gemini-pro", [("ok", "good")])

    results = await fanout_chat(
        [{"role": "user", "content": "x"}],
        ["gpt-5", "claude-opus", "gemini-pro"],
        chat_fn=fake.chat, node="analyze",
    )
    assert len(results) == 3
    statuses = [(r.model, r.ok) for r in results]
    assert statuses == [("gpt-5", True), ("claude-opus", False), ("gemini-pro", True)]
    assert results[1].error and "rate limit" in results[1].error


@pytest.mark.asyncio
async def test_fanout_all_fail_raises(fake: FakeTransport) -> None:
    """Strict semantics for the all-fail case: no survivors → raise
    EnsembleError. The merger never gets a chance to produce a
    useless empty output."""
    fake.script("a", [("err", "boom-1")])
    fake.script("b", [("err", "boom-2")])
    with pytest.raises(EnsembleError) as exc_info:
        await fanout_chat(
            [{"role": "user", "content": "x"}], ["a", "b"],
            chat_fn=fake.chat, node="ideate",
        )
    assert "all 2" in str(exc_info.value)
    assert "boom-1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_fanout_empty_models_raises() -> None:
    """Defensive: empty models list is a caller bug, not a no-op."""
    with pytest.raises(EnsembleError):
        await fanout_chat([{"role": "user", "content": "x"}], [],
                          chat_fn=lambda *a, **k: None, node="ideate")


# ---------------------------------------------------------------------------
# merge_tournament
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tournament_moderator_picks_winner(fake: FakeTransport) -> None:
    """Moderator emits ``WINNER: [B]`` → the second candidate is returned
    verbatim (not rewritten)."""
    raw = [
        FanoutResponse(model="m1", text="bad answer"),
        FanoutResponse(model="m2", text="GOOD ANSWER"),
        FanoutResponse(model="m3", text="ok answer"),
    ]
    fake.script("moderator", [("ok", "WINNER: [B]\nB is the most rigorous.")])
    result = await merge_tournament(
        raw, moderator_model="moderator", chat_fn=fake.chat,
        node="ideate", prompt_summary="brainstorm",
    )
    assert result.merged == "GOOD ANSWER"
    assert "[B]" in result.notes and "m2" in result.notes


@pytest.mark.asyncio
async def test_tournament_unparseable_moderator_falls_back(fake: FakeTransport) -> None:
    """Moderator returns something we can't parse → take the first
    survivor + flag in notes. The node keeps running."""
    raw = [
        FanoutResponse(model="m1", text="first"),
        FanoutResponse(model="m2", text="second"),
    ]
    fake.script("mod", [("ok", "hmm, I think B but also kinda A")])
    result = await merge_tournament(
        raw, moderator_model="mod", chat_fn=fake.chat, node="ideate",
    )
    assert result.merged == "first"
    assert "unparseable" in result.notes.lower()


@pytest.mark.asyncio
async def test_tournament_single_survivor_skips_moderator(fake: FakeTransport) -> None:
    """One fan-out success → no moderator call (saves an LLM round-trip);
    survivor returned with explanatory note."""
    raw = [
        FanoutResponse(model="m1", text="only"),
        FanoutResponse(model="m2", error="failed"),
    ]
    result = await merge_tournament(
        raw, moderator_model="mod", chat_fn=fake.chat, node="ideate",
    )
    assert result.merged == "only"
    assert "single-survivor" in result.notes
    assert fake.calls == []  # moderator never called


@pytest.mark.asyncio
async def test_tournament_empty_survivors_raises(fake: FakeTransport) -> None:
    """Defensive contract: passing an all-failed raw list directly to
    the merger surfaces EnsembleError, not IndexError. fanout_chat
    already enforces this upstream but the merger is callable on its
    own."""
    raw = [FanoutResponse(model="m1", error="x"),
           FanoutResponse(model="m2", error="y")]
    with pytest.raises(EnsembleError):
        await merge_tournament(
            raw, moderator_model="mod", chat_fn=fake.chat, node="ideate",
        )


@pytest.mark.asyncio
async def test_tournament_over_26_candidates_raises(fake: FakeTransport) -> None:
    """27 candidates breaks the single-letter A-Z labeling scheme; the
    merger refuses rather than emit ``[\\` as a label."""
    raw = [FanoutResponse(model=f"m{i}", text=f"c{i}") for i in range(27)]
    with pytest.raises(EnsembleError) as exc:
        await merge_tournament(
            raw, moderator_model="mod", chat_fn=fake.chat, node="ideate",
        )
    assert "26" in str(exc.value)


@pytest.mark.asyncio
async def test_synthesize_empty_survivors_raises(fake: FakeTransport) -> None:
    """Same defensive contract for synthesize: no survivors → raise,
    don't return an empty-string merged result."""
    raw = [FanoutResponse(model="m1", error="x")]
    with pytest.raises(EnsembleError):
        await merge_synthesize(
            raw, moderator_model="mod", chat_fn=fake.chat, node="analyze",
        )


@pytest.mark.asyncio
async def test_synthesize_disagreement_no_double_count(fake: FakeTransport) -> None:
    """A single ``⚠ Disagreement:`` line counts ONCE, not twice. The
    old regex matched both ``⚠ Disagreement`` and ``Disagreement:`` on
    the same line and inflated the score."""
    raw = [FanoutResponse(model="m1", text="a"), FanoutResponse(model="m2", text="b")]
    merged_text = "Summary text.\n⚠ Disagreement: a vs b. b wins."
    fake.script("mod", [("ok", merged_text)])
    result = await merge_synthesize(
        raw, moderator_model="mod", chat_fn=fake.chat, node="analyze",
    )
    # One flag → score == 1/5 == 0.2 exactly. The buggy version saw 2 → 0.4.
    assert result.disagreement_score == pytest.approx(0.2, abs=1e-3)


@pytest.mark.asyncio
async def test_tournament_no_moderator_returns_first_survivor(fake: FakeTransport) -> None:
    """``moderator_model=None`` → can't pick by quality; default to the
    first survivor + note. Useful for tests + ultra-cheap configs."""
    raw = [
        FanoutResponse(model="m1", text="alpha"),
        FanoutResponse(model="m2", text="beta"),
    ]
    result = await merge_tournament(
        raw, moderator_model=None, chat_fn=fake.chat, node="ideate",
    )
    assert result.merged == "alpha"
    assert "no moderator" in result.notes


# ---------------------------------------------------------------------------
# merge_synthesize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_merges_and_counts_disagreements(fake: FakeTransport) -> None:
    """Moderator merges N analyses + flags disagreements with the
    explicit ⚠ marker; the result carries a disagreement_score derived
    from the count of those markers."""
    raw = [
        FanoutResponse(model="m1", text="The trend is up."),
        FanoutResponse(model="m2", text="The trend is down."),
    ]
    merged_text = (
        "The trend direction is contested.\n"
        "⚠ Disagreement: model m1 says up; model m2 says down. "
        "m2 is more defensible because the dataset post-2024 reverses."
    )
    fake.script("mod", [("ok", merged_text)])
    result = await merge_synthesize(
        raw, moderator_model="mod", chat_fn=fake.chat,
        node="analyze", prompt_summary="trend analysis",
    )
    assert result.merged == merged_text
    assert result.disagreement_score > 0.0
    assert "disagreement flag" in result.notes.lower()


@pytest.mark.asyncio
async def test_tournament_moderator_failure_falls_back(fake: FakeTransport) -> None:
    """When the moderator call itself raises (e.g. transport timeout),
    the merger falls back to first survivor + note. Fan-out work isn't
    wasted, and the engine never has to catch the moderator's raw
    transport exception."""
    raw = [
        FanoutResponse(model="m1", text="alpha"),
        FanoutResponse(model="m2", text="beta"),
    ]
    fake.script("mod", [("err", "timeout")])
    result = await merge_tournament(
        raw, moderator_model="mod", chat_fn=fake.chat, node="ideate",
    )
    assert result.merged == "alpha"
    assert "moderator call failed" in result.notes
    assert "timeout" in result.notes
    # No moderator_model set → cost row will skip the moderator entry.
    assert result.moderator_model is None


@pytest.mark.asyncio
async def test_synthesize_moderator_failure_concatenates(fake: FakeTransport) -> None:
    """Same fallback policy for synthesize: moderator failure → raw
    concatenation of survivors so no analysis is lost."""
    raw = [
        FanoutResponse(model="m1", text="alpha report"),
        FanoutResponse(model="m2", text="beta report"),
    ]
    fake.script("mod", [("err", "rate limit")])
    result = await merge_synthesize(
        raw, moderator_model="mod", chat_fn=fake.chat, node="analyze",
    )
    assert "alpha report" in result.merged
    assert "beta report" in result.merged
    assert "moderator call failed" in result.notes
    assert result.moderator_model is None


@pytest.mark.asyncio
async def test_synthesize_no_moderator_concatenates(fake: FakeTransport) -> None:
    """Without a moderator, the merger crudely concatenates the
    survivors so no information is lost. Truthful > pretty."""
    raw = [
        FanoutResponse(model="m1", text="alpha report"),
        FanoutResponse(model="m2", text="beta report"),
    ]
    result = await merge_synthesize(
        raw, moderator_model=None, chat_fn=fake.chat, node="analyze",
    )
    assert "m1" in result.merged and "m2" in result.merged
    assert "alpha report" in result.merged
    assert "beta report" in result.merged


# ---------------------------------------------------------------------------
# merge_vote (no LLM call — pure tally)
# ---------------------------------------------------------------------------


def test_vote_majority_wins() -> None:
    """Three models classify a finding; majority verdict wins."""
    raw = [
        FanoutResponse(model="m1", text=json.dumps({"verdict": "supporting"})),
        FanoutResponse(model="m2", text=json.dumps({"verdict": "supporting"})),
        FanoutResponse(model="m3", text=json.dumps({"verdict": "neutral"})),
    ]
    result = merge_vote(raw, key="verdict")
    assert result.merged["verdict"] == "supporting"
    assert result.merged["tally"] == {"supporting": 2, "neutral": 1}
    assert result.merged["tie"] is False
    assert result.disagreement_score == pytest.approx(1.0 / 3, abs=1e-3)


def test_vote_tie_flagged() -> None:
    """1-1-1 across three buckets → tie flagged in result."""
    raw = [
        FanoutResponse(model="m1", text=json.dumps({"verdict": "supporting"})),
        FanoutResponse(model="m2", text=json.dumps({"verdict": "conflicting"})),
        FanoutResponse(model="m3", text=json.dumps({"verdict": "neutral"})),
    ]
    result = merge_vote(raw, key="verdict")
    assert result.merged["tie"] is True
    # Disagreement maxes near 1.0 for full ties.
    assert result.disagreement_score > 0.6


def test_vote_tolerates_non_json_responses() -> None:
    """When a model returns prose instead of JSON, sniff common verdict
    tokens. Don't crash."""
    raw = [
        FanoutResponse(model="m1", text=json.dumps({"verdict": "supporting"})),
        FanoutResponse(model="m2", text="I think this is supporting evidence honestly"),
        FanoutResponse(model="m3", text=json.dumps({"verdict": "conflicting"})),
    ]
    result = merge_vote(raw, key="verdict")
    assert result.merged["verdict"] == "supporting"
    assert result.merged["tally"]["supporting"] == 2


def test_vote_all_failures_raises() -> None:
    """No surviving responses → EnsembleError (consistent with
    fanout_chat's contract)."""
    raw = [FanoutResponse(model="m1", error="x"),
           FanoutResponse(model="m2", error="y")]
    with pytest.raises(EnsembleError):
        merge_vote(raw, key="verdict")


# ---------------------------------------------------------------------------
# cost_jsonl_entries — structural breadcrumb for the cost tool.
# ---------------------------------------------------------------------------


def test_cost_jsonl_tournament_has_fanout_plus_moderator() -> None:
    raw = [
        FanoutResponse(model="m1", text="x"),
        FanoutResponse(model="m2", text="y"),
    ]
    result = EnsembleResult(merged="x", raw=raw, moderator_model="mod-7")
    rows = cost_jsonl_entries(result, base_node="ideate", merge_strategy="tournament")
    assert len(rows) == 3  # 2 fanout + 1 moderator
    roles = [r["role"] for r in rows]
    assert roles.count("fanout") == 2
    assert roles.count("moderator") == 1
    # Each fan-out row carries the model identity so the cost tool can
    # bucket spend per model.
    assert {r["model"] for r in rows if r["role"] == "fanout"} == {"m1", "m2"}
    # Moderator row carries model + ok + error too, mirroring fan-out
    # rows so cost aggregation doesn't need a role-specific branch.
    mod = next(r for r in rows if r["role"] == "moderator")
    assert mod["model"] == "mod-7"
    assert mod["ok"] is True
    assert mod["error"] is None


def test_cost_jsonl_vote_has_no_moderator_row() -> None:
    """Vote has no moderator call — pure tally — so no moderator row."""
    raw = [FanoutResponse(model="m1", text="{}"),
           FanoutResponse(model="m2", text="{}")]
    result = EnsembleResult(merged={}, raw=raw)
    rows = cost_jsonl_entries(result, base_node="cross_check", merge_strategy="vote")
    assert len(rows) == 2
    assert all(r["role"] == "fanout" for r in rows)


def test_cost_jsonl_skips_moderator_row_when_none_was_called() -> None:
    """When the merger took an early-return path (single survivor /
    no moderator configured / moderator call failed), no LLM moderator
    call actually happened — so no moderator cost row should be emitted.
    The moderator_model field on the result is the truthy signal."""
    raw = [FanoutResponse(model="m1", text="x")]
    # Single-survivor early-return → moderator_model stays None.
    result = EnsembleResult(merged="x", raw=raw, moderator_model=None)
    rows = cost_jsonl_entries(result, base_node="ideate", merge_strategy="tournament")
    assert all(r["role"] != "moderator" for r in rows)


def test_cost_jsonl_marks_failed_calls() -> None:
    """A failed fan-out call still gets a row (so the cost tool can
    show wasted attempts), with ok=False + the error message."""
    raw = [
        FanoutResponse(model="m1", text="ok"),
        FanoutResponse(model="m2", error="timeout"),
    ]
    result = EnsembleResult(merged="ok", raw=raw)
    rows = cost_jsonl_entries(result, base_node="ideate", merge_strategy="tournament")
    failed = [r for r in rows if r.get("role") == "fanout" and not r["ok"]]
    assert len(failed) == 1
    assert failed[0]["model"] == "m2"
    assert "timeout" in failed[0]["error"]
