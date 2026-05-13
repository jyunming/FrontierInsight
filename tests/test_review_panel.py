"""Phase N — reviewer panel tests.

Validates:
  - `_load_persona_prefix`: built-in personas resolve to their prefix
    files; custom persona names fall back to the generic template.
  - `_aggregate_panel_reviews`: deterministic aggregation rules
    (verdict, score median, strengths intersection, weaknesses union,
    suggestions attribution, agreement classification).
  - `_node_review`: panel mode fires N parallel persona reviews,
    aggregates, and calls a moderator for the prose.
  - Backwards compat: empty `review_panel` preserves single-reviewer
    behavior exactly.
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
from core.engine import (
    Engine, _aggregate_panel_reviews, _load_persona_prefix,
)


def _mk_cfg(tmp_path: Path, *, panel=None, node_models=None) -> Config:
    return Config(
        topic="panel review tests",
        title="rp",
        provider=ProviderConfig(name="openai", model="default", node_models=node_models),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="off",
            ideate_reflect=False,
            review_panel=panel or [],
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


# --- _load_persona_prefix --------------------------------------------------


def test_load_persona_prefix_builtin_personas_each_have_distinct_prefix() -> None:
    """All four built-in personas must produce non-empty, distinct
    prefixes. Catches accidental file-loading regressions."""
    builtins = ["methodologist", "statistician", "devil_advocate", "reproducibility"]
    # A first-token in the persona file that the test asserts shows up
    # — these are the human-readable role names the prefix files use.
    expected_token = {
        "methodologist": "methodologist",
        "statistician": "statistician",
        "devil_advocate": "devil",   # file says "Devil's advocate"
        "reproducibility": "reproducibility",
    }
    prefixes = {p: _load_persona_prefix(p) for p in builtins}
    for name, text in prefixes.items():
        assert text, f"persona {name!r} produced empty prefix"
        assert expected_token[name] in text.lower(), (
            f"persona {name!r} prefix did not contain {expected_token[name]!r}"
        )
    assert len({v for v in prefixes.values()}) == len(builtins)


def test_load_persona_prefix_custom_name_falls_back_to_generic_template() -> None:
    """A persona name with no corresponding file falls back to the
    generic template so users can declare ad-hoc roles in YAML."""
    out = _load_persona_prefix("hardware_engineer")
    assert "hardware_engineer" in out


def test_load_persona_prefix_rejects_invalid_names() -> None:
    """Invalid persona names (path-traversal attempts, etc.) raise."""
    with pytest.raises(ValueError):
        _load_persona_prefix("../etc/passwd")
    with pytest.raises(ValueError):
        _load_persona_prefix("Bad Name")
    with pytest.raises(ValueError):
        _load_persona_prefix("UPPERCASE")


# --- _aggregate_panel_reviews — the deterministic rules ------------------


def _r(persona: str, verdict: str, score: int, *, strengths=None, weaknesses=None,
       suggestions=None, blocking="") -> dict:
    return {
        "persona": persona, "verdict": verdict, "score": score,
        "strengths": strengths or [], "weaknesses": weaknesses or [],
        "suggestions": suggestions or [], "blocking": blocking,
    }


def test_aggregate_empty_panel_returns_default_accept() -> None:
    agg = _aggregate_panel_reviews([])
    assert agg["verdict"] == "accept"
    assert agg["agreement"] == "unanimous"


def test_aggregate_unanimous_accept() -> None:
    agg = _aggregate_panel_reviews([
        _r("m", "accept", 4), _r("s", "accept", 4), _r("d", "accept", 5),
    ])
    assert agg["verdict"] == "accept"
    assert agg["agreement"] == "unanimous"
    assert agg["score"] == 4  # median


def test_aggregate_unanimous_revise() -> None:
    agg = _aggregate_panel_reviews([
        _r("m", "revise", 2), _r("s", "revise", 3), _r("d", "revise", 1),
    ])
    assert agg["verdict"] == "revise"
    assert agg["agreement"] == "unanimous"
    assert agg["score"] == 2


def test_aggregate_majority_revise() -> None:
    """2-1 in favor of revise → revise."""
    agg = _aggregate_panel_reviews([
        _r("m", "revise", 2), _r("s", "revise", 3), _r("d", "accept", 4),
    ])
    assert agg["verdict"] == "revise"
    assert agg["agreement"] == "split"


def test_aggregate_majority_accept() -> None:
    """2-1 in favor of accept → accept (no low-score-revise dragging)."""
    agg = _aggregate_panel_reviews([
        _r("m", "accept", 4), _r("s", "accept", 4), _r("d", "revise", 3),
    ])
    assert agg["verdict"] == "accept"
    assert agg["agreement"] == "split"


def test_aggregate_low_revise_dominates_even_with_majority_accept() -> None:
    """A single very low score (< 3) on a revise vote forces revise,
    even if the majority would have accepted. The conservative rule —
    a strongly-objecting reviewer can block."""
    agg = _aggregate_panel_reviews([
        _r("m", "accept", 4), _r("s", "accept", 4), _r("d", "revise", 1),
    ])
    assert agg["verdict"] == "revise"


def test_aggregate_tie_goes_to_revise() -> None:
    agg = _aggregate_panel_reviews([
        _r("m", "accept", 4), _r("s", "revise", 4),  # exactly tied, both >= 3
    ])
    assert agg["verdict"] == "revise"


def test_aggregate_strengths_are_intersection() -> None:
    agg = _aggregate_panel_reviews([
        _r("m", "accept", 4, strengths=["A", "B", "C"]),
        _r("s", "accept", 4, strengths=["B", "C", "D"]),
        _r("d", "accept", 4, strengths=["C"]),
    ])
    assert agg["strengths"] == ["C"]  # only every reviewer agreed


def test_aggregate_weaknesses_are_union_deduped() -> None:
    agg = _aggregate_panel_reviews([
        _r("m", "accept", 4, weaknesses=["W1", "w1"]),  # same after lowercase
        _r("s", "accept", 4, weaknesses=["W1", "W2"]),
    ])
    # Deduplicated case-insensitively but original form preserved.
    assert len(agg["weaknesses"]) == 2
    assert any("W1" in w for w in agg["weaknesses"])
    assert any("W2" in w for w in agg["weaknesses"])


def test_aggregate_suggestions_carry_persona_attribution() -> None:
    agg = _aggregate_panel_reviews([
        _r("methodologist", "accept", 4, suggestions=["increase N"]),
        _r("statistician", "accept", 4, suggestions=["report CIs"]),
    ])
    assert "[methodologist] increase N" in agg["suggestions"]
    assert "[statistician] report CIs" in agg["suggestions"]


def test_aggregate_blocking_picks_first_non_empty() -> None:
    agg = _aggregate_panel_reviews([
        _r("m", "accept", 4),
        _r("s", "revise", 2, blocking="missing CIs"),
        _r("d", "revise", 2, blocking="alt explanation likely"),
    ])
    # Either is acceptable per the rule "first non-empty"; the
    # important thing is `blocking` is not empty when any persona set it.
    assert agg["blocking"] in {"missing CIs", "alt explanation likely"}


# --- _node_review panel path ----------------------------------------------


def _panel_responses() -> dict[str, str]:
    """Canned LLM responses per persona, plus the moderator's prose."""
    return {
        "methodologist": json.dumps({
            "verdict": "accept", "score": 4,
            "strengths": ["clear design"], "weaknesses": [],
            "suggestions": [], "blocking": "",
        }),
        "statistician": json.dumps({
            "verdict": "revise", "score": 2,
            "strengths": [], "weaknesses": ["no CIs reported"],
            "suggestions": ["bootstrap a 95% CI on the headline number"],
            "blocking": "",
        }),
        "devil_advocate": json.dumps({
            "verdict": "accept", "score": 3,
            "strengths": ["clear design"], "weaknesses": ["alt explanation: confound"],
            "suggestions": [], "blocking": "",
        }),
        "moderator": json.dumps({
            "verdict": "revise", "score": 3, "agreement": "split",
            "strengths": [], "weaknesses": [],
            "suggestions": [
                "[statistician] bootstrap a 95% CI on the headline number",
            ],
            "blocking": "",
            "rationale": "statistician's low-score revise dominates; methodologist accepted but the bar is uncertainty quantification",
        }),
    }


@pytest.mark.asyncio
async def test_review_panel_fires_all_personas_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three-persona panel → three persona LLM calls + one moderator
    call = 4 total. Final verdict comes from the deterministic
    aggregator; rationale prose from the moderator."""
    cfg = _mk_cfg(tmp_path, panel=["methodologist", "statistician", "devil_advocate"])
    eng = Engine(cfg)

    # Make sure paper_md exists on disk so the review prompt builds.
    paper_path = eng.quest_root / "paper" / "paper.md"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text("# title\n\nbody.\n", encoding="utf-8")

    responses = _panel_responses()
    seen_nodes: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        text = messages[-1]["content"]
        if "review moderator" in text or "Per-persona reviews" in text:
            seen_nodes.append("moderator")
            return responses["moderator"]
        # Distinct keywords from each persona's prefix file.
        if "Methodologist." in text:
            seen_nodes.append("methodologist")
            return responses["methodologist"]
        if "Statistician." in text:
            seen_nodes.append("statistician")
            return responses["statistician"]
        if "Devil's advocate." in text:
            seen_nodes.append("devil_advocate")
            return responses["devil_advocate"]
        return "{}"
    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    from core.provider import LLMClient, resolve_endpoint
    ep = resolve_endpoint(ProviderConfig(name="openai", model="x"))
    eng._client = LLMClient(ep)

    state = {
        "topic": "t", "design": {}, "analysis": {}, "paper_md": str(paper_path),
        "iteration": 0,
    }
    patch = await eng._node_review(state)

    # 3 personas + 1 moderator = 4 LLM calls.
    assert seen_nodes.count("moderator") == 1
    assert set(seen_nodes[:3]) == {"methodologist", "statistician", "devil_advocate"}

    # `review_panel` carries the per-persona dicts (for the GUI).
    panel = patch["review_panel"]
    assert len(panel) == 3
    by_persona = {r["persona"]: r for r in panel}
    assert by_persona["methodologist"]["verdict"] == "accept"
    assert by_persona["statistician"]["verdict"] == "revise"

    # Aggregator-driven final verdict: low-score-revise dominates.
    final = patch["review"]
    assert final["verdict"] == "revise"
    assert final["agreement"] == "split"
    # Median score across [4, 2, 3] → 3.
    assert final["score"] == 3
    # Iteration was bumped by the revise verdict.
    assert patch["iteration"] == 1
    # Moderator's rationale was kept (prose-level).
    assert "statistician" in final.get("rationale", "")


@pytest.mark.asyncio
async def test_review_panel_empty_preserves_single_reviewer_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`review_panel=[]` (the default) must NOT invoke the panel path —
    exactly one LLM call, exact same JSON shape as before Phase N."""
    cfg = _mk_cfg(tmp_path, panel=[])
    eng = Engine(cfg)
    paper_path = eng.quest_root / "paper" / "paper.md"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text("# t\n", encoding="utf-8")

    n_calls = {"v": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        n_calls["v"] += 1
        return json.dumps({"verdict": "accept", "score": 4, "strengths": [],
                           "weaknesses": [], "suggestions": [], "blocking": ""})
    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    from core.provider import LLMClient, resolve_endpoint
    ep = resolve_endpoint(ProviderConfig(name="openai", model="x"))
    eng._client = LLMClient(ep)

    patch = await eng._node_review({
        "topic": "t", "design": {}, "analysis": {}, "paper_md": str(paper_path),
    })

    assert n_calls["v"] == 1
    assert "review_panel" not in patch
    assert patch["review"]["verdict"] == "accept"


@pytest.mark.asyncio
async def test_review_panel_uses_per_persona_model_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `provider.node_models` declares
    `review_panel.<persona>: <model>`, each persona's LLM call uses
    its assigned model. This is the cross-model debate plumbing —
    different personas can run on different models within one quest."""
    cfg = _mk_cfg(
        tmp_path,
        panel=["methodologist", "devil_advocate"],
        node_models={
            "review_panel.methodologist": "gpt-5",
            "review_panel.devil_advocate": "claude-opus-4-7",
            "review_moderator": "gpt-5-mini",
        },
    )
    eng = Engine(cfg)
    paper_path = eng.quest_root / "paper" / "paper.md"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text("# t\n", encoding="utf-8")

    seen: list[tuple[str, str | None]] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        text = messages[-1]["content"]
        if "review moderator" in text or "Per-persona reviews" in text:
            persona = "moderator"
        elif "Methodologist." in text:
            persona = "methodologist"
        elif "Devil's advocate." in text:
            persona = "devil_advocate"
        else:
            persona = "?"
        seen.append((persona, kw.get("model")))
        return json.dumps({"verdict": "accept", "score": 4, "strengths": [],
                           "weaknesses": [], "suggestions": [], "blocking": ""})
    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    from core.provider import LLMClient, resolve_endpoint
    ep = resolve_endpoint(ProviderConfig(name="openai", model="x"))
    eng._client = LLMClient(ep)

    await eng._node_review({
        "topic": "t", "design": {}, "analysis": {}, "paper_md": str(paper_path),
    })

    by_persona = {p: m for p, m in seen}
    assert by_persona["methodologist"] == "gpt-5"
    assert by_persona["devil_advocate"] == "claude-opus-4-7"
    assert by_persona["moderator"] == "gpt-5-mini"


def test_aggregate_carries_rigor_and_depth_medians_when_present() -> None:
    """Regression for PR #34 review: the aggregator was returning a
    fixed dict without rigor_score / depth_score, silently dropping
    the per-persona depth signal that motivates a revise. Pin that
    medians of both axes flow through."""
    panel = [
        {**_r("m", "accept", 4), "rigor_score": 5, "depth_score": 3},
        {**_r("s", "accept", 4), "rigor_score": 4, "depth_score": 2},
        {**_r("d", "accept", 5), "rigor_score": 4, "depth_score": 4},
    ]
    agg = _aggregate_panel_reviews(panel)
    assert agg["verdict"] == "accept"
    # rigor: sorted [4,4,5] → median 4. depth: sorted [2,3,4] → median 3.
    assert agg["rigor_score"] == 4
    assert agg["depth_score"] == 3


def test_aggregate_falls_back_to_3_when_personas_omit_rigor_depth() -> None:
    """Personas that don't return the new axes (older prompts, parse
    failures) should fall back to 3, not crash the aggregator."""
    panel = [_r("m", "accept", 4), _r("s", "accept", 5)]
    agg = _aggregate_panel_reviews(panel)
    assert agg["rigor_score"] == 3
    assert agg["depth_score"] == 3


def test_aggregate_empty_panel_provides_rigor_and_depth_defaults() -> None:
    agg = _aggregate_panel_reviews([])
    assert agg["rigor_score"] == 3
    assert agg["depth_score"] == 3
