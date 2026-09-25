"""Pin the tier classification + tier-2/3 derivation in core/interview.py.

Every Question carries a ``tier`` field in {1, 2, 3}:

- tier 1 — always-ask (topic / paper_format / output_kinds /
  study_depth, plus provider + provider_model for CLI / web only).
- tier 2 — auto-derive from tier-1, shown on a review screen
  with click-to-edit (title / no_simulation / clarify_mode /
  review_panel / knowledge_enabled / audience).
- tier 3 — advanced, behind a "Show advanced" toggle
  (comparative_baseline / success_metric / budget /
  knowledge_top_k).

This module pins the classification so a future edit can't quietly
demote topic to tier-2 or promote audience to tier-1 without a
test failure surfacing the intent shift.
"""
from __future__ import annotations

from core.interview import (
    QUESTIONS,
    derive_tier2,
    derive_tier3,
    questions_for_tier,
    smart_default_audience,
    smart_default_clarify_mode,
    smart_default_review_panel,
)


# ---- Tier-1 contract ----


def test_tier1_cli_asks_the_topic_what_it_is_for_and_the_model() -> None:
    """The first screen asks what only the person can say: the topic, what the result is for (it sets how strictly the
    quest is checked, so it is asked, never hidden) and the provider and model. The paper format, the deliverables and
    the study depth are worked out from the topic and shown on the review screen; the multi-model ensemble is in
    Advanced. The author line comes last and is asked only while no profile is saved (core/profile.py)."""
    ids = [q.id for q in questions_for_tier(1, "cli")]
    assert ids == [
        "topic",
        "result_use",
        "provider",
        "provider_model",
        "author",
        "affiliation",
        "contact_email",
        "url",
    ]


def test_tier1_serve_matches_cli() -> None:
    """Web (frontend=serve) sees the same tier-1 set as CLI — both
    are unattended-by-Copilot frontends, so provider + model are
    real choices."""
    cli_ids = [q.id for q in questions_for_tier(1, "cli")]
    serve_ids = [q.id for q in questions_for_tier(1, "serve")]
    assert cli_ids == serve_ids


def test_tier1_vscode_asks_the_topic_and_what_it_is_for() -> None:
    """VSCode pins provider=vscode_extension and takes the Copilot model the person picked in the chat picker, so its
    first screen is the topic and what the result is for (and the author line on the first interview)."""
    ids = [q.id for q in questions_for_tier(1, "vscode")]
    assert ids == [
        "topic",
        "result_use",
        "author",
        "affiliation",
        "contact_email",
        "url",
    ]


def test_author_line_questions_are_optional_one_line_text() -> None:
    """Every author-line question accepts a blank answer (default "") and
    can be changed mid-quest without re-running any LLM node."""
    from core.interview import STAGE_INVALIDATION

    for qid in ("author", "affiliation", "contact_email", "url"):
        q = next(q for q in QUESTIONS if q.id == qid)
        assert (q.kind, q.default, q.tier, q.mid_quest_editable) == ("text", "", 1, True)
        assert STAGE_INVALIDATION[qid] == ()


def test_topic_is_tier1_and_not_mid_quest_editable() -> None:
    """topic is THE input — must always be asked, never editable
    mid-quest (that would invalidate every downstream node)."""
    topic_q = next(q for q in QUESTIONS if q.id == "topic")
    assert topic_q.tier == 1
    assert topic_q.mid_quest_editable is False


# ---- Tier-2 contract ----


def test_tier2_covers_the_derived_fields() -> None:
    """Tier-2 is the auto-derive set that lands on the review
    screen. New fields land here when they're cheaply derivable
    from tier-1 answers — ``knowledge_top_k`` joined this tier
    when the Axon RAG cap became user-tunable (separate from the
    web/external cap), and ``pause_for_user_input`` joined when
    the mid-quest pause-drop gate was added."""
    ids = [q.id for q in questions_for_tier(2, "cli")]
    assert set(ids) == {
        "paper_format",
        "output_kinds",
        "study_depth",
        "title",
        "no_simulation",
        "survey_mode",
        "clarify_mode",
        "review_panel",
        "pause_for_user_input",
        "knowledge_enabled",
        "web_research",
        "audience",
        "knowledge_top_k",
    }


def test_derive_tier2_slugs_title_from_topic() -> None:
    """title falls out of slugify(topic)."""
    out = derive_tier2({"topic": "Why Do EUV Contacts Fail?"})
    assert out["title"] == "why-do-euv-contacts-fail"


def test_derive_tier2_picks_no_simulation_for_prose_formats() -> None:
    """prose paper_format (essay / report / policy_brief / whitepaper)
    → no_simulation defaults to True."""
    for fmt in ("essay", "report", "policy_brief", "whitepaper"):
        out = derive_tier2({"topic": "x", "paper_format": fmt})
        assert out["no_simulation"] is True, f"prose fmt {fmt} should default no_simulation=True"


def test_derive_tier2_keeps_simulation_for_scientific_formats() -> None:
    """generic / neurips / iclr / ieee_access / nature_mi default to
    no_simulation=False — they expect a Python experiment."""
    for fmt in ("generic", "neurips", "iclr", "ieee_access", "nature_mi"):
        out = derive_tier2({"topic": "x", "paper_format": fmt})
        assert out["no_simulation"] is False


def test_derive_tier2_defaults_audience_external() -> None:
    """audience defaults to external — safer for a one-shot paper."""
    out = derive_tier2({"topic": "x", "paper_format": "generic"})
    assert out["audience"] == "external"


def test_derive_tier2_defaults_clarify_auto_and_three_persona_panel() -> None:
    """clarify_mode=auto and a 3-persona review panel are the
    smart-default helpers' picks. The panel default flipped from
    [] (single reviewer, cost-conscious) to the 3-persona list
    because the methodologist must fire for the non-bypassable
    must-flag enforcement to take effect — defaulting to no panel
    silently loses that protection."""
    out = derive_tier2({"topic": "x", "paper_format": "generic", "result_use": "explore"})
    assert out["clarify_mode"] == "auto"
    assert out["review_panel"] == [
        "methodologist", "statistician", "devil_advocate",
    ]
    # Research (the default) and a decision add the reviewer the research profile requires, before any review
    # screen shows the panel.
    for use in ("research", "decision", None):
        partial = {"topic": "x", "paper_format": "generic", **({"result_use": use} if use else {})}
        assert derive_tier2(partial)["review_panel"] == [
            "methodologist", "statistician", "devil_advocate", "reproducibility",
        ]


def test_smart_default_audience_and_clarify_are_static() -> None:
    """The two helpers don't read partial state today — keep that
    contract so adding new tier-1 fields can't accidentally make
    audience topic-dependent."""
    assert smart_default_audience({}) == "external"
    assert smart_default_audience({"topic": "anything", "paper_format": "report"}) == "external"
    assert smart_default_clarify_mode({}) == "auto"
    assert smart_default_review_panel({"result_use": "explore"}) == [
        "methodologist", "statistician", "devil_advocate",
    ]


# ---- Tier-3 contract ----


def test_tier3_covers_the_advanced_fields() -> None:
    """Tier-3 hides behind a 'Show advanced' toggle on each frontend.
    Three are topic-tuned (preflight LLM call suggests a value);
    ``knowledge_external_top_k`` is the web-search cap (Axon RAG cap
    ``knowledge_top_k`` lives in Tier-2); ``max_iterations`` is the
    design-revise loop hard cap; ``node_models`` is the per-node model
    override (comma-separated node:model pairs, empty by default).
    ``ensemble_profile`` / ``ensemble_models`` are here too: the ensemble
    multiplies the cost and its models are the person's to name. ``provider_base_url`` /
    ``provider_api_key_env`` / ``provider_fixed_temperature`` are the
    HTTP-direct-transport connection overrides (a custom OpenAI-compatible
    endpoint, its key variable, a model-mandated fixed temperature) — CLI/
    serve only, same as ``provider``/``provider_model`` in tier 1."""
    ids = [q.id for q in questions_for_tier(3, "cli")]
    assert set(ids) == {
        "supply_papers",
        "pause_for_plan",
        "comparative_baseline",
        "success_metric",
        "budget",
        "knowledge_external_top_k",
        "max_iterations",
        "paper_style",
        "poster_size",
        "node_models",
        "reasoning_effort",
        "page_limit",
        "provider_base_url",
        "provider_api_key_env",
        "provider_fixed_temperature",
        "ensemble_profile",
        "ensemble_models",
    }


def test_derive_tier3_returns_empty_strings_until_preflight_fills_them() -> None:
    """No smart-default for the three text slots — the preflight LLM
    call fills them at quest start. The interview just collects an
    override if the user wants to type one upfront."""
    out = derive_tier3({"topic": "x", "paper_format": "generic"})
    assert out["comparative_baseline"] == ""
    assert out["success_metric"] == ""
    assert out["budget"] == ""


def test_derive_tier3_external_top_k_default_is_20() -> None:
    """knowledge_external_top_k carries a real static default (20) —
    web search is coarser than RAG, so breadth matters even without
    a preflight LLM call. Comprehensive reviews bump to 30."""
    out = derive_tier3({"topic": "x", "paper_format": "generic"})
    assert out["knowledge_external_top_k"] == 20
    out = derive_tier3({"topic": "x", "paper_format": "generic",
                         "study_depth": "comprehensive review"})
    assert out["knowledge_external_top_k"] == 30


# ---- Cross-tier invariants ----


def test_every_question_has_a_tier_in_1_2_3() -> None:
    """No question may forget its tier. If a new question is added
    without a tier, frontends won't know whether to ask it or
    auto-derive it."""
    for q in QUESTIONS:
        assert q.tier in (1, 2, 3), f"{q.id}: tier={q.tier!r}"


def test_tiers_partition_the_full_question_list() -> None:
    """Every question appears in exactly one tier. Catches a future
    edit that lands a new tier without updating any of the
    helpers."""
    by_id = {q.id for q in QUESTIONS}
    union = (
        {q.id for q in questions_for_tier(1, "cli")}
        | {q.id for q in questions_for_tier(2, "cli")}
        | {q.id for q in questions_for_tier(3, "cli")}
    )
    # Tier-1 has CLI-only entries (provider, provider_model). Cover
    # VSCode's 2 missing entries by adding them back from raw list.
    missing = by_id - union
    assert missing == set(), f"unclassified question ids: {missing}"



def test_derive_tier2_works_the_paper_format_out_first_and_the_depth_from_it() -> None:
    """The paper format is derived now, not asked, and what follows from it (the study depth) reads the derived one."""
    derived = derive_tier2({"topic": "A policy brief on congestion pricing"})
    assert derived["paper_format"] == "generic" and derived["output_kinds"] == ["paper_md", "paper_pdf"]
    assert derived["study_depth"]
    held = derive_tier2({"topic": "t", "paper_format": "policy_brief"})
    from core.interview import smart_default_study_depth
    assert held["study_depth"] == smart_default_study_depth({"paper_format": "policy_brief"})
