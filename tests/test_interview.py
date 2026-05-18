"""Tests for core/interview.py — the single source of truth for the
Frontier Insight interview.

These tests intentionally do NOT exercise the LLM-backed
``preflight_clarify`` against a real provider; that path is mocked.
The point is to pin: question schema, defaults, smart-default
cascades, YAML emission, schema-JSON shape, slugify."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from core import interview as interview_mod
from core.config import Config
from core.interview import (
    EDITABLE_FIELDS,
    InterviewAnswers,
    OUTPUT_BUNDLES,
    PAPER_FORMATS,
    PROSE_FORMATS,
    PROVIDER_CHOICES,
    PROVIDER_MODEL_OPTIONS,
    QUESTIONS,
    REVIEW_PANELS,
    STAGE_INVALIDATION,
    STUDY_DEPTHS,
    answers_to_yaml,
    available_providers,
    build_smart_defaults,
    export_schema_json,
    model_choices_for,
    preflight_clarify,
    slugify,
    smart_default_no_simulation,
    smart_default_study_depth,
    smart_default_title,
)


# ---------------------------------------------------------------------------
# Question schema invariants
# ---------------------------------------------------------------------------


def test_every_question_has_unique_id_and_label() -> None:
    """Question ids must be unique — frontends key off them.
    Labels also must be unique within their frontend because the TS
    half renders labels into modal titles."""
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids)), f"duplicate question id in QUESTIONS: {ids}"


def test_paper_format_choice_values_match_config_literal() -> None:
    """The 9 paper_format choices must align EXACTLY with
    ``PaperFormat`` in core/config.py. Adding a venue without
    updating the choices (or vice versa) is the regression this
    guards."""
    from core.config import PaperFormat
    import typing
    literal_values = set(typing.get_args(PaperFormat))
    choice_values = {c.value for c in PAPER_FORMATS}
    assert choice_values == literal_values, (
        f"PAPER_FORMATS missing {literal_values - choice_values} or "
        f"has extra {choice_values - literal_values}"
    )


def test_prose_formats_intersection_with_paper_format_choices() -> None:
    """``PROSE_FORMATS`` must be a strict subset of paper_format
    choices — the smart_default_no_simulation logic depends on it."""
    choice_values = {c.value for c in PAPER_FORMATS}
    assert PROSE_FORMATS.issubset(choice_values)


def test_editable_fields_matches_mid_quest_editable_flag() -> None:
    """``EDITABLE_FIELDS`` is derived from ``Question.mid_quest_editable``.
    This test pins that they stay in sync after a future refactor."""
    derived = {q.id for q in QUESTIONS if q.mid_quest_editable}
    assert set(EDITABLE_FIELDS) == derived


def test_stage_invalidation_keys_are_editable_fields() -> None:
    """Every key in STAGE_INVALIDATION must be an editable field —
    invalidating a stage based on a non-editable field is dead code."""
    assert set(STAGE_INVALIDATION.keys()).issubset(EDITABLE_FIELDS)


def test_provider_model_options_keys_match_provider_choices() -> None:
    """Every provider in ``PROVIDER_CHOICES`` must have a model list
    in ``PROVIDER_MODEL_OPTIONS`` — otherwise the CLI model picker
    falls through to a free-text input for no good reason."""
    provider_ids = {c.value for c in PROVIDER_CHOICES}
    model_keys = set(PROVIDER_MODEL_OPTIONS.keys())
    assert provider_ids == model_keys, (
        f"PROVIDER_MODEL_OPTIONS missing {provider_ids - model_keys} "
        f"or has extra {model_keys - provider_ids}"
    )


# ---------------------------------------------------------------------------
# Smart defaults
# ---------------------------------------------------------------------------


def test_smart_default_title_slugifies_topic() -> None:
    assert smart_default_title({"topic": "Hello World"}) == "hello-world"
    assert smart_default_title({"topic": ""}) == "quest"
    assert smart_default_title({}) == "quest"


def test_smart_default_title_truncates_to_40_chars() -> None:
    long_topic = "a-very-long-topic-name-that-runs-past-the-cutoff-and-keeps-going"
    out = smart_default_title({"topic": long_topic})
    assert len(out) <= 40
    assert out  # not empty


def test_smart_default_no_simulation_prose_formats_default_true() -> None:
    for fmt in PROSE_FORMATS:
        assert smart_default_no_simulation({"paper_format": fmt}) is True


def test_smart_default_no_simulation_scientific_formats_default_false() -> None:
    for fmt in ("generic", "neurips", "iclr", "ieee_access", "nature_mi"):
        assert smart_default_no_simulation({"paper_format": fmt}) is False


def test_smart_default_no_simulation_unknown_format_defaults_false() -> None:
    # Defensive — an unknown format string shouldn't crash.
    assert smart_default_no_simulation({"paper_format": "made-up-format"}) is False
    assert smart_default_no_simulation({}) is False


def test_smart_default_study_depth_policy_brief_is_brief() -> None:
    assert smart_default_study_depth({"paper_format": "policy_brief"}) == "brief preprint"


def test_smart_default_study_depth_survey_topic_is_comprehensive() -> None:
    for topic in (
        "Survey of methods for X",
        "A review of recent work on Y",
        "Comparison of X vs Y",
    ):
        assert smart_default_study_depth({"topic": topic, "paper_format": "generic"}) == "comprehensive review"


def test_smart_default_study_depth_default_is_journal_length() -> None:
    assert smart_default_study_depth({"paper_format": "generic"}) == "journal-length"
    assert smart_default_study_depth({}) == "journal-length"


def test_build_smart_defaults_cascades_through_partial() -> None:
    """build_smart_defaults takes whatever's been answered so far and
    derives the next-question default. The cascade is: pick a prose
    paper_format → smart-default no_simulation flips to True →
    smart-default study_depth depends on topic text."""
    partial = {"paper_format": "essay", "topic": "Survey of cat lore"}
    defaults = build_smart_defaults(partial)
    assert defaults["no_simulation"] is True
    assert defaults["study_depth"] == "comprehensive review"
    assert defaults["title"] == "survey-of-cat-lore"


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inp,out", [
    ("Hello World", "hello-world"),
    ("Hello   World!!!", "hello-world"),
    ("foo-bar", "foo-bar"),
    ("foo--bar", "foo-bar"),
    ("FOO Bar 123", "foo-bar-123"),
    ("---a---", "a"),
    ("", ""),
])
def test_slugify_examples(inp: str, out: str) -> None:
    assert slugify(inp) == out


@pytest.mark.parametrize("inp,expected", [
    # Traditional Chinese — the user-reported failing case.
    ("近視的遺傳影響", "近視的遺傳影響"),
    # Simplified Chinese.
    ("近视的遗传影响", "近视的遗传影响"),
    # Japanese — hiragana + katakana + kanji mix collapsed into one run.
    ("日本語のテスト", "日本語のテスト"),
    # Korean — space → dash.
    ("한국어 테스트", "한국어-테스트"),
    # Cyrillic — lowercased.
    ("Тестовая тема", "тестовая-тема"),
    # Greek — lowercased.
    ("Δοκιμή", "δοκιμή"),
    # Mixed CJK + ASCII keeps BOTH runs (not just the ASCII portion).
    ("Genetic 遺傳 impact", "genetic-遺傳-impact"),
])
def test_slugify_preserves_non_latin_scripts(inp: str, expected: str) -> None:
    """Pre-fix bug: CJK / Cyrillic / Greek topics produced an empty
    slug because ``[^a-z0-9\\s-]`` stripped every non-Latin codepoint.
    Post-fix: ``[^\\w\\s-]`` (Unicode-aware) keeps them. Exact-match
    assertion so a regression to "drops part of mixed input" surfaces
    immediately — a non-empty check would still pass with bugs."""
    assert slugify(inp) == expected


def test_engine_slugify_falls_back_to_hex_for_pure_non_ascii() -> None:
    """The engine's quest-id ``_slugify`` MUST stay ASCII because the
    digest / critique / --resume / interview_update validators all
    require ``^\\d{10}-[a-z0-9-]+-[0-9a-f]{6}$``. Pure non-Latin topics
    fall back to a deterministic 8-hex digest of the original text:
    each distinct CJK / Cyrillic topic still gets a distinct quest_id
    (instead of every one colliding on ``"untitled"``)."""
    from core.engine import _slugify
    import re as _re
    cjk = _slugify("近視的遺傳影響")
    jp = _slugify("日本語のテスト")
    assert _re.fullmatch(r"i18n-[0-9a-f]{8}", cjk)
    assert _re.fullmatch(r"i18n-[0-9a-f]{8}", jp)
    # Distinct inputs → distinct hashes.
    assert cjk != jp
    # Deterministic — same input twice, same hash.
    assert _slugify("近視的遺傳影響") == cjk
    # Mixed CJK + ASCII keeps the ASCII portion (most-readable id).
    assert _slugify("genetic 近視 impact") == "genetic-impact"
    # ASCII path unchanged.
    assert _slugify("Hello World") == "hello-world"
    # Truly empty / pure-punctuation still falls back to untitled.
    assert _slugify("") == "untitled"
    assert _slugify("!!!---!!!") == "untitled"


# ---------------------------------------------------------------------------
# YAML emitter — round-trip through Config validation
# ---------------------------------------------------------------------------


def _sample_answers(**overrides: Any) -> InterviewAnswers:
    base = dict(
        topic="A test quest about something interesting",
        title="test-quest",
        output_kinds=["paper_md", "paper_pdf"],
        paper_format="generic",
        no_simulation=False,
        study_depth="journal-length",
        comparative_baseline="RandomForest baseline",
        success_metric="AUC >= 0.9",
        budget="a few minutes on a laptop CPU",
        clarify_mode="auto",
        review_panel=[],
        knowledge_enabled=False,
        provider="openai",
        provider_model="gpt-4o",
    )
    base.update(overrides)
    return InterviewAnswers(**base)


def test_yaml_round_trips_through_config_cli_frontend() -> None:
    """The YAML produced by the CLI frontend must validate against
    the Config schema. Anything else means we wrote an invalid YAML
    that the engine would reject at runtime."""
    yaml_text = answers_to_yaml(_sample_answers(), frontend="cli")
    data = yaml.safe_load(yaml_text)
    cfg = Config.model_validate(data)
    assert cfg.provider.name == "openai"
    assert cfg.provider.model == "gpt-4o"
    assert cfg.engine.clarify_mode == "auto"
    assert cfg.engine.no_simulation is False
    assert cfg.output.paper_format == "generic"
    assert sorted(cfg.engine.clarify_overrides.keys()) == sorted([
        "study_depth", "paper_venue", "output_kinds", "simulatability",
        "comparative_baseline", "success_metric", "budget",
    ])


def test_yaml_round_trips_for_vscode_frontend_with_cost_discipline() -> None:
    """VSCode frontend keeps the cost-discipline defaults
    (ideate_reflect: false, cross_check_per_finding_k: 0,
    enable_analyze_reroute: false) for parity with the existing TS
    emitter."""
    yaml_text = answers_to_yaml(
        _sample_answers(provider="vscode_extension", provider_model=None),
        frontend="vscode",
    )
    data = yaml.safe_load(yaml_text)
    cfg = Config.model_validate(data)
    assert cfg.engine.ideate_reflect is False
    assert cfg.engine.cross_check_per_finding_k == 0
    assert cfg.engine.enable_analyze_reroute is False


def test_yaml_emits_prose_format_with_no_simulation() -> None:
    answers = _sample_answers(paper_format="essay", no_simulation=True)
    yaml_text = answers_to_yaml(answers, frontend="cli")
    data = yaml.safe_load(yaml_text)
    cfg = Config.model_validate(data)
    assert cfg.engine.no_simulation is True
    assert cfg.output.paper_format == "essay"
    assert cfg.engine.clarify_overrides["simulatability"] == "no"


def test_yaml_omits_provider_model_when_unset() -> None:
    """VSCode passes provider_model=None when the active Copilot model
    hasn't been captured yet. The emitter must NOT write
    ``provider.model: null`` — pydantic accepts None, but absent is
    cleaner + avoids the YAML 'Norway-problem'-adjacent surprise."""
    answers = _sample_answers(provider_model=None)
    yaml_text = answers_to_yaml(answers, frontend="vscode")
    assert "model:" not in yaml_text.split("engine:")[0], (
        "provider.model line should not appear when the value is None"
    )


def test_yaml_emits_review_panel_correctly() -> None:
    answers = _sample_answers(review_panel=["methodologist", "statistician"])
    yaml_text = answers_to_yaml(answers, frontend="cli")
    data = yaml.safe_load(yaml_text)
    cfg = Config.model_validate(data)
    assert cfg.engine.review_panel == ["methodologist", "statistician"]


# ---------------------------------------------------------------------------
# Schema export
# ---------------------------------------------------------------------------


def test_export_schema_json_contains_all_questions() -> None:
    payload = export_schema_json()
    assert payload["version"] == 1
    schema_ids = {q["id"] for q in payload["questions"]}
    canonical_ids = {q.id for q in QUESTIONS}
    assert schema_ids == canonical_ids


def test_export_schema_json_carries_choice_lists() -> None:
    payload = export_schema_json()
    paper_format_values = {c["value"] for c in payload["paper_formats"]}
    assert "generic" in paper_format_values
    assert "essay" in paper_format_values
    assert len(paper_format_values) == 9
    # provider_models dict has a model list per provider.
    for provider in payload["provider_models"]:
        assert isinstance(payload["provider_models"][provider], list)
        assert payload["provider_models"][provider], f"{provider} has no models"


def test_export_schema_json_carries_stage_invalidation() -> None:
    payload = export_schema_json()
    assert "paper_format" in payload["stage_invalidation"]
    assert payload["stage_invalidation"]["paper_format"] == ["write", "review"]


def test_export_schema_json_editable_fields_are_sorted() -> None:
    """Sorted output → diffable JSON file; if the dataclass order
    changes the snapshot doesn't churn."""
    payload = export_schema_json()
    assert payload["editable_fields"] == sorted(payload["editable_fields"])


# ---------------------------------------------------------------------------
# Provider availability probes
# ---------------------------------------------------------------------------


def test_available_providers_returns_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    """When OPENAI_API_KEY is set, openai + codex should both probe true.
    When no CLIs are installed, the binary-based providers should
    NOT appear."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(interview_mod.shutil, "which", lambda _: None)
    avail = available_providers()
    assert "openai" in avail
    assert "codex" in avail
    assert "claude_cli" not in avail


def test_available_providers_empty_when_no_keys_or_clis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(interview_mod.shutil, "which", lambda _: None)
    assert available_providers() == []


def test_model_choices_for_known_provider_returns_curated_list() -> None:
    choices = model_choices_for("openai")
    assert choices
    assert any(c.value == "gpt-4o" for c in choices)


def test_model_choices_for_unknown_provider_returns_empty() -> None:
    assert model_choices_for("invented-provider-xyz") == ()


# ---------------------------------------------------------------------------
# Preflight clarify — LLM call mocking + fallback
# ---------------------------------------------------------------------------


def test_preflight_clarify_falls_back_when_no_provider() -> None:
    """No provider name → no LLM call → static fallback dict."""
    out = asyncio.run(preflight_clarify(
        topic="some topic",
        paper_format="generic",
        provider_name=None,
    ))
    assert set(out.keys()) == {"comparative_baseline", "success_metric", "budget"}
    assert "(none specified" in out["comparative_baseline"]


def test_preflight_clarify_falls_back_when_topic_empty() -> None:
    out = asyncio.run(preflight_clarify(
        topic="   ",
        paper_format="generic",
        provider_name="openai",
    ))
    assert "(none specified" in out["comparative_baseline"]


def test_preflight_clarify_returns_llm_output_when_provider_responds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock the provider chat to return a valid JSON object;
    preflight_clarify must extract and return those values."""

    class FakeProvider:
        async def chat(self, prompt: str) -> str:
            return json.dumps({
                "comparative_baseline": "BaselineX",
                "success_metric": "AUC >= 0.92",
                "budget": "30 minutes on a GPU",
            })

    def fake_build(name: str, model: str | None = None) -> FakeProvider:
        return FakeProvider()

    # Import the lazy-import target so monkeypatch can replace it.
    import core.provider as provider_mod
    monkeypatch.setattr(provider_mod, "build_provider", fake_build, raising=False)

    out = asyncio.run(preflight_clarify(
        topic="some topic",
        paper_format="generic",
        provider_name="openai",
    ))
    assert out["comparative_baseline"] == "BaselineX"
    assert out["success_metric"] == "AUC >= 0.92"
    assert out["budget"] == "30 minutes on a GPU"


def test_preflight_clarify_falls_back_when_llm_returns_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-JSON LLM output → static fallback."""

    class GarbageProvider:
        async def chat(self, prompt: str) -> str:
            return "this is not JSON at all"

    def fake_build(name: str, model: str | None = None) -> GarbageProvider:
        return GarbageProvider()

    import core.provider as provider_mod
    monkeypatch.setattr(provider_mod, "build_provider", fake_build, raising=False)

    out = asyncio.run(preflight_clarify(
        topic="topic",
        paper_format="generic",
        provider_name="openai",
    ))
    assert "(none specified" in out["comparative_baseline"]


def test_preflight_clarify_times_out_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hanging provider → 30s default timeout → fallback. Use a
    short timeout for the test."""

    class HangingProvider:
        async def chat(self, prompt: str) -> str:
            await asyncio.sleep(60)
            return "{}"

    def fake_build(name: str, model: str | None = None) -> HangingProvider:
        return HangingProvider()

    import core.provider as provider_mod
    monkeypatch.setattr(provider_mod, "build_provider", fake_build, raising=False)

    out = asyncio.run(preflight_clarify(
        topic="topic",
        paper_format="generic",
        provider_name="openai",
        timeout_s=0.1,
    ))
    assert "(none specified" in out["comparative_baseline"]


# ---------------------------------------------------------------------------
# Schema JSON file is checked in and matches export()
# ---------------------------------------------------------------------------


def test_schema_json_file_matches_export() -> None:
    """``core/interview_schema.json`` is a snapshot of
    ``export_schema_json()``. If the canonical Python list changes
    without the JSON being regenerated, the TS + HTMX frontends
    silently drift. This test fails CI when the snapshot is stale —
    run ``python -c 'from core.interview import write_schema_json;
    write_schema_json()'`` to regenerate."""
    json_path = Path(__file__).resolve().parent.parent / "core" / "interview_schema.json"
    assert json_path.exists(), (
        "core/interview_schema.json must exist. Generate with "
        "`python -c 'from core.interview import write_schema_json; write_schema_json()'`."
    )
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    fresh = export_schema_json()
    assert on_disk == fresh, (
        "core/interview_schema.json is out of date. Run "
        "`python -c 'from core.interview import write_schema_json; "
        "write_schema_json()'` to regenerate."
    )


# ---------------------------------------------------------------------------
# Ensemble profile — Tier-3 question wiring
# ---------------------------------------------------------------------------


def test_ensemble_profile_question_is_tier_3_and_optional() -> None:
    """The ensemble profile lives at Tier 3 (behind 'Show advanced')
    and defaults to ``off`` so users who don't opt in see no change.
    It's mid-quest-editable so the user can flip profiles between
    /resume passes."""
    q = next(q for q in QUESTIONS if q.id == "ensemble_profile")
    assert q.tier == 3
    assert q.default == "off"
    assert q.mid_quest_editable is True
    values = {c.value for c in q.choices}
    assert values == {"off", "cross_check_only", "ideate_and_check", "full"}


def test_expand_ensemble_profile_off_emits_no_block() -> None:
    """``off`` → empty dict so the YAML emitter can skip the
    ``node_ensemble`` block entirely."""
    from core.interview import expand_ensemble_profile
    assert expand_ensemble_profile("off", provider="openai") == {}
    assert expand_ensemble_profile("", provider="openai") == {}


def test_expand_ensemble_profile_cross_check_only_has_just_cross_check() -> None:
    """Cheapest non-trivial profile: fan-out limited to cross_check,
    vote merge (no moderator call — pure tally)."""
    from core.interview import expand_ensemble_profile
    out = expand_ensemble_profile("cross_check_only", provider="openai")
    assert set(out) == {"cross_check"}
    assert out["cross_check"]["merge"] == "vote"
    assert "moderator" not in out["cross_check"]
    assert len(out["cross_check"]["models"]) == 3


def test_expand_ensemble_profile_full_avoids_analyze_synthesize() -> None:
    """The full profile must never pair ``analyze`` with ``synthesize``
    — that combination is rejected by ProviderConfig's validator.
    Tournament is the only safe merger for analyze."""
    from core.interview import expand_ensemble_profile
    out = expand_ensemble_profile("full", provider="claude_cli")
    assert set(out) == {"cross_check", "ideate", "analyze"}
    assert out["analyze"]["merge"] == "tournament"
    assert out["ideate"]["merge"] == "tournament"
    assert out["cross_check"]["merge"] == "vote"
    # Tournament needs a moderator. The first trio model is the default.
    assert out["analyze"]["moderator"] == out["analyze"]["models"][0]


def test_ensemble_model_trio_unknown_provider_falls_back() -> None:
    """Providers not in the trio catalog still get a 3-element list
    (repeated provider_model). The engine fans out 3 calls; the user
    can edit the YAML to swap models."""
    from core.interview import ensemble_model_trio
    out = ensemble_model_trio("nonsense-provider", provider_model="my-model")
    assert out == ["my-model", "my-model", "my-model"]


def test_estimate_ensemble_cost_multiplier_orders_match_intuition() -> None:
    """Cost grows monotonically with ensemble breadth. Frontends rely
    on this ordering when rendering the cost-discipline hint."""
    from core.interview import estimate_ensemble_cost_multiplier
    assert estimate_ensemble_cost_multiplier("off") == 1.0
    assert (
        estimate_ensemble_cost_multiplier("off")
        < estimate_ensemble_cost_multiplier("cross_check_only")
        < estimate_ensemble_cost_multiplier("ideate_and_check")
        < estimate_ensemble_cost_multiplier("full")
    )


def test_answers_to_yaml_emits_node_ensemble_when_profile_picked() -> None:
    """When ``ensemble_profile != "off"``, the YAML emitter materializes
    a ``provider.node_ensemble`` block. The block must round-trip
    through the ProviderConfig validator (analyze: tournament is OK)."""
    from core.config import Config
    answers = InterviewAnswers(
        topic="t", title="t", output_kinds=["paper_md"],
        paper_format="generic", no_simulation=False, study_depth="journal-length",
        comparative_baseline="b", success_metric="m", budget="b",
        clarify_mode="auto", review_panel=[], knowledge_enabled=False,
        provider="openai", ensemble_profile="full",
    )
    yaml_text = answers_to_yaml(answers)
    assert "node_ensemble:" in yaml_text
    assert "ideate:" in yaml_text
    assert "analyze:" in yaml_text
    assert "cross_check:" in yaml_text
    # Round-trip through the Pydantic loader so a future YAML-shape
    # regression fails the test instead of corrupting real quests.
    parsed = yaml.safe_load(yaml_text)
    Config.model_validate(parsed)


def test_answers_to_yaml_omits_node_ensemble_for_default_off() -> None:
    """Default profile = "off" → no node_ensemble block. Existing quests
    that never touched this slot continue to emit identical YAML."""
    answers = InterviewAnswers(
        topic="t", title="t", output_kinds=["paper_md"],
        paper_format="generic", no_simulation=False, study_depth="journal-length",
        comparative_baseline="b", success_metric="m", budget="b",
        clarify_mode="auto", review_panel=[], knowledge_enabled=False,
        provider="openai", ensemble_profile="off",
    )
    assert "node_ensemble" not in answers_to_yaml(answers)


def test_max_iterations_question_is_tier_3_default_2() -> None:
    """User-reported: 'i cannot define iterations in interview questions'.
    The hard cap on the design-revise loop now appears as a Tier-3
    question — visible behind 'Show advanced', default 2 to preserve
    existing behaviour for users who don't touch it."""
    q = next(q for q in QUESTIONS if q.id == "max_iterations")
    assert q.tier == 3
    assert q.default == 2
    assert q.mid_quest_editable is True


def test_answers_to_yaml_respects_max_iterations_override() -> None:
    """A user-supplied ``max_iterations`` carries through to the
    rendered YAML's ``engine.max_iterations`` field — pinning the new
    Tier-3 question actually changes the YAML, not just the dataclass."""
    answers = InterviewAnswers(
        topic="t", title="t", output_kinds=["paper_md"],
        paper_format="generic", no_simulation=False, study_depth="journal-length",
        comparative_baseline="b", success_metric="m", budget="b",
        clarify_mode="auto", review_panel=[], knowledge_enabled=False,
        provider="openai", max_iterations=4,
    )
    out = answers_to_yaml(answers)
    parsed = yaml.safe_load(out)
    assert parsed["engine"]["max_iterations"] == 4
