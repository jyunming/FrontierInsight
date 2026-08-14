"""Interview → YAML → Config round-trip guard (PR-10).

``answers_to_yaml`` hand-writes YAML. Nothing checked that the emitted keys
still MATCH the real ``Config`` schema, so a renamed/dropped config field (or a
non-default answer wired to the wrong key — the "dead pause_for_user_input" /
"dropped paper_style" drift class) could silently stop taking effect. These
tests load the emitted YAML through the real ``Config.from_yaml`` and assert the
non-default choices actually land on the config keys the engine reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Config
from core.interview import InterviewAnswers, answers_to_yaml


def _full_answers() -> InterviewAnswers:
    """A fully-populated set with EVERY toggle off its default, so each one has
    to survive the round-trip to be observed."""
    return InterviewAnswers(
        topic="round-trip probe topic",
        title="round-trip",
        output_kinds=["paper_pdf", "slides"],
        paper_format="neurips",
        no_simulation=True,
        study_depth="comprehensive review",
        comparative_baseline="a strong baseline",
        success_metric="accuracy",
        budget="2 hours",
        clarify_mode="off",
        review_panel=["methodologist", "skeptic"],
        knowledge_enabled=True,
        pause_for_user_input="never",
        provider="openai",
        provider_model="gpt-5",
        max_iterations=3,
        audience="internal",
        paper_style="briefing",
        knowledge_top_k=12,
        knowledge_external_top_k=30,
        survey_mode=True,
        web_research=True,
        supply_papers=True,
        ensemble_profile="full",
    )


def _load(tmp_path: Path, answers: InterviewAnswers) -> Config:
    yaml_text = answers_to_yaml(answers, frontend="cli")
    p = tmp_path / "quest.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return Config.from_yaml(p)


def test_emitted_yaml_parses_as_config(tmp_path: Path):
    """The hand-written YAML must load through the real Config schema without
    raising — the fundamental emitter-vs-schema lock."""
    cfg = _load(tmp_path, _full_answers())
    # topic is emitted as a YAML block literal (``topic: |``) on BOTH the
    # Python and TS surfaces, so it round-trips with a trailing newline — a
    # benign, parity-consistent quirk (the topic is slug-stripped and
    # prompt-embedded). Assert on the stripped content.
    assert cfg.topic.strip() == "round-trip probe topic"
    assert cfg.title == "round-trip"


def test_non_default_choices_reach_the_right_config_keys(tmp_path: Path):
    cfg = _load(tmp_path, _full_answers())

    # paper_style → output.paper_style (the "dropped paper_style" class).
    assert cfg.output.paper_style == "briefing"
    # audience → output.audience.
    assert cfg.output.audience == "internal"
    # output kinds survive verbatim.
    assert cfg.output.kinds == ["paper_pdf", "slides"]
    # survey_mode → engine.survey_mode (and implies no_simulation).
    assert cfg.engine.survey_mode is True
    assert cfg.engine.no_simulation is True
    # review panel personas survive.
    assert cfg.engine.review_panel == ["methodologist", "skeptic"]
    # knowledge tuning survives.
    assert cfg.knowledge.enabled is True
    assert cfg.knowledge.top_k == 12
    assert cfg.knowledge.external_top_k == 30
    assert cfg.knowledge.web_search is True
    # supply_papers → pauses.papers (the "dead pause" class).
    assert cfg.pauses.papers is True
    # ensemble_profile=full → provider.node_ensemble is populated.
    assert cfg.provider.node_ensemble, "ensemble profile did not reach node_ensemble"
    assert "ideate" in cfg.provider.node_ensemble


def test_defaults_stay_clean(tmp_path: Path):
    """A default-latex, ensemble-off answer set must NOT emit the optional
    keys (keeps generated YAML minimal) — and still round-trips."""
    ans = _full_answers()
    ans.paper_style = "latex"
    ans.ensemble_profile = "off"
    ans.audience = "external"
    yaml_text = answers_to_yaml(ans, frontend="cli")
    assert "paper_style:" not in yaml_text
    assert "node_ensemble:" not in yaml_text
    # Still a valid config.
    p = tmp_path / "q.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = Config.from_yaml(p)
    assert cfg.output.paper_style == "latex"
    assert not cfg.provider.node_ensemble
