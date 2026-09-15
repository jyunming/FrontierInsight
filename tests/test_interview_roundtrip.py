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


def test_turning_the_paper_pause_off_survives_the_round_trip(tmp_path: Path):
    """The pause for paywalled PDFs is on by default, so an interview answer of
    "off" has to be written out explicitly or it silently reverts to on."""
    ans = _full_answers()
    ans.supply_papers = False
    assert "papers: false" in answers_to_yaml(ans, frontend="cli")
    assert _load(tmp_path, ans).pauses.papers is False


def test_author_line_and_poster_size_reach_the_output_config(tmp_path: Path):
    ans = _full_answers()
    ans.author = "陳 Jane"
    ans.affiliation = "R&D Lab, Example University"
    ans.contact_email = "jane_doe@example.org"
    ans.url = "https://example.org/project#poster"
    ans.poster_size = "landscape_48x36"
    cfg = _load(tmp_path, ans)
    assert (cfg.output.author, cfg.output.affiliation) == ("陳 Jane", "R&D Lab, Example University")
    assert (cfg.output.contact_email, cfg.output.url) == (
        "jane_doe@example.org", "https://example.org/project#poster",
    )
    assert cfg.output.poster_size == "landscape_48x36"


def test_an_unset_author_line_writes_nothing(tmp_path: Path):
    yaml_text = answers_to_yaml(_full_answers(), frontend="cli")
    for key in ("author:", "affiliation:", "contact_email:", "url:", "poster_size:"):
        assert key not in yaml_text
    cfg = _load(tmp_path, _full_answers())
    assert (cfg.output.author, cfg.output.url, cfg.output.poster_size) == ("", "", "a1_portrait")


def test_update_keeps_the_author_line_and_clearing_a_field_removes_it(tmp_path: Path):
    """--update loads the author line from the quest's YAML, and a field the
    user clears must not come back from the old YAML: the emitter leaves an
    empty field out, so the merge has to treat these keys as managed."""
    from dataclasses import replace

    from core.interview_update import load_current_answers, rewrite_yaml_with_new_answers

    quest = tmp_path / "quest"
    quest.mkdir()
    ans = _full_answers()
    ans.author, ans.url, ans.poster_size = "Jane Chen", "https://example.org", "a0_portrait"
    (quest / "config.yaml").write_text(answers_to_yaml(ans, frontend="cli"), encoding="utf-8")

    current, _yaml_path, raw = load_current_answers(quest)
    assert (current.author, current.url, current.poster_size) == (
        "Jane Chen", "https://example.org", "a0_portrait",
    )
    (quest / "config.yaml").write_text(
        rewrite_yaml_with_new_answers(raw, replace(current, url="")), encoding="utf-8",
    )
    cfg = Config.from_yaml(quest / "config.yaml")
    assert (cfg.output.author, cfg.output.url, cfg.output.poster_size) == (
        "Jane Chen", "", "a0_portrait",
    )


def test_reasoning_effort_reaches_the_provider_config_and_default_writes_nothing(tmp_path: Path):
    ans = _full_answers()
    ans.reasoning_effort = "high"
    assert 'reasoning_effort: "high"' in answers_to_yaml(ans, frontend="cli")
    assert _load(tmp_path, ans).provider.reasoning_effort == "high"
    ans.reasoning_effort = "default"
    assert "reasoning_effort:" not in answers_to_yaml(ans, frontend="cli")
    assert _load(tmp_path, ans).provider.reasoning_effort is None


def test_update_keeps_reasoning_effort_and_resetting_to_default_removes_it(tmp_path: Path):
    """--update loads the level from the quest's YAML; setting it back to
    "default" must drop the key rather than merge the old level back."""
    from dataclasses import replace

    from core.interview_update import load_current_answers, rewrite_yaml_with_new_answers

    quest = tmp_path / "quest"
    quest.mkdir()
    ans = _full_answers()
    ans.reasoning_effort = "xhigh"
    (quest / "config.yaml").write_text(answers_to_yaml(ans, frontend="cli"), encoding="utf-8")

    current, _yaml_path, raw = load_current_answers(quest)
    assert current.reasoning_effort == "xhigh"
    (quest / "config.yaml").write_text(
        rewrite_yaml_with_new_answers(raw, replace(current, reasoning_effort="default")),
        encoding="utf-8",
    )
    assert Config.from_yaml(quest / "config.yaml").provider.reasoning_effort is None


def test_page_limit_reaches_the_output_config_and_blank_writes_nothing(tmp_path: Path):
    from core.config import resolve_page_limit

    ans = _full_answers()
    ans.page_limit = 4
    assert "  page_limit: 4" in answers_to_yaml(ans, frontend="cli").splitlines()
    cfg = _load(tmp_path, ans)
    assert cfg.output.page_limit == 4 and resolve_page_limit(cfg) == 4
    ans.page_limit = None
    assert "page_limit:" not in answers_to_yaml(ans, frontend="cli")
    assert _load(tmp_path, ans).output.page_limit is None


@pytest.mark.parametrize("raw, expected", [
    (None, None), ("", None), ("  ", None), ("none", None), ("None", None),
    ("4", 4), (" 4 ", 4), ("+4", 4), (4, 4), (1, 1),
])
def test_page_limit_answers_that_are_accepted(raw, expected):  # noqa: ANN001
    from core.interview import parse_page_limit_answer

    assert parse_page_limit_answer(raw) == expected


@pytest.mark.parametrize("raw", ["0", 0, -1, "-1", "four", "4.5", 4.5, "4 pages", True, "²", [4]])
def test_page_limit_answers_that_are_refused(raw):  # noqa: ANN001
    from core.interview import parse_page_limit_answer

    with pytest.raises(ValueError):
        parse_page_limit_answer(raw)
    seen: list[str] = []
    assert parse_page_limit_answer(raw, on_error=seen.append) is None and len(seen) == 1


def test_update_keeps_the_page_limit_and_clearing_it_removes_it(tmp_path: Path):
    """--update loads the limit from the quest's YAML and keeps it; clearing it
    drops the key rather than merging the old limit back, and re-runs the
    write and review stages."""
    from dataclasses import replace

    from core.interview_update import (
        compute_invalidated_stages, diff_answers, load_current_answers, rewrite_yaml_with_new_answers,
    )

    quest = tmp_path / "quest"
    quest.mkdir()
    ans = _full_answers()
    ans.page_limit = 4
    (quest / "config.yaml").write_text(answers_to_yaml(ans, frontend="cli"), encoding="utf-8")

    current, _yaml_path, raw = load_current_answers(quest)
    assert current.page_limit == 4
    (quest / "config.yaml").write_text(rewrite_yaml_with_new_answers(raw, current), encoding="utf-8")
    assert Config.from_yaml(quest / "config.yaml").output.page_limit == 4

    cleared = replace(current, page_limit=None)
    changes = diff_answers(current, cleared)
    assert changes == {"page_limit": (4, None)}
    assert compute_invalidated_stages(changes) == ["write", "review"]
    (quest / "config.yaml").write_text(rewrite_yaml_with_new_answers(raw, cleared), encoding="utf-8")
    assert Config.from_yaml(quest / "config.yaml").output.page_limit is None


def test_output_config_keeps_each_author_field_on_one_line():
    cfg = Config(topic="t", output={"author": "Jane\n  Chen", "affiliation": None})
    assert (cfg.output.author, cfg.output.affiliation) == ("Jane Chen", "")


def _find_question(node, qid: str):
    if isinstance(node, dict):
        if node.get("id") == qid and "default" in node:
            return node
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            found = _find_question(child, qid)
            if found is not None:
                return found
    return None


def test_new_quests_fetch_full_text_and_pause_for_paywalled_papers_by_default():
    import json

    cfg = Config(topic="t", title="t")
    assert cfg.pauses.papers is True
    assert cfg.knowledge.try_fetch_full_text is True
    # The legacy flag is merged into pauses.papers, and a programmatic Config
    # copies it through, so its default must agree with the new one.
    assert cfg.knowledge.pause_for_user_papers is True
    schema = json.loads(
        (Path(__file__).resolve().parent.parent / "core" / "interview_schema.json").read_text(encoding="utf-8"))
    question = _find_question(schema, "supply_papers")
    assert question is not None and question["default"] is True
