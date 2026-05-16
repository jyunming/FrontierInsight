"""Tests for core/interview_update.py — the mid-quest re-entry flow.

Focus areas:
* ``load_current_answers`` correctly projects YAML → InterviewAnswers.
* ``diff_answers`` only reports editable-field changes.
* ``compute_invalidated_stages`` honors the STAGE_INVALIDATION matrix.
* ``keys_to_clear`` maps stage names to QuestState keys.
* ``rewrite_yaml_with_new_answers`` preserves non-interview keys.

These tests do NOT run the actual ``run_update_flow`` end-to-end —
that requires a real LangGraph checkpoint sqlite + a runnable Engine.
The flow is exercised by integration tests in a future PR; here we
pin the pure-logic helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from core.config import Config
from core.interview import (
    EDITABLE_FIELDS,
    InterviewAnswers,
    STAGE_INVALIDATION,
    answers_to_yaml,
)
from core.interview_update import (
    HARD_REFUSE_FIELDS,
    STAGE_STATE_KEYS,
    compute_invalidated_stages,
    diff_answers,
    keys_to_clear,
    load_current_answers,
    rewrite_yaml_with_new_answers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample(**overrides: Any) -> InterviewAnswers:
    base = dict(
        topic="Sample topic",
        title="sample",
        output_kinds=["paper_md", "paper_pdf"],
        paper_format="generic",
        no_simulation=False,
        study_depth="journal-length",
        comparative_baseline="BaselineA",
        success_metric="AUC >= 0.9",
        budget="5 minutes",
        clarify_mode="auto",
        review_panel=[],
        knowledge_enabled=False,
        provider="openai",
        provider_model="gpt-4o",
    )
    base.update(overrides)
    return InterviewAnswers(**base)


# ---------------------------------------------------------------------------
# load_current_answers
# ---------------------------------------------------------------------------


def test_load_current_answers_round_trips(tmp_path: Path) -> None:
    quest_root = tmp_path / "quest-x"
    quest_root.mkdir()
    original = _sample()
    yaml_text = answers_to_yaml(original, frontend="cli")
    (quest_root / "config.yaml").write_text(yaml_text, encoding="utf-8")

    loaded, yaml_path, raw = load_current_answers(quest_root)
    assert yaml_path.name == "config.yaml"
    assert loaded.topic == original.topic
    assert loaded.paper_format == original.paper_format
    assert loaded.study_depth == original.study_depth
    assert loaded.comparative_baseline == original.comparative_baseline
    assert loaded.success_metric == original.success_metric
    assert loaded.review_panel == original.review_panel
    assert isinstance(raw, dict)


def test_load_current_answers_missing_config(tmp_path: Path) -> None:
    quest_root = tmp_path / "empty"
    quest_root.mkdir()
    with pytest.raises(FileNotFoundError):
        load_current_answers(quest_root)


def test_load_current_answers_handles_yaml_literal_topic(tmp_path: Path) -> None:
    """Topics are usually written as YAML literal blocks (``topic: |``).
    The parser may return them as a single multi-line string or a
    list of lines depending on YAML quirks — both should work."""
    quest_root = tmp_path / "quest-lit"
    quest_root.mkdir()
    yaml_text = (
        "topic: |\n"
        "  line one\n"
        "  line two\n"
        "title: \"q\"\n"
        "provider:\n  name: \"openai\"\n"
        "engine:\n  framework: \"langgraph\"\n  clarify_mode: \"auto\"\n  clarify_overrides:\n    paper_venue: \"generic\"\n"
        "output:\n  kinds: [\"paper_md\"]\n  paper_format: \"generic\"\n"
        "knowledge:\n  enabled: false\n"
    )
    (quest_root / "config.yaml").write_text(yaml_text, encoding="utf-8")
    loaded, _, _ = load_current_answers(quest_root)
    assert "line one" in loaded.topic
    assert "line two" in loaded.topic


# ---------------------------------------------------------------------------
# diff_answers
# ---------------------------------------------------------------------------


def test_diff_answers_empty_when_unchanged() -> None:
    a = _sample()
    b = _sample()
    assert diff_answers(a, b) == {}


def test_diff_answers_reports_only_editable_fields() -> None:
    """Changing a non-editable field (topic, title, provider, model,
    no_simulation) must NOT appear in the diff — the interview
    forbids editing those mid-quest."""
    a = _sample()
    b = _sample(topic="DIFFERENT topic", provider="codex")
    diff = diff_answers(a, b)
    assert "topic" not in diff
    assert "provider" not in diff


def test_diff_answers_reports_paper_format_change() -> None:
    a = _sample(paper_format="generic")
    b = _sample(paper_format="neurips")
    diff = diff_answers(a, b)
    assert "paper_format" in diff
    assert diff["paper_format"] == ("generic", "neurips")


def test_diff_answers_reports_review_panel_change() -> None:
    a = _sample(review_panel=[])
    b = _sample(review_panel=["methodologist"])
    diff = diff_answers(a, b)
    assert "review_panel" in diff


# ---------------------------------------------------------------------------
# Stage invalidation
# ---------------------------------------------------------------------------


def test_compute_invalidated_stages_paper_format() -> None:
    """Per STAGE_INVALIDATION, paper_format triggers write + review."""
    stages = compute_invalidated_stages({"paper_format": ("generic", "neurips")})
    assert stages == ["write", "review"]


def test_compute_invalidated_stages_topic_tuned_slot() -> None:
    """success_metric triggers analyze + cross_check + write + review."""
    stages = compute_invalidated_stages({"success_metric": ("X", "Y")})
    assert stages == ["analyze", "cross_check", "write", "review"]


def test_compute_invalidated_stages_review_panel_only() -> None:
    stages = compute_invalidated_stages({"review_panel": ([], ["methodologist"])})
    assert stages == ["review"]


def test_compute_invalidated_stages_output_kinds_is_noop() -> None:
    """output_kinds doesn't invalidate any LLM stage — generators
    just re-fire."""
    stages = compute_invalidated_stages({"output_kinds": (["paper_md"], ["paper_md", "paper_pdf"])})
    assert stages == []


def test_compute_invalidated_stages_clarify_mode_is_noop() -> None:
    stages = compute_invalidated_stages({"clarify_mode": ("auto", "interactive")})
    assert stages == []


def test_compute_invalidated_stages_multiple_changes_union() -> None:
    """When several fields change, the affected stages are the
    UNION, returned in pipeline order."""
    changes = {
        "paper_format": ("generic", "neurips"),
        "review_panel": ([], ["methodologist"]),
    }
    stages = compute_invalidated_stages(changes)
    # write from paper_format, review from both — order: write, review.
    assert stages == ["write", "review"]


def test_keys_to_clear_maps_stages_to_quest_state_keys() -> None:
    """When write is invalidated, we clear `paper_md` from the
    checkpoint. When review is invalidated, we clear `review` +
    `review_panel`."""
    assert keys_to_clear(["write"]) == ["paper_md"]
    assert sorted(keys_to_clear(["review"])) == sorted(["review", "review_panel"])
    assert sorted(keys_to_clear(["write", "review"])) == sorted(
        ["paper_md", "review", "review_panel"]
    )


def test_stage_state_keys_cover_every_invalidated_stage() -> None:
    """Every stage mentioned in STAGE_INVALIDATION values must have
    an entry in STAGE_STATE_KEYS — otherwise a stage invalidation
    silently fails to clear anything."""
    all_stages = set()
    for stages in STAGE_INVALIDATION.values():
        all_stages.update(stages)
    for stage in all_stages:
        assert stage in STAGE_STATE_KEYS, (
            f"STAGE_INVALIDATION references stage {stage!r} but "
            f"STAGE_STATE_KEYS doesn't list any state keys for it"
        )


# ---------------------------------------------------------------------------
# YAML rewrite preserves non-interview keys
# ---------------------------------------------------------------------------


def test_rewrite_yaml_preserves_local_papers(tmp_path: Path) -> None:
    """``knowledge.local_papers`` is set by --proposal seeding and
    must survive an --update round-trip."""
    quest_root = tmp_path / "quest-prop"
    quest_root.mkdir()
    raw = yaml.safe_load(answers_to_yaml(_sample(), frontend="cli"))
    raw.setdefault("knowledge", {})["local_papers"] = ["/path/to/proposal.md"]
    raw.setdefault("execution", {})["timeout_s"] = 1800
    new_yaml_text = rewrite_yaml_with_new_answers(raw, _sample(paper_format="neurips"))
    new_data = yaml.safe_load(new_yaml_text)
    assert new_data["knowledge"]["local_papers"] == ["/path/to/proposal.md"]
    assert new_data["execution"]["timeout_s"] == 1800
    # The interview-managed field DID change.
    assert new_data["output"]["paper_format"] == "neurips"


def test_rewrite_yaml_round_trips_through_config_validation(tmp_path: Path) -> None:
    raw = yaml.safe_load(answers_to_yaml(_sample(), frontend="cli"))
    new_yaml_text = rewrite_yaml_with_new_answers(raw, _sample(paper_format="essay"))
    cfg = Config.model_validate(yaml.safe_load(new_yaml_text))
    assert cfg.output.paper_format == "essay"


# ---------------------------------------------------------------------------
# Hard-refuse fields
# ---------------------------------------------------------------------------


def test_run_update_flow_rejects_path_traversal(tmp_path: Path) -> None:
    """``python launch.py --update '../somewhere'`` must NOT escape
    the output root. The flow returns rc=1 with a clear error
    message instead of read/writing arbitrary paths."""
    import asyncio
    from core.interview_update import run_update_flow

    output_root = tmp_path / "outputs"
    output_root.mkdir()

    async def fake_run_one(*args, **kwargs):
        raise AssertionError("run_one must NOT be called for hostile quest_id")

    def fake_apply_bridge(cfg, port):
        pass

    for hostile in ("../somewhere", "a/b", "a\\b", ""):
        rc = asyncio.run(run_update_flow(
            quest_id=hostile,
            output_root=output_root,
            vscode_bridge_port=0,
            interactive=False,
            supervisor=None,
            run_one=fake_run_one,
            apply_vscode_bridge_override=fake_apply_bridge,
        ))
        assert rc == 1, f"hostile quest_id {hostile!r} returned rc={rc}"


def test_no_simulation_in_hard_refuse() -> None:
    """no_simulation is not editable mid-quest. The HARD_REFUSE_FIELDS
    set is consulted by run_update_flow to skip it during the
    interview walk."""
    assert "no_simulation" in HARD_REFUSE_FIELDS
