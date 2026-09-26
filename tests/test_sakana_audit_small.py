"""The 2026-09-26 Sakana re-audit's small items and the user's two decisions.

- A config that does not say what its result is for is an exploration (never publication-ready, its paper marked
  preliminary), unless it asks for ``rigor_profile: research``.
- ``rigor_profile: research`` keeps the decision trace on, and a trace that is missing or broken is a gap.
- ``rigor_profile: research`` needs a review panel with at least one reviewer on another model; each review records
  the model that gave it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import audit_log, evidence
from core.config import Config
from core.engine import PRELIMINARY_NOTE, Engine, _mark_preliminary
from tests.test_evidence import ON, _quest, _state


def test_a_config_that_does_not_say_what_it_is_for_is_an_exploration() -> None:
    assert Config(topic="t").result_use == "explore"
    assert Config.model_validate({"topic": "t", "rigor_profile": "research"}).result_use == "research"
    assert Config.model_validate({"topic": "t", "result_use": "decision"}).result_use == "decision"


def test_research_keeps_the_decision_trace_on() -> None:
    assert Config.model_validate({"topic": "t", "rigor_profile": "research"}).engine.audit_trace is True
    with pytest.raises(ValueError, match="audit_trace"):
        Config.model_validate({"topic": "t", "rigor_profile": "research", "engine": {"audit_trace": False}})


def test_under_research_a_missing_or_broken_trace_is_a_gap(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    research = {**ON, "rigor_profile": "research"}
    trace = root / ".fi" / "audit.jsonl"
    trace.unlink(missing_ok=True)
    missing = evidence.assess(root, _state(), settings=research)
    assert any("decision trace" in g and "missing" in g for g in missing["gaps"])
    audit_log.AuditLog(trace, root.name).append("quest_started")
    audit_log.AuditLog(trace, root.name).append("node_completed", node="design")
    assert not any("decision trace" in g for g in evidence.assess(root, _state(), settings=research)["gaps"])
    lines = trace.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["kind"] = "edited"
    trace.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", encoding="utf-8")
    broken = evidence.assess(root, _state(), settings=research)
    assert broken["status"] != "publication_ready"
    assert any("no longer checks out" in g for g in broken["gaps"])
    # Outside research, as before.
    assert not any("decision trace" in g for g in evidence.assess(root, _state(), settings=ON)["gaps"])


def test_an_exploration_s_paper_says_it_is_preliminary_once() -> None:
    marked = _mark_preliminary("# Title\n\nAbstract.")
    assert marked.index("# Title") < marked.index(PRELIMINARY_NOTE) < marked.index("Abstract.")
    assert _mark_preliminary(marked) == marked
    assert _mark_preliminary("No title.") == "\n" + PRELIMINARY_NOTE + "\n\nNo title."


def _research(tmp_path: Path, **provider: object) -> Config:
    return Config.model_validate({
        "topic": "t", "rigor_profile": "research", "provider": {"name": "openai", "model": "m-main", **provider},
        "knowledge": {"enabled": False}, "output": {"output_dir": str(tmp_path / "out")},
    })


def test_research_with_every_reviewer_on_one_model_stops_before_anything_runs(tmp_path: Path) -> None:
    engine = Engine(_research(tmp_path))
    artifacts = engine._review_models_stop()
    assert artifacts is not None
    text = (engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "one reviewer on a different model" in text and "review_panel.statistician" in text and "m-main" in text


def test_research_with_one_reviewer_on_another_model_goes_on(tmp_path: Path) -> None:
    engine = Engine(_research(tmp_path, node_models={"review_panel.statistician": "m-other"}))
    assert engine._review_models_stop() is None
    assert engine._reviewer_model("statistician") == "m-other" and engine._reviewer_model("methodologist") == "m-main"


def test_the_default_profile_has_no_such_requirement(tmp_path: Path) -> None:
    engine = Engine(Config.model_validate({
        "topic": "t", "provider": {"name": "openai", "model": "m"}, "engine": {"review_panel": ["methodologist"]},
        "knowledge": {"enabled": False}, "output": {"output_dir": str(tmp_path / "out")},
    }))
    assert engine._review_models_stop() is None
