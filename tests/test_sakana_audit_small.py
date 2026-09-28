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
    unsaid = Config(topic="t")
    assert unsaid.result_use == "" and unsaid.effective_result_use == "explore", "the field keeps what the file said"
    assert Config.model_validate({"topic": "t", "rigor_profile": "research"}).effective_result_use == "research"
    assert Config.model_validate({"topic": "t", "result_use": "decision"}).effective_result_use == "decision"
    # Worked out when read, so a copy with another profile gets its own answer.
    assert unsaid.model_copy(update={"rigor_profile": "research"}).effective_result_use == "research"


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
    log = audit_log.AuditLog(trace, root.name)
    log.append("quest_started")
    log.append("node_completed", node="write")
    log.append("node_completed", node="review")
    for rel in evidence.SEALED_LEDGERS:  # as a finishing quest does: an unused record is an empty file
        (root / rel).touch()
    (root / "needs" / "EVIDENCE.json").write_text("{}", encoding="utf-8")  # written before the seal
    log.append("quest_finalized", events_before=3, write_errors=0, nodes_completed=["review", "write"],
               model_calls={"lines": 0, "counts": {}, "gaps": []},
               files={rel: evidence._file_sha256(root / rel) for rel in evidence.SEALED_FILES},
               paper_path="paper/paper.md", paper_sha256=evidence._file_sha256(root / "paper" / "paper.md"))
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
    fronted = _mark_preliminary("---\ntitle: x\n---\nBody.")
    assert fronted.startswith("---\ntitle: x\n---\n") and fronted.index(PRELIMINARY_NOTE) < fronted.index("Body.")
    fenced = _mark_preliminary("```python\n# a comment\n```\n# Real title\nText.")
    assert fenced.index("# Real title") < fenced.index(PRELIMINARY_NOTE)


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
    assert "under its `provider:` section" in text, "no approved record yet: the config is edited by hand"
    assert json.loads((engine.fi_dir / "pause.json").read_text(encoding="utf-8"))["kind"] == "review_models"
    assert any(e.get("kind") == "pause_requested" and e.get("pause") == "review_models"
               for e in audit_log.read(engine.audit.path))


def test_with_an_approved_record_the_steps_go_through_update(tmp_path: Path) -> None:
    from core import plan_settings
    engine = Engine(_research(tmp_path))
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    (engine.fi_dir / plan_settings.NAME).write_text("{}", encoding="utf-8")
    assert engine._review_models_stop() is not None
    text = (engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "--update" in text and "Per-node model overrides" in text


def test_a_panel_that_already_reviewed_is_not_stopped(tmp_path: Path) -> None:
    engine = Engine(_research(tmp_path))
    engine._audit("node_completed", node="review")
    assert engine._review_models_stop() is None


def test_research_with_one_reviewer_on_another_model_goes_on(tmp_path: Path) -> None:
    engine = Engine(_research(tmp_path, node_models={"review_panel.statistician": "m-other"}))
    assert engine._review_models_stop() is None
    assert engine._reviewer_model("statistician") == "m-other" and engine._reviewer_model("methodologist") == "m-main"


def test_set_up_with_only_one_model_the_quest_runs(tmp_path: Path) -> None:
    """The interview's "I only have one model": no stop; the result says so instead (core/evidence.py)."""
    cfg = _research(tmp_path)
    engine = Engine(cfg.model_copy(update={"engine": cfg.engine.model_copy(update={"one_model_review": True})}))
    assert engine._review_models_stop() is None
    assert not (engine.fi_dir / "pause.json").exists()


def test_the_default_profile_has_no_such_requirement(tmp_path: Path) -> None:
    engine = Engine(Config.model_validate({
        "topic": "t", "provider": {"name": "openai", "model": "m"}, "engine": {"review_panel": ["methodologist"]},
        "knowledge": {"enabled": False}, "output": {"output_dir": str(tmp_path / "out")},
    }))
    assert engine._review_models_stop() is None
