"""Topic-shape clarify-slot tests.

Validates:

  - ``_log_topic_shape_mismatch`` warns when topic_shape is non-
    experimental but the engine resolved to SIMULATE (mismatch).
  - The helper is silent when shapes are aligned (experimental →
    SIMULATE, non-experimental → NO_SIMULATION).
  - ``_default_clarify_questions`` includes the ``topic_shape``
    slot so legacy fallback paths populate it too.
  - ``_format_clarify`` renders the slot when present so downstream
    prompts (design, write) see it.
  - The clarify-prompt file carries the four-shape vocabulary and
    the mismatch guidance so the LLM emits the right defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import (
    Engine, _default_clarify_questions, _format_clarify,
)


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents"


class _RecordCollector(logging.Handler):
    """Manual log capture: the engine's per-quest logger has
    ``propagate=False`` (set in ``_quest_logger``), so pytest's
    caplog — which hooks the root logger — never sees its records.
    Attach this handler directly to ``engine._log`` to collect."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def warnings(self) -> list[str]:
        return [
            r.getMessage() for r in self.records
            if r.levelno >= logging.WARNING
        ]


def _mk_engine(tmp_path: Path) -> Engine:
    cfg = Config(
        topic="OPC importance in low-k1 era",
        title="opc-review",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="auto",
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    return Engine(cfg)


def test_default_clarify_questions_includes_topic_shape(tmp_path: Path) -> None:
    """Legacy fallback (LLM JSON un-parseable) must still populate
    ``topic_shape`` so downstream prompts never see a missing slot."""
    q = _default_clarify_questions("any topic")
    assert "topic_shape" in q
    assert isinstance(q["topic_shape"], dict)
    assert q["topic_shape"]["default"] in {
        "experimental", "review", "case_study", "opinion",
    }


def test_format_clarify_renders_topic_shape(tmp_path: Path) -> None:
    """``_format_clarify`` flows the topic_shape value into the
    rendered bulleted block so design / write prompts see it."""
    out = _format_clarify({
        "clarify_answers": {
            "topic_shape": "review",
            "comparative_baseline": "(none)",
        }
    })
    assert "Topic shape" in out
    assert "review" in out


def _capture(eng: Engine) -> _RecordCollector:
    collector = _RecordCollector()
    eng._log.addHandler(collector)
    return collector


def test_topic_shape_mismatch_warns_when_review_meets_simulate(
    tmp_path: Path,
) -> None:
    """topic_shape=review + engine-resolved SIMULATE → WARNING.
    This is the canary for "engine is about to burn compute on a
    survey-shaped topic"."""
    eng = _mk_engine(tmp_path)
    coll = _capture(eng)
    eng._log_topic_shape_mismatch(
        {"topic_shape": "review"},
        no_simulation_resolved=False,  # engine routes to SIMULATE
    )
    assert any("topic_shape" in m and "SIMULATE" in m for m in coll.warnings), (
        f"expected mismatch WARNING; got: {coll.warnings}"
    )


@pytest.mark.parametrize("shape", ["case_study", "opinion"])
def test_topic_shape_mismatch_warns_for_all_non_experimental_shapes(
    tmp_path: Path, shape: str,
) -> None:
    """All three non-experimental shapes (review / case_study /
    opinion) trigger the warning when paired with SIMULATE."""
    eng = _mk_engine(tmp_path)
    coll = _capture(eng)
    eng._log_topic_shape_mismatch(
        {"topic_shape": shape}, no_simulation_resolved=False,
    )
    assert any(shape in m for m in coll.warnings)


def test_topic_shape_mismatch_silent_when_aligned_experimental(
    tmp_path: Path,
) -> None:
    """topic_shape=experimental + SIMULATE → no warning. The most
    common path; must stay quiet."""
    eng = _mk_engine(tmp_path)
    coll = _capture(eng)
    eng._log_topic_shape_mismatch(
        {"topic_shape": "experimental"},
        no_simulation_resolved=False,
    )
    assert coll.warnings == []


def test_topic_shape_mismatch_silent_when_aligned_review_with_no_simulation(
    tmp_path: Path,
) -> None:
    """topic_shape=review + NO_SIMULATION → no warning. The user
    already pivoted simulatability=no; engine and shape are
    consistent."""
    eng = _mk_engine(tmp_path)
    coll = _capture(eng)
    eng._log_topic_shape_mismatch(
        {"topic_shape": "review"}, no_simulation_resolved=True,
    )
    assert coll.warnings == []


def test_topic_shape_mismatch_silent_when_slot_missing(
    tmp_path: Path,
) -> None:
    """Legacy quests (resumed from a pre-topic_shape checkpoint)
    have no ``topic_shape`` key; the helper must be a no-op rather
    than warn-spam every clarify pass."""
    eng = _mk_engine(tmp_path)
    coll = _capture(eng)
    eng._log_topic_shape_mismatch({}, no_simulation_resolved=False)
    eng._log_topic_shape_mismatch(
        {"clarify_done": True}, no_simulation_resolved=False,
    )
    assert coll.warnings == []


def test_clarify_prompt_carries_topic_shape_vocabulary() -> None:
    """The clarify prompt must list the four shape values + give
    explicit trigger words so the LLM can self-classify reliably."""
    text = (PROMPTS_DIR / "clarify.md").read_text(encoding="utf-8")
    assert "topic_shape" in text
    for shape in ("experimental", "review", "case_study", "opinion"):
        assert shape in text, f"clarify prompt missing shape {shape!r}"
    # Trigger words must be present — the slot's reliability depends
    # on the LLM having concrete cues to pattern-match against the
    # topic string.
    assert "importance" in text.lower() or "history" in text.lower(), (
        "clarify prompt missing review-shape trigger words"
    )


def test_design_prompt_reads_topic_shape_signal() -> None:
    """Design prompt must adapt scope to topic_shape so a review-
    shaped topic doesn't produce a full-bore comparative experiment.
    The four shape-specific scope rules are the contract."""
    text = (PROMPTS_DIR / "design.md").read_text(encoding="utf-8")
    assert "topic_shape" in text
    # All four shapes get explicit scope guidance.
    for shape in ("experimental", "review", "case_study", "opinion"):
        assert shape in text, f"design prompt missing topic_shape={shape}"
    # Review shape specifically must instruct "narrow illustrative
    # measurement" — that's the difference between a useful narrow
    # experiment and the toy-experiment-pretending-to-be-comprehensive
    # failure mode this PR is preventing.
    assert "narrow" in text.lower() and "illustrative" in text.lower(), (
        "design prompt missing the narrow-illustrative scope guidance "
        "for review-shaped topics"
    )
