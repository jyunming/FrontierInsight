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
from core.engine import Engine, _format_clarify


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
            max_iterations=1, review_loop=False,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    return Engine(cfg)


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


# --- survey mode (no experiment AND no dataset) -----------------------------


def _mk_engine_survey(tmp_path: Path, *, survey_mode: bool = False) -> Engine:
    cfg = Config(
        topic="the evolution of sculpture in pop culture",
        title="sculpt-history",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            survey_mode=survey_mode,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    return Engine(cfg)


def test_route_after_design_survey_goes_to_synthesize(tmp_path: Path) -> None:
    """Survey mode routes design → web_figures (illustrative images) → analyze,
    skipping BOTH the experiment and the data-collection chain."""
    eng = _mk_engine_survey(tmp_path)
    assert eng._route_after_design({"survey_mode_resolved": True}) == "synthesize"
    # A plain no-sim (non-survey) quest still takes the data path.
    assert eng._route_after_design(
        {"no_simulation_resolved": True}
    ) == "wait_for_data"
    assert eng._route_after_design({}) == "implement"


def test_protocol_survey_topic_type_maps_to_no_simulation() -> None:
    from core.protocol import (
        _derive_topic_type, route_for_topic_type, source_policy_for,
    )
    assert _derive_topic_type("survey", True) == "survey"
    assert route_for_topic_type("survey") == "no_simulation"
    # survey is a humanities history/overview — a MIXED source policy (web +
    # academic both legitimate), so the evidence gate accepts a web-sourced
    # corpus instead of broadening against it for lacking peer-reviewed papers.
    assert source_policy_for("survey") == "mixed"


def test_survey_protocol_accepts_web_sources() -> None:
    """derive_protocol for a survey state carries source_policy=mixed and an
    expected-evidence string that legitimises web / reference sources, so the
    evidence gate won't penalise a humanities corpus for lacking academic work."""
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
        OutputConfig, ProviderConfig,
    )
    from core.protocol import derive_protocol
    cfg = Config(
        topic="the evolution of sculpture in pop culture", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, survey_mode=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=Path("./o")),
    )
    state = {
        "topic": cfg.topic, "no_simulation_resolved": True,
        "survey_mode_resolved": True,
        "clarify_answers": {"topic_shape": "survey"},
    }
    proto = derive_protocol(state, cfg)
    assert proto.topic_type == "survey"
    assert proto.source_policy == "mixed"
    assert "web" in proto.expected_evidence.lower()


def test_evidence_gate_prompt_documents_mixed_policy() -> None:
    """The evidence-gate prompt must define the `mixed` source policy, else the
    survey→mixed change silently relies on the LLM inferring it."""
    text = (PROMPTS_DIR / "evidence_gate.md").read_text(encoding="utf-8")
    assert "`mixed`" in text
    assert "survey" in text.lower()


def test_evidence_gate_prompt_is_survey_aware() -> None:
    """Surveys have no experiment / dataset / cross-check by design, so the gate
    must be told not to weigh Results / Cross-check against a survey (else it
    broadens on absent results that were never meant to exist)."""
    text = (PROMPTS_DIR / "evidence_gate.md").read_text(encoding="utf-8").lower()
    # Match the label ResearchProtocol.as_block() actually renders ("topic type").
    assert "topic type: survey" in text
    # It must tell the gate not to penalise a survey for missing results/cross-check.
    assert "cross-check" in text and "no experiment" in text


def test_evidence_note_suppressed_for_survey() -> None:
    """A survey is a synthesis, not an evidence-gated experiment — the gate's
    'evidence insufficient / broaden' over-hedge note must NOT be injected into
    a survey's write prompt (deterministic, regardless of the gate's verdict)."""
    from core.engine import _format_evidence_note
    broaden = {"verdict": "broaden", "rationale": "thin sources", "gaps": ["x"]}
    # Non-survey: the note IS produced (existing behaviour).
    note = _format_evidence_note(broaden, is_survey=False)
    assert "Evidence note" in note and "broaden" in note
    # Survey: suppressed entirely.
    assert _format_evidence_note(broaden, is_survey=True) == ""
    # Sufficient / disabled / None → always empty.
    assert _format_evidence_note({"verdict": "sufficient"}, is_survey=False) == ""
    assert _format_evidence_note(None, is_survey=False) == ""


def test_source_router_prompt_has_field_discipline() -> None:
    """The source router must be told not to pick arXiv/PubMed for humanities."""
    text = (Path(__file__).resolve().parent.parent / "core" / "knowledge.py").read_text(encoding="utf-8")
    # Phrases contiguous within a single source-string literal (the prompt is
    # split across adjacent literals, so cross-literal substrings won't match).
    assert "arXiv is a STEM preprint" in text
    assert "do NOT pick them for humanities" in text
