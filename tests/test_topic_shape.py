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


def test_resolver_coerces_yaml_bool_false_to_no(tmp_path: Path) -> None:
    """PyYAML parses bare ``no`` as boolean ``False``, which is how
    users idiomatically write the slot in YAML. The resolver must
    treat ``simulatability: False`` as ``"no"`` (route to
    NO_SIMULATION), not fall through silently to the legacy fallback.

    Before this fix, ``clarify_overrides: simulatability: no``
    (unquoted) parsed to ``{simulatability: False}``, the resolver
    rejected the bool, and the engine ran SIMULATE despite the user
    explicitly asking for NO_SIMULATION.
    """
    eng = _mk_engine(tmp_path)
    # Bool False at top level.
    assert eng._resolve_no_simulation_from_clarify(
        {"simulatability": False},
    ) is True
    # Bool True at top level.
    assert eng._resolve_no_simulation_from_clarify(
        {"simulatability": True},
    ) is False
    # Bool nested in the dict-shape (full slot with reason).
    assert eng._resolve_no_simulation_from_clarify(
        {"simulatability": {"default": False, "reason": "real-data quest"}},
    ) is True


@pytest.mark.asyncio
async def test_short_circuit_populates_topic_shape_when_unset(
    tmp_path: Path,
) -> None:
    """When clarify_mode=auto and all 7 ORIGINAL slots are pinned via
    clarify_overrides (the common case from the interview), the
    short-circuit must still fire AND populate ``topic_shape`` with
    a safe default. Otherwise downstream consumers see a missing
    slot and the mismatch helper is unable to fire.

    Regression test for: adding ``topic_shape`` to ``known_slots``
    would have broken the short-circuit for every existing interview-
    generated YAML (which doesn't pin the new slot yet)."""
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
        OutputConfig, ProviderConfig,
    )
    overrides = {
        "comparative_baseline": "(none)",
        "empirical_vs_theoretical": "empirical",
        "simulatability": "yes",
        "success_metric": "F1 >= 0.9",
        "budget": "5 min CPU",
        "output_kinds": ["paper_md"],
        "study_depth": "journal-length",
        "paper_venue": "generic",
        # NB: topic_shape intentionally not pinned — emulates current
        # interview output. The short-circuit must still fire.
    }
    cfg = Config(
        topic="some topic",
        title="x",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="auto",
            clarify_overrides=overrides,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    # Build a stub client that asserts chat is NEVER called — the
    # short-circuit's whole purpose is to skip the LLM round-trip.
    from unittest.mock import AsyncMock
    chat_mock = AsyncMock()
    eng._client = type("Stub", (), {"chat": chat_mock})()

    patch = await eng._node_clarify({"topic": "x"})

    chat_mock.assert_not_called()
    assert patch["clarify_done"] is True
    # The original 7 survive verbatim.
    for k, v in overrides.items():
        assert patch["clarify_answers"][k] == v
    # The 8th appeared with a safe default.
    assert patch["clarify_answers"]["topic_shape"] == "experimental"


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


# --- survey mode (no experiment AND no dataset) -----------------------------


def _mk_engine_survey(tmp_path: Path, *, survey_mode: bool = False) -> Engine:
    cfg = Config(
        topic="the evolution of sculpture in pop culture",
        title="sculpt-history",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="auto",
            survey_mode=survey_mode,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    return Engine(cfg)


def test_survey_via_topic_shape_resolves_and_implies_no_sim(tmp_path: Path) -> None:
    """The clarify agent classifying ``topic_shape == 'survey'`` turns on
    survey mode, and survey ALWAYS implies no-simulation (no experiment,
    no dataset)."""
    eng = _mk_engine_survey(tmp_path)
    assert eng._resolve_survey_from_clarify({"topic_shape": "survey"}) is True
    modes = eng._resolve_modes({"topic_shape": "survey"})
    assert modes == {
        "survey_mode_resolved": True, "no_simulation_resolved": True,
    }


def test_survey_via_yaml_flag_resolves(tmp_path: Path) -> None:
    """``engine.survey_mode: true`` pins survey mode regardless of clarify."""
    eng = _mk_engine_survey(tmp_path, survey_mode=True)
    assert eng._resolve_survey_from_clarify({}) is True
    assert eng._resolve_modes({}) == {
        "survey_mode_resolved": True, "no_simulation_resolved": True,
    }


def test_non_survey_shapes_are_not_survey(tmp_path: Path) -> None:
    """experimental / review are NOT survey. review still resolves to the
    data (no-sim) path only when simulatability says so — never survey."""
    eng = _mk_engine_survey(tmp_path)
    assert eng._resolve_modes({"topic_shape": "experimental"}) == {
        "survey_mode_resolved": False, "no_simulation_resolved": False,
    }
    # review + simulatability:no → no-sim data path, but NOT survey.
    assert eng._resolve_modes(
        {"topic_shape": "review", "simulatability": "no"}
    ) == {"survey_mode_resolved": False, "no_simulation_resolved": True}


def test_route_after_design_survey_goes_to_synthesize(tmp_path: Path) -> None:
    """Survey mode routes design → web_figures (illustrative images) → analyze,
    skipping BOTH the experiment and the data-collection chain."""
    eng = _mk_engine_survey(tmp_path)
    assert eng._route_after_design({"survey_mode_resolved": True}) == "synthesize"
    # A plain no-sim (non-survey) quest still takes the data path.
    assert eng._route_after_design(
        {"no_simulation_resolved": True}
    ) == "auto_collect_data"
    assert eng._route_after_design({}) == "implement"


def test_default_clarify_questions_include_survey(tmp_path: Path) -> None:
    q = _default_clarify_questions("history of X")
    assert "survey" in q["topic_shape"]["question"].lower()


def test_clarify_prompt_carries_survey_vocabulary() -> None:
    text = (PROMPTS_DIR / "clarify.md").read_text(encoding="utf-8")
    assert "survey" in text.lower(), "clarify prompt missing survey shape"
    # The design prompt must tell the survey path to NOT design an experiment.
    design = (PROMPTS_DIR / "design.md").read_text(encoding="utf-8")
    assert "survey" in design.lower()


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
