"""Tests for core.protocol.ResearchProtocol + derive_protocol."""
from __future__ import annotations

import pytest

from core.config import Config, EngineConfig, OutputConfig, ProviderConfig
from core.protocol import ResearchProtocol, derive_protocol


def _cfg(*, max_iterations=2, evidence_gate=True, execute_replicates=1) -> Config:
    return Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=max_iterations,
            evidence_gate=evidence_gate,
            execute_replicates=execute_replicates,
        ),
        output=OutputConfig(),
    )


@pytest.mark.parametrize("shape,no_sim,expected", [
    ("review", False, "literature_review"),
    ("review", True, "literature_review"),
    ("case_study", True, "case_study"),
    ("opinion", False, "opinion"),
    ("experimental", False, "simulation"),
    ("experimental", True, "data_analysis"),
    ("", True, "data_analysis"),        # unset shape, no-sim
    ("", False, "simulation"),          # unset shape, sim
])
def test_topic_type_derivation(shape, no_sim, expected) -> None:
    state = {
        "topic": "Why overlay metrology matters",
        "clarify_answers": {"topic_shape": shape},
        "no_simulation_resolved": no_sim,
    }
    p = derive_protocol(state, _cfg())
    assert p.topic_type == expected


def test_source_policy_follows_topic_type() -> None:
    # market_current isn't reachable from topic_shape yet, but the
    # per-type default table must give web_current the right policy.
    review = derive_protocol(
        {"topic": "x", "clarify_answers": {"topic_shape": "review"}}, _cfg())
    assert review.source_policy == "academic"
    data = derive_protocol(
        {"topic": "x", "clarify_answers": {"topic_shape": "experimental"},
         "no_simulation_resolved": True}, _cfg())
    assert data.source_policy == "user_data"
    assert data.requires_user_input is True


def test_placeholder_clarify_defaults_are_dropped() -> None:
    """The clarify '(none specified …)' placeholder must NOT leak into the
    protocol as a real baseline/metric."""
    state = {
        "topic": "x",
        "clarify_answers": {
            "comparative_baseline": "(none specified — agent will pick a baseline)",
            "success_metric": "RMSE on the held-out set",
        },
    }
    p = derive_protocol(state, _cfg())
    assert p.baseline == ""
    assert p.success_metric == "RMSE on the held-out set"


def test_replication_and_stopping_from_config() -> None:
    p = derive_protocol(
        {"topic": "x"}, _cfg(max_iterations=3, execute_replicates=5))
    assert p.replication == 5
    assert "3 revise" in p.stopping_criteria
    assert "evidence gate active" in p.stopping_criteria


def test_derive_protocol_is_total_on_empty_state() -> None:
    """A quest with clarify off (no answers) still yields a usable, valid
    protocol — every field defaulted."""
    p = derive_protocol({}, _cfg())
    assert isinstance(p, ResearchProtocol)
    assert p.research_question == "(unspecified)"
    assert p.topic_type in ("simulation", "data_analysis")  # depends on no_sim default
    assert p.as_block()  # renders without error


def test_as_block_includes_key_fields() -> None:
    p = derive_protocol(
        {"topic": "Overlay error variance in EUV",
         "clarify_answers": {"topic_shape": "review",
                             "success_metric": "share of die out of budget"}},
        _cfg())
    block = p.as_block()
    assert "topic type: literature_review" in block
    assert "source policy: academic" in block
    assert "share of die out of budget" in block


def test_malformed_clarify_answers_do_not_crash() -> None:
    p = derive_protocol({"topic": "x", "clarify_answers": "not-a-dict"}, _cfg())
    assert isinstance(p, ResearchProtocol)
