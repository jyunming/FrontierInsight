"""Typed ``ResearchProtocol`` — the explicit contract for a quest.

Today a quest's "contract" (what it IS and what counts as enough) lives
scattered across loosely-typed clarify slots (``topic_shape``,
``simulatability``, ``comparative_baseline``, ``success_metric`` …) plus
rules embedded in prompts. This module consolidates that into ONE typed,
validated object derived from the already-resolved clarify answers + the
engine config — no new LLM call.

This first increment makes the implicit contract **explicit and typed**
and hands it to the ``evidence_gate`` so its sufficiency judgement is
contract-aware (it can see the declared topic type, source policy,
baseline, and success metric). A later increment promotes ``topic_type``
to a first-class graph route controller; the model is intentionally the
foundation those follow-ups build on.

Pure-stdlib + Pydantic; no engine import (avoids a cycle — the engine
imports this, not the reverse).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# What kind of research this is. Broader than clarify's four
# ``topic_shape`` values because the *routing/source* implications differ
# (a market/current-events question NEEDS fresh web sources; a literature
# review needs academic ones). The deterministic deriver below maps the
# available slots onto this; the richer LLM-classified version is a
# follow-up.
TopicType = Literal[
    "simulation",         # run code, measure something
    "data_analysis",      # analyse supplied / collected data, no simulation
    "literature_review",  # synthesise existing published work
    "case_study",         # characterise one system
    "market_current",     # markets / current events — needs recent sources
    "engineering",        # build/spec a system
    "opinion",            # argue a position
    "unknown",
]

# Which sources the question REQUIRES to be answerable.
SourcePolicy = Literal[
    "academic",     # peer-reviewed / arXiv literature
    "web_current",  # recent web sources are mandatory (markets, current events)
    "user_data",    # analyses supplied/collected data — the user's drop, or
                    # what the no-sim auto-collector gathered — not a simulation
    "mixed",        # academic + web both legitimate
]


class ResearchProtocol(BaseModel):
    """The validated contract a quest is held to. Derived from clarify +
    config; consumed (for now) by the evidence_gate."""

    research_question: str
    topic_type: TopicType = "unknown"
    # Plain-language statement of what evidence would actually answer the
    # question (helps the gate decide whether the assembled evidence fits).
    expected_evidence: str = ""
    source_policy: SourcePolicy = "mixed"
    # The comparator the study should be measured against ("" = none pinned).
    baseline: str = ""
    # What number, moving which way, is the headline result ("" = unset).
    success_metric: str = ""
    # True for no-simulation studies, which run on supplied/collected data
    # rather than a simulation and so MAY pause for the user to drop files
    # (only if the auto-collector gathered nothing). Not a hard guarantee
    # the quest will block — a signal that user data may be needed.
    requires_user_input: bool = False
    # How many independent runs back a quantitative claim (1 = single-shot).
    replication: int = Field(default=1, ge=1)
    # Human-readable stopping criteria (iteration cap + whether the gate
    # is active), so a reader knows when the quest will stop.
    stopping_criteria: str = ""

    def as_block(self) -> str:
        """Render as a compact markdown block for a prompt."""
        lines = [
            f"- research question: {self.research_question}",
            f"- topic type: {self.topic_type}",
            f"- source policy: {self.source_policy}",
        ]
        if self.expected_evidence:
            lines.append(f"- expected evidence: {self.expected_evidence}")
        if self.baseline:
            lines.append(f"- baseline / comparator: {self.baseline}")
        if self.success_metric:
            lines.append(f"- success metric: {self.success_metric}")
        lines.append(f"- replication: {self.replication} run(s)")
        if self.requires_user_input:
            lines.append("- may need user-supplied data: yes")
        if self.stopping_criteria:
            lines.append(f"- stopping criteria: {self.stopping_criteria}")
        return "\n".join(lines)


# Default expected-evidence + source-policy per topic type. Deterministic
# v1; a later LLM-classified deriver can override these per question.
_TYPE_DEFAULTS: dict[str, tuple[str, SourcePolicy]] = {
    "simulation": (
        "quantitative results from a reproducible experiment, plus prior "
        "work to contextualise them", "academic"),
    "data_analysis": (
        "the user's / collected dataset, analysed; supporting literature",
        "user_data"),
    "literature_review": (
        "a representative, on-topic body of published sources to synthesise",
        "academic"),
    "case_study": (
        "concrete evidence about the one system in question + comparable "
        "prior cases", "mixed"),
    "market_current": (
        "recent, dated web sources (reports, filings, news) — not just "
        "academic papers", "web_current"),
    "engineering": (
        "a concrete design/spec grounded in established techniques",
        "academic"),
    "opinion": (
        "evidence for the specific claim the position rests on", "mixed"),
    "unknown": ("evidence that directly addresses the question", "mixed"),
}


def _derive_topic_type(topic_shape: str, no_simulation: bool) -> TopicType:
    """Map the four clarify ``topic_shape`` values × the simulatability
    decision onto a TopicType. (market_current / engineering aren't in
    topic_shape yet — the LLM-classified follow-up will reach them.)"""
    shape = (topic_shape or "").strip().lower()
    if shape == "review":
        return "literature_review"
    if shape == "case_study":
        return "case_study"
    if shape == "opinion":
        return "opinion"
    # shape == "experimental" or unset
    return "data_analysis" if no_simulation else "simulation"


def derive_protocol(state: dict[str, Any], config: Any) -> ResearchProtocol:
    """Build the typed protocol from the resolved clarify answers + engine
    config. Pure + total: every field has a sensible default, so a quest
    with clarify off still gets a usable protocol."""
    answers = state.get("clarify_answers") or {}
    if not isinstance(answers, dict):
        answers = {}
    no_simulation = bool(state.get("no_simulation_resolved"))
    topic_type = _derive_topic_type(
        str(answers.get("topic_shape") or ""), no_simulation)
    expected_evidence, source_policy = _TYPE_DEFAULTS.get(
        topic_type, _TYPE_DEFAULTS["unknown"])

    def _clean(v: object) -> str:
        s = str(v or "").strip()
        # Drop the clarify "(none specified …)" placeholder defaults.
        return "" if s.lower().startswith("(none") else s

    eng = getattr(config, "engine", None)
    max_iter = getattr(eng, "max_iterations", 0) if eng else 0
    gate_on = bool(getattr(eng, "evidence_gate", False)) if eng else False
    replication = int(getattr(eng, "execute_replicates", 1) or 1) if eng else 1

    return ResearchProtocol(
        research_question=str(state.get("topic") or "").strip() or "(unspecified)",
        topic_type=topic_type,
        expected_evidence=expected_evidence,
        source_policy=source_policy,
        baseline=_clean(answers.get("comparative_baseline")),
        success_metric=_clean(answers.get("success_metric")),
        requires_user_input=no_simulation,
        replication=max(1, replication),
        stopping_criteria=(
            f"at most {max_iter} revise iteration(s)"
            + ("; evidence gate active" if gate_on else "")
        ),
    )
