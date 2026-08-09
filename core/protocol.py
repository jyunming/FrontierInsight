"""Typed ``ResearchProtocol`` — the explicit contract for a quest.

Today a quest's "contract" (what it IS and what counts as enough) lives
scattered across loosely-typed clarify slots (``topic_shape``,
``simulatability``, ``comparative_baseline``, ``success_metric`` …) plus
rules embedded in prompts. This module consolidates that into ONE typed,
validated object derived from the already-resolved clarify answers + the
engine config — no new LLM call.

The first increment made the implicit contract **explicit and typed**
and handed it to the ``evidence_gate`` so its sufficiency judgement is
contract-aware (it can see the declared topic type, source policy,
baseline, and success metric).

This increment promotes ``topic_type`` to a first-class routing input
via ``ROUTE_MATRIX`` below: a single declarative table mapping every
topic type to the graph path it expects (``simulation`` vs
``no_simulation``) and the source policy it implies. The engine consults
it at the design fork — the resolved ``no_simulation`` flag stays
authoritative (config/clarify wins), but when the topic type's expected
path disagrees with the resolved route the engine logs a route-mismatch
WARNING so the divergence is visible in run.log. The same table is the
single source of truth for each type's source policy.

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
    "survey",             # descriptive / historical synthesis, no experiment AND no dataset
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

# The two outcomes of the design fork in ``_build_graph``:
#   "simulation"     → design → implement → execute → execute_reflect → analyze
#   "no_simulation"  → design → auto_collect_data → wait_for_data → data_load
RoutePath = Literal["simulation", "no_simulation"]

# Per-topic-type contract: the graph path the type expects + the source
# policy it implies. SINGLE SOURCE OF TRUTH consulted in two places:
#   * the design fork (``path``) — for the route-mismatch WARNING; the
#     resolved ``no_simulation`` flag still decides the actual edge.
#   * ``derive_protocol`` (``source_policy``) — the live protocol's policy.
# ``engineering`` / ``market_current`` paths are forward-looking: the
# current deterministic ``_derive_topic_type`` can't yet produce those
# types (they await the LLM-classified deriver), but the table is ready.
ROUTE_MATRIX: dict[str, dict[str, str]] = {
    "simulation":        {"path": "simulation",    "source_policy": "academic"},
    "data_analysis":     {"path": "no_simulation", "source_policy": "user_data"},
    "literature_review": {"path": "no_simulation", "source_policy": "academic"},
    "survey":            {"path": "no_simulation", "source_policy": "academic"},
    "case_study":        {"path": "no_simulation", "source_policy": "mixed"},
    "market_current":    {"path": "no_simulation", "source_policy": "web_current"},
    "engineering":       {"path": "simulation",    "source_policy": "academic"},
    "opinion":           {"path": "no_simulation", "source_policy": "mixed"},
    "unknown":           {"path": "simulation",    "source_policy": "mixed"},
}


def route_for_topic_type(topic_type: str) -> RoutePath:
    """The graph path a topic type expects: ``"simulation"`` (run code) or
    ``"no_simulation"`` (analyse supplied/collected data). Falls back to
    ``"simulation"`` (the historical default) for unrecognised types."""
    entry = ROUTE_MATRIX.get(str(topic_type or ""), ROUTE_MATRIX["unknown"])
    return entry["path"]  # type: ignore[return-value]


def source_policy_for(topic_type: str) -> SourcePolicy:
    """The source policy a topic type implies (academic / web_current /
    user_data / mixed). Single source of truth = ``ROUTE_MATRIX``."""
    entry = ROUTE_MATRIX.get(str(topic_type or ""), ROUTE_MATRIX["unknown"])
    return entry["source_policy"]  # type: ignore[return-value]


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


# Default expected-evidence per topic type (the source policy lives in
# ``ROUTE_MATRIX`` — single source of truth). Deterministic v1; a later
# LLM-classified deriver can override these per question.
_EXPECTED_EVIDENCE: dict[str, str] = {
    "simulation":
        "quantitative results from a reproducible experiment, plus prior "
        "work to contextualise them",
    "data_analysis":
        "the user's / collected dataset, analysed; supporting literature",
    "literature_review":
        "a representative, on-topic body of published sources to synthesise",
    "survey":
        "a representative, on-topic body of published sources to synthesise "
        "into a descriptive history / overview — no experiment or dataset",
    "case_study":
        "concrete evidence about the one system in question + comparable "
        "prior cases",
    "market_current":
        "recent, dated web sources (reports, filings, news) — not just "
        "academic papers",
    "engineering":
        "a concrete design/spec grounded in established techniques",
    "opinion":
        "evidence for the specific claim the position rests on",
    "unknown": "evidence that directly addresses the question",
}


# Markers that flag a markets / current-events question — one whose answer
# turns on RECENT, dated web sources (filings, reports, news), not just
# academic papers. Deliberately finance/geopolitics/current-events specific
# (no bare "current"/"trend", which collide with physics: "current density",
# "trend line"), so the deterministic upgrade below doesn't misfire on
# scientific topics. The richer LLM-classified deriver will replace this.
_MARKET_CURRENT_MARKERS = (
    "market share", "stock price", "share price", "stock market", "revenue",
    "valuation", "gdp", "inflation", "interest rate", "election", "sanction",
    "earnings", "quarterly", "ipo", "recession", "geopolitic", "tariff",
    "exchange rate", "commodity price", "market cap", "financial status",
    "box office", "unemployment rate",
)


def _looks_market_current(topic: object) -> bool:
    """True when the question text reads as a markets / current-events
    topic — the one case whose ``source_policy`` must be ``web_current``.
    Conservative substring match against :data:`_MARKET_CURRENT_MARKERS`."""
    t = str(topic or "").lower()
    return bool(t) and any(m in t for m in _MARKET_CURRENT_MARKERS)


def _derive_topic_type(topic_shape: str, no_simulation: bool) -> TopicType:
    """Map the four clarify ``topic_shape`` values × the simulatability
    decision onto a TopicType. (market_current / engineering aren't in
    topic_shape yet — the LLM-classified follow-up will reach them.)"""
    shape = (topic_shape or "").strip().lower()
    if shape == "survey":
        return "survey"
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
    # Deterministic upgrade: a markets / current-events question (detected
    # from the topic text) needs recent web sources. The clarify
    # ``topic_shape`` vocabulary can't express this, so without it the
    # ``market_current`` branch of ROUTE_MATRIX (source_policy=web_current)
    # was unreachable dead code. Only upgrade non-experimental shapes — a
    # genuine simulation keeps its type.
    if topic_type in ("data_analysis", "literature_review", "survey",
                      "case_study", "unknown") and _looks_market_current(state.get("topic")):
        topic_type = "market_current"
    expected_evidence = _EXPECTED_EVIDENCE.get(
        topic_type, _EXPECTED_EVIDENCE["unknown"])
    source_policy = source_policy_for(topic_type)

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
