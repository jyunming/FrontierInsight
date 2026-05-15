"""Tests for ``core/proposal_seed.py`` — the helper that parses a
``/proposal``-generated markdown into ``clarify_answers`` so the
engine can skip the clarify LLM call when running a quest from
that companion YAML.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.proposal_seed import (
    find_proposal_md,
    parse_proposal_md,
    seed_clarify_from_local_papers,
)

_REAL_PROPOSAL = """\
# Pre-quest proposal

## TL;DR
Three integrators (RK4, Verlet, Euler) over 10000 periods will rank
in the expected order. The biggest risk is the timestep being
chosen poorly and inflating Euler's drift artificially.

## Background and prior work
RK4 has 4th-order convergence; Velocity-Verlet is symplectic so
energy drift is bounded; forward Euler is 1st-order and diverges
on long integrations. Standard textbook material.

## Hypothesis

> H: For the same wall-clock budget, RK4 < Verlet < Euler in
> energy drift, with Euler diverging by 100x at the longest run.

## Experimental plan

- Numerical simulation of the damped harmonic oscillator over 10000
  periods (~5-minute simulation on CPU).
- Compute relative energy drift per integrator per timestep.
- Sweep dt across {0.01, 0.001} (Monte Carlo over 100 trials each).

## Success criteria

The primary result is the relative energy drift `|E(T) - E(0)| /
E(0)` ranked monotonically across the three integrators.

A null outcome would be: all three integrators show identical
drift within 1% (would refute the hypothesis).

## Risks and gotchas

- Floating-point underflow on Verlet at dt < 0.0001.
- Euler diverging so fast we lose the comparison.

## What this quest will NOT do

- Compare against semi-implicit schemes.
- Test on stiff systems.

## Recommended next step

**Proceed** — the plan is solid and the cost is bounded.
"""


def test_parse_proposal_md_returns_clarify_shaped_dict() -> None:
    answers = parse_proposal_md(_REAL_PROPOSAL)
    assert answers is not None
    # All 8 documented clarify slots populated.
    for slot in (
        "comparative_baseline", "empirical_vs_theoretical",
        "simulatability", "success_metric", "budget",
        "output_kinds", "study_depth", "paper_venue",
    ):
        assert slot in answers, f"missing slot: {slot}"
    # Bonus slots carry through for downstream nodes.
    assert "_proposal_hypothesis" in answers
    assert "_proposal_scope_limits" in answers


def test_parse_proposal_md_extracts_hypothesis_from_blockquote() -> None:
    answers = parse_proposal_md(_REAL_PROPOSAL)
    assert answers is not None
    h = answers["_proposal_hypothesis"]
    assert "RK4" in h and "Verlet" in h, f"hypothesis not extracted: {h!r}"


def test_parse_proposal_md_detects_simulatable_topic() -> None:
    """The plan contains ``simulation`` and ``Monte Carlo`` keywords
    so simulatability must be ``yes``, not ``uncertain``."""
    answers = parse_proposal_md(_REAL_PROPOSAL)
    assert answers is not None
    assert answers["simulatability"] == "yes"


def test_parse_proposal_md_detects_non_simulatable_topic() -> None:
    """Qualitative / archival topics map to ``simulatability: no``."""
    md = """\
## TL;DR
Compare the 1968 protests in Paris vs Mexico City.

## Background and prior work
Cited in standard reference works.

## Hypothesis

> H: The two were structurally similar but had divergent outcomes.

## Experimental plan

- Qualitative comparison of archival sources from each city.
- Historical record analysis of newspaper reports.
- Case study writeup, no simulation.

## Success criteria

A defensible side-by-side analysis with at least 5 primary sources cited.
"""
    answers = parse_proposal_md(md)
    assert answers is not None
    assert answers["simulatability"] == "no"


def test_parse_proposal_md_returns_none_for_non_proposal_doc() -> None:
    """Random markdown that happens to share one heading must not
    short-circuit the LLM clarify call. We require 3+ proposal-shape
    headings to false-positive only on docs we're certain about."""
    assert parse_proposal_md("# Just a note\n\n## Hypothesis\n\nx") is None
    assert parse_proposal_md("just a paragraph") is None
    assert parse_proposal_md("") is None


def test_parse_proposal_md_extracts_budget_from_plan_cost_cue() -> None:
    answers = parse_proposal_md(_REAL_PROPOSAL)
    assert answers is not None
    # The plan has ``(~5-minute simulation on CPU)`` — should land in budget.
    assert "5-minute" in answers["budget"] or "minute" in answers["budget"]


def test_find_proposal_md_matches_only_proposal_suffix(tmp_path: Path) -> None:
    # Two files, only one has the right suffix.
    p1 = tmp_path / "1778800000-foo-aabbcc-proposal.md"
    p1.write_text("# P", encoding="utf-8")
    p2 = tmp_path / "user-notes.md"
    p2.write_text("# Notes", encoding="utf-8")
    found = find_proposal_md([p2, p1])
    assert found == p1


def test_find_proposal_md_skips_missing_files(tmp_path: Path) -> None:
    """A pinned path that doesn't exist on disk shouldn't match."""
    p1 = tmp_path / "ghost-proposal.md"  # never created
    p2 = tmp_path / "real-proposal.md"
    p2.write_text("# Real", encoding="utf-8")
    found = find_proposal_md([p1, p2])
    assert found == p2


def test_seed_clarify_from_local_papers_end_to_end(tmp_path: Path) -> None:
    p = tmp_path / "x-proposal.md"
    p.write_text(_REAL_PROPOSAL, encoding="utf-8")
    result = seed_clarify_from_local_papers([p])
    assert result is not None
    proposal_path, answers = result
    assert proposal_path == p
    assert answers["simulatability"] == "yes"


def test_seed_clarify_from_local_papers_returns_none_without_proposal(
    tmp_path: Path,
) -> None:
    """No ``*-proposal.md`` in local_papers → no short-circuit, the
    engine falls through to the regular LLM clarify path."""
    p = tmp_path / "just-a-paper.md"
    p.write_text("# Random paper", encoding="utf-8")
    assert seed_clarify_from_local_papers([p]) is None
    assert seed_clarify_from_local_papers([]) is None
