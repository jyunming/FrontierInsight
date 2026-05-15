"""Seed ``clarify_answers`` from a ``/proposal`` planning markdown.

The companion to ``core/proposal.py``. After the proposal generator
pins the proposal MD into ``knowledge.local_papers`` (so retrieval
sees it), this module short-circuits the clarify node when that
proposal MD is present — parsing the structured H2 sections directly
into the eight clarify slots and saving the engine an LLM call.

Detection contract: a file is treated as a proposal iff its filename
ends with ``-proposal.md``. That suffix is emitted only by
``core.proposal.generate_proposal``; nothing else writes it. So the
short-circuit is safe to fire whenever the suffix is present.

Parsing contract: lenient. If a section is missing or empty, the
slot falls back to a sensible default rather than erroring — the
engine's clarify resolver already tolerates partial answers and
the goal here is "save the LLM call when we can," not "block the
quest if the proposal MD is slightly malformed."
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

_log = logging.getLogger("frontier_insight.proposal_seed")

# Recognized proposal H2 sections, lowercased. Used both for parsing
# and for the "does this look like a proposal at all?" check.
_PROPOSAL_HEADINGS = {
    "tl;dr",
    "background and prior work",
    "hypothesis",
    "experimental plan",
    "success criteria",
    "risks and gotchas",
    "what this quest will not do",
    "recommended next step",
}


def _split_h2_sections(md: str) -> dict[str, str]:
    """Parse the markdown into ``{lowercased-heading: body}``. Headings
    are matched at ``^## `` (loose enough to absorb trailing
    punctuation; the proposal generator emits the canonical names but
    LLMs sometimes append ``.`` / ``:``)."""
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    for line in md.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            heading = m.group(1).rstrip(".:").strip().lower()
            current_key = heading
            current_lines = []
            continue
        if current_key is not None:
            current_lines.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections


def _extract_hypothesis(body: str) -> str:
    """Pull the actual hypothesis sentence from the Hypothesis section.
    The prompt template asks for ``> H: <sentence>`` format; LLMs
    sometimes drop the ``H:`` prefix or the blockquote. Try the
    structured form first, fall back to the first non-empty line."""
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        # Blockquote + H: prefix (the canonical form).
        m = re.match(r"^>\s*H\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Blockquote without H: prefix.
        m = re.match(r"^>\s*(.+)$", line)
        if m:
            return m.group(1).strip()
        # Bare first line (very-lenient fallback).
        return line
    return ""


def _detect_simulatability(plan_body: str) -> str:
    """Decide ``yes`` / ``no`` / ``uncertain`` based on the experimental
    plan's keywords. Same documented set as the clarify agent's
    simulatability slot. Bias toward "yes" — the dominant case is a
    Python experiment, and the engine logs the source so a wrong
    pick is recoverable.

    The keyword lists are intentionally narrow + specific so noise
    in the plan doesn't flip the decision."""
    text = plan_body.lower()
    # Strong "no-simulation" indicators: the user is going to gather
    # qualitative/archival/survey data.
    no_sim_markers = [
        "qualitative", "ethnograph", "interview", "survey ",
        "archival", "historical record", "case study", "field note",
        "policy analysis", "literature synthesis only",
    ]
    if any(m in text for m in no_sim_markers):
        return "no"
    # Strong "yes-simulation" indicators: the plan calls for code.
    yes_sim_markers = [
        "simulation", "monte carlo", "numerical experiment",
        "benchmark", "fit ", "regression", "neural network",
        "matplotlib", "numpy", "scipy", "compute", "algorithm",
    ]
    if any(m in text for m in yes_sim_markers):
        return "yes"
    return "uncertain"


def parse_proposal_md(md: str) -> dict[str, Any] | None:
    """Parse a proposal MD into a clarify_answers-shaped dict.
    Returns ``None`` when the document doesn't look like a proposal
    at all (no recognizable H2 sections from the documented set).

    The returned dict has the engine's auto-mode reduced shape
    (bare-string values, not ``{question, default}`` dicts) so it
    can drop straight into ``state["clarify_answers"]`` without
    further massaging."""
    sections = _split_h2_sections(md)
    overlap = _PROPOSAL_HEADINGS & set(sections.keys())
    # Require at least 3 recognized sections so we don't false-fire on
    # random markdown that happens to share one heading name.
    if len(overlap) < 3:
        return None

    hypothesis_body = sections.get("hypothesis", "")
    success_body = sections.get("success criteria", "")
    plan_body = sections.get("experimental plan", "")
    bg_body = sections.get("background and prior work", "")
    not_do_body = sections.get("what this quest will not do", "")

    hypothesis = _extract_hypothesis(hypothesis_body)
    simulatability = _detect_simulatability(plan_body)

    # comparative_baseline — first line of Background, since the
    # proposal template asks for "what's already known" up top.
    bg_first = next(
        (ln.strip() for ln in bg_body.splitlines() if ln.strip() and not ln.startswith("#")),
        "",
    )

    # success_metric — first non-bullet sentence of Success criteria.
    success_first = ""
    for ln in success_body.splitlines():
        s = ln.strip().lstrip("-•*").strip()
        if s and not s.startswith("#"):
            success_first = s
            break

    # budget — extract any "compute cost" cue from the plan. The
    # prompt template asks for cost annotations per bullet
    # ("single 5-minute simulation on CPU"). Capture the first
    # parenthesised cost-like phrase OR fall back to "a few minutes
    # on a laptop CPU".
    budget = "a few minutes on a laptop CPU"
    cost_match = re.search(
        r"\(([^)]*?(?:minute|hour|gpu|cpu|second|day)[^)]*?)\)",
        plan_body, re.IGNORECASE,
    )
    if cost_match:
        budget = cost_match.group(1).strip()

    # output_kinds + study_depth + paper_venue: the proposal doesn't
    # capture these explicitly. Use the same defaults the auto-mode
    # clarify would have picked.
    answers: dict[str, Any] = {
        "comparative_baseline": bg_first or "(inherited from proposal — see local_papers)",
        "empirical_vs_theoretical": "mixed",
        "simulatability": simulatability,
        "success_metric": success_first or "(see Success criteria in proposal)",
        "budget": budget,
        "output_kinds": ["paper_md", "paper_pdf"],
        "study_depth": "journal-length",
        "paper_venue": "generic",
    }
    if hypothesis:
        # Non-clarify slot but cheap to carry — downstream design /
        # write nodes can read it via state.get for a more concrete
        # prompt.
        answers["_proposal_hypothesis"] = hypothesis
    if not_do_body:
        answers["_proposal_scope_limits"] = not_do_body[:800]
    return answers


def find_proposal_md(local_papers: list[Path]) -> Path | None:
    """Walk ``knowledge.local_papers`` and return the first entry
    whose filename ends with ``-proposal.md`` AND exists on disk.
    That suffix is emitted only by ``core.proposal.generate_proposal``
    so the match is a reliable signal."""
    for p in local_papers:
        try:
            if p.name.endswith("-proposal.md") and p.is_file():
                return p
        except OSError:
            continue
    return None


def seed_clarify_from_local_papers(
    local_papers: list[Path],
) -> tuple[Path, dict[str, Any]] | None:
    """Top-level helper for the engine. Returns ``(proposal_path,
    seeded_answers)`` when a proposal MD is found and parses to a
    non-empty clarify_answers dict; ``None`` otherwise.

    Caller (``Engine._node_clarify``) uses this to short-circuit
    the LLM call when the user is running a quest from a
    ``/proposal``-generated YAML."""
    proposal_path = find_proposal_md(local_papers)
    if proposal_path is None:
        return None
    try:
        text = proposal_path.read_text(encoding="utf-8")
    except OSError as e:
        _log.warning(
            "proposal-seed: could not read %s (%r); falling back to LLM clarify",
            proposal_path, e,
        )
        return None
    answers = parse_proposal_md(text)
    if not answers:
        _log.info(
            "proposal-seed: %s parsed to empty answers; falling back to LLM clarify",
            proposal_path.name,
        )
        return None
    return proposal_path, answers
