"""Stratification-by-factor pipeline tests.

Pin the three-prompt contract that lets the engine surface
per-stratum findings (per-clip-class, per-method, per-dataset, ...)
instead of collapsing everything into aggregate means:

  1. ``implement`` prompt must instruct the experiment script to emit
     a ``by_<factor>`` key in ``RESULT_JSON`` when natural strata exist.
  2. ``analyze`` prompt must instruct the LLM to surface per-stratum
     findings as separately tagged bullets, not merge them into the
     aggregate.
  3. ``write`` prompt must forbid collapsing stratified findings into
     an aggregate-only Results section.

These are prompt-content tests — they don't fire LLMs. The point is
"if someone trims the prompt, the contract regression is caught
before it ships."
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents"


def _read(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def test_implement_prompt_requires_by_factor_stratification() -> None:
    """``implement`` prompt must (a) instruct experiment.py to emit
    ``by_<factor>`` when natural strata exist, (b) show a stratified
    RESULT_JSON example so the LLM has a concrete template, and (c)
    explicitly carve out the no-natural-strata case so single-condition
    experiments aren't padded with fake singleton groups."""
    text = _read("implement")
    assert "Stratify when natural strata exist" in text, (
        "implement prompt missing the stratify-when-natural-strata "
        "header — without it the LLM defaults to aggregate-only."
    )
    assert "by_<factor>" in text or "by_clip_class" in text, (
        "implement prompt missing the by_<factor> key convention."
    )
    # Concrete example so the LLM has a template to follow, not just a rule.
    assert "by_clip_class" in text, (
        "implement prompt missing the stratified RESULT_JSON example."
    )
    # Single-condition carve-out so the LLM doesn't manufacture
    # fake strata for genuinely-aggregate experiments.
    assert "single-condition" in text.lower(), (
        "implement prompt missing the single-condition exception — "
        "without it the LLM may invent fake singleton strata."
    )


def test_analyze_prompt_surfaces_per_stratum_findings() -> None:
    """``analyze`` must tell the LLM that when result_json has a
    ``by_<factor>`` key, per-stratum findings must surface as tagged
    bullets in ``key_findings`` so the writer can render them as a
    per-stratum table. A uniform-effect carve-out keeps the prompt
    from forcing useless per-stratum prose when the strata all behave
    the same way."""
    text = _read("analyze")
    assert "by_<factor>" in text, (
        "analyze prompt missing the by_<factor> trigger — without it "
        "the LLM has no signal that the result_json carries strata."
    )
    assert "per-stratum" in text.lower() or "stratum-level" in text.lower(), (
        "analyze prompt missing the per-stratum-findings instruction."
    )
    # The tag convention is how the writer downstream knows which
    # findings to render as a per-stratum table.
    assert "by_clip_class:" in text or "[by_" in text, (
        "analyze prompt missing the tagged-bullet convention "
        "(``[by_<factor>:<stratum>]`` prefix)."
    )
    # Uniform-effect carve-out so the LLM doesn't pad with N
    # bullets when one ``effect is uniform across strata`` suffices.
    assert "uniform" in text.lower(), (
        "analyze prompt missing the uniform-effect carve-out."
    )


def test_write_prompt_forbids_collapsing_stratified_findings() -> None:
    """``write`` must include a Honesty-constraints bullet forbidding
    aggregate-only Results when the analysis block carries
    stratum-level findings. The reader-utility rationale ("did this
    work for MY use case") is the contract; without it the writer
    will silently merge strata to keep the prose flowing."""
    text = _read("write")
    # The bullet must reference the analyze-emitted tag convention so
    # the writer knows what shape of input to look for.
    assert "[by_" in text or "by_<factor>" in text, (
        "write prompt missing the stratum-tag reference — without it "
        "the writer has no signal that the analysis carries strata."
    )
    assert "Collapsing stratified findings" in text or "stratified" in text.lower(), (
        "write prompt missing the don't-collapse-strata rule."
    )
    # The fix is a per-stratum table OR per-stratum subsection; pin
    # at least one of those terms so future trimming doesn't leave
    # the rule without an actionable remedy.
    assert "per-stratum table" in text or "per-stratum subsection" in text, (
        "write prompt mentions stratification but doesn't tell the "
        "writer HOW to render it (table or subsection)."
    )
