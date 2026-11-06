"""The prompts that teach a figure body FI's drawing calls name only calls FI draws: a theory curve is a reference of the
plan (``fig.ref(panel, "<id>")``, which FI evaluates along x from the plan's formula), never a curve the body computes, and
a typed constant is drawn only when it equals one reference of the plan."""

from __future__ import annotations

from pathlib import Path

from core import analysis_frame, figure_data_check, run_checks

ROOT = Path(__file__).resolve().parents[1]
PRIMITIVES = (ROOT / "agents" / "figure_primitives.md").read_text(encoding="utf-8")


def _prompt_texts() -> dict[str, str]:
    """Every text a model is shown about figure bodies."""
    texts = {f"agents/{p.name}": p.read_text(encoding="utf-8") for p in (ROOT / "agents").glob("*.md")}
    texts["analysis_frame._CALL_NAMES"] = analysis_frame._CALL_NAMES
    texts["run_checks.analysis_bodies_rules"] = run_checks.analysis_bodies_rules(calls=True)
    texts["figure_data_check.directive"] = figure_data_check.directive([])
    return texts


def test_prompt_mentions_formula_and_not_curve_call() -> None:
    assert "fig.curve" not in PRIMITIVES
    # a theory curve is a reference of the plan, drawn by FI from the plan's formula
    assert "theory" in PRIMITIVES and "formula" in PRIMITIVES
    assert 'fig.ref(panel, "<reference id>")' in PRIMITIVES or "fig.ref(panel, \"id\")" in PRIMITIVES
    # a typed constant is drawn only when it equals a reference of the plan; otherwise it is left out
    assert "left out" in PRIMITIVES


def test_no_prompt_names_a_removed_call() -> None:
    named = [where for where, text in _prompt_texts().items() if "fig.curve" in text]
    assert named == [], named
    # the constant form no longer reads as a free number "stated in the figure"
    assert "stated in the figure" not in PRIMITIVES
