"""A paper that leaves out a figure its design planned and its run drew gets it
back when the draft is written (``core/engine.py``: ``_place_missing_figures``,
called by the write node), and the review still catches one that goes missing
some other way (``_missing_planned_figures``, the review node and the rewrite
route)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import (
    Engine,
    _figure_caption_findings,
    _format_review_for_writer,
    _hits_need_only_a_rewrite,
    _missing_planned_figures,
    _place_missing_figures,
)

DESIGN = {"figures_planned": ["final_size.png", "outbreak_prob.png", "gap_fraction.png"]}
FIGURES = ["gap_fraction.png", "final_size.png", "outbreak_prob.png"]
# The shape of a real paper that kept 1 of its 3 figures.
PAPER = (
    "## Results\n\nThe distributions became bimodal, as shown in Figure 1.\n\n"
    "![**Figure 1.** Final size densities.](figures/final_size.png)\n"
)


def _names(hits: list[str]) -> list[str]:
    return [h.split("figures/", 1)[1].split(",", 1)[0] for h in hits]


def test_a_figure_planned_and_drawn_but_left_out_is_missing() -> None:
    hits = _missing_planned_figures(PAPER, {"design": DESIGN, "figures": FIGURES})
    assert _names(hits) == ["outbreak_prob.png", "gap_fraction.png"]
    assert all(h.startswith("figure_missing: ") for h in hits)


@pytest.mark.parametrize("state", [
    # Planned but never drawn: the writer describes it in prose instead.
    {"design": DESIGN, "figures": ["final_size.png"]},
    # Drawn but never planned: the paper may leave it out.
    {"design": {"figures_planned": ["final_size.png"]}, "figures": FIGURES},
    # No design (a survey), or a plan that is not a list.
    {"figures": FIGURES},
    {"design": {"figures_planned": "outbreak_prob.png"}, "figures": FIGURES},
])
def test_only_a_figure_both_planned_and_drawn_counts(state: dict) -> None:
    assert _missing_planned_figures(PAPER, state) == []


def test_a_paper_with_every_figure_is_clear() -> None:
    paper = PAPER + "![Figure 2.](figures/outbreak_prob.png)\n![Figure 3.](./figures/gap_fraction.png)\n"
    assert _missing_planned_figures(paper, {"design": DESIGN, "figures": FIGURES}) == []


def test_a_missing_figure_is_a_rewrite_not_a_rerun() -> None:
    hits = _missing_planned_figures(PAPER, {"design": DESIGN, "figures": FIGURES})
    assert _hits_need_only_a_rewrite(hits)
    assert _hits_need_only_a_rewrite(["unsupported_claim", *hits])


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


@pytest.mark.parametrize("panel", [False, True])
def test_review_forces_the_rewrite_even_when_the_reviewer_accepts(tmp_path: Path, panel: bool) -> None:
    eng = _engine(tmp_path)
    if panel:
        eng.config.engine.review_panel = ["methodologist", "statistician"]
    nodes: list[str] = []

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001
        nodes.append(node or "")
        if node == "review_moderator":
            return json.dumps({"rationale": "both reviewers accept"})
        return json.dumps({"verdict": "accept", "score": 4, "suggestions": [], "must_flag_hits": []})

    eng._chat = fake_chat  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text(PAPER, encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper),
        "design": DESIGN, "figures": FIGURES,
    }))
    review = patch["review"]

    assert any(n.startswith("review_panel.") for n in nodes) == panel
    assert _names(review["must_flag_hits"]) == ["outbreak_prob.png", "gap_fraction.png"]
    assert patch["iteration"] == 1, "a forced rewrite spends budget"
    assert eng._route_after_review({"review": review, "iteration": 1}) == "rewrite"  # type: ignore[arg-type]
    assert "figures/gap_fraction.png" in _format_review_for_writer({"review": review})  # type: ignore[typeddict-item]


# ---- the write node puts a left-out figure back, before the review ---------

# The design's order, which is the order the writer numbers its figures in.
PLAN = {"figures_planned": ["outbreak_prob.png", "final_size.png", "gap_fraction.png"]}
RECORDS = {
    "outbreak_prob.png": {"axes": [{"title": "Outbreak probability vs R0", "ylabel": "P", "series": [
        {"label": "N=100", "min": 0.1, "max": 0.7, "shows": "yes"}]}]},
    "final_size.png": {"axes": [{"title": "Final size densities", "ylabel": "runs", "series": []}]},
    "gap_fraction.png": {"axes": [{"title": "Gap fraction by N", "ylabel": "gap", "series": []}]},
}
# The shape of two real drafts that discussed every figure and embedded none.
DRAFT_WITHOUT_FIGURES = (
    "# T\n\n## Results\n\n"
    "**Outbreak probability.** Figure 1 shows the simulated probability of a major outbreak,\n"
    "plotted against R0 for each population size.\n\n"
    "**Final-size distributions.** Figure 2 shows the final-size histogram for each cell.\n\n"
    "Figure 3 illustrates the standard-deviation comparison across the cells shown.\n\n"
    "## References\n\n1. A cited paper.\n"
)


def _embed_line(markdown: str, name: str) -> int:
    lines = markdown.split("\n")
    return next(i for i, line in enumerate(lines) if f"](figures/{name})" in line)


def _paragraph_above(lines: list[str], at: int) -> str:
    """The prose paragraph the line at ``at`` was put after."""
    end = at - 1
    while end >= 0 and not lines[end].strip():
        end -= 1
    start = end
    while start >= 0 and lines[start].strip():
        start -= 1
    return " ".join(lines[start + 1:end + 1])


def test_a_left_out_figure_is_placed_where_the_prose_discusses_it() -> None:
    state = {"design": PLAN, "figures": FIGURES, "figure_records": RECORDS}
    repaired, placed = _place_missing_figures(DRAFT_WITHOUT_FIGURES, state)

    assert placed == ["outbreak_prob.png", "final_size.png", "gap_fraction.png"]
    # Every figure is in the paper, so the review has nothing left to force.
    assert _missing_planned_figures(repaired, state) == []
    lines = repaired.split("\n")
    # Each figure takes its place in the plan as its number, and lands in the
    # paragraph break after the prose that already names that number.
    for number, name in enumerate(placed, start=1):
        at = _embed_line(repaired, name)
        assert lines[at] == (
            f"![**Figure {number}.** {RECORDS[name]['axes'][0]['title']}.](figures/{name})"
        )
        assert f"Figure {number}" in _paragraph_above(lines, at)
    # The engine's own source list stays last.
    assert _embed_line(repaired, "gap_fraction.png") < lines.index("## References")


def test_a_figure_the_draft_kept_keeps_its_number_and_the_rest_follow() -> None:
    # PAPER embeds final_size.png as Figure 1 and names no other number.
    state = {"design": DESIGN, "figures": FIGURES, "figure_records": RECORDS}
    repaired, placed = _place_missing_figures(PAPER, state)

    assert placed == ["outbreak_prob.png", "gap_fraction.png"]
    assert repaired.count("](figures/final_size.png)") == 1, "the kept figure is not duplicated"
    assert "![**Figure 2.** Outbreak probability vs R0.](figures/outbreak_prob.png)" in repaired
    assert "![**Figure 3.** Gap fraction by N.](figures/gap_fraction.png)" in repaired
    assert _missing_planned_figures(repaired, state) == []


@pytest.mark.parametrize("state", [
    # Planned but never drawn, drawn but never planned, no design, a plan that
    # is not a list: the same four cases the review-time check leaves alone.
    {"design": DESIGN, "figures": ["final_size.png"]},
    {"design": {"figures_planned": ["final_size.png"]}, "figures": FIGURES},
    {"figures": FIGURES},
    {"design": {"figures_planned": "outbreak_prob.png"}, "figures": FIGURES},
])
def test_a_draft_with_no_figure_to_restore_is_untouched(state: dict) -> None:
    assert _place_missing_figures(PAPER, state) == (PAPER, [])


def test_the_repair_is_idempotent() -> None:
    state = {"design": PLAN, "figures": FIGURES, "figure_records": RECORDS}
    repaired, _ = _place_missing_figures(DRAFT_WITHOUT_FIGURES, state)
    assert _place_missing_figures(repaired, state) == (repaired, [])


def test_an_engine_written_caption_never_names_a_series_its_figure_hides() -> None:
    # A panel titled after a series the figure draws flat: naming it in the
    # caption would be a forced ``figure_caption`` hit, costing the very
    # iteration this repair saves. The caption falls back to the file name.
    record = {"axes": [{"title": "rk4 energy drift", "ylabel": "dE", "series": [
        {"label": "rk4", "min": -1e-8, "max": 1.2e-8, "shows": "flat"}]}]}
    state = {
        "design": {"figures_planned": ["drift_check.png"]},
        "figures": ["drift_check.png"],
        "figure_records": {"drift_check.png": record},
    }
    repaired, placed = _place_missing_figures("## Results\n\nThe drift stayed small.\n", state)

    assert placed == ["drift_check.png"]
    assert "![**Figure 1.** drift check.](figures/drift_check.png)" in repaired
    assert _figure_caption_findings(repaired, state["figure_records"]) == []


def test_a_figure_whose_record_is_missing_is_captioned_from_its_file_name() -> None:
    # A figure saved in a sandbox that could not write the record has no panel
    # title to caption it with.
    state = {"design": {"figures_planned": ["gap_fraction.png"]}, "figures": ["gap_fraction.png"]}
    repaired, placed = _place_missing_figures("## Results\n\nThe gap closed.\n", state)

    assert placed == ["gap_fraction.png"]
    assert "![**Figure 1.** gap fraction.](figures/gap_fraction.png)" in repaired


def test_a_grid_of_panels_is_captioned_by_name_and_panel_count() -> None:
    # Nine panel titles are a list of settings, not a description of the grid.
    record = {"axes": [
        {"title": f"R0={r}, N={n}", "series": []}
        for r in (0.9, 1.5, 3.0) for n in (100, 1000, 5000)
    ]}
    state = {
        "design": {"figures_planned": ["final_size_histograms_grid.png"]},
        "figures": ["final_size_histograms_grid.png"],
        "figure_records": {"final_size_histograms_grid.png": record},
    }
    repaired, _ = _place_missing_figures("## Results\n\nThe distributions split.\n", state)

    assert (
        "![**Figure 1.** final size histograms grid (9 panels).]"
        "(figures/final_size_histograms_grid.png)"
    ) in repaired


def test_the_write_node_restores_the_figures_without_spending_an_iteration(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)

    async def fake_chat(prompt, *, node=None, temperature=None):  # noqa: ANN001
        return DRAFT_WITHOUT_FIGURES

    eng._chat = fake_chat  # type: ignore[assignment]
    eng._writing_skills_block = lambda state: ""  # type: ignore[assignment]
    eng._resolve_write_persona = lambda state: ""  # type: ignore[assignment]
    patch = asyncio.run(eng._node_write({  # type: ignore[arg-type]
        "topic": "t", "title": "t", "iteration": 0,
        "design": PLAN, "figures": FIGURES, "figure_records": RECORDS,
    }))

    written = Path(patch["paper_md"]).read_text(encoding="utf-8")
    for name in FIGURES:
        assert f"](figures/{name})" in written, f"{name} reached the delivered paper"
    assert "iteration" not in patch, "restoring a figure costs no iteration"
    # And the review the draft now goes to has no missing figure to force.
    assert _missing_planned_figures(written, {"design": PLAN, "figures": FIGURES}) == []
