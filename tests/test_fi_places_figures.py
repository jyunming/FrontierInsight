"""The figures of a paper are FI's (``core/engine.py``: ``_figure_numbers``, ``_place_figures``, ``_figure_caption_text``,
called by the write node): every figure the run drew goes in under the number the writer was given, with a caption FI
writes from its own record of what the figure draws, and every image the draft wrote itself is taken out."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core import paper_scope
from core.engine import (
    Engine,
    _figure_caption_text,
    _figure_list_for_prompt,
    _figure_numbers,
    _place_figures,
    _without_figure_embeds,
)

PLAN = {"figures_planned": ["error_vs_step.png", "cost_by_size.png", "spread_map.png"]}
FIGURES = ["spread_map.png", "cost_by_size.png", "error_vs_step.png"]
RECORDS = {
    "error_vs_step.png": {"axes": [{"ylabel": "error", "series": [
        {"label": "scheme A", "min": 0.001, "max": 0.2, "shows": "yes"}]}]},
    "cost_by_size.png": {"axes": [{"ylabel": "seconds", "series": []}]},
    "spread_map.png": {"axes": [{"ylabel": "Spread over the grid", "series": []}]},
}
STATE = {"design": PLAN, "figures": FIGURES, "figure_records": RECORDS}
# A draft that discusses every figure by number and embeds none.
DRAFT = (
    "# T\n\n## Results\n\n"
    "**Accuracy.** Figure 1 shows the error falling as the step shrinks,\n"
    "for each scheme.\n\n"
    "**Cost.** Figure 2 shows the time each size took.\n\n"
    "Figure 3 maps the spread across the grid.\n\n"
    "## References\n\n1. A cited paper.\n"
)


def _embed_line(markdown: str, name: str) -> int:
    lines = markdown.split("\n")
    return next(i for i, line in enumerate(lines) if f"](figures/{name})" in line)


def _paragraph_above(lines: list[str], at: int) -> str:
    end = at - 1
    while end >= 0 and not lines[end].strip():
        end -= 1
    start = end
    while start >= 0 and lines[start].strip():
        start -= 1
    return " ".join(lines[start + 1:end + 1])


# ---- the numbers, fixed before the paper is written ---------------------------------------------------------------------

def test_the_planned_figures_come_first_in_the_planned_order_then_the_others_as_drawn() -> None:
    state = {**STATE, "design": {"figures_planned": ["cost_by_size.png", "never_drawn.png"]}}
    assert _figure_numbers(state) == {"cost_by_size.png": 1, "spread_map.png": 2, "error_vs_step.png": 3}


def test_a_figure_whose_record_says_it_draws_nothing_has_no_number() -> None:
    records = {**RECORDS, "cost_by_size.png": {"axes": [{"title": "Cost", "series": [{"label": "a", "shows": "no points"}]}]}}
    numbers = _figure_numbers({**STATE, "figure_records": records})
    assert "cost_by_size.png" not in numbers
    assert numbers == {"error_vs_step.png": 1, "spread_map.png": 2}


def test_the_writer_is_given_the_numbers() -> None:
    listing = _figure_list_for_prompt(STATE)  # type: ignore[arg-type]
    assert "- Figure 1: figures/error_vs_step.png" in listing
    assert "- Figure 2: figures/cost_by_size.png" in listing
    assert "- Figure 3: figures/spread_map.png" in listing


# ---- placing ------------------------------------------------------------------------------------------------------------

def test_each_figure_goes_after_the_paragraph_that_first_names_its_number() -> None:
    out, placed, written = _place_figures(DRAFT, STATE)  # type: ignore[arg-type]

    assert placed == ["error_vs_step.png", "cost_by_size.png", "spread_map.png"] and written == 0
    lines = out.split("\n")
    for number, name in enumerate(placed, start=1):
        at = _embed_line(out, name)
        assert lines[at].startswith(f"![**Figure {number}.** {RECORDS[name]['axes'][0]['ylabel']}.")
        assert f"Figure {number}" in _paragraph_above(lines, at)
    assert _embed_line(out, "spread_map.png") < lines.index("## References")


def test_the_drafts_own_images_and_captions_are_replaced_by_fis() -> None:
    draft = DRAFT.replace(
        "for each scheme.\n",
        "for each scheme.\n\n![**Figure 1.** The error is flat for every scheme.](figures/error_vs_step.png){width=80%}\n")
    out, placed, written = _place_figures(draft, STATE)  # type: ignore[arg-type]

    assert written == 1
    assert "flat for every scheme" not in out, "no caption in the paper is the writer's"
    assert out.count("](figures/error_vs_step.png)") == 1
    assert len(placed) == 3


def test_a_figure_the_prose_names_out_of_order_still_follows_the_one_before_it() -> None:
    draft = "## Results\n\nFigure 2 shows the cost.\n\nFigure 1 shows the error.\n\n## References\n\n1. A.\n"
    out, _, _ = _place_figures(draft, STATE)  # type: ignore[arg-type]
    assert _embed_line(out, "error_vs_step.png") < _embed_line(out, "cost_by_size.png") < _embed_line(out, "spread_map.png")


def test_a_figure_the_prose_never_names_goes_before_the_source_lists() -> None:
    out, placed, _ = _place_figures("## Results\n\nText.\n\n## References\n\n1. A.\n", STATE)  # type: ignore[arg-type]
    lines = out.split("\n")
    assert len(placed) == 3
    assert all(_embed_line(out, name) < lines.index("## References") for name in placed)


def test_an_unplanned_figure_the_run_drew_is_placed_too() -> None:
    state = {**STATE, "design": {"figures_planned": ["error_vs_step.png"]}}
    _, placed, _ = _place_figures(DRAFT, state)  # type: ignore[arg-type]
    assert sorted(placed) == sorted(FIGURES)


def test_placing_the_figures_again_gives_the_same_paper() -> None:
    once, _, _ = _place_figures(DRAFT, STATE)  # type: ignore[arg-type]
    twice, placed, written = _place_figures(once, STATE)  # type: ignore[arg-type]
    assert twice == once and written == 3 and len(placed) == 3


def test_taking_the_figures_out_gives_back_the_text_they_were_placed_in() -> None:
    once, _, _ = _place_figures(DRAFT, STATE)  # type: ignore[arg-type]
    assert _without_figure_embeds(once) == (DRAFT, 3)


def test_an_image_inside_a_code_fence_is_left_as_it_is() -> None:
    fenced = "```\n![x](figures/error_vs_step.png)\n```\n"
    assert _without_figure_embeds(fenced) == (fenced, 0)


def test_with_no_figure_nothing_is_placed() -> None:
    assert _place_figures(DRAFT, {"figures": []}) == (DRAFT, [], 0)  # type: ignore[arg-type]


# ---- the caption, from the records only --------------------------------------------------------------------------------

def test_a_caption_is_the_panel_title_with_the_series_and_what_the_record_says_of_them() -> None:
    record = {"axes": [{"ylabel": "Energy drift", "series": [
        {"label": "scheme A", "shows": "yes"}, {"label": "scheme B", "shows": "flat"},
        {"label": "limit", "y": 0.5, "shows": "reference line"}]}]}
    caption = _figure_caption_text("drift.png", record, n_seeds=None)
    assert caption == "Energy drift. It draws scheme A, scheme B. scheme B is flat at this scale. Marked: limit at 0.5."


def test_a_panel_with_no_title_is_described_by_its_axes() -> None:
    record = {"axes": [{"ylabel": "error", "xlabel": "step size", "series": []}]}
    assert _figure_caption_text("e.png", record, n_seeds=None) == "error against step size."


def test_a_figure_with_no_record_is_described_by_its_file_name_and_never_left_empty() -> None:
    assert _figure_caption_text("cost_by_size.png", None, n_seeds=None) == "cost by size."


def test_a_grid_of_panels_is_described_by_name_and_panel_count() -> None:
    record = {"axes": [{"title": f"a={a}, b={b}", "series": []} for a in (1, 2, 3) for b in (4, 5, 6)]}
    assert _figure_caption_text("spread_grid.png", record, n_seeds=None).startswith("spread grid (9 panels).")


def test_a_panel_that_draws_nothing_is_left_out_of_the_caption() -> None:
    record = {"axes": [{"ylabel": "Rate", "series": [{"label": "a", "shows": "yes"}]},
                       {"ylabel": "Odds", "series": [{"label": "b", "shows": "no points"}]}]}
    caption = _figure_caption_text("r.png", record, n_seeds=None)
    assert "Rate" in caption and "Odds" not in caption


def test_the_layout_and_line_styles_come_from_fis_figure_facts() -> None:
    record = {"axes": [{"title": "Left", "series": []}, {"title": "Right", "series": []}]}
    facts = {"two:layout": "two panels side by side", "two:lines": "the dashed line is the reference"}
    caption = _figure_caption_text("two.png", record, n_seeds=None, facts=facts)
    assert caption.endswith("Two panels side by side. The dashed line is the reference.")


def test_a_mean_over_seeds_is_said_in_the_caption() -> None:
    record = {"axes": [{"title": "Rate", "series": []}], "replicate_mean": {"n": 5}}
    assert "mean of 5 seeds" in _figure_caption_text("r.png", record, n_seeds=5)


def test_a_sourced_figure_says_where_it_is_from_and_that_it_is_not_the_studys_data() -> None:
    credit = {"file": "diagram.png", "caption": "A schematic of the apparatus", "source_url": "https://example.org/x",
              "license": "CC BY 4.0"}
    caption = _figure_caption_text("diagram.png", None, n_seeds=None, credit=credit)
    assert caption == ("A schematic of the apparatus. Source: https://example.org/x, CC BY 4.0; an illustration from "
                       "that source, not this study's data.")


# ---- the write node -----------------------------------------------------------------------------------------------------

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


@pytest.mark.parametrize("draft", [
    DRAFT,
    # A draft that wrote its own image with a caption the record contradicts.
    DRAFT.replace("Figure 3 maps", "![**Figure 1.** Nothing changes.](figures/error_vs_step.png)\n\nFigure 3 maps"),
])
def test_the_write_node_delivers_every_figure_with_fis_captions_without_spending_an_iteration(
    tmp_path: Path, draft: str,
) -> None:
    eng = _engine(tmp_path)
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)

    async def fake_chat(prompt, *, node=None, temperature=None):  # noqa: ANN001
        return draft

    eng._chat = fake_chat  # type: ignore[assignment]
    eng._writing_skills_block = lambda state: ""  # type: ignore[assignment]
    eng._resolve_write_persona = lambda state: ""  # type: ignore[assignment]
    patch = asyncio.run(eng._node_write({  # type: ignore[arg-type]
        "topic": "t", "title": "t", "iteration": 0, **STATE,
    }))

    written = Path(patch["paper_md"]).read_text(encoding="utf-8")
    for number, name in enumerate(["error_vs_step.png", "cost_by_size.png", "spread_map.png"], start=1):
        assert written.count(f"](figures/{name})") == 1, name
        assert f"![**Figure {number}.** {RECORDS[name]['axes'][0]['ylabel']}." in written
    assert "Nothing changes" not in written
    assert "iteration" not in patch, "placing the figures costs no iteration"


# ---- after the first review -----------------------------------------------------------------------------------------

def test_a_series_flat_in_one_of_two_untitled_panels_is_said_to_be_flat_there_only() -> None:
    record = {"axes": [{"series": [{"label": "A", "shows": "flat"}]}, {"series": [{"label": "A", "shows": "yes"}]}]}
    caption = _figure_caption_text("a.png", record, n_seeds=None)
    assert "A is flat at this scale in panel 1." in caption
    assert "A is flat at this scale." not in caption


def test_a_series_outside_the_range_in_every_panel_is_not_listed_as_drawn() -> None:
    record = {"axes": [{"title": "Rate", "series": [{"label": "A", "shows": "yes"}, {"label": "B", "shows": "outside the axis"}]}]}
    caption = _figure_caption_text("a.png", record, n_seeds=None)
    assert "It draws A." in caption and "B is not visible: it lies outside the plotted range." in caption


def test_one_drawing_saved_in_two_formats_is_one_figure_shown_as_its_png() -> None:
    state = {"design": {"figures_planned": ["fit.pdf"]}, "figures": ["fit.png", "fit.pdf", "other.png"]}
    assert _figure_numbers(state) == {"fit.png": 1, "other.png": 2}  # type: ignore[arg-type]
    out, placed, _ = _place_figures("## Results\n\nFigure 1 shows the fit.\n", state)  # type: ignore[arg-type]
    assert placed == ["fit.png", "other.png"] and "fit.pdf" not in out
    assert "the same drawing as Figure 1 (figures/fit.png)" in _figure_list_for_prompt(state)  # type: ignore[arg-type]


def test_a_caption_holds_no_square_bracket() -> None:
    record = {"axes": [{"ylabel": "r = 0.99 [0.98, 1.00]", "series": []}, {"ylabel": "rate in [2%, 30%)", "series": []}]}
    caption = _figure_caption_text("a.png", record, n_seeds=None)
    assert "[" not in caption and "]" not in caption and "(0.98, 1.00)" in caption


def test_a_figure_named_only_in_the_source_lists_goes_before_them() -> None:
    draft = "## Results\n\nText.\n\n## References\n\n1. A. Author, Fig. 1 of the original.\n"
    out, _, _ = _place_figures(draft, {"figures": ["a.png"]})  # type: ignore[arg-type]
    lines = out.split("\n")
    assert _embed_line(out, "a.png") < lines.index("## References")


def test_a_paper_without_a_final_line_break_is_placed_the_same_way_twice() -> None:
    once, _, _ = _place_figures("## Results\n\nFigure 1 shows it.", {"figures": ["a.png"]})  # type: ignore[arg-type]
    assert _place_figures(once, {"figures": ["a.png"]})[0] == once  # type: ignore[arg-type]


def test_a_drafts_dot_slash_image_is_taken_out_too() -> None:
    out, _, written = _place_figures("## R\n\n![mine](./figures/a.png)\n\nFigure 1.\n", {"figures": ["a.png"]})  # type: ignore[arg-type]
    assert written == 1 and out.count("figures/a.png") == 1 and "mine" not in out


def test_fis_figures_are_not_units_a_checker_can_name() -> None:
    marked, table = paper_scope.mark("## Results\nThe rate fell.\n\n![**Figure 1.** It draws A.](figures/a.png)\n")
    passages = [p.passage for p in table.values() if p.passage]
    assert len(passages) == 1 and "The rate fell" in passages[0], passages
    assert "<<" not in marked.split("![", 1)[1]


def test_an_untitled_panel_is_named_by_its_place_in_the_saved_figure_when_an_empty_panel_is_left_out() -> None:
    record = {"axes": [{"series": [{"label": "B", "shows": "no points"}]}, {"series": [{"label": "A", "shows": "flat"}]}, {"series": [{"label": "A", "shows": "yes"}]}]}
    caption = _figure_caption_text("a.png", record, n_seeds=None)
    assert "A is flat at this scale in panel 2." in caption, caption


# ---- a caption never copies the script's own title -----------------------------------------------------------------

def test_a_caption_does_not_copy_a_title_that_names_a_line_the_figure_does_not_draw() -> None:
    record = {"axes": [{"title": "Drift (dashed: the model)", "ylabel": "Drift", "xlabel": "noise",
                        "series": [{"label": "n = 6", "shows": "yes"}]}]}
    facts = {"s:lines": "solid lines and markers are drawn; no dashed or dotted line is drawn"}
    caption = _figure_caption_text("s.png", record, n_seeds=None, facts=facts)
    assert "the model" not in caption
    assert caption == ("Drift against noise. It draws n = 6. Solid lines and markers are drawn; "
                       "no dashed or dotted line is drawn.")


def test_a_figure_title_the_paper_copy_drops_is_not_copied_into_the_caption() -> None:
    record = {"axes": [{"ylabel": "Rate", "series": []}], "paper_dropped": ["Rate, dotted: the model"]}
    assert _figure_caption_text("r.png", record, n_seeds=None) == "Rate."
