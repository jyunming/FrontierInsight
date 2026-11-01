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


# ---- a caption states what the saved axes and legends say, never what FI guesses from which results the body read -------

_UNREAD = {"shows": ["first", "second"], "drawn": ["first"], "not_drawn": ["second"], "labels": {"first": "First error", "second": "Second error"}}


def test_a_quantity_the_body_worked_out_from_other_reads_is_not_said_missing() -> None:
    # The body drew "Second error" from the first result and an exact formula, so no read returned it under its own id.
    record = {"axes": [{"ylabel": "First error", "xlabel": "step", "series": [{"label": "RK4", "shows": "yes"}]},
                       {"ylabel": "Second error", "xlabel": "step", "series": [{"label": "exact", "shows": "yes"}]}],
              "declared": _UNREAD}
    caption = _figure_caption_text("e.png", record, n_seeds=None)
    assert "not drawn" not in caption and "no value" not in caption
    assert caption.startswith("First error against step; Second error against step.")


def test_a_caption_names_what_the_axis_says_not_the_declared_result_the_body_read() -> None:
    # The body read the estimate and plotted its spread; the y axis says so, and the declared label of the read is not used.
    record = {"axes": [{"ylabel": "Standard deviation of the estimate", "xlabel": "noise",
                        "series": [{"label": "n=5", "shows": "yes"}]}],
              "declared": {"shows": ["estimate"], "drawn": ["estimate"], "not_drawn": [], "labels": {"estimate": "Estimate"}}}
    caption = _figure_caption_text("e.png", record, n_seeds=None)
    assert caption.startswith("Standard deviation of the estimate against noise.")
    assert "Estimate against" not in caption


def test_the_label_of_an_empty_panel_is_not_named() -> None:
    record = {"axes": [{"ylabel": "Error", "xlabel": "step", "series": [{"label": "RK4", "shows": "yes"}]},
                       {"ylabel": "Energy drift", "xlabel": "step", "series": [{"label": "RK4", "shows": "no points"}]}]}
    caption = _figure_caption_text("e.png", record, n_seeds=None)
    assert "drift" not in caption and caption.startswith("Error against step.")


def test_a_declared_result_no_read_returned_is_a_line_for_run_log_not_a_caption_claim() -> None:
    from core.engine import _unread_results

    lines = _unread_results({"e.png": {"declared": _UNREAD}, "f.png": {"declared": dict(_UNREAD, not_drawn=[])}, "g.png": {}})
    assert lines == ["e.png: no read of the figure's body returned a value of Second error"]


def test_a_series_with_no_data_point_is_not_named_in_a_caption() -> None:
    record = {"axes": [{"ylabel": "Error", "xlabel": "t",
                        "series": [{"label": "A", "shows": "yes"}, {"label": "Ghost", "shows": "no points"}]}]}
    assert "Ghost" not in _figure_caption_text("e.png", record, n_seeds=None)


def test_a_colorbar_or_hidden_axes_is_not_a_panel_of_the_caption() -> None:
    from core import figure_facts

    record = {"axes": [{"ylabel": "y", "xlabel": "x", "series": [], "place": [0.1, 0.1, 0.6, 0.8]},
                       {"ylabel": "Density", "series": [], "colorbar": True, "place": [0.8, 0.1, 0.05, 0.8]}]}
    caption = _figure_caption_text("e.png", record, n_seeds=None, facts={"e:layout": figure_facts.layout_sentence(record)})
    assert "Density" not in caption and "Two panels" not in caption
    assert figure_facts.layout(record)["panels"] == 1


def test_the_layout_and_the_description_leave_out_the_same_panels() -> None:
    from core import figure_facts

    full = {"ylabel": "Error", "series": [{"label": "A", "shows": "yes"}]}
    empty = {"ylabel": "Drift", "series": [{"label": "B", "shows": "no points"}]}
    record = {"axes": [dict(full, title="one", place=[0.05, 0.1, 0.25, 0.8]), dict(full, title="two", place=[0.37, 0.1, 0.25, 0.8]),
                       dict(empty, title="Drift", place=[0.7, 0.1, 0.25, 0.8])]}
    sentence = figure_facts.layout_sentence(record)
    assert "Drift" not in sentence and "two panels" in sentence.lower()
    assert "Drift" not in _figure_caption_text("e.png", record, n_seeds=None, facts={"e:layout": sentence})


def test_an_unread_results_record_with_a_non_list_is_read_as_empty() -> None:
    from core.engine import _unread_results

    assert _unread_results({"e.png": {"declared": dict(_UNREAD, not_drawn=5)}}) == []


# ---- the writer chooses among statements FI has checked -----------------------------------------------------------------

import json  # noqa: E402

from core import figure_claims as _claims  # noqa: E402

LEGACY_LIST = ('- Figure 3: figures/spread_map.png\n- Figure 2: figures/cost_by_size.png\n'
               '- Figure 1: figures/error_vs_step.png \u2014 series: scheme A 0.001 to 0.2')


def _series(ident: str, label: str, ys=(3.0, 2.0, 1.0)) -> dict:
    return {"id": ident, "label": label, "kind": "line", "step": False, "x": [1.0, 2.0, 3.0], "y": list(ys),
            "yerr": None, "merged_with": []}


def _panel(k: int = 1, *, raw: bool = False, series=None) -> dict:
    return {"panel": k, "raw": raw, "raw_reason": "fig.raw" if raw else None,
            "x": {"id": "size", "kind": "setting", "label": "size", "scale": "linear"},
            "y": {"what": "error", "label": "error", "scale": "linear"},
            "series": [] if raw else (series if series is not None else [_series("error", "scheme A")]),
            "bands": [], "refs": [], "notes": [], "texts_raw": ["slope 2.0"] if raw else []}


def _quest(tmp_path: Path, calls: dict | None, stem: str = "error_vs_step") -> dict:
    folder = tmp_path / ".fi" / "figure_records"
    folder.mkdir(parents=True)
    if calls is not None:
        (folder / f"{stem}.calls.json").write_text(json.dumps(calls), encoding="utf-8")
    return {**STATE, "quest_root": str(tmp_path)}


def _record(*panels: dict, fallback: bool = False) -> dict:
    return {"version": 1, "fallback": fallback, "panels": list(panels)}


def test_prompt_lists_true_claims_as_marks_for_verified_figure(tmp_path: Path) -> None:
    state = _quest(tmp_path, _record(_panel()))
    listing = _figure_list_for_prompt(state)  # type: ignore[arg-type]
    assert "{{figure:error_vs_step:drawn @ series=error, panel=1}}" in listing
    assert "scheme A is drawn" in listing, "the writer sees the sentence FI will write"
    assert "{{figure:error_vs_step:trend @ series=error, panel=1}}" in listing
    # a claim that is not true of the drawing is not offered
    assert "{{figure:error_vs_step:crossing" not in listing and "increases" not in listing
    # a figure with no call record keeps its old line
    assert "- Figure 2: figures/cost_by_size.png\n" in listing + "\n"


def test_prompt_says_raw_panel_has_none(tmp_path: Path) -> None:
    state = _quest(tmp_path, _record(_panel(1), _panel(2, raw=True)))
    listing = _figure_list_for_prompt(state)  # type: ignore[arg-type]
    assert "no statements can be made about panel 2 (custom drawing)" in listing
    assert "panel=2" not in listing


def test_prompt_says_a_fallback_figure_is_drawn_from_the_plans_description(tmp_path: Path) -> None:
    state = _quest(tmp_path, _record(_panel(), fallback=True))
    assert "drawn by FI from the plan's description" in _figure_list_for_prompt(state)  # type: ignore[arg-type]


def test_legacy_figure_prompt_is_unchanged(tmp_path: Path) -> None:
    # RF1: golden text of the list before the menu existed (no call record), with or without a quest folder.
    assert _figure_list_for_prompt(STATE) == LEGACY_LIST  # type: ignore[arg-type]
    state = _quest(tmp_path, None)
    assert _figure_list_for_prompt(state) == LEGACY_LIST  # type: ignore[arg-type]


def test_no_selected_claim_gets_one_composed_sentence_inline(tmp_path: Path) -> None:
    state = _quest(tmp_path, _record(_panel()))
    out, placed, _ = _place_figures(DRAFT, state)  # type: ignore[arg-type]
    assert "error_vs_step.png" in placed
    sentence = "{{figure:error_vs_step:drawn @ series=error, panel=1}} (Figure 1)."
    assert out.count("{{figure:error_vs_step") == 1 and sentence in out
    lines = out.split("\n")
    # at the figure's first reference: the paragraph that names it, before the figure
    assert sentence in _paragraph_above(lines, _embed_line(out, "error_vs_step.png"))
    # the figures with no call record get nothing
    assert "{{figure:cost_by_size" not in out and "{{figure:spread_map" not in out
    # placing again adds nothing
    again, _, _ = _place_figures(out, state)  # type: ignore[arg-type]
    assert again == out


def test_selected_claims_are_not_duplicated_by_minimum_sentence(tmp_path: Path) -> None:
    state = _quest(tmp_path, _record(_panel()))
    chosen = DRAFT.replace("for each scheme.", "and {{figure:error_vs_step:trend @ series=error, panel=1}}.")
    out, _, _ = _place_figures(chosen, state)  # type: ignore[arg-type]
    assert out.count("{{figure:error_vs_step") == 1 and ":drawn" not in out
    rendered = chosen.replace("{{figure:error_vs_step:trend @ series=error, panel=1}}",
                              "<!-- fi:fact-figure-trend-error_vs_step-args00 -->x<!-- /fi:fact-figure-trend-error_vs_step-args00 -->")
    out, _, _ = _place_figures(rendered, state)  # type: ignore[arg-type]
    assert "{{figure:" not in out


def test_a_raw_only_or_fallback_figure_gets_no_minimum_sentence(tmp_path: Path) -> None:
    for calls in (_record(_panel(1, raw=True)), _record(_panel(), fallback=True)):
        state = _quest(tmp_path / str(len(calls["panels"])) / str(calls["fallback"]), calls)
        out, _, _ = _place_figures(DRAFT, state)  # type: ignore[arg-type]
        assert "{{figure:" not in out


def test_plan_of_reads_metrics_and_a_single_numeric_grid() -> None:
    plan = _claims.plan_of({"metrics": [{"id": "error"}, {"id": "cost"}], "grid": {"size": [1, 2, 4]}})
    assert plan == {"series": [{"id": "error", "label": ""}, {"id": "cost", "label": ""}],
                    "grid": [1.0, 2.0, 4.0], "grid_id": "size"}
    assert "grid" not in _claims.plan_of({"metrics": [{"id": "e"}], "grid": {"a": [1], "b": [2]}})
    assert _claims.plan_of(None) is None
    calls = _record(_panel(series=[{**_series("error", "scheme A"), "kind": "points"}]))
    gap = {"series": [], "grid": [1.0, 2.0, 3.0, 4.0], "grid_id": "size"}
    assert "has no point at size 4" in _claims.evaluate(calls, "missing_x", {"series": "error", "panel": "1"}, plan=gap)[0]
    other = {**gap, "grid_id": "step"}
    assert _claims.evaluate(calls, "missing_x", {"series": "error", "panel": "1"}, plan=other) is None


def test_pointer_rule_runs_in_write_post_processing(tmp_path: Path) -> None:
    from tests.test_paper_line_endings import _quest as quest_with_engine

    text = ("# T\n\n## Results\n\nFigure 1 shows the paired gain. Figure 1 is a placeholder.\n\n"
            "Figure 2 shows the time each size took.\n\n## Limitations\n\nFew.\n")
    eng, paper = quest_with_engine(tmp_path, text)
    state = _quest(eng.quest_root, _record(_panel()))
    state = {**state, "paper_md": str(paper), "paper_keys": True, "iteration": 1}
    eng.config.engine.max_iterations = 2
    hits = eng._paper_record_hits(paper.read_text(encoding="utf-8"), state)  # type: ignore[arg-type]
    after = paper.read_bytes().decode("utf-8")
    # Figure 1 has a call record: a sentence that points at it and is not FI's is gone; Figure 2 is legacy: untouched
    assert "paired gain" not in after and "placeholder" not in after
    assert "Figure 2 shows the time each size took." in after
    from core import paper_facts
    kept = paper_facts.kept_check(eng.quest_root)
    left_out = [d for d in kept["taken_out_values"] if str(d.get("key", "")).startswith("figure:")]
    assert {d["key"].split(":")[1] for d in left_out} == {"error_vs_step"} and {d["figure"] for d in left_out} == {"1"}
    # one disclosure for the figure, in the Limitations of the same write
    assert after.count("some statements about this figure could not be checked") == 1
    assert not hits or all("figure" not in h.lower() for h in hits)


def test_pointer_rule_leaves_a_paper_of_legacy_figures_alone(tmp_path: Path) -> None:
    from tests.test_paper_line_endings import _quest as quest_with_engine

    text = "# T\n\n## Results\n\nFigure 1 shows the paired gain. Figure 9 is odd.\n\n## Limitations\n\nFew.\n"
    eng, paper = quest_with_engine(tmp_path, text)
    state = {**STATE, "quest_root": str(eng.quest_root), "paper_md": str(paper), "paper_keys": True, "iteration": 1}
    eng.config.engine.max_iterations = 2
    before = paper.read_bytes()
    eng._paper_record_hits(paper.read_text(encoding="utf-8"), state)  # type: ignore[arg-type]
    assert paper.read_bytes() == before


# ---- the pointer rule and the take-out of figure statements run on every write ---------------------------------------------

FREE = "Figure 1 shows that the error falls steeply, a drop of about three orders of magnitude over the whole range."
WRITER_DRAFT = (
    "# T\n\n## Abstract\nWe compare schemes.\n\n## Results\n\n"
    f"**Accuracy.** {FREE}\n\n"
    "**Cost.** Figure 2 shows the time each size took.\n\n"
    "## Discussion\nIt is small.\n"
)
DISCLOSURE = "some statements about this figure could not be checked"


def _write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drafts: list[str], calls: dict | None, *,
           iteration: int = 0, max_iterations: int = 3, state_extra: dict | None = None):
    """A write node run on a quest with a call record for Figure 1 (error_vs_step) only; ``drafts`` are the writer's
    replies in turn. Returns ``(engine, state, [state after each write])``."""
    import json as _json
    from tests.test_disclosure import _engine as make_engine

    replies = list(drafts)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return replies.pop(0) if len(replies) > 1 else replies[0]

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = make_engine(tmp_path)
    eng.config.engine.max_iterations = max_iterations
    import logging

    eng.said = []  # type: ignore[attr-defined]  # what the engine put in its run log

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            eng.said.append(record.getMessage())  # type: ignore[attr-defined]

    eng._log.addHandler(_Keep())
    state = {**_quest(tmp_path, calls), "topic": "t", "title": "t", "iteration": iteration, **(state_extra or {})}
    outs = []
    for _ in drafts:
        out = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
        state = {**state, **out}
        outs.append(dict(state))
    return eng, state, outs


def _paper(state: dict) -> str:
    return Path(state["paper_md"]).read_text(encoding="utf-8")


def test_an_unkept_figure_sentence_is_gone_after_the_first_write_with_rounds_left(tmp_path, monkeypatch) -> None:
    eng, state, _ = _write(tmp_path, monkeypatch, [WRITER_DRAFT], _record(_panel()))
    paper = _paper(state)
    assert "falls steeply" not in paper and "three orders" not in paper
    assert "scheme A is drawn" in paper and "(Figure 1)." in paper, "FI's own sentence is there"
    assert "Figure 2 shows the time each size took." in paper, "a figure with no call record is left as it was"
    assert paper.count(DISCLOSURE) == 1, "the Limitations say so in the same write"
    # one line per figure: the count and the first 80 characters of each removed sentence
    lines = [ln for ln in eng.said if "statements about Figure 1" in ln]
    assert len(lines) == 1 and "left out (1)" in lines[0]
    assert '"' + ("**Accuracy.** " + FREE)[:80] + '"' in lines[0], "the first 80 characters of the sentence, as it read"
    assert ("**Accuracy.** " + FREE)[:81] not in lines[0]


def test_the_disclosure_survives_the_review_that_follows_and_a_clean_rewrite_clears_it(tmp_path, monkeypatch) -> None:
    from core import paper_facts

    eng, state, _ = _write(tmp_path, monkeypatch, [WRITER_DRAFT], _record(_panel()), max_iterations=3)
    eng._paper_record_hits(_paper(state), state)  # type: ignore[arg-type]  # the review's record check, rounds left
    assert _paper(state).count(DISCLOSURE) == 1
    assert any(str(d.get("key")).startswith("figure:") for d in paper_facts.kept_check(tmp_path)["taken_out_values"])
    eng._restate_limitations(state, "again")  # type: ignore[arg-type]
    assert _paper(state).count(DISCLOSURE) == 1, "a later restatement of the Limitations still says it"
    # the paper written again with nothing to take out says nothing older
    clean = WRITER_DRAFT.replace(FREE, "The error is compared across the schemes.")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return clean

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    out = asyncio.run(eng._node_write({**state, "iteration": 1}))  # type: ignore[arg-type]
    assert DISCLOSURE not in Path(out["paper_md"]).read_text(encoding="utf-8")
    assert not any(str(d.get("key")).startswith("figure:")
                   for d in (paper_facts.kept_check(tmp_path).get("taken_out_values") or []))


def test_a_rewrite_after_a_first_write_is_taken_out_too(tmp_path, monkeypatch) -> None:
    again = WRITER_DRAFT.replace("**Cost.**", "**Cost.** Figure 1 also shows a clear bend.")
    eng, state, outs = _write(tmp_path, monkeypatch, [WRITER_DRAFT, again], _record(_panel()), max_iterations=5)
    assert "bend" not in _paper(outs[1]) and _paper(outs[1]).count(DISCLOSURE) == 1
    # the last round, or not, paper_keys or not: the record check finds nothing left to take out
    assert eng._paper_record_hits(_paper(outs[1]), outs[1]) is not None  # type: ignore[arg-type]
    assert "bend" not in _paper(outs[1])


def test_fis_sentence_stays_when_the_writer_puts_other_words_in_it(tmp_path, monkeypatch) -> None:
    mixed = WRITER_DRAFT.replace(
        FREE, "Beyond doubt, {{figure:error_vs_step:trend @ series=error, panel=1}} and so the scheme wins (Figure 1).")
    _eng, state, _ = _write(tmp_path, monkeypatch, [mixed], _record(_panel()))
    paper = _paper(state)
    assert "scheme wins" not in paper and "Beyond doubt" not in paper
    assert "(Figure 1)." in paper and "scheme A" in paper, "FI's sentence is kept, with its pointer"


def test_a_figure_with_no_call_record_keeps_its_free_sentence_through_every_write(tmp_path, monkeypatch) -> None:
    legacy = "# T\n\n## Results\n\nFigure 1 shows the paired gain. Figure 2 and Figure 9 differ.\n\n## Discussion\nOk.\n"
    _eng, state, _ = _write(tmp_path, monkeypatch, [legacy], None)
    paper = _paper(state)
    assert "Figure 1 shows the paired gain. Figure 2 and Figure 9 differ." in paper and DISCLOSURE not in paper


def test_a_sentence_naming_only_legacy_figures_stays_when_another_figure_has_a_record(tmp_path, monkeypatch) -> None:
    draft = ("# T\n\n## Results\n\nThe error is compared. Figure 2 shows the time each size took, "
             "and Figure 9 is odd.\n\n## Discussion\nOk.\n")
    _eng, state, _ = _write(tmp_path, monkeypatch, [draft], _record(_panel()))
    assert "Figure 2 shows the time each size took, and Figure 9 is odd." in _paper(state)
    assert DISCLOSURE not in _paper(state)
