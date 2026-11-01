"""The writer's "Figure N." comes off a caption before a renderer that numbers
figures prints it (``generation/_figure_captions.py``)."""

from __future__ import annotations

import pytest

from generation._figure_captions import blank_lines_around_figures, numbers_off_figure_captions, without_number


@pytest.mark.parametrize("caption", [
    "**Figure 1.** Log-log plot of trajectory error versus time step $h$.",
    "**Figure 1:** Log-log plot of trajectory error versus time step $h$.",
    "*Fig. 1.* Log-log plot of trajectory error versus time step $h$.",
    "**Figure 1** — Log-log plot of trajectory error versus time step $h$.",
    "Figure 1. Log-log plot of trajectory error versus time step $h$.",
    "Fig 1: Log-log plot of trajectory error versus time step $h$.",
])
def test_the_writers_number_comes_off_the_caption(caption: str) -> None:
    assert without_number(caption) == "Log-log plot of trajectory error versus time step $h$."


@pytest.mark.parametrize("caption", [
    "Figure 2 shows the drift of each integrator.",   # a sentence, not a number
    "Energy decay against the analytical reference.",
    "**Figure 3.**",                                   # nothing else to print
])
def test_a_caption_without_a_leading_number_stays_as_written(caption: str) -> None:
    assert without_number(caption) == caption


def test_only_figure_lines_outside_code_change() -> None:
    md = "\n".join([
        "## Results",
        "",
        "![**Figure 1.** Error versus step, as in [3].](figures/error.png)",
        "",
        "![**Figure 2.** Energy decay.](figures/energy.png){width=80%}",
        "",
        "Figure 1 shows the error; an inline ![**Figure 9.** icon](figures/i.png) stays.",
        "",
        "```markdown",
        "![**Figure 4.** An example in a code block.](figures/x.png)",
        "```",
    ])
    assert numbers_off_figure_captions(md).split("\n") == [
        "## Results",
        "",
        "![Error versus step, as in [3].](figures/error.png)",
        "",
        "![Energy decay.](figures/energy.png){width=80%}",
        "",
        "Figure 1 shows the error; an inline ![**Figure 9.** icon](figures/i.png) stays.",
        "",
        "```markdown",
        "![**Figure 4.** An example in a code block.](figures/x.png)",
        "```",
    ]


# --- a figure is a paragraph of its own ----------------------------------------------------

def test_figures_written_one_under_the_other_are_paragraphs_of_their_own() -> None:
    """Three in a row, as a stored quest wrote them, printed with no caption at all: pandoc read them
    as one paragraph of three inline images."""
    md = (
        "Text.\n![**Figure 1.** One.](figures/a.png)\n![**Figure 2.** Two.](figures/b.png)\n"
        "![Three.](figures/c.png)\nMore text.\n"
    )
    assert blank_lines_around_figures(md) == (
        "Text.\n\n![**Figure 1.** One.](figures/a.png)\n\n![**Figure 2.** Two.](figures/b.png)\n\n"
        "![Three.](figures/c.png)\n\nMore text.\n"
    )


def test_a_figure_that_is_already_a_paragraph_and_a_code_block_are_left_alone() -> None:
    spaced = "Text.\n\n![A.](figures/a.png)\n\nMore.\n"
    assert blank_lines_around_figures(spaced) == spaced
    once = blank_lines_around_figures("a\n![A.](x.png)\nb")
    assert once == "a\n\n![A.](x.png)\n\nb" and blank_lines_around_figures(once) == once
    fenced = "```\n![A.](x.png)\n![B.](y.png)\n```\n"
    assert blank_lines_around_figures(fenced) == fenced
    # An image inside a sentence is not a figure line.
    inline = "See ![icon](i.png) here.\nNext line.\n"
    assert blank_lines_around_figures(inline) == inline
    assert blank_lines_around_figures("") == "" and blank_lines_around_figures("![A.](x.png)") == "![A.](x.png)"


def test_figures_out_of_number_order_have_their_references_renumbered_to_where_they_stand() -> None:
    """A paper placed Figure 1, 3, 2; the renderer numbers by place, so the PDF printed the text's "Figure 3" over the
    plot captioned Figure 2. Every reference is renumbered to its figure's place; code is left alone."""
    from generation._figure_captions import numbers_off_figure_captions

    md = ("See Figure 3 and Figures 2 and 3.\n\n![**Figure 1.** A](a.png)\n\n![**Figure 3.** C, beside Figure 2](c.png)"
          "\n\n![**Figure 2.** B](b.png)\n\n```\nFigure 3 in code\n```")
    out = numbers_off_figure_captions(md)
    assert out.startswith("See Figure 2 and Figures 3 and 2.")
    assert "![C, beside Figure 3](c.png)" in out and "Figure 3 in code" in out
    in_order = "See Figure 2.\n\n![**Figure 1.** A](a.png)\n\n![**Figure 2.** B](b.png)"
    assert numbers_off_figure_captions(in_order).startswith("See Figure 2."), "figures in order: references untouched"


def test_renumbering_leaves_file_names_alone_and_keeps_a_range_in_order() -> None:
    from generation._figure_captions import numbers_off_figure_captions
    md = (
        "![**Figure 1.** a](figures/a.png)\n\n![**Figure 3.** c](figures/figure3.png)\n\n![**Figure 2.** b](figures/b.png)\n\n"
        "See Figure 3 and Figs. 2-3; inline ![x](figures/figure3.png) and figure3.png stay.\n"
    )
    out = numbers_off_figure_captions(md)
    assert "See Figure 2 and Figs. 2-3;" in out
    assert out.count("figures/figure3.png") == 2 and "and figure3.png stay" in out


def test_a_range_whose_figures_are_no_longer_consecutive_is_listed() -> None:
    from generation._figure_captions import _renumbered
    assert _renumbered("Figs. 2-3 show it.", {1: 2, 2: 1, 3: 3}) == "Figs. 1 and 3 show it."
    assert _renumbered("Figs. 2-3 show it.", {2: 3, 3: 2}) == "Figs. 2-3 show it."
    assert _renumbered("Figs. 2-3 show it.", {2: 4, 3: 1, 4: 2, 1: 3}) == "Figs. 1 and 4 show it."
    assert _renumbered("Figures 2 to 3.", {2: 3, 3: 2}) == "Figures 2 to 3."


# ---- A figure's caption from FI's record of the drawing calls ----------------------------------------------------------

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from core.engine import _calls_of, _figure_caption_text, _figure_record_note  # noqa: E402


def _series(ident, label, kind="line", ys=(1.0, 2.0, 3.0), yerr=None, meaning=None):
    return {"id": ident, "label": label, "kind": kind, "step": False, "x": [1.0, 2.0, 3.0], "y": list(ys),
            "yerr": yerr, "yerr_meaning": meaning, "merged_with": []}


def _panel(n=1, series=(), bands=(), refs=(), raw=False, texts=(), y_what="error", y_label="Error of the estimate"):
    return {"panel": n, "raw": raw, "raw_reason": "an artist FI did not draw" if raw else None,
            "x": {"id": "size", "kind": "setting", "label": "Problem size", "scale": "linear"},
            "y": {"what": y_what, "label": y_label, "scale": "linear"},
            "series": list(series), "bands": list(bands), "refs": list(refs), "notes": [], "texts_raw": list(texts)}


def _calls(*panels, fallback=False):
    return {"version": 1, "fallback": fallback, "panels": list(panels)}


# What the old code says about a recorded panel with a body-typed series label and a stale "not drawn" list.
_RECORD = {"axes": [{"title": "load 3", "xlabel": "Step size", "ylabel": "Error of the estimate",
                     "series": [{"label": "Method A", "shows": "yes", "min": 0.1, "max": 0.9},
                                {"label": "Method B", "shows": "flat", "min": 0.2, "max": 0.2}]}],
           "replicate_mean": {"n": 4}}


def test_caption_names_computed_quantity_as_drawn() -> None:
    # Fault 3: the body worked out a quantity no table column holds; the call says so (what == "other"), so the caption
    # names it neutrally and does not call it a result of the plan.
    calls = _calls(_panel(series=[_series("other", "computed value")], y_what="other", y_label="computed value"))
    caption = _figure_caption_text("a.png", _RECORD, n_seeds=None, calls=calls)
    assert caption.startswith("computed value against Problem size.")
    assert "Error of the estimate" not in caption


def test_caption_never_quotes_body_typed_label() -> None:
    # Fault 9: a curve carries a label the body typed; the caption states a curve, never its words.
    curve = _series("curve1", "Exact solution (computed in the figure)", kind="curve")
    curve["source"] = "computed in the figure"
    calls = _calls(_panel(series=[_series("error", "Method A"), curve]))
    record = {"axes": [{"xlabel": "Step size", "ylabel": "Typed axis words",
                        "series": [{"label": "Exact solution (computed in the figure)", "shows": "yes", "min": 0, "max": 1}]}]}
    caption = _figure_caption_text("a.png", record, n_seeds=None, calls=calls)
    assert "Method A" in caption and "a curve" in caption
    assert "Exact solution" not in caption and "Typed axis words" not in caption


def test_caption_lists_only_drawn_series() -> None:
    # Fault 2: the plan declared three results; the calls drew two. The record's "declared" list never adds a third.
    calls = _calls(_panel(series=[_series("error", "Method A"), _series("cost", "Method B")]))
    calls["declared"] = {"not_drawn": ["cost_c"], "labels": {"cost_c": "Method C"}}
    record = dict(_RECORD, declared=calls["declared"])
    caption = _figure_caption_text("a.png", record, n_seeds=None, calls=calls)
    assert "It draws Method A, Method B." in caption
    assert "Method C" not in caption


def test_caption_states_bands_error_bars_and_references() -> None:
    calls = _calls(_panel(
        series=[_series("error", "Method A", yerr=[0.1, 0.1, 0.1], meaning="sd")],
        bands=[{"series": "error", "x": [1.0, 2.0, 3.0], "lo": [0, 1, 2], "hi": [2, 3, 4], "meaning": "ci95"}],
        refs=[{"id": "limit", "axis": "y", "value": 1.5, "label": "Target"}]))
    caption = _figure_caption_text("a.png", None, n_seeds=None, calls=calls)
    assert "Method A has a band: a 95% confidence interval." in caption
    assert "Error bars show one standard deviation." in caption
    assert "Marked: Target at 1.5." in caption


def test_raw_panel_caption_is_custom_drawing_and_quotes_digit_text() -> None:
    # Fault 4: a raw panel says only that it is a custom drawing, plus the digit-bearing words it prints, verbatim.
    calls = _calls(_panel(1, series=[_series("error", "Method A")]),
                   _panel(2, raw=True, texts=["fit slope 2.04", "label without digits"]))
    caption = _figure_caption_text("a.png", _RECORD, n_seeds=None, calls=calls)
    assert "panel 2: custom drawing" in caption
    assert "text in the panel, not checked by FI: 'fit slope 2.04'" in caption
    assert "label without digits" not in caption


def test_fallback_caption_has_no_apology() -> None:
    calls = _calls(_panel(series=[_series("error", "Method A")]), fallback=True)
    caption = _figure_caption_text("a.png", None, n_seeds=None, calls=calls)
    assert caption == "Error of the estimate against Problem size. It draws Method A."
    assert "fallback" not in caption.lower() and "could not" not in caption.lower()


def test_caption_without_call_record_is_unchanged() -> None:
    # RF1: this string is what the code before the call record wrote for this record.
    golden = ("Error of the estimate against Step size. It draws Method A, Method B. Method B is flat at this scale. "
              "Each line is the mean of 4 seeds, shaded with its 95% confidence interval.")
    assert _figure_caption_text("err_by_step.png", _RECORD, n_seeds=4) == golden
    assert _figure_caption_text("err_by_step.png", _RECORD, n_seeds=4, calls=None) == golden


def test_record_note_without_calls_is_unchanged_and_with_calls_names_the_drawn() -> None:
    plain = _figure_record_note(_RECORD, n_seeds=4)
    assert _figure_record_note(_RECORD, n_seeds=4, calls=None) == plain
    calls = _calls(_panel(series=[_series("error", "Method A")]), _panel(2, raw=True, texts=["slope 2.0"]))
    note = _figure_record_note(_RECORD, n_seeds=4, calls=calls)
    assert "Method A" in note and "custom drawing" in note and "slope 2.0" in note
    assert "Method B" not in note


def test_calls_of_reads_the_record_beside_the_recorders_files(tmp_path: Path) -> None:
    folder = tmp_path / ".fi" / "figure_records"
    folder.mkdir(parents=True)
    state = {"quest_root": str(tmp_path)}
    assert _calls_of(state, "a") is None
    (folder / "a.calls.json").write_text(json.dumps(_calls(_panel())), encoding="utf-8")
    assert _calls_of(state, "a")["version"] == 1
    (folder / "b.calls.json").write_text("not json", encoding="utf-8")
    assert _calls_of(state, "b") is None
    assert _calls_of({}, "a") is None


def test_layout_fact_with_a_typed_title_never_reaches_a_call_caption() -> None:
    calls = _calls(_panel(1, series=[_series("error", "Method A")]), _panel(2, series=[_series("cost", "Method B")]))
    facts = {"a:layout": 'two panels, the left "Typed title words", the right y axis "Typed axis words"'}
    caption = _figure_caption_text("a.png", _RECORD, n_seeds=None, facts=facts, calls=calls)
    assert "Typed" not in caption and "Panel 1 draws Method A." in caption and "Panel 2 draws Method B." in caption


def test_merged_series_is_named_once_as_the_legend_does_and_same_label_series_stay_two() -> None:
    first, second = _series("a", "Method A"), _series("b", "Method B")
    first["merged_with"] = ["b"]
    caption = _figure_caption_text("a.png", None, n_seeds=None, calls=_calls(_panel(series=[first, second])))
    assert "It draws Method A = Method B." in caption
    note = _figure_record_note(None, calls=_calls(_panel(series=[first, second])))
    assert "Method A = Method B (line)" in note and note.count("(line)") == 1
    twins = _calls(_panel(series=[_series("a", "Same"), _series("b", "Same", ys=(5.0, 6.0, 7.0))]))
    assert "It draws Same, Same." in _figure_caption_text("a.png", None, n_seeds=None, calls=twins)


def test_a_merged_curve_is_never_quoted_and_an_empty_primary_label_keeps_the_rest() -> None:
    first = _series("a", "Method A")
    curve = _series("curve1", "Body typed curve (Some source)", kind="curve")
    first["merged_with"] = ["curve1"]
    calls = _calls(_panel(series=[first, curve]))
    caption = _figure_caption_text("a.png", None, n_seeds=None, calls=calls)
    assert "Body typed" not in caption and "Some source" not in caption
    assert "Method A = a curve" in caption
    assert "Body typed" not in _figure_record_note(None, calls=calls)
    blank, other = _series("a", ""), _series("b", "Method B")
    blank["merged_with"] = ["b"]
    assert "It draws Method B." in _figure_caption_text("a.png", None, n_seeds=None, calls=_calls(_panel(series=[blank, other])))


def test_caption_and_note_name_a_panel_by_the_setting_it_holds() -> None:
    from core.engine import _note_from_calls
    calls = _calls(_panel(1, series=[_series("a", "signal")]), _panel(2, series=[_series("a", "signal")]))
    plain = json.loads(json.dumps(calls))
    assert "Panel 2 draws" in _figure_caption_text("a.png", None, n_seeds=None, calls=plain)
    calls["panels"][0]["hold"], calls["panels"][1]["hold"] = {"m": 100}, {"m": 200}
    caption = _figure_caption_text("a.png", None, n_seeds=None, calls=calls)
    assert "The panel at m = 100 draws" in caption and "The panel at m = 200 draws" in caption and "Panel 2" not in caption
    note = _note_from_calls(calls)
    assert note[0].startswith("the panel at m = 100 (") and note[1].startswith("the panel at m = 200 (")
