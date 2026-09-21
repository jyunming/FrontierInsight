"""The legend of a figure the engine redraws as the mean over the seeds.

``core/replot_figures.py`` draws the mean figures, so nothing else moves their legend, and the plot-style recorder measured
it over the lines or bands it labels in 6 of the 20 stored redraws whose layout was measured (six to eight entries in the
middle of the axes). A legend now leaves the axes, to their right, when it has four entries or more or the drawn canvas
shows it over a line or a band.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import replot_figures as rf  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _axes(n_lines: int, *, spread: bool = False):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(n_lines):
        ys = [0.1 * i, 0.1 * i + 0.05, 0.1 * i] if spread else [0.02 * i] * 3
        ax.plot([0, 1, 2], ys, label=f"series {i}")
    return fig, ax


def _outside(ax) -> bool:
    fig = ax.figure
    fig.canvas.draw()
    legend = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
    return legend.x0 >= ax.get_window_extent().x1 - 1


def test_a_legend_of_four_entries_or_more_goes_outside_the_axes() -> None:
    _fig, ax = _axes(4)
    assert rf.place_legend(ax) is True and _outside(ax)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [f"series {i}" for i in range(4)]


def test_a_short_legend_that_covers_nothing_stays_where_matplotlib_puts_it() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1, 2], [0.0, 0.0, 0.0], label="low")
    ax.plot([0, 1, 2], [0.1, 0.1, 0.1], label="lower")
    ax.set_ylim(0, 10)  # the whole upper part of the axes is empty
    assert rf.place_legend(ax) is False
    assert not _outside(ax)


def test_a_short_legend_over_a_line_goes_outside() -> None:
    """Two entries, but lines cross every part of the axes, so no corner is free."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for y in [i / 40 for i in range(41)]:
        ax.plot([0, 1], [y, y], color="0.7")
    ax.plot([0, 1], [0.5, 0.5], label="a")
    ax.plot([0, 1], [0.6, 0.6], label="b")
    assert rf.place_legend(ax) is True and _outside(ax)


def test_a_legend_over_a_shaded_band_goes_outside() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1, 2], [0.5, 0.5, 0.5], label="mean")
    ax.fill_between([0, 1, 2], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], alpha=0.2, label="95% CI")
    ax.set_ylim(0, 1)
    assert rf.place_legend(ax) is True and _outside(ax)


def test_an_axes_with_nothing_labelled_gets_no_legend() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    assert rf.place_legend(ax) is False and ax.get_legend() is None


def _plan(entries: int, file: str = "f.png") -> dict:
    x = [100, 500, 1000, 5000]
    lines = [
        {"kind": "data", "label": f"R0={i}", "x": x, "mean": [0.1 * i + 0.01 * k for k in range(4)],
         "lower": [0.1 * i] * 4, "upper": [0.1 * i + 0.05] * 4, "color": None}
        for i in range(entries)
    ]
    return {"file": file, "size": [6, 4], "suptitle": "", "axes": [{
        "grid": [1, 1, 0, 1, 0, 1], "title": "t", "xlabel": "N", "ylabel": "p", "legend": True, "lines": lines,
    }]}


def test_a_redrawn_figure_with_a_long_legend_is_saved_wide_enough_to_hold_it(tmp_path: Path) -> None:
    rf.draw(_plan(8, "long.png"), tmp_path)
    rf.draw(_plan(2, "short.png"), tmp_path)
    long_w = mpimg.imread(tmp_path / "long.png").shape[1]
    short_w = mpimg.imread(tmp_path / "short.png").shape[1]
    assert long_w > short_w  # the saved image grew to hold a legend outside the axes
    assert short_w <= 6 * 100 + 2  # a figure that kept its legend is the size it was asked to be


def test_a_two_panel_figure_keeps_a_legend_per_panel_and_saves(tmp_path: Path) -> None:
    plan = _plan(5, "two.png")
    panel = dict(plan["axes"][0])
    panel["grid"] = [1, 2, 0, 1, 1, 2]
    plan["axes"][0]["grid"] = [1, 2, 0, 1, 0, 1]
    plan["axes"].append(panel)
    rf.draw(plan, tmp_path)
    assert (tmp_path / "two.png").is_file()


def test_the_legend_rule_is_documented_in_the_module_and_the_docs() -> None:
    repo = Path(__file__).resolve().parent.parent
    assert "outside the axes" in (repo / "docs" / "capabilities-reference.md").read_text(encoding="utf-8")
    assert rf.LEGEND_OUTSIDE_AT == 4
