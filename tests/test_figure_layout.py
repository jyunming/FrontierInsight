"""One legend per figure, a layout that keeps every text clear of every other, one colour per series (core/frame_figure.py,
core/analysis_figures.py, core/plot_style.py). Made-up study: a colony whose ``growth`` and ``load`` are read at settings of ``size``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import analysis_figures, frame_figure  # noqa: E402
from core.plot_style import write_boot  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _Table:
    declared = ["growth", "load"]

    def keys(self):
        return ["size=%d" % v for v in (1, 2, 3)]

    def mean(self, name, key):  # every result's mean at size=v is v: a drawn y = x is the table's own value
        return float(key.partition("=")[2])

    def n(self, name, key):
        return 4

    def interval(self, name, key):
        return (1.25, 3.75)


LONG = "Growth of the colony in the long run, uncorrected for the size of the dish"
LOAD = "Load carried by the colony over the whole season"
LABELS = {"growth": {"label": LONG}, "load": {"label": LOAD},
          "fast": "Fast colony", "slow": "Slow colony", "size": "Size"}


class _Host(frame_figure.FigureCalls):
    def __init__(self, panels=2):
        _plt, self.fig, self.axes = analysis_figures._new_figure(panels)
        self.table = _Table()
        self.x = "size"
        self.settings = ["size"]
        self.series_ids = ["fast", "slow"]
        self.references = {}
        self.labels = LABELS


def _two_panel_host():
    h = _Host(2)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.line(0, "slow", [1, 2, 3], [1.0, 2.0, 3.0], "load")
    h.line(1, "slow", [1, 2, 3], [1.0, 2.0, 3.0], "load")  # the second panel draws them in the other order
    h.line(1, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.finish(LABELS, {})
    return h


def _boxes(fig):
    """The text boxes of every panel (title, labels, tick labels in range) and the legends, in pixels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for i, ax in enumerate(fig.axes):
        for kind, text in [("title", ax.title), ("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)]:
            if text.get_text().strip():
                out.append((i, kind, text.get_window_extent(renderer)))
        leg = ax.get_legend()
        if leg is not None:
            out.append((i, "legend", leg.get_window_extent(renderer)))
    for leg in fig.legends:
        out.append((-1, "figlegend", leg.get_window_extent(renderer)))
    return out


def _overlaps(fig, share=0.05):
    boxes = _boxes(fig)
    found = []
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            (pa, ka, ba), (pb, kb, bb) = boxes[a], boxes[b]
            w, h = min(ba.x1, bb.x1) - max(ba.x0, bb.x0), min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            small = min(ba.width * ba.height, bb.width * bb.height)
            if w > 0 and h > 0 and small > 0 and w * h / small >= share:
                found.append((pa, ka, pb, kb))
    return found


def test_a_two_panel_figure_has_one_legend_with_each_series_once():
    h = _two_panel_host()
    assert all(ax.get_legend() is None for ax in h.axes)
    assert len(h.fig.legends) == 1
    assert [t.get_text() for t in h.fig.legends[0].get_texts()] == ["Fast colony", "Slow colony"]
    assert h.record()["legend"]["scope"] == "figure" and h.record()["legend"]["entries"] == ["Fast colony", "Slow colony"]


def test_the_figure_legend_is_outside_the_axes():
    h = _two_panel_host()
    h.fig.canvas.draw()
    renderer = h.fig.canvas.get_renderer()
    box = h.fig.legends[0].get_window_extent(renderer)
    for ax in h.axes:
        axes_box = ax.get_window_extent(renderer)
        assert box.y1 <= axes_box.y0 + 1 or box.x0 >= axes_box.x1 - 1


def test_no_text_box_overlaps_another_in_a_two_panel_figure_with_long_labels():
    h = _two_panel_host()
    assert _overlaps(h.fig) == []
    assert all(ax.get_window_extent(h.fig.canvas.get_renderer()).width / h.fig.dpi > 1.5 for ax in h.axes)


def test_a_long_y_label_is_wrapped_to_the_panel_height_and_not_cut():
    h = _two_panel_host()
    h.fig.canvas.draw()
    renderer = h.fig.canvas.get_renderer()
    for ax in h.axes:
        label = ax.yaxis.label
        assert " ".join(label.get_text().split()) == " ".join(ax.get_ylabel().split())
        assert label.get_window_extent(renderer).height <= ax.get_window_extent(renderer).height * 1.02
    assert " ".join(h.axes[0].get_ylabel().split()) == LONG + " / " + LOAD and "\n" in h.axes[0].get_ylabel()
    assert h.record()["panels"][0]["y"]["label"] == LONG + " / " + LOAD  # the record keeps the text as written


def test_a_series_has_the_same_colour_in_every_panel():
    h = _two_panel_host()
    colours = {}
    for ax in h.axes:
        for line in ax.lines:
            colours.setdefault(line.get_label(), set()).add(matplotlib.colors.to_hex(line.get_color()))
    assert all(len(v) == 1 for v in colours.values()) and colours["Fast colony"] != colours["Slow colony"]


def test_the_colour_is_the_series_even_when_another_series_comes_first_in_a_panel():
    h = _Host(2)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.line(1, "slow", [1, 2, 3], [1.0, 2.0, 3.0], "load")
    h.line(1, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.finish(LABELS, {})
    colour = {i: [matplotlib.colors.to_hex(line.get_color()) for line in ax.lines] for i, ax in enumerate(h.axes)}
    assert colour[0][0] == colour[1][1] and colour[1][0] != colour[1][1]  # fast is one colour; slow is not fast's


def test_a_figure_with_entries_on_one_panel_keeps_its_legend_inside_that_panel():
    h = _Host(2)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.finish(LABELS, {})
    assert h.fig.legends == [] and h.axes[0].get_legend() is not None
    assert h.record()["legend"]["scope"] == "panel"


def test_a_crowded_single_panel_legend_goes_outside_and_clear_of_the_axes_labels():
    h = _Host(1)
    spots = [[1], [2], [3], [1, 2], [2, 3], [1, 3], [1, 2, 3]]  # seven series that are not one line to the reader
    h.series_ids = ["series%d" % k for k in range(len(spots))]
    for k, at in enumerate(spots):
        h.line(0, "series%d" % k, at, [float(v) for v in at], "growth")
    h.finish(LABELS, {})
    assert len(h.fig.legends) == 1 and h.axes[0].get_legend() is None
    assert _overlaps(h.fig) == []


def test_no_text_box_overlaps_another_in_a_three_panel_figure_with_a_crowded_legend():
    h = _Host(3)
    spots = [[1], [2], [3], [1, 2], [2, 3], [1, 3], [1, 2, 3]]
    h.series_ids = ["series%d" % k for k in range(len(spots))]
    for panel in range(3):
        for k, at in enumerate(spots):
            h.line(panel, "series%d" % k, at, [float(v) for v in at], "growth" if panel % 2 == 0 else "load")
    h.finish(LABELS, {})
    assert len(h.fig.legends) == 1 and all(ax.get_legend() is None for ax in h.axes)
    assert _overlaps(h.fig) == []


def test_the_layout_engine_is_constrained():
    plt_, fig, axes = analysis_figures._new_figure(3)
    assert type(fig.get_layout_engine()).__name__ == "ConstrainedLayoutEngine" and len(axes) == 3


def _paper_copy(tmp_path, panels):
    """A saved figure with a figure legend, through the bootstrap: the record of the figure and its paper copy."""
    boot_dir = write_boot(tmp_path, "latex")
    records = tmp_path / "records"
    figures = tmp_path / "figures"
    figures.mkdir()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p),
           "FI_FIGURE_RECORDS": str(records)}
    script = tmp_path / "draw.py"
    script.write_text(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "fig, axes = plt.subplots(1, %d, figsize=(7, 3.2), layout='constrained')\n"
        "axes = list(axes) if %d > 1 else [axes]\n"
        "for ax in axes:\n"
        "    ax.plot([1, 2, 3], [1, 2, 3], label='Fast colony'); ax.plot([1, 2, 3], [3, 2, 1], label='Slow colony')\n"
        "    ax.set_xlabel('Size'); ax.set_ylabel('Growth of the colony in the long run, uncorrected for the size of the dish')\n"
        "fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)\n"
        "fig.savefig('figures/g.png')\n" % (panels, panels), encoding="utf-8")
    out = subprocess.run([sys.executable, str(script)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    return json.loads((records / "g.json").read_text(encoding="utf-8")), figures / "paper"


def test_the_paper_copy_accepts_a_figure_legend(tmp_path):
    record, paper = _paper_copy(tmp_path, 2)
    info = record["paper"]
    assert "error" not in info and info["fit"].startswith("constrained") and not info.get("layouts_failed")
    assert (paper / "g.pdf").is_file() and (paper / "g.png").is_file()


def test_the_record_holds_the_figure_legend_box_and_words_and_a_wrapped_label_as_one_line(tmp_path):
    record, _paper = _paper_copy(tmp_path, 2)
    box = record["legend_box"]
    assert 0 <= box["x0"] < box["x1"] <= 1.0001 and 0 <= box["y0"] < box["y1"] <= 1.0001
    assert "Fast colony" in record["texts"] and "Slow colony" in record["texts"]
    assert all(ax["ylabel"].count("\n") == 0 for ax in record["axes"])


def test_geometry_findings_check_the_figures_own_legend_box():
    from core import figure_facts

    axes = [{"place": [0.1, 0.3, 0.8, 0.6]}]
    inside = {"axes": axes, "legend_box": {"x0": 0.1, "y0": 0.02, "x1": 0.9, "y1": 0.15}}
    over = {"axes": axes, "legend_box": {"x0": 0.1, "y0": 0.4, "x1": 0.9, "y1": 0.5}}
    off = {"axes": axes, "legend_box": {"x0": 0.1, "y0": -0.2, "x1": 0.9, "y1": 0.1}}
    assert figure_facts.geometry_findings(inside) == []
    assert any("covers part of" in s for s in figure_facts.geometry_findings(over))
    assert any("reaches outside" in s for s in figure_facts.geometry_findings(off))
