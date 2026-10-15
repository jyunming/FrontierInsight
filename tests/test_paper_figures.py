"""The paper copy of each experiment figure (core/plot_style.py).

FI papers printed figure text at about twice the body size: the scripts draw on
canvases 10-15 in wide with slide-sized text, and the page shrinks them into a
6.3 in column. The bootstrap now saves a second copy of every figure saved into
``figures/`` under ``figures/paper/``: a vector PDF and a PNG at the column's
width, text at 8-9 pt as printed, no title inside the plot, a legend that covered
the data moved off it. The slide copy stays exactly as the script drew it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.plot_style import (
    HOUSE_TICK_PT,
    PAPER_FULL_WIDTH_IN,
    PAPER_HALF_WIDTH_IN,
    PAPER_LABEL_PT,
    PAPER_TICK_PT,
    _boot_source,
    fi_rcparams,
    paper_figure_size,
    paper_profile,
    write_boot,
)


def test_the_slide_profile_is_unchanged_and_the_paper_profile_is_body_sized() -> None:
    slide = fi_rcparams("latex")
    assert slide == fi_rcparams("latex", "slide")
    assert slide["savefig.bbox"] == "tight" and slide["xtick.labelsize"] == HOUSE_TICK_PT
    paper = fi_rcparams("latex", "paper")
    assert paper["figure.figsize"][0] == PAPER_FULL_WIDTH_IN
    assert paper["xtick.labelsize"] == paper["legend.fontsize"] == 8.0
    assert paper["axes.labelsize"] == 9.0
    assert paper["figure.constrained_layout.use"] is True
    assert paper["savefig.bbox"] == "standard"  # no tight crop: it changes the size after the fact
    assert paper["pdf.fonttype"] == 42  # text stays text in the PDF
    assert paper["axes.facecolor"] == slide["axes.facecolor"]


@pytest.mark.parametrize(("drawn", "rows", "cols", "expected"), [
    ((14, 4.2), 1, 2, (PAPER_FULL_WIDTH_IN, 2.0, "full")),  # 6.3 x 1.89 is raised to one row's minimum
    ((15, 4.4), 1, 3, (PAPER_FULL_WIDTH_IN, 2.0, "full")),
    ((6, 5), 1, 1, (PAPER_HALF_WIDTH_IN, round(3.1 * 5 / 6, 4), "half")),
    ((7, 4.3), 1, 1, (PAPER_HALF_WIDTH_IN, 2.0, "half")),
    ((10, 4), 1, 1, (PAPER_FULL_WIDTH_IN, 2.52, "full")),  # one wide panel: a time series
    ((8, 10), 2, 2, (PAPER_FULL_WIDTH_IN, 5.8, "full")),  # 7.875 tall is capped at two rows' maximum
    ((4, 8), 1, 1, (PAPER_HALF_WIDTH_IN, 2.9, "half")),
])
def test_the_paper_size_rule(drawn, rows, cols, expected) -> None:
    width, height, rule = paper_figure_size(drawn, rows, cols)
    assert (width, round(height, 2), rule) == (expected[0], round(expected[1], 2), expected[2])


def test_the_bootstrap_embeds_the_one_size_rule_and_compiles() -> None:
    source = _boot_source("latex")
    compile(source, "sitecustomize.py", "exec")
    assert "def _fi_paper_figure_size(" in source
    assert repr(paper_profile()) in source
    assert "@@" not in source


# ---- the bootstrap in a subprocess, as the execute node runs it ----------


def _run(tmp_path: Path, script: str, *, records: bool = True, extra_env: dict | None = None) -> Path:
    pytest.importorskip("matplotlib")
    boot_dir = write_boot(tmp_path / ".fi", "latex")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p),
        **(extra_env or {}),
    }
    env.pop("FI_FIGURE_RECORDS", None)
    if records:
        env["FI_FIGURE_RECORDS"] = str(tmp_path / ".fi" / "figure_records")
    (tmp_path / "figures").mkdir(exist_ok=True)
    path = tmp_path / "plot.py"
    path.write_text("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n" + script,
                    encoding="utf-8")
    out = subprocess.run([sys.executable, str(path)], env=env, cwd=tmp_path, capture_output=True, text=True,
                         timeout=180)
    assert out.returncode == 0, out.stderr
    return tmp_path


def _record(root: Path, stem: str) -> dict:
    return json.loads((root / ".fi" / "figure_records" / f"{stem}.json").read_text(encoding="utf-8"))


def _pdf_text(path: Path):
    from generation._pdf_measure import measure_pdf

    doc = measure_pdf(path)
    assert doc is not None and len(doc.pages) == 1
    return doc.pages[0]


TWO_PANELS = (
    "import numpy as np\n"
    "x = np.linspace(0, 1, 50)\n"
    "fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.2))\n"
    "a.plot(x, x ** 2, label='quadratic'); a.plot(x, x, label='linear'); a.legend(loc='upper left')\n"
    "a.set_title('m = 100'); a.set_xlabel('time (s)'); a.set_ylabel('distance (m)')\n"
    "b.plot(x, np.sqrt(x)); b.set_title('A very long panel title that cannot fit over a third of a column at nine points')\n"
    "b.set_xlabel('time (s)')\n"
    "fig.suptitle('Everything the experiment found')\n"
    "fig.savefig('figures/two_panels.png', dpi=150, bbox_inches='tight')\n"
)


def test_a_wide_two_panel_figure_prints_at_the_column_width_with_body_sized_text(tmp_path: Path) -> None:
    root = _run(tmp_path, TWO_PANELS)
    pdf, png = root / "figures" / "paper" / "two_panels.pdf", root / "figures" / "paper" / "two_panels.png"
    assert pdf.is_file() and png.is_file()
    # The paper copy lives in a folder of its own: the figure list of the run is unchanged.
    assert sorted(p.name for p in (root / "figures").iterdir() if p.is_file()) == ["two_panels.png"]
    page = _pdf_text(pdf)
    assert abs(page.width / 72 - PAPER_FULL_WIDTH_IN) < 0.01
    assert abs(page.height / 72 - 2.0) < 0.01
    sizes = sorted({round(line.size, 1) for line in page.lines})
    assert sizes and min(sizes) >= 7.5 and max(sizes) <= 9.5, sizes
    texts = " ".join(line.text for line in page.lines)
    assert "Everything the experiment found" not in texts  # no figure title in a paper figure
    # One panel title does not fit over its panel at 9 pt: the panels' titles go together, never one of them.
    assert "m = 100" not in texts and "cannot fit" not in texts
    from PIL import Image

    with Image.open(png) as image:
        assert image.size[0] == round(PAPER_FULL_WIDTH_IN * 300)

    record = _record(root, "two_panels")
    paper = record["paper"]
    assert paper["file"] == "paper/two_panels.pdf" and paper["png"] == "paper/two_panels.png"
    assert paper["drawn_in"] == [14.0, 4.2] and paper["width"] == "full"
    assert paper["width_in"] == PAPER_FULL_WIDTH_IN and paper["height_in"] == 2.0
    assert paper["tick_pt"] == PAPER_TICK_PT and paper["label_pt"] == PAPER_LABEL_PT
    assert paper["suptitle_dropped"] == "Everything the experiment found"
    assert paper["titles_kept"] == []
    assert paper["titles_dropped"] == [
        "m = 100", "A very long panel title that cannot fit over a third of a column at nine points"]
    assert paper["fit"] == "constrained" and paper["layout"] == []
    # The slide copy and its record are as the script drew them.
    assert record["tick_pt"] == HOUSE_TICK_PT
    seed = json.loads((root / ".fi" / "figure_records" / "two_panels.seed0.json").read_text(encoding="utf-8"))
    assert seed["size"] == [14.0, 4.2] and seed["suptitle"] == "Everything the experiment found"


def test_short_panel_titles_name_their_panels_and_stay(tmp_path: Path) -> None:
    root = _run(tmp_path, (
        "fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))\n"
        "for ax, m in zip(axes, (20, 200, 2000)):\n"
        "    ax.plot([0, 0.05, 0.2], [0.6, 0.4, 0.2]); ax.set_title(f'm = {m}')\n"
        "fig.savefig('figures/panels.png')\n"
    ))
    paper = _record(root, "panels")["paper"]
    assert paper["titles_kept"] == ["m = 20", "m = 200", "m = 2000"] and paper["titles_dropped"] == []
    texts = " ".join(line.text for line in _pdf_text(root / "figures" / "paper" / "panels.pdf").lines)
    assert "m = 2000" in texts


def test_an_axis_label_longer_than_its_axis_is_set_on_two_lines(tmp_path: Path) -> None:
    root = _run(tmp_path, (
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))\n"
        "for ax in axes:\n"
        "    ax.plot([0, 1], [1, 2])\n"
        "axes[2].set_ylabel('observed SD / linearised SE of the fitted slope')\n"
        "fig.savefig('figures/long_label.png')\n"
    ))
    paper = _record(root, "long_label")["paper"]
    assert paper["labels_wrapped"] == ["observed SD / linearised SE of the fitted slope"]
    page = _pdf_text(root / "figures" / "paper" / "long_label.pdf")
    assert all(-0.5 <= line.box[1] and line.box[3] <= page.height + 0.5 for line in page.lines)


def test_a_one_panel_figure_prints_at_half_width_without_its_title(tmp_path: Path) -> None:
    root = _run(tmp_path, (
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "ax.scatter([1, 2, 3], [3, 1, 2], label='estimates')\n"
        "ax.set_title('sigma=0.02, n=12: observed r=0.9992'); ax.set_xlabel('Ea_hat (kJ/mol)')\n"
        "fig.savefig('figures/scatter.png')\n"
    ))
    page = _pdf_text(root / "figures" / "paper" / "scatter.pdf")
    assert abs(page.width / 72 - PAPER_HALF_WIDTH_IN) < 0.01
    assert "observed r" not in " ".join(line.text for line in page.lines)
    paper = _record(root, "scatter")["paper"]
    assert paper["width"] == "half" and paper["titles_dropped"] == ["sigma=0.02, n=12: observed r=0.9992"]


def test_a_legend_over_the_data_is_moved_off_it_in_the_paper_copy(tmp_path: Path) -> None:
    root = _run(tmp_path, (
        "import numpy as np\n"
        "x = np.linspace(0, 1, 200)\n"
        "fig, ax = plt.subplots(figsize=(7, 4.3))\n"
        # Lines that fill the whole axes: no corner is empty, so the legend goes under the axes.
        "for k in range(12):\n"
        "    ax.plot(x, (x * (k + 1)) % 1, label=f'run {k}' if k < 3 else None)\n"
        "ax.legend(loc='center')\n"
        "fig.savefig('figures/covered.png')\n"
    ))
    record = _record(root, "covered")
    assert [f["check"] for f in record["layout"]] == ["legend_over_data"]  # the slide copy as the script drew it
    paper = record["paper"]
    assert paper["legend_moved"] and paper["legend_moved"][-1]["to"] in ("best", "below the axes")
    assert paper["layout"] == []
    page = _pdf_text(root / "figures" / "paper" / "covered.pdf")
    assert {"run 0", "run 1", "run 2"} <= {line.text.strip() for line in page.lines} | set(
        " ".join(line.text for line in page.lines).split("  "))


def test_a_figure_legend_set_below_the_canvas_is_kept_on_the_page(tmp_path: Path) -> None:
    """The kinetics quest's figure 1: three panels and one figure legend anchored
    under the canvas, which only a tight crop kept in the image."""
    root = _run(tmp_path, (
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))\n"
        "for ax in axes:\n"
        "    ax.plot([0, 1], [0, 1], label='n = 5'); ax.plot([0, 1], [1, 0], label='n = 12')\n"
        "handles, labels = axes[0].get_legend_handles_labels()\n"
        "fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2)\n"
        "fig.savefig('figures/legend_below.png', dpi=150, bbox_inches='tight')\n"
    ))
    paper = _record(root, "legend_below")["paper"]
    assert {"panel": "figure", "to": "outside lower center"} in paper["legend_moved"]
    assert paper["width_in"] == PAPER_FULL_WIDTH_IN
    page = _pdf_text(root / "figures" / "paper" / "legend_below.pdf")
    legend_lines = [line for line in page.lines if "n = 5" in line.text]
    assert legend_lines and all(0 <= line.box[1] and line.box[3] <= page.height for line in legend_lines)


def test_saving_the_same_figure_twice_saves_it_as_the_script_drew_it_both_times(tmp_path: Path) -> None:
    root = _run(tmp_path, (
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "ax.plot([0, 1, 2], [0, 1, 4], label='squares'); ax.legend(); ax.set_title('Squares')\n"
        "fig.savefig('figures/twice.png')\n"
        "first = (list(fig.get_size_inches()), ax.get_title('left'), ax.xaxis.get_ticklabels()[0].get_fontsize(),\n"
        "         ax.get_lines()[0].get_linewidth(), fig.get_layout_engine())\n"
        "fig.savefig('figures/twice.svg')\n"
        "fig.canvas.draw()\n"
        "after = (list(fig.get_size_inches()), ax.get_title('left'), ax.xaxis.get_ticklabels()[0].get_fontsize(),\n"
        "         ax.get_lines()[0].get_linewidth(), fig.get_layout_engine())\n"
        "assert [list(map(float, first[0]))] + list(first[1:]) == [list(map(float, after[0]))] + list(after[1:]), (first, after)\n"
        "assert after[0] == [9.0, 4.0] and after[1] == 'Squares' and after[4] is None\n"
    ))
    assert _record(root, "twice")["tick_pt"] == HOUSE_TICK_PT
    assert (root / "figures" / "paper" / "twice.pdf").is_file()


def test_no_paper_copy_outside_the_figures_folder_or_when_turned_off(tmp_path: Path) -> None:
    root = _run(tmp_path, (
        "import io\n"
        "fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1])\n"
        "fig.savefig('elsewhere.png'); fig.savefig(io.BytesIO(), format='png')\n"
    ))
    assert not (root / "paper").exists() and not (root / "figures" / "paper").exists()
    assert "paper" not in _record(root, "elsewhere")
    off = _run(tmp_path / "off", (
        "fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1]); fig.savefig('figures/plain.png')\n"
    ), extra_env={"FI_PAPER_FIGURES": "0"})
    assert not (off / "figures" / "paper").exists()


def test_the_paper_copy_is_made_without_records_too(tmp_path: Path) -> None:
    """A redraw from the saved numbers runs without the records; its figures still get their paper copy."""
    root = _run(tmp_path, (
        "fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1]); fig.savefig('figures/bare.png')\n"
    ), records=False)
    assert (root / "figures" / "paper" / "bare.pdf").is_file()
    assert not (root / ".fi" / "figure_records").exists()
