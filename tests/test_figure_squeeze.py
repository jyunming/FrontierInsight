"""A figure drawn squeezed is caught on the finished PDF, and a paper copy that squeezes its panels is not made or used.

Cause this guards against: a figure with a wide legend outside its last panel, long labels and several panels, redrawn
as a paper copy 6.3 in wide, made matplotlib's constrained layout give up ("axes sizes collapsed to zero"); the copy was
kept with panels a hair wide and no visible data, and paper.pdf placed it. The panel sizes are measured by the
bootstrap on matplotlib's own axes (``core/plot_style.py``) and recorded per copy; the paper and its visual check read
that record for the copy FI placed (``generation/_paper_figures.py``, ``generation/_pdf_measure.py``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from core.plot_style import PAPER_PANEL_MIN_SHARE, write_boot  # noqa: E402
from generation._paper_figures import copy_is_unusable, placed_copies, prefer_paper_figures, recorded_panel_state  # noqa: E402
from generation._pdf_measure import Document, Line, Page, paper_report, squeezed_copy_findings  # noqa: E402

# A figure of the pattern that collapsed: three panels, a legend outside the last, long labels and a text running past
# its panel. VARIANT adds the kinds of axes a panel check could miss.
SCRIPT = r'''
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
variant = os.environ.get("VARIANT", "")
xs = [20, 200, 2000]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
for i, ax in enumerate(axes):
    for k, name in enumerate(["Alpha", "Beta", "Gamma"]):
        ax.errorbar(xs, [0.1 * (k + 1) * (i + 1) + 0.01 * j for j in range(3)], yerr=0.01, marker="o", label=name)
    ax.axhline(0.05, ls="--")
    ax.text(0.03, 0.05, "reference level for the dashed line = %d" % i, transform=ax.get_yaxis_transform(), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Number of tests $m$")
    ax.set_ylabel("Measured response of the system")
    ax.set_title("panel %d" % i)
    if variant == "twin":
        ax.twinx().plot(xs, [3, 2, 1], color="k")
    if variant == "inset":
        ax.inset_axes([0.5, 0.5, 0.4, 0.4]).plot(xs, [1, 2, 3])
    if variant == "grey":
        ax.set_facecolor("#eeeeee")
if variant == "colorbar":
    im = axes[0].imshow(np.arange(12.0).reshape(3, 4), aspect="auto")
    fig.colorbar(im, ax=axes[0])
axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.suptitle("Title")
fig.tight_layout()
fig.savefig("figures/fig1_x.png", dpi=150, bbox_inches="tight")
'''


def _run(tmp_path: Path, variant: str) -> dict:
    boot = write_boot(tmp_path / "fi")
    (tmp_path / "figures").mkdir()
    (tmp_path / "rec").mkdir()
    script = tmp_path / "make.py"
    script.write_text(SCRIPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(boot), "FI_FIGURE_RECORDS": str(tmp_path / "rec"), "VARIANT": variant}
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-800:]
    return json.loads((tmp_path / "rec" / "fig1_x.json").read_text(encoding="utf-8"))


# Every axes of the figure counts, the twin, the inset and the colorbar included: (variant, axes measured).
@pytest.mark.parametrize(("variant", "axes"), [("", 3), ("twin", 6), ("inset", 6), ("grey", 3), ("colorbar", 4)])
def test_the_bootstrap_measures_every_axes_and_keeps_no_layout_that_squeezes_one(
    tmp_path: Path, variant: str, axes: int,
) -> None:
    record = _run(tmp_path, variant)
    paper = record.get("paper")
    assert paper, "a copy is made by a layout that keeps the panels: %s" % record.get("paper_error")
    assert len(paper["panels_in"]) == axes
    assert paper["panel_share"] >= PAPER_PANEL_MIN_SHARE
    if variant != "colorbar":
        assert paper["layouts_failed"], "the layout that collapsed the panels was tried and not kept"


def _doc(captions: dict[int, int]) -> Document:
    """A paper whose figure N has its caption on page captions[N]."""
    pages = []
    for number in sorted(set(captions.values())):
        lines = [Line(f"Figure {n}. A caption.", 9.0, False, (50.0, 100.0, 300.0, 110.0))
                 for n, page in captions.items() if page == number]
        pages.append(Page(number, 612.0, 792.0, lines, [], []))
    return Document(pages)


def _paper_with(root: Path, records: dict[str, dict | None], source: str | None = None) -> None:
    """Figures with a copy and a record each: ``records[stem]`` is the "paper" part of the record (None: no record)."""
    (root / ".fi" / "figure_records").mkdir(parents=True)
    (root / "figures" / "paper").mkdir(parents=True)
    text = ""
    for stem, paper in records.items():
        (root / "figures" / f"{stem}.png").write_bytes(b"png")
        (root / "figures" / "paper" / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n/MediaBox [0 0 453.6 144]\n")
        if paper is not None:
            (root / ".fi" / "figure_records" / f"{stem}.json").write_text(
                json.dumps({"file": f"{stem}.png", "axes": [], "paper": paper}), encoding="utf-8")
        text += f"![Cap.](figures/paper/{stem}.pdf){{width=100% height=33%}}\n\n"
    (root / "paper_pdf_source.md").write_text(text if source is None else source, encoding="utf-8")


OK = {"panel_measure": "ok"}


def test_of_two_figures_of_the_same_shape_the_one_squeezed_is_the_one_flagged(tmp_path: Path) -> None:
    _paper_with(tmp_path, {"fig_a": {**OK, "panel_share": 1.0}, "fig_b": {**OK, "panel_share": 0.01}})
    copies = placed_copies(tmp_path)
    assert copies == [(1, "fig_a", "ok", 1.0), (2, "fig_b", "ok", 0.01)]
    found = squeezed_copy_findings(_doc({1: 1, 2: 2}), copies)
    assert [(f["check"], f["figure"], f["page"]) for f in found] == [("figure_squeezed", "fig_b", 2)]
    report = paper_report(_doc({1: 1, 2: 2}), placed_copies=copies)
    assert [f["figure"] for f in report["findings"] if f["check"] == "figure_squeezed"] == ["fig_b"]


def test_a_copy_whose_panels_could_not_be_measured_is_not_used_and_is_reported(tmp_path: Path) -> None:
    records = {
        "fig_a": {**OK, "panel_share": 1.0},
        "fig_b": {**OK, "panel_share": 0.01},
        "fig_c": {"panel_measure": "failed", "panel_share": None},
        "fig_d": {"panel_measure": "failed", "panel_share": 0.9},  # measured in part: still not trusted
        "fig_e": {"panel_share": None},  # made before a measurement was recorded: used as before
        "fig_f": None,  # no record at all: used as before
        "fig_g": {**OK, "panel_share": None},  # nothing with a size to measure: nothing squeezed
    }
    _paper_with(tmp_path, records)
    states = {s: recorded_panel_state(tmp_path, s)[0] for s in records}
    assert states == {"fig_a": "ok", "fig_b": "ok", "fig_c": "failed", "fig_d": "failed", "fig_e": "old",
                      "fig_f": "old", "fig_g": "ok"}
    md = "".join(f"![Cap.](figures/{s}.png)\n" for s in records)
    out, n = prefer_paper_figures(md, tmp_path)
    used = {s for s in records if f"](figures/paper/{s}.pdf)" in out}
    assert used == {"fig_a", "fig_e", "fig_f", "fig_g"} and n == 4
    assert [copy_is_unusable(tmp_path, s) for s in ("fig_b", "fig_c", "fig_a")] == [True, True, False]
    found = squeezed_copy_findings(_doc({3: 1}), [(3, "fig_c", "failed", None)])
    assert [f["check"] for f in found] == ["figure_not_measured"]


def test_figure_numbers_skip_fenced_code_as_the_caption_numbering_does(tmp_path: Path) -> None:
    source = (
        "![Cap.](figures/paper/fig_a.pdf){width=100% height=33%}\n\n"
        "```\n![Not a figure.](figures/paper/fig_a.pdf)\n```\n\n"
        "Text with an inline ![icon](x.png) image.\n\n"
        "![Cap.](figures/paper/fig_b.pdf){width=100% height=33%}\n"
    )
    _paper_with(tmp_path, {"fig_a": {**OK, "panel_share": 1.0}, "fig_b": {**OK, "panel_share": 0.01}}, source)
    assert [(n, stem) for n, stem, _s, _sh in placed_copies(tmp_path)] == [(1, "fig_a"), (2, "fig_b")]


BOOM = r'''
import sys
from matplotlib.axes import Axes
_orig = Axes.get_window_extent
_seen = set()
def _boom(self, *a, **k):
    frame = sys._getframe(1).f_code
    if "sitecustomize" in frame.co_filename:
        if MODE == "drawn" and id(self.figure) not in _seen:  # the first measurement of each figure, on the drawn one
            _seen.add(id(self.figure))
            raise RuntimeError("cannot measure")
        if MODE == "copy" and frame.co_name == "_fi_paper_attempt":  # the measurement on the copy
            raise RuntimeError("cannot measure")
    return _orig(self, *a, **k)
Axes.get_window_extent = _boom
'''


@pytest.mark.parametrize("mode", ["drawn", "copy"])
def test_a_copy_whose_axes_could_not_be_measured_is_recorded_so_and_not_used(tmp_path: Path, mode: str) -> None:
    boot = write_boot(tmp_path / "fi")
    (tmp_path / "figures").mkdir()
    (tmp_path / "rec").mkdir()
    script = tmp_path / "make.py"
    script.write_text(f"MODE = {mode!r}\n" + BOOM + SCRIPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(boot), "FI_FIGURE_RECORDS": str(tmp_path / "rec"), "VARIANT": ""}
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-800:]
    record = json.loads((tmp_path / "rec" / "fig1_x.json").read_text(encoding="utf-8"))
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "figure_records").mkdir()
    (tmp_path / ".fi" / "figure_records" / "fig1_x.json").write_text(json.dumps(record), encoding="utf-8")
    assert record["paper"]["panel_measure"] == "failed", record["paper"]
    assert copy_is_unusable(tmp_path, "fig1_x"), "the paper sets the figure from its PNG"
