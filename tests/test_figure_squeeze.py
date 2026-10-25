"""A figure drawn squeezed is caught on the finished PDF, and a paper copy that squeezes its panels is not made or used.

Cause this guards against: a figure with a wide legend outside its last panel, long labels and several panels, redrawn
as a paper copy 6.3 in wide, made matplotlib's constrained layout give up ("axes sizes collapsed to zero"); the copy was
kept with panels a hair wide and no visible data, and paper.pdf placed it. The panel sizes are measured: on the canvas
the copy is made from (``core/plot_style.py``), on the copy file before the paper uses it
(``generation/_paper_figures.py``) and on the placed figure in paper.pdf (``generation/_pdf_measure.py``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pdfium = pytest.importorskip("pypdfium2")
matplotlib = pytest.importorskip("matplotlib")

from core.plot_style import PAPER_PANEL_MIN_SHARE, write_boot  # noqa: E402
from generation._paper_figures import copy_is_squeezed, paper_figure, prefer_paper_figures  # noqa: E402
from generation._pdf_measure import (  # noqa: E402
    Page,
    PaperFigure,
    figure_frames,
    figure_squeeze_findings,
    measure_pdf,
    paper_report,
    panel_share,
)

# Three panels side by side on a 12 x 3.8 in canvas, each 23% x 46% of it.
PLACES = tuple((0.08 + 0.29 * i, 0.23, 0.23, 0.46) for i in range(3))
FIGURE = PaperFigure("fig_a", 6.3, 1.9, (12.0, 3.8), PLACES)


def _copy_pdf(path: Path, widths: tuple[float, float, float], *, size=(6.3, 1.9)) -> Path:
    """A paper-copy-like PDF: a white canvas and three white plotting areas with the given widths (shares of the
    canvas width), each with a data line."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=size)
    for i, w in enumerate(widths):
        ax = fig.add_axes((0.08 + 0.3 * i, 0.25, w, 0.55))
        ax.plot([0, 1, 2], [1, 3, 2])
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def _page(fills, images=(), image_px=()):
    return Page(1, 612.0, 792.0, [], list(images), [], list(image_px), list(fills))


def test_panel_share_is_the_smallest_panel_against_its_drawn_size_at_the_placed_width() -> None:
    frame = (50.0, 600.0, 50.0 + 6.3 * 72, 600.0 + 1.9 * 72)
    scale = 6.3 / 12.0
    ok = [(60.0 + 140 * i, 620.0, 60.0 + 140 * i + 0.23 * 12 * scale * 72, 620.0 + 0.46 * 3.8 * scale * 72)
          for i in range(3)]
    assert panel_share(frame, ok, FIGURE) == pytest.approx(1.0, abs=0.02)
    sliver = list(ok)
    sliver[1] = (sliver[1][0], sliver[1][1], sliver[1][0] + 1.0, sliver[1][3])
    assert panel_share(frame, sliver, FIGURE) < 0.05
    assert panel_share(frame, ok[:2], FIGURE) is None  # not the panels the figure recorded


def test_a_collapsed_vector_figure_is_flagged_and_a_sound_one_is_not(tmp_path: Path) -> None:
    for widths, flagged in (((0.22, 0.22, 0.22), False), ((0.004, 0.004, 0.004), True)):
        pdf = _copy_pdf(tmp_path / "f.pdf", widths)
        doc = measure_pdf(pdf)
        assert doc is not None
        frames = figure_frames(doc.pages[0])
        assert len(frames) == 1 and len(frames[0][1]) == 3
        found = figure_squeeze_findings(doc.pages[0], [FIGURE])
        assert bool(found) is flagged, found
        if flagged:
            assert found[0]["check"] == "figure_squeezed" and found[0]["figure"] == "fig_a"
            report = paper_report(doc, figures=[FIGURE])
            assert any(f["check"] == "figure_squeezed" for f in report["findings"])


def test_a_picture_placed_at_another_shape_than_its_file_is_flagged() -> None:
    page = _page([], images=[(100.0, 400.0, 400.0, 500.0), (100.0, 100.0, 400.0, 250.0)],
                 image_px=[(900, 300), (900, 300)])
    found = figure_squeeze_findings(page, [])
    # The first is placed 3.0 : 1 as stored; the second 2.0 : 1, squeezed from 3.0 : 1.
    assert [f["check"] for f in found] == ["figure_squeezed"]
    assert found[0]["placed_aspect"] == pytest.approx(2.0) and found[0]["stored_aspect"] == pytest.approx(3.0)


def test_a_copy_is_not_used_when_its_panels_are_slivers_and_is_when_they_are_not(tmp_path: Path) -> None:
    root = tmp_path
    (root / ".fi" / "figure_records").mkdir(parents=True)
    (root / "figures" / "paper").mkdir(parents=True)
    (root / "figures" / "fig_a.png").write_bytes(b"png")
    record = {"file": "fig_a.png", "axes": [{"place": list(p)} for p in PLACES],
              "paper": {"drawn_in": [12.0, 3.8], "width_in": 6.3, "height_in": 1.9}}
    (root / ".fi" / "figure_records" / "fig_a.json").write_text(json.dumps(record), encoding="utf-8")
    copy = root / "figures" / "paper" / "fig_a.pdf"
    md = "![Cap.](figures/fig_a.png)\n"

    _copy_pdf(copy, (0.004, 0.004, 0.004))
    assert paper_figure(root, "fig_a") is not None
    assert copy_is_squeezed(copy, paper_figure(root, "fig_a")) is not None
    assert prefer_paper_figures(md, root) == (md, 0)  # the PNG is used

    _copy_pdf(copy, (0.22, 0.22, 0.22))
    assert copy_is_squeezed(copy, paper_figure(root, "fig_a")) is None
    out, n = prefer_paper_figures(md, root)
    assert n == 1 and "figures/paper/fig_a.pdf" in out


# A figure of the pattern that collapsed: three panels, a legend outside the last, long labels and a text running past
# its panel.
SCRIPT = r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
xs = [20, 200, 2000]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
for i, ax in enumerate(axes):
    for k, name in enumerate(["Alpha", "Beta", "Gamma"]):
        ax.errorbar(xs, [0.1 * (k + 1) * (i + 1) + 0.01 * j for j in range(3)], yerr=0.01, marker="o", label=name)
    ax.axhline(0.05, ls="--")
    ax.text(0.03, 0.05, "reference level for the dashed line = %d" % i, transform=ax.get_yaxis_transform(), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Number of tests $m$")
    ax.set_ylabel("Realised false discovery rate")
    ax.set_title("panel %d" % i)
axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.suptitle("Title")
fig.tight_layout()
fig.savefig("figures/fig1_x.png", dpi=150, bbox_inches="tight")
'''


def test_the_bootstrap_does_not_keep_a_layout_that_collapses_the_panels(tmp_path: Path) -> None:
    boot = write_boot(tmp_path / "fi")
    (tmp_path / "figures").mkdir()
    (tmp_path / "rec").mkdir()
    script = tmp_path / "make.py"
    script.write_text(SCRIPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(boot), "FI_FIGURE_RECORDS": str(tmp_path / "rec")}
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-800:]
    paper = json.loads((tmp_path / "rec" / "fig1_x.json").read_text(encoding="utf-8")).get("paper")
    assert paper, "a copy is made by a layout that keeps the panels"
    assert paper["panel_share"] >= PAPER_PANEL_MIN_SHARE
    assert paper["layouts_failed"], "the layout that collapsed the panels was tried and not kept"
    # And the file itself, measured the way the paper measures it, has panels of a size.
    doc = measure_pdf(tmp_path / "figures" / "paper" / "fig1_x.pdf", max_pages=1)
    frames = figure_frames(doc.pages[0])
    assert len(frames) == 1 and len(frames[0][1]) == 3
    assert all(box[2] - box[0] > 36.0 for box in frames[0][1])
