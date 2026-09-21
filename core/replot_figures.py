"""Redraw an experiment's line figures as the mean over its seeds.

FrontierInsight copies this file into a quest's ``code/`` folder, with
``replot_figures.json`` beside it, after the experiment has run at several
seeds. The JSON holds each figure whose panels are all lines drawn at the same
x in every seed, and for each line its mean and the bounds of its 95%
confidence interval, which the engine computed. This script only draws them:
each line at its mean, shaded between its bounds, under the figure's own
titles, labels, scales and colours. It runs in the quest's Python, so the
house style and the figure recorder apply as they do to ``experiment.py``.
Self-contained: the quest's Python cannot import FrontierInsight.

A legend is drawn where matplotlib puts it, unless that is over what it
labels: a redrawn figure is the engine's, so nothing else moves its legend. Of
the 20 figures redrawn as a mean in the stored quests whose layout was measured,
6 (in 4 of 9 quests) had their legend over their lines or bands, a legend of
three to eight entries in the middle of the axes, and nothing acted on it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

BAND_ALPHA = 0.18
STYLE_KEYS = (
    "color", "linestyle", "marker", "markersize", "markerfacecolor",
    "linewidth", "alpha", "drawstyle",
)
# A legend of this many entries is set outside the axes whatever they hold: six
# to eight entries cannot sit in a corner of a small panel without covering a line.
LEGEND_OUTSIDE_AT = 4
# Points tested along each line segment, and across the legend, for something under it.
_SEGMENT_SAMPLES = 25
_LEGEND_GRID = 6


def _legend_covers_data(ax, legend) -> bool:
    """Whether ``legend``, as it is now drawn, has a line or a shaded band of
    ``ax`` under it. Measured on the drawn canvas: only the positions can say."""
    fig = ax.figure
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer())
    for line in ax.get_lines():
        if not line.get_visible():
            continue
        points = np.asarray(line.get_xydata(), dtype=float)
        if not len(points):
            continue
        points = line.get_transform().transform(points)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) > 1:
            # A line with five points is a few long segments, and a legend can sit
            # between two of its points: test along each segment as well.
            t = np.linspace(0.0, 1.0, _SEGMENT_SAMPLES)[:, None, None]
            points = (points[None, :-1] * (1.0 - t) + points[None, 1:] * t).reshape(-1, 2)
        inside = (
            (points[:, 0] >= box.x0) & (points[:, 0] <= box.x1)
            & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1)
        )
        if inside.any():
            return True
    xs = np.linspace(box.x0, box.x1, _LEGEND_GRID)
    ys = np.linspace(box.y0, box.y1, _LEGEND_GRID)
    grid = np.array([(x, y) for x in xs for y in ys])
    for collection in ax.collections:
        transform = collection.get_transform()
        for path in collection.get_paths():
            if not len(path.vertices):
                continue
            vertices = transform.transform(path.vertices)
            if MplPath(vertices).contains_points(grid).any():
                return True
    return False


def place_legend(ax) -> bool:
    """Draw ``ax``'s legend, and take it outside the axes, to the right of them,
    when it has ``LEGEND_OUTSIDE_AT`` entries or more or would cover what it
    labels. Returns whether it was moved outside."""
    _handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return False
    legend = ax.legend()
    if len(labels) < LEGEND_OUTSIDE_AT and not _legend_covers_data(ax, legend):
        return False
    legend.remove()
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize="small")
    return True


def draw(figure: dict, out_dir: Path) -> None:
    fig = plt.figure(figsize=figure["size"]) if figure.get("size") else plt.figure()
    grids: dict[tuple[int, int], object] = {}
    legends = []
    for panel in figure["axes"]:
        rows, cols, row_start, row_stop, col_start, col_stop = panel["grid"]
        if (rows, cols) not in grids:
            grids[(rows, cols)] = fig.add_gridspec(rows, cols)
        ax = fig.add_subplot(grids[(rows, cols)][row_start:row_stop, col_start:col_stop])
        for line in panel["lines"]:
            style = {key: line[key] for key in STYLE_KEYS if line.get(key) is not None}
            label = line.get("label") or None
            if line["kind"] == "hline":
                ax.axhline(line["mean"][0], label=label, **style)
            elif line["kind"] == "vline":
                ax.axvline(line["x"][0], label=label, **style)
            else:
                ax.plot(line["x"], line["mean"], label=label, **style)
                if line["lower"] != line["upper"]:
                    ax.fill_between(
                        line["x"], line["lower"], line["upper"],
                        color=style.get("color"), alpha=BAND_ALPHA, linewidth=0,
                    )
        ax.set_xscale(panel.get("xscale") or "linear")
        ax.set_yscale(panel.get("yscale") or "linear")
        ax.set_title(panel.get("title") or "")
        ax.set_xlabel(panel.get("xlabel") or "")
        ax.set_ylabel(panel.get("ylabel") or "")
        if panel.get("legend"):
            legends.append(ax)
    if figure.get("suptitle"):
        fig.suptitle(figure["suptitle"])
    if len(figure["axes"]) > 1:
        fig.tight_layout()
    # After the layout: where a legend lands depends on the size of its panel.
    moved = [place_legend(ax) for ax in legends]
    if any(moved):
        if len(figure["axes"]) > 1:
            fig.tight_layout()
        # A legend outside the axes is outside the canvas too, unless the saved image grows to hold it.
        fig.savefig(out_dir / figure["file"], bbox_inches="tight")
    else:
        fig.savefig(out_dir / figure["file"])
    plt.close(fig)


def main() -> int:
    # The recorder keeps each seed's lines for the redraw; a redrawn figure is
    # no seed's, so it records only what the figure shows.
    os.environ["FI_REPLOT"] = "1"
    data = json.loads(Path(__file__).with_name("replot_figures.json").read_text(encoding="utf-8"))
    failed = 0
    for figure in data.get("figures") or []:
        try:
            draw(figure, Path("figures"))
        except Exception as exc:  # one figure's failure leaves the others drawn
            failed += 1
            plt.close("all")
            print(f"could not redraw {figure.get('file')}: {exc!r}", file=sys.stderr)
        else:
            print(f"REPLOTTED: {figure['file']}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
