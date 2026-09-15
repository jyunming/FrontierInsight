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
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BAND_ALPHA = 0.18
STYLE_KEYS = (
    "color", "linestyle", "marker", "markersize", "markerfacecolor",
    "linewidth", "alpha", "drawstyle",
)


def draw(figure: dict, out_dir: Path) -> None:
    fig = plt.figure(figsize=figure["size"]) if figure.get("size") else plt.figure()
    grids: dict[tuple[int, int], object] = {}
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
            ax.legend()
    if figure.get("suptitle"):
        fig.suptitle(figure["suptitle"])
    if len(figure["axes"]) > 1:
        fig.tight_layout()
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
