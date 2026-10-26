"""FrontierInsight matplotlib house style.

Experiments are LLM-authored single-file scripts that run in an isolated
venv and call matplotlib with whatever defaults the model picks — so every
figure came out looking like raw matplotlib (DejaVu font, the blue/orange
cycle, boxed axes). This module gives the engine a *model-independent* way
to brand every figure: it emits a guarded ``sitecustomize.py`` bootstrap
that the execute node drops on the experiment subprocess's ``PYTHONPATH``.
Python imports ``sitecustomize`` at interpreter startup — before the script
imports matplotlib — so the house rcParams + a branded teal colormap are in
place no matter what the generated code does, with zero edits to that code.

The figure backdrop tracks the chosen ``output.paper_style`` so figures
blend into the paper they land in: warm cream for the briefing style, white
for the LaTeX style.
"""
from __future__ import annotations

from pathlib import Path

# Brand palette (mirrors templates/paper/_html/briefing.css).
INK = "#16222b"
PAPER_CREAM = "#f8f7f3"
MUTED = "#65727c"
HAIR = "#e4e2da"
ACCENT = "#0e6e6b"

# Desaturated, brand-anchored qualitative cycle (teal → terracotta → slate →
# olive → plum → gold → …). Reads as "designed", not the default tab10.
FI_COLOR_CYCLE = [
    "#0e6e6b", "#c2693c", "#36618e", "#8a8d3a",
    "#9b4d6b", "#c79a3e", "#5f8f8b", "#a85440",
]

# Light→dark teal sequential ramp for heatmaps/imshow. Monotonic in
# lightness so it stays readable; anchored on the brand teal.
FI_TEAL_STOPS = [
    "#f6f4ee", "#bcd9d5", "#6fb3ae", "#2e8b87",
    "#0e6e6b", "#0a4f4d", "#063634",
]
FI_TEAL_CMAP_NAME = "fi_teal"

# Name of the dir the engine drops the bootstrap into, under <quest>/.fi/.
BOOT_DIRNAME = "plotstyle"

# The size (pt) the house style first set its text at, and the factor it is
# enlarged by now. A figure is shown at a fraction of the width it was drawn
# at: the experiments draw 8-13 in wide (the house canvas is 7 in), a slide's
# side pane is 5.6 in, so ticks that were 10 pt came out at 2.9-6.9 pt (median
# 4.6) on the slides of six stored quests, and 3.3-7.3 pt in their papers.
# Larger text in the same canvas fixes both without touching the canvas or the
# dpi. Measured on those quests at 1.5x: the papers keep their page counts, the
# recorder's layout check stays silent, and clipping only starts at about 1.8x
# (a y label). An explicit ``fontsize=`` in a script (a legend's) is not scaled.
FONT_SCALE = 1.5
_TEXT_PT = {
    "font.size": 11.5,
    "axes.titlesize": 13.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
}
# The tick label size of a figure drawn in the house style, in pt. The slide
# check falls back to it for a figure whose record holds no tick size.
HOUSE_TICK_PT = round(_TEXT_PT["xtick.labelsize"] * FONT_SCALE, 2)


def _text_pt(key: str) -> float:
    return round(_TEXT_PT[key] * FONT_SCALE, 2)


# The paper figure profile. The slide profile above draws text large on the
# canvas the script picks, and a figure 14 in wide shrunk into a 6.3 in column
# printed its text at about twice the body size, under a caption that looked
# tiny beside it. A paper figure is drawn at the width it prints at, so a point
# on the canvas is a point on the page: the bootstrap saves a second copy of
# every figure under figures/paper/ (a vector PDF and a PNG), with the canvas
# set to one of two widths of a one-column page, text at the body's size, no
# title inside the plot (the caption carries it) and a constrained layout in
# place of a tight crop (a tight crop changes the size after the fact).
PAPER_DIRNAME = "paper"
#: Width of the text block of a one-column article page, in inches.
PAPER_FULL_WIDTH_IN = 6.3
#: Half of it, less a gap: a single panel prints at this width.
PAPER_HALF_WIDTH_IN = 3.1
#: Text sizes of a paper figure, in points as printed.
PAPER_TICK_PT = 8.0
PAPER_LEGEND_PT = 8.0
PAPER_LABEL_PT = 9.0
PAPER_TITLE_PT = 9.0
#: Height of a row of panels, in inches: at least this much, so 8 pt ticks and
#: a 9 pt label leave room for the data, and at most this much for one row, so
#: a figure with its caption shares a page with text (the paper template caps a
#: figure at a third of the text height).
PAPER_ROW_MIN_IN = 2.0
PAPER_ROW_MAX_IN = 2.9
#: The whole figure, a legend under it included, is at most this tall (or a row
#: of panels at their least, when that is taller): the copy is placed at its own
#: width, which bypasses the template's own cap of a third of the text height.
PAPER_MAX_HEIGHT_IN = 3.2
#: Each panel of a paper copy keeps at least this share of the size it had in the figure as drawn, the whole figure
#: taken at the copy's width (a panel half as wide or half as tall as that is squeezed). A layout that leaves less is
#: not kept.
PAPER_PANEL_MIN_SHARE = 0.5
#: A single panel at least this many times as wide as it is tall (a time series)
#: prints at the full width.
PAPER_WIDE_ASPECT = 2.0
PAPER_DPI = 300
#: A scatter (or mesh) of more points than this is set as an image inside the paper copy's vector PDF.
PAPER_RASTER_OVER = 5000


def paper_profile() -> dict[str, object]:
    """The paper figure profile as plain values, for the bootstrap to embed."""
    return {
        "dirname": PAPER_DIRNAME,
        "full_in": PAPER_FULL_WIDTH_IN, "half_in": PAPER_HALF_WIDTH_IN,
        "tick_pt": PAPER_TICK_PT, "legend_pt": PAPER_LEGEND_PT,
        "label_pt": PAPER_LABEL_PT, "title_pt": PAPER_TITLE_PT,
        # Any other text (an annotation, a colorbar's tick text set by hand) is
        # scaled as the ticks are and kept between these sizes.
        "other_scale": round(PAPER_TICK_PT / HOUSE_TICK_PT, 4), "other_min_pt": 7.0, "other_max_pt": 9.0,
        "row_min_in": PAPER_ROW_MIN_IN, "row_max_in": PAPER_ROW_MAX_IN, "max_height_in": PAPER_MAX_HEIGHT_IN,
        "wide_aspect": PAPER_WIDE_ASPECT, "dpi": PAPER_DPI, "panel_min_share": PAPER_PANEL_MIN_SHARE,
        # A collection of more points or cells than this is an image inside the vector PDF.
        "raster_over": PAPER_RASTER_OVER,
        # A copy whose first layout failed tries another only within this many seconds of starting.
        "budget_s": 20.0,
    }


def paper_figure_size(drawn_in, rows=1, cols=1, profile=None):
    """``(width, height, "full" | "half")`` in inches of the paper copy of a figure the script drew at ``drawn_in``
    with ``rows`` x ``cols`` panels: two or more panels side by side, or one wide panel, take the full width, one panel
    half of it; the height keeps the drawn aspect within the profile's row heights. The bootstrap embeds this very
    function (its source), so the rule is written once; it therefore uses nothing but its arguments."""
    p = profile or paper_profile()
    w0, h0 = (float(v) for v in drawn_in)
    full = cols >= 2 or (w0 > 0 and h0 > 0 and w0 / h0 >= p["wide_aspect"])
    width = p["full_in"] if full else p["half_in"]
    height = width * h0 / w0 if w0 > 0 and h0 > 0 else width * 0.62
    rows = max(1, int(rows))
    height = min(max(height, rows * p["row_min_in"]), rows * p["row_max_in"])
    height = min(height, max(p["max_height_in"], rows * p["row_min_in"]))
    return width, round(height, 4), "full" if full else "half"


def remove_paper_copies(figures_dir: Path, name: str) -> None:
    """Remove the paper copy of figure ``name`` (``figures/paper/<stem>.pdf`` and ``.png``): a figure FI takes out of
    ``figures/`` must not reach ``paper.pdf`` through its copy. Never raises."""
    for suffix in (".pdf", ".png"):
        try:
            (Path(figures_dir) / PAPER_DIRNAME / f"{Path(name).stem}{suffix}").unlink(missing_ok=True)
        except OSError:
            pass


def figure_facecolor(paper_style: str) -> str:
    """Backdrop for figures, keyed to the paper they'll be embedded in."""
    return PAPER_CREAM if str(paper_style).lower() == "briefing" else "white"


def fi_rcparams(paper_style: str, target: str = "slide") -> dict[str, object]:
    """The house rcParams. ``paper_style`` only changes the face colors.

    ``target="slide"`` (the default) is the profile every experiment draws in: large text, for slides and posters.
    ``target="paper"`` is the paper figure profile: the canvas at the page's text width, text at the body's size,
    a constrained layout and no tight crop, text kept as text in a PDF."""
    rc = _slide_rcparams(paper_style)
    if target == "paper":
        rc.update({
            "figure.figsize": [PAPER_FULL_WIDTH_IN, round(PAPER_FULL_WIDTH_IN * 0.4, 2)],
            "figure.constrained_layout.use": True,
            "savefig.bbox": "standard",
            "savefig.dpi": PAPER_DPI,
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "font.size": PAPER_LABEL_PT,
            "axes.titlesize": PAPER_TITLE_PT,
            "axes.titlepad": 4,
            "axes.labelsize": PAPER_LABEL_PT,
            "axes.labelpad": 3,
            "xtick.labelsize": PAPER_TICK_PT,
            "ytick.labelsize": PAPER_TICK_PT,
            "xtick.major.pad": 3,
            "ytick.major.pad": 3,
            "legend.fontsize": PAPER_LEGEND_PT,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
        })
    return rc


def _slide_rcparams(paper_style: str) -> dict[str, object]:
    face = figure_facecolor(paper_style)
    return {
        # figure / output
        "figure.figsize": [7.0, 4.3],
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "figure.facecolor": face,
        "savefig.facecolor": face,
        "axes.facecolor": face,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.10,
        # typography
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans",
        ],
        "font.size": _text_pt("font.size"),
        "axes.titlesize": _text_pt("axes.titlesize"),
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 11,
        "axes.labelsize": _text_pt("axes.labelsize"),
        "axes.labelpad": 6,
        "xtick.labelsize": _text_pt("xtick.labelsize"),
        "ytick.labelsize": _text_pt("ytick.labelsize"),
        "legend.fontsize": _text_pt("legend.fontsize"),
        # ink
        "text.color": INK,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#c7c4ba",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        # despined, hairline y-grid
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": HAIR,
        "grid.linewidth": 0.9,
        # ticks: labels only, no marks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        # lines / markers
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
        "lines.solid_capstyle": "round",
        # frameless legend
        "legend.frameon": False,
        "legend.borderaxespad": 0.5,
        "legend.handlelength": 1.6,
        # branded heatmap default
        "image.cmap": FI_TEAL_CMAP_NAME,
    }


def apply_style(paper_style: str = "briefing", target: str = "slide") -> None:
    """Apply the house style to the live matplotlib in THIS process: register
    the teal colormap and push the rcParams. Imports matplotlib lazily and
    never imports pyplot, so it won't lock a GUI backend before the caller
    selects Agg. Used by the in-process demo/tests; the bootstrap file emits
    the same logic for the experiment subprocess."""
    import matplotlib
    from cycler import cycler
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(FI_TEAL_CMAP_NAME, FI_TEAL_STOPS)
    try:
        matplotlib.colormaps.register(cmap, force=True)            # mpl >= 3.6
    except (AttributeError, ValueError):
        try:
            matplotlib.cm.register_cmap(name=FI_TEAL_CMAP_NAME, cmap=cmap)
        except Exception:
            pass
    matplotlib.rcParams.update(fi_rcparams(paper_style, target))
    # prop_cycle needs a cycler object, so it can't live in the JSON rc dict.
    matplotlib.rcParams["axes.prop_cycle"] = cycler(color=FI_COLOR_CYCLE)


def _boot_source(paper_style: str) -> str:
    """Render the self-contained ``sitecustomize.py`` text. Everything is
    inlined (the venv can't import this package) and fully guarded — any
    failure is swallowed so it can never break the experiment or pip."""
    # repr(), NOT json.dumps — the output is embedded in a .py file, so the
    # dict needs Python literals (True/False/None), not JSON (true/false/null)
    # which would NameError at runtime and get swallowed by the guard.
    rc = repr(fi_rcparams(paper_style))
    stops = repr(FI_TEAL_STOPS)
    cycle = repr(FI_COLOR_CYCLE)
    return f'''\
# Auto-generated by FrontierInsight (core/plot_style.py). Applies the house
# matplotlib style to this experiment's figures. Safe no-op if anything is
# missing — never raises into the experiment or pip.
try:
    import matplotlib
    from cycler import cycler
    from matplotlib.colors import LinearSegmentedColormap

    _cmap = LinearSegmentedColormap.from_list({FI_TEAL_CMAP_NAME!r}, {stops})
    try:
        matplotlib.colormaps.register(_cmap, force=True)
    except (AttributeError, ValueError):
        try:
            matplotlib.cm.register_cmap(name={FI_TEAL_CMAP_NAME!r}, cmap=_cmap)
        except Exception:
            pass
    matplotlib.rcParams.update({rc})
    matplotlib.rcParams["axes.prop_cycle"] = cycler(color={cycle})
except Exception:
    pass
''' + _RECORDER_SOURCE.replace("@@PAPER_SOURCE@@", _boot_paper_source())


# Name of the dir, under <quest>/.fi/, where the bootstrap records what each
# saved figure draws; the engine passes it in FI_FIGURE_RECORDS.
RECORDS_DIRNAME = "figure_records"

# Appended to the bootstrap. A figure of the validation quest was captioned
# "energy drift rates for RK4 and Velocity-Verlet" while both lines lay flat at
# 0 under forward Euler's 0.44 J spike: the writer knew only the file name.
# On each savefig this writes <stem>.json with every Axes' title, labels, y
# scale and limits, and each labelled series' range and whether it shows on
# that axis ("yes", "flat" under 1% of the axis span, or "outside the axis").
# Its "layout" lists what the saved canvas draws over what a reader needs: a
# suptitle over a panel title (a real quest's three-panel figure), or a legend
# over its data (a real quest's legend sat on the line it named). Both are
# measured on the canvas after the figure is saved, since only the drawn
# positions can say so; a record without "layout" was not measured, and one
# with an empty list was measured and is clean. Its "tick_pt" is the smallest
# tick label size in points (null: no tick labels), which the slide check
# multiplies by how much smaller a slide draws the figure than it was drawn.
# Beside it, <stem>.seed<k>.json keeps the run's lines themselves (points and
# style, per panel, and whether the panel holds anything but lines and error
# bars), which the engine uses to redraw a figure as the mean over the seeds.
_RECORDER_SOURCE = '''\
try:
    import os as _fi_os

    _fi_records = _fi_os.environ.get("FI_FIGURE_RECORDS")
    # The paper copy of each figure is made with or without the records;
    # FI_PAPER_FIGURES=0 turns it off.
    if _fi_records or _fi_os.environ.get("FI_PAPER_FIGURES", "1") != "0":
        import json as _fi_json
        import math as _fi_math
        import time as _fi_time
        import warnings as _fi_warnings
        import numpy as _fi_np
        from matplotlib.collections import LineCollection as _FiLineCollection
        from matplotlib.collections import PathCollection as _FiPathCollection
        from matplotlib.collections import PolyCollection as _FiPolyCollection
        from matplotlib.figure import Figure as _FiFigure

        def _fi_values(values):
            out = []
            for value in values:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if _fi_math.isfinite(value):
                    out.append(value)
            return out

        def _fi_series(label, ys, lo, hi, log):
            if log:
                ys = [y for y in ys if y > 0]
            if not ys:
                return {"label": label, "shows": "no points"}
            low, high = min(ys), max(ys)
            bottom, top = min(lo, hi), max(lo, hi)
            if high < bottom or low > top:
                return {"label": label, "min": low, "max": high, "shows": "outside the axis"}
            if log and bottom <= 0:
                return {"label": label, "min": low, "max": high, "shows": "yes"}
            pos = _fi_math.log10 if log else float
            span = abs(pos(top) - pos(bottom))
            own = abs(pos(min(high, top)) - pos(max(low, bottom)))
            # One point is a marker, not a line lying flat.
            flat = len(ys) > 1 and span > 0 and own < 0.01 * span
            return {"label": label, "min": low, "max": high, "shows": "flat" if flat else "yes"}

        def _fi_style(artist):
            try:
                if hasattr(artist, "get_marker"):
                    return ("line", str(artist.get_color()), str(artist.get_linestyle()), str(artist.get_marker()))
                return ("points", str([round(float(v), 3) for v in artist.get_facecolor()[0]]))
            except Exception:
                return None

        def _fi_marker(artist):
            try:
                if hasattr(artist, "get_marker"):
                    return ("line", str(artist.get_marker()))
                path = artist.get_paths()[0]
                return ("points", tuple(round(float(v), 2) for v in path.vertices.flatten()))
            except Exception:
                return None

        def _fi_legend_names(ax, artists):
            # A series named only in ax.legend(["a", "b"]) keeps its "_child" label.
            # The legend draws each entry in its series' style, which pairs the two.
            legend = ax.get_legend()
            if legend is None:
                return {}
            handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", None) or []
            labelled = {str(a.get_label() or "") for a in artists}
            names = {}
            for handle, text in zip(handles, legend.get_texts()):
                style, name = _fi_style(handle), text.get_text()
                if style is None or not name or name in labelled:
                    continue
                for artist in artists:
                    if (id(artist) not in names and str(artist.get_label() or "").startswith("_")
                            and _fi_style(artist) == style):
                        names[id(artist)] = name
                        break
            return names

        def _fi_error_bars(ax):
            # ax.errorbar draws a data line, its bars (line collections) and its
            # caps (lines), all labelled "_nolegend_"; the label sits on their
            # container. The data line is the series; the bars and caps are one
            # run's error, which a redraw over the seeds replaces with their
            # interval. Bars with no data line, or across x, are more than that.
            names, pieces = {}, set()
            try:
                from matplotlib.container import ErrorbarContainer

                for container in ax.containers:
                    if not isinstance(container, ErrorbarContainer):
                        continue
                    data_line, caps, bars = container.lines
                    if data_line is None or container.has_xerr:
                        continue
                    label = container.get_label()
                    if label and not str(label).startswith("_"):
                        names[id(data_line)] = str(label)
                    pieces.update(id(artist) for artist in (*caps, *bars))
            except Exception:
                return {}, set()
            return names, pieces

        def _fi_collection_ys(ax, artist):
            # The y values a collection draws, in data units: a scatter's points, the
            # ends of the segments of hlines / vlines, the corners of a fill_between
            # band. Only a scatter has offsets that say where it is: every other
            # collection reports [0, 0], which recorded an hlines drawn at 0.94 as a
            # series at 0. None for a collection read no such way (a mesh, a contour,
            # one drawn in axes coordinates), which is left out rather than recorded.
            if isinstance(artist, _FiPathCollection):
                return _fi_values(point[1] for point in artist.get_offsets())
            if artist.get_transform() != ax.transData:
                return None
            if isinstance(artist, _FiLineCollection):
                return _fi_values(point[1] for segment in artist.get_segments() for point in segment)
            if isinstance(artist, _FiPolyCollection):
                return _fi_values(point[1] for path in artist.get_paths() for point in path.vertices)
            return None

        def _fi_axes(ax):
            lo, hi = ax.get_ylim()
            log = ax.get_yscale() == "log"
            bar_names, bar_pieces = _fi_error_bars(ax)
            artists = [a for a in list(ax.get_lines()) + list(ax.collections) if id(a) not in bar_pieces]
            try:
                names = _fi_legend_names(ax, artists)
            except Exception:
                names = {}
            names.update(bar_names)
            # A series can be drawn by several calls: a scatter per point, with
            # the label on the first call only. Unlabelled markers join the
            # labelled series drawn in the same style, as the legend pairs them;
            # failing that, the one labelled series with the same marker shape,
            # since an uncoloured call takes the next colour in the cycle.
            # Otherwise that series' first point alone reads as "flat".
            groups, unlabelled, verticals = [], [], []
            for artist in artists:
                label = names.get(id(artist)) or str(artist.get_label() or "")
                try:
                    if hasattr(artist, "get_ydata"):
                        kind = _fi_kind(ax, artist)
                        if kind == "vline":
                            # An axvline's y is [0, 1] in axes coordinates, not data: read as data
                            # it was a series from 0 to 1, "flat" on any axis reaching past 100, and
                            # a caption was then told to call a vertical line horizontal.
                            xs = _fi_values(artist.get_xdata())
                            if xs and label and not label.startswith("_"):
                                verticals.append({"label": label, "x": xs[0], "shows": "vertical line"})
                            continue
                        if kind == "hline":
                            # An axhline is a reference drawn flat on purpose (a threshold, a known
                            # value): a series lying flat is a finding, this is not one.
                            ys = _fi_values(artist.get_ydata())
                            if ys and label and not label.startswith("_"):
                                verticals.append({"label": label, "y": ys[0], "shows": "reference line"})
                            continue
                        if kind == "other":
                            continue  # drawn in coordinates that are not the data's: nothing to read
                        ys = _fi_values(artist.get_ydata())
                    else:
                        ys = _fi_collection_ys(ax, artist)
                except Exception:
                    continue
                if ys is None:
                    continue
                style, marker = _fi_style(artist), _fi_marker(artist)
                if label and not label.startswith("_"):
                    groups.append((label, style, marker, ys))
                elif style is not None and (style[0] == "points" or style[3] not in ("None", "", " ")):
                    # Only markers join a series drawn by several calls: a band or a
                    # segment collection is not one of its points.
                    if hasattr(artist, "get_ydata") or isinstance(artist, _FiPathCollection):
                        unlabelled.append((style, marker, ys))
                elif style is not None and style[0] == "line":
                    # A family of runs drawn in a loop, the label on the first only: the
                    # others join it when drawn in exactly its style, so the family's range
                    # is recorded, not the first run's (a real figure's labelled run lay
                    # flat at 1.0 while the others rose and fell).
                    unlabelled.append((style, None, ys))
            for style, marker, ys in unlabelled:
                same_style = [g for g in groups if g[1] == style]
                same_marker = [g for g in groups if marker is not None and g[2] == marker]
                if marker is None and not same_style:
                    continue  # a plain line joins only a series in exactly its style
                target = same_style[0] if same_style else same_marker[0] if len(same_marker) == 1 else None
                if target is not None:
                    target[3].extend(ys)
            series = [
                {"label": label, "y": ys[0], "shows": "reference line"}
                if style and style[0] == "line" and style[2] in ("--", ":", "-.", "dashed", "dotted", "dashdot")
                and len(ys) > 1 and max(ys) == min(ys)
                else _fi_series(label, ys, lo, hi, log)
                for label, style, _marker, ys in groups
            ] + verticals  # a dashed line at one value is a reference drawn flat on purpose, not a flat series
            # The house style puts titles on the left, where get_title() alone misses them.
            title = " ".join(t for t in (ax.get_title("left"), ax.get_title(), ax.get_title("right")) if t)
            return {
                "title": title, "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
                "yscale": ax.get_yscale(), "ylim": [float(lo), float(hi)], "series": series,
            }

        # Tick labels a redraw gets back by setting the same scale.
        _FI_DEFAULT_FORMATTERS = {
            "ScalarFormatter", "LogFormatter", "LogFormatterSciNotation", "LogFormatterMathtext",
        }

        def _fi_kind(ax, line):
            transform = line.get_transform()
            if transform == ax.transData:
                return "data"
            if transform == ax.get_yaxis_transform():
                return "hline"
            if transform == ax.get_xaxis_transform():
                return "vline"
            return "other"

        def _fi_points(line):
            # Paired, so x and y stay aligned; None when an x is not a number.
            xs, ys = list(line.get_xdata()), list(line.get_ydata())
            if len(xs) != len(ys) or len(xs) > 20000:
                return None, None
            px, py = [], []
            for x, y in zip(xs, ys):
                try:
                    x, y = float(x), float(y)
                except (TypeError, ValueError):
                    return None, None
                if _fi_math.isfinite(x) and _fi_math.isfinite(y):
                    px.append(x)
                    py.append(y)
            return px, py

        def _fi_colour(value):
            from matplotlib.colors import to_hex

            return "none" if str(value).lower() == "none" else to_hex(value)

        def _fi_missing(line):
            # The x values at which the line has no y value (None or NaN): matplotlib
            # draws nothing there, and _fi_points drops them, so without this the
            # record could not tell a setting never plotted from one with no value.
            try:
                import warnings

                xs, ys = list(line.get_xdata()), list(line.get_ydata())
                if len(xs) != len(ys) or len(xs) > 20000:
                    return []
                out = []
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for x, y in zip(xs, ys):
                        try:
                            fx = float(x)
                        except (TypeError, ValueError):
                            continue
                        try:
                            fy = float(y)
                        except (TypeError, ValueError):
                            fy = float("nan")
                        if _fi_math.isfinite(fx) and not _fi_math.isfinite(fy):
                            out.append(fx)
                return out[:50]
            except Exception:
                return []

        def _fi_line(ax, line, names):
            x, y = _fi_points(line)
            return {
                "missing_x": _fi_missing(line),
                "label": names.get(id(line)) or str(line.get_label() or ""),
                "kind": _fi_kind(ax, line), "x": x, "y": y,
                "color": _fi_colour(line.get_color()), "linestyle": line.get_linestyle(),
                "marker": str(line.get_marker()), "markersize": float(line.get_markersize()),
                "markerfacecolor": _fi_colour(line.get_markerfacecolor()),
                "linewidth": float(line.get_linewidth()), "alpha": line.get_alpha(),
                "drawstyle": line.get_drawstyle(),
            }

        def _fi_draws(artist):
            # A collection with no paths draws nothing: seaborn leaves an empty
            # confidence band per line when each x holds one observation.
            try:
                return not hasattr(artist, "get_paths") or bool(len(artist.get_paths()))
            except Exception:
                return True

        def _fi_where(ax):
            # Where the panel sits on the figure: its cell of the subplot grid, and its box in figure coordinates (twin
            # axes share it, also when made with add_axes, which has no grid).
            spec = ax.get_subplotspec()
            grid = None
            if spec is not None:
                rows, cols = spec.get_gridspec().get_geometry()
                grid = [rows, cols, spec.rowspan.start, spec.rowspan.stop, spec.colspan.start, spec.colspan.stop]
            try:
                place = [round(float(v), 4) for v in ax.get_position().bounds]
            except Exception:
                place = None
            return {"grid": grid, "place": place}

        def _fi_panel(ax):
            bar_names, bar_pieces = _fi_error_bars(ax)
            try:
                names = _fi_legend_names(ax, list(ax.get_lines()) + list(ax.collections))
            except Exception:
                names = {}
            names.update(bar_names)
            lines = [_fi_line(ax, line, names) for line in ax.get_lines() if id(line) not in bar_pieces]
            formatters = {type(axis.get_major_formatter()).__name__ for axis in (ax.xaxis, ax.yaxis)}
            others = (ax.patches, ax.collections, ax.images, ax.texts, ax.tables, ax.artists, ax.child_axes)
            return {
                **_fi_where(ax),
                "title": " ".join(t for t in (ax.get_title("left"), ax.get_title(), ax.get_title("right")) if t),
                "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
                "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
                "legend": ax.get_legend() is not None,
                # Whether the panel draws anything but lines (bars, an image, a patch, a collection that has paths): a panel
                # with no lines and nothing else is empty.
                "draws_other": any(_fi_draws(artist) for group in (ax.patches, ax.collections, ax.images, ax.tables,
                                                                   ax.artists, ax.child_axes)
                                   for artist in group if id(artist) not in bar_pieces),
                # Only lines (error bars included), on default ticks: what a redraw can reproduce.
                "line_only": bool(lines) and not any(
                    _fi_draws(artist) for group in others for artist in group if id(artist) not in bar_pieces
                )
                and formatters <= _FI_DEFAULT_FORMATTERS
                and all(line["kind"] != "other" and line["x"] is not None for line in lines),
                "lines": lines,
            }

        def _fi_tick_pt(fig):
            # The smallest size (pt) of a tick label the saved figure draws, or None
            # when it draws none. Read from the labels themselves, so a script's own
            # tick_params(labelsize=...) counts, not only the house style's size.
            sizes = []
            for ax in fig.get_axes():
                for axis in (ax.xaxis, ax.yaxis):
                    for label in axis.get_ticklabels():
                        if label.get_visible() and label.get_text().strip():
                            sizes.append(float(label.get_fontsize()))
            return min(sizes) if sizes else None

        def _fi_plot_data(fig, name):
            suptitle = getattr(fig, "_suptitle", None)
            return {
                "file": name, "size": [float(v) for v in fig.get_size_inches()],
                "suptitle": suptitle.get_text() if suptitle is not None else "",
                "axes": [_fi_panel(ax) for ax in fig.get_axes()],
            }

        # A suptitle covering at least this share of a panel title is drawn over it.
        _FI_TITLE_SHARE = 0.2
        # A legend's box is measured this many pixels in from its frame, so a line
        # that only grazes the frame is not "under" it.
        _FI_LEGEND_INSET = 3

        def _fi_share(a, b):
            # The part of the smaller of two boxes that the other one covers, 0 to 1.
            x0, y0, x1, y1 = max(a.x0, b.x0), max(a.y0, b.y0), min(a.x1, b.x1), min(a.y1, b.y1)
            smaller = min(a.width * a.height, b.width * b.height)
            if x1 <= x0 or y1 <= y0 or smaller <= 0:
                return 0.0
            return (x1 - x0) * (y1 - y0) / smaller

        def _fi_titles(ax):
            # The house style puts titles on the left, so all three are read.
            texts = (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None))
            return [t for t in texts if t is not None and t.get_visible() and t.get_text().strip()]

        def _fi_length_inside(a, b, box):
            # Total length of the segments a[i] -> b[i] that lie inside box (pixels):
            # Liang-Barsky, over every segment at once. NaN ends draw nothing.
            np = _fi_np
            x0, y0, x1, y1 = box
            d = b - a
            t0, t1 = np.zeros(len(a)), np.ones(len(a))
            with np.errstate(divide="ignore", invalid="ignore"):
                for p, q in ((-d[:, 0], a[:, 0] - x0), (d[:, 0], x1 - a[:, 0]),
                             (-d[:, 1], a[:, 1] - y0), (d[:, 1], y1 - a[:, 1])):
                    r = q / p
                    t0 = np.where(p < 0, np.maximum(t0, r), t0)
                    t1 = np.where(p > 0, np.minimum(t1, r), t1)
                    t1 = np.where((p == 0) & (q < 0), -1.0, t1)
                inside = np.clip(t1 - t0, 0.0, None) * np.hypot(d[:, 0], d[:, 1])
            return float(np.nansum(np.where(np.isfinite(inside), inside, 0.0)))

        def _fi_under(ax, artist, box, axes_box):
            # What a data line or a scatter has inside box, on the canvas:
            # (the length of its line in pixels, the number of its markers or points).
            np = _fi_np
            if not artist.get_visible():
                return 0.0, 0
            if artist.get_clip_on():
                # Outside the axes a line is clipped, so nothing of it is drawn there.
                box = (max(box[0], axes_box.x0), max(box[1], axes_box.y0),
                       min(box[2], axes_box.x1), min(box[3], axes_box.y1))
                if box[2] <= box[0] or box[3] <= box[1]:
                    return 0.0, 0
            if hasattr(artist, "get_xydata"):
                if artist.get_transform() != ax.transData:
                    return 0.0, 0
                xy = np.asarray(artist.get_xydata(), dtype=float)
                if len(xy) > 20000:
                    xy = xy[:: -(-len(xy) // 20000)]
                if len(xy) == 0:
                    return 0.0, 0
                pts = ax.transData.transform(xy)
                joined = str(artist.get_linestyle()) not in ("None", "none", "", " ")
                marked = str(artist.get_marker()) not in ("None", "none", "", " ")
                length = _fi_length_inside(pts[:-1], pts[1:], box) if joined and len(pts) > 1 else 0.0
                hit = (pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])
                return length, int(hit.sum()) if (marked or len(pts) == 1) else 0
            if isinstance(artist, _FiPathCollection):
                offsets = np.ma.filled(np.ma.asarray(artist.get_offsets(), dtype=float), np.nan)
                if len(offsets) == 0:
                    return 0.0, 0
                pts = artist.get_offset_transform().transform(offsets)
                hit = (pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])
                return 0.0, int(hit.sum())
            return 0.0, 0

        def _fi_legend_hits(ax, legend, renderer):
            # (the length in pixels of the lines, the number of the points) of ax drawn
            # under its legend, measured on the drawn canvas.
            lb = legend.get_window_extent(renderer)
            inset = _FI_LEGEND_INSET
            if lb.width <= 2 * inset or lb.height <= 2 * inset:
                return 0.0, 0
            inner = (lb.x0 + inset, lb.y0 + inset, lb.x1 - inset, lb.y1 - inset)
            axes_box = ax.get_window_extent(renderer)
            line_px, points = 0.0, 0
            for artist in list(ax.get_lines()) + list(ax.collections):
                try:
                    length, count = _fi_under(ax, artist, inner, axes_box)
                except Exception:
                    continue
                line_px += length
                points += count
            return line_px, points

        def _fi_axis_texts(ax, axis):
            # The tick labels of one axis that are drawn: only the ticks inside the
            # visible range (matplotlib keeps spare tick objects beyond it), and only
            # the labels that show text.
            lo, hi = sorted(axis.get_view_interval())
            texts = []
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi:
                    for label in (tick.label1, tick.label2):
                        if label.get_visible() and label.get_text().strip():
                            texts.append(label)
            return texts

        def _fi_label_findings(fig, renderer, cropped):
            # Tick labels run into each other, an axis label is drawn over the tick
            # labels, or a label runs past the edge of the figure. Measured on the drawn
            # canvas, with the same share as a title: a little touching is not a clash.
            # A label past the canvas edge only counts when the saved picture is that
            # canvas: a tight crop widens to keep every label (see _fi_overflow_box).
            found = []
            figure_box = fig.bbox
            for ax in fig.get_axes():
                for axis_name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
                    ticks = _fi_axis_texts(ax, axis)
                    boxes = [(t, t.get_window_extent(renderer)) for t in ticks]
                    for i, (first, a) in enumerate(boxes):
                        for second, b in boxes[i + 1:]:
                            share = _fi_share(a, b)
                            if share >= _FI_TITLE_SHARE:
                                found.append({"check": "tick_label_overlap", "axis": axis_name,
                                              "text": first.get_text(), "over": second.get_text(),
                                              "share": round(share, 2)})
                    label = axis.label
                    if not label.get_visible() or not label.get_text().strip():
                        continue
                    label_box = label.get_window_extent(renderer)
                    all_ticks = _fi_axis_texts(ax, ax.xaxis) + _fi_axis_texts(ax, ax.yaxis)
                    for tick in all_ticks:
                        share = _fi_share(label_box, tick.get_window_extent(renderer))
                        if share >= _FI_TITLE_SHARE:
                            found.append({"check": "axis_label_overlap", "axis": axis_name,
                                          "text": label.get_text(), "over": tick.get_text(),
                                          "share": round(share, 2)})
            # A label past the canvas edge, a pixel of rounding allowed.
            edge = 1.0
            for ax in ([] if cropped else fig.get_axes()):
                texts = [ax.xaxis.label, ax.yaxis.label] + _fi_axis_texts(ax, ax.xaxis) + _fi_axis_texts(ax, ax.yaxis)
                for text in texts:
                    if not text.get_visible() or not text.get_text().strip():
                        continue
                    box = text.get_window_extent(renderer)
                    if box.width <= 0 and box.height <= 0:
                        continue
                    if (box.x0 < figure_box.x0 - edge or box.x1 > figure_box.x1 + edge
                            or box.y0 < figure_box.y0 - edge or box.y1 > figure_box.y1 + edge):
                        found.append({"check": "label_clipped", "text": text.get_text()})
            return found

        def _fi_layout(fig, cropped=False):
            # Text drawn over text, and a legend drawn over the data, measured on the
            # canvas the figure was saved from. None when it could not be measured.
            found = []
            try:
                try:
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                except Exception:
                    renderer = fig._get_renderer()
                suptitle = getattr(fig, "_suptitle", None)
                if suptitle is not None and suptitle.get_visible() and suptitle.get_text().strip():
                    box = suptitle.get_window_extent(renderer)
                    for ax in fig.get_axes():
                        for text in _fi_titles(ax):
                            share = _fi_share(box, text.get_window_extent(renderer))
                            if share >= _FI_TITLE_SHARE:
                                found.append({"check": "title_overlap", "text": suptitle.get_text(),
                                              "over": text.get_text(), "share": round(share, 2)})
                for ax in fig.get_axes():
                    legend = ax.get_legend()
                    if legend is None or not legend.get_visible():
                        continue
                    line_px, points = _fi_legend_hits(ax, legend, renderer)
                    if line_px > 0 or points > 0:
                        found.append({"check": "legend_over_data",
                                      "axes_title": " ".join(t.get_text() for t in _fi_titles(ax)),
                                      "legend": [t.get_text() for t in legend.get_texts()],
                                      "line_px": round(line_px, 1), "points": points})
            except Exception:
                return None
            # The label checks have their own guard: a defect in them must not drop the
            # title and legend findings above.
            try:
                found.extend(_fi_label_findings(fig, renderer, cropped))
            except Exception:
                pass
            return found

        _fi_savefig = _FiFigure.savefig

@@PAPER_SOURCE@@

        def _fi_recording_savefig(self, fname, *args, **kwargs):
            started = _fi_time.perf_counter()
            # Read before the crop below rewrites bbox_inches: a tight crop keeps every label.
            cropped = kwargs.get("bbox_inches", _fi_mpl.rcParams["savefig.bbox"]) == "tight"
            # A tight crop keeps an axis label that runs past the canvas edge: the crop is the widened box
            # (see _fi_overflow_box), since savefig's own tight box leaves that label out.
            if cropped:
                try:
                    pad = float(kwargs.get("pad_inches", _fi_mpl.rcParams["savefig.pad_inches"]))
                    box = _fi_overflow_box(self, _fi_draw(self))
                    kwargs["bbox_inches"] = _fi_mpl.transforms.Bbox.from_extents(
                        box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
                    kwargs["pad_inches"] = 0
                except Exception:
                    pass
            result = _fi_savefig(self, fname, *args, **kwargs)
            saved_in = _fi_time.perf_counter() - started
            record = plot = name = None
            if _fi_records:
                try:
                    name = _fi_os.path.basename(_fi_os.fspath(fname))
                    record = {"file": name, "axes": [{**_fi_axes(ax), **_fi_where(ax)} for ax in self.get_axes()]}
                    # A failure here leaves the key out: a record without "tick_pt" was
                    # not measured, one with null draws no tick labels.
                    try:
                        record["tick_pt"] = _fi_tick_pt(self)
                    except Exception:
                        pass
                    # Measuring draws the figure once more, so a figure that was slow to
                    # save is not measured.
                    layout = _fi_layout(self, cropped) if saved_in < 10 else None
                    if layout is not None:
                        record["layout"] = layout
                    # Each seed's lines, for the engine to redraw a figure as the
                    # mean over the seeds. A redrawn figure is no seed's. Read before
                    # the paper copy, from the figure as the script drew it.
                    if not _fi_os.environ.get("FI_REPLOT"):
                        plot = _fi_plot_data(self, name)
                except Exception:
                    record = None
            # The paper copy (figures/paper/<stem>.pdf and .png): drawn from a copy of
            # the figure, which stays as the script left it. This file name's earlier
            # copy goes first, whatever happens next: a copy that is not made again
            # (the save was slow, the copy failed) must not show an older figure.
            paper_dir = _fi_paper_dir(fname)
            if paper_dir is not None:
                try:
                    _fi_paper_clear(_fi_paper_outputs(fname, paper_dir)[1])
                    info = _fi_paper_copy(self, fname, paper_dir) if saved_in < 10 else {"skipped": "slow to save"}
                except Exception as exc:
                    info = {"error": repr(exc)[:300]}
                if record is not None:
                    record["paper"] = info
                    # Titles the paper copy does not show: the writer is told, so the caption can say it.
                    gone = ([info["suptitle_dropped"]] if info.get("suptitle_dropped") else []) + list(
                        info.get("titles_dropped") or [])
                    if "error" not in info and "skipped" not in info and gone:
                        record["paper_dropped"] = gone
            if record is None:
                return result
            try:
                _fi_os.makedirs(_fi_records, exist_ok=True)
                stem = _fi_os.path.join(_fi_records, _fi_os.path.splitext(name)[0])
                with open(stem + ".json", "w", encoding="utf-8") as handle:
                    _fi_json.dump(record, handle)
                if plot is not None:
                    # Keyed on WHICH replicate this is, not on what it seeds
                    # with. The two used to be the same small integer; the
                    # seeds now stride far apart to keep the replicates' random
                    # streams disjoint, while the engine still looks these
                    # files up by ordinal (0, 1, 2 ...).
                    seed = "".join(c for c in _fi_os.environ.get("FI_REPLICATE_INDEX", "") if c.isdigit()) or "0"
                    with open(stem + ".seed" + seed + ".json", "w", encoding="utf-8") as handle:
                        _fi_json.dump(plot, handle)
            except Exception:
                pass
            return result

        _FiFigure.savefig = _fi_recording_savefig
except Exception:
    pass
'''


# The paper copy, inserted into the recorder above (at @@PAPER_SOURCE@@, inside
# its guarded block). A paper figure is the script's own figure drawn again under
# the paper profile, never a redraw from numbers: what it plots is what the
# script plotted. It is drawn from an independent copy of the figure (a pickle
# round trip), so the script's figure is never changed: a script that saves the
# same figure again (a PNG, then a PDF) saves it as it drew it, and nothing has
# to be undone. Only text changes size on the copy; lines, markers and a
# scatter's points are as the script drew them (a bubble's size is data).
_PAPER_SOURCE = '''\
        import copyreg as _fi_copyreg
        import io as _fi_io
        import pickle as _fi_pickle
        import types as _fi_types
        import matplotlib as _fi_mpl
        from matplotlib.legend import Legend as _FiLegend

        try:
            from matplotlib.backend_bases import _get_renderer as _fi_get_renderer
        except Exception:
            _fi_get_renderer = None

        _FI_PAPER = @@PAPER_PROFILE@@

@@PAPER_SIZE@@

        # Each legend keeps the artists it was made from, beside the texts it shows: the paper copy sets a legend
        # elsewhere with the same entries, each drawn as the legend drew it (its own drawn keys are only the first
        # piece of an error bar's).
        try:
            _fi_init_legend_box = _FiLegend._init_legend_box

            def _fi_legend_box(self, handles, labels, *args, **kwargs):
                handles, labels = list(handles), list(labels)
                result = _fi_init_legend_box(self, handles, labels, *args, **kwargs)
                try:
                    keys = list(self.legend_handles)
                    self._fi_entries = [(label, handle) for handle, label, key in zip(handles, labels, keys)
                                        if key is not None]
                except Exception:
                    pass
                return result

            _FiLegend._init_legend_box = _fi_legend_box
        except Exception:
            pass

        def _fi_paper_dir(fname):
            # The paper copy goes beside the figure, in figures/paper/, only for a
            # figure saved into a folder called "figures": FI's figure folder.
            if _fi_os.environ.get("FI_PAPER_FIGURES", "1") == "0":
                return None
            try:
                path = _fi_os.fspath(fname)
            except TypeError:
                return None
            if isinstance(path, bytes):
                path = _fi_os.fsdecode(path)
            parent = _fi_os.path.dirname(_fi_os.path.abspath(path))
            if _fi_os.path.basename(parent).lower() != "figures":
                return None
            return _fi_os.path.join(parent, _FI_PAPER["dirname"])

        def _fi_paper_outputs(fname, paper_dir):
            stem = _fi_os.path.splitext(_fi_os.path.basename(_fi_os.fsdecode(_fi_os.fspath(fname))))[0]
            return stem, [_fi_os.path.join(paper_dir, stem + ext) for ext in (".pdf", ".png")]

        def _fi_paper_clear(paths):
            for path in paths:
                try:
                    _fi_os.remove(path)
                except OSError:
                    pass

        class _FiCopier(_fi_pickle.Pickler):
            # Pickles the figure as one pyplot does not manage, so the copy is never made pyplot's current figure
            # (a later plt.savefig() of the script saves its own figure), and a plain function (a tick formatter's
            # lambda, which cannot be pickled) by reference: the copy shares it.
            def __init__(self, handle, fig, shared):
                _fi_pickle.Pickler.__init__(self, handle, protocol=_fi_pickle.HIGHEST_PROTOCOL)
                self._fi_fig = fig
                self._fi_shared = shared

            def reducer_override(self, obj):
                if obj is self._fi_fig:
                    state = obj.__getstate__()
                    state.pop("_restore_to_pylab", None)
                    return (_fi_copyreg._reconstructor, (type(obj), object, None), state)
                return NotImplemented

            def persistent_id(self, obj):
                if isinstance(obj, _fi_types.FunctionType):
                    self._fi_shared.append(obj)
                    return len(self._fi_shared) - 1
                return None

        class _FiCopyLoader(_fi_pickle.Unpickler):
            def __init__(self, handle, shared):
                _fi_pickle.Unpickler.__init__(self, handle)
                self._fi_shared = shared

            def persistent_load(self, pid):
                return self._fi_shared[pid]

        def _fi_copy_maker(fig):
            # A function that returns a new independent copy of fig at each call, with a canvas of its own. Raises
            # when the figure cannot be copied.
            shared = []
            handle = _fi_io.BytesIO()
            _FiCopier(handle, fig, shared).dump(fig)
            data = handle.getvalue()

            def make():
                from matplotlib.backends.backend_agg import FigureCanvasAgg

                copy = _FiCopyLoader(_fi_io.BytesIO(data), shared).load()
                FigureCanvasAgg(copy)
                return copy
            return make

        def _fi_spec(ax):
            # The place of an Axes in the figure's own grid: a colorbar made beside a panel puts the panel in a grid
            # of its own inside that place.
            try:
                spec = ax.get_subplotspec()
            except Exception:
                return None
            if spec is None:
                return None
            try:
                return spec.get_topmost_subplotspec()
            except Exception:
                return spec

        def _fi_panels(fig):
            # What a reader sees as panels: every visible Axes but a colorbar's.
            return [ax for ax in fig.get_axes() if ax.get_visible() and getattr(ax, "_colorbar", None) is None]

        def _fi_grid(panels):
            rows = cols = 1
            for ax in panels:
                spec = _fi_spec(ax)
                if spec is None:
                    continue
                r, c = spec.get_gridspec().get_geometry()
                rows, cols = max(rows, int(r)), max(cols, int(c))
            return rows, cols

        def _fi_place(ax):
            spec = _fi_spec(ax)
            try:
                if spec is not None:
                    return (spec.rowspan.start, spec.rowspan.stop, spec.colspan.start, spec.colspan.stop)
            except Exception:
                pass
            return tuple(round(float(v), 3) for v in ax.get_position().bounds)

        def _fi_cap(rows):
            # The tallest a paper copy is saved: the profile's height, or its rows at their least when that is more.
            p = _FI_PAPER
            return max(p["max_height_in"], max(1, int(rows)) * p["row_min_in"])

        def _fi_draw(fig):
            # Lays the figure out and places its text without painting it (a scatter of many points costs nothing
            # here); the renderer returned measures what is drawn where.
            try:
                renderer = _fi_get_renderer(fig)
                with renderer._draw_disabled():
                    fig.draw(renderer)
                return renderer
            except (AttributeError, TypeError):
                fig.canvas.draw()
                return fig.canvas.get_renderer()

        def _fi_overflow_box(fig, renderer):
            # The box a saved copy must keep, in inches. get_tightbbox leaves out an axis label that runs past
            # the canvas edge (a long x label on the right panel), so the crop would cut it; the box is widened
            # to every visible text's own extent, so a label that fits changes nothing.
            box = fig.get_tightbbox(renderer)
            x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
            dpi = float(fig.dpi)
            texts = list(fig.texts)
            for ax in _fi_all_axes(fig):
                texts += [ax.xaxis.label, ax.yaxis.label] + _fi_all_titles(ax) + list(ax.texts)
                for axis in (ax.xaxis, ax.yaxis):
                    # Only the ticks inside the axis range are drawn; matplotlib keeps spare tick objects beyond it.
                    lo, hi = sorted(axis.get_view_interval())
                    for tick in axis.get_major_ticks():
                        if lo <= tick.get_loc() <= hi:
                            texts += [tick.label1, tick.label2]
            for text in texts:
                if not text.get_visible() or not text.get_text().strip():
                    continue
                ext = text.get_window_extent(renderer)
                if ext.width <= 0 and ext.height <= 0:
                    continue
                x0 = min(x0, ext.x0 / dpi)
                y0 = min(y0, ext.y0 / dpi)
                x1 = max(x1, ext.x1 / dpi)
                y1 = max(y1, ext.y1 / dpi)
            return _fi_mpl.transforms.Bbox.from_extents(x0, y0, x1, y1)

        def _fi_all_axes(fig):
            # Every Axes, an inset's (a child of its Axes) included.
            out, seen, todo = [], set(), list(fig.get_axes())
            while todo:
                ax = todo.pop(0)
                if id(ax) in seen:
                    continue
                seen.add(id(ax))
                out.append(ax)
                todo.extend(getattr(ax, "child_axes", None) or [])
            return out

        def _fi_all_titles(ax):
            return [t for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None))
                    if t is not None]

        def _fi_legends(fig):
            out = [ax.get_legend() for ax in _fi_all_axes(fig)]
            return [leg for leg in out + list(getattr(fig, "legends", [])) if leg is not None]

        def _fi_other_pt(text):
            p = _FI_PAPER
            return min(max(float(text.get_fontsize()) * p["other_scale"], p["other_min_pt"]), p["other_max_pt"])

        def _fi_paper_text(fig):
            # Text at the body's sizes. Nothing else changes size.
            p = _FI_PAPER
            insets = {id(child) for ax in fig.get_axes() for child in (getattr(ax, "child_axes", None) or [])}
            for ax in _fi_all_axes(fig):
                inset = id(ax) in insets
                try:
                    ax.tick_params(axis="both", which="both", labelsize=p["tick_pt"])
                except Exception:
                    pass
                for axis in (ax.xaxis, ax.yaxis):
                    axis.label.set_fontsize(p["tick_pt"] if inset else p["label_pt"])
                    axis.get_offset_text().set_fontsize(p["tick_pt"])
                if inset:
                    for title in _fi_all_titles(ax):
                        title.set_fontsize(p["tick_pt"])
                for text in ax.texts:
                    text.set_fontsize(_fi_other_pt(text))
            for leg in _fi_legends(fig):
                for text in leg.get_texts():
                    text.set_fontsize(p["legend_pt"])
                leg.get_title().set_fontsize(p["legend_pt"])
            for key in ("_supxlabel", "_supylabel"):
                text = getattr(fig, key, None)
                if text is not None:
                    text.set_fontsize(p["label_pt"])
            skip = {id(getattr(fig, k, None)) for k in ("_suptitle", "_supxlabel", "_supylabel")}
            for text in fig.texts:
                if id(text) not in skip:
                    text.set_fontsize(_fi_other_pt(text))

        def _fi_rasterize_big(fig):
            # A collection of many points or cells is set as an image inside the vector PDF: drawn as vectors, a
            # scatter of 200,000 points made a 9 MB file that took a minute to save.
            out = 0
            for ax in _fi_all_axes(fig):
                for coll in ax.collections:
                    try:
                        if hasattr(coll, "get_coordinates"):
                            shape = coll.get_coordinates().shape
                            n = int(shape[0]) * int(shape[1])
                        else:
                            n = max(len(coll.get_offsets()), len(getattr(coll, "_paths", None) or []))
                    except Exception:
                        continue
                    if n > _FI_PAPER["raster_over"]:
                        coll.set_rasterized(True)
                        out += 1
            return out

        def _fi_set_loc(leg, loc):
            try:
                leg.set_loc(loc)
            except AttributeError:
                leg._loc = {"best": 0, "upper center": 9}[loc]

        def _fi_below(fig, ax, renderer):
            # Where, in axes coordinates, a legend set under the axes starts: under its
            # x tick labels and x label, 3 pt down.
            box = ax.get_window_extent(renderer)
            bottom = box.y0
            try:
                bottom = min(bottom, ax.xaxis.get_tightbbox(renderer).y0)
            except Exception:
                pass
            return (bottom - 3 * fig.dpi / 72 - box.y0) / max(box.height, 1.0)

        def _fi_legend_entries(leg):
            # (label, artist) of each entry the legend shows, in its order, a label shown twice twice: the artists
            # it was made from, else its own drawn keys.
            texts = [t.get_text() for t in leg.get_texts()]
            kept = getattr(leg, "_fi_entries", None)
            if kept is not None and len(kept) == len(texts):
                return [(text, handle) for text, (_label, handle) in zip(texts, kept)]
            keys = [k for k in (getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", None) or [])
                    if k is not None]
            return [(text, keys[i]) for i, text in enumerate(texts) if i < len(keys)]

        def _fi_key_colour(value):
            from matplotlib.colors import to_hex

            try:
                arr = _fi_np.asarray(value, dtype=float)
                if arr.ndim == 2:
                    if not len(arr):
                        return None
                    arr = arr[0]
                return to_hex(arr, keep_alpha=True)
            except Exception:
                try:
                    return to_hex(value, keep_alpha=True)
                except Exception:
                    return str(value)

        def _fi_key_style(handle):
            # How a legend key draws its series: colour, line style and width, marker and its size, and alpha (fill,
            # edge, hatch, marker shape and size for a patch or a scatter). A container (error bars, bars) is drawn as
            # its first piece.
            artist, depth = handle, 0
            while isinstance(artist, (tuple, list)) and depth < 4:
                artist = next((a for a in artist if a is not None), None)
                depth += 1
            if artist is None:
                return None
            try:
                alpha = artist.get_alpha()
                alpha = None if alpha is None else round(float(_fi_np.ravel(alpha)[0]), 3)
                if hasattr(artist, "get_marker"):
                    return ("line", _fi_key_colour(artist.get_color()), str(artist.get_linestyle()),
                            round(float(artist.get_linewidth()), 3), str(artist.get_marker()),
                            round(float(artist.get_markersize()), 3), _fi_key_colour(artist.get_markerfacecolor()),
                            alpha)
                if hasattr(artist, "get_facecolor"):
                    hatch = artist.get_hatch() if hasattr(artist, "get_hatch") else None
                    shape = sizes = None
                    if hasattr(artist, "get_paths") and hasattr(artist, "get_sizes"):
                        paths = artist.get_paths()
                        shape = (tuple(round(float(v), 3) for v in paths[0].vertices.flatten()) if len(paths)
                                 else None)
                        drawn = _fi_np.asarray(artist.get_sizes(), dtype=float)
                        sizes = (round(float(drawn.min()), 3), round(float(drawn.max()), 3)) if drawn.size else None
                    return (type(artist).__name__, _fi_key_colour(artist.get_facecolor()),
                            _fi_key_colour(artist.get_edgecolor()), str(hatch), shape, sizes, alpha)
            except Exception:
                pass
            return (type(artist).__name__, id(artist))

        def _fi_mergeable(groups):
            # One legend for several panels says a name is drawn alike in every panel: only when it is, and no
            # legend names two series alike.
            styles = {}
            for entries in groups:
                names = [label for label, _ in entries]
                if len(set(names)) != len(names):
                    return False
                for label, handle in entries:
                    style = _fi_key_style(handle)
                    if styles.setdefault(label, style) != style:
                        return False
            return True

        def _fi_legend_kwargs(leg):
            kw = {}
            for key in ("numpoints", "scatterpoints", "markerscale"):
                value = getattr(leg, key, None)
                if value is not None:
                    kw[key] = value
            handler_map = getattr(leg, "_custom_handler_map", None)
            if handler_map:
                kw["handler_map"] = handler_map
            return kw

        def _fi_fitted_legend(make, n, room, renderer, columns):
            # The legend make(ncol) with the most columns (up to `columns`) whose width fits
            # `room` pixels; one column when none fits. Earlier tries are removed.
            new = None
            for ncol in range(max(1, min(n, columns)), 0, -1):
                if new is not None:
                    new.remove()
                new = make(ncol)
                if new.get_window_extent(renderer).width <= room:
                    break
            return new

        def _fi_titled(new, title):
            if title:
                new.get_title().set_fontsize(_FI_PAPER["legend_pt"])
            return new

        def _fi_figure_legend(fig, legends, loc, renderer, merge):
            # One legend for the figure at `loc` ("outside lower center", ...), in place of `legends`; a constrained
            # layout makes room for it. Merged, a name the legends share is listed once (they draw it alike).
            entries, seen = [], set()
            for leg in legends:
                for label, handle in _fi_legend_entries(leg):
                    if merge and label in seen:
                        continue
                    seen.add(label)
                    entries.append((label, handle))
            if not entries:
                return None
            labels, handles = [label for label, _ in entries], [handle for _, handle in entries]
            title = next((leg.get_title().get_text() for leg in legends if leg.get_title().get_text()), "")
            columns = 6 if loc.endswith("center") else 1
            kw = _fi_legend_kwargs(legends[0])

            def make(ncol):
                return _fi_titled(fig.legend(handles, labels, loc=loc, ncol=ncol, frameon=False,
                                             fontsize=_FI_PAPER["legend_pt"], title=title or None, **kw), title)
            new = _fi_fitted_legend(make, len(labels), fig.bbox.width * 0.98, renderer, columns)
            for leg in legends:
                leg.set_visible(False)
            return new

        def _fi_outside_figure_legends(fig, renderer):
            # A figure legend placed off the canvas (bbox_to_anchor below it, kept by a
            # tight crop) is set "outside" the panels instead.
            moved, below = [], []
            fb = fig.bbox
            for leg in list(getattr(fig, "legends", [])):
                if not leg.get_visible():
                    continue
                box = leg.get_window_extent(renderer)
                if box.y0 < fb.y0 - 1:
                    loc = "outside lower center"
                elif box.y1 > fb.y1 + 1:
                    loc = "outside upper center"
                elif box.x1 > fb.x1 + 1:
                    loc = "outside right upper"
                elif box.x0 < fb.x0 - 1:
                    loc = "outside left upper"
                else:
                    continue
                new = _fi_figure_legend(fig, [leg], loc, renderer, False)
                if new is None:
                    continue
                moved.append({"panel": "figure", "to": loc})
                if loc == "outside lower center":
                    below.append(new)
            return moved, below

        def _fi_move_legends(fig, panels, renderer, several, constrained):
            # A legend over the data: first where matplotlib finds the emptiest corner; then, if that is over the
            # data too, under the plot: one legend for the figure under its panels when there are several and every
            # panel's legend draws each name alike, else under its own axes.
            moved, below, still, over = [], [], [], []
            for index, ax in enumerate(panels):
                leg = ax.get_legend()
                if leg is None or not leg.get_visible() or sum(_fi_legend_hits(ax, leg, renderer)) <= 0:
                    continue
                _fi_set_loc(leg, "best")
                leg.set_bbox_to_anchor(None)
                over.append((index, ax, leg))
            if over:
                # One draw places them all: each legend's place depends on its own axes only.
                renderer = _fi_draw(fig)
            for index, ax, leg in over:
                if sum(_fi_legend_hits(ax, leg, renderer)) <= 0:
                    moved.append({"panel": index, "to": "best"})
                else:
                    still.append((index, ax, leg))
            if still and several and constrained:
                shown = [ax.get_legend() for ax in panels]
                groups = [_fi_legend_entries(leg) for leg in shown if leg is not None and leg.get_visible()]
                if _fi_mergeable(groups):
                    new = _fi_figure_legend(fig, [leg for _, _, leg in still], "outside lower center", renderer, True)
                    if new is not None:
                        moved += [{"panel": index, "to": "below the panels"} for index, _, _ in still]
                        below.append(new)
                        return moved, below, renderer
            for index, ax, leg in still:
                entries = _fi_legend_entries(leg)
                if not entries:
                    continue
                labels, handles = [label for label, _ in entries], [handle for _, handle in entries]
                title = leg.get_title().get_text()
                kw = _fi_legend_kwargs(leg)

                def make(ncol, ax=ax, labels=labels, handles=handles, title=title, kw=kw):
                    return _fi_titled(ax.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False,
                                                fontsize=_FI_PAPER["legend_pt"], title=title or None, **kw), title)
                new = _fi_fitted_legend(make, len(labels), ax.get_window_extent(renderer).width, renderer, 6)
                new.set_bbox_to_anchor((0.5, _fi_below(fig, ax, renderer)), transform=ax.transAxes)
                moved.append({"panel": index, "to": "below the axes"})
                below.append(new)
            return moved, below, renderer

        def _fi_wrap(text, lines):
            # The label on `lines` lines of about equal length, broken at spaces outside
            # $...$ math; unchanged when it is one word.
            words, word, math = [], "", False
            for ch in text:
                if ch == "$":
                    math = not math
                if ch == " " and not math:
                    if word:
                        words.append(word)
                    word = ""
                else:
                    word += ch
            if word:
                words.append(word)
            if len(words) < 2:
                return text
            target = len(text) / lines
            out, line = [], ""
            for word in words:
                if line and len(line) + 1 + len(word) > target and len(out) < lines - 1:
                    out.append(line)
                    line = word
                else:
                    line = line + " " + word if line else word
            out.append(line)
            return "\\n".join(out)

        def _fi_wrap_labels(fig, panels, renderer):
            # An axis label longer than its axis, or cut by the canvas edge, is set on two
            # (or three) lines.
            wrapped = []
            fb = fig.bbox
            for ax in panels:
                box = ax.get_window_extent(renderer)
                for label, along_x in ((ax.xaxis.label, True), (ax.yaxis.label, False)):
                    text = label.get_text()
                    if not label.get_visible() or not text.strip() or "\\n" in text:
                        continue
                    ext = label.get_window_extent(renderer)
                    length, room = (ext.width, box.width) if along_x else (ext.height, box.height)
                    cut = ext.x0 < fb.x0 - 0.5 or ext.x1 > fb.x1 + 0.5 or ext.y0 < fb.y0 - 0.5 or ext.y1 > fb.y1 + 0.5
                    if length <= 1.1 * room and not cut:
                        continue
                    new = _fi_wrap(text, 2 if length <= 2 * room else 3)
                    if new == text:
                        continue
                    label.set_text(new)
                    wrapped.append(text)
            return wrapped

        def _fi_paper_attempt(fig, mode, stem, outputs):
            # Draws `fig` (a copy) as the paper copy under one layout, "constrained", or "tight" or "none" when that
            # fails, and saves it to `outputs`. Raises when it cannot.
            p = _FI_PAPER
            panels = _fi_panels(fig)
            rows, cols = _fi_grid(panels)
            cap = _fi_cap(rows)
            drawn = [float(v) for v in fig.get_size_inches()]
            width, height, rule = _fi_paper_figure_size(drawn, rows, cols, p)
            info = {"file": p["dirname"] + "/" + stem + ".pdf", "png": p["dirname"] + "/" + stem + ".png",
                    "drawn_in": [round(v, 3) for v in drawn], "width": rule}
            # The size of each panel's plotting area as the figure was drawn, in inches: what the copy is held to.
            measure_failed = False
            try:
                first = _fi_draw(fig)
                # Every visible Axes, a twin, an inset and a colorbar's included: the copy is made from this very
                # figure, so each is the same object before and after.
                measured_axes = [ax for ax in _fi_all_axes(fig) if ax.get_visible()]
                drawn_panels = {id(ax): (ax.get_window_extent(first).width / fig.dpi,
                                         ax.get_window_extent(first).height / fig.dpi) for ax in measured_axes}
            except Exception:
                drawn_panels = None
                measured_axes = []
                measure_failed = True
            fig.set_layout_engine(mode)
            constrained = mode == "constrained"
            fig.set_size_inches(width, height)
            # No title inside the plot: the caption carries it. A one-panel figure
            # loses its title. In a figure of several panels the panels' titles name
            # them and stay, all of them, when each fits over its panel; else all go.
            dropped, kept = [], []
            sup = getattr(fig, "_suptitle", None)
            if sup is not None and sup.get_visible() and sup.get_text().strip():
                info["suptitle_dropped"] = sup.get_text()
                sup.set_visible(False)
            several = len({_fi_place(ax) for ax in panels}) > 1
            titles = [t for ax in panels for t in _fi_all_titles(ax) if t.get_visible() and t.get_text().strip()]

            def drop(texts):
                for text in texts:
                    dropped.append(text.get_text())
                    text.set_text("")
            if several:
                for text in titles:
                    text.set_fontsize(p["title_pt"])
            else:
                drop(titles)
            _fi_paper_text(fig)
            rasterized = _fi_rasterize_big(fig)
            renderer = _fi_draw(fig)
            if several and titles:
                room = {id(t): ax.get_window_extent(renderer).width for ax in panels for t in _fi_all_titles(ax)}
                if all(t.get_window_extent(renderer).width <= room.get(id(t), 0) for t in titles):
                    kept = [t.get_text() for t in titles]
                else:
                    drop(titles)
                    renderer = _fi_draw(fig)
            wrapped = _fi_wrap_labels(fig, panels, renderer)
            if wrapped:
                renderer = _fi_draw(fig)
            moved, below = _fi_outside_figure_legends(fig, renderer) if constrained else ([], [])
            if moved:
                renderer = _fi_draw(fig)
            more, under, renderer = _fi_move_legends(fig, panels, renderer, several, constrained)
            moved += more
            below += under
            if below:
                # A legend set under the plot takes room of its own: the canvas grows by
                # it, up to the height cap, so the panels keep their height.
                renderer = _fi_draw(fig)
                extra = max(leg.get_window_extent(renderer).height for leg in below) / fig.dpi + 0.05
                height = min(height + extra, cap)
                fig.set_size_inches(width, height)
                renderer = _fi_draw(fig)
                for _ in range(2):
                    for leg in below:
                        if leg.axes is not None:
                            leg.set_bbox_to_anchor((0.5, _fi_below(fig, leg.axes, renderer)),
                                                   transform=leg.axes.transAxes)
                    renderer = _fi_draw(fig)
            more_wrapped = _fi_wrap_labels(fig, panels, renderer)
            if more_wrapped:
                wrapped += more_wrapped
                renderer = _fi_draw(fig)
            # Anything still off the canvas (text placed outside the axes by hand) is kept by a tight crop, as is
            # everything when the layout is not a constrained one; the canvas is then narrowed (and shortened) by
            # what the crop adds, so the saved copy is no wider than the paper width and no taller than the cap.
            pad = 0.02
            tight = not constrained
            box = _fi_overflow_box(fig, renderer)
            if tight or box.x0 < -0.02 or box.x1 > width + 0.02 or box.y0 < -0.02 or box.y1 > height + 0.02:
                tight = True
                for _ in range(4):
                    wide = box.width + 2 * pad - width
                    tall = box.height + 2 * pad - cap
                    if wide <= 0.02 and tall <= 0.02:
                        break
                    w_now, h_now = (float(v) for v in fig.get_size_inches())
                    fig.set_size_inches(max(w_now - max(wide, 0.0), width * 0.5), max(h_now - max(tall, 0.0), cap * 0.3))
                    renderer = _fi_draw(fig)
                    box = _fi_overflow_box(fig, renderer)
                if box.height + 2 * pad > cap + 0.05:
                    # Taller than a page allows even after shortening: no copy rather than one the page cannot hold.
                    raise ValueError("the copy is %.2f in tall, more than the %.2f in a paper figure may be"
                                     % (box.height + 2 * pad, cap))
            saved = ([box.width + 2 * pad, box.height + 2 * pad] if tight
                     else [float(v) for v in fig.get_size_inches()])
            layout = []
            for index, ax in enumerate(panels):
                leg = ax.get_legend()
                if leg is not None and leg.get_visible():
                    line_px, points = _fi_legend_hits(ax, leg, renderer)
                    if line_px > 0 or points > 0:
                        layout.append({"check": "legend_over_data", "panel": index,
                                       "legend": [t.get_text() for t in leg.get_texts()],
                                       "line_px": round(line_px, 1), "points": points})
            # How much of its drawn size each panel kept in the copy (both sizes taken from the drawn panels).
            panels_in, panel_share = [], None
            scale = width / drawn[0] if drawn[0] > 0 else 1.0
            try:
                for ax in (measured_axes if drawn_panels else []):
                    box_now = ax.get_window_extent(renderer)
                    now = (box_now.width / fig.dpi, box_now.height / fig.dpi)
                    panels_in.append([round(now[0], 3), round(now[1], 3)])
                    before = drawn_panels[id(ax)]
                    if before[0] > 0 and before[1] > 0:
                        share = min(now[0] / (before[0] * scale), now[1] / (before[1] * scale))
                        panel_share = share if panel_share is None else min(panel_share, share)
            except Exception:
                measure_failed = True
            if panel_share is not None and panel_share < p["panel_min_share"]:
                raise ValueError("a panel is left %.0f%% of its drawn size, under the %.0f%% a copy keeps"
                                 % (100 * panel_share, 100 * p["panel_min_share"]))
            ticks = _fi_tick_pt(fig)
            labels = [float(t.get_fontsize()) for ax in panels for t in (ax.xaxis.label, ax.yaxis.label)
                      if t.get_visible() and t.get_text().strip()]
            _fi_os.makedirs(_fi_os.path.dirname(outputs[0]), exist_ok=True)
            # The crop is passed as the box itself: savefig's own "tight" would leave out the same label.
            crop = _fi_mpl.transforms.Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
            kw = {"bbox_inches": crop if tight else None, "pad_inches": 0 if tight else pad, "dpi": p["dpi"]}
            with _fi_mpl.rc_context({"pdf.fonttype": 42, "savefig.bbox": "standard"}):
                _fi_savefig(fig, outputs[0], format="pdf", **kw)
                _fi_savefig(fig, outputs[1], format="png", **kw)
            info.update({
                "width_in": round(saved[0], 3), "height_in": round(saved[1], 3),
                "fit": mode + (", cropped" if tight else ""),
                "tick_pt": ticks, "label_pt": min(labels) if labels else None,
                "titles_dropped": dropped, "titles_kept": kept, "labels_wrapped": wrapped,
                "legend_moved": moved, "layout": layout, "panels_in": panels_in,
                "panel_share": None if panel_share is None else round(panel_share, 3),
                # "ok" when every axes was measured; "failed" when a measurement was tried and not completed: the
                # paper then does not use this copy.
                "panel_measure": "failed" if measure_failed else "ok",
            })
            if rasterized:
                info["rasterized"] = rasterized
            return info

        def _fi_paper_copy(fig, fname, paper_dir):
            # The paper copy of fig, drawn from an independent copy of it: fig itself is never changed. The caller
            # has removed this file name's earlier copy; when no copy can be made, none is left. Never raises.
            started = _fi_time.perf_counter()
            stem, outputs = _fi_paper_outputs(fname, paper_dir)
            try:
                _fi_os.makedirs(paper_dir, exist_ok=True)
            except OSError as exc:
                return {"error": "no folder for the paper copy: " + repr(exc)[:240]}
            try:
                make = _fi_copy_maker(fig)
            except Exception as exc:
                return {"error": "the figure could not be copied: " + repr(exc)[:240]}
            failed = []
            for mode in ("constrained", "tight", "none"):
                if failed and _fi_time.perf_counter() - started > _FI_PAPER["budget_s"]:
                    failed.append("no time left for another layout")
                    break
                try:
                    # matplotlib itself says when a constrained layout gave up ("axes sizes collapsed to zero") and
                    # drew the panels as slivers: a copy drawn that way is not kept; the next layout is tried, and
                    # with none left the figure is placed from its PNG.
                    with _fi_warnings.catch_warnings(record=True) as caught:
                        _fi_warnings.simplefilter("always")
                        info = _fi_paper_attempt(make(), mode, stem, outputs)
                    if any("collapsed" in str(w.message) for w in caught):
                        raise ValueError("the layout collapsed the panels")
                except Exception as exc:
                    _fi_paper_clear(outputs)
                    failed.append(mode + ": " + repr(exc)[:200])
                    continue
                if failed:
                    info["layouts_failed"] = failed
                return info
            return {"error": "; ".join(failed)[:600]}
'''


def _boot_paper_source() -> str:
    """The paper copy's part of the bootstrap: the profile's values and the size rule, written once in this module."""
    import inspect
    import textwrap

    size_rule = inspect.getsource(paper_figure_size).replace("def paper_figure_size(", "def _fi_paper_figure_size(", 1)
    size_rule = size_rule.replace("p = profile or paper_profile()", "p = profile", 1)
    return (_PAPER_SOURCE
            .replace("@@PAPER_PROFILE@@", repr(paper_profile()))
            .replace("@@PAPER_SIZE@@", textwrap.indent(size_rule, " " * 8)))


def write_boot(fi_dir: Path, paper_style: str = "briefing") -> Path:
    """Write the ``sitecustomize.py`` bootstrap under ``<fi_dir>/plotstyle/``
    and return that directory. The engine prepends it to the experiment
    subprocess's ``PYTHONPATH`` so Python auto-imports it at startup."""
    boot_dir = Path(fi_dir) / BOOT_DIRNAME
    boot_dir.mkdir(parents=True, exist_ok=True)
    (boot_dir / "sitecustomize.py").write_text(
        _boot_source(paper_style), encoding="utf-8",
    )
    return boot_dir
