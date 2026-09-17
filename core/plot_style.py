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


def figure_facecolor(paper_style: str) -> str:
    """Backdrop for figures, keyed to the paper they'll be embedded in."""
    return PAPER_CREAM if str(paper_style).lower() == "briefing" else "white"


def fi_rcparams(paper_style: str) -> dict[str, object]:
    """The house rcParams. ``paper_style`` only changes the face colors."""
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
        "font.size": 11.5,
        "axes.titlesize": 13.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 11,
        "axes.labelsize": 11.5,
        "axes.labelpad": 6,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
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


def apply_style(paper_style: str = "briefing") -> None:
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
    matplotlib.rcParams.update(fi_rcparams(paper_style))
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
''' + _RECORDER_SOURCE


# Name of the dir, under <quest>/.fi/, where the bootstrap records what each
# saved figure draws; the engine passes it in FI_FIGURE_RECORDS.
RECORDS_DIRNAME = "figure_records"

# Appended to the bootstrap. A figure of the validation quest was captioned
# "energy drift rates for RK4 and Velocity-Verlet" while both lines lay flat at
# 0 under forward Euler's 0.44 J spike: the writer knew only the file name.
# On each savefig this writes <stem>.json with every Axes' title, labels, y
# scale and limits, and each labelled series' range and whether it shows on
# that axis ("yes", "flat" under 1% of the axis span, or "outside the axis").
# Beside it, <stem>.seed<k>.json keeps the run's lines themselves (points and
# style, per panel, and whether the panel holds anything but lines and error
# bars), which the engine uses to redraw a figure as the mean over the seeds.
_RECORDER_SOURCE = '''\
try:
    import os as _fi_os

    _fi_records = _fi_os.environ.get("FI_FIGURE_RECORDS")
    if _fi_records:
        import json as _fi_json
        import math as _fi_math
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
            groups, unlabelled = [], []
            for artist in artists:
                label = names.get(id(artist)) or str(artist.get_label() or "")
                try:
                    if hasattr(artist, "get_ydata"):
                        ys = _fi_values(artist.get_ydata())
                    else:
                        ys = _fi_values(point[1] for point in artist.get_offsets())
                except Exception:
                    continue
                style, marker = _fi_style(artist), _fi_marker(artist)
                if label and not label.startswith("_"):
                    groups.append((label, style, marker, ys))
                elif style is not None and (style[0] == "points" or style[3] not in ("None", "", " ")):
                    unlabelled.append((style, marker, ys))
            for style, marker, ys in unlabelled:
                same_style = [g for g in groups if g[1] == style]
                same_marker = [g for g in groups if marker is not None and g[2] == marker]
                target = same_style[0] if same_style else same_marker[0] if len(same_marker) == 1 else None
                if target is not None:
                    target[3].extend(ys)
            series = [_fi_series(label, ys, lo, hi, log) for label, _style, _marker, ys in groups]
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

        def _fi_line(ax, line, names):
            x, y = _fi_points(line)
            return {
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

        def _fi_panel(ax):
            bar_names, bar_pieces = _fi_error_bars(ax)
            try:
                names = _fi_legend_names(ax, list(ax.get_lines()) + list(ax.collections))
            except Exception:
                names = {}
            names.update(bar_names)
            lines = [_fi_line(ax, line, names) for line in ax.get_lines() if id(line) not in bar_pieces]
            spec = ax.get_subplotspec()
            grid = None
            if spec is not None:
                rows, cols = spec.get_gridspec().get_geometry()
                grid = [rows, cols, spec.rowspan.start, spec.rowspan.stop, spec.colspan.start, spec.colspan.stop]
            formatters = {type(axis.get_major_formatter()).__name__ for axis in (ax.xaxis, ax.yaxis)}
            others = (ax.patches, ax.collections, ax.images, ax.texts, ax.tables, ax.artists, ax.child_axes)
            return {
                "grid": grid,
                "title": " ".join(t for t in (ax.get_title("left"), ax.get_title(), ax.get_title("right")) if t),
                "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
                "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
                "legend": ax.get_legend() is not None,
                # Only lines (error bars included), on default ticks: what a redraw can reproduce.
                "line_only": bool(lines) and not any(
                    _fi_draws(artist) for group in others for artist in group if id(artist) not in bar_pieces
                )
                and formatters <= _FI_DEFAULT_FORMATTERS
                and all(line["kind"] != "other" and line["x"] is not None for line in lines),
                "lines": lines,
            }

        def _fi_plot_data(fig, name):
            suptitle = getattr(fig, "_suptitle", None)
            return {
                "file": name, "size": [float(v) for v in fig.get_size_inches()],
                "suptitle": suptitle.get_text() if suptitle is not None else "",
                "axes": [_fi_panel(ax) for ax in fig.get_axes()],
            }

        _fi_savefig = _FiFigure.savefig

        def _fi_recording_savefig(self, fname, *args, **kwargs):
            result = _fi_savefig(self, fname, *args, **kwargs)
            try:
                name = _fi_os.path.basename(_fi_os.fspath(fname))
                record = {"file": name, "axes": [_fi_axes(ax) for ax in self.get_axes()]}
                _fi_os.makedirs(_fi_records, exist_ok=True)
                stem = _fi_os.path.join(_fi_records, _fi_os.path.splitext(name)[0])
                with open(stem + ".json", "w", encoding="utf-8") as handle:
                    _fi_json.dump(record, handle)
                # Each seed's lines, for the engine to redraw a figure as the
                # mean over the seeds. A redrawn figure is no seed's.
                if not _fi_os.environ.get("FI_REPLOT"):
                    # Keyed on WHICH replicate this is, not on what it seeds
                    # with. The two used to be the same small integer; the
                    # seeds now stride far apart to keep the replicates' random
                    # streams disjoint, while the engine still looks these
                    # files up by ordinal (0, 1, 2 ...).
                    seed = "".join(c for c in _fi_os.environ.get("FI_REPLICATE_INDEX", "") if c.isdigit()) or "0"
                    with open(stem + ".seed" + seed + ".json", "w", encoding="utf-8") as handle:
                        _fi_json.dump(_fi_plot_data(self, name), handle)
            except Exception:
                pass
            return result

        _FiFigure.savefig = _fi_recording_savefig
except Exception:
    pass
'''


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
