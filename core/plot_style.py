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
