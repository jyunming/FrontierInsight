"""The paper copy is drawn from an independent copy of the script's figure (core/plot_style.py), and FI keeps each copy
with its figure (core/engine.py, core/result_checks.py, generation/_paper_figures.py).

The claims, each checked here on the matplotlib patterns a review of the first version found broken:
- the copy shows the same data as the slide figure: the same lines, points (and their sizes), images, and legend
  entries in the same colours, a label two series share kept twice;
- the script's figure is not changed: saving it again gives the same bytes as with no paper copy at all;
- a figure FI drops loses its copy, and paper.pdf never shows a copy whose figure is gone;
- a copy that is not made again (the save was slow, the figure cannot be copied) leaves no older copy behind;
- a copy is no taller than the cap, on every layout path.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.plot_style import PAPER_MAX_HEIGHT_IN, PAPER_ROW_MIN_IN, write_boot

# Records each print_figure (the slide save, then the paper copy's PDF and PNG) after the draw: what each draws.
SNAP_HEAD = r'''
import atexit, hashlib, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import backend_bases
from matplotlib.colors import to_hex

SNAPS = []

def _h(a):
    arr = np.ma.filled(np.ma.asarray(a, dtype=float), np.nan)
    return hashlib.sha1(np.round(arr, 9).tobytes()).hexdigest()[:12] + str(arr.shape)

def _c(c):
    try:
        return to_hex(c, keep_alpha=True)
    except Exception:
        try:
            return to_hex(np.asarray(c)[0], keep_alpha=True)
        except Exception:
            return str(c)

def _legend(leg):
    if leg is None or not leg.get_visible():
        return []
    keys = list(leg.legend_handles)
    out = []
    for i, t in enumerate(leg.get_texts()):
        k = keys[i] if i < len(keys) else None
        colour = size = shape = None
        if k is not None:
            colour = _c(k.get_color()) if hasattr(k, "get_color") else _c(k.get_facecolor())
            if hasattr(k, "get_sizes"):
                size = [round(float(s), 3) for s in k.get_sizes()]
            if hasattr(k, "get_marker"):
                shape = [str(k.get_marker()), round(float(k.get_linewidth()), 3), k.get_alpha()]
            elif hasattr(k, "get_paths") and len(k.get_paths()):
                shape = len(k.get_paths()[0].vertices)
        out.append([t.get_text(), colour, size, shape])
    return out

def _snap(fig):
    axes = []
    legends = [e for leg in fig.legends for e in _legend(leg)]
    for ax in fig.get_axes():
        legends += _legend(ax.get_legend())
        axes.append({
            "xlim": [float(v) for v in ax.get_xlim()], "ylim": [float(v) for v in ax.get_ylim()],
            "aspect": str(ax.get_aspect()), "scales": [ax.get_xscale(), ax.get_yscale()],
            "lines": [[l.get_visible(), _c(l.get_color()), str(l.get_linestyle()), str(l.get_marker()),
                       _h(l.get_xydata())] for l in ax.get_lines()],
            "colls": [[type(c).__name__, c.get_visible(),
                       _h(c.get_offsets()) if hasattr(c, "get_offsets") else None,
                       [round(float(s), 4) for s in (c.get_sizes() if hasattr(c, "get_sizes") else [])]]
                      for c in ax.collections],
            "images": [[_h(im.get_array()), [float(v) for v in im.get_clim()]] for im in ax.images],
            "texts": [t.get_text() for t in ax.texts],
            "ticks": sorted({round(float(t.get_fontsize()), 2) for t in ax.xaxis.get_ticklabels()}),
        })
    return {"axes": axes, "legend": sorted(legends), "size": [float(v) for v in fig.get_size_inches()]}

_print = backend_bases.FigureCanvasBase.print_figure

def _wrapped(self, filename, *a, **k):
    out = _print(self, filename, *a, **k)
    name = "<buffer>" if hasattr(filename, "write") else os.fspath(filename).replace("\\", "/")
    SNAPS.append({"file": name, "snap": _snap(self.figure)})
    return out

backend_bases.FigureCanvasBase.print_figure = _wrapped
os.makedirs("figures", exist_ok=True)
os.makedirs("out", exist_ok=True)
atexit.register(lambda: open("out/snaps.json", "w").write(json.dumps(SNAPS)))

def finish(fig):
    # The script's figure afterwards: the state, and a save of it again.
    SNAPS.append({"file": "after", "snap": _snap(fig), "current": plt.gcf() is fig, "open": plt.get_fignums()})
    fig.savefig("out/second.png", dpi=80)
    fig.savefig("out/second.pdf", metadata={"CreationDate": None, "ModDate": None})
'''

CASES = {
    # Panel 2 draws its series in the other order: the same names, other colours.
    "same_label_diff_color": """
x = np.linspace(0, 1, 200)
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
a.plot(x, x, label='Euler'); a.plot(x, x**2, label='RK4'); a.legend(loc='lower right')
for k in range(10):
    b.plot(x, (x*(k+1)) % 1, color='0.8')
b.plot(x, x, label='RK4'); b.plot(x, 1 - x, label='Euler')
b.legend(loc='center')
fig.savefig('figures/swap.png')
finish(fig)
""",
    # The second review's probes: the same name and colour, but another marker (a scatter) or another line width and
    # alpha. One legend for both panels would key one panel's series in the other's style.
    "merge_scatter_diff_marker": """
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
rng = np.random.default_rng(1)
axes[0].scatter(rng.random(300), rng.random(300), color='C0', marker='o', label='trial')
axes[1].scatter(rng.random(300), rng.random(300), color='C0', marker='^', label='trial')
for ax in axes:
    ax.legend(loc='center')
fig.savefig('figures/sc.png')
finish(fig)
""",
    "merge_same_style_diff_alpha_lw": """
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = np.linspace(0, 1, 40)
for k in range(12):
    for ax in axes:
        ax.plot(x, (x * (k + 1)) % 1, color='0.85')
axes[0].plot(x, 0.5 + 0.4 * np.sin(8 * x), color='C0', lw=1, label='model')
axes[1].plot(x, 0.5 + 0.4 * np.sin(8 * x), color='C0', lw=4, alpha=0.4, label='model')
for ax in axes:
    ax.legend(loc='center')
fig.savefig('figures/lw.png')
finish(fig)
""",
    "dup_label_one_legend": """
fig, ax = plt.subplots(figsize=(7, 4.3))
x = np.linspace(0, 1, 200)
for k in range(10):
    ax.plot(x, (x*(k+1)) % 1, color='0.85')
ax.plot(x, x, 'r', label='fit'); ax.plot(x, 1-x, 'b', label='fit')
ax.legend(loc='center')
fig.savefig('figures/dup.png')
finish(fig)
""",
    "bubble_size_legend": """
fig, ax = plt.subplots(figsize=(6, 5))
rng = np.random.default_rng(1)
ax.scatter(rng.random(15), rng.random(15), s=rng.integers(20, 400, 15), alpha=0.5)
for s in (50, 200, 400):
    ax.scatter([], [], s=s, color='k', alpha=0.4, label=f'{s} people')
ax.legend(title='population', loc='upper left', bbox_to_anchor=(1.0, 1.0))
fig.savefig('figures/bubble.png', bbox_inches='tight')
finish(fig)
""",
    "errorbar_fill": """
fig, ax = plt.subplots(figsize=(7, 4.3))
x = np.arange(5.0)
ax.errorbar(x, x**2, yerr=0.5 + x, label='mean +/- SD', capsize=3)
ax.fill_between(x, x**2 - 3, x**2 + 3, alpha=0.2, label='95% band')
ax.legend()
fig.savefig('figures/eb.png')
finish(fig)
""",
    "axis_equal_datalim": """
fig, ax = plt.subplots(figsize=(10, 4))
t = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(t), 0.5*np.sin(t), label='orbit'); ax.plot([0.9], [0.0], 'o', label='perihelion')
ax.axis('equal')
fig.savefig('figures/orbit.png')
finish(fig)
""",
    "tick_label_size_by_hand": """
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([0, 1, 2], [3, 1, 2])
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(6)
fig.savefig('figures/handticks.png')
finish(fig)
""",
    "suptitle_autopos": """
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
a.plot([0, 1], [0, 1]); b.plot([0, 1], [1, 0])
fig.suptitle('Overview')
fig.savefig('figures/sup.png')
finish(fig)
""",
    "layout_engine_tight": """
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='tight')
for ax in axes:
    im = ax.pcolormesh(np.random.default_rng(0).random((8, 8))); fig.colorbar(im, ax=ax)
fig.savefig('figures/lt.png')
finish(fig)
""",
    "tight_then_colorbar": """
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax in axes:
    im = ax.imshow(np.random.default_rng(0).random((8, 8)))
    fig.colorbar(im, ax=ax)
    ax.set_title('field')
fig.tight_layout()
fig.savefig('figures/tightcb.png')
finish(fig)
""",
    "colorbar_imshow": """
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(np.arange(100).reshape(10, 10), extent=(0, 1, 0, 2), aspect='auto')
fig.colorbar(im, ax=ax, label='density (kg/m^3)')
ax.set_title('t = 3.2 s')
fig.savefig('figures/cbar.png')
finish(fig)
""",
    "lambda_formatter_and_pyplot_save": """
from matplotlib.ticker import FuncFormatter
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([0, 1, 2], [0.1, 0.5, 0.9])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{100 * v:.0f}%'))
fig.savefig('figures/fmt.png')
plt.savefig('out/pyplot_save.png')
finish(fig)
""",
    "inset_axes": """
fig, ax = plt.subplots(figsize=(7, 4.3))
x = np.linspace(0, 10, 300)
ax.plot(x, np.exp(-x)*np.sin(5*x))
ins = ax.inset_axes([0.55, 0.55, 0.4, 0.4]); ins.plot(x[:50], np.exp(-x[:50])*np.sin(5*x[:50]))
ins.set_xlabel('time t (s)')
fig.savefig('figures/inset.png')
finish(fig)
""",
}


def _run_case(tmp_path: Path, name: str, on: bool) -> dict:
    pytest.importorskip("matplotlib")
    root = tmp_path / name / ("on" if on else "off")
    root.mkdir(parents=True)
    boot = write_boot(root / ".fi", "latex")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(boot), os.environ.get("PYTHONPATH", "")) if p),
           "FI_FIGURE_RECORDS": str(root / ".fi" / "figure_records"), "FI_PAPER_FIGURES": "1" if on else "0",
           "SOURCE_DATE_EPOCH": "0", "PYTHONIOENCODING": "utf-8"}
    (root / "case.py").write_text(SNAP_HEAD + CASES[name], encoding="utf-8")
    out = subprocess.run([sys.executable, "case.py"], cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    snaps = json.loads((root / "out" / "snaps.json").read_text(encoding="utf-8"))
    records = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (root / ".fi" / "figure_records").glob("*.json")
               if ".seed" not in p.name}
    return {"root": root, "snaps": snaps, "records": records,
            "second": [(root / "out" / f).read_bytes() for f in ("second.png", "second.pdf")]}


# Each case is run once with the copy on and once off, however many tests read it (the file runs on one worker).
_RUNS: dict[tuple[str, bool], dict] = {}


def _case(factory: pytest.TempPathFactory, name: str, on: bool) -> dict:
    if (name, on) not in _RUNS:
        _RUNS[(name, on)] = _run_case(factory.mktemp(name), name, on)
    return _RUNS[(name, on)]


def _slide_and_paper(run: dict) -> tuple[dict, dict]:
    snaps = run["snaps"]
    slide = next(s["snap"] for s in snaps if s["file"].startswith("figures/") and "/paper/" not in s["file"])
    paper = next(s["snap"] for s in snaps if "/paper/" in s["file"] and s["file"].endswith(".pdf"))
    return slide, paper


def _data(snap: dict) -> list:
    return [{k: ax[k] for k in ("lines", "colls", "images", "texts", "scales")} for ax in snap["axes"]]


@pytest.mark.parametrize("name", ["same_label_diff_color", "merge_scatter_diff_marker", "merge_same_style_diff_alpha_lw",
                                  "dup_label_one_legend", "bubble_size_legend",
                                  "errorbar_fill", "axis_equal_datalim", "inset_axes"])
def test_the_copy_shows_the_same_data_as_the_slide_figure(tmp_path_factory: pytest.TempPathFactory,
                                                        name: str) -> None:
    slide, paper = _slide_and_paper(_case(tmp_path_factory, name, True))
    # The same lines (colour, style, marker, points), the same points and their sizes, images and texts.
    assert _data(paper) == _data(slide)
    # Every legend entry, in the colour (and size) of its key on the slide: a label two series share is kept twice,
    # and a name drawn in other colours in two panels is not merged into one entry.
    assert paper["legend"] == slide["legend"]
    for s, p in zip(slide["axes"], paper["axes"]):
        if s["aspect"] == "1.0":  # axis('equal'): a canvas of another shape shows more around the same data
            assert p["xlim"][0] <= s["xlim"][0] and p["xlim"][1] >= s["xlim"][1]
            assert p["ylim"][0] <= s["ylim"][0] and p["ylim"][1] >= s["ylim"][1]
        else:
            assert p["xlim"] == s["xlim"] and p["ylim"] == s["ylim"]


@pytest.mark.parametrize("name", ["tick_label_size_by_hand", "suptitle_autopos", "layout_engine_tight",
                                  "axis_equal_datalim", "same_label_diff_color", "lambda_formatter_and_pyplot_save"])
def test_a_second_save_of_the_scripts_figure_is_unchanged(tmp_path_factory: pytest.TempPathFactory,
                                                          name: str) -> None:
    on, off = _case(tmp_path_factory, name, True), _case(tmp_path_factory, name, False)
    assert (on["root"] / "figures" / "paper").is_dir() and not (off["root"] / "figures" / "paper").exists()
    assert on["second"] == off["second"], "the figure saved again differs from one never copied"
    after_on = next(s for s in on["snaps"] if s["file"] == "after")
    after_off = next(s for s in off["snaps"] if s["file"] == "after")
    assert after_on["snap"] == after_off["snap"]
    # The copy is never pyplot's current figure: a later plt.savefig() saves the script's own.
    assert after_on["current"] is True and after_on["open"] == after_off["open"]
    stem = next(iter(on["records"]))
    assert on["records"][stem]["tick_pt"] == off["records"][stem]["tick_pt"]


def _paper_info(run: dict) -> dict:
    return next(r["paper"] for r in run["records"].values() if "paper" in r)


@pytest.mark.parametrize("name", ["layout_engine_tight", "tight_then_colorbar", "colorbar_imshow",
                                  "same_label_diff_color", "bubble_size_legend"])
def test_every_layout_path_makes_a_copy_no_taller_than_the_cap(tmp_path_factory: pytest.TempPathFactory,
                                                               name: str) -> None:
    from generation._pdf_measure import measure_pdf

    run = _case(tmp_path_factory, name, True)
    info = _paper_info(run)
    assert "error" not in info, info
    pdf = next((run["root"] / "figures" / "paper").glob("*.pdf"))
    page = measure_pdf(pdf).pages[0]
    cap = max(PAPER_MAX_HEIGHT_IN, PAPER_ROW_MIN_IN)  # each of these is one row of panels
    assert page.height / 72 <= cap + 0.06 and page.width / 72 <= 6.3 + 0.05, (page.width / 72, page.height / 72)
    if name == "tight_then_colorbar":
        # Two panels with a colorbar each: one row of two (the colorbars' own grid is not more rows), titles kept.
        assert info["titles_kept"] == ["field", "field"]
    if name == "colorbar_imshow":
        # The constrained layout fails on it (ZeroDivisionError); a plain layout makes the copy.
        assert info["fit"].startswith("tight") and info.get("layouts_failed")


def test_inset_text_is_set_at_the_bodys_size(tmp_path_factory: pytest.TempPathFactory) -> None:
    from generation._pdf_measure import measure_pdf

    run = _case(tmp_path_factory, "inset_axes", True)
    page = measure_pdf(run["root"] / "figures" / "paper" / "inset.pdf").pages[0]
    inset_label = [line for line in page.lines if "time t (s)" in line.text]
    assert inset_label and all(line.size <= 9.5 for line in page.lines), [(l.text, l.size) for l in page.lines]


# ---- a copy that is not made again leaves no older one ---------------------------------------------------------


def _run_script(root: Path, script: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    boot = write_boot(root / ".fi", "latex")
    env = {**os.environ, "PYTHONPATH": str(boot), "FI_FIGURE_RECORDS": str(root / ".fi" / "figure_records")}
    env.pop("FI_PAPER_FIGURES", None)
    (root / "case.py").write_text(script, encoding="utf-8")
    out = subprocess.run([sys.executable, "case.py"], cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    return json.loads((root / ".fi" / "figure_records" / "x.json").read_text(encoding="utf-8"))


SAVE_TWICE = """
import time, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
line, = ax.plot([0, 1, 2], [1, 2, 3])
fig.savefig("figures/x.png")
assert os.path.exists("figures/paper/x.pdf")
line.set_ydata([30, 20, 10]); ax.relim(); ax.autoscale_view()
"""


def test_a_slow_save_leaves_no_older_copy(tmp_path: Path) -> None:
    """The review's probe: the second save of a changed figure reads 11 s on the hook's clock, so no copy is made."""
    record = _run_script(tmp_path, SAVE_TWICE + (
        "real = time.perf_counter\n"
        "state = {'n': 0}\n"
        "def fake():\n"
        "    state['n'] += 1\n"
        "    return real() + (11.0 if state['n'] == 2 else 0.0)\n"
        "time.perf_counter = fake\n"
        "fig.savefig('figures/x.png')\n"
        "time.perf_counter = real\n"))
    assert record["paper"] == {"skipped": "slow to save"}
    assert not any((tmp_path / "figures" / "paper").glob("x.*"))


def test_a_figure_that_cannot_be_copied_gets_no_copy_and_leaves_no_older_one(tmp_path: Path) -> None:
    record = _run_script(tmp_path, SAVE_TWICE + (
        "import threading\n"
        "fig._not_copyable = threading.Lock()\n"
        "fig.savefig('figures/x.png')\n"))
    assert "could not be copied" in record["paper"]["error"]
    assert not any((tmp_path / "figures" / "paper").glob("x.*"))
    assert (tmp_path / "figures" / "x.png").is_file()


def test_a_large_scatter_is_an_image_inside_the_vector_copy(tmp_path: Path) -> None:
    record = _run_script(tmp_path, (
        "import matplotlib, os\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt, numpy as np\n"
        "os.makedirs('figures', exist_ok=True)\nrng = np.random.default_rng(0)\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\nax.scatter(rng.random(20000), rng.random(20000), s=4)\n"
        "fig.savefig('figures/x.png')\n"))
    assert record["paper"]["rasterized"] == 1
    assert (tmp_path / "figures" / "paper" / "x.pdf").stat().st_size < 400_000


# ---- a figure FI drops loses its copy ----------------------------------------------------------------------------


def _quest_with_copies(root: Path) -> Path:
    (root / "figures" / "paper").mkdir(parents=True)
    (root / ".fi" / "figure_records").mkdir(parents=True)
    for name in ("typed.png", "ok.png"):
        (root / "figures" / name).write_bytes(b"png")
    for name in ("typed.pdf", "typed.png", "ok.pdf", "ok.png"):
        (root / "figures" / "paper" / name).write_bytes(b"%PDF-1.4 /MediaBox [0 0 453.6 180]")
    return root


def test_a_figure_a_result_check_drops_loses_its_copy(tmp_path: Path) -> None:
    from core.result_checks import ResultChecksMixin

    root = _quest_with_copies(tmp_path)
    fake = SimpleNamespace(quest_root=root, fi_dir=root / ".fi", _log=logging.getLogger("t"))
    ResultChecksMixin._drop_figures(fake, ["typed.png"])  # type: ignore[arg-type]
    left = sorted(str(p.relative_to(root / "figures")).replace("\\", "/") for p in (root / "figures").rglob("*")
                  if p.is_file())
    assert left == ["ok.png", "paper/ok.pdf", "paper/ok.png"]


def test_a_figure_drawn_from_typed_in_numbers_loses_its_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import figure_data_check as figure_data
    from core.engine import Engine

    root = _quest_with_copies(tmp_path)
    monkeypatch.setattr(figure_data, "figures_to_drop", lambda found, figures: ["typed.png"])
    monkeypatch.setattr(figure_data, "plain_note", lambda removed, where: "dropped")
    monkeypatch.setattr(figure_data, "record", lambda removed, where: {"removed_figures": removed})
    fake = SimpleNamespace(quest_root=root, _typed_figures={"experiment.py": ["typed.png"]}, _log=logging.getLogger("t"))
    kept = Engine._drop_typed_figures(fake, ["typed.png", "ok.png"], root / ".fi" / "figure_records")  # type: ignore[arg-type]
    assert kept == ["ok.png"]
    assert not (root / "figures" / "paper" / "typed.pdf").exists() and (root / "figures" / "paper" / "ok.pdf").exists()


def test_the_record_check_and_paper_pdf_do_not_see_a_copy_without_its_figure(tmp_path: Path) -> None:
    from core import record_claims
    from core.engine import _figure_files_on_disk
    from generation._paper_figures import prefer_paper_figures

    root = _quest_with_copies(tmp_path)
    (root / "figures" / "typed.png").unlink()  # dropped by some path that left the copy
    on_disk = _figure_files_on_disk(root)
    assert "typed.png" not in on_disk and "ok.png" in on_disk
    md = "Results.\n\n![Power by method](figures/typed.png)\n\n![ok](figures/ok.png)\n"
    assert record_claims.contradictions(md, figures_on_disk=on_disk), "the paper cites a figure that is not there"
    swapped, n = prefer_paper_figures(md, root)
    assert n == 1 and "](figures/typed.png)" in swapped and "](figures/paper/ok.pdf)" in swapped


def test_a_redraw_that_adds_a_figure_it_does_not_name_leaves_no_copy_of_it(tmp_path: Path) -> None:
    """The layout redraw removes a figure file it added without naming it; its paper copy goes too, and a file name
    the run already had gets its earlier copy back."""
    from core.engine import _put_back_paper_copies

    backup, figures = tmp_path / "backup" / "figures", tmp_path / "figures"
    (backup / "paper").mkdir(parents=True)
    (figures / "paper").mkdir(parents=True)
    (backup / "paper" / "fig1.pdf").write_bytes(b"before")
    for name in ("stray.pdf", "stray.png", "fig1.pdf"):
        (figures / "paper" / name).write_bytes(b"redraw")
    _put_back_paper_copies(backup, figures, "figures/stray.png")
    _put_back_paper_copies(backup, figures, "figures/fig1.svg")
    assert sorted(p.name for p in (figures / "paper").iterdir()) == ["fig1.pdf"]
    assert (figures / "paper" / "fig1.pdf").read_bytes() == b"before"


# ---- the engine side of the cost and of the dropped titles ------------------------------------------------------


def test_replicate_runs_make_no_paper_copies() -> None:
    from core.engine import _replicate_env

    assert "FI_PAPER_FIGURES" not in _replicate_env({"PATH": "x"}, 0, 1000)
    assert _replicate_env({"PATH": "x"}, 2, 1000)["FI_PAPER_FIGURES"] == "0"


def test_the_writer_is_told_which_titles_the_paper_copy_does_not_show(tmp_path: Path) -> None:
    from core.engine import _figure_record_note, _read_figure_records

    record = {"file": "scat.png", "axes": [{"title": "sigma=0.02, n=12: observed r=0.9992", "series": []}],
              "paper": {"titles_dropped": ["sigma=0.02, n=12: observed r=0.9992"], "width_in": 3.1},
              "paper_dropped": ["All runs at T = 300 K", "sigma=0.02, n=12: observed r=0.9992"]}
    (tmp_path / "scat.json").write_text(json.dumps(record), encoding="utf-8")
    read = _read_figure_records(tmp_path, ["scat.png"])["scat.png"]
    assert "paper" not in read and read["paper_dropped"] == record["paper_dropped"]
    note = _figure_record_note(read)
    assert ('in a one-column paper.pdf this figure does not show its titles "All runs at T = 300 K"; '
            '"sigma=0.02, n=12: observed r=0.9992": put in the caption') in note
    assert "does not show" not in _figure_record_note({"axes": [{"title": "x", "series": []}]})


def test_the_bootstrap_records_the_titles_the_copy_drops(tmp_path: Path) -> None:
    record = _run_script(tmp_path, (
        "import matplotlib, os\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "os.makedirs('figures', exist_ok=True)\n"
        "fig, ax = plt.subplots(figsize=(6, 5)); ax.plot([0, 1], [0, 1])\n"
        "ax.set_title('dashed: true 50 kJ/mol'); fig.suptitle('All runs at T = 300 K')\n"
        "fig.savefig('figures/x.png')\n"))
    assert record["paper_dropped"] == ["All runs at T = 300 K", "dashed: true 50 kJ/mol"]


# ---- where each panel sits, for the caption ------------------------------------------------------------------------


@pytest.mark.parametrize(("layout", "places"), [
    ("plt.subplots(1, 2)", ["left", "right"]),
    ("plt.subplots(2, 1)", ["top", "bottom"]),
    ("plt.subplots(1, 3)", ["left", "centre", "right"]),
    ("plt.subplots(2, 2)", ["top left", "top right", "bottom left", "bottom right"]),
    ("plt.subplots()", [""]),
])
def test_the_record_says_where_each_panel_sits(tmp_path: Path, layout: str, places: list[str]) -> None:
    from core.engine import _panel_place

    record = _run_script(tmp_path, (
        "import matplotlib, os\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport numpy as np\n"
        "os.makedirs('figures', exist_ok=True)\n"
        f"fig, axes = {layout}\n"
        "for i, ax in enumerate(np.atleast_1d(axes).ravel()):\n"
        "    ax.plot([0, 1, 2], [i, i + 1, i + 3], label=f's{i}'); ax.set_title(f'panel {i}')\n"
        "fig.savefig('figures/x.png')\n"
    ))
    assert [_panel_place(ax["grid"]) for ax in record["axes"]] == places


def test_the_writer_is_told_a_side_by_side_figure_is_left_and_right() -> None:
    # The live kinetics run (round s3): Figure 1 drew its two panels side by side, the writer saw only "panel 1,
    # panel 2" and captioned them "Top ... Bottom ...". The grids are that run's own record.
    from core.engine import _figure_list_for_prompt, _figure_record_note

    record = {"file": "fig1.png", "axes": [
        {"title": "Bias of recovered E_a", "grid": [1, 2, 0, 1, 0, 1], "series": []},
        {"title": "Spread of E_a", "grid": [1, 2, 0, 1, 1, 2], "series": []}]}
    note = _figure_record_note(record)
    assert 'left panel, "Bias of recovered E_a"' in note and 'right panel, "Spread of E_a"' in note
    assert "top" not in note and "bottom" not in note
    listing = _figure_list_for_prompt({"figures": ["fig1.png"], "figure_records": {"fig1.png": record}})
    assert "Where a panel sits (left, right, top, bottom) is FI's record" in listing
    # A record from before the grid was kept says nothing about place, rather than guess one.
    old = {"axes": [{"title": "a", "series": []}, {"title": "b", "series": []}]}
    assert "panel" not in _figure_record_note(old)
    assert "Where a panel sits" not in _figure_list_for_prompt({"figures": ["o.png"], "figure_records": {"o.png": old}})


def test_a_panel_spanning_the_grid_or_an_odd_grid_gets_no_guess() -> None:
    from core.engine import _panel_place

    assert _panel_place([2, 2, 0, 1, 0, 2]) == "top"            # spans both columns
    assert _panel_place([4, 1, 1, 2, 0, 1]) == "row 2 of 4"
    assert _panel_place(None) == "" and _panel_place([1, 2]) == "" and _panel_place([1, 2, 0, 1, 5, 6]) == ""
