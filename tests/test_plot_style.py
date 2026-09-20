"""Tests for the FrontierInsight matplotlib house style (core/plot_style.py).

Covers the pure config (rcParams, facecolor-by-paper-style), the generated
bootstrap's validity (a regression guard for the json-vs-repr boolean bug),
and — the load-bearing claim — that the bootstrap actually applies the style
in a *separate* subprocess the way the execute node runs experiments.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from core.plot_style import (
    BOOT_DIRNAME,
    FI_COLOR_CYCLE,
    FI_TEAL_CMAP_NAME,
    apply_style,
    fi_rcparams,
    figure_facecolor,
    write_boot,
    _boot_source,
)


# ---- facecolor tracks the paper style -----------------------------------


def test_figure_facecolor_briefing_is_warm_cream() -> None:
    assert figure_facecolor("briefing") == "#f8f7f3"


def test_figure_facecolor_latex_is_white() -> None:
    assert figure_facecolor("latex") == "white"


def test_figure_facecolor_unknown_defaults_white() -> None:
    # Anything that isn't the briefing style gets the white card.
    assert figure_facecolor("something_else") == "white"


# ---- rcParams ------------------------------------------------------------


def test_fi_rcparams_briefing_paints_all_faces_cream() -> None:
    rc = fi_rcparams("briefing")
    assert rc["figure.facecolor"] == "#f8f7f3"
    assert rc["savefig.facecolor"] == "#f8f7f3"
    assert rc["axes.facecolor"] == "#f8f7f3"


def test_fi_rcparams_latex_paints_all_faces_white() -> None:
    rc = fi_rcparams("latex")
    assert rc["figure.facecolor"] == "white"
    assert rc["axes.facecolor"] == "white"


def test_fi_rcparams_despines_and_sets_teal_heatmap() -> None:
    rc = fi_rcparams("briefing")
    assert rc["axes.spines.top"] is False
    assert rc["axes.spines.right"] is False
    assert rc["image.cmap"] == FI_TEAL_CMAP_NAME


# ---- generated bootstrap source -----------------------------------------


def test_boot_source_compiles_as_python() -> None:
    compile(_boot_source("briefing"), "<sitecustomize>", "exec")


def test_boot_source_uses_python_literals_not_json() -> None:
    """Regression: an earlier cut embedded the rc dict via json.dumps, so
    booleans rendered as JSON ``false``/``true`` and the bootstrap NameError'd
    at runtime (swallowed by its guard) — the style silently never applied.
    The rc dict must serialize with Python literals."""
    src = _boot_source("briefing")
    assert "False" in src                 # repr() of the despine flags
    assert ": false" not in src
    assert ": true" not in src
    assert ": null" not in src


def test_write_boot_creates_sitecustomize(tmp_path) -> None:
    boot_dir = write_boot(tmp_path, "briefing")
    assert boot_dir.name == BOOT_DIRNAME
    assert boot_dir == tmp_path / BOOT_DIRNAME
    assert (boot_dir / "sitecustomize.py").is_file()


# ---- in-process application ---------------------------------------------


@pytest.fixture
def _restore_rcparams():
    # matplotlib isn't in the fast-tier CI env; skip the tests that need it.
    matplotlib = pytest.importorskip("matplotlib")

    snapshot = dict(matplotlib.rcParams)
    try:
        yield
    finally:
        matplotlib.rcParams.update(snapshot)


def test_apply_style_registers_cmap_and_sets_brand_cycle(_restore_rcparams) -> None:
    import matplotlib

    apply_style("briefing")
    assert FI_TEAL_CMAP_NAME in matplotlib.colormaps
    first = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][0]
    assert first == FI_COLOR_CYCLE[0]
    assert matplotlib.rcParams["axes.facecolor"] == "#f8f7f3"
    assert matplotlib.rcParams["axes.spines.top"] is False


# ---- the load-bearing test: applies in a real subprocess ----------------


def test_bootstrap_applies_in_subprocess(tmp_path) -> None:
    """Drop the bootstrap, put it on a child process's PYTHONPATH, and confirm
    a script that imports ONLY matplotlib (never core.plot_style) inherits the
    full house style — exactly how the execute node styles experiments."""
    # The probe subprocess imports matplotlib; same interpreter, so if it's
    # absent here (fast-tier CI) the child can't import it either — skip.
    pytest.importorskip("matplotlib")
    boot_dir = write_boot(tmp_path, "briefing")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p
        ),
    }
    probe = (
        "import json, matplotlib; rc = matplotlib.rcParams; "
        "print(json.dumps({"
        "'face': rc['axes.facecolor'], "
        "'cyc': rc['axes.prop_cycle'].by_key()['color'][0], "
        "'cmap': rc['image.cmap'], "
        "'top': rc['axes.spines.top'], "
        "'reg': 'fi_teal' in matplotlib.colormaps}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["face"] == "#f8f7f3"
    assert data["cyc"] == "#0e6e6b"
    assert data["cmap"] == "fi_teal"
    assert data["top"] is False
    assert data["reg"] is True


def test_bootstrap_records_what_each_saved_figure_draws(tmp_path) -> None:
    """The validation quest's drift figure: forward Euler's 0.44 J spike on a
    linear axis, with RK4 and Velocity-Verlet flat at 0 under it."""
    pytest.importorskip("matplotlib")
    boot_dir = write_boot(tmp_path, "latex")
    records = tmp_path / "records"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p),
        "FI_FIGURE_RECORDS": str(records),
    }
    script = tmp_path / "plot.py"
    script.write_text(
        "import math\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "t = [i * 0.1 for i in range(2000)]\n"
        "fig, (drift, errors) = plt.subplots(1, 2)\n"
        "drift.plot(t, [0.44 * math.exp(-(x - 2) ** 2) for x in t], label='forward_euler')\n"
        "drift.plot(t, [1e-8 * math.sin(x) for x in t], label='rk4')\n"
        "drift.plot(t, [4e-5 * x / 200 for x in t], label='velocity_verlet')\n"
        "drift.axhline(0.0)\n"
        "drift.set_title('Long-term Energy Drift'); drift.set_ylabel('Energy Difference (J)')\n"
        "errors.set_yscale('log')\n"
        "errors.scatter([0.01, 0.05, 0.1], [5e-9, 3e-6, 5e-5], label='rk4')\n"
        "errors.plot([0.01, 0.05, 0.1], [float('nan'), 1e-3, 1e-1], label='euler')\n"
        "errors.set_ylim(1e-10, 1)\n"
        "plt.savefig('long_term_energy_drift.png')\n",
        encoding="utf-8",
    )
    out = subprocess.run([sys.executable, str(script)], env=env, cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "long_term_energy_drift.png").is_file()
    record = json.loads((records / "long_term_energy_drift.json").read_text(encoding="utf-8"))
    assert record["file"] == "long_term_energy_drift.png"
    drift, errors = record["axes"]
    assert drift["title"] == "Long-term Energy Drift" and drift["yscale"] == "linear"
    assert {s["label"]: s["shows"] for s in drift["series"]} == {
        "forward_euler": "yes", "rk4": "flat", "velocity_verlet": "flat",
    }
    assert {s["label"]: s["shows"] for s in errors["series"]} == {"euler": "yes", "rk4": "yes"}


def test_bootstrap_names_series_labelled_only_in_the_legend(tmp_path) -> None:
    """``ax.legend(["euler", ...])`` names the series without labelling them."""
    pytest.importorskip("matplotlib")
    boot_dir = write_boot(tmp_path, "latex")
    records = tmp_path / "records"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p),
        "FI_FIGURE_RECORDS": str(records),
    }
    probe = (
        "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([0, 1], [0, 10]); ax.plot([0, 1], [0, 0.01], '--')\n"
        "ax.scatter([0, 1], [5, 6]); ax.plot([0, 1], [3, 4], color='k', label='kept')\n"
        "ax.legend(['euler', 'rk4', 'points', 'kept'])\n"
        # Drawn like "kept" but not in the legend: it stays unnamed.
        "ax.axhline(0, color='k')\n"
        "fig.savefig('legend.png')\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], env=env, cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    (axes,) = json.loads((records / "legend.json").read_text(encoding="utf-8"))["axes"]
    assert [(s["label"], s["shows"]) for s in axes["series"]] == [
        ("euler", "yes"), ("rk4", "flat"), ("kept", "yes"), ("points", "yes"),
    ]


def test_bootstrap_joins_a_series_drawn_one_point_at_a_time(tmp_path) -> None:
    """A real quest plotted its final sizes with one scatter call per R0 and
    the label on the first call only. Recorded as that first point alone, each
    series read as "flat", and a caption that was right got flagged."""
    pytest.importorskip("matplotlib")
    boot_dir = write_boot(tmp_path, "latex")
    records = tmp_path / "records"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p),
        "FI_FIGURE_RECORDS": str(records),
    }
    probe = (
        "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "det = {0.9: 0.0001, 1.5: 0.58, 3.0: 0.94}\n"
        "sto = {0.9: 0.11, 1.5: 0.58, 3.0: 0.94}\n"
        "for r0 in det:\n"
        # No colour, as in the quest: each call takes the next one in the cycle.
        "    ax.scatter(r0, sto[r0], label='Stochastic' if r0 == 0.9 else '')\n"
        "    ax.scatter(r0, det[r0], marker='x', color='red', label='Deterministic' if r0 == 0.9 else '')\n"
        # A marker of its own is a point, not a flat series.
        "ax.plot([2.0], [0.5], 'o', color='k', label='single')\n"
        # A line that lies flat still does, and a reference line drawn in the
        # same style is not part of it: only markers join a series.
        "ax.plot([1, 3], [0.2, 0.2], color='k', label='level')\n"
        "ax.axhline(0.94, color='k')\n"
        "ax.legend(); fig.savefig('final_size.png')\n"
        # Two labelled round-marker series and a third-colour round point:
        # which series it belongs to is unknowable, so it joins neither.
        "fig2, ax2 = plt.subplots()\n"
        "ax2.scatter([1], [0.1], color='blue', label='a'); ax2.scatter([2], [0.2], color='green', label='b')\n"
        "ax2.scatter([3], [0.9], color='orange', label='')\n"
        "ax2.legend(); fig2.savefig('ambiguous.png')\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], env=env, cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    (axes,) = json.loads((records / "final_size.json").read_text(encoding="utf-8"))["axes"]
    series = {s["label"]: s for s in axes["series"]}
    assert set(series) == {"Stochastic", "Deterministic", "single", "level"}
    assert (series["Stochastic"]["min"], series["Stochastic"]["max"]) == pytest.approx((0.11, 0.94))
    assert (series["Deterministic"]["min"], series["Deterministic"]["max"]) == pytest.approx((0.0001, 0.94))
    assert {label: s["shows"] for label, s in series.items()} == {
        "Stochastic": "yes", "Deterministic": "yes", "single": "yes", "level": "flat",
    }
    (axes,) = json.loads((records / "ambiguous.json").read_text(encoding="utf-8"))["axes"]
    assert [(s["label"], s["min"], s["max"]) for s in axes["series"]] == [("a", 0.1, 0.1), ("b", 0.2, 0.2)]


def test_bootstrap_records_the_values_of_hlines_vlines_and_bands(tmp_path) -> None:
    """A real quest drew "Large-N final-size root" with ``ax.hlines`` at 0.94 and
    the record said it lay at 0. The offsets of a segment collection or a
    fill_between band are always [0, 0]; where they are drawn is in their
    segments and their polygons."""
    pytest.importorskip("matplotlib")
    boot_dir = write_boot(tmp_path, "latex")
    records = tmp_path / "records"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p),
        "FI_FIGURE_RECORDS": str(records),
    }
    probe = (
        "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([0, 1, 2], [0.1, 0.5, 0.9], label='line')\n"
        "ax.hlines(0.94, 0, 2, linestyle=':', label='root')\n"
        "ax.hlines(0.0, 0, 2, label='zero')\n"
        "ax.vlines(1.0, 0.2, 0.6, label='marker')\n"
        "ax.fill_between([0, 1, 2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8], alpha=0.2, label='band')\n"
        "ax.scatter([1.0], [0.5], label='dot')\n"
        # In axes coordinates the numbers are fractions of the axes, not data; and a
        # mesh has no y values of its own: neither is a series of this figure.
        "ax.vlines(0.5, 0, 1, transform=ax.get_xaxis_transform(), label='across the axes')\n"
        "ax.pcolormesh([[0.0, 1.0], [0.0, 1.0]], label='mesh')\n"
        "ax.set_ylim(0, 1)\n"
        "fig.savefig('collections.png')\n"
        # A band drawn in the log axis: only its positive corners count.
        "fig2, ax2 = plt.subplots()\n"
        "ax2.set_yscale('log'); ax2.fill_between([0, 1], [1e-3, 1e-2], [1e-1, 1.0], label='band')\n"
        "ax2.hlines(1e-2, 0, 1, label='floor'); fig2.savefig('log.png')\n"
        # A band drawn in the colour of a labelled scatter is not one of its points.
        "fig3, ax3 = plt.subplots()\n"
        "ax3.scatter([0.5], [0.5], color='C0', alpha=0.2, label='pts')\n"
        "ax3.fill_between([0, 1], [0.6, 0.6], [0.9, 0.9], color='C0', alpha=0.2)\n"
        "fig3.savefig('join.png')\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], env=env, cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    (axes,) = json.loads((records / "collections.json").read_text(encoding="utf-8"))["axes"]
    series = {s["label"]: (s["min"], s["max"], s["shows"]) for s in axes["series"]}
    assert series == {
        "line": (0.1, 0.9, "yes"),
        "root": (0.94, 0.94, "flat"),
        "zero": (0.0, 0.0, "flat"),
        "marker": (0.2, 0.6, "yes"),
        "band": (0.3, 0.8, "yes"),
        "dot": (0.5, 0.5, "yes"),
    }
    (axes,) = json.loads((records / "log.json").read_text(encoding="utf-8"))["axes"]
    series = {s["label"]: (s["min"], s["max"], s["shows"]) for s in axes["series"]}
    assert series == {"band": (1e-3, 1.0, "yes"), "floor": (1e-2, 1e-2, "flat")}
    (axes,) = json.loads((records / "join.json").read_text(encoding="utf-8"))["axes"]
    assert [(s["label"], s["min"], s["max"]) for s in axes["series"]] == [("pts", 0.5, 0.5)]


def _layout_records(tmp_path, probe: str):
    """Run ``probe`` under the bootstrap, as the executor does, and return a
    function from a saved figure's stem to its layout findings."""
    pytest.importorskip("matplotlib")
    boot_dir = write_boot(tmp_path, "latex")
    records = tmp_path / "records"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p),
        "FI_FIGURE_RECORDS": str(records),
    }
    out = subprocess.run([sys.executable, "-c", "import matplotlib; matplotlib.use('Agg')\n" + probe],
                         env=env, cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr

    def layout(stem: str) -> list[dict]:
        return json.loads((records / f"{stem}.json").read_text(encoding="utf-8"))["layout"]

    return layout


def test_bootstrap_records_a_suptitle_drawn_over_a_panel_title(tmp_path) -> None:
    """A real quest's three-panel figure: a short, wide figure with a suptitle
    and no room made for it, so it covers the middle panel's title."""
    layout = _layout_records(tmp_path, (
        "import matplotlib.pyplot as plt\n"
        "fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))\n"
        "for ax, n in zip(axes, (100, 1000, 5000)):\n"
        "    ax.hist([0.0, 0.0, 0.6, 0.62], bins=5); ax.set_title(f'N = {n}')\n"
        "plt.suptitle('Final Size Distribution for R0 = 1.5')\n"
        "fig.savefig('titles.png')\n"
        # The same figure with the room made: nothing is drawn over anything.
        "fig2, axes2 = plt.subplots(1, 3, sharey=True, figsize=(12, 4))\n"
        "for ax, n in zip(axes2, (100, 1000, 5000)):\n"
        "    ax.hist([0.0, 0.0, 0.6, 0.62], bins=5); ax.set_title(f'N = {n}')\n"
        "fig2.suptitle('Final Size Distribution for R0 = 1.5'); fig2.subplots_adjust(top=0.76)\n"
        "fig2.savefig('room.png')\n"
    ))
    found = layout("titles")
    assert found and {f["check"] for f in found} == {"title_overlap"}
    assert "N = 1000" in {f["over"] for f in found}
    assert all(f["text"] == "Final Size Distribution for R0 = 1.5" and f["share"] >= 0.2 for f in found)
    assert layout("room") == []


def test_bootstrap_records_a_legend_drawn_over_the_lines_it_names(tmp_path) -> None:
    layout = _layout_records(tmp_path, (
        "import matplotlib.pyplot as plt\n"
        # A legend placed where the lines run.
        "fig, ax = plt.subplots()\n"
        "ax.plot([0, 1], [0, 1], label='rising'); ax.plot([0, 1], [0.5, 0.5], label='level')\n"
        "ax.legend(loc='center'); fig.savefig('covered.png')\n"
        # A legend in an empty corner, with a title and no suptitle.
        "fig2, ax2 = plt.subplots()\n"
        "ax2.plot([0, 1], [0, 0.2], label='low'); ax2.set_ylim(0, 1)\n"
        "ax2.legend(loc='upper left'); ax2.set_title('Clean'); fig2.savefig('clean.png')\n"
    ))
    (found,) = layout("covered")
    assert found["check"] == "legend_over_data" and found["legend"] == ["rising", "level"]
    assert found["line_px"] > 0
    assert layout("clean") == []


def test_bootstrap_records_a_legend_drawn_over_scatter_points(tmp_path) -> None:
    layout = _layout_records(tmp_path, (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.scatter([0.1, 0.5, 0.9], [0.1, 0.5, 0.9], label='runs'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)\n"
        "ax.legend(loc='center'); fig.savefig('covered.png')\n"
        "fig2, ax2 = plt.subplots()\n"
        "ax2.scatter([0.1, 0.5, 0.9], [0.1, 0.5, 0.9], label='runs'); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)\n"
        "ax2.legend(loc='upper left'); fig2.savefig('clean.png')\n"
    ))
    (found,) = layout("covered")
    assert found["check"] == "legend_over_data" and found["legend"] == ["runs"] and found["points"] >= 1
    assert layout("clean") == []


def test_bootstrap_measures_a_legend_on_a_log_axis_where_it_is_drawn(tmp_path) -> None:
    """On a log-log axis a diagonal line is still a diagonal: the legend is
    measured in pixels, not in data units. Data that lies past the axis limits
    is clipped, so a legend beside the axes has nothing under it."""
    layout = _layout_records(tmp_path, (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.loglog([1, 10, 100], [1, 10, 100], label='power law'); ax.legend(loc='center'); fig.savefig('covered.png')\n"
        "fig2, ax2 = plt.subplots()\n"
        "ax2.loglog([1, 10, 100], [1, 10, 100], label='power law'); ax2.legend(loc='upper left')\n"
        "fig2.savefig('clean.png')\n"
        # The line runs on past x = 5, where it is clipped; the legend sits there.
        "fig3, ax3 = plt.subplots()\n"
        "ax3.plot([0, 10], [0, 10], label='a'); ax3.set_xlim(0, 5)\n"
        "ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5)); fig3.savefig('beside.png')\n"
    ))
    assert [f["check"] for f in layout("covered")] == ["legend_over_data"]
    assert layout("clean") == []
    assert layout("beside") == []


def test_bootstrap_records_nothing_without_a_records_folder(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    boot_dir = write_boot(tmp_path, "latex")
    env = {k: v for k, v in os.environ.items() if k != "FI_FIGURE_RECORDS"}
    env["PYTHONPATH"] = str(boot_dir)
    probe = ("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; "
             "plt.plot([1, 2], label='a'); plt.savefig('a.png')")
    out = subprocess.run([sys.executable, "-c", probe], env=env, cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".json") == []
