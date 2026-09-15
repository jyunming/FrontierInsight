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
