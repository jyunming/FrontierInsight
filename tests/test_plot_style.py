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
    import matplotlib

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
