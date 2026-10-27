"""What each experiment figure draws reaches the writer (``core/engine.py`` figure records, recorded by
``core/plot_style.py``); the captions themselves are FI's (``tests/test_fi_places_figures.py``)."""

from __future__ import annotations

import json
from pathlib import Path

from core.engine import _figure_list_for_prompt, _read_figure_records

# The validation quest's figure 3, as the bootstrap records it.
DRIFT = {
    "file": "long_term_energy_drift.png",
    "axes": [{
        "title": "Long-term Energy Drift (h=0.01)", "xlabel": "Time (s)", "ylabel": "Energy Difference ΔE (J)",
        "yscale": "linear", "ylim": [-0.022, 0.462],
        "series": [
            {"label": "forward_euler", "min": 0.0, "max": 0.44, "shows": "yes"},
            {"label": "rk4", "min": -1.1e-08, "max": 1.2e-08, "shows": "flat"},
            {"label": "velocity_verlet", "min": 0.0, "max": 3.9e-05, "shows": "flat"},
        ],
    }],
}
PAPER = (
    "# P\n\n## Results\n\n"
    "![**Figure 1.** Error of Forward Euler and RK4.](figures/error.png)\n\n"
    "![**Figure 3.** Long-term energy drift rates for RK4 and Velocity-Verlet.](figures/long_term_energy_drift.png)\n"
)


def test_records_are_read_by_figure_file_name(tmp_path: Path) -> None:
    (tmp_path / "long_term_energy_drift.json").write_text(json.dumps(DRIFT), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    records = _read_figure_records(tmp_path, ["long_term_energy_drift.png", "broken.png", "missing.png"])
    assert list(records) == ["long_term_energy_drift.png"]


def test_the_writer_sees_what_each_figure_draws() -> None:
    listing = _figure_list_for_prompt({
        "figures": ["error.png", "long_term_energy_drift.png"],
        "figure_records": {"long_term_energy_drift.png": DRIFT},
    })
    lines = listing.splitlines()
    assert lines[0] == "- Figure 1: figures/error.png"
    assert lines[1].startswith('- Figure 2: figures/long_term_energy_drift.png — "Long-term Energy Drift (h=0.01)", '
                               'y axis "Energy Difference ΔE (J)" (linear, -0.022 to 0.462), series: ')
    assert "forward_euler 0 to 0.44;" in lines[1]
    assert "rk4 -1.1e-08 to 1.2e-08, FLAT on this axis" in lines[1]
    assert "velocity_verlet 0 to 3.9e-05, FLAT on this axis" in lines[1]
    assert "does not describe how such a series changes" in lines[2]
    # No record, no note.
    assert _figure_list_for_prompt({"figures": ["a.png"]}) == "- Figure 1: figures/a.png"


# --- a figure that draws unlabelled dashed lines: they are in its record, and the writer is told -------------------

def _line(label, style, marker, ys, kind="data"):
    return {"label": label, "kind": kind, "x": [0.005, 0.01, 0.02, 0.05, 0.1][:len(ys)], "y": ys, "linestyle": style,
            "marker": marker}


SPREAD = {
    "file": "spread.png",
    "axes": [
        {"title": "Spread of $E_a$ vs noise", "ylabel": "Relative SD", "yscale": "log", "ylim": [0.001, 0.15],
         "series": [{"label": "n=5", "min": 0.0065, "max": 0.1186, "shows": "yes"},
                    {"label": "n=8", "min": 0.005, "max": 0.095, "shows": "yes"}]},
        {"title": "Bias relative to spread", "ylabel": "Bias / SD", "ylim": [-0.5, 0.6],
         "series": [{"label": "n=5", "min": -0.09, "max": 0.40, "shows": "yes"}]},
    ],
}
SPREAD_LINES = {
    "file": "spread.png",
    "axes": [
        {"title": "Spread of $E_a$ vs noise", "lines": [
            _line("n=5", "-", "o", [0.0065, 0.0125, 0.0231, 0.062, 0.1186]),
            _line("_child1", "--", "None", [0.0054, 0.0108, 0.0216, 0.0541, 0.1083]),
            _line("n=8", "-", "o", [0.005, 0.0097, 0.0166, 0.0464, 0.095]),
            _line("_child3", "--", "None", [0.0043, 0.0086, 0.0171, 0.0428, 0.0856])]},
        {"title": "Bias relative to spread", "lines": [
            _line("n=5", "-", "o", [-0.09, 0.1, 0.4, 0.0, 0.2]),
            _line("_child20", "--", "None", [0.0, 0.0], kind="hline")]},
    ],
}


def _spread_records(tmp_path: Path, lines: dict | None = None) -> dict:
    (tmp_path / "spread.json").write_text(json.dumps(SPREAD), encoding="utf-8")
    (tmp_path / "spread.seed0.json").write_text(json.dumps(lines or SPREAD_LINES), encoding="utf-8")
    return _read_figure_records(tmp_path, ["spread.png"])


def test_the_dashed_lines_a_figure_draws_without_a_label_are_in_its_record(tmp_path: Path) -> None:
    record = _spread_records(tmp_path)["spread.png"]
    first, second = record["axes"]
    assert first["unlabelled_lines"] == [{"style": "dashed", "count": 2, "min": 0.0043, "max": 0.1083}]
    assert second["unlabelled_lines"] == [{"style": "dashed", "count": 1, "min": 0.0, "max": 0.0}]


def test_a_record_without_the_line_list_has_no_unlabelled_lines(tmp_path: Path) -> None:
    (tmp_path / "spread.json").write_text(json.dumps(SPREAD), encoding="utf-8")
    (record,) = _read_figure_records(tmp_path, ["spread.png"]).values()
    assert all("unlabelled_lines" not in ax for ax in record["axes"])
    # Solid unlabelled lines (a family of runs) and markers or error-bar caps are not reference lines.
    solid = {"file": "spread.png", "axes": [{"title": "a", "lines": [
        _line("_child1", "-", "None", [1.0, 2.0, 3.0]), _line("_child2", "None", "_", [1.0, 1.0]),
        _line("_child3", "--", "o", [1.0, 2.0])]}, {"title": "b", "lines": []}]}
    record = _spread_records(tmp_path, solid)["spread.png"]
    assert all("unlabelled_lines" not in ax for ax in record["axes"])


def test_the_writer_is_told_about_the_lines_the_legend_does_not_name(tmp_path: Path) -> None:
    records = _spread_records(tmp_path)
    note = _figure_list_for_prompt({"figures": ["spread.png"], "figure_records": records})
    assert "also draws 2 dashed lines with no legend label (0.0043 to 0.108)" in note
    assert "also draws 1 dashed line with no legend label (0 to 0)" in note
