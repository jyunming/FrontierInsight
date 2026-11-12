"""The coarse scan's map puts every scanned point at its own position.

The two most-varied settings are the axes; every combination of the further settings gets its own panel (named above it), so
points that differ only in a further setting never share a position. More than 12 panels are drawn as several figures. Repeat
runs of one design are one marker at the design's setting, coloured by their mean, and the note gives their number."""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from core import analysis_figures as af  # noqa: E402


@pytest.fixture
def made(monkeypatch):
    figures = []
    orig = af._new_figure

    def spy(*a, **k):
        got = orig(*a, **k)
        figures.append(got[1])
        return got

    monkeypatch.setattr(af, "_new_figure", spy)
    return figures


def _panels(fig):
    return [ax for ax in fig.axes if ax.get_visible() and ax.collections and ax.get_label() != "<colorbar>"]


def _positions(ax):
    out = []
    for coll in ax.collections:
        out += [tuple(o) for o in coll.get_offsets().tolist()]
    return out


def _three(c_values=(0, 1)):
    return [{"design": {"a": a, "b": b, "c": c}, "objective": a + b + 10 * c} for a in (1, 2, 3) for b in (1, 2, 3) for c in c_values]


def _sixteen():
    """Two settings of five values on the axes, two further settings of four values: 16 combinations."""
    return [{"design": {"a": a, "b": b, "c": c, "d": d}, "objective": float(a + b + c + d)}
            for a in range(1, 6) for b in range(1, 6) for c in range(1, 5) for d in range(1, 5)]


def test_points_that_differ_only_in_a_further_setting_are_in_separate_panels(tmp_path, made):
    out = af._scan_figure({"scan": _three()}, str(tmp_path))
    assert [o["status"] for o in out] == ["ok"]
    panels = _panels(made[0])
    assert len(panels) == 2 and [ax.get_title() for ax in panels] == ["c = 0", "c = 1"]
    for ax in panels:
        pos = _positions(ax)
        assert len(pos) == 9 and len(set(pos)) == 9  # every point has its own position
    assert "one panel for each combination" in out[0]["note"] and "c" in out[0]["note"]


def test_one_colour_scale_is_shared_by_every_panel_and_figure(tmp_path, made):
    af._scan_figure({"scan": _sixteen()}, str(tmp_path))
    limits = {coll.get_clim() for fig in made for ax in _panels(fig) for coll in ax.collections}
    assert len(made) == 2 and limits == {(4.0, 18.0)}


def test_more_than_twelve_combinations_are_drawn_as_several_figures(tmp_path, made):
    scan = _sixteen()
    out = af._scan_figure({"scan": scan}, str(tmp_path))
    assert [(o["id"], o["file"], o["status"]) for o in out] == [
        ("search_scan_map", af.SCAN_FILE, "ok"), ("search_scan_map_2", "search_scan_map_2.png", "ok")]
    assert [len(_panels(fig)) for fig in made] == [12, 4]
    assert all((tmp_path / o["file"]).is_file() for o in out)
    assert "part 1 of 2" in out[0]["note"] and "part 2 of 2" in out[1]["note"]
    for fig in made:
        for ax in _panels(fig):
            pos = _positions(ax)
            assert len(pos) == 25 and len(set(pos)) == 25


def test_a_part_left_from_an_earlier_run_is_removed(tmp_path, made):
    (tmp_path / "search_scan_map_2.png").write_bytes(b"old")
    out = af._scan_figure({"scan": _three()}, str(tmp_path))
    assert len(out) == 1 and not (tmp_path / "search_scan_map_2.png").exists()


def test_repeat_runs_of_one_design_are_one_marker_with_their_count(tmp_path, made):
    scan = [{"design": {"a": a, "b": b}, "objective": float(a + b)} for a in (1, 2) for b in (1, 2)]
    scan += [{"design": {"a": 1, "b": 1}, "objective": 4.0}, {"design": {"a": 1, "b": 1}, "objective": 5.0}]
    out = af._scan_figure({"scan": scan}, str(tmp_path))
    (ax,) = _panels(made[0])
    pos = _positions(ax)
    assert len(pos) == 4 and len(set(pos)) == 4
    colours = {tuple(o): v for coll in ax.collections for o, v in zip(coll.get_offsets().tolist(), coll.get_array().tolist())}
    assert colours[(1.0, 1.0)] == pytest.approx((2.0 + 4.0 + 5.0) / 3)  # the mean of its three runs
    assert "3 runs" in out[0]["note"] and "a = 1, b = 1" in out[0]["note"]


def test_two_varying_settings_without_repeats_are_one_panel_with_no_note(tmp_path, made):
    scan = [{"design": {"a": a, "b": b}, "objective": a * 1.5 + b} for a in (1, 2, 3) for b in (10, 20)]
    out = af._scan_figure({"scan": scan}, str(tmp_path))
    assert [o["status"] for o in out] == ["ok"] and "note" not in out[0]
    (ax,) = _panels(made[0])
    assert ax.get_legend() is None and len(_positions(ax)) == 6


def test_search_figures_lists_each_part_as_its_own_outcome(tmp_path, monkeypatch):
    scan = _sixteen()
    monkeypatch.setattr(af, "_read_record", lambda root: {"record": {}, "rows": [], "scan": scan, "objective": {}})
    files = [o["file"] for o in af.search_figures(str(tmp_path), str(tmp_path / "figs"))]
    assert files[-2:] == [af.SCAN_FILE, "search_scan_map_2.png"]
