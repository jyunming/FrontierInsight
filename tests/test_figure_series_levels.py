"""A figure whose plan draws one curve per level of a setting: each curve is named by its level, and the caption counts them
(core/frame_figure.py, the caption in core/engine.py). Made-up study: a colony whose ``growth`` is read at settings of ``size``
for each value of ``temp``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import engine, frame_figure  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _Grid:
    declared = ["growth"]

    def __init__(self, temps):
        self.temps = temps

    def keys(self):
        return ["size=%d,temp=%s" % (s, t) for t in self.temps for s in (1, 2, 3)]

    def mean(self, name, key):
        return 1.0


class _Host(frame_figure.FigureCalls):
    def __init__(self, temps=(10, 20, 30), series="temp", fallback=False):
        self.fig, axes = plt.subplots(1, 1, squeeze=False)
        self.axes = list(axes[0])
        self.table = _Grid(temps)
        self.x = "size"
        self.series = series
        self.fallback = fallback
        self.settings = ["size", "temp"]
        self.series_ids = []
        self.labels = {"growth": "Growth", "size": "Size", "temp": "Temperature (K)"}


def _caption(h):
    return engine._figure_caption_text("growth.png", None, n_seeds=None, calls=h.record())


def test_a_result_id_as_the_series_of_a_curve_per_level_is_a_plain_error():
    h = _Host()
    with pytest.raises(ValueError) as err:
        h.line(0, "growth", [1, 2, 3], [1.0, 1.0, 1.0], "growth")
    text = str(err.value)
    assert "one curve per Temperature (K)" in text and "pass each curve's Temperature (K) as its series" in text
    assert "10, 20, 30" in text


def test_a_value_that_is_no_level_of_the_setting_is_refused():
    with pytest.raises(ValueError, match="is not a value of Temperature"):
        _Host().line(0, 99, [1, 2, 3], [1.0, 1.0, 1.0], "growth")


def test_each_curve_is_named_by_its_level_and_the_caption_counts_them():
    h = _Host()
    for t in (10, 20, 30):
        h.line(0, t, [1, 2, 3], [1.0 + t, 2.0, 3.0], "growth")
    series = h.record()["panels"][0]["series"]
    assert [s["label"] for s in series] == ["Temperature (K) = 10", "Temperature (K) = 20", "Temperature (K) = 30"]
    assert len({s["id"] for s in series}) == 3 and series[0]["level"]["value"] == 10
    assert "3 curves of Growth, one per Temperature (K)" in _caption(h)
    h.finish({}, {})
    assert [t.get_text() for t in h.axes[0].get_legend().get_texts()] == [s["label"] for s in series]


def test_many_levels_are_named_by_the_lowest_and_the_highest_and_the_caption_says_the_range():
    temps = tuple(range(1, 21))
    h = _Host(temps)
    for t in temps:
        h.line(0, t, [1, 2, 3], [float(t), 2.0, 3.0], "growth")
    h.finish({}, {})
    assert [t.get_text() for t in h.axes[0].get_legend().get_texts()] == ["Temperature (K) = 1", "Temperature (K) = 20"]
    text = _caption(h)
    assert "20 curves of Growth, one per Temperature (K) (from 1 to 20" in text


def test_a_band_names_its_curve_by_the_level_too():
    h = _Host()
    h.line(0, 10, [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.band(0, 10, [1, 2, 3], [0.5, 1.5, 2.5], [1.5, 2.5, 3.5], "growth", "other")
    assert len(h.record()["panels"][0]["bands"]) == 1


def test_the_same_level_twice_on_one_panel_is_an_error():
    h = _Host()
    h.line(0, 10, [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    with pytest.raises(ValueError, match="a second time on panel 0 with the same name"):
        h.line(0, 10, [1, 2, 3], [2.0, 3.0, 4.0], "growth")


def test_the_same_id_twice_on_one_panel_is_an_error_without_a_series_setting():
    h = _Host(series=None)
    h.series_ids = ["fast"]
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    with pytest.raises(ValueError, match="a second time"):
        h.line(0, "fast", [1, 2, 3], [2.0, 3.0, 4.0], "growth")


def test_a_figure_without_a_series_setting_is_drawn_as_before():
    h = _Host(series=None)
    h.series_ids = ["fast"]
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.points(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")  # a line and its markers are one series drawn in two kinds
    s = h.record()["panels"][0]["series"]
    assert len(s) == 2 and all("level" not in e for e in s)
    assert "one per" not in _caption(h)


def test_fi_default_figure_is_not_held_to_the_series_setting():
    h = _Host(fallback=True)
    h.series_ids = ["growth, temp = 10"]
    h.line(0, "growth, temp = 10", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    assert "level" not in h.record()["panels"][0]["series"][0]
