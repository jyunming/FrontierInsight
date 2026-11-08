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
        size, temp = (float(part.split("=")[1]) for part in key.split(","))
        return temp + size - 1  # one value per curve and x, so the drawn curves are the table's and no two are identical


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


def _legend_texts(h):
    legend = h.axes[0].get_legend() or h.fig.legends[0]  # inside the panel, or the figure's own when there are many entries
    return [t.get_text() for t in legend.get_texts()]


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
        h.line(0, t, [1, 2, 3], [t, t + 1.0, t + 2.0], "growth")
    series = h.record()["panels"][0]["series"]
    assert [s["label"] for s in series] == ["Temperature (K) = 10", "Temperature (K) = 20", "Temperature (K) = 30"]
    assert len({s["id"] for s in series}) == 3 and series[0]["level"]["value"] == 10
    assert "3 curves of Growth, one per Temperature (K)" in _caption(h)
    h.finish({}, {})
    assert _legend_texts(h) == [s["label"] for s in series]


def test_many_levels_are_named_by_the_lowest_and_the_highest_and_the_caption_says_the_range():
    temps = tuple(range(1, 21))
    h = _Host(temps)
    for t in temps:
        h.line(0, t, [1, 2, 3], [t, t + 1.0, t + 2.0], "growth")
    h.finish({}, {})
    assert _legend_texts(h) == ["Temperature (K) = 1", "Temperature (K) = 20"]
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


def test_fi_default_figure_is_held_to_the_series_setting_like_a_body():
    h = _Host(fallback=True)
    with pytest.raises(ValueError, match="names a result, not a curve"):
        h.line(0, "growth", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.line(0, 10, [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    assert h.record()["panels"][0]["series"][0]["level"]["value"] == 10


# ---- fix round 1 -------------------------------------------------------------------------------------------------------------

import json  # noqa: E402

import numpy as np  # noqa: E402

from core import figure_claims  # noqa: E402


def test_a_level_drawn_as_line_and_points_is_one_curve():
    temps = tuple(range(1, 8))
    h = _Host(temps)
    for t in temps:
        h.line(0, t, [1, 2, 3], [t, t + 1.0, t + 2.0], "growth")
        h.points(0, t, [1, 2, 3], [t, t + 1.0, t + 2.0], "growth")
    h.finish({}, {})
    assert "7 curves of Growth, one per Temperature (K)" in _caption(h)
    assert "14 curves" not in _caption(h)
    legend = [t.get_text() for t in h.fig.legends[0].get_texts()]  # more than six entries: the figure's legend, outside
    assert legend == ["Temperature (K) = %d" % t for t in temps]  # not compacted at 7 levels, named once each


def test_thirteen_levels_drawn_as_line_and_points_count_thirteen_and_name_the_ends():
    temps = tuple(range(1, 14))
    h = _Host(temps)
    for t in temps:
        h.line(0, t, [1, 2, 3], [t, t + 1.0, t + 2.0], "growth")
        h.points(0, t, [1, 2, 3], [t, t + 1.0, t + 2.0], "growth")
    h.finish({}, {})
    assert "13 curves of Growth, one per Temperature (K) (from 1 to 13" in _caption(h)
    assert _legend_texts(h) == ["Temperature (K) = 1", "Temperature (K) = 13"]


def test_the_line_markers_error_bars_and_band_of_one_level_share_a_colour():
    from matplotlib.colors import to_rgb
    h = _Host()
    for t in (10, 20):
        h.line(0, t, [1, 2, 3], [float(t), 2.0, 3.0], "growth", yerr=[0.1, 0.1, 0.1])
        h.points(0, t, [1, 2, 3], [float(t), 2.0, 3.0], "growth")
        h.band(0, t, [1, 2, 3], [t - 1.0, 1.0, 2.0], [t + 1.0, 3.0, 4.0], "growth", "other")
    h.finish({}, {})
    extra = h._fc_extra[0]
    rec = h.record()["panels"][0]["series"]
    for level in (10, 20):
        ks = [k for k, e in enumerate(rec) if e["level"]["value"] == level]
        line_rgb = to_rgb(extra["series"][ks[0]][0].get_color())
        assert to_rgb(extra["series"][ks[1]][0].get_color()) == line_rgb  # the markers
        bars = extra["bars"][ks[0]]
        assert to_rgb(bars.lines[2][0].get_color()[0]) == line_rgb  # the error bars' stems
        assert all(to_rgb(c.get_color()) == line_rgb for c in bars.lines[1])  # and their caps
        band = [art for sid, art in extra["bands"] if sid == rec[ks[0]]["id"]][0]
        assert tuple(band.get_facecolor()[0][:3]) == pytest.approx(line_rgb)  # the band
    assert to_rgb(extra["series"][0][0].get_color()) != to_rgb(extra["series"][2][0].get_color())


def test_numpy_integer_and_float_levels_are_levels():
    h = _Host()
    h.line(0, np.int64(10), [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.line(0, np.float64(20.0), [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.line(0, np.int32(30), [1, 2, 3], [1.0, 2.0, 5.0], "growth")
    series = h.record()["panels"][0]["series"]
    assert [s["level"]["text"] for s in series] == ["10", "20", "30"]
    assert [s["label"] for s in series] == ["Temperature (K) = 10", "Temperature (K) = 20", "Temperature (K) = 30"]
    json.dumps(h.record())  # a plain number in the record, never numpy's


def test_a_bool_is_no_level():
    with pytest.raises(ValueError, match="is not a value of Temperature"):
        _Host().line(0, True, [1, 2, 3], [1.0, 2.0, 3.0], "growth")


def test_a_level_that_is_the_sum_of_floats_matches_the_table_level():
    h = _Host((0.3, 0.5))
    h.line(0, 0.1 + 0.2, [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    assert h.record()["panels"][0]["series"][0]["level"]["text"] == "0.3"


def test_tiny_levels_are_matched_by_their_own_scale_not_a_fixed_absolute_tolerance():
    # Values on a 1e-12 scale: 4e-12 is no level (an absolute tolerance of 1e-12 took it for 3e-12), and 1e-12 is one level only.
    with pytest.raises(ValueError, match="is not a value of Temperature"):
        _Host((1e-12, 3e-12)).line(0, 4e-12, [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h = _Host((1e-12, 3e-12))
    h.line(0, 1e-12, [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    assert h.record()["panels"][0]["series"][0]["level"]["text"] == "1e-12"
    assert h._keys_at("temp", 1e-12) == ["size=1,temp=1e-12", "size=2,temp=1e-12", "size=3,temp=1e-12"]
    assert h._keys_at("temp", 0.0) == []


def test_the_level_curve_id_has_no_comma_or_equals_sign():
    h = _Host()
    h.line(0, 10, [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.line(0, 20, [1, 2, 3], [1.0, 2.0, 4.0], "growth")
    ids = [s["id"] for s in h.record()["panels"][0]["series"]]
    assert ids == ["growth@temp:10", "growth@temp:20"]
    assert all("," not in i and "=" not in i for i in ids)


def _claim_calls():
    h = _Host()
    h.line(0, 10, [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.line(0, 20, [1, 2, 3], [2.0, 4.0, 6.0], "growth")
    h.line(0, 30, [1, 2, 3], [2.0, 4.0, 6.0], "growth")
    return h.record()


def test_a_true_claim_about_one_level_curve_is_confirmed():
    calls = _claim_calls()
    args = figure_claims.parse_args("series=growth@temp:20, panel=1")
    assert args == {"series": "growth@temp:20", "panel": "1"}
    got = figure_claims.evaluate(calls, "trend", args)
    assert got is not None and "Temperature (K) = 20 increases" in got[0]
    assert figure_claims.evaluate(calls, "drawn", args) is not None
    same = figure_claims.evaluate(calls, "identical", {"a": "growth@temp:20", "b": "growth@temp:30", "panel": "1"})
    assert same is not None  # two levels that drew the same numbers


def test_a_false_claim_about_one_level_curve_is_caught():
    calls = _claim_calls()
    # the curves at 10 and 20 are not the same numbers; a curve of a level that was not drawn is no curve
    assert figure_claims.evaluate(calls, "identical", {"a": "growth@temp:10", "b": "growth@temp:20", "panel": "1"}) is None
    assert figure_claims.evaluate(calls, "drawn", {"series": "growth@temp:99", "panel": "1"}) is None
    assert figure_claims.evaluate(calls, "value_at", {"series": "growth@temp:10", "panel": "1", "x": "7"}) is None
    offers = [o for o in figure_claims.menu(calls) if o["args"].get("series") == "growth@temp:20"]
    assert offers and all("," not in o["args"]["series"] and "=" not in o["args"]["series"] for o in offers)


def test_a_claim_about_a_level_curve_in_the_paper_is_filled_and_a_false_one_is_not(tmp_path):
    from core import figure_facts, paper_facts
    folder = tmp_path / figure_facts.RECORDS_DIR
    folder.mkdir(parents=True)
    (folder / "gr.calls.json").write_text(json.dumps(_claim_calls()), encoding="utf-8")
    good = paper_facts.render("A. {{figure:gr:trend @ series=growth@temp:20, panel=1}} B.", tmp_path)
    assert "Temperature (K) = 20 increases over the drawn points" in good.text and not good.unfilled
    bad = paper_facts.render("A. {{figure:gr:identical @ a=growth@temp:10, b=growth@temp:20, panel=1}} B.", tmp_path)
    assert bad.unfilled == ("figure:gr:identical",)


# ---- a curve whose values are not the table's is named for what it is ---------------------------------------------------------

def test_curves_whose_values_are_not_the_tables_are_captioned_and_legended_as_a_computed_value():
    h = _Host()
    for t in (10, 20, 30):
        h.line(0, t, [1, 2, 3], [1.0 + t, 2.0, 3.0], "growth")  # not the table's value of the result
    series = h.record()["panels"][0]["series"]
    assert "3 curves of computed value, one per Temperature (K)" in _caption(h)
    assert "Growth" not in _caption(h)
    h.finish({}, {})
    assert _legend_texts(h) == [s["label"] for s in series] == ["Temperature (K) = %d" % t for t in (10, 20, 30)]


def test_many_curves_that_are_not_the_tables_values_say_the_range_of_a_computed_value():
    temps = tuple(range(1, 21))
    h = _Host(temps)
    for t in temps:
        h.line(0, t, [1, 2, 3], [t * 1e-10, 2e-10, 3e-10], "growth")  # no value of the table's
    h.finish({}, {})
    assert _legend_texts(h) == ["Temperature (K) = 1", "Temperature (K) = 20"]
    assert "20 curves of computed value, one per Temperature (K) (from 1 to 20" in _caption(h)
