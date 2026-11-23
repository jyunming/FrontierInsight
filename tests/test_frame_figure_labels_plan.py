"""A drawn curve keeps the plan's name of the result its code drew it as (core/frame_figure.py ``_add_series``, ``_set_y``,
``_write_labels``): FI's check of the values stays in the record (``declared_as``, ``what``), it never renames a curve. A
curve FI compared and found not to be its record is left out of the figure (the panel record's ``left_out``). Made-up
study: a culture's ``growth`` read at settings of ``dose``, one curve per ``feed`` in the figure with levels."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import frame_figure  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _Table:
    declared = ["growth", "uptake"]

    def __init__(self, levels):
        self.levels = levels

    def keys(self):
        if not self.levels:
            return ["dose=%d" % d for d in (1, 2, 3)]
        return ["dose=%d,feed=%d" % (d, f) for f in (1, 2) for d in (1, 2, 3)]

    def mean(self, name, key):
        got = dict(part.split("=") for part in key.split(","))
        dose, feed = float(got["dose"]), float(got.get("feed", 1))
        return 2 * dose + feed if name == "growth" else dose * feed


LABELS = {"growth": "Growth (1/h)", "uptake": "Uptake (mg)", "dose": "Dose (mg)", "feed": "Feed level",
          "yield_ratio": "Yield ratio", "arm_a": "Arm A"}


class _Host(frame_figure.FigureCalls):
    def __init__(self, levels=False):
        self.fig, axes = plt.subplots(1, 1, squeeze=False)
        self.axes = list(axes[0])
        self.table = _Table(levels)
        self.x = "dose"
        self.series = "feed" if levels else None
        self.fallback = False
        self.settings = ["dose", "feed"] if levels else ["dose"]
        self.series_ids = ["arm_a", "arm_b"]  # arm_b has no label in the plan
        self.derived_ids = ["yield_ratio"]  # a quantity FI holds no table value of: never compared
        self.labels = dict(LABELS)


def _legend(h):
    leg = h.axes[0].get_legend()
    return [t.get_text() for t in leg.get_texts()] if leg else []


def _words(h):
    """Every word FI writes on the figure and in its record."""
    ax = h.axes[0]
    return " | ".join([ax.get_xlabel(), ax.get_ylabel(), *_legend(h), json.dumps(h.record())])


def test_a_series_fi_could_not_compare_keeps_the_plan_label_on_legend_and_axis():
    h = _Host()
    h.line(0, "yield_ratio", [1, 2, 3], [0.5, 0.6, 0.7], "yield_ratio")
    h.finish(h.labels, {})
    s = h.record()["panels"][0]["series"][0]
    assert s["what"] == "other"  # no claim is offered from it
    assert s["declared_as"] == {"id": "yield_ratio", "label": "Yield ratio", "checked": "not compared"}
    assert _legend(h) == ["Yield ratio"] and h.axes[0].get_ylabel() == "Yield ratio"
    y = h.record()["panels"][0]["y"]
    assert y["what"] == "other" and y["shown"] == "yield_ratio"


def test_a_level_curve_fi_could_not_compare_keeps_the_result_label():
    h = _Host(levels=True)
    for f in (1, 2):
        h.line(0, f, [1, 2, 3], [0.5 * f, 0.6 * f, 0.7 * f], "yield_ratio")
    h.finish(h.labels, {})
    series = h.record()["panels"][0]["series"]
    assert [s["level"]["result_label"] for s in series] == ["Yield ratio", "Yield ratio"]
    assert all(s["level"]["checked"] == "not compared" and s["what"] == "other" for s in series)
    assert h.axes[0].get_ylabel() == "Yield ratio"
    assert _legend(h) == ["Feed level = 1", "Feed level = 2"]


def test_a_series_fi_found_differs_is_not_drawn_and_is_listed_as_left_out():
    h = _Host()
    arts = h.line(0, "growth", [1, 2, 3], [9.0, 9.5, 9.9], "growth")  # FI's record of growth is 3, 5, 7: not these
    assert arts == []
    rec = h.record()["panels"][0]
    assert rec["series"] == []
    assert rec["left_out"] == [{"id": "growth", "kind": "curve", "declared": "growth", "label": "Growth (1/h)",
                                "name": "Growth (1/h)", "setting": None, "setting_label": None,
                                "reason": "the drawn values are not FI's record of Growth (1/h)"}]
    assert not h.axes[0].lines


def test_a_level_curve_fi_found_differs_is_left_out_and_the_matching_one_drawn():
    h = _Host(levels=True)
    h.line(0, 1, [1, 2, 3], [3.0, 5.0, 7.0], "growth")  # FI's record at feed 1: drawn (the known positive)
    gone = h.line(0, 2, [1, 2, 3], [40.0, 50.0, 60.0], "growth")  # not FI's record at feed 2
    h.finish(h.labels, {})
    rec = h.record()["panels"][0]
    assert gone == [] and [s["level"]["text"] for s in rec["series"]] == ["1"]
    assert rec["series"][0]["what"] == "growth"
    assert rec["left_out"] == [{"id": "growth@feed:2", "kind": "curve", "declared": "growth", "label": "Growth (1/h)",
                                "name": "Growth (1/h) at Feed level = 2", "setting": "2", "setting_label": "Feed level",
                                "reason": "the drawn values are not FI's record of Growth (1/h) at Feed level = 2"}]
    assert _legend(h) == ["Feed level = 1"]


def test_a_band_of_a_left_out_series_is_left_out_too():
    h = _Host()
    h.line(0, "growth", [1, 2, 3], [9.0, 9.5, 9.9], "growth")
    assert h.band(0, "growth", [1, 2, 3], [8, 9, 9], [10, 10, 11], "growth", "other") is None
    assert h.record()["panels"][0]["bands"] == []


def test_a_series_drawn_as_other_uses_the_series_label_else_value():
    h = _Host()
    h.line(0, "arm_a", [1, 2, 3], [1.0, 4.0, 2.0], "other")
    h.line(0, "arm_b", [1, 2, 3], [2.0, 1.0, 3.0], "other")
    h.finish(h.labels, {})
    assert _legend(h) == ["Arm A", "value"]  # the plan's label of the series id, else "value" (never the bare id)
    assert h.axes[0].get_ylabel() == "value" and h.record()["panels"][0]["y"]["shown"] == "other"
    assert all(s["what"] == "other" for s in h.record()["panels"][0]["series"])


def test_a_series_named_by_a_result_and_drawn_as_other_uses_the_result_label():
    h = _Host()
    h.line(0, "growth", [1, 2, 3], [1.0, 4.0, 2.0], "other")
    h.finish(h.labels, {})
    assert _legend(h) == ["Growth (1/h)"] and h.axes[0].get_ylabel() == "Growth (1/h)"
    assert h.record()["panels"][0]["series"][0]["what"] == "other"


def test_an_x_fi_could_not_check_keeps_the_plan_label_and_the_flag():
    h = _Host()
    h.line(0, "arm_a", ("dose", [0.1, 0.2, 0.3]), [1.0, 2.0, 3.0], "other")  # no setting holds these doses
    h.finish(h.labels, {})
    x = h.record()["panels"][0]["x"]
    assert x["neutral"] is True and x["label"] == "Dose (mg)"
    assert h.axes[0].get_xlabel() == "Dose (mg)"


def test_two_results_fi_could_not_compare_both_name_the_axis():
    h = _Host()
    h.line(0, "yield_ratio", [1, 2, 3], [0.5, 0.6, 0.7], "yield_ratio")
    h.line(0, "uptake", [1, 1, 2], [1.0, 2.0, 3.0], "uptake")  # two points at one x: no one setting per point
    h.finish(h.labels, {})
    y = h.record()["panels"][0]["y"]
    assert y["what"] == "other" and y["shown"] == "yield_ratio / uptake"
    assert h.axes[0].get_ylabel() == "Yield ratio / Uptake (mg)"


def test_no_figure_word_says_computed_value_or_computed_x():
    for draw in (lambda h: h.line(0, "yield_ratio", [1, 2, 3], [0.5, 0.6, 0.7], "yield_ratio"),
                 lambda h: h.line(0, "arm_b", ("dose", [0.1, 0.2, 0.3]), [1.0, 2.0, 3.0], "other"),
                 lambda h: h.line(0, "growth", [1, 2, 3], [1.0, 4.0, 2.0], "other"),
                 lambda h: h.image(0, [[1.0, 2.0], [3.0, 4.0]], ("dose", [1, 2]), ("feed", [1, 2]), "uptake")):
        h = _Host()
        draw(h)
        h.finish(h.labels, {})
        words = _words(h)
        assert "computed value" not in words and "computed x" not in words, words
    assert not hasattr(frame_figure, "NEUTRAL_Y") and not hasattr(frame_figure, "NEUTRAL_X")
