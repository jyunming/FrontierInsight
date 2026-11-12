"""FI's own figure in place of a figure body (``default_figure``) when the plan puts a RESULT on the x axis, not a setting: each
trial's pair of the two results is drawn as points (paired by the record's trial numbers, never by position), a text value of a
derived quantity on x is a category, and a figure with a setting on x is drawn as before. Made-up study: plants grown at settings
of ``water`` (and ``soil``), each trial recording a plant's ``height`` and ``leaf_area``."""

from __future__ import annotations

import json
import sys
import time
import types
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import analysis_figures as af  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def frame5(tmp_path, monkeypatch):
    """The frame of the drawing calls beside the figure code, the way a quest has it."""
    from core import frame_table, skeleton

    code = tmp_path / "code"
    code.mkdir()
    (code / "fi_frame.py").write_text(skeleton.RUNTIME, encoding="utf-8")
    (code / skeleton.FIGURE_CALLS_FILE).write_text(skeleton.FRAME_FIGURE_SOURCE, encoding="utf-8")
    for name in ("fi_frame", "fi_figure_calls"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "fi_table", frame_table)
    monkeypatch.syspath_prepend(str(code))
    records = tmp_path / ".fi" / "figure_records"
    records.mkdir(parents=True)
    monkeypatch.setenv("FI_FIGURE_RECORDS", str(records))
    yield records
    for name in ("fi_frame", "fi_figure_calls"):
        sys.modules.pop(name, None)


def _table(declared=("height", "leaf_area"), soils=("clay",)):
    """FI's own Table over a made-up record: three settings of ``water`` for each soil, three trials each. ``leaf_area`` is twice
    ``height`` in the same trial; the trial numbers of ``leaf_area`` are listed in another order than ``height``'s, so a pairing
    by position would give other points than a pairing by trial."""
    from core import frame_table, skeleton

    sys.modules["fi_table"] = frame_table
    ns: dict[str, Any] = {"fi_table": frame_table}
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    cells = []
    for soil in soils:
        for w in (1, 2, 3):
            heights = {t: 10.0 * w + t for t in (0, 1, 2)}
            order = [2, 0, 1]
            cells.append({"cell": {"water": w, "soil": soil}, "key": "water=%d,soil=%s" % (w, soil), "planned": 3, "ok": 3,
                          "failed": 0, "metrics": {
                              "height": {"values": [heights[t] for t in (0, 1, 2)], "trials": [0, 1, 2]},
                              "leaf_area": {"values": [2.0 * heights[t] for t in order], "trials": order}}})
    return ns["Table"](cells, list(declared))


def _record(records, file):
    return json.loads((records / file.replace(".png", ".calls.json")).read_text(encoding="utf-8"))


def test_a_result_on_x_is_drawn_as_each_trial_s_paired_points(tmp_path, frame5):
    spec = {"id": "f", "file": "f.png", "shows": ["leaf_area"], "x": "height", "series": None, "panels": 1}
    out = af.default_figure(spec, _table(), str(tmp_path))
    assert out["status"] == "ok" and out["fallback"] is True and out.get("paired") is True
    assert (tmp_path / "f.png").exists()
    series = _record(frame5, "f.png")["panels"][0]["series"]
    assert [e["kind"] for e in series] == ["points"]
    pairs = sorted(zip(series[0]["x"], series[0]["y"]))
    assert len(pairs) == 9  # every trial of every setting: no setting is held
    assert all(abs(y - 2.0 * x) < 1e-12 for x, y in pairs)  # paired by the record's trial numbers, never by position
    assert _record(frame5, "f.png")["panels"][0]["x"]["id"] == "height"


def test_a_result_on_x_is_drawn_without_the_drawing_calls_too(tmp_path):
    spec = {"id": "f", "file": "f.png", "shows": ["leaf_area"], "x": "height"}
    out = af.default_figure(spec, _table(), str(tmp_path))
    assert out["status"] == "ok" and out.get("paired") is True and (tmp_path / "f.png").exists()


def test_the_series_setting_splits_the_points_into_one_cloud_for_each_value(tmp_path, frame5):
    spec = {"id": "f", "file": "f.png", "shows": ["leaf_area"], "x": "height", "series": "soil", "panels": 1}
    out = af.default_figure(spec, _table(soils=("clay", "sand")), str(tmp_path))
    assert out["status"] == "ok"
    series = _record(frame5, "f.png")["panels"][0]["series"]
    assert len(series) == 2 and all(e["kind"] == "points" and len(e["x"]) == 9 for e in series)


def test_a_derived_text_result_on_x_is_drawn_as_categories(tmp_path, frame5):
    table = _table()
    derived = {"best_soil": {"status": "ok", "by_setting": {k: ("loam" if table.setting(k)["water"] < 3 else "peat")
                                                             for k in table.keys()}}}
    spec = {"id": "f", "file": "f.png", "shows": ["height"], "x": "best_soil"}
    plan = types.SimpleNamespace(DERIVED={"best_soil": {"by": "derive_best_soil", "inputs": ["height"], "role": "result"}})
    out = af.default_figure(spec, table, str(tmp_path), plan, derived)
    assert out["status"] == "ok" and out.get("paired") is True
    series = _record(frame5, "f.png")["panels"][0]["series"]
    assert sorted(series[0]["x"]) == ["loam", "loam", "peat"]  # one point for each setting: the derived value and the mean there


def test_a_result_no_trial_gave_is_said_by_name_not_as_a_value_along_x(tmp_path):
    spec = {"id": "f", "file": "f.png", "shows": ["root_depth"], "x": "height"}
    out = af.default_figure(spec, _table(declared=("height", "leaf_area", "root_depth")), str(tmp_path))
    assert out["status"] == "dropped" and out["fallback"] is True
    assert "no trial gave a value of `root_depth`" in out["why"] and "along" not in out["why"]
    assert not (tmp_path / "f.png").exists()


def test_a_setting_on_x_is_still_drawn_as_lines_along_the_setting(tmp_path, frame5):
    spec = {"id": "f", "file": "f.png", "shows": ["leaf_area"], "x": "water", "series": None, "panels": 1}
    out = af.default_figure(spec, _table(), str(tmp_path))
    assert out["status"] == "ok" and "paired" not in out
    series = _record(frame5, "f.png")["panels"][0]["series"]
    assert [e["kind"] for e in series] == ["line"] and series[0]["x"] == [1, 2, 3]


def test_a_shown_result_that_gives_no_point_is_not_recorded_as_drawn(tmp_path, frame5):
    """``stem`` is recorded (without trial numbers) only at a setting where no trial gave ``height``: it has a value, but no
    point against ``height``, so the figure's record must not count it as drawn."""
    table = _table(declared=("height", "leaf_area", "stem"))
    extra = {"cell": {"water": 4, "soil": "clay"}, "key": "water=4,soil=clay", "planned": 1, "ok": 1, "failed": 0,
             "metrics": {"stem": {"values": [5.0]}}}
    table = type(table)(table._cells + [extra], list(table.declared))
    spec = {"id": "f", "file": "f.png", "shows": ["leaf_area", "stem"], "x": "height"}
    out = af.default_figure(spec, table, str(tmp_path))
    assert out["status"] == "ok"
    assert out["declared"]["drawn"] == ["leaf_area"] and out["declared"]["not_drawn"] == ["stem"]


class _Many:
    """A light table of many settings, each with its own value of the series setting ``plot`` and one trial."""

    def __init__(self, n):
        self.declared = ["height", "leaf_area"]
        self.n = n

    def keys(self):
        return ["plot=%d" % i for i in range(self.n)]

    def setting(self, key):
        return {"plot": int(key.split("=")[1])}

    def trials(self, name, key):
        return [0]

    def paired(self, a, b, key):
        i = float(key.split("=")[1])
        return [i], [2.0 * i]


def test_the_points_of_each_series_value_are_gathered_in_one_pass_over_the_settings():
    """A series with as many values as there are settings must not scan every setting once for each value: on a record of tens of
    thousands of settings that is billions of comparisons."""
    spec = {"id": "f", "file": "f.png", "shows": ["leaf_area"], "x": "height", "series": "plot"}
    began = time.monotonic()
    lines = af._paired_lines(spec, _Many(4000), {})
    assert len(lines) == 4000 and all(len(line[1]) == 1 for line in lines)
    assert time.monotonic() - began < 3.0  # one pass: a few hundredths of a second; once per value: well over ten seconds


def test_a_result_read_for_a_compared_value_with_no_point_against_x_is_not_recorded_as_drawn(tmp_path, frame5):
    """Two arms compared inside each run: ``height`` and ``leaf_area`` are recorded for arm ``a``, ``stem`` only for arm ``b``.
    Reading ``stem`` gives a value, but no point against ``height`` (another arm), so it is not drawn and not recorded as drawn."""
    from core import frame_table, skeleton

    sys.modules["fi_table"] = frame_table
    ns: dict[str, Any] = {"fi_table": frame_table}
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    cells = []
    for w in (1, 2):
        key = "water=%d" % w
        cells.append({"cell": {"water": w}, "key": key, "planned": 1, "ok": 1, "failed": 0, "metrics": {}})
        cells.append({"cell": {"water": w, "arm": "a"}, "level": {"arm": "a"}, "key": key + ",arm=a", "planned": 1, "ok": 1,
                      "failed": 0, "metrics": {"height": {"values": [1.0 * w]}, "leaf_area": {"values": [2.0 * w]}}})
        cells.append({"cell": {"water": w, "arm": "b"}, "level": {"arm": "b"}, "key": key + ",arm=b", "planned": 1, "ok": 1,
                      "failed": 0, "metrics": {"stem": {"values": [3.0 * w]}}})
    table = ns["Table"](cells, ["height", "leaf_area", "stem"])
    spec = {"id": "f", "file": "f.png", "shows": ["leaf_area", "stem"], "x": "height"}
    out = af.default_figure(spec, table, str(tmp_path))
    assert out["status"] == "ok" and out.get("paired") is True
    assert out["declared"]["drawn"] == ["leaf_area"] and out["declared"]["not_drawn"] == ["stem"]
    series = _record(frame5, "f.png")["panels"][0]["series"]
    assert len(series) == 1 and sorted(zip(series[0]["x"], series[0]["y"])) == [(1.0, 2.0), (2.0, 4.0)]


def test_a_derived_shown_result_drawn_without_the_drawing_calls_is_recorded_as_drawn(tmp_path):
    table = _table()
    derived = {"vigour": {"status": "ok", "by_setting": {k: float(table.setting(k)["water"]) for k in table.keys()}}}
    spec = {"id": "f", "file": "f.png", "shows": ["vigour"], "x": "height"}
    out = af.default_figure(spec, table, str(tmp_path), None, derived)
    assert out["status"] == "ok" and out.get("paired") is True
    assert out["declared"]["drawn"] == ["vigour"] and out["declared"]["not_drawn"] == []


# ---- the names of the two axes: FI names a result on x only when every point is one trial's pair --------------------------------

_NAMES = {"height": "Plant height", "leaf_area": "Leaf area"}


def _axes_of(records, file):
    panel = _record(records, file)["panels"][0]
    return panel["x"]["label"], panel["y"]["label"], panel["series"]


def _true_pairs(table, x="height", y="leaf_area", **hold):
    xs, ys = [], []
    for key in table.keys():
        if all(table.setting(key).get(a) == v for a, v in hold.items()):
            a, b = table.paired(x, y, key)
            xs += a
            ys += b
    return xs, ys


def test_the_default_figure_of_one_result_against_another_names_both_axes(tmp_path, frame5):
    spec = {"id": "f", "file": "f.png", "shows": ["leaf_area"], "x": "height", "result_labels": _NAMES}
    out = af.default_figure(spec, _table(), str(tmp_path))
    assert out["status"] == "ok" and out.get("paired") is True
    xlabel, ylabel, series = _axes_of(frame5, "f.png")
    assert (xlabel, ylabel) == ("Plant height", "Leaf area")
    assert series[0]["what"] == "leaf_area" and series[0]["x_paired"] is True and series[0]["label"] == "Leaf area"
    assert series[0]["x_checked"] is False and "declared_as" not in series[0]  # no claim along x; nothing to disclose


def test_each_series_value_keeps_both_names_when_its_points_are_its_own_trials(tmp_path, frame5):
    spec = {"id": "f", "file": "f.png", "shows": ["leaf_area"], "x": "height", "series": "soil", "result_labels": _NAMES}
    out = af.default_figure(spec, _table(soils=("clay", "sand")), str(tmp_path))
    assert out["status"] == "ok"
    xlabel, ylabel, series = _axes_of(frame5, "f.png")
    assert (xlabel, ylabel) == ("Plant height", "Leaf area") and all(e["what"] == "leaf_area" for e in series)


def test_pairs_of_results_recorded_without_trial_numbers_are_named_from_the_setting_values(tmp_path, frame5):
    """The record says no trial numbers: FI's pairs are the two means at each setting, and a figure of those is named."""
    from core import frame_table, skeleton

    sys.modules["fi_table"] = frame_table
    ns: dict[str, Any] = {"fi_table": frame_table}
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    cells = [{"cell": {"water": w}, "key": "water=%d" % w, "planned": 2, "ok": 2, "failed": 0,
              "metrics": {"height": {"values": [w, w + 1.0]}, "leaf_area": {"values": [3.0 * w, 3.0 * w + 1]}}} for w in (1, 2, 3)]
    spec = {"id": "f", "file": "f.png", "shows": ["leaf_area"], "x": "height", "result_labels": _NAMES}
    out = af.default_figure(spec, ns["Table"](cells, ["height", "leaf_area"]), str(tmp_path))
    assert out["status"] == "ok"
    xlabel, ylabel, series = _axes_of(frame5, "f.png")
    assert (xlabel, ylabel) == ("Plant height", "Leaf area") and len(series[0]["x"]) == 3


def _body_figure(tmp_path, table, body, series=None, ok=True):
    spec = {"id": "g", "file": "g.png", "shows": ["leaf_area"], "x": "height", "series": series, "result_labels": _NAMES}
    plan = af._plan_for(spec, None, table)
    out = af._draw_one(spec, body, table, str(tmp_path), plan)
    assert not ok or out["status"] == "ok", out
    return out


def test_a_figure_body_that_draws_the_trials_own_pairs_gets_both_names(tmp_path, frame5):
    table = _table()

    def figure_g(fig, tbl):
        xs, ys = _true_pairs(tbl)
        fig.points(0, "leaf_area", ("height", xs), ys, "leaf_area")

    _body_figure(tmp_path, table, figure_g)
    xlabel, ylabel, series = _axes_of(frame5, "g.png")
    assert (xlabel, ylabel) == ("Plant height", "Leaf area") and series[0]["what"] == "leaf_area"


def test_a_figure_whose_points_are_not_one_trial_s_pairs_keeps_the_computed_names(tmp_path, frame5):
    """Known positive: the same values, but the y of each point is another trial's (shuffled pairing)."""
    from core import frame_figure

    table = _table()

    def figure_g(fig, tbl):
        xs, ys = _true_pairs(tbl)
        fig.points(0, "leaf_area", ("height", xs), ys[1:] + ys[:1], "leaf_area")

    _body_figure(tmp_path, table, figure_g)
    xlabel, ylabel, series = _axes_of(frame5, "g.png")
    assert (xlabel, ylabel) == (frame_figure.NEUTRAL_X, frame_figure.NEUTRAL_Y)
    assert series[0]["what"] == "other" and "x_paired" not in series[0]
    assert series[0]["declared_as"]["checked"] == "differs"


def test_a_trial_s_pair_drawn_twice_is_not_two_trials(tmp_path, frame5):
    from core import frame_figure

    table = _table()

    def figure_g(fig, tbl):
        xs, ys = _true_pairs(tbl)
        fig.points(0, "leaf_area", ("height", xs + xs[:1]), ys + ys[:1], "leaf_area")

    _body_figure(tmp_path, table, figure_g)
    assert _axes_of(frame5, "g.png")[:2] == (frame_figure.NEUTRAL_X, frame_figure.NEUTRAL_Y)


def test_the_points_of_one_series_value_must_be_that_value_s_trials(tmp_path, frame5):
    """The legend says "soil = sand", but the points are the clay trials': not the curve it is named as."""
    from core import frame_figure

    table = _table(soils=("clay", "sand"))
    for c in table._cells:  # the sand trials give other values than the clay ones
        if c["cell"]["soil"] == "sand":
            c["metrics"]["height"]["values"] = [v + 100.0 for v in c["metrics"]["height"]["values"]]

    def figure_g(fig, tbl):
        xs, ys = _true_pairs(tbl, soil="clay")
        fig.points(0, "sand", ("height", xs), ys, "leaf_area")

    _body_figure(tmp_path, table, figure_g, series="soil")
    assert _axes_of(frame5, "g.png")[:2] == (frame_figure.NEUTRAL_X, frame_figure.NEUTRAL_Y)


def test_tiny_values_a_factor_apart_are_not_one_trial_s_pair(tmp_path, frame5):
    """Values near zero (1e-15 and 3e-15) are two values: a shuffled pairing of them is no trial's pair, whatever their size."""
    from core import frame_figure, frame_table, skeleton

    sys.modules["fi_table"] = frame_table
    ns: dict[str, Any] = {"fi_table": frame_table}
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    cells = [{"cell": {"water": w}, "key": "water=%d" % w, "planned": 3, "ok": 3, "failed": 0,
              "metrics": {"height": {"values": [10.0 * w + t for t in (0, 1, 2)], "trials": [0, 1, 2]},
                          "leaf_area": {"values": [1e-15 * (1 + 2 * t) for t in (0, 1, 2)], "trials": [0, 1, 2]}}} for w in (1, 2)]
    table = ns["Table"](cells, ["height", "leaf_area"])

    def figure_g(fig, tbl):
        xs, ys = _true_pairs(tbl)
        fig.points(0, "leaf_area", ("height", xs), ys[1:] + ys[:1], "leaf_area")

    _body_figure(tmp_path, table, figure_g)
    assert _axes_of(frame5, "g.png")[:2] == (frame_figure.NEUTRAL_X, frame_figure.NEUTRAL_Y)


def test_the_pair_check_does_not_count_as_a_read_of_the_body(tmp_path, frame5):
    """A body that read nothing from the table (it typed the pairs in) is still seen to have read nothing after FI checked its
    points: the figure is the body's fault, as for any body that draws nothing from the trial record."""
    table = _table()
    xs, ys = _true_pairs(table)
    table.reset_reads()
    table.served.clear()
    seen = {}

    def figure_g(fig, tbl):
        fig.points(0, "leaf_area", ("height", list(xs)), list(ys), "leaf_area")
        seen.update(gave=set(tbl.gave_values), served=list(tbl.served), errors=dict(tbl.last_error))

    out = _body_figure(tmp_path, table, figure_g, ok=False)
    assert seen == {"gave": set(), "served": [], "errors": {}}
    assert out["status"] == "failed" and "drew nothing from the trial record" in out["said"]


def test_a_band_on_a_named_pair_series_keeps_the_names(tmp_path, frame5):
    table = _table()

    def figure_g(fig, tbl):
        xs, ys = _true_pairs(tbl)
        fig.points(0, "leaf_area", ("height", xs), ys, "leaf_area")
        fig.band(0, "leaf_area", ("height", xs), [y - 1 for y in ys], [y + 1 for y in ys], "leaf_area", "other")

    _body_figure(tmp_path, table, figure_g)
    assert _axes_of(frame5, "g.png")[:2] == ("Plant height", "Leaf area")


class _Pairs:
    """A light table of one setting whose trials gave the pairs ``pairs`` (x, y), trial numbers 0, 1, ..."""

    def __init__(self, pairs):
        self.declared = ["height", "leaf_area"]
        self.pairs = pairs

    def keys(self):
        return ["water=1"]

    def setting(self, key):
        return {"water": 1}

    def trials(self, name, key):
        return list(range(len(self.pairs)))

    def paired(self, a, b, key):
        return [p[0] for p in self.pairs], [p[1] for p in self.pairs]


def _calls(table):
    from core import frame_figure

    host = type("Host", (frame_figure.FigureCalls,), {})()
    host.table, host.settings, host.x = table, ["water"], "height"
    return host


def test_a_recorded_pair_is_matched_exactly_and_once():
    host = _calls(_Pairs([(1.0, 1.0), (1.0, 1.0000000015)]))
    assert host._pair_check("leaf_area", "height", [1.0, 1.0], [1.0000000015, 1.0]) == "matches"  # in any order
    assert host._pair_check("leaf_area", "height", [1.0, 1.0], [1.0, 1.0]) == "differs"  # one trial drawn twice
    assert host._pair_check("leaf_area", "height", [1.0, 1.0], [1.00000000075, 1.0]) == "differs"  # a value no trial gave


def test_many_trials_at_one_x_are_checked_in_one_pass():
    n = 50000
    host = _calls(_Pairs([(1.0, float(i)) for i in range(n)]))
    began = time.monotonic()
    assert host._pair_check("leaf_area", "height", [1.0] * n, [float(i) for i in reversed(range(n))]) == "matches"
    assert time.monotonic() - began < 3.0
    near = [float(i) * (1 + 1e-10) for i in reversed(range(1, n))]  # no recorded value: no name, and no long search
    began = time.monotonic()
    assert host._pair_check("leaf_area", "height", [1.0] * (n - 1), near) == "differs"
    assert time.monotonic() - began < 3.0
