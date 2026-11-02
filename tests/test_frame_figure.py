"""The drawing calls of a figure and the call record (core/frame_figure.py). Made-up study: a colony whose ``growth`` is read at
settings of ``size``. The host is a minimal stand-in on real matplotlib axes (Agg)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

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
    declared = ["growth", "load"]

    def keys(self):
        return ["size=1"]

    def mean(self, name, key):
        return 2.5

    def n(self, name, key):
        return 4

    def interval(self, name, key):
        return (1.25, 3.75)


class _IdentityTable(_Table):
    """Every result's mean at ``size=v`` is ``v``: a test that only reads labels draws y = x, which is the table's own value."""

    def keys(self):
        return ["size=%d" % v for v in (1, 2, 3, 10, 100)]

    def mean(self, name, key):
        return float(key.partition("=")[2])


class _Host(frame_figure.FigureCalls):
    def __init__(self, panels=2, figsize=None, identity=False):
        self.fig, axes = plt.subplots(1, panels, squeeze=False, figsize=figsize)
        self.axes = list(axes[0])
        self.table = _IdentityTable() if identity else _Table()
        self.x = "size"
        self.settings = ["size", "step"]
        self.series_ids = ["fast", "slow"]
        self.references = {"limit": {"value": 3.0, "label": "upper limit"}}
        self.labels = {"growth": {"label": "Growth", "scale": "linear"}, "fast": "Fast colony", "size": "Size"}


def _data(host):
    return [a for ax in host.axes for a in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.images)]


def test_line_records_series_x_y_what():
    h = _Host(identity=True)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    p = h.record()["panels"][0]
    assert p["panel"] == 1 and p["raw"] is False and p["raw_reason"] is None
    assert p["x"] == {"id": "size", "kind": "setting", "label": "Size", "scale": "linear"}
    assert p["y"] == {"what": "growth", "label": "Growth", "scale": "linear"}
    s = p["series"][0]
    assert (s["id"], s["label"], s["kind"], s["step"]) == ("fast", "Fast colony", "line", False)
    assert s["x"] == [1.0, 2.0, 3.0] and s["y"] == [1.0, 2.0, 3.0] and s["yerr"] is None and s["merged_with"] == []
    assert h.record()["panels"][1]["series"] == []


def test_series_must_be_a_plan_id():
    h = _Host()
    with pytest.raises(ValueError) as e:
        h.line(0, "Weighted estimator", [1, 2], [1, 2], "growth")
    assert "fast" in str(e.value) and "slow" in str(e.value) and "growth" in str(e.value)
    with pytest.raises(ValueError) as e:
        h.line(0, "fast", [1, 2], [1, 2], "speed")
    assert "growth" in str(e.value) and "other" in str(e.value)
    h.line(0, "fast", [1, 2], [1, 2], "other")
    assert h.record()["panels"][0]["y"]["label"] == "computed value"


def test_x_may_be_a_result_id():
    h = _Host()
    h.points(0, "fast", ("load", [0.5, 1.5]), [1.0, 2.0], "growth")
    x = h.record()["panels"][0]["x"]
    assert x["id"] == "load" and x["kind"] == "result"
    with pytest.raises(ValueError) as e:
        h.points(0, "slow", ("size", [1, 2]), [1.0, 2.0], "growth")
    assert "one x axis" in str(e.value)
    with pytest.raises(ValueError) as e:
        h.points(1, "fast", ("nonsense", [1]), [1.0], "growth")
    assert "size" in str(e.value) and "load" in str(e.value)


def test_band_meaning_closed_list():
    h = _Host()
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    for m in ("ci95", "sd", "range", "other"):
        h.band(0, "fast", [1, 2], [0, 1], [2, 3], "growth", m)
    # typed here, and the table gives none of these widths or edges at these x: no meaning is kept
    assert [b["meaning"] for b in h.record()["panels"][0]["bands"]] == ["other", "other", "other", "other"]
    with pytest.raises(ValueError) as e:
        h.band(0, "fast", [1, 2], [0, 1], [2, 3], "growth", "stderr")
    assert "ci95" in str(e.value) and "range" in str(e.value)


def test_ref_unknown_id_raises_naming_known():
    h = _Host()
    h.ref(0, "limit")
    h.ref(1, "limit", axis="x")
    r = h.record()["panels"][0]["refs"][0]
    assert r == {"id": "limit", "axis": "y", "value": 3.0, "label": "upper limit", "source": "plan"}
    assert h.axes[0].lines[-1].get_ydata()[0] == 3.0
    with pytest.raises(ValueError) as e:
        h.ref(0, "bound")
    assert "limit" in str(e.value)


def test_note_with_typed_digits_is_refused():
    h = _Host()
    with pytest.raises(ValueError) as e:
        h.note(0, 1, 1, "mean growth 2.5 at {growth}")
    assert "digits" in str(e.value) and "table" in str(e.value)
    h.note(0, 1.0, 2.0, "mean growth: {growth}")
    n = h.record()["panels"][0]["notes"][0]
    assert n == {"id": "growth", "text": "mean growth: 2.5", "x": 1.0, "y": 2.0}
    h.note(0, 1, 1, "runs: {growth}", form="n")
    assert h.record()["panels"][0]["notes"][1]["text"] == "runs: 4"
    h.note(0, 1, 1, "upper {growth}", form="high")
    assert h.record()["panels"][0]["notes"][2]["text"] == "upper 3.75"
    with pytest.raises(ValueError):
        h.note(0, 1, 1, "no placeholder")
    with pytest.raises(ValueError):
        h.note(0, 1, 1, "bad {speed}")


def test_curve_recorded_as_kind_curve_with_source():
    h = _Host()
    with pytest.raises(ValueError):
        h.curve(0, "theory", [1, 2], [1, 2])  # nothing drawn yet: no x to draw on
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.curve(0, "theory", [1, 3], [1.0, 3.0])
    h.curve(0, "fit", [1, 3], [1.5, 2.5], source="fitted to the drawn points")
    s = h.record()["panels"][0]["series"]
    assert [x["kind"] for x in s] == ["line", "curve", "curve"]
    assert s[1]["source"] == "computed in the figure" and s[2]["source"] == "fitted to the drawn points"
    assert [x["label"] for x in s[1:]] == ["curve 1", "curve 2"]  # FI's names, never the body's words
    assert all(x["what"] == "other" for x in s[1:])


def test_image_is_recorded_with_its_grid():
    h = _Host()
    h.image(0, [[1, 2, 3], [4, 5, 6]], x=[0, 1, 2], y=[10, 20], what="growth")
    s = h.record()["panels"][0]["series"][0]
    assert s["kind"] == "image" and s["x"] == [0.0, 1.0, 2.0] and s["y"] == [10.0, 20.0] and s["z"][1] == [4.0, 5.0, 6.0]
    assert h.axes[0].images[0].get_gid().startswith("fi:")


def test_raw_marks_panel_raw():
    h = _Host()
    ax = h.raw(1)
    assert ax is h.axes[1]
    p = h.record()["panels"]
    assert p[1]["raw"] is True and p[1]["raw_reason"] == "fig.raw" and p[0]["raw"] is False
    with pytest.raises(IndexError):
        h.raw(2)


def test_artist_gids_are_tagged():
    h = _Host()
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth", yerr=[0.1, 0.1, 0.2])
    h.bars(1, "slow", [1, 2], [1.0, 2.0], "growth", yerr=[0.1, 0.1])
    h.band(0, "fast", [1, 2, 3], [0, 1, 2], [2, 3, 5], "growth", "sd")
    h.ref(0, "limit")
    h.curve(0, "theory", [1, 3], [1, 4])
    h.points(1, "fast", [1, 2], [1.0, 2.0], "growth")
    data = _data(h)
    assert data
    assert all((a.get_gid() or "").startswith("fi:") for a in data), [(type(a).__name__, a.get_gid()) for a in data]
    gids = {a.get_gid() for a in data}
    assert len(gids) >= 6  # one tag for each call


def test_equal_series_are_kept_as_drawn():
    h = _Host()
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth")
    h.line(0, "slow", [1, 2, 3], [1.0, 2.0, 4.0], "growth", yerr=[0.1, 0.1, 0.1])
    s = h.record()["panels"][0]["series"]
    assert [x["id"] for x in s] == ["fast", "slow"] and s[0]["merged_with"] == [] and s[1]["yerr"] == [0.1, 0.1, 0.1]


def test_missing_values_are_null_gaps_never_text():
    h = _Host()
    h.line(0, "fast", [1, 2, 3], [1, None, 3], "growth")
    h.bars(1, "fast", [1, 2], [1.0, None], "growth", yerr=[None, 0.2])
    h.image(1, [[1, None], [3, 4]], what="growth")
    r = h.record()["panels"]
    assert r[0]["series"][0]["y"] == [1.0, None, 3.0]
    assert r[1]["series"][0]["y"] == [1.0, None] and r[1]["series"][0]["yerr"] == [None, 0.2]
    assert r[1]["series"][1]["z"][0] == [1.0, None]
    ydata = h.axes[0].lines[0].get_ydata()
    assert len(ydata) == 3 and math.isnan(ydata[1]) and not any(isinstance(v, str) for v in ydata)
    json.dumps(h.record(), allow_nan=False)


def test_non_numeric_y_is_refused_before_anything_is_drawn():
    h = _Host()
    for bad in (["a", "b"], [True, 1.0]):
        with pytest.raises(ValueError) as e:
            h.line(0, "fast", [1, 2], bad, "growth")
        assert "numbers" in str(e.value)
    with pytest.raises(ValueError):
        h.band(0, "fast", [1, 2], [0, "x"], [1, 2], "growth", "sd")
    assert h.record()["panels"][0]["series"] == [] and not h.axes[0].lines


def test_failed_draw_leaves_no_record(monkeypatch):
    h = _Host()

    def boom(*a, **k):
        raise RuntimeError("draw failed")

    monkeypatch.setattr(h.axes[0], "plot", boom)
    with pytest.raises(RuntimeError):
        h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    p = h.record()["panels"][0]
    assert p["series"] == [] and p["x"] is None and p["y"] is None
    monkeypatch.undo()
    monkeypatch.setattr(h.axes[0], "errorbar", boom)  # fails after the line was drawn: the line goes too
    with pytest.raises(RuntimeError):
        h.line(0, "fast", [1, 2], [1.0, 2.0], "growth", yerr=[0.1, 0.1])
    assert h.record()["panels"][0]["series"] == [] and not h.axes[0].lines


def test_band_records_x_and_needs_its_series():
    h = _Host()
    with pytest.raises(ValueError) as e:
        h.band(0, "fast", [1, 2], [0, 1], [2, 3], "growth", "sd")
    assert "not been drawn" in str(e.value)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth")
    with pytest.raises(ValueError) as e:
        h.band(0, "fast", [1, 2, 4], [0, 1, 2], [2, 3, 5], "growth", "sd")
    assert "does not match" in str(e.value)
    with pytest.raises(ValueError):
        h.band(0, "slow", [1, 2, 3], [0, 1, 2], [2, 3, 5], "growth", "sd")
    assert h.record()["panels"][0]["bands"] == []
    h.band(0, "fast", [1, 2, 3], [0, 1, 2], [2, 3, 5], "growth", "sd")
    b = h.record()["panels"][0]["bands"][0]
    assert b["x"] == [1.0, 2.0, 3.0] and b["lo"] == [0.0, 1.0, 2.0] and b["meaning"] == "other"  # typed sd, not the table's


def test_error_bars_record_their_meaning():
    h = _Host()
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth", yerr=[0.1, 0.2], yerr_meaning="range")
    h.points(0, "slow", [1, 2], [1.0, 2.5], "growth", yerr=[0.1, 0.2])
    h.bars(1, "fast", [1, 2], [1.0, 2.0], "growth", yerr=[-0.1, 0.2], yerr_meaning="sd")
    h.line(1, "slow", [1, 2], [1.0, 2.0], "growth")
    r = h.record()["panels"]
    assert [s["yerr_meaning"] for s in r[0]["series"]] == ["other", "other"]  # range is the table's min and max: typed ones are not
    assert r[1]["series"][0]["yerr_meaning"] == "other" and r[1]["series"][0]["yerr"] == [0.1, 0.2]  # sd typed, table says 0
    assert r[1]["series"][1]["yerr_meaning"] is None
    with pytest.raises(ValueError) as e:
        h.line(0, "fast", [3], [1.0], "growth", yerr=[0.1], yerr_meaning="stderr")
    assert "ci95" in str(e.value) and "range" in str(e.value)


def test_step_and_non_finite_values():
    h = _Host()
    h.line(0, "fast", [1, 2, 3], [1.0, math.nan, 4.0], "growth", step=True)
    s = h.record()["panels"][0]["series"][0]
    assert s["step"] is True and s["y"] == [1.0, None, 4.0]
    with pytest.raises(ValueError):
        h.line(0, "slow", [1, 2], [1.0], "growth")


def test_record_roundtrips_json(tmp_path):
    h = _Host()
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth", yerr=[0.1, 0.2, 0.3])
    h.band(0, "fast", [1, 2, 3], [0, 1, 2], [2, 3, 5], "growth", "ci95")
    h.ref(0, "limit")
    h.note(0, 1, 1, "mean: {growth}")
    h.raw(1)
    out = h.write(tmp_path / "sub" / "fig1.png")
    assert out.name == "fig1.calls.json"
    assert json.loads(out.read_text(encoding="utf-8")) == h.record()
    assert h.record()["version"] == 2 and h.record()["fallback"] is False
    assert h.write(tmp_path / "other.calls.json").name == "other.calls.json"


def test_carried_text_imports_nothing_of_fi(tmp_path):
    src = Path(frame_figure.__file__).read_text(encoding="utf-8")
    assert "from core" not in src and "import core" not in src
    (tmp_path / "fi_figure.py").write_text(src, encoding="utf-8")
    code = textwrap.dedent("""
        import fi_figure
        class H(fi_figure.FigureCalls):
            axes = []
        print(H().record())
    """)
    run = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r)\n%s" % (str(tmp_path), code)],
                         capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert run.returncode == 0, run.stderr


# ---- labels, legend, limits (finish) ---------------------------------------------------------------------------------------

SPEC = {"growth": {"label": "Growth", "unit": "1/h"}, "size": {"label": "Size", "unit": "mm"}, "fast": "Fast colony", "slow": "Slow colony"}
INFO = {"growth": {"description": "Mean growth rate of the colony, measured per hour (see plan).", "unit": "1/h", "scale": "linear"},
        "load": {"description": "Load carried by the colony at the end"}}


def test_label_comes_from_plan_not_body(caplog):
    h = _Host(panels=1, identity=True)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.axes[0].set_ylabel("energy drift")
    h.axes[0].set_title("A title")
    with caplog.at_level("WARNING", logger="frame_figure"):
        h.finish(SPEC, INFO)
    assert h.axes[0].get_ylabel() == "Growth (1/h)" == h.record()["panels"][0]["y"]["label"]
    assert h.axes[0].get_xlabel() == "Size (mm)" and h.axes[0].get_title() == ""


def test_label_fallback_order():
    h = _Host(panels=3, identity=True)
    for i, what in enumerate(["growth", "load", "load"]):
        h.line(i, "fast", [1, 2], [1.0, 2.0], what)
    h.finish({"growth": {"label": "Growth", "unit": "1/h"}}, {"load": {"description": "Load carried by the colony at the end; read at 3 h"}})
    assert h.axes[0].get_ylabel() == "Growth (1/h)"
    assert h.axes[1].get_ylabel() == "Load carried by the colony at the end"
    h2 = _Host(panels=1, identity=True)
    h2.line(0, "fast", [1, 2], [1.0, 2.0], "load")
    h2.finish({}, {"load": {"unit": "kg"}})
    assert h2.axes[0].get_ylabel() == "value (kg)" and h2.axes[0].get_xlabel() == "value"
    assert "load" not in h2.axes[0].get_ylabel()


def test_two_results_and_other_quantity_labels():
    h = _Host(panels=2, identity=True)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.line(0, "slow", [1, 2], [1.0, 2.0], "load")
    h.line(1, "fast", [1, 2], [1.0, 2.0], "other")
    h.finish(SPEC, INFO)
    assert h.axes[0].get_ylabel() == "Growth (1/h) / Load carried by the colony at the end"
    assert h.axes[1].get_ylabel() == "computed value"


def test_legend_has_no_empty_entry():
    h = _Host(panels=1, identity=True)
    ax = h.axes[0]
    ax.plot([1, 2], [1, 2], label="_stale")  # a body's leftover artist with a private label
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth", yerr=[0.1, 0.1, 0.1])
    h.band(0, "fast", [1, 2, 3], [0, 1, 2], [2, 3, 5], "growth", "ci95")
    h.ref(0, "limit")
    ax.legend(["body legend"])
    h.finish(SPEC, INFO)
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts == ["Fast colony", "upper limit"]
    assert all(texts)


def test_limits_include_errorbars_bands_refs():
    h = _Host(panels=1)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth", yerr=[0.5, 0.5, 2.0])
    h.band(0, "fast", [1, 2, 3], [0.0, 1.0, 2.0], [2.0, 3.0, 4.0], "growth", "ci95")
    h.ref(0, "limit")
    h.finish(SPEC, INFO)
    lo, hi = h.axes[0].get_ylim()
    assert lo < 0.0 and hi > 5.0 and lo > -1.0 and hi < 6.0  # error bar top 5.0, band floor 0.0, ref 3.0, 5% pad
    x0, x1 = h.axes[0].get_xlim()
    assert x0 < 1 and x1 > 3


def test_bars_reach_their_baseline():
    h = _Host(panels=1)
    h.bars(0, "fast", [1, 2, 3], [5.0, 6.0, 7.0], "growth")
    h.finish(SPEC, INFO)
    assert h.axes[0].get_ylim()[0] <= 0.0 and h.axes[0].get_xlim()[0] <= 0.5


def test_identical_series_are_merged_and_recorded():
    h = _Host(panels=1, identity=True)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth", yerr=[0.1, 0.2, 0.3])
    h.line(0, "slow", [1, 2, 3], [1.0 + 1e-14, 2.0, 3.0], "growth", yerr=[0.1, 0.2, 0.3])
    h.finish(SPEC, INFO)
    s = h.record()["panels"][0]["series"]
    assert s[0]["merged_with"] == ["slow"] and s[1]["merged_with"] == [] and len(s) == 2
    assert [t.get_text() for t in h.axes[0].get_legend().get_texts()] == ["Fast colony = Slow colony"]


def test_near_equal_series_are_not_merged():
    h = _Host(panels=1)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth")
    h.line(0, "slow", [1, 2, 3], [1.0 + 1e-6, 2.0, 4.0], "growth")
    h.finish(SPEC, INFO)
    assert [e["merged_with"] for e in h.record()["panels"][0]["series"]] == [[], []]
    assert len(h.axes[0].get_legend().get_texts()) == 2
    # same y but different error bars, or a different result: not identical either
    g = _Host(panels=1)
    g.line(0, "fast", [1, 2], [1.0, 2.0], "growth", yerr=[0.1, 0.1])
    g.line(0, "slow", [1, 2], [1.0, 2.0], "growth", yerr=[0.2, 0.1])
    g.finish(SPEC, INFO)
    assert [e["merged_with"] for e in g.record()["panels"][0]["series"]] == [[], []]


def test_body_set_limits_are_overwritten_and_logged_once(caplog):
    h = _Host(panels=2)
    for i in (0, 1):
        h.line(i, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth")
        h.axes[i].set_ylim(-100, 100)
        h.axes[i].set_xlabel("body x")
        h.axes[i].set_title("body title")
    with caplog.at_level("WARNING", logger="frame_figure"):
        h.finish(SPEC, INFO)
        h.finish(SPEC, INFO)
    msgs = [r for r in caplog.records if r.name == "frame_figure"]
    assert len(msgs) == 1 and "limits" in msgs[0].getMessage() and "label" in msgs[0].getMessage()
    assert all(ax.get_ylim()[1] < 10 for ax in h.axes)
    assert h.record()["panels"][0]["raw"] is False


def test_log_axis_with_zero_in_band_keeps_finite_limits():
    h = _Host(panels=1)
    h.line(0, "fast", [1, 2, 3], [1.0, 10.0, 100.0], "growth")
    h.band(0, "fast", [1, 2, 3], [0.0, 5.0, 50.0], [2.0, 20.0, 200.0], "growth", "range")
    h.finish(SPEC, {"growth": {"scale": "log"}})
    ax = h.axes[0]
    lo, hi = ax.get_ylim()
    assert ax.get_yscale() == "log" and math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= 1.0 and hi >= 200.0
    assert h.record()["panels"][0]["y"]["scale"] == "log"


def test_legend_stays_off_the_data_and_outside_when_crowded():
    h = _Host(panels=2)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")  # rising line: upper left is free
    h.finish(SPEC, INFO)
    assert h.axes[0].get_legend()._loc == 2  # "upper left"


def _crowded(host, panel, n=2):
    """Lines through every corner of the panel, so no corner is free."""
    for k in range(n):
        host.line(panel, "fast" if k % 2 == 0 else "slow", [1, 2, 3, 4], [1.0 + k, 4.0 + k, 1.0 + k, 4.0 + k], "growth")
    host.line(panel, "slow", [1, 2, 3, 4], [4.0, 1.0, 4.0, 1.0], "growth")


@pytest.mark.parametrize("columns,series", [(1, 2), (2, 2), (3, 2), (3, 8), (2, 8)])
def test_outside_legend_clears_other_panels_and_the_canvas(columns, series):
    h = _Host(panels=columns, figsize=(6.4 if columns == 1 else 9.6, 4.8))
    for i in range(columns):
        _crowded(h, i, series)
        h.ref(i, "limit")
    h.finish(SPEC, INFO)
    r = h.fig.canvas.get_renderer()
    canvas = h.fig.bbox
    for i, ax in enumerate(h.axes):
        ext = ax.get_legend().get_window_extent(r)
        assert ext.x0 >= canvas.x0 and ext.x1 <= canvas.x1 and ext.y0 >= canvas.y0 and ext.y1 <= canvas.y1, (i, ext)
        for j, other in enumerate(h.axes):
            if j != i:
                b = other.get_tightbbox(r)
                assert not (ext.x0 < b.x1 and b.x0 < ext.x1 and ext.y0 < b.y1 and b.y0 < ext.y1), (i, j, ext, b)


def test_merged_legend_entry_never_shows_a_raw_id():
    h = _Host(panels=1, identity=True)
    h.series_ids = ["fast", "slow", "mid"]
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.line(0, "mid", [1, 2], [1.0, 2.0], "growth")
    h.finish({"growth": {"label": "Growth"}}, {"mid": {"description": "Middle colony; third arm"}})
    assert [t.get_text() for t in h.axes[0].get_legend().get_texts()] == ["Fast colony = Middle colony"]
    g = _Host(panels=1, identity=True)
    g.labels = {}
    g.line(0, "slow", [1, 2], [1.0, 2.0], "growth")
    g.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    g.finish({}, {})
    assert [t.get_text() for t in g.axes[0].get_legend().get_texts()] == ["value = value"]


def test_log_axis_ignores_nonpositive_error_bar_end():
    h = _Host(panels=1, identity=True)
    h.line(0, "fast", [1, 10, 100], [1.0, 10.0, 100.0], "growth", yerr=[2.0, 2.0, 2.0])  # 1.0 - 2.0 is below zero
    h.finish(SPEC, {"growth": {"scale": "log"}})
    lo, hi = h.axes[0].get_ylim()
    assert h.axes[0].get_yscale() == "log" and 0 < lo < 1.0 and 100.0 < hi < 1000.0


class _GridTable(_Table):
    """Two settings; the mean of ``growth`` at size=a, step=b is 10a + b."""

    def keys(self):
        return ["size=%d,step=%d" % (a, b) for a in (1, 2) for b in (3, 4)]

    def mean(self, name, key):
        a, b = (int(p.partition("=")[2]) for p in key.split(","))
        return float(10 * a + b)


def _grid_host(z):
    h = _Host(panels=1)
    h.table = _GridTable()
    h.image(0, z, x=("size", [1, 2]), y=("step", [3, 4]), what="growth")
    return h


def test_image_panel_labels():
    h = _grid_host([[13.0, 23.0], [14.0, 24.0]])
    h.finish(SPEC, INFO)
    p = h.record()["panels"][0]
    assert h.axes[0].get_ylabel() == "row" and h.axes[0].get_xlabel() == "column"
    assert p["y"]["label"] == "Growth (1/h)" and h.axes[0].get_legend() is None


def test_image_keeps_its_quantity_only_when_every_cell_is_the_tables():
    honest = _grid_host([[13.0, 23.0], [14.0, 24.0]]).record()["panels"][0]
    assert honest["series"][0]["what"] == "growth" and honest["y"]["what"] == "growth"
    for z in ([[13.0, 23.0], [14.0, 24.5]], [[14.0, 24.0], [13.0, 23.0]]):  # one cell off; rows swapped
        bad = _grid_host(z).record()["panels"][0]
        assert bad["series"][0]["what"] == "other" and bad["y"]["what"] == "other"
    unknown = _Host(panels=1)  # no ids: the rows cannot be tied to a setting of this table
    unknown.image(0, [[1.0, 2.0], [3.0, 4.0]], what="growth")
    assert unknown.record()["panels"][0]["series"][0]["what"] == "other"


class _OneKey(_Table):
    """One setting: mean 10, 95% interval 9 to 11, sd 1."""

    def keys(self):
        return ["size=1"]

    def mean(self, name, key):
        return 10.0

    def interval(self, name, key):
        return (9.0, 11.0)

    def sd(self, name, key):
        return 1.0


def test_band_and_bars_compare_their_edges_with_the_table_not_only_their_widths():
    h = _Host(panels=1)
    h.table = _OneKey()
    h.line(0, "fast", [1], [12.0], "growth")  # a y that is not the table's mean
    h.band(0, "fast", [1], [11.0], [13.0], "growth", "ci95")  # the table's width, around the wrong centre
    h.points(0, "slow", [1], [12.0], "growth", yerr=[1.0], yerr_meaning="sd")
    r = h.record()["panels"][0]
    assert r["bands"][0]["meaning"] == "other" and r["series"][1]["yerr_meaning"] == "other"
    g = _Host(panels=1)
    g.table = _OneKey()
    g.line(0, "fast", [1], [10.0], "growth")
    g.band(0, "fast", [1], [9.0], [11.0], "growth", "ci95")
    g.points(0, "slow", [1], [10.0], "growth", yerr=[1.0], yerr_meaning="sd")
    r = g.record()["panels"][0]
    assert r["bands"][0]["meaning"] == "ci95" and r["series"][1]["yerr_meaning"] == "sd"


class _ThreeSettings(_Table):
    """size x step x arm; the mean is 100 * arm + 10 * size + step, so no two cells share a value."""

    def keys(self):
        return ["size=%d,step=%d,arm=%d" % (a, b, c) for a in (1, 2) for b in (3, 4) for c in (0, 1)]

    def mean(self, name, key):
        a, b, c = (int(p.partition("=")[2]) for p in key.split(","))
        return float(100 * c + 10 * a + b)


def _arm_host(z, x=("size", [1, 2]), y=("step", [3, 4])):
    h = _Host(panels=1)
    h.settings = ["size", "step", "arm"]
    h.table = _ThreeSettings()
    h.image(0, z, x=x, y=y, what="growth")
    return h.record()["panels"][0]["series"][0]["what"]


def test_image_needs_one_slice_two_different_settings_and_placed_coordinates():
    assert _arm_host([[13.0, 23.0], [14.0, 24.0]]) == "growth"  # arm 0 throughout
    assert _arm_host([[113.0, 123.0], [114.0, 124.0]]) == "growth"  # arm 1 throughout
    assert _arm_host([[13.0, 123.0], [14.0, 24.0]]) == "other"  # a patchwork of two slices
    assert _arm_host([[13.0, 23.0], [14.0, 24.0]], y=("size", [3, 4])) == "other"  # both ids name one setting
    assert _arm_host([[23.0, 13.0], [24.0, 14.0]], x=("size", [2, 1])) == "other"  # descending columns
    assert _arm_host([[13.0, 23.0, 33.0], [14.0, 24.0, 34.0]], x=("size", [1, 2, 4])) == "other"  # uneven columns
    assert _arm_host([[13.0, 23.0], [14.0, 24.0]], x=[1, 2]) == "other"  # no ids


def test_band_keeps_ci95_or_sd_only_when_it_is_the_tables_own():
    h = _Host(panels=1)
    h.table = _TableByX()
    keys = h.table.keys()
    xs = [float(k.partition("=")[2]) for k in keys]
    ys = [h.table.mean("growth", k) for k in keys]
    h.line(0, "fast", xs, ys, "growth")
    ci = [h.table.interval("growth", k) for k in keys]
    h.band(0, "fast", xs, [a for a, _b in ci], [b for _a, b in ci], "growth", "ci95")
    h.band(0, "fast", xs, [a - 0.5 for a, _b in ci], [b + 0.5 for _a, b in ci], "growth", "ci95")
    assert [b["meaning"] for b in h.record()["panels"][0]["bands"]] == ["ci95", "other"]


class _TableByX(_Table):
    """Two settings with their own mean, 95% interval and sd (the widths a drawing's bars may be checked against)."""

    def keys(self):
        return ["size=1", "size=2"]

    def mean(self, name, key):
        return {"size=1": 2.0, "size=2": 4.0}[key]

    def interval(self, name, key):
        return {"size=1": (1.5, 3.0), "size=2": (3.0, 6.0)}[key]

    def sd(self, name, key):
        return {"size=1": 0.5, "size=2": 1.0}[key]


def _host_by_x(panels=2):
    h = _Host(panels)
    h.table = _TableByX()
    return h


def test_uneven_error_bars_are_recorded_and_their_meaning_is_checked_against_the_table():
    h = _host_by_x()
    h.line(0, "fast", [1, 2], [2.0, 4.0], "growth", yerr=([0.5, 1.0], [1.0, 2.0]), yerr_meaning="ci95")
    h.points(0, "slow", [1, 2], [2.0, 4.0], "growth", yerr=([0.1, 0.2], [0.3, 0.4]), yerr_meaning="ci95")
    h.bars(1, "fast", [1, 2], [2.0, 4.0], "growth", yerr=[0.5, 1.0], yerr_meaning="sd")
    h.line(1, "slow", [1, 2], [2.0, 4.0], "growth", yerr="ci95")
    h.points(1, "fast", [1, 2], [2.0, 4.0], "growth", yerr="sd")
    a, b = h.record()["panels"]
    s = a["series"]
    assert (s[0]["yerr_lo"], s[0]["yerr_hi"], s[0]["yerr"], s[0]["yerr_meaning"]) == ([0.5, 1.0], [1.0, 2.0], None, "ci95")
    assert s[1]["yerr_meaning"] == "other" and s[1]["yerr_lo"] == [0.1, 0.2]  # widths the table does not give: no meaning claim
    assert (b["series"][0]["yerr"], b["series"][0]["yerr_lo"], b["series"][0]["yerr_meaning"]) == ([0.5, 1.0], [0.5, 1.0], "sd")
    assert (b["series"][1]["yerr_lo"], b["series"][1]["yerr_hi"], b["series"][1]["yerr_meaning"]) == ([0.5, 1.0], [1.0, 2.0], "ci95")
    assert (b["series"][2]["yerr_lo"], b["series"][2]["yerr_meaning"]) == ([0.5, 1.0], "sd")
    with pytest.raises(ValueError):
        h.line(0, "fast", [1, 2], [2.0, 4.0], "growth", yerr="ci95", yerr_meaning="sd")
    # the drawn bars reach the recorded sides: the y limits hold the upper ends
    h.finish({}, {})
    assert h.axes[0].get_ylim()[1] > 6.0


def test_ref_has_three_sources_and_a_stated_value_gets_a_neutral_label():
    h = _host_by_x()
    h.ref(0, "limit")
    h.ref(0, at="growth", key="size=2")
    h.ref(0, value=1.5)
    h.ref(1, value=7, axis="x")
    refs = h.record()["panels"][0]["refs"]
    assert [r["source"] for r in refs] == ["plan", "table", "stated in the figure"]
    assert refs[1]["value"] == 4.0 and refs[1]["axis"] == "y"
    assert refs[2]["value"] == 1.5 and refs[2]["label"] == "Reference line"
    assert len({r["id"] for r in refs}) == 3
    with pytest.raises(ValueError):
        h.ref(0, "limit", value=1.0)
    with pytest.raises(ValueError):
        h.ref(0, value=math.nan)
    with pytest.raises(ValueError):
        h.ref(0)
    h.bars(1, "fast", ["a", "b"], [1.0, 2.0], "growth")
    with pytest.raises(ValueError):
        h.ref(1, value=1.0, axis="x")  # categories have no coordinate


def test_note_without_a_position_goes_to_a_free_corner_away_from_the_legend():
    h = _host_by_x(1)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth")  # data runs from the lower left to the upper right
    h.note(0, what="mean growth: {growth}", key="size=1")
    h.note(0, "runs: {growth}", form="n", key="size=1")
    n = h.record()["panels"][0]["notes"]
    assert n[0]["x"] is None and n[0]["text"] == "mean growth: 2" and n[0]["corner"] is None
    h.finish({}, {})
    n = h.record()["panels"][0]["notes"]
    legend = h._fc_extra[0]["legend_corner"]
    assert n[0]["corner"] in ("upper left", "lower right") and n[0]["corner"] != legend
    assert n[1]["corner"] != n[0]["corner"] and n[1]["corner"] != legend
    pos = h._fc_extra[0]["notes"][0].get_position()
    assert pos[0] in (0.02, 0.98) and h._fc_extra[0]["notes"][0].get_transform() is h.axes[0].transAxes
    h.derived_ids = ["slope"]
    h.table.mean = lambda name, key: 0.5
    h.note(0, 1.0, 1.0, "fit: {slope}", key="size=1")
    assert h.record()["panels"][0]["notes"][2]["text"] == "fit: 0.5"


def test_band_around_a_step_line_follows_the_steps():
    h = _host_by_x(1)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth", step=True)
    h.line(0, "slow", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    stepped = h.band(0, "fast", [1, 2, 3], [0.0, 1.0, 2.0], [2.0, 3.0, 4.0], "growth", "sd")
    plain = h.band(0, "slow", [1, 2, 3], [0.0, 1.0, 2.0], [2.0, 3.0, 4.0], "growth", "sd")
    assert any(abs(x - 2.0) < 1e-9 and abs(y - 0.0) < 1e-9 for x, y in stepped.get_paths()[0].vertices)  # the lo of x=1 holds up to x=2
    assert not any(abs(x - 2.0) < 1e-9 and abs(y - 0.0) < 1e-9 for x, y in plain.get_paths()[0].vertices)


def test_note_writes_a_setting_value_and_names_the_known_settings_when_unknown():
    h = _host_by_x(1)
    h.line(0, "fast", [1, 2], [2.0, 4.0], "growth")
    h.note(0, 1.0, 1.0, "run at size {size}", key="size=2")
    n = h.record()["panels"][0]["notes"][0]
    assert n["text"] == "run at size 2" and n["source"] == "setting" and n["setting"] == "size"
    with pytest.raises(ValueError) as e:
        h.note(0, 1.0, 1.0, "at {width}", key="size=2")
    assert "size" in str(e.value) and "step" in str(e.value)
    with pytest.raises(ValueError):
        h.note(0, 1.0, 1.0, "size 640 is {size}", key="size=2")


def test_a_note_refuses_a_setting_key_the_study_does_not_hold():
    h = _host_by_x(1)
    with pytest.raises(ValueError) as e:
        h.note(0, 1.0, 1.0, "run at size {size}", key="size=640")
    assert "size=1" in str(e.value) and "size=2" in str(e.value)
    assert h.record()["panels"][0]["notes"] == []


def test_bars_keep_ci95_only_when_every_point_matches_the_table():
    h = _host_by_x(1)
    h.line(0, "fast", [1, 2], [2.0, 4.0], "growth", yerr=([0.5, 9.0], [1.0, 9.0]), yerr_meaning="ci95")  # right at size 1 only
    h.line(0, "slow", [1, 7], [2.0, 4.0], "growth", yerr=([0.5, 9.0], [1.0, 9.0]), yerr_meaning="ci95")  # size 7 has no key
    h.points(0, "fast", [1, 2], [2.0, 4.0], "growth", yerr=([0.5, 1.0], [1.0, 2.0]), yerr_meaning="ci95")
    assert [s["yerr_meaning"] for s in h.record()["panels"][0]["series"]] == ["other", "other", "ci95"]


def test_sd_without_an_sd_read_is_the_sample_sd_of_the_values():
    class _NoSd(_Table):  # no `sd` read at all
        def keys(self):
            return ["size=1", "size=2"]

        def values(self, name, key):
            return [1.0, 3.0] if key == "size=1" else [2.0, 4.0, 6.0]

    h = _host_by_x(1)
    h.table = _NoSd()
    h.line(0, "fast", [1, 2], [2.0, 4.0], "growth", yerr="sd")
    s = h.record()["panels"][0]["series"][0]
    assert s["yerr_lo"] == pytest.approx([math.sqrt(2.0), 2.0]) and s["yerr_meaning"] == "sd"


def test_bars_are_checked_against_the_key_whose_mean_is_the_plotted_point():
    class _Two(_Table):  # two settings vary: several keys share one x
        def keys(self):
            return ["size=1,step=a", "size=1,step=b"]

        def mean(self, name, key):
            return 2.0 if key.endswith("a") else 5.0

        def interval(self, name, key):
            return (1.0, 3.0) if key.endswith("a") else (4.0, 7.0)

    h = _Host(1)
    h.table = _Two()
    h.line(0, "fast", [1], [2.0], "growth", yerr=([1.0], [1.0]), yerr_meaning="ci95")
    h.line(0, "slow", [1], [5.0], "growth", yerr=([1.0], [1.0]), yerr_meaning="ci95")  # the widths of the other key
    assert [s["yerr_meaning"] for s in h.record()["panels"][0]["series"]] == ["ci95", "other"]


def test_y_check_skips_gaps_and_flags_a_wrong_point():
    h = _Host(panels=2, identity=True)
    h.line(0, "fast", [1, 2, 3], [1.0, None, float("nan")], "growth")  # gaps are not compared
    h.line(1, "fast", [1, 2, 3], [1.0, 2.5, 3.0], "growth")
    p0, p1 = h.record()["panels"]
    assert p0["series"][0]["what"] == "growth" and p1["series"][0]["what"] == "other"
    assert p1["y"]["what"] == "other"


def test_y_form_names_the_statistic_drawn():
    h = _Host(panels=1, identity=True)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth", form="sd")  # the identity table has no spread equal to y
    h.line(0, "slow", [1, 2], [1.0, 2.0], "growth", form="no_such_form")
    assert [s["what"] for s in h.record()["panels"][0]["series"]] == ["other", "other"]
