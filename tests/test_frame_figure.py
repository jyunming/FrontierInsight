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


class _Host(frame_figure.FigureCalls):
    def __init__(self, panels=2):
        self.fig, axes = plt.subplots(1, panels, squeeze=False)
        self.axes = list(axes[0])
        self.table = _Table()
        self.x = "size"
        self.settings = ["size", "step"]
        self.series_ids = ["fast", "slow"]
        self.references = {"limit": {"value": 3.0, "label": "upper limit"}}
        self.labels = {"growth": {"label": "Growth", "scale": "linear"}, "fast": "Fast colony", "size": "Size"}


def _data(host):
    return [a for ax in host.axes for a in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.images)]


def test_line_records_series_x_y_what():
    h = _Host()
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth")
    p = h.record()["panels"][0]
    assert p["panel"] == 1 and p["raw"] is False and p["raw_reason"] is None
    assert p["x"] == {"id": "size", "kind": "setting", "label": "Size", "scale": "linear"}
    assert p["y"] == {"what": "growth", "label": "Growth", "scale": "linear"}
    s = p["series"][0]
    assert (s["id"], s["label"], s["kind"], s["step"]) == ("fast", "Fast colony", "line", False)
    assert s["x"] == [1.0, 2.0, 3.0] and s["y"] == [1.0, 2.0, 4.0] and s["yerr"] is None and s["merged_with"] == []
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
    assert [b["meaning"] for b in h.record()["panels"][0]["bands"]] == ["ci95", "sd", "range", "other"]
    with pytest.raises(ValueError) as e:
        h.band(0, "fast", [1, 2], [0, 1], [2, 3], "growth", "stderr")
    assert "ci95" in str(e.value) and "range" in str(e.value)


def test_ref_unknown_id_raises_naming_known():
    h = _Host()
    h.ref(0, "limit")
    h.ref(1, "limit", axis="x")
    r = h.record()["panels"][0]["refs"][0]
    assert r == {"id": "limit", "axis": "y", "value": 3.0, "label": "upper limit"}
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
    assert s[1]["source"] == "computed in the figure" and "theory" in s[1]["label"]
    assert s[2]["source"] == "fitted to the drawn points"


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
    assert b["x"] == [1.0, 2.0, 3.0] and b["lo"] == [0.0, 1.0, 2.0] and b["meaning"] == "sd"


def test_error_bars_record_their_meaning():
    h = _Host()
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth", yerr=[0.1, 0.2], yerr_meaning="ci95")
    h.points(0, "slow", [1, 2], [1.0, 2.5], "growth", yerr=[0.1, 0.2])
    h.bars(1, "fast", [1, 2], [1.0, 2.0], "growth", yerr=[-0.1, 0.2], yerr_meaning="sd")
    h.line(1, "slow", [1, 2], [1.0, 2.0], "growth")
    r = h.record()["panels"]
    assert [s["yerr_meaning"] for s in r[0]["series"]] == ["ci95", "other"]
    assert r[1]["series"][0]["yerr_meaning"] == "sd" and r[1]["series"][0]["yerr"] == [0.1, 0.2]
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
    assert h.record()["version"] == 1 and h.record()["fallback"] is False
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
    h = _Host(panels=1)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth")
    h.axes[0].set_ylabel("energy drift")
    h.axes[0].set_title("A title")
    with caplog.at_level("WARNING", logger="frame_figure"):
        h.finish(SPEC, INFO)
    assert h.axes[0].get_ylabel() == "Growth (1/h)" == h.record()["panels"][0]["y"]["label"]
    assert h.axes[0].get_xlabel() == "Size (mm)" and h.axes[0].get_title() == ""


def test_label_fallback_order():
    h = _Host(panels=3)
    for i, what in enumerate(["growth", "load", "load"]):
        h.line(i, "fast", [1, 2], [1.0, 2.0], what)
    h.finish({"growth": {"label": "Growth", "unit": "1/h"}}, {"load": {"description": "Load carried by the colony at the end; read at 3 h"}})
    assert h.axes[0].get_ylabel() == "Growth (1/h)"
    assert h.axes[1].get_ylabel() == "Load carried by the colony at the end"
    h2 = _Host(panels=1)
    h2.line(0, "fast", [1, 2], [1.0, 2.0], "load")
    h2.finish({}, {"load": {"unit": "kg"}})
    assert h2.axes[0].get_ylabel() == "value (kg)" and h2.axes[0].get_xlabel() == "value"
    assert "load" not in h2.axes[0].get_ylabel()


def test_two_results_and_other_quantity_labels():
    h = _Host(panels=2)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.line(0, "slow", [1, 2], [3.0, 2.0], "load")
    h.line(1, "fast", [1, 2], [1.0, 2.0], "other")
    h.finish(SPEC, INFO)
    assert h.axes[0].get_ylabel() == "Growth (1/h) / Load carried by the colony at the end"
    assert h.axes[1].get_ylabel() == "computed value"


def test_legend_has_no_empty_entry():
    h = _Host(panels=1)
    ax = h.axes[0]
    ax.plot([1, 2], [1, 2], label="_stale")  # a body's leftover artist with a private label
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth", yerr=[0.1, 0.1, 0.1])
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
    h = _Host(panels=1)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 4.0], "growth", yerr=[0.1, 0.2, 0.3])
    h.line(0, "slow", [1, 2, 3], [1.0 + 1e-14, 2.0, 4.0], "growth", yerr=[0.1, 0.2, 0.3])
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
    c = _Host(panels=1)
    for k in range(7):
        c.line(0, "fast" if k % 2 == 0 else "slow", [1, 2, 3], [1.0 + k, 2.0 + k, 3.0 + k], "growth")
    c.finish(SPEC, INFO)
    assert c.axes[0].get_position().width < 0.8 * _Host(panels=1).axes[0].get_position().width
