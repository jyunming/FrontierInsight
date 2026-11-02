"""The claims FI can state about a figure from its call record (core/figure_claims.py)."""

from __future__ import annotations

import json
import math
import random
import re

import pytest

from core import figure_claims as fc


def _series(sid, xs, ys, kind="line", **kw):
    return {"id": sid, "label": sid.upper(), "kind": kind, "step": False, "x": xs, "y": ys, "yerr": None,
            "merged_with": [], **kw}


def _panel(series, n=1, what="signal", **kw):
    return {"panel": n, "raw": False, "raw_reason": None,
            "x": {"id": "size", "kind": "setting", "label": "size", "scale": "linear"},
            "y": {"what": what, "label": "signal level", "scale": "linear"},
            "series": series, "bands": [], "refs": [], "notes": [], "texts_raw": [], **kw}


def _calls(*panels, fallback=False):
    return {"version": 1, "fallback": fallback, "panels": list(panels)}


def _base():
    return _calls(_panel([
        _series("a", [1, 2, 3, 4], [1.0, 2.0, 3.0, 5.0]),
        _series("b", [1, 2, 3, 4], [4.0, 3.0, 2.5, 0.5]),
        _series("c", [1, 2, 3, 4], [1.0, 2.0, 3.0, 5.0]),
    ]))


def ev(calls, cid, **args):
    return fc.evaluate(calls, cid, {k: str(v) for k, v in args.items()})


def test_drawn():
    got = ev(_base(), "drawn", series="a")
    assert got and "A is drawn" in got[0] and got[1] == {"series": "a", "panel": "1"}
    assert ev(_base(), "drawn", series="zz") is None


def test_not_drawn():
    plan = {"series": ["a", "ghost"]}
    assert fc.evaluate(_base(), "not_drawn", {"series": "ghost"}, plan=plan)
    assert fc.evaluate(_base(), "not_drawn", {"series": "a"}, plan=plan) is None   # drawn
    assert fc.evaluate(_base(), "not_drawn", {"series": "invented"}, plan=plan) is None  # not a plan series
    assert fc.evaluate(_base(), "not_drawn", {"series": "ghost"}) is None


def test_range():
    got = ev(_base(), "range", series="b", axis="y")
    assert got and got[0].startswith("B spans 0.5 to 4")
    assert ev(_base(), "range", series="b", axis="x")[0].startswith("B spans 1 to 4")
    other = _calls(_panel([_series("a", [1, 2], [1, 2])], what="other"))
    assert ev(other, "range", series="a", axis="y") is None
    assert ev(other, "range", series="a", axis="x")


def test_value_at():
    got = ev(_base(), "value_at", series="a", x=3)
    assert got and "A is 3 at size 3" in got[0]
    assert ev(_base(), "value_at", series="a", x=9) is None


def test_value_at_between_drawn_points_is_not_evaluable():
    assert ev(_base(), "value_at", series="a", x=2.5) is None


def test_trend():
    assert "increases" in ev(_base(), "trend", series="a")[0]
    assert "decreases" in ev(_base(), "trend", series="b")[0]
    mixed = _calls(_panel([_series("m", [1, 2, 3, 4], [1, 3, 2, 4])]))
    assert "does not change monotonically" in ev(mixed, "trend", series="m")[0]
    tied = _calls(_panel([_series("t", [1, 2, 3, 4], [1, 2, 2.0000001, 4])]))
    assert "increases" in ev(tied, "trend", series="t")[0]
    two = _calls(_panel([_series("t", [1, 2], [1, 2])]))
    assert ev(two, "trend", series="t") is None
    flat = _calls(_panel([_series("t", [1, 2, 3], [2, 2, 2])]))
    assert ev(flat, "trend", series="t") is None
    step = _calls(_panel([_series("t", [1, 2, 3], [3, 2, 1], step=True)]))
    assert ev(step, "trend", series="t") is None


def test_steps():
    step = _calls(_panel([_series("t", [0, 5, 10], [1.0, 0.7, 0.4], step=True)]))
    got = ev(step, "steps", series="t")
    assert got and "falls from 1 to 0.4" in got[0]
    assert ev(_base(), "steps", series="a") is None


def test_order():
    got = ev(_base(), "order", at=3)
    assert got is None  # a and c are the same data: a tie has no order
    assert ev(_base(), "order", at=1) is None
    cc = _calls(_panel([_series("a", [1, 2], [1, 5]), _series("b", [1, 2], [3, 2]), _series("c", [2], [9])]))
    assert ev(cc, "order", at=2)[0].split("are ")[1] == "C, A, B"
    solo = _calls(_panel([_series("a", [1, 2], [1, 5]), _series("b", [1], [3])]))
    assert ev(solo, "order", at=2) is None


def test_crossing_and_no_crossing():
    got = ev(_base(), "crossing", a="a", b="b")
    assert got and "between size 2 and 3" in got[0]
    assert ev(_base(), "no_crossing", a="a", b="b") is None
    apart = _calls(_panel([_series("a", [1, 2, 3], [1, 2, 3]), _series("b", [1, 2, 3], [5, 6, 7])]))
    assert "do not cross" in ev(apart, "no_crossing", a="a", b="b")[0]
    assert ev(apart, "crossing", a="a", b="b") is None
    one = _calls(_panel([_series("a", [1], [1]), _series("b", [1], [5])]))
    assert ev(one, "no_crossing", a="a", b="b") is None


def test_identical():
    got = ev(_base(), "identical", a="a", b="c")
    assert got and "same data" in got[0]
    assert ev(_base(), "identical", a="a", b="b") is None
    merged = _calls(_panel([_series("a", [1, 2], [1, 2], merged_with=["z"])]))
    assert "drawn once" in ev(merged, "identical", a="a", b="z")[0]


def test_reference():
    c = _calls(_panel([_series("a", [1, 2], [1, 2])], refs=[{"id": "r1", "axis": "y", "value": 0.05, "label": "limit"}]))
    got = ev(c, "reference", ref="r1")
    assert got and "y = 0.05" in got[0] and "limit" in got[0]
    assert ev(c, "reference", ref="r2") is None


def test_uneven_errorbars_and_stated_reference_lines():
    lo_hi = {"yerr": None, "yerr_lo": [0.1, 0.2], "yerr_hi": [0.3, 0.5], "yerr_meaning": "ci95"}
    c = _calls(_panel([_series("a", [1, 2], [1, 2], **lo_hi)]))
    text = ev(c, "errorbars", series="a")[0]
    assert "95% confidence interval" in text and "-0.2" in text and "+0.5" in text
    other = _calls(_panel([_series("a", [1, 2], [1, 2], **{**lo_hi, "yerr_meaning": "other"})]))
    assert ev(other, "errorbars", series="a") is None
    same = _calls(_panel([_series("a", [1, 2], [1, 2], **{**lo_hi, "yerr_hi": [0.1, 0.2]})]))
    assert "reaching" not in ev(same, "errorbars", series="a")[0]
    ref = {"id": "refline1", "axis": "y", "value": 1.0, "label": "Reference line", "source": "stated in the figure"}
    c = _calls(_panel([_series("a", [1, 2], [1, 2])], refs=[ref]))
    got = ev(c, "reference", ref="refline1")[0]
    assert "y = 1" in got and "Reference line" not in got and "limit" not in got and got.startswith("a line is drawn")


def test_errorbars():
    c = _calls(_panel([_series("a", [1, 2], [1, 2], yerr=[0.1, 0.2], yerr_meaning="sd")]))
    assert "standard deviation" in ev(c, "errorbars", series="a")[0]
    zero = _calls(_panel([_series("a", [1, 2], [1, 2], yerr=[0, 0])]))
    assert ev(zero, "errorbars", series="a") is None
    other = _calls(_panel([_series("a", [1, 2], [1, 2], yerr=[1, 1], yerr_meaning="other")]))
    assert ev(other, "errorbars", series="a") is None
    none = _calls(_panel([_series("a", [1, 2], [1, 2])]))
    assert ev(none, "errorbars", series="a") is None
    assert ev(c, "errorbar", series="a") is None  # the id is errorbars


def test_band():
    band = {"series": "a", "lo": [0, 1], "hi": [2, 3], "meaning": "ci95"}
    c = _calls(_panel([_series("a", [1, 2], [1, 2])], bands=[band]))
    assert "95% confidence interval" in ev(c, "band", series="a")[0]
    c2 = _calls(_panel([_series("a", [1, 2], [1, 2])], bands=[{**band, "meaning": "other"}]))
    assert ev(c2, "band", series="a") is None


def test_missing_x():
    c = _calls(_panel([_series("a", [1, 2, 4], [1, 2, 4], kind="points")]))
    got = fc.evaluate(c, "missing_x", {"series": "a"}, plan={"grid": [1, 2, 3, 4]})
    assert got and "no point at size 3" in got[0]
    assert fc.evaluate(c, "missing_x", {"series": "a"}) is None
    assert fc.evaluate(c, "missing_x", {"series": "a"}, plan={"grid": [1, 2, 4]}) is None
    line = _calls(_panel([_series("a", [1, 2, 4], [1, 2, 4])]))
    assert fc.evaluate(line, "missing_x", {"series": "a"}, plan={"grid": [1, 2, 3]}) is None


def test_scale():
    c = _calls(_panel([_series("a", [1, 2], [1, 2])]))
    c["panels"][0]["x"]["scale"] = "log"
    assert "x axis is logarithmic" in ev(c, "scale", axis="x")[0]
    assert "y axis is linear" in ev(c, "scale", axis="y")[0]
    assert ev(c, "scale", axis="z") is None


def test_layout_and_lines_delegate():
    record = {"axes": [{"title": "A", "ylabel": "y", "place": [0.1, 0.15, 0.35, 0.7], "series": []},
                       {"title": "B", "ylabel": "y", "place": [0.55, 0.15, 0.35, 0.7], "series": []}]}
    lines = {"axes": [{"lines": [{"label": "s", "kind": "data", "x": [0, 1, 2], "y": [1, 2, 3], "linestyle": "-",
                                  "marker": "None"}]}]}
    got = fc.evaluate(_base(), "layout", {}, record=record, lines_data=lines)
    assert got and got[0].startswith("two panels side by side")
    got = fc.evaluate(_base(), "lines", {}, record=record, lines_data=lines)
    assert got and "solid lines are drawn" in got[0]
    assert fc.evaluate(_base(), "layout", {}) is None
    assert fc.evaluate(_base(), "lines", {}) is None


def test_note():
    c = _calls(_panel([_series("a", [1, 2], [1, 2])], notes=[{"id": "n1", "text": "onset", "x": 1, "y": 1}]))
    assert ev(c, "note", note="n1")[0] == "onset"
    assert ev(c, "note", note="n2") is None


def test_curve_supports_only_drawn_and_range():
    c = _calls(_panel([_series("k", [1, 2, 3, 4], [1, 2, 3, 4], kind="curve"), _series("a", [1, 2, 3, 4], [4, 3, 2, 1])]))
    assert ev(c, "drawn", series="k") and ev(c, "range", series="k", axis="y")
    assert ev(c, "trend", series="k") is None
    assert ev(c, "value_at", series="k", x=2) is None
    assert ev(c, "crossing", a="k", b="a") is None
    assert ev(c, "identical", a="k", b="a") is None
    assert all(o["claim"] in ("drawn", "range") for o in fc.menu(c) if "series" in o["args"] and o["args"]["series"] == "k")


def test_unknown_claim_id_is_not_evaluable():
    assert ev(_base(), "argmax", series="a") is None
    assert fc.evaluate(_base(), 5, {}) is None  # type: ignore[arg-type]
    assert ev(_base(), "drawn", series="a", extra="x") is None  # an argument the claim does not take
    assert ev(_base(), "drawn") is None


def test_claim_on_raw_panel_is_not_evaluable():
    raw = _panel([_series("a", [1, 2, 3], [1, 2, 3])], raw=True, raw_reason="custom artist")
    c = _calls(raw)
    for cid, args in (("drawn", {"series": "a"}), ("trend", {"series": "a"}), ("scale", {"axis": "x"})):
        assert fc.evaluate(c, cid, args) is None
    assert fc.menu(c) == []


def test_panel_argument():
    two = _calls(_panel([_series("a", [1, 2, 3], [1, 2, 3])], n=1), _panel([_series("b", [1, 2, 3], [3, 2, 1])], n=2))
    assert ev(two, "drawn", series="a") is None  # which panel?
    assert ev(two, "drawn", series="b", panel=2)[0].endswith("in panel 2")
    assert ev(two, "drawn", series="b", panel=1) is None
    assert ev(two, "drawn", series="b", panel="x") is None


def test_no_call_record_is_not_evaluable():
    assert fc.evaluate(None, "drawn", {"series": "a"}) is None
    assert fc.evaluate({}, "drawn", {"series": "a"}) is None
    assert fc.menu(None) == []


def test_other_quantity_offers_shape_claims_but_no_value_claims():
    c = _calls(_panel([_series("a", [1, 2, 3], [3, 1, 2]), _series("b", [1, 2, 3], [1, 2, 3])], what="other"))
    claims = {o["claim"] for o in fc.menu(c)}
    assert "drawn" in claims and "scale" in claims
    assert not claims & {"value_at", "steps"}  # no y number
    assert {"trend", "order", "crossing"} <= claims  # the shape (rising, order, crossing) is still true of the drawing
    assert all(o["args"].get("axis") == "x" for o in fc.menu(c) if o["claim"] == "range")


def test_nan_and_single_point_series_fail_closed_without_raising():
    nan = float("nan")
    c = _calls(_panel([_series("n", [1, 2, 3], [nan, nan, nan]), _series("s", [1], [1.0]),
                       _series("m", [1, 2, 3], [1.0, nan, 3.0]), _series("bad", [1, 2], [1.0]),
                       _series("str", ["p", "q"], [1.0, 2.0], kind="bars"), _series("inf", [1, 2, 3], [1, math.inf, 3])]))
    for sid in ("n", "s", "m", "bad", "str", "inf"):
        for cid in ("trend", "steps", "value_at", "crossing", "identical", "errorbars"):
            args = {"series": sid, "x": "2"} if cid == "value_at" else {"a": sid, "b": "n"} if cid in (
                "crossing", "identical") else {"series": sid}
            assert fc.evaluate(c, cid, args) is None
    assert ev(c, "value_at", series="s", x=1) and ev(c, "value_at", series="m", x=1)  # finite points still answer
    assert ev(c, "drawn", series="n") is None and ev(c, "drawn", series="s")
    fc.menu(c)  # does not raise
    assert fc.evaluate({"panels": [None, 3, {"series": "x"}]}, "drawn", {"series": "a"}) is None
    assert fc.evaluate({"panels": "x"}, "scale", {"axis": "x"}) is None


def test_parse_args():
    assert fc.parse_args("series=a, panel=1") == {"series": "a", "panel": "1"}
    assert fc.parse_args("") == {}
    assert fc.evaluate(_base(), "drawn", fc.parse_args("series=a, junk")) is None
    assert fc.evaluate(_base(), "drawn", fc.parse_args("series=a, series=b")) is None


def test_load_calls(tmp_path):
    folder = tmp_path / ".fi" / "figure_records"
    folder.mkdir(parents=True)
    (folder / "fig1.calls.json").write_text(json.dumps(_base()), encoding="utf-8")
    (folder / "bad.calls.json").write_text("{not json", encoding="utf-8")
    (folder / "list.calls.json").write_text("[]", encoding="utf-8")
    assert fc.load_calls(tmp_path, "fig1") == _base()
    assert fc.load_calls(str(tmp_path), "fig1") == _base()
    assert fc.load_calls(tmp_path, "bad") is None
    assert fc.load_calls(tmp_path, "list") is None
    assert fc.load_calls(tmp_path, "missing") is None
    assert fc.load_calls(tmp_path, "../fig1") is None


def test_catalogue_ids():
    assert set(fc.CLAIMS) == {"drawn", "not_drawn", "range", "value_at", "trend", "order", "crossing", "no_crossing",
                              "identical", "reference", "errorbars", "band", "missing_x", "scale", "layout", "lines",
                              "note", "steps"}


def _random_record(rng):
    xs = sorted(rng.sample(range(1, 30), rng.randint(3, 7)))
    series = []
    for i in range(rng.randint(2, 4)):
        ys = [round(rng.uniform(-5, 5), 3) for _ in xs]
        series.append(_series(f"s{i}", list(xs), ys))
    return _calls(_panel(series, refs=[{"id": "r", "axis": "y", "value": 1.5, "label": "limit"}]))


def _recompute(calls, offer):
    """A plain recomputation of an offered claim from the arrays, independent of the module."""
    panel = calls["panels"][0]
    by = {s["id"]: s for s in panel["series"]}
    a = offer["args"]
    cid = offer["claim"]
    if cid == "drawn":
        return any(math.isfinite(v) for v in by[a["series"]]["y"])
    if cid == "range":
        vals = by[a["series"]]["x" if a["axis"] == "x" else "y"]
        return offer["text"].split("spans ")[1].startswith(f"{min(vals):.4g} to {max(vals):.4g}")
    if cid == "value_at":
        s = by[a["series"]]
        i = s["x"].index(float(a["x"]))
        return f"is {s['y'][i]:.4g} at" in offer["text"]
    if cid == "trend":
        ys = by[a["series"]]["y"]
        d = [q - p for p, q in zip(ys, ys[1:])]
        word = "increases" if all(v > 0 for v in d) else "decreases" if all(v < 0 for v in d) else \
            "does not change monotonically"
        return f" {word} " in offer["text"]
    if cid == "order":
        x = float(a["at"])
        got = sorted(((s["y"][s["x"].index(x)], s["label"]) for s in by.values() if x in s["x"]), reverse=True)
        return offer["text"].endswith("are " + ", ".join(n for _, n in got))
    if cid in ("crossing", "no_crossing"):
        da = [p - q for p, q in zip(by[a["a"]]["y"], by[a["b"]]["y"])]
        crossed = any((p > 0) != (q > 0) for p, q in zip(da, da[1:]))
        return crossed == (cid == "crossing")
    if cid == "identical":
        return by[a["a"]]["y"] == by[a["b"]]["y"]
    if cid == "reference":
        return "y = 1.5" in offer["text"]
    if cid == "scale":
        return offer["text"].startswith(f"the {a['axis']} axis is linear")
    return False


@pytest.mark.parametrize("seed", range(8))
def test_independent_recompute_agrees_on_every_offered_claim(seed):
    calls = _random_record(random.Random(seed))
    offers = fc.menu(calls)
    assert offers
    seen = set()
    for o in offers:
        assert _recompute(calls, o), o
        seen.add(o["claim"])
    assert {"drawn", "range", "value_at", "scale", "reference"} <= seen


def test_menu_offers_only_true_claims():
    calls = _base()
    offers = fc.menu(calls, stem="fig1")
    kinds = {(o["claim"], tuple(sorted(o["args"].items()))) for o in offers}
    assert ("trend", (("panel", "1"), ("series", "a"))) in kinds
    assert ("trend", (("panel", "1"), ("series", "b"))) in kinds
    assert ("identical", (("a", "a"), ("b", "c"), ("panel", "1"))) in kinds
    assert not any(o["claim"] == "identical" and o["args"]["b"] == "b" for o in offers)
    assert not any(o["claim"] == "no_crossing" and {o["args"]["a"], o["args"]["b"]} == {"a", "b"} for o in offers)
    assert any(o["claim"] == "crossing" for o in offers)
    for o in offers:
        assert fc.evaluate(calls, o["claim"], o["args"]) == (o["text"], o["args"])
        claim = o["claim"]
        assert o["mark"].startswith("{{figure:fig1:" + claim) and o["mark"].endswith("}}")
        if o["args"]:
            assert fc.parse_args(o["mark"].split(" @ ", 1)[1][:-2]) == o["args"]
    mark = next(o["mark"] for o in offers if o["claim"] == "trend" and o["args"]["series"] == "a")
    assert mark == "{{figure:fig1:trend @ series=a, panel=1}}"


def test_menu_layout_lines_and_not_drawn_need_their_inputs():
    plain = {o["claim"] for o in fc.menu(_base())}
    assert not plain & {"layout", "lines", "not_drawn", "missing_x"}
    more = {o["claim"] for o in fc.menu(_base(), {"series": ["ghost"]}, record={"axes": [
        {"title": "A", "ylabel": "y", "place": [0.1, 0.15, 0.8, 0.7], "series": []}]})}
    assert {"layout", "not_drawn"} <= more


def test_custom_formatter_is_used():
    got = fc.evaluate(_base(), "value_at", {"series": "a", "x": "3"}, lambda v: f"<{v}>")
    assert got and "<3.0>" in got[0]


# --- fix round 1 -------------------------------------------------------------------------------------------------------

def test_not_drawn_counts_a_merged_id_as_drawn():
    c = _calls(_panel([_series("a", [1, 2], [1, 2], merged_with=["z"])]))
    plan = {"series": ["a", "z", "ghost"]}
    assert fc.evaluate(c, "not_drawn", {"series": "z"}, plan=plan) is None
    assert fc.evaluate(c, "not_drawn", {"series": "ghost"}, plan=plan)
    offered = {o["args"]["series"] for o in fc.menu(c, plan) if o["claim"] == "not_drawn"}
    assert offered == {"ghost"}


def _bars():
    return _calls(_panel([_series("a", ["p", "q", "r"], [3.0, 1.0, None], kind="bars"),
                          _series("b", ["p", "q", "r"], [2.0, 4.0, 5.0], kind="bars")]))


def test_categorical_x_offers_drawn_value_at_order_only():
    c = _bars()
    assert ev(c, "drawn", series="a")
    assert "A is 3 at size p" in ev(c, "value_at", series="a", x="p")[0]
    assert ev(c, "value_at", series="a", x="r") is None   # a null point is skipped
    assert ev(c, "value_at", series="a", x="zz") is None
    assert ev(c, "order", at="q")[0].endswith("are B, A")
    for cid, args in (("trend", {"series": "a"}), ("crossing", {"a": "a", "b": "b"}),
                      ("no_crossing", {"a": "a", "b": "b"}), ("range", {"series": "a", "axis": "x"}),
                      ("missing_x", {"series": "a"})):
        assert fc.evaluate(c, cid, args, plan={"grid": [1, 2]}) is None
    assert ev(c, "range", series="a", axis="y")[0].startswith("A spans 1 to 3")
    claims = {o["claim"] for o in fc.menu(c)}
    assert {"drawn", "value_at", "order"} <= claims
    assert not claims & {"trend", "crossing", "no_crossing", "missing_x"}


def _image(nrow=2, ncol=3):
    z = [[float(r * ncol + k) for k in range(ncol)] for r in range(nrow)]
    return {"id": "img", "label": "IMG", "kind": "image", "step": False, "x": list(range(ncol)),
            "y": [100.0 * r for r in range(nrow)], "z": z, "yerr": None, "merged_with": []}


def test_image_series_only_drawn_and_range_of_z():
    line = _series("a", [1, 2, 3], [1.0, 2.0, 3.0])
    c = _calls(_panel([_image(), line]))
    assert ev(c, "drawn", series="img")           # not square: 3 columns, 2 rows
    assert ev(c, "range", series="img", axis="y")[0].startswith("IMG has values from 0 to 5")
    assert ev(c, "range", series="img", axis="x") is None
    assert ev(c, "range", axis="y")[0].startswith("the series spans 1 to 3")   # the rows never feed it
    assert ev(c, "range", axis="x")[0].startswith("the series spans 1 to 3")
    for cid, args in (("trend", {"series": "img"}), ("value_at", {"series": "img", "x": "1"}),
                      ("identical", {"a": "img", "b": "a"})):
        assert fc.evaluate(c, cid, args) is None
    assert not any(o["args"].get("series") == "img" and o["claim"] not in ("drawn", "range", "scale")
                   for o in fc.menu(c))
    empty = _calls(_panel([{**_image(), "z": [[None, float("nan")]]}]))
    assert ev(empty, "drawn", series="img") is None


def test_menu_never_raises_on_malformed_input():
    bad = _calls(_panel([_series("a", [1, 2, 3], [1, 2, 3], merged_with=5), _series("b", [1, 2], [10 ** 400, 2])],
                        refs="x", notes=5, bands=3))
    assert isinstance(fc.menu(bad, {"series": 5, "grid": "x"}), list)
    assert isinstance(fc.menu(bad, {"series": [5, {"id": 3}]}), list)
    assert isinstance(fc.menu(_calls(_panel("nope")), None), list)
    assert isinstance(fc.menu(_calls(_panel([{"id": "q", "x": 5, "y": 5}, "junk", None])), None), list)
    assert fc.evaluate(bad, "not_drawn", {"series": "z"}, plan={"series": ["z"]}) is None   # unreadable merged_with


def test_no_crossing_needs_the_same_x_points():
    cross = _calls(_panel([_series("a", [1, 2, 3], [0.0, 10.0, 0.0]), _series("b", [1, 3], [5.0, 5.0])]))
    assert ev(cross, "no_crossing", a="a", b="b") is None   # they visibly cross between the shared points
    assert not any(o["claim"] == "no_crossing" for o in fc.menu(cross))
    same = _calls(_panel([_series("a", [1, 2, 3], [0.0, 1.0, 0.0]), _series("b", [1, 2, 3], [5.0, 6.0, 5.0])]))
    assert "do not cross" in ev(same, "no_crossing", a="a", b="b")[0]


def test_errorbars_need_a_stated_meaning_and_band_a_known_series():
    for meaning in (None, "other"):
        c = _calls(_panel([_series("a", [1, 2], [1, 2], yerr=[1, 1], **({"yerr_meaning": meaning} if meaning else {}))]))
        assert ev(c, "errorbars", series="a") is None
    c = _calls(_panel([_series("a", [1, 2], [1, 2], yerr=[1, 1], yerr_meaning="ci95")]))
    assert "95% confidence interval" in ev(c, "errorbars", series="a")[0]
    band = {"series": "zz", "lo": [0, 1], "hi": [2, 3], "meaning": "sd"}
    c = _calls(_panel([_series("a", [1, 2], [1, 2])], bands=[band]))
    assert ev(c, "band", series="zz") is None


def test_trend_is_not_fooled_by_a_chain_of_small_steps():
    ys = [1.0, 0.9996, 0.9992, 0.9988, 0.9984, 1.0]
    c = _calls(_panel([_series("a", list(range(6)), ys)]))
    assert "does not change monotonically" in ev(c, "trend", series="a")[0]


# --- the pointer rule --------------------------------------------------------------------------------------------------

_MENTION = re.compile(r"(?<![A-Za-z])(?:figures?|figs?\.?)\s*(\d+)", re.IGNORECASE)


def _v(metric, text="0.5"):
    return f"<!-- fi:v {{{{v:{metric}@size=1}}}} -->{text}<!-- /fi:v -->"


def _claim(stem="fig_a", text="Panel 1 draws a rising line."):
    return f"<!-- fi:fact-figure-lines-{stem} -->{text}<!-- /fi:fact-figure-lines-{stem} -->"


def _rule(md, numbers=None, drawn=None):
    return fc.apply_pointer_rule(md, {"fig_a.png": 1} if numbers is None else numbers,
                                 {1: {"signal"}} if drawn is None else drawn, _MENTION)


def test_result_mark_sentence_with_pointer_kept_when_figure_draws_it():
    md = f"Intro text.\n\nThe signal is {_v('signal')} at size 1 (Figure 1). Next.\n"
    out, taken = _rule(md)
    assert out == md and taken == []


def test_gain_sentence_pointing_at_figure_that_does_not_draw_it_is_dropped():
    md = f"Intro text.\n\nThe gain is {_v('gain')} over the baseline (Figure 1). The signal is {_v('signal')}.\n"
    out, taken = _rule(md)
    assert "gain" not in out and "Figure 1" not in out
    assert f"The signal is {_v('signal')}." in out
    assert len(taken) == 1 and taken[0]["key"] == "Figure 1" and taken[0]["how"] == "sentence removed"
    assert taken[0]["sentence"] == "The gain is 0.5 over the baseline (Figure 1)."


def test_free_sentence_naming_figure_is_dropped():
    md = "First sentence. Figure 1 shows a piecewise linear curve. Last sentence.\n"
    out, taken = _rule(md)
    assert out == "First sentence. Last sentence.\n"
    assert len(taken) == 1


def test_pointer_not_closing_the_sentence_is_dropped():
    md = f"As (Figure 1) shows, the signal is {_v('signal')}.\n\nSee Figure 1 for the signal {_v('signal')}.\n"
    out, taken = _rule(md)
    assert "Figure" not in out and len(taken) == 2


def test_whole_sentence_figure_claim_kept():
    md = f"Intro. {_claim()} Later text.\n"
    out, taken = _rule(md)
    assert out == md and taken == []
    # FI's own sentence ends with its "(Figure N)": the claim and the pointer are one sentence, and it stays
    for tail in (" (Figure 1)", " (Figure 1)."):
        md = f"Intro. {_claim()}{tail}\n"
        assert _rule(md) == (md, [])


def test_figure_claim_with_other_words_is_dropped():
    md = f"{_claim()} (Figure 1) It also clearly flattens.\n"
    out, taken = _rule(md)
    assert "flattens" not in out and "Panel 1" in out
    assert len(taken) == 1


def test_caption_and_table_row_with_figure_number_untouched():
    md = ("# Figure 1 results\n\n![**Figure 1.** A caption that says Figure 1 shows a gain.](figures/fig_a.png)\n\n"
          "| Figure 1 shows | gain |\n|---|---|\n| Figure 1 | 3 |\n\n> Figure 1 quote.\n\n```\nFigure 1 shows x.\n```\n"
          "<!-- fi:fact-limitations -->\nFigure 1: some statements were left out.\n<!-- /fi:fact-limitations -->\n")
    out, taken = _rule(md)
    assert out == md and taken == []


def test_figure_number_not_issued_by_fi_is_dropped_sentence_when_it_names_a_checked_figure():
    md = f"Good one. The signal is {_v('signal')} (Figures 1 and 7). Last.\n"
    out, taken = _rule(md)
    assert out == "Good one. Last.\n"
    assert taken[0]["key"] == "Figure 1"


def test_a_sentence_naming_only_figures_without_a_call_record_is_left_exactly_as_it_was():
    # Figure 1 has a record; Figure 2 is legacy and Figure 9 does not exist: neither is judged (RF1)
    md = "Figure 2 shows a curve, and Figure 9 shows another.\n\nFigure 2 (the line) rises. See also Figure 9.\n"
    nums = {"fig_a.png": 1, "old.png": 2}
    assert fc.apply_pointer_rule(md, nums, {1: {"signal"}}, _MENTION) == (md, [])
    md_both = md + "Figure 1 shows a rise and Figure 2 a fall.\n"
    out, taken = fc.apply_pointer_rule(md_both, nums, {1: {"signal"}}, _MENTION)
    assert out == md and len(taken) == 1, "a sentence that names a figure with a record is judged"


def test_legacy_figure_passes_through():
    md = "Figure 2 shows a rising curve.\n"
    out, taken = _rule(md, numbers={"fig_a.png": 1, "old.png": 2}, drawn={1: {"signal"}})
    assert out == md and taken == []


def test_unfilled_mark_does_not_count_as_drawn():
    md = f"The signal is {_v('signal', 'not recorded')} (Figure 1).\n"
    assert _rule(md)[0] == ""




def test_run_in_heading_stays_when_sentence_dropped():
    out, taken = _rule("**Result.** Figure 1 shows a rise.\n")
    assert out == "**Result.**\n"
    assert len(taken) == 1


def test_claim_marker_with_arguments_is_kept():
    md = f"Intro. {_claim('fig_a-args1f')} Later text.\n"
    assert _rule(md) == (md, [])


def test_pointer_listing_several_figures():
    nums = {"fig_a.png": 1, "fig_b.png": 2}
    drawn = {1: {"signal"}, 2: {"gain"}}
    for pointer in ("(Figures 1 and 2)", "(Figures 1, 2)", "(Figures 1–2)", "(Fig. 1 and Fig. 2)"):
        md = f"The signal is {_v('signal')} {pointer}. Next.\n"
        assert fc.apply_pointer_rule(md, nums, drawn, _MENTION) == (md, []), pointer
    md = f"The gain is {_v('gain')} (Figures 1 and 2). Next.\n"
    assert fc.apply_pointer_rule(md, nums, drawn, _MENTION)[0] == md
    md = f"The size is {_v('size')} (Figures 1 and 2). Next.\n"
    out, taken = fc.apply_pointer_rule(md, nums, drawn, _MENTION)
    assert out == "Next.\n" and taken[0]["sentence"] == "The size is 0.5 (Figures 1 and 2)."
    md = f"The signal is {_v('signal')} (Figures 1 and 5). Next.\n"
    assert fc.apply_pointer_rule(md, nums, drawn, _MENTION)[0] == "Next.\n"


def test_abbreviations_do_not_cut_a_removed_sentence():
    gain = _v("gain")
    out, taken = _rule(f"The gain is {gain} (Figure 1), e.g. at size 1. Next.\n")
    assert out == "Next.\n" and taken[0]["sentence"] == "The gain is 0.5 (Figure 1), e.g. at size 1."
    out, taken = _rule(f"Smith et al. report the gain {gain} (Figure 1). Next.\n")
    assert out == "Next.\n" and taken[0]["sentence"].startswith("Smith et al. report")
    out, taken = _rule(f"The gain is {gain} (Fig. 2 and Fig. 3). Next.\n", numbers={"a.png": 2, "b.png": 3},
                       drawn={2: {"signal"}, 3: {"signal"}})
    assert out == "Next.\n" and taken[0]["sentence"] == "The gain is 0.5 (Fig. 2 and Fig. 3)."


def test_list_items_and_wrapped_lines_are_read_alone():
    md = "- Figure 1 shows a rise.\n"
    assert _rule(md) == (md, [])
    out, _ = _rule("First part of a sentence\nthat ends here (Figure 1).\n")
    assert out == "First part of a sentence\n"


def test_sentence_after_a_removed_pointer_sentence_survives():
    gain = _v("gain")
    for nxt in ("10 seeds were used.", "mm-scale runs follow.", "0.5 is the baseline.", "5% of runs failed."):
        out, taken = _rule(f"The gain is {gain} (Figure 1). {nxt}\n")
        assert out == f"{nxt}\n", nxt
        assert [t["sentence"] for t in taken] == ["The gain is 0.5 (Figure 1)."], nxt


def test_a_panel_with_a_hold_is_named_by_its_setting_in_a_claim():
    p0 = _panel([_series("a", [1, 2], [1.0, 2.0])], 1, hold={"m": 100})
    p1 = _panel([_series("a", [1, 2], [3.0, 4.0])], 2, hold={"m": 200})
    calls = _calls(p0, p1)
    got = ev(calls, "drawn", series="a", panel=2)
    assert got and got[0] == "A is drawn in the panel at m = 200"
    assert ev(_calls(_panel([_series("a", [1, 2], [1.0, 2.0])], 1), _panel([_series("a", [1, 2], [1.0, 2.0])], 2)),
              "drawn", series="a", panel=2)[0] == "A is drawn in panel 2"
    assert fc.panel_name(p0) == "the panel at m = 100" and fc.panel_name(_panel([], 3)) == "panel 3"
    assert fc.hold_text(_panel([], 1, hold={"m": 0.25, "k": "x"})) == "m = 0.25, k = x"


def test_claims_name_a_hold_by_the_plans_label_and_distinct_values_read_distinct():
    p0 = _panel([_series("a", [1, 2], [1.0, 2.0])], 1, hold={"m": 1.23441}, hold_label={"m": "Mass"})
    p1 = _panel([_series("a", [1, 2], [3.0, 4.0])], 2, hold={"m": 1.23449}, hold_label={"m": "Mass"})
    assert fc.hold_text(p0) == "Mass = 1.23441" and fc.hold_text(p1) == "Mass = 1.23449"
    assert ev(_calls(p0, p1), "drawn", series="a", panel=2)[0] == "A is drawn in the panel at Mass = 1.23449"


def test_fis_composed_sentence_is_never_removed_the_writers_words_in_it_are():
    nums = {"fig_a.png": 1}
    drawn = {1: {"signal"}}
    claim = _claim()
    for md, expect in (
        (f"Intro. Beyond doubt, {claim} (Figure 1). Next.\n", f"Intro. {claim} (Figure 1). Next.\n"),
        (f"Intro. {claim} (Figure 1) It also clearly flattens. Next.\n", f"Intro. {claim} (Figure 1). Next.\n"),
        (f"Intro. Clearly {claim} and it proves the point (Figure 1). Next.\n", f"Intro. {claim} (Figure 1). Next.\n"),
        (f"{claim} and the gain is large, see Figure 1 for that. Last.\n", f"{claim} (Figure 1). Last.\n"),
        # a sentence with FI's statement is judged although it names no figure
        (f"Intro. {claim} and this proves the method is best. Next.\n", f"Intro. {claim} (Figure 1). Next.\n"),
        # ... and a result mark that matches the figure does not keep the writer's words beside it
        (f"Intro. {claim} and the signal is {_v('signal')} (Figure 1). Next.\n", f"Intro. {claim} (Figure 1). Next.\n"),
        (f"Intro. The signal is {_v('signal')}, so {claim} (Figure 1). Next.\n", f"Intro. {claim} (Figure 1). Next.\n"),
    ):
        out, taken = fc.apply_pointer_rule(md, nums, drawn, _MENTION)
        assert out == expect, (md, out)
        assert len(taken) == 1
        assert "Panel 1 draws a rising line." in out and "proves" not in out and "flattens" not in out
        assert "large" not in out and "see Figure" not in out and "doubt" not in out
        assert taken[0]["sentence"] and "Panel 1" not in taken[0]["sentence"]
    # a statement without a full stop of its own (as FI's real ones are) gets one where the sentence after it goes
    bare = _claim(text="Panel 1 draws a rising line")
    out, _ = fc.apply_pointer_rule(f"{bare} (Figure 1). See Figure 1 for more. Next.\n", nums, drawn, _MENTION)
    assert out == f"{bare} (Figure 1). Next.\n"
    out, _ = fc.apply_pointer_rule(f"{bare}. See Figure 1 for more. Next.\n", nums, drawn, _MENTION)
    assert out == f"{bare}. Next.\n"
    # the writer's own sentence beside FI's is judged on its own: a result mark under a closing pointer stays
    md = f"{claim} (Figure 1). The signal is {_v('signal')} (Figure 1).\n"
    assert fc.apply_pointer_rule(md, nums, drawn, _MENTION) == (md, [])


def _many(n_series: int, n_points: int = 7):
    series = [_series(f"s{i}", list(range(n_points)), [float(i + j) for j in range(n_points)]) for i in range(n_series)]
    return _calls(_panel(series))


def test_a_menu_is_cut_to_forty_statements_in_the_order_of_what_matters():
    calls = _many(10)
    full = fc.menu(calls)
    assert len(full) > 40
    capped = fc.cap_menu(full, calls)
    assert len(capped) == 40 and capped == fc.cap_menu(full, calls), "deterministic"
    kinds = [o["claim"] for o in capped]
    assert kinds.count("drawn") == 10, "every line's drawn statement is kept"
    assert kinds.count("no_crossing") < sum(1 for o in full if o["claim"] == "no_crossing"), "no_crossing is cut first"
    assert [o["mark"] for o in capped] == [o["mark"] for o in full if o in capped], "the menu's own order"
    short = _many(2)
    assert fc.cap_menu(fc.menu(short), short) == fc.menu(short)


def test_the_pairs_of_lines_next_to_each_other_in_the_legend_are_kept_first_in_no_crossing():
    calls = _many(8, n_points=3)
    full = fc.menu(calls)
    others = len([o for o in full if o["claim"] != "no_crossing"])
    capped = fc.cap_menu(full, calls, limit=others + 5)
    pairs = [(o["args"]["a"], o["args"]["b"]) for o in capped if o["claim"] == "no_crossing"]
    assert len(pairs) == 5
    assert all(abs(int(b[1:]) - int(a[1:])) == 1 for a, b in pairs), pairs


def test_a_result_key_fi_could_not_fill_in_a_sentence_the_figure_pass_removes_stays_on_record():
    nums = {"fig_a.png": 1}
    md = f"Intro. {_claim()} and the gain was {_v('gain', 'not recorded')} overall. Figure 1 shows {_v('gain', 'not recorded')}.\n"
    out, taken = fc.apply_pointer_rule(md, nums, {1: {"signal"}}, _MENTION)
    assert "not recorded" not in out and "Panel 1 draws a rising line." in out
    listed = [t for t in taken if t.get("at_write")]
    assert [t["key"] for t in listed] == ["gain@size=1", "gain@size=1"] and all(t["how"] == "sentence removed" for t in listed)
    # a filled key is not a number lost
    out, taken = fc.apply_pointer_rule(f"Figure 1 shows {_v('gain')}.\n", nums, {1: {"signal"}}, _MENTION)
    assert not [t for t in taken if t.get("at_write")]


def test_a_removed_sentence_is_attributed_to_every_checked_figure_it_names():
    nums = {"fig_a.png": 1, "old.png": 2, "fig_c.png": 3}
    out, taken = fc.apply_pointer_rule("Figure 2 and Figure 1 show a rise. Next.\n", nums, {1: {"s"}, 3: {"s"}}, _MENTION)
    assert out == "Next.\n" and len(taken) == 1 and taken[0]["figures"] == [1, 2]
    out, taken = fc.apply_pointer_rule("Figure 3 and Figure 1 differ, unlike Figure 2.\n", nums, {1: {"s"}, 3: {"s"}}, _MENTION)
    assert taken[0]["figures"] == [1, 2, 3]


def test_a_bold_run_in_heading_naming_a_figure_before_prose_is_not_removed():
    text = "**Figure 1.**\n\nThe fit is close.\n"
    assert _rule(text)[0] == text  # the text stays (the entry it logs is a Task 16 matter, not shown here)


def test_a_fully_bold_fi_marked_statement_is_kept_as_it_was():
    text = "**" + _claim() + " (Figure 1).**\n"
    assert _rule(text)[0] == text


# --- a leading bold span: judged and cut as one span -------------------------------------------------------------------

def test_a_bold_false_sentence_closed_before_its_period_is_removed_whole():
    out, taken = _rule("Intro. **Figure 1 shows the energy drifting upward**. More words here.\n")
    assert out == "Intro. More words here.\n"
    assert [t["sentence"] for t in taken] == ["**Figure 1 shows the energy drifting upward**."]


def test_a_bold_false_sentence_closed_after_its_period_is_removed_and_the_next_one_stays():
    out, taken = _rule("Intro. **Figure 1 shows the energy drifting upward.** More words here.\n")
    assert out == "Intro. More words here.\n"
    assert [t["sentence"] for t in taken] == ["**Figure 1 shows the energy drifting upward.**"]


def test_the_plain_false_sentence_is_removed_as_before():
    out, _ = _rule("Intro. Figure 1 shows the energy drifting upward. More words here.\n")
    assert out == "Intro. More words here.\n"


def test_a_bare_figure_label_heading_stays_and_is_not_on_record():
    for head in ("**Figure 1.**", "**Fig. 2.**", "**Figures 1 and 2.**"):
        md = f"{head} The study ran at several sizes.\n"
        assert _rule(md, {"fig_a.png": 1, "fig_b.png": 2}, {1: {"signal"}, 2: {"signal"}}) == (md, []), head
    out, taken = _rule("**Figure 1.** Figure 1 shows a rise. Next.\n")
    assert out == "**Figure 1.** Next.\n" and [t["sentence"] for t in taken] == ["Figure 1 shows a rise."]


def test_a_run_in_heading_before_an_honest_sentence_with_a_pointer_stays():
    md = f"**Results.** The signal is {_v('signal')} at size 1 (Figure 1). Next.\n"
    assert _rule(md) == (md, [])
    out, _ = _rule("**Results.** Figure 1 shows a rise. Next.\n")
    assert out == "**Results.** Next.\n"


def test_fis_own_figure_claim_in_bold_with_its_pointer_stays():
    md = f"**{_claim()}** (Figure 1). Next.\n"
    assert _rule(md) == (md, [])


def test_an_image_embed_inside_a_line_is_never_judged_and_never_cut():
    emb = "![**Figure 1.** Bias against noise; Figure 1 shows a rise.](figures/x.png)"
    for md in (f"**Spread.** {emb}\n", f"Some prose here. {emb}\n", f"{emb} More prose.\n"):
        out, taken = _rule(md)
        assert emb in out and taken == [], md
    out, taken = _rule(f"Intro. {emb} Figure 1 shows a rise. Next.\n")
    assert emb in out and "shows a rise. Next" not in out and "Next." in out
    assert [t["sentence"] for t in taken] == ["Figure 1 shows a rise."]


def test_a_bold_label_is_a_heading_only_when_it_closes_with_a_full_stop_or_colon():
    for head in ("**Figure 1.**", "**Figure 1:**", "**Results.**", "**Figure 1**."):
        md = f"{head} The study ran at several sizes.\n"
        assert _rule(md) == (md, []), head
    out, taken = _rule("Intro. **Figure 1** shows a rise. Next.\n")
    assert out == "Intro. Next.\n" and [t["sentence"] for t in taken] == ["**Figure 1** shows a rise."]


def test_a_bold_span_holding_a_result_mark_is_judged_and_cut_with_its_sentence():
    wrong = f"Intro. **The signal is {_v('other')} here (Figure 1).** Next.\n"
    out, taken = _rule(wrong)
    assert out == "Intro. Next.\n" and len(taken) == 1
    out, taken = _rule(f"Intro. **The signal is {_v('other')} here** and Figure 1 shows it. Next.\n")
    assert out == "Intro. Next.\n"
    ok = f"Intro. **The signal is {_v('signal')} here (Figure 1).** Next.\n"
    assert _rule(ok) == (ok, [])


def test_what_the_record_lists_as_removed_is_gone_from_the_output():
    cases = [f"Intro. **The signal is {_v('other')} here (Figure 1).** Next.\n",
             "Intro. **Figure 1 shows the energy drifting upward**. More words here.\n",
             "Intro. **Figure 1 shows a rise.** More words here.\n",
             f"**Spread.** ![**Figure 1.** cap](f/x.png) Figure 1 shows a rise. Next.\n",
             f"**A.** {_v('other')} in Figure 1 and **bold** words. Next.\n"]
    for md in cases:
        out, taken = _rule(md)
        for t in taken:
            assert t["sentence"] not in fc._values.without_marks(out), (md, t)


def test_an_image_embed_with_parentheses_in_its_target_or_a_title_is_whole():
    for emb in ("![Figure 1 plot](figures/a_(b).png)", '![cap](figures/x.png "Figure 1 title")',
                "![a [Figure 1] b](figures/x.png)"):
        md = f"**Spread.** {emb} Figure 1 shows a rise. Next.\n"
        out, taken = _rule(md)
        assert emb in out and out.endswith("Next.\n") and len(taken) == 1, emb


def test_a_line_with_an_embed_that_cannot_be_parsed_is_left_alone():
    md = "Some words. ![cap](figures/x.png Figure 1 shows a rise. Next.\n"
    assert _rule(md) == (md, [])
