"""The replay script (dev/evaluation/figure_replay.py): how it classes a text with a number, how it places a note, and that its
independent recomputation of a claim disagrees when a fault is planted in each kind of claim."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import figure_claims  # noqa: E402
from tests import test_figure_fixtures as fixtures  # noqa: E402

_spec = importlib.util.spec_from_file_location("figure_replay", Path(__file__).resolve().parents[1] / "dev" / "evaluation" / "figure_replay.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _clean_host(tmp_path):
    """A host on the spectra fixture's table whose axes are empty."""
    contract, drawn, _stem = fixtures._run("spectra", tmp_path)
    return contract, fixtures._Host(contract, drawn.table)


def _run_on_spectra(tmp_path, grid, draw, held=()):
    contract, host = _clean_host(tmp_path)
    spec = {"x": contract["x"], "series": None, "shows": [], "file": "f.png", "panels": 1}
    run = replay._Run(host, host.table, {}, grid, spec, {}, held=held)
    draw(replay._Panel(run, 0, host.axes[0]))
    run.sort_texts()
    return run


def test_a_number_equal_to_a_setting_of_another_axis_is_typed(tmp_path):
    # "1" is a value on another axis of the study, but not an x of this panel's series: the body typed it
    grid = {"wavelength": [400, 450, 500], "repeat": [1, 2, 3]}

    def draw(p):
        p.plot([400, 450, 500], [1.2, 2.8, 6.5], label="a")
        p.annotate("1", xy=(450, 3))

    run = _run_on_spectra(tmp_path, grid, draw)
    assert dict(run.text_classes) == {"number typed by the body": 1}
    assert run.causes[0][-1][0] == "text with a number the body typed"


def test_a_number_equal_to_an_x_of_the_panel_is_a_setting_value(tmp_path):
    grid = {"wavelength": [400, 450, 500], "repeat": [1, 2, 3]}

    def draw(p):
        p.plot([400, 450, 500], [1.2, 2.8, 6.5], label="a")
        p.annotate("450", xy=(450, 3))

    run = _run_on_spectra(tmp_path, grid, draw)
    assert dict(run.text_classes) == {"number is a setting value": 1}
    assert run.causes[0][-1][0] == "text with a setting value in it"


def test_a_value_held_for_the_figure_is_a_setting_value(tmp_path):
    def draw(p):
        p.plot([400, 450], [1.2, 2.8], label="a")
        p.text(430, 3, "2.5")

    run = _run_on_spectra(tmp_path, {"wavelength": [400]}, draw, held=[2.5])
    assert dict(run.text_classes) == {"number is a setting value": 1}


def test_a_text_before_the_data_is_still_classed_by_the_data(tmp_path):
    def draw(p):
        p.annotate("450", xy=(450, 3))
        p.plot([400, 450, 500], [1.2, 2.8, 6.5], label="a")

    run = _run_on_spectra(tmp_path, {"wavelength": [400, 450, 500]}, draw)
    assert dict(run.text_classes) == {"number is a setting value": 1}


def _note_run(tmp_path):
    contract, host = _clean_host(tmp_path)
    spec = {"x": contract["x"], "series": None, "shows": [], "file": "f.png", "panels": 1}
    run = replay._Run(host, host.table, {}, {"wavelength": [400, 450, 500]}, spec, {})
    key = host.table.keys()[0]
    value = host.table.mean("signal", key)
    run.reads.append((float(value), "signal", key, "mean"))
    return host, run, "peak %.2f" % value


def test_a_note_in_axes_coordinates_is_raw_not_a_note_at_a_wrong_place(tmp_path):
    host, run, words = _note_run(tmp_path)
    ax = host.axes[0]
    replay._Panel(run, 0, ax).text(0.9, 0.9, words, transform=ax.transAxes)
    assert run.causes[0][0][0] == "a note placed in axes coordinates"
    assert host._state()[0]["notes"] == []


def test_the_same_note_in_data_coordinates_is_a_note_at_that_place(tmp_path):
    host, run, words = _note_run(tmp_path)
    replay._Panel(run, 0, host.axes[0]).text(450, 3, words)
    assert not run.causes
    assert host._state()[0]["notes"][0]["x"] == 450.0


# ---- the independent recomputation fires on a planted fault in each kind of claim ----------------------------------------

def _first(offers, claim, **args):
    return next(o for o in offers if o["claim"] == claim and all(o["args"].get(k) == v for k, v in args.items()))


def _panel(calls, n):
    return next(p for p in calls["panels"] if p["panel"] == n)


def _series(calls, n, sid):
    return next(s for s in _panel(calls, n)["series"] if s["id"] == sid)


def _built(name, tmp_path):
    contract, host, stem = fixtures._run(name, tmp_path)
    calls = figure_claims.load_calls(tmp_path, stem)
    offers = figure_claims.menu(calls, contract["plan"], stem=stem)

    def scale_of(n, axis):
        return host.axes[n - 1].get_xscale() if axis == "x" else host.axes[n - 1].get_yscale()

    return calls, offers, scale_of


def _number(o, calls):
    """The sentence states a number the record does not hold."""
    o["text"] = replay.NUMBER.sub(lambda m: "%.4g" % (float(m.group()) * 3 + 7), o["text"], count=1)


def _words(old, new):
    def go(o, calls):
        assert old in o["text"], o["text"]
        o["text"] = o["text"].replace(old, new)
    return go


def _record(change):
    def go(o, calls):
        change(calls)
    return go


def _a_series_is_drawn(calls):
    s = copy.deepcopy(_series(calls, 1, "sample_a"))
    s["id"] = "sample_c"
    _panel(calls, 1)["series"].append(s)


def _sample_b_above(calls):
    s = _series(calls, 1, "sample_b")
    s["y"] = [v + 100 for v in s["y"]]


def _rows_cross(calls):
    low, high = _series(calls, 2, "row_0"), _series(calls, 2, "row_2")
    low["y"][-2:], high["y"][-2:] = list(high["y"][-2:]), list(low["y"][-2:])  # the order flips at the last two columns


def _negative_bars(calls):
    s = _series(calls, 1, "sample_a")
    s["yerr"] = [-0.1] * len(s["y"])


def _band_upside_down(calls):
    b = _panel(calls, 1)["bands"][0]
    b["lo"], b["hi"] = b["hi"], b["lo"]


def _not_a_step(calls):
    _series(calls, 1, "group_a")["step"] = False


# one fault in each claim kind the fixtures offer: the sentence or the record is changed so that the sentence is false
FAULTS = [
    ("spectra", "drawn", {"series": "sample_a"}, _words("Sample A", "Sample Z")),
    ("spectra", "not_drawn", {"series": "sample_c"}, _record(_a_series_is_drawn)),
    ("spectra", "range", {"series": "sample_a", "axis": "y"}, _number),
    ("spectra", "value_at", {"series": "sample_a", "x": "450"}, _number),
    ("spectra", "trend", {"series": "sample_a"}, _words("does not change monotonically", "increases")),
    ("spectra", "order", {"at": "400"}, _words("Sample A, Sample B", "Sample B, Sample A")),
    ("spectra", "crossing", {"a": "sample_a", "b": "sample_b"}, _record(_sample_b_above)),
    ("spectra", "reference", {"ref": "baseline"}, _number),
    ("spectra", "errorbars", {"series": "sample_a"}, _record(_negative_bars)),
    ("spectra", "band", {"series": "sample_b"}, _record(_band_upside_down)),
    ("spectra", "scale", {"axis": "x", "panel": "1"}, _words("is linear", "is logarithmic")),
    ("spectra", "note", {"note": "signal"}, _words("6.5", "6.4")),
    ("map", "no_crossing", {"a": "row_0", "b": "row_2"}, _record(_rows_cross)),
    ("survival", "steps", {"series": "group_a"}, _number),
    ("survival", "steps", {"series": "group_a"}, _record(_not_a_step)),
]


@pytest.mark.parametrize("name,claim,args,plant", FAULTS, ids=[f"{f[0]}-{f[1]}-{i}" for i, f in enumerate(FAULTS)])
def test_the_recompute_agrees_with_the_truth_and_disagrees_with_a_planted_fault(tmp_path, name, claim, args, plant):
    calls, offers, scale_of = _built(name, tmp_path)
    offer = _first(offers, claim, **args)
    assert replay._agree(calls, offer, [], scale_of)[0] == "agree", offer
    bad_calls, bad_offer = copy.deepcopy(calls), copy.deepcopy(offer)
    plant(bad_offer, bad_calls)
    assert replay._agree(bad_calls, bad_offer, [], scale_of)[0] == "disagree", (claim, bad_offer["text"])


def test_every_claim_kind_the_fixtures_offer_has_a_planted_fault(tmp_path):
    offered = set()
    for name in ("spectra", "map", "survival", "bars_heatmap"):
        offered |= {o["claim"] for o in _built(name, tmp_path / name)[1]}
    assert offered <= {f[1] for f in FAULTS}, offered - {f[1] for f in FAULTS}


def test_identical_and_missing_x_disagree_when_planted(tmp_path):
    calls, _offers, scale_of = _built("spectra", tmp_path)
    twin = copy.deepcopy(_series(calls, 1, "sample_a"))
    twin["id"] = "twin"
    _panel(calls, 1)["series"].append(twin)
    both = figure_claims.menu(calls, {"series": ["sample_a", "twin"], "grid": []}, stem="s")
    same = next(o for o in both if o["claim"] == "identical")
    assert replay._agree(calls, same, [], scale_of)[0] == "agree"
    _series(calls, 1, "twin")["y"][0] += 5
    assert replay._agree(calls, same, [], scale_of)[0] == "disagree"
    # missing_x: a points series that lacks a grid x; the fault draws that x after all
    pts = _series(calls, 1, "sample_a")
    gone = pts["x"][-1]
    pts["kind"], pts["x"], pts["y"] = "points", pts["x"][:-1], pts["y"][:-1]
    offer = {"claim": "missing_x", "args": {"series": "sample_a", "panel": "1"}, "text": "Sample A has no point at %g" % gone}
    assert replay._agree(calls, offer, [gone], scale_of)[0] == "agree"
    pts["x"].append(gone)
    pts["y"].append(1.0)
    assert replay._agree(calls, offer, [gone], scale_of)[0] == "disagree"


def test_a_label_that_opens_another_label_is_found_as_a_whole_item_in_the_order_sentence():
    """The order sentence lists labels such as "S#29, S#3, S#2": "S#2" is the start of "S#29", so a plain search for it finds the
    wrong place. The recomputation finds each label after the one before and agrees with the true sentence (a replayed figure with
    34 lines named computed#1 ... computed#33 was reported as 3 disagreements by the plain search)."""
    def line(sid, y):
        return {"id": sid, "label": sid, "kind": "line", "step": False, "x": [1.0, 2.0], "y": [y, y], "yerr": None, "merged_with": [], "x_checked": True}

    panel = {"panel": 1, "raw": False, "raw_reason": None, "x": {"id": "size", "kind": "setting", "label": "size", "scale": "linear"},
             "y": {"what": "signal", "label": "signal", "scale": "linear"}, "bands": [], "refs": [], "notes": [], "texts_raw": [],
             "series": [line("S#29", 3.0), line("S#3", 2.0), line("S#2", 1.0)]}
    calls = {"version": 2, "fallback": False, "panels": [panel]}
    order = next(o for o in figure_claims.menu(calls, None, stem="s") if o["claim"] == "order")
    assert "S#29, S#3, S#2" in order["text"]
    scale_of = lambda n, axis: "linear"  # noqa: E731
    assert replay._agree(calls, order, [], scale_of)[0] == "agree"
    order["text"] = order["text"].replace("S#29, S#3, S#2", "S#3, S#29, S#2")  # the planted fault: two labels swapped
    assert replay._agree(calls, order, [], scale_of)[0] == "disagree"
