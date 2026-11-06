"""Final-review fixes of the drawing record and its claims: a curve's name and y are FI's (no body words or digits reach a checked
panel), a series FI found is not its result keeps neither the result's name nor an unchecked x, and error bars / bands called
"range" are kept only when they are the table's min and max. The studies are the invented fixtures of
``tests/fixtures/figure_claims/``."""

from __future__ import annotations

import json
import runpy

import pytest

from core import figure_claims
from tests.test_figure_fixtures import FIXTURES, _Host, _Table

HEAD = "def figure_spectra(fig, table):\n    waves = sorted({table.setting(k)['wavelength'] for k in table.keys()})\n"


def _draw(body, tmp_path):
    """Run ``body`` (the lines inside ``figure_spectra`` after ``waves``) on the spectra fixture; ``(calls, offers, host)``."""
    folder = FIXTURES / "spectra"
    contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
    cells = json.loads((folder / "trials.json").read_text(encoding="utf-8"))["cells"]
    path = tmp_path / "body.py"
    path.write_text(HEAD + "".join("    " + line + "\n" for line in body.strip().splitlines()), encoding="utf-8")
    host = _Host(contract, _Table(cells, contract["results"]))
    runpy.run_path(str(path))["figure_spectra"](host, host.table)
    host.write(tmp_path / ".fi" / "figure_records" / "spectra.png")
    calls = figure_claims.load_calls(tmp_path, "spectra")
    return calls, figure_claims.menu(calls, contract["plan"], stem="spectra"), host


HONEST = '''
keys = [table.at(wavelength=w, sample="a")[0] for w in waves]
means = [table.mean("signal", k) for k in keys]
halves = [(table.interval("signal", k)[1] - table.interval("signal", k)[0]) / 2 for k in keys]
'''
LINE = 'fig.line(0, "sample_a", waves, means, "signal")\n'
CURVE = 'fig.curve(0, "theory says error vanishes, r = 0.9991", waves, [50.0] * len(waves))\n'


def _texts(offers):
    return " | ".join(o["text"] for o in offers)


def _legend(host):
    return [t.get_text() for ax in host.axes for t in (ax.get_legend().get_texts() if ax.get_legend() else [])]


# ---- C2 --------------------------------------------------------------------------------------------------------------------

def test_a_curve_the_body_computes_is_not_drawn(tmp_path):
    with pytest.raises(ValueError):
        _draw(HONEST + LINE + CURVE, tmp_path)


# ---- I1 --------------------------------------------------------------------------------------------------------------------

INVERSE = HONEST + 'fig.line(0, "sample_a", [1 / w for w in waves], means, "signal")\n'


def test_a_series_drawn_against_a_transform_of_x_keeps_neither_name_nor_x_claims(tmp_path):
    calls, offers, host = _draw(INVERSE, tmp_path)
    s = calls["panels"][0]["series"][0]
    assert s["what"] == "other" and s["label"] == "computed value"
    assert calls["panels"][0]["x"]["label"] != "Wavelength (nm)"
    by = {o["claim"] for o in offers}
    assert not by & {"trend", "order", "crossing", "no_crossing", "steps", "value_at", "missing_x"}
    assert not [o for o in offers if o["claim"] == "range" and o["args"].get("axis") == "x"]
    assert "Sample A" not in _texts(offers) and "decreases" not in _texts(offers)


def test_the_x_axis_of_such_a_series_is_named_neutrally_in_the_figure(tmp_path):
    _c, _o, host = _draw(INVERSE, tmp_path)
    host.finish(host.labels, {})
    assert host.axes[0].get_xlabel() != "Wavelength (nm)"


def test_missing_x_is_not_offered_for_points_on_a_transformed_x(tmp_path):
    calls, offers, _h = _draw(HONEST + 'fig.points(0, "sample_a", [1 / w for w in waves], means, "signal")\n', tmp_path)
    assert not [o for o in offers if o["claim"] == "missing_x"]


def test_an_honest_line_on_the_settings_own_x_keeps_trend_and_range(tmp_path):
    calls, offers, _h = _draw(HONEST + LINE, tmp_path)
    by = {(o["claim"], o["args"].get("axis")) for o in offers}
    assert ("trend", None) in by and ("range", "x") in by and ("value_at", None) in by
    assert calls["panels"][0]["series"][0]["label"] == "Sample A"


def test_two_downgraded_series_get_numbered_neutral_names(tmp_path):
    body = HONEST + ('fig.line(0, "sample_a", waves, [m + 0.3 for m in means], "signal")\n'
                     'fig.line(0, "sample_b", waves, [m + 0.5 for m in means], "signal")\n')
    calls, _o, host = _draw(body, tmp_path)
    assert [s["label"] for s in calls["panels"][0]["series"]] == ["computed value 1", "computed value 2"]
    host.finish(host.labels, {})
    assert _legend(host) == ["computed value 1", "computed value 2"]


# ---- I6 --------------------------------------------------------------------------------------------------------------------

RANGE_LO = 'lows = [table.mean("signal", k) - min(table.values("signal", k)) for k in keys]\n'
RANGE_HI = 'highs = [max(table.values("signal", k)) - table.mean("signal", k) for k in keys]\n'


def test_error_bars_called_range_that_are_not_the_min_and_max_become_other(tmp_path):
    body = HONEST + 'fig.line(0, "sample_a", waves, means, "signal", yerr=[0.3] * len(waves), yerr_meaning="range")\n'
    calls, offers, _h = _draw(body, tmp_path)
    assert calls["panels"][0]["series"][0]["yerr_meaning"] == "other"
    assert not [o for o in offers if o["claim"] == "errorbars"]


def test_uneven_range_bars_that_are_not_the_table_print_no_number(tmp_path):
    body = HONEST + 'fig.line(0, "sample_a", waves, means, "signal", yerr=([0.1] * len(waves), [0.9] * len(waves)), yerr_meaning="range")\n'
    _c, offers, _h = _draw(body, tmp_path)
    assert not [o for o in offers if o["claim"] == "errorbars"]


def test_honest_range_bars_are_kept(tmp_path):
    body = HONEST + RANGE_LO + RANGE_HI + 'fig.line(0, "sample_a", waves, means, "signal", yerr=(lows, highs), yerr_meaning="range")\n'
    calls, offers, _h = _draw(body, tmp_path)
    assert calls["panels"][0]["series"][0]["yerr_meaning"] == "range"
    assert any(o["claim"] == "errorbars" and "range" in o["text"] for o in offers)


def test_a_band_called_range_is_checked_and_an_honest_one_kept(tmp_path):
    bad = HONEST + LINE + 'fig.band(0, "sample_a", waves, [m - 0.3 for m in means], [m + 0.3 for m in means], "signal", "range")\n'
    calls, offers, _h = _draw(bad, tmp_path)
    assert calls["panels"][0]["bands"][0]["meaning"] == "other" and not [o for o in offers if o["claim"] == "band"]
    good = (HONEST + LINE + 'fig.band(0, "sample_a", waves, [min(table.values("signal", k)) for k in keys], '
            '[max(table.values("signal", k)) for k in keys], "signal", "range")\n')
    calls, offers, _h = _draw(good, tmp_path)
    assert calls["panels"][0]["bands"][0]["meaning"] == "range" and any(o["claim"] == "band" for o in offers)


# ---- fix round 1 -------------------------------------------------------------------------------------------------------------

def test_a_series_declared_other_never_carries_a_results_plan_label(tmp_path):
    body = HONEST + 'fig.line(0, "signal", waves, [9, 7, 5, 3, 2, 1], "other")\n'
    calls, offers, host = _draw(body, tmp_path)
    host.finish(host.labels, {})
    assert calls["panels"][0]["series"][0]["label"] == "computed value"
    assert _legend(host) == ["computed value"]
    assert "Signal" not in _texts(offers)
    # the known positive: an arm id (not a result) declared `other` keeps its own name
    calls, offers, host = _draw(HONEST + 'fig.line(0, "sample_a", waves, [9, 7, 5, 3, 2, 1], "other")\n', tmp_path)
    assert calls["panels"][0]["series"][0]["label"] == "Sample A"


def test_a_merged_legend_entry_never_names_a_result_whose_values_fi_did_not_find(tmp_path):
    # the arm's series is drawn first and kept; the result-named one with the same values merges into it
    body = HONEST + ('fig.line(0, "sample_a", waves, [9, 7, 5, 3, 2, 1], "other")\n'
                     'fig.line(0, "signal", waves, [9, 7, 5, 3, 2, 1], "other")\n')
    calls, offers, host = _draw(body, tmp_path)
    host.finish(host.labels, {})
    legend = " | ".join(_legend(host))
    assert "Sample A" in legend and "Signal" not in legend, legend


def _hand(series_extra=None, **top):
    s = {"id": "a", "label": "A", "kind": "line", "step": False, "x": [1, 2, 3], "y": [1.0, 2.0, 3.0], "yerr": None,
         "merged_with": [], "what": "signal", **(series_extra or {})}
    panel = {"panel": 1, "raw": False, "raw_reason": None, "x": {"id": "size", "kind": "setting", "label": "size", "scale": "linear"},
             "y": {"what": "signal", "label": "signal", "scale": "linear"}, "series": [s], "bands": [], "refs": [], "notes": [],
             "texts_raw": []}
    return {"version": 2, "fallback": False, "panels": [panel], **top}


def test_a_series_record_without_the_x_flag_reads_as_unchecked():
    assert figure_claims.evaluate(_hand({"x_checked": True}), "trend", {"series": "a"}) is not None
    assert figure_claims.evaluate(_hand(), "trend", {"series": "a"}) is None
    assert figure_claims.evaluate(_hand(), "range", {"series": "a", "axis": "x"}) is None
    assert figure_claims.evaluate(_hand(), "drawn", {"series": "a"}) is not None


def test_a_curve_is_named_by_fi_whatever_the_record_stores():
    got = figure_claims.evaluate(_hand({"kind": "curve", "label": "theory says r = 0.9991 (x)", "id": "curve1"}), "drawn",
                                 {"series": "curve1"})
    assert got is not None and got[0].startswith("a curve is drawn") and "0.9991" not in got[0]


def test_the_record_is_version_two_and_an_older_one_is_not_read(tmp_path):
    calls, _o, _h = _draw(HONEST + LINE, tmp_path)
    assert calls["version"] == 2
    path = tmp_path / ".fi" / "figure_records" / "spectra.calls.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    for version in (1, None):
        if version is None:
            old.pop("version")
        else:
            old["version"] = version
        path.write_text(json.dumps(old), encoding="utf-8")
        assert figure_claims.load_calls(tmp_path, "spectra") is None
    old["version"] = 2
    path.write_text(json.dumps(old), encoding="utf-8")
    assert figure_claims.load_calls(tmp_path, "spectra") is not None


def test_a_gap_at_an_x_off_the_grid_keeps_the_x_claims(tmp_path):
    body = HONEST + ('xs = list(waves) + [999]\n'
                     'fig.line(0, "sample_a", xs, means + [float("nan")], "signal")\n')
    calls, offers, _h = _draw(body, tmp_path)
    assert calls["panels"][0]["series"][0]["x_checked"] is True
    assert any(o["claim"] == "trend" for o in offers)


def test_two_identical_downgraded_series_share_one_legend_entry_naming_both(tmp_path):
    body = HONEST + ('fig.line(0, "sample_a", waves, [m + 0.3 for m in means], "signal")\n'
                     'fig.line(0, "sample_b", waves, [m + 0.3 for m in means], "signal")\n')
    _c, _o, host = _draw(body, tmp_path)
    host.finish(host.labels, {})
    assert _legend(host) == ["computed value 1 = computed value 2"]
