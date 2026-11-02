"""The drawn y of a series is compared with the results table: a line that does not match keeps its shape, but FI states no
value from it (its ``what`` becomes ``other``). The faults are planted in copies of the invented fixture bodies in
``tests/fixtures/figure_claims/``."""

from __future__ import annotations

import json
import runpy

import pytest

from core import figure_claims
from tests.test_figure_fixtures import FIXTURES, _Host, _Table

VALUE_CLAIMS = {"value_at", "steps"}  # claims that state a y number
SHAPE_CLAIMS = {"trend", "order", "crossing", "no_crossing"}  # claims about the shape: no y number in their text


def _draw(name, tmp_path, old=None, new=None):
    """Run the fixture's body (with ``old`` replaced by ``new``); return ``(calls, offers)``."""
    folder = FIXTURES / name
    contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
    cells = json.loads((folder / "trials.json").read_text(encoding="utf-8"))["cells"]
    src = (folder / "body.py").read_text(encoding="utf-8")
    if old is not None:
        assert old in src, old
        src = src.replace(old, new)
    path = tmp_path / "body.py"
    path.write_text(src, encoding="utf-8")
    host = _Host(contract, _Table(cells, contract["results"]))
    runpy.run_path(str(path))[contract["figure"]["body"]](host, host.table)
    stem = contract["figure"]["id"]
    host.write(tmp_path / ".fi" / "figure_records" / f"{stem}.png")
    calls = figure_claims.load_calls(tmp_path, stem)
    return calls, figure_claims.menu(calls, contract["plan"], stem=stem)


def _states_values(offers, panel="1"):
    """The offered claims of a panel that state a value of a drawn y (the range of y included)."""
    return [o for o in offers if o["args"].get("panel") == panel
            and (o["claim"] in VALUE_CLAIMS or (o["claim"] == "range" and o["args"].get("axis") == "y"))]


def _whats(calls, panel_index=0):
    return {s["id"]: s.get("what") for s in calls["panels"][panel_index]["series"]}


MEANS_SIGNAL = 'means = [table.mean("signal", k) for k in keys]'
MEANS_SURVIVAL = 'means = [table.mean("survival", k) for k in keys]'
MEANS_GROWTH = 'means = [table.mean("growth", k) for k in keys]'


def test_the_unchanged_body_keeps_its_value_claims(tmp_path):
    calls, offers = _draw("spectra", tmp_path)
    assert set(_whats(calls).values()) == {"signal"}
    assert {"value_at", "order", "trend", "crossing"} <= {o["claim"] for o in offers}


@pytest.mark.parametrize("name, old, new", [
    ("spectra", MEANS_SIGNAL, 'means = [table.mean("signal", k) + 0.3 for k in keys]'),  # a shift
    ("survival", MEANS_SURVIVAL, MEANS_SURVIVAL + "\n        means[1], means[2] = means[2], means[1]"),  # two settings swapped
    ("survival", MEANS_SURVIVAL, 'means = [0.97, 0.91, 0.84, 0.70, 0.52, 0.31, 0.20, 0.11, 0.05, 0.02][:len(keys)]'),  # typed
    ("spectra", MEANS_SIGNAL, 'means = [table.sd("signal", k) for k in keys]'),  # sd drawn as the result
    ("bars_heatmap", MEANS_GROWTH, 'means = [100 * table.mean("growth", k) for k in keys]'),  # a unit mistake
], ids=["shift", "swap", "typed", "wrong_reducer", "unit"])
def test_a_planted_fault_loses_its_value_claims_but_not_its_shape(tmp_path, name, old, new):
    calls, offers = _draw(name, tmp_path, old, new)
    assert set(_whats(calls).values()) == {"other"}
    assert _states_values(offers) == []  # no value, step value or y range from the wrong y
    assert any(o["claim"] == "drawn" for o in offers)  # the series is still drawn and its x range stated
    assert any(o["claim"] == "range" and o["args"].get("axis") == "x" for o in offers) or name == "bars_heatmap"
    panel = calls["panels"][0]
    # the axis names the quantity neutrally (a band of the real result may add its own name to it, the series stays flagged)
    assert panel["raw"] is False and panel["y"]["label"].startswith("computed value")


def test_a_shifted_line_keeps_its_shape_claims_without_a_y_number(tmp_path):
    calls, offers = _draw("spectra", tmp_path, MEANS_SIGNAL, 'means = [table.mean("signal", k) + 0.3 for k in keys]')
    kinds = {o["claim"] for o in offers}
    assert {"trend", "order", "crossing"} <= kinds and not kinds & VALUE_CLAIMS
    ys = {"%.3g" % y for e in calls["panels"][0]["series"] for y in e["y"] if y is not None}
    for o in offers:
        if o["claim"] in SHAPE_CLAIMS:
            assert not any(y in o["text"] for y in ys if "." in y), o["text"]


def test_a_real_table_value_of_another_slice_is_a_known_gap(tmp_path):
    # No option catches these (a y that is a real value of another setting at the same x): the values still pass the check.
    calls, offers = _draw("bars_heatmap", tmp_path,
                          'keys = [table.at(method=m, arm=arm, dose=2)[0] for m in methods]',
                          'keys = [table.at(method=m, arm=arm, dose=(3 if arm == "treated" else 2))[0] for m in methods]')
    assert set(_whats(calls).values()) == {"growth"}
    assert _states_values(offers)
    calls, offers = _draw("map", tmp_path, 'table.at(col=c, row=r)[0]) for c in cols]',
                          'table.at(col=c, row=(1 if r == 2 else r))[0]) for c in cols]')
    assert set(_whats(calls, 1).values()) == {"load"}
    assert _states_values(offers, "2")


def test_an_honestly_drawn_sd_with_its_form_keeps_its_value_claims(tmp_path):
    sd = MEANS_SIGNAL.replace("table.mean", "table.sd")
    old = 'fig.line(0, "sample_" + sample, waves, means, "signal", yerr=halves, yerr_meaning="ci95")'
    new = 'fig.line(0, "sample_" + sample, waves, means, "signal", yerr=halves, yerr_meaning="ci95", form="sd")'
    both = tmp_path / "g"
    both.mkdir()
    contract = json.loads((FIXTURES / "spectra" / "contract.json").read_text(encoding="utf-8"))
    cells = json.loads((FIXTURES / "spectra" / "trials.json").read_text(encoding="utf-8"))["cells"]
    text = (FIXTURES / "spectra" / "body.py").read_text(encoding="utf-8").replace(MEANS_SIGNAL, sd).replace(old, new)
    assert new in text
    (both / "body.py").write_text(text, encoding="utf-8")
    host = _Host(contract, _Table(cells, contract["results"]))
    runpy.run_path(str(both / "body.py"))["figure_spectra"](host, host.table)
    host.write(both / ".fi" / "figure_records" / "spectra.png")
    calls = figure_claims.load_calls(both, contract["figure"]["id"])
    offers = figure_claims.menu(calls, contract["plan"], stem=contract["figure"]["id"])
    assert set(_whats(calls).values()) == {"signal"}
    assert {"value_at", "order", "trend"} <= {o["claim"] for o in offers}


def test_the_same_sd_without_a_form_loses_its_value_claims_and_does_not_stop(tmp_path):
    calls, offers = _draw("spectra", tmp_path, MEANS_SIGNAL, MEANS_SIGNAL.replace("table.mean", "table.sd"))
    assert set(_whats(calls).values()) == {"other"}
    assert _states_values(offers) == []
    assert calls["panels"][0]["raw"] is False


def test_a_flagged_series_stays_flagged_when_an_honest_one_is_drawn_after_it(tmp_path):
    old = 'fig.line(0, "sample_" + sample, waves, means, "signal", yerr=halves, yerr_meaning="ci95")'
    new = ('means = [m + 0.3 for m in means] if sample == "a" else means\n'
           '        fig.line(0, "sample_" + sample, waves, means, "signal")')
    calls, offers = _draw("spectra", tmp_path, old, new)
    assert _whats(calls) == {"sample_a": "other", "sample_b": "signal"}
    assert calls["panels"][0]["y"]["what"] == "other / signal"  # the panel's axis names both; the claims follow each series
    by = {(o["claim"], o["args"].get("series")) for o in offers}
    assert ("value_at", "sample_b") in by and ("trend", "sample_b") in by
    assert not [o for o in offers if o["args"].get("series") == "sample_a" and o["claim"] in VALUE_CLAIMS]
    # the shape claims stay, and name no y number
    assert ("trend", "sample_a") in by and ("trend", "sample_b") in by
    assert [o for o in offers if o["claim"] in ("order", "crossing")]
    ys = {"%.3g" % y for e in calls["panels"][0]["series"] for y in e["y"] if y is not None}
    for o in offers:  # a shape claim quotes no y value: its text holds only the x positions and names
        if o["claim"] in SHAPE_CLAIMS:
            assert not any(("%s" % y) in o["text"] for y in ys if "." in y), o["text"]
    assert not [o for o in offers if o["claim"] == "range" and o["args"].get("axis") == "y" and o["args"].get("series") == "sample_a"]


# ---- fix round 2: what FI cannot compare is flagged, never kept ---------------------------------------------------------

LINE_CALL = 'fig.line(0, "sample_" + sample, waves, means, "signal", yerr=halves, yerr_meaning="ci95")'
UNEVEN = ('fig.line(0, "sample_" + sample, waves, means, "signal", yerr=([h / 2 for h in halves], [h * 2 for h in halves]), '
          'yerr_meaning="range")')

HONEST_RANGE = ('fig.line(0, "sample_" + sample, waves, means, "signal", yerr=([m - min(table.values("signal", k)) for m, k in zip(means, keys)], '
                '[max(table.values("signal", k)) - m for m, k in zip(means, keys)]), yerr_meaning="range")')


def _uneven_bar_claims(offers):
    return [o for o in offers if o["claim"] == "errorbars" and "reaching" in o["text"]]


def test_uneven_error_bars_of_a_flagged_series_state_no_number(tmp_path):
    (tmp_path / "ok").mkdir()
    (tmp_path / "bad").mkdir()
    (tmp_path / "typed").mkdir()
    # control: the honest line with the table's own range as bars keeps its claim (their sides are even here: no reach is quoted)
    calls, offers = _draw("spectra", tmp_path / "ok", LINE_CALL, HONEST_RANGE)
    assert set(_whats(calls).values()) == {"signal"} and len([o for o in offers if o["claim"] == "errorbars" and "range" in o["text"]]) == 2
    # uneven typed widths that are not the table's are not its range: no claim, no number
    calls, offers = _draw("spectra", tmp_path / "typed", LINE_CALL, UNEVEN)
    assert [o for o in offers if o["claim"] == "errorbars"] == []
    # a shifted line: no claim with a bar width, but the call is not refused and the panel is not raw
    calls, offers = _draw("spectra", tmp_path / "bad", LINE_CALL, UNEVEN.replace("waves, means,", "waves, [m + 0.3 for m in means],"))
    assert set(_whats(calls).values()) == {"other"} and calls["panels"][0]["raw"] is False
    assert _uneven_bar_claims(offers) == []
    assert any(o["claim"] == "trend" for o in offers)  # the shape stays


def test_a_series_whose_x_is_a_result_is_flagged(tmp_path):
    folder = FIXTURES / "spectra"
    contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
    cells = json.loads((folder / "trials.json").read_text(encoding="utf-8"))["cells"]
    host = _Host(contract, _Table(cells, contract["results"]))
    means = [host.table.mean("signal", k) for k in host.table.at(sample="a")]
    host.line(0, "sample_a", ("signal", means), means, "signal")
    host.line(0, "sample_b", ("signal", means), means, "signal")
    host.write(tmp_path / ".fi" / "figure_records" / "spectra.png")
    calls = figure_claims.load_calls(tmp_path, "spectra")
    assert set(_whats(calls).values()) == {"other"} and calls["panels"][0]["raw"] is False
    assert _states_values(figure_claims.menu(calls, contract["plan"], stem="spectra")) == []


def test_a_derived_quantity_and_a_missing_table_are_flagged(tmp_path):
    folder = FIXTURES / "spectra"
    contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
    cells = json.loads((folder / "trials.json").read_text(encoding="utf-8"))["cells"]
    for derived, table in (("ratio", True), ("signal", False)):
        host = _Host(contract, _Table(cells, contract["results"]))
        host.derived_ids = ["ratio"]
        waves = [500, 600]
        if not table:
            host.table = None  # nothing to compare with
            host.derived_ids = ["signal"]
        host.line(0, "sample_a", waves, [1.0, 2.0], derived)
        host.line(0, "sample_b", waves, [2.0, 1.0], derived)
        host.write(tmp_path / derived / ".fi" / "figure_records" / "spectra.png")
        calls = figure_claims.load_calls(tmp_path / derived, "spectra")
        assert set(_whats(calls).values()) == {"other"}, derived
        assert calls["panels"][0]["raw"] is False
        offers = figure_claims.menu(calls, contract["plan"], stem="spectra")
        assert _states_values(offers) == []


def test_a_series_with_no_finite_y_to_check_is_flagged(tmp_path):
    calls, offers = _draw("spectra", tmp_path, 'fig.line(0, "sample_" + sample, waves, means, "signal", yerr=halves, yerr_meaning="ci95")',
                          'fig.line(0, "sample_" + sample, waves, [None] * len(waves), "signal")')
    assert set(_whats(calls).values()) == {"other"}
