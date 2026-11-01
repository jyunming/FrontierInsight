"""Four invented studies a figure statement is tested on (spectra, a map, a survival curve, bars with a heat map).

Each folder in ``tests/fixtures/figure_claims/`` holds a made-up contract (plan ids, settings, references, labels), a trial
record, and a hand-written ``figure_<id>`` body that draws only through the drawing calls (core/frame_figure.py). The host is a
minimal stand-in on real matplotlib axes (Agg), the way tests/test_frame_figure.py builds one."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core import figure_claims, frame_figure  # noqa: E402

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


FIXTURES = Path(__file__).parent / "fixtures" / "figure_claims"


class _Table:
    """The read-only table of a fixture's trial record (mean, sd, n and a 95% interval of the mean per setting)."""

    def __init__(self, cells, declared):
        self.declared = list(declared)
        self._cells = {c["key"]: c for c in cells}

    def keys(self):
        return list(self._cells)

    def setting(self, key):
        return dict(self._cells[key]["cell"])

    def at(self, **settings):
        return [k for k, c in self._cells.items() if all(c["cell"].get(a) == v for a, v in settings.items())]

    def _values(self, name, key):
        assert name in self.declared, name
        return self._cells[key]["metrics"][name]["values"]

    def values(self, name, key):
        return list(self._values(name, key))

    def n(self, name, key):
        return len(self._values(name, key))

    def mean(self, name, key):
        v = self._values(name, key)
        return sum(v) / len(v)

    def sd(self, name, key):
        v, m = self._values(name, key), self.mean(name, key)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

    def interval(self, name, key):
        half = 2.262 * self.sd(name, key) / math.sqrt(self.n(name, key))  # t, nine degrees of freedom
        return self.mean(name, key) - half, self.mean(name, key) + half


class _Host(frame_figure.FigureCalls):
    def __init__(self, contract, table):
        self.fig, axes = plt.subplots(1, contract["figure"]["panels"], squeeze=False)
        self.axes = list(axes[0])
        self.table = table
        self.x = contract["x"]
        self.settings = contract["settings"]
        self.series_ids = contract["series_ids"]
        self.references = contract["references"]
        self.labels = contract["labels"]


class _Shifted:
    """The table a faulty body reads: every mean is ``by`` too high. FI's own record (``host.table``) is the true one."""

    def __init__(self, table, by):
        self._table, self._by = table, by

    def __getattr__(self, name):
        return getattr(self._table, name)

    def mean(self, name, key):
        return self._table.mean(name, key) + self._by


def _run(name, tmp_path, *, shift=0.0):
    """The fixture's body run on its trial record; returns ``(contract, host, stem)``. The call record is written.
    ``shift``: the body reads means that many too high (a drawn value that is not the table's)."""
    folder = FIXTURES / name
    contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
    cells = json.loads((folder / "trials.json").read_text(encoding="utf-8"))["cells"]
    host = _Host(contract, _Table(cells, contract["results"]))
    body = runpy.run_path(str(folder / "body.py"))[contract["figure"]["body"]]
    body(host, _Shifted(host.table, shift) if shift else host.table)
    stem = contract["figure"]["id"]
    host.write(tmp_path / ".fi" / "figure_records" / f"{stem}.png")
    return contract, host, stem


def _check_draws(name, tmp_path):
    contract, host, stem = _run(name, tmp_path)
    calls = figure_claims.load_calls(tmp_path, stem)
    assert calls is not None and calls["fallback"] is False
    assert len(calls["panels"]) == contract["figure"]["panels"]
    for panel in calls["panels"]:
        assert panel["raw"] is False and panel["series"], panel["panel"]
    drawn = [a for ax in host.axes for a in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.images)]
    assert drawn and all(str(a.get_gid()).startswith("fi:") for a in drawn)  # every data artist came from a call


def _check_claims_offered(name, tmp_path):
    contract, _host_, stem = _run(name, tmp_path)
    calls = figure_claims.load_calls(tmp_path, stem)
    plan = contract["plan"]
    offers = figure_claims.menu(calls, plan, stem=stem)
    assert offers
    for want in contract["expected"]:
        hits = [o for o in offers if o["claim"] == want["claim"]
                and all(o["args"].get(k) == v for k, v in want["args"].items())]
        assert hits, f"the menu does not offer {want}"
    for o in offers:
        again = figure_claims.evaluate(calls, o["claim"], o["args"], plan=plan)
        assert again is not None and again[0] == o["text"], o


def test_spectra_draws(tmp_path):
    _check_draws("spectra", tmp_path)


def test_spectra_claims_offered(tmp_path):
    _check_claims_offered("spectra", tmp_path)


def test_map_draws(tmp_path):
    _check_draws("map", tmp_path)


def test_map_claims_offered(tmp_path):
    _check_claims_offered("map", tmp_path)


def test_survival_draws(tmp_path):
    _check_draws("survival", tmp_path)
    calls = figure_claims.load_calls(tmp_path, "survival")
    assert all(s["step"] for s in calls["panels"][0]["series"])  # a step curve is recorded as one


def test_survival_claims_offered(tmp_path):
    _check_claims_offered("survival", tmp_path)


def test_bars_heatmap_draws(tmp_path):
    _check_draws("bars_heatmap", tmp_path)


def test_bars_heatmap_claims_offered(tmp_path):
    _check_claims_offered("bars_heatmap", tmp_path)


# ---- a heat map's range and a band's meaning are compared with FI's table ---------------------------------------------------


def _image_range(name, tmp_path, shift):
    contract, _h, stem = _run(name, tmp_path, shift=shift)
    offers = figure_claims.menu(figure_claims.load_calls(tmp_path, stem), contract["plan"], stem=stem)
    return [o["text"] for o in offers if o["claim"] == "range" and "has values" in o["text"]]


@pytest.mark.parametrize("name", ["map", "bars_heatmap"])
def test_a_heat_map_of_values_the_table_does_not_hold_states_no_range(name, tmp_path):
    assert _image_range(name, tmp_path / "honest", 0.0), "the honest heat map states its range"
    assert not _image_range(name, tmp_path / "shifted", 0.5)


def _survival_band(tmp_path, widen):
    folder = FIXTURES / "survival"
    contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
    cells = json.loads((folder / "trials.json").read_text(encoding="utf-8"))["cells"]
    host = _Host(contract, _Table(cells, contract["results"]))
    times = sorted({host.table.setting(k)["time"] for k in host.table.keys()})
    keys = [host.table.at(time=t, group="b")[0] for t in times]
    means = [host.table.mean("survival", k) for k in keys]
    host.line(0, "group_b", times, means, "survival", step=True)
    if widen:
        lo, hi = [m - 0.3 for m in means], [m + 0.3 for m in means]  # not the table's interval
    else:
        lo, hi = [host.table.interval("survival", k)[0] for k in keys], [host.table.interval("survival", k)[1] for k in keys]
    host.band(0, "group_b", times, lo, hi, "survival", "ci95")
    host.write(tmp_path / ".fi" / "figure_records" / "survival.png")
    return figure_claims.load_calls(tmp_path, "survival")


def test_a_band_that_is_not_the_tables_interval_states_no_meaning(tmp_path):
    calls = _survival_band(tmp_path, widen=True)
    assert figure_claims.evaluate(calls, "band", {"series": "group_b", "panel": "1"}) is None


def test_a_band_that_is_the_tables_interval_keeps_its_meaning(tmp_path):
    calls = _survival_band(tmp_path, widen=False)
    got = figure_claims.evaluate(calls, "band", {"series": "group_b", "panel": "1"})
    assert got is not None and "95% confidence interval" in got[0]
