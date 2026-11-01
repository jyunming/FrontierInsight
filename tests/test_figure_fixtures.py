"""Four invented studies a figure statement is tested on (spectra, a map, a survival curve, bars with a heat map).

Each folder in ``tests/fixtures/figure_claims/`` holds a made-up contract (plan ids, settings, references, labels), a trial
record, and a hand-written ``figure_<id>`` body that draws only through the drawing calls (core/frame_figure.py). The host is a
minimal stand-in on real matplotlib axes (Agg), the way tests/test_frame_figure.py builds one."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core import figure_claims, frame_figure  # noqa: E402

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


def _run(name, tmp_path):
    """The fixture's body run on its trial record; returns ``(contract, host, stem)``. The call record is written."""
    folder = FIXTURES / name
    contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
    cells = json.loads((folder / "trials.json").read_text(encoding="utf-8"))["cells"]
    host = _Host(contract, _Table(cells, contract["results"]))
    body = runpy.run_path(str(folder / "body.py"))[contract["figure"]["body"]]
    body(host, host.table)
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
