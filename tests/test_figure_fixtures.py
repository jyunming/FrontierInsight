"""Four invented studies a figure statement is tested on (spectra, a map, a survival curve, bars with a heat map).

Each folder in ``tests/fixtures/figure_claims/`` holds a made-up contract (plan ids, settings, references, labels), a trial
record, and a hand-written ``figure_<id>`` body that draws only through the drawing calls (core/frame_figure.py). The host is a
minimal stand-in on real matplotlib axes (Agg), the way tests/test_frame_figure.py builds one."""

from __future__ import annotations

import asyncio
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


def _run(name, tmp_path, *, shift=0.0, raw=None):
    """The fixture's body run on its trial record; returns ``(contract, host, stem)``. The call record is written.
    ``shift``: the body reads means that many too high (a drawn value that is not the table's). ``raw``: the 0-based panel
    that is then handed to the body's own drawing (``fig.raw``)."""
    folder = FIXTURES / name
    contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
    cells = json.loads((folder / "trials.json").read_text(encoding="utf-8"))["cells"]
    host = _Host(contract, _Table(cells, contract["results"]))
    body = runpy.run_path(str(folder / "body.py"))[contract["figure"]["body"]]
    body(host, _Shifted(host.table, shift) if shift else host.table)
    if raw is not None:
        host.raw(raw)
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


# ---- statements FI cannot stand behind fail closed, through the write node ---------------------------------------------------

DISCLOSURE = "some statements about this figure could not be checked"
BAD_ID = "glows"  # a claim FI does not make


def _honest(name, tmp_path):
    """``(contract, stem, offers)`` of the fixture drawn as written."""
    contract, _host_, stem = _run(name, tmp_path / "honest")
    calls = figure_claims.load_calls(tmp_path / "honest", stem)
    return contract, stem, figure_claims.menu(calls, contract["plan"], stem=stem)


def _write_paper(tmp_path, monkeypatch, calls, stem, marks):
    """The paper the write node leaves when the writer's draft states ``marks`` (one paragraph each) about the figure whose call
    record is ``calls``; returns ``(paper, kept record of what was taken out)``."""
    from core import paper_facts
    from tests.test_disclosure import _engine as make_engine

    draft = ("# T\n\n## Abstract\nWe compare things.\n\n## Results\n\n"
             + "\n\n".join(f"**Point {i}.** {mark} (Figure 1)." for i, mark in enumerate(marks, 1)) + "\n\n## Discussion\nIt is small.\n")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return draft

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    quest = tmp_path / "quest"
    eng = make_engine(quest)
    eng.config.engine.max_iterations = 3
    folder = quest / ".fi" / "figure_records"
    folder.mkdir(parents=True)
    (folder / f"{stem}.calls.json").write_text(json.dumps(calls), encoding="utf-8")
    image = f"{stem}.png"
    state = {"design": {"figures_planned": [image]}, "figures": [image], "quest_root": str(quest), "topic": "t", "title": "t",
             "iteration": 0, "figure_records": {image: {"axes": [{"ylabel": "y", "series": [{"label": "s", "shows": "yes"}]}]}}}
    out = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    paper = Path(out["paper_md"]).read_text(encoding="utf-8")
    return paper, paper_facts.kept_check(quest).get("taken_out_values") or []


def _first(offers, claim):
    return next(o for o in offers if o["claim"] == claim)


def _check_fails_closed(name, tmp_path, monkeypatch):
    contract, stem, offers = _honest(name, tmp_path)
    lines = {s["id"] for p in figure_claims.load_calls(tmp_path / "honest", stem)["panels"] for s in p["series"] if s["kind"] != "image"}
    # a line keeps its own name when its values are not the table's; an image is named by its quantity, which FI then withholds
    true_one = next(o for o in offers if o["claim"] == "drawn" and o["args"]["series"] in lines)
    value = _first(offers, "value_at")
    series = true_one["args"]["series"]
    # (a) value_at on an `other` quantity: the body drew values that are not the table's, so FI holds no value to state
    _run(name, tmp_path / "other", shift=0.5)
    other = figure_claims.load_calls(tmp_path / "other", stem)
    assert any(s["what"] == "other" for p in other["panels"] for s in p["series"] if s["kind"] != "image"), "the record says `other`"
    assert figure_claims.evaluate(other, "value_at", value["args"], plan=contract["plan"]) is None
    assert figure_claims.evaluate(other, "drawn", true_one["args"]) is not None, "the shape of the drawing is still stated"
    # (b) any claim on a raw panel: nothing is offered or evaluated for the panel the body drew itself
    shown = true_one["args"]["panel"]  # the panel of the statement the writer states below
    _run(name, tmp_path / "raw", raw=int(shown) - 1)
    raw = figure_claims.load_calls(tmp_path / "raw", stem)
    on_panel = [o for o in offers if o["args"].get("panel") == shown]
    assert on_panel and all(figure_claims.evaluate(raw, o["claim"], o["args"], plan=contract["plan"]) is None for o in on_panel)
    assert not [o for o in figure_claims.menu(raw, contract["plan"], stem=stem) if o["args"].get("panel") == shown]
    # (c) an unknown claim id
    assert figure_claims.evaluate(other, BAD_ID, {"series": series}) is None
    assert figure_claims.evaluate(other, BAD_ID, {}) is None
    unknown = "{{figure:%s:%s @ series=%s, panel=1}}" % (stem, BAD_ID, series)
    # through the write node: the writer states (a) and (c); the true statement stays, the others are gone with one disclosure
    paper, kept = _write_paper(tmp_path / "w1", monkeypatch, other, stem, [true_one["mark"], value["mark"], unknown])
    # the series FI did not find to be the table's result is named neutrally now, not by the result's or the plan's label
    stays = figure_claims.evaluate(other, "drawn", true_one["args"])[0]
    assert stays in paper, "the statement FI can check stays"
    assert value["text"] not in paper and "not recorded" not in paper and BAD_ID not in paper, paper
    assert paper.count(DISCLOSURE) == 1
    keys = [d["key"] for d in kept if str(d.get("key", "")).startswith("figure:")]
    assert any("value_at" in k for k in keys) and any(BAD_ID in k for k in keys), keys
    # through the write node: the writer states a claim on the raw panel
    paper, kept = _write_paper(tmp_path / "w2", monkeypatch, raw, stem, [true_one["mark"]])
    assert true_one["text"] not in paper and "not recorded" not in paper
    assert paper.count(DISCLOSURE) == 1 and any(str(d.get("key", "")).startswith("figure:") for d in kept)


def test_spectra_unsupported_claim_fails_closed(tmp_path, monkeypatch):
    _check_fails_closed("spectra", tmp_path, monkeypatch)


def test_map_unsupported_claim_fails_closed(tmp_path, monkeypatch):
    _check_fails_closed("map", tmp_path, monkeypatch)


def test_survival_unsupported_claim_fails_closed(tmp_path, monkeypatch):
    _check_fails_closed("survival", tmp_path, monkeypatch)


def test_bars_heatmap_unsupported_claim_fails_closed(tmp_path, monkeypatch):
    _check_fails_closed("bars_heatmap", tmp_path, monkeypatch)


# ---- a mutation of each study's own -------------------------------------------------------------------------------------------

VALUE_CLAIMS = ("value_at", "steps")  # claims that state a number read off a drawn value


def _states_a_value(o):
    return o["claim"] in VALUE_CLAIMS or (o["claim"] == "range" and o["args"].get("axis") == "y" and "has values" not in o["text"])


def _check_shifted_values_are_withheld(name, tmp_path, *, kept_shape):
    """The body draws means 0.5 too high. Caught: every statement of a drawn value (a value, a y range, steps) is withdrawn on the
    lines, the statements of shape that remain are true, and the honest figure offered the statements of value."""
    contract, stem, honest = _honest(name, tmp_path)
    _run(name, tmp_path / "bad", shift=0.5)
    calls = figure_claims.load_calls(tmp_path / "bad", stem)
    bad = figure_claims.menu(calls, contract["plan"], stem=stem)
    line_panels = {str(p["panel"]) for p in calls["panels"] if any(s["kind"] != "image" for s in p["series"])}

    def on_lines(offers):
        return [o for o in offers if str(o["args"].get("panel")) in line_panels and _states_a_value(o)]

    assert on_lines(honest), "the honest figure states values"
    assert not on_lines(bad), [o["text"] for o in on_lines(bad)]
    for claim, args in kept_shape:
        assert any(o["claim"] == claim and all(o["args"].get(k) == v for k, v in args.items()) for o in bad), (claim, args)
    return honest, bad


def test_spectra_own_mutation_caught(tmp_path):
    """Drawn values that are not the table's: the 95% error bars, which are checked against the table too, are withdrawn."""
    honest, bad = _check_shifted_values_are_withheld(
        "spectra", tmp_path, kept_shape=[("drawn", {"series": "sample_a"}), ("crossing", {"a": "sample_a"})])
    assert [o for o in honest if o["claim"] == "errorbars"] and not [o for o in bad if o["claim"] == "errorbars"]


def test_map_own_mutation_caught(tmp_path):
    _check_shifted_values_are_withheld("map", tmp_path, kept_shape=[("drawn", {"series": "row_0"}), ("trend", {"series": "row_0"})])


def test_survival_own_mutation_caught(tmp_path):
    """Drawn values that are not the table's: the 95% band, whose edges are checked against the table's interval, is withdrawn."""
    honest, bad = _check_shifted_values_are_withheld("survival", tmp_path, kept_shape=[("drawn", {"series": "group_a"})])
    assert [o for o in honest if o["claim"] == "band"] and not [o for o in bad if o["claim"] == "band"]


def test_a_heat_map_whose_values_are_not_the_tables_is_named_neutrally(tmp_path):
    """The image keeps its shape statement, under the neutral name an axis of an unchecked quantity has, never the id `other`."""
    contract, _h, stem = _run("map", tmp_path, shift=0.5)
    offers = figure_claims.menu(figure_claims.load_calls(tmp_path, stem), contract["plan"], stem=stem)
    drawn = [o["text"] for o in offers if o["claim"] == "drawn" and o["args"].get("panel") == "1"]
    assert drawn == ["computed value is drawn in panel 1"], drawn


def test_bars_heatmap_own_mutation_caught(tmp_path):
    """Drawn values that are not the table's: the standard-deviation bars, checked against the table, are withdrawn."""
    honest, bad = _check_shifted_values_are_withheld("bars_heatmap", tmp_path, kept_shape=[("drawn", {"series": "treated"})])
    assert [o for o in honest if o["claim"] == "errorbars"] and not [o for o in bad if o["claim"] == "errorbars"]


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
