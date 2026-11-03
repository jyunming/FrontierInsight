"""A figure mark round-trips for every series id FI makes, and no raw ``{{figure:...}}`` mark reaches a delivered paper.

Made-up study: a signal read at settings of ``size``, one curve per level of ``temp`` (ids ``<result>@<setting>:<level>``,
which hold ``@``, ``:`` and spaces, and for an odd level also a comma, an equals sign, a quote or a brace)."""

from __future__ import annotations

import json
import re

import pytest

from core import figure_claims as fc
from core import paper_values as pv
from core import paper_facts as pf
from core.frame_figure import RECORD_VERSION

LEVELS = ["cold room", "20", "1.5", "a,b", "x=y", 'say "hi"', "{open}", "back\\slash", "a}}b", "line\nbreak"]


def _series(sid, ys):
    return {"id": sid, "label": sid.upper(), "kind": "line", "step": False, "x": [1, 2, 3, 4], "y": ys, "yerr": None,
            "merged_with": [], "x_checked": True}


def _calls():
    series = [_series(f"signal@temp:{lv}", [1.0 + i, 2.0 + i, 3.0 + 2 * i, 5.0 + i * i]) for i, lv in enumerate(LEVELS)]
    panel = {"panel": 1, "raw": False, "raw_reason": None,
             "x": {"id": "size", "kind": "setting", "label": "size", "scale": "linear"},
             "y": {"what": "signal", "label": "signal level", "scale": "linear"},
             "series": series, "bands": [], "refs": [], "notes": [], "texts_raw": []}
    return {"version": max(RECORD_VERSION, 2), "fallback": False, "panels": [panel]}


def _quest(tmp_path, calls):
    d = tmp_path / ".fi" / "figure_records"
    d.mkdir(parents=True)
    (d / "fig1.calls.json").write_text(json.dumps(calls), encoding="utf-8")
    return tmp_path


def _inner(mark):
    return mark[len("{{figure:"):-2]


def test_every_offered_mark_parses_back_to_its_claim_and_args():
    offers = fc.menu(_calls(), stem="fig1")
    assert {o["claim"] for o in offers} >= {"drawn", "trend", "range"}
    assert any(o["args"].get("series") == "signal@temp:a,b" for o in offers)
    for o in offers:
        head, at, tail = _inner(o["mark"]).partition(" @ ")
        assert head == f"fig1:{o['claim']}"
        assert fc.parse_args(tail if at else "") == o["args"], o["mark"]
        assert "\n" not in o["mark"] and "}}" not in o["mark"][:-2]


@pytest.mark.parametrize("value", LEVELS + ["plain", "two words", "a@b:c d", " lead", "trail ", "", "\u00e9\u4e2d"])
def test_format_then_parse_is_identity(value):
    if value != value.strip() or not value:
        pytest.skip("a value with an edge space or nothing is no id FI makes")
    args = {"series": f"signal@temp:{value}", "panel": "1"}
    assert fc.parse_args(fc.format_args(args)) == args


def test_plain_ids_are_written_as_before():
    assert fc.format_args({"series": "period_s@period_references:exact elliptic-integral period", "panel": "1"}) == \
        "series=period_s@period_references:exact elliptic-integral period, panel=1"


def test_paper_fills_every_offered_mark_with_the_offered_sentence(tmp_path):
    calls = _calls()
    root = _quest(tmp_path, calls)
    offers = fc.menu(calls, None, pv.sig, stem="fig1")
    md = "\n\n".join(o["mark"] for o in offers)
    r = pf.render(md, root)
    assert r.unfilled == ()
    text = pf.without_marks(r.text)
    assert "{{" not in text
    for o in offers:
        assert o["text"] in text, o["mark"]


def test_mark_longer_than_eighty_characters_is_filled(tmp_path):
    calls = _calls()
    root = _quest(tmp_path, calls)
    offer = next(o for o in fc.menu(calls, None, pv.sig, stem="fig1") if o["claim"] == "drawn" and "cold room" in o["args"]["series"])
    assert len(_inner(offer["mark"])) > 30
    long_id = "signal@temp:" + "cold room"
    mark = "{{figure:fig1:drawn @ series=" + long_id + ", panel=1}}"
    assert pf.without_marks(pf.render(mark, root).text).strip() == offer["text"]


@pytest.mark.parametrize("mark", [
    "{{figure:fig1:no_such_claim @ series=a, panel=1}}",
    "{{figure:fig1:drawn @ series=ghost, panel=1}}",
    '{{figure:fig1:drawn @ series="unterminated, panel=1}}',
    "{{figure:fig1:drawn @ series=a{b, panel=1}}",
    "{{figure:fig1:drawn @ " + "q" * 1500 + "}}",
])
def test_a_mark_fi_cannot_fill_is_taken_out_with_its_sentence_and_recorded(tmp_path, mark):
    root = _quest(tmp_path, _calls())
    r = pf.render(f"Before.\n\n{mark}\n\nAfter.", root)
    out, taken = pf.take_out_unfilled(r.text, figures_only=True)
    shown = pf.without_marks(out)
    assert "{{" not in shown and "figure:" not in shown and "not recorded" not in shown
    assert "Before." in shown and "After." in shown
    assert taken and all(t["key"].startswith("figure:") for t in taken)


def test_no_raw_mark_in_paper_slides_or_poster_text(tmp_path):
    root = _quest(tmp_path, _calls())
    r = pf.render("Intro.\n\n{{figure:fig1:no_such_claim}}\n\n{{figure:fig1:drawn @ series=a{b}}}\n\nEnd.", root)
    out, _taken = pf.take_out_unfilled(r.text, figures_only=True)
    for delivered in (out, pf.without_marks(out)):
        assert not re.search(r"\{\{\s*figure", delivered)


def test_an_unclosed_long_placeholder_renders_quickly(tmp_path):
    import time
    root = _quest(tmp_path, _calls())
    for text in ("{{x: a" + " " * 1000 + "b", "{{x: " + " " * 990, "{{figure:fig1:drawn @ " + " " * 990):
        start = time.perf_counter()
        pf.render("Before. " + text + " After.", root)
        assert time.perf_counter() - start < 0.5


def test_an_unfillable_stray_mark_is_recorded_under_its_own_figure_and_stays_stable(tmp_path):
    root = _quest(tmp_path, _calls())
    md = "![cap](figures/fig1.png)\n\nBefore.\n\n{{figure:fig1:bogus @ series=a{b, panel=1}}\n\nAfter."
    first = pf.render(md, root)
    assert first.unfilled == ("figure:fig1:bogus",)
    text = first.text
    for _ in range(3):
        again = pf.render(text, root)
        assert again.unfilled == ("figure:fig1:bogus",)
        text = again.text
    assert text == first.text
    out, taken = pf.take_out_unfilled(text, figures_only=True)
    assert taken and taken[0]["key"].startswith("figure:fig1:bogus") and taken[0]["figure"] == "1"
    assert "figure:claim" not in text and "Before." in out and "After." in out


@pytest.mark.parametrize("mark", [
    "{{{figure:fig1:drawn @ series=signal@temp:20, panel=1}}}",
    "{{figure:fig1:drawn @ series=signal@temp:20}}b, panel=1}}",
    "{{figure:fig1:bogus @ series=a}}b, panel=1}}",
])
def test_no_brace_or_tail_of_a_mark_is_left_in_the_paper(tmp_path, mark):
    root = _quest(tmp_path, _calls())
    out, _ = pf.take_out_unfilled(pf.render(f"Before.\n\n{mark}\n\nAfter.", root).text, figures_only=True)
    shown = pf.without_marks(out)
    assert "}" not in shown and "{" not in shown and "panel=1" not in shown


@pytest.mark.parametrize("prose", [" The set A}} is odd.", " The mean is 4.5}} units.", " Use `a}}` to close.", " | 4.5}} |"])
def test_prose_after_a_figure_statement_is_never_removed(tmp_path, prose):
    root = _quest(tmp_path, _calls())
    mark = "{{figure:fig1:drawn @ series=signal@temp:20, panel=1}}"
    out, _ = pf.take_out_unfilled(pf.render(f"Before.\n\n{mark}{prose}\n\nAfter.", root).text, figures_only=True)
    assert prose.strip() in pf.without_marks(out)
