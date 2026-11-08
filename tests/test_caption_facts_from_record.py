"""The caption's account of what a figure draws comes from FI's figure record: every reference line drawn (named by the
plan, with its value), a plan reference curve by its plan name, and a panel that draws nothing. The model never counts
them. Neutral made-up records; no model is called."""
from __future__ import annotations

from core.engine import _caption_from_calls


def _series(ident="r1", label="Result one", kind="line", **kw):
    return {"id": ident, "label": label, "kind": kind, "x": [1.0, 2.0], "y": [1.0, 2.0], "merged_with": [], **kw}


def _panel(n, series=(), refs=(), **kw):
    return {"panel": n, "raw": False, "x": {"id": "s1", "label": "Size"}, "y": {"label": "Value"},
            "series": list(series), "refs": list(refs), "bands": [], **kw}


def _ref(ident, label, value):
    return {"id": ident, "axis": "y", "value": value, "label": label, "source": "plan"}


def test_caption_lists_every_drawn_reference_with_name_and_value():
    refs = [_ref("ref_a", "Bound A", 0.01), _ref("ref_b", "Bound B", 0.05), _ref("ref_c", "Bound C", 0.1),
            _ref("ref_d", "Bound D", 0.2)]
    text = _caption_from_calls("f.png", {"panels": [_panel(1, [_series()], refs)]}, None, n_seeds=None, facts=None)
    for label, value in (("Bound A", "0.01"), ("Bound B", "0.05"), ("Bound C", "0.1"), ("Bound D", "0.2")):
        assert f"{label} at {value}" in text, text


def test_caption_names_a_plan_reference_curve_by_its_plan_name():
    curve = _series("ref_a", "Theory A", kind="curve", source="plan")
    text = _caption_from_calls("f.png", {"panels": [_panel(1, [_series(), curve])]}, None, n_seeds=None, facts=None)
    assert "Theory A" in text and "a curve" not in text, text


def test_caption_names_a_panel_that_draws_nothing():
    calls = {"panels": [_panel(1, [_series()]), _panel(2)]}
    text = _caption_from_calls("f.png", calls, None, n_seeds=None, facts=None)
    assert "panel 2 draws nothing" in text.lower(), text
