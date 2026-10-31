"""A key FI fills inside math keeps the key in the paper (the span is wrapped in a marker holding it as written), so the
next render, the passage editor, the typed-number check and the goal coverage still see it; the words that qualify a value
never go inside math."""
from __future__ import annotations

from core import paper_facts as pf
from core import paper_values as pv


def _cell(a: int, y: float) -> dict:
    return {"cell": {"a": a}, "key": f"a={a}", "planned": 1, "ok": 1, "failed": 0,
            "metrics": {"y": {"values": [y], "trials": [0], "count": 1, "total": y, "non_finite": 0}}}


_REC = pv.Records(cells=[_cell(a, 22.8138) for a in range(1, 6)])
_KEY = "{{v:y:mean_ci @ a=all}}"


def _render(tmp_path, md):
    return pf.render(md, tmp_path, records=_REC).text


def test_a_value_in_math_is_the_bare_number_inside_a_marker_that_keeps_the_key(tmp_path):
    text = _render(tmp_path, f"The root is $A = {_KEY}$ degrees.")
    assert "$A = 22.81$" in text and "settings" not in text.split("-->$A")[1]
    assert pf.without_marks(text) == "The root is $A = 22.81$ degrees."
    assert pf.to_placeholders(text) == f"The root is $A = {_KEY}$ degrees."


def test_the_next_render_restores_the_key_exactly(tmp_path):
    text = _render(tmp_path, f"The root is $A = {_KEY}$ degrees.")
    assert _render(tmp_path, text) == text


def test_a_number_filled_into_math_is_not_listed_as_untraced(tmp_path):
    text = _render(tmp_path, f"The root is $A = {_KEY}$ degrees.")
    assert pf.untraced_typed(text, None, "") == []
    assert pf.typed_hits(text, None, "") == []


def test_the_paper_still_names_the_quantity_placed_in_math(tmp_path):
    assert pv.placed_metrics(_render(tmp_path, f"The root is $A = {_KEY}$ degrees.")) == {"y"}


def test_known_positive_a_number_typed_in_no_record_is_still_listed(tmp_path):
    text = _render(tmp_path, f"The root is $A = {_KEY}$ degrees, and the other is 41.37 degrees.")
    assert [n for n, _s in pf.untraced_typed(text, None, "")] == ["41.37"]


def test_known_positive_a_number_typed_inside_the_same_math_span_is_listed(tmp_path):
    text = _render(tmp_path, "The root is $A = 22.81$ degrees.")
    assert [n for n, _s in pf.untraced_typed(text, None, "")] == ["22.81"]
