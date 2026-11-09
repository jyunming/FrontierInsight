"""A constant reference line typed in a figure body (``fig.ref(panel, value=...)``) is never bound to a plan reference by
its value (a number that happens to equal a reference is a coincidence, not an identity): it is refused, and only
``fig.ref(panel, "<id>")`` draws a plan reference. FI never draws a line it cannot name. Neutral made-up study; no model is called."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from tests.test_frame_figure import _Host, _IdentityTable, _Table  # noqa: E402


class _RefTable(_Table):
    """Two settings of `step`; the plan reference `per_step` is 0.05 at step=1 and 0.04 at step=2."""

    def keys(self):
        return ["size=1,step=1", "size=1,step=2"]

    _references = {"per_step": {"size=1,step=1": 0.05, "size=1,step=2": 0.04}}


def _host(table=None, references=None):
    h = _Host()
    if table is not None:
        h.table = table
    h.references = references if references is not None else {"limit": {"value": 3.0, "label": "upper limit"}}
    return h


def test_value_equal_to_one_plan_reference_is_not_bound_to_it():
    h = _host(references={"half": {"value": 0.5, "label": "half level"}})
    with pytest.raises(ValueError) as e:
        h.ref(0, value=0.5)
    assert "half" in str(e.value) and "fig.ref(panel" in str(e.value)  # the message names the id to draw it by
    assert h.record()["panels"][0]["refs"] == [] and not h.axes[0].lines


def test_known_positive_reference_by_id_is_recorded_as_plan_and_found_drawn():
    from core import figure_text_check as ftc
    protocol = {"metrics": [{"id": "level", "label": "Level", "reference": {"value": 0.5, "kind": "value"}}]}
    series = {"id": "level", "label": "level", "kind": "line", "what": "level", "x": [1, 2], "y": [1.0, 2.0]}
    for drawn_by_id in (True, False):
        h = _host(references={"level": {"value": 0.5, "label": "half level"}})
        if drawn_by_id:
            h.ref(0, "level")
        else:
            with pytest.raises(ValueError):
                h.ref(0, value=0.5)  # an unrelated line at 0.5 is not the plan's reference
        refs = h.record()["panels"][0]["refs"]
        calls = {"panels": [{"panel": 1, "raw": False, "series": [series], "refs": refs}]}
        missing = ftc.missing_references(calls, protocol)
        assert (missing == []) is drawn_by_id
        if drawn_by_id:
            assert (refs[0]["id"], refs[0]["source"], refs[0]["value"]) == ("level", "plan", 0.5)


def test_value_matching_two_plan_references_is_refused():
    h = _host(references={"limit": {"value": 3.0, "label": "upper limit"}, "cap": {"value": 3.0, "label": "cap"}})
    with pytest.raises(ValueError) as e:
        h.ref(0, value=3.0)
    assert "limit" in str(e.value) and "cap" in str(e.value)
    assert h.record()["panels"][0]["refs"] == [] and not h.axes[0].lines


def test_value_matching_only_a_result_value_is_refused():
    h = _host()
    h.table = _IdentityTable()
    with pytest.raises(ValueError) as e:
        h.ref(0, value=2.0)  # the table's mean at size=2, not a reference of the plan
    assert "at=" in str(e.value)
    assert h.record()["panels"][0]["refs"] == []


def test_a_reference_by_id_is_found_at_the_panels_own_setting_only():
    h = _host(table=_RefTable(), references={})
    h._panel(0)[1]["hold"] = {"step": 2}
    with pytest.raises(ValueError):
        h.ref(0, value=0.04)  # equal to the reference at step=2, but a typed number is not bound by value
    h.ref(0, "per_step")
    ref = h.record()["panels"][0]["refs"][0]
    assert (ref["id"], ref["source"], ref["value"]) == ("per_step", "plan", 0.04)


def test_stated_source_no_longer_written():
    h = _host()
    h.ref(0, "limit")
    h.ref(0, at="growth", key="size=1")
    assert all(r["source"] != "stated in the figure" for r in h.record()["panels"][0]["refs"])
