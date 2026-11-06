"""A constant reference line typed in a figure body (``fig.ref(panel, value=...)``) takes the identity of the one plan
reference it equals at the panel's setting, or it is refused: FI never draws a line it cannot name (a legend entry is
written from FI's records, not guessed). Neutral made-up study; no model is called."""
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


def test_value_matching_one_plan_reference_takes_its_identity():
    h = _host()
    h.ref(0, value=3.0)
    ref = h.record()["panels"][0]["refs"][0]
    assert (ref["id"], ref["source"], ref["label"], ref["value"]) == ("limit", "plan", "upper limit", 3.0)


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


def test_value_matches_only_at_the_panels_own_setting():
    h = _host(table=_RefTable(), references={})
    h._panel(0)[1]["hold"] = {"step": 2}
    with pytest.raises(ValueError):
        h.ref(0, value=0.05)  # the reference at step=1, not at this panel's step=2
    h.ref(0, value=0.04)
    ref = h.record()["panels"][0]["refs"][0]
    assert (ref["id"], ref["source"], ref["value"]) == ("per_step", "plan", 0.04)


def test_a_reference_that_differs_within_the_panel_is_no_constant():
    h = _host(table=_RefTable(), references={})  # no hold: the panel spans step=1 and step=2
    with pytest.raises(ValueError):
        h.ref(0, value=0.05)


def test_value_within_typed_digits():
    h = _host(references={"limit": {"value": 0.0500000001, "label": "upper limit"}})
    h.ref(0, value=0.05)
    assert h.record()["panels"][0]["refs"][0]["id"] == "limit"
    h2 = _host(references={"limit": {"value": 0.05, "label": "upper limit"}})
    with pytest.raises(ValueError):
        h2.ref(0, value=0.0475)


def test_stated_source_no_longer_written():
    h = _host()
    h.ref(0, value=3.0)
    h.ref(0, "limit")
    h.ref(0, at="growth", key="size=1")
    assert all(r["source"] != "stated in the figure" for r in h.record()["panels"][0]["refs"])
