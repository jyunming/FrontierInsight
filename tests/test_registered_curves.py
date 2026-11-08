"""Every curve a figure draws has an identity in FI's records: a plan reference that changes along the panel's x is drawn
by FI from its own evaluation of the plan's formula and named by the plan; a curve the body computes itself is not drawn.
Neutral made-up study; no model is called."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from tests.test_frame_figure import _Host, _Table  # noqa: E402


class _Grid(_Table):
    """size 1..3 at step 1 and 2; the plan reference `law` is size * step at every setting."""

    def keys(self):
        return ["size=%d,step=%d" % (s, t) for t in (1, 2) for s in (1, 2, 3)]

    _references = {"law": {"size=%d,step=%d" % (s, t): float(s * t) for t in (1, 2) for s in (1, 2, 3)}}


def _host():
    h = _Host()
    h.table = _Grid()
    h.references = {}
    h.labels = {**h.labels, "law": "Scaling law"}
    return h


def test_a_plan_reference_that_changes_along_x_is_drawn_as_its_curve_and_named_by_the_plan():
    h = _host()
    h._panel(0)[1]["hold"] = {"step": 2}
    h.line(0, "fast", [1, 2, 3], [2.0, 4.0, 6.0], "growth")
    h.ref(0, "law")
    s = h.record()["panels"][0]["series"][-1]
    assert (s["id"], s["kind"], s["label"], s["source"]) == ("law", "curve", "Scaling law", "plan")
    assert s["x"] == [1.0, 2.0, 3.0] and s["y"] == [2.0, 4.0, 6.0]


def test_a_reference_that_changes_with_a_setting_the_panel_does_not_hold_is_refused():
    h = _host()  # no hold: at size=1 the reference is 1 (step 1) and 2 (step 2)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    with pytest.raises(ValueError) as e:
        h.ref(0, "law")
    assert "step" in str(e.value)


def test_a_reference_curve_needs_the_data_first():
    h = _host()
    h._panel(0)[1]["hold"] = {"step": 1}
    with pytest.raises(ValueError):
        h.ref(0, "law")


def test_a_curve_the_body_computes_itself_is_refused_and_points_to_the_records():
    h = _host()
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    with pytest.raises(ValueError) as e:
        h.curve(0, "trend", [1, 2, 3], [1.0, 2.0, 3.0])
    assert "plan" in str(e.value) and "analysis" in str(e.value)
    assert all(s["kind"] != "curve" for s in h.record()["panels"][0]["series"])


def test_no_placeholder_name_is_recorded_for_a_curve():
    h = _host()
    h._panel(0)[1]["hold"] = {"step": 1}
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.ref(0, "law")
    labels = [s["label"] for s in h.record()["panels"][0]["series"]]
    assert "a curve" not in labels and not any(str(lb).startswith("curve ") for lb in labels)
