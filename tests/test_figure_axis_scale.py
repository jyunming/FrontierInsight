"""The scale of a figure's axes: the plan declares ``x_scale`` / ``y_scale`` ("linear" or "log") and FI draws it; when the plan is
silent FI's own rule decides (log when every value drawn along the axis is above zero and the largest is 1000 times the
smallest) and the record says which one set the scale. Neutral made-up study: a colony's ``growth`` read at settings of ``size``."""

from __future__ import annotations

import json
import string
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import analysis_frame as af  # noqa: E402
from core import contract, contract_parts as cp, frame_figure  # noqa: E402
from tests.test_frame_figure import SPEC, _Host  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


X = [1, 2, 3, 4]
SMALL = [1e-10, 1e-8, 1e-6, 1e-4]


def _drawn(y, declared=None, x=X, **kw):
    h = _Host(panels=1, **kw)
    h.line(0, "fast", x, y, "growth")
    h.finish(SPEC, {}, declared)
    return h, h.record()["panels"][0]


def test_a_declared_log_axis_is_drawn_log_and_the_record_names_the_plan():
    h, rec = _drawn([0.2, 0.4, 0.6, 0.8], {"y": "log"})  # a narrow range: FI's rule alone would say linear
    assert h.axes[0].get_yscale() == "log" and h.axes[0].get_xscale() == "linear"
    assert rec["y"]["scale"] == "log" and rec["y"]["scale_source"] == "plan" and "scale_note" not in rec["y"]


def test_a_declared_linear_axis_stays_linear_over_many_decades():
    h, rec = _drawn(SMALL, {"y": "linear"})
    assert h.axes[0].get_yscale() == "linear" and rec["y"]["scale_source"] == "plan"


def test_a_silent_plan_gets_fis_rule_on_the_drawn_values():
    h, rec = _drawn(SMALL)
    assert h.axes[0].get_yscale() == "log" and rec["y"]["scale"] == "log" and rec["y"]["scale_source"] == "FI rule"
    h, rec = _drawn([0.1, 0.3, 0.5, 0.9])
    assert h.axes[0].get_yscale() == "linear" and rec["y"]["scale"] == "linear" and rec["y"]["scale_source"] == "FI rule"
    assert rec["x"]["scale"] == "linear"  # the x values 1..4 span a factor of four


def test_the_rule_applies_to_each_axis_alone():
    h, rec = _drawn([0.2, 0.4, 0.6, 0.8], x=[1, 10, 100, 1000])
    assert h.axes[0].get_xscale() == "log" and h.axes[0].get_yscale() == "linear"
    assert rec["x"]["scale_source"] == "FI rule"


def test_a_declared_log_axis_with_a_zero_is_drawn_linear_and_says_why():
    h, rec = _drawn([0.0, 1e-8, 1e-6, 1e-4], {"y": "log"})
    assert h.axes[0].get_yscale() == "linear" and rec["y"]["scale"] == "linear"
    assert rec["y"]["scale_source"] == "plan" and "zero or negative" in rec["y"]["scale_note"]


def test_a_zero_makes_the_rule_say_linear_without_a_note():
    h, rec = _drawn([0.0, 1e-8, 1e-6, 1e-4])
    assert h.axes[0].get_yscale() == "linear" and "scale_note" not in rec["y"]


def test_the_end_of_an_error_bar_below_zero_does_not_stop_a_declared_log_axis():
    h = _Host(panels=1, identity=True)
    h.line(0, "fast", [1, 10, 100], [1.0, 10.0, 100.0], "growth", yerr=[2.0, 2.0, 2.0])
    h.finish(SPEC, {}, {"y": "log"})
    assert h.axes[0].get_yscale() == "log"


# ---- the plan's declaration reaches the drawing -----------------------------------------------------------------------------

TEMPLATE = string.Template((Path(__file__).parent.parent / "agents" / "plan_figures.md").read_text(encoding="utf-8"))
DESIGN = {"hypothesis": "h", "method": "Simulate colony growth and count the peak.",
          "figures_planned": ["colony_peak.png (peak against size)"],
          "protocol": {"grid": {"size": [1, 2, 4]}, "runs_per_setting": 5,
                       "metrics": [{"id": "peak", "kind": "mean", "estimand": "peak count", "unit": "one colony run"}]}}


def _asked():
    return cp.figure_faults(cp.strip_fi_keys(DESIGN))


def test_a_bad_scale_answer_is_named_and_a_good_one_is_not():
    held = cp.strip_fi_keys(DESIGN)
    row = {"shows": ["peak"], "x": "size", "series": None, "panels": 1}
    name = "colony_peak.png"
    assert cp._check_figures({name: {**row, "x_scale": "log", "y_scale": "linear"}}, _asked(), held) == []
    assert cp._check_figures({name: row}, _asked(), held) == []  # silent is fine
    text = " | ".join(cp._check_figures({name: {**row, "y_scale": "logarithmic"}}, _asked(), held))
    assert "`y_scale`" in text and '"linear" or "log"' in text


def test_the_declared_scale_is_kept_and_reaches_the_analysis_plan():
    name = "colony_peak.png"
    answer = {"shows": ["peak"], "x": "size", "series": None, "panels": 1, "x_scale": "log", "y_scale": "bogus"}
    design = json.loads(json.dumps(DESIGN))
    design["protocol"] = cp._apply_figures(design["protocol"], design, {name: answer}, _asked())
    kept = design["protocol"]["figure_declarations"][name]
    assert kept["x_scale"] == "log" and "y_scale" not in kept  # a value that is not one of the two is not kept
    figs = {f.file: f for f in contract.from_design(design, fi_analysis=True).figures}
    assert (figs[name].x_scale, figs[name].y_scale) == ("log", None)
    plan = af.plan_of(design)["figures"][figs[name].id]
    assert plan["x_scale"] == "log" and "y_scale" not in plan


def test_the_question_asks_for_the_scale_in_plain_words():
    text = TEMPLATE.template
    assert "x_scale" in text and "y_scale" in text and '"linear" or "log"' in text


def test_the_plans_scale_is_what_a_drawn_figure_gets(tmp_path):
    from core import analysis_figures as afig
    plan = afig._plan_for({"id": "f", "x_scale": "log", "y_scale": "linear"}, {}, None)
    assert plan["x_scale"] == "log" and plan["y_scale"] == "linear"
    assert "x_scale" not in afig._plan_for({"id": "f", "x_scale": "weird"}, {}, None)
