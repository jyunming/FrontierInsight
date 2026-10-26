"""The plan is asked to declare a known true value (with the interval's multiplier) and a worked example by an
independent route, and FI says plainly when a plan leaves the multiplier out. No model is called."""
from __future__ import annotations

from core import engine as E
from core import equation_tests as eqt
from core import metric_spec as ms
from core import plan_steps as ps
from core import protocol_check as pc


def _metric(**kv):
    spec = {"id": "gain_hat", "estimand": "the gain", "kind": "mean", "unit": "one run"}
    spec["known_value"] = {"true_value": 2.0, "estimate": "gain_hat", "standard_error": "gain_se",
                           "interval": ["gain_lo", "gain_hi"], "level": 0.95, **kv}
    return spec


def test_plan_directive_asks_for_the_known_value_and_multiplier_of_every_metric():
    """Old plan request: `known_value` was described as a field to use 'for a number the simulation knows the true value
    of' and `degrees_of_freedom` as 'when your design fixes the multiplier', so real plans left both out and the
    calibration and quantile checks never ran. Now the request asks metric by metric and says ALWAYS."""
    text = E._PLAN_DIRECTIVE
    assert "@@" not in text
    assert "Go through EVERY metric" in text
    assert "ALWAYS give `degrees_of_freedom`" in text
    # a value the plan's model derived is not a known answer: it belongs in an oracle, unverified
    assert "FI reports it as unverified" in text


def test_plan_directive_asks_for_examples_and_not_as_an_option():
    """Old step text: '`example` and `relations` are optional'. A step or equation example copied from its own formula
    passes whatever convention the formula has; the request now asks for an independent route."""
    assert "are optional" not in ps.DIRECTIVE_TEXT
    assert "Give `example` for every step" in ps.DIRECTIVE_TEXT
    assert "DIFFERENT route" in eqt.EXAMPLE_RULE
    assert "DIFFERENT route" in E._PLAN_DIRECTIVE


def test_multiplier_unstated_lists_only_the_metrics_that_left_it_out():
    """Old code had no such list. A metric with an interval and a standard error but no multiplier is named."""
    missing = {"metrics": [_metric()]}
    assert ms.multiplier_unstated(missing) == ["gain_hat"]
    given = {"metrics": [_metric(degrees_of_freedom=3)]}
    assert ms.multiplier_unstated(given) == []
    normal = {"metrics": [_metric(degrees_of_freedom="normal")]}
    assert ms.multiplier_unstated(normal) == []
    no_interval = _metric()
    del no_interval["known_value"]["interval"]
    assert ms.multiplier_unstated({"metrics": [no_interval]}) == []


def test_plan_notes_say_when_the_multiplier_is_missing_and_stay_quiet_when_given():
    protocol = {"metrics": [_metric()], "runs_per_setting": 50}
    notes = pc.metric_notes(protocol)
    assert any("degrees_of_freedom" in n and "gain_hat" in n for n in notes)
    protocol = {"metrics": [_metric(degrees_of_freedom=3)], "runs_per_setting": 50}
    assert not any("degrees_of_freedom" in n for n in pc.metric_notes(protocol))


def test_an_example_written_by_another_route_catches_a_convention_slip():
    """Known positive: the equation is written with a period where a rate is meant. An example whose expected value comes
    from the known limit (rate 0.5 gives 0.5 at t=1 ... written by another route) disagrees with the equation as
    written, and FI says the example and the equation differ; an example copied from the equation's own form agrees."""
    protocol = {"model": {"equations": [
        {"id": "E1", "formula": "y = exp(-t / k)", "role": "analyses", "source": "derivation",
         "example": {"inputs": {"t": 1.0, "k": 0.5}, "expected_formula": "exp(-0.5 * t)"}}]}}
    rows = eqt.example_rows(protocol)
    assert rows[0]["state"] == "unusable" and "equation as written" in rows[0]["why"]
    protocol["model"]["equations"][0]["example"] = {"inputs": {"t": 1.0, "k": 0.5}, "expected_formula": "exp(-t / k)"}
    assert eqt.example_rows(protocol)[0]["state"] == "ok"


def test_a_step_may_say_it_cannot_be_tested_with_a_reason():
    cards, problems = ps.parse({"model": {"equations": [{"id": "E1", "formula": "x = a", "role": "generates"}], "steps": [
        {"id": "S1", "name": "decay", "does": "decays x", "inputs": {"k": "1/s"}, "output": "x",
         "example": {"untestable": "the output is a fitted curve with no closed form"}}]}})
    assert problems == [] and cards[0]["example_state"] == "untestable"
