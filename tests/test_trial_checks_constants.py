"""A true value the plan fixes under a name in its ``constants`` (an activation energy of 50 kJ/mol, the log of a frequency
factor that follows from it) is a true value FI measures the reported uncertainty against, like a threshold or a grid
parameter. Before, only thresholds and grid parameters were looked up: live kinetics rounds declared
``true_value: Ea_true_kJ_per_mol`` with the number in ``constants`` and the whole calibration was skipped
("calibration not checked"). A true value may also be a formula of the plan's fixed numbers, which FI computes."""

from __future__ import annotations

import math
import random

from core import metric_spec, protocol_check
from core import trial_checks as tc


def _metric(truth, *, ident: str = "x_hat") -> dict:
    return {"id": ident, "estimand": "x", "kind": "mean", "unit": "one trial",
            "known_value": {"true_value": truth, "estimate": "x_hat", "standard_error": "x_se",
                            "interval": ["x_lo", "x_hi"], "level": 0.95}}


def _cells(true: float = 50.0, *, se_scale: float = 1.0, shift: float = 0.0, seed: int = 3, n: int = 200,
           settings: int = 3) -> list[dict]:
    """Trials of a t-3 interval around an estimate of ``true`` whose spread is the reported standard error; the reported
    standard error is multiplied by ``se_scale`` and the data shifted by ``shift`` to plant a fault."""
    rng = random.Random(seed)
    out = []
    for i in range(settings):
        sd = 0.5 + i
        est = [true + shift + rng.gauss(0, sd) for _ in range(n)]
        se = [sd * se_scale for _ in range(n)]
        out.append({"key": f"s{i}", "cell": {"n": i}, "planned": n, "failed": 0, "metrics": {
            "x_hat": {"values": est, "trials": list(range(n))}, "x_se": {"values": se, "trials": list(range(n))},
            "x_lo": {"values": [e - 1.96 * s for e, s in zip(est, se)], "trials": list(range(n))},
            "x_hi": {"values": [e + 1.96 * s for e, s in zip(est, se)], "trials": list(range(n))}}})
    return out


def _protocol(truth, constants: dict | None = None, thresholds: dict | None = None) -> dict:
    return {"grid": {"n": [0, 1, 2]}, "runs_per_setting": 200, "thresholds": thresholds or {},
            "constants": constants if constants is not None else {"Ea_true": 50.0}, "metrics": [_metric(truth)]}


def test_a_true_value_named_in_the_constants_is_measured_not_skipped():
    cal = tc.calibration(_cells(), _protocol("Ea_true"))
    assert cal.checked and not cal.not_checked
    assert {q.kind for q in cal.quantities} == {"coverage", "se_ratio", "bias"}
    assert all("Ea_true = 50" in q.truth for q in cal.quantities) and not cal.faults


def test_a_formula_of_the_constants_is_computed_by_fi():
    consts = {"k_ref": 0.01, "Ea": 50000.0, "R": 8.314462618, "T_ref": 300.0}
    truth = "log(k_ref) + Ea / (R * T_ref)"
    want = math.log(0.01) + 50000.0 / (8.314462618 * 300.0)
    assert abs(metric_spec.fixed_value(truth, metric_spec.fixed_numbers({"constants": consts})) - want) < 1e-12
    cal = tc.calibration(_cells(want), _protocol(truth, consts))
    assert cal.checked and not cal.faults


def test_a_true_value_that_is_nowhere_in_the_plan_is_still_not_checked_and_says_so():
    cal = tc.calibration(_cells(), _protocol("Ea_true", constants={"other": 1.0}))
    assert not cal.quantities and "neither a threshold, a constant nor a grid parameter" in cal.not_checked[0]
    notes = protocol_check.metric_notes({**_protocol("Ea_true", constants={"other": 1.0})})
    assert any("`Ea_true`" in n and "constant" in n for n in notes)
    # a formula over names the plan does not fix is not computed either
    cal = tc.calibration(_cells(), _protocol("log(missing)", constants={"other": 1.0}))
    assert not cal.quantities and cal.not_checked


def test_the_plan_check_accepts_a_constant_and_a_computable_formula():
    assert protocol_check.metric_notes(_protocol("Ea_true")) == []
    assert protocol_check.metric_notes(_protocol("2 * Ea_true")) == []


def test_a_planted_miscalibration_of_a_constant_true_value_is_caught():
    for kwargs in ({"se_scale": 0.5}, {"shift": 2.0}):
        cal = tc.calibration(_cells(**kwargs), _protocol("Ea_true"))
        assert cal.faults, kwargs
    # the true value the plan fixes is wrong by a little: found
    cal = tc.calibration(_cells(), _protocol("Ea_true", constants={"Ea_true": 52.0}))
    assert cal.faults


def test_the_reported_coverage_is_recounted_against_a_constant_true_value():
    cells = _cells()
    proto = _protocol("Ea_true")
    proto["metrics"].append({"id": "cover", "estimand": "coverage", "kind": "proportion", "unit": "one trial",
                             "known_value": proto["metrics"][0]["known_value"]})
    assert tc.reported_coverage_mismatches(cells, proto, {"cover": 0.0}) != []
