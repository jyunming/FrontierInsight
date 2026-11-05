"""A trial that returns no number for a value compared inside each run belongs to every record of its setting.

Which setting a number belongs to is FI's own fact (the setting it ran and the values the plan has each run compare inside
it), not a record formed by the names a run happened to return. A study of a decay rate compares two fitting rules (``rule``)
and two interval widths (``width``) inside each run; a fit that fails returns ``None`` under the plain names while a fit that
works returns each number keyed by the rule (and the width). The failed trials must not become a record of their own that
holds only missing numbers (judged as "no estimate in 100% of the trials"), and the records of a rule and a width must still be
completed with the estimate that belongs to the same rule."""

from __future__ import annotations

import copy
import math
import random

from core import in_run, trial_checks

NAN = float("nan")
RULES = ["plain", "robust"]
WIDTHS = [1.96, 3.18]
INSIDE = {"rule": RULES, "width": WIDTHS}
LEVELS = in_run.level_cells(INSIDE)
TRUE = 10.0
PROTOCOL = {
    "grid": {"noise": [0.1, 0.4], "points": [5, 12]},
    "in_each_run": INSIDE,
    "runs_per_setting": 200,
    "metrics": [{"id": "est", "estimand": "the decay rate", "kind": "mean", "unit": "a trial",
                 "known_value": {"true_value": TRUE, "estimate": "est", "standard_error": "se", "interval": ["lo", "hi"],
                                 "level": 0.95}}],
}
THRESHOLDS = {"truth": TRUE}


def _rows(noise: float, points: int, fail: float = 0.0, shift: float = 0.0, seed: int = 0, trials: int = 200) -> list[dict]:
    """Trials of one setting: a fit that works keys each number by the rule (the estimate and the standard error) and by the
    rule and the width (the interval ends); a fit that fails returns ``None`` under the plain names."""
    rng = random.Random(seed * 1000 + points)
    sd = noise * 2.0
    rows = []
    for t in range(trials):
        if rng.random() < fail:
            rows.append({"trial": t, "status": "ok", "values": {"est": NAN, "se": NAN, "lo": NAN, "hi": NAN, "miss": 1.0}})
            continue
        values: dict = {"est": {}, "se": {}, "lo": {}, "hi": {}}
        for rule in RULES:
            e = TRUE + rng.gauss(0.0, sd)
            values["est"][f"rule={rule}"] = e
            values["se"][f"rule={rule}"] = sd
            for w in WIDTHS:
                key = f"rule={rule},width={w:g}"
                half = 1.96 * sd if w == 1.96 else 3.18 * sd
                values["lo"][key] = e - half + shift * half
                values["hi"][key] = e + half + shift * half
        values["miss"] = 0.0
        rows.append({"trial": t, "status": "ok", "values": values})
    return rows


def _cells(fail_at_noise: float = 0.0, fail: float = 0.0, shift: float = 0.0, rows_fn=None) -> list[dict]:
    out = []
    for noise in PROTOCOL["grid"]["noise"]:
        for points in PROTOCOL["grid"]["points"]:
            rows = _rows(noise, points, fail if noise >= fail_at_noise else 0.0, shift)
            out.extend(in_run.summary_cells({"noise": noise, "points": points}, f"noise={noise},points={points}", 200, rows, LEVELS))
    return out


def _judge(cells: list[dict]) -> trial_checks.Verdict:
    return trial_checks.judge(cells, PROTOCOL, THRESHOLDS, {}, repairs_left=False)


def test_failed_trials_under_the_plain_names_are_not_a_record_of_their_own() -> None:
    """The failing input: a setting where a few percent of the fits failed (plain ``None``) next to the keyed records."""
    cells = _cells(fail_at_noise=0.4, fail=0.1)
    # the old formation made, for each setting that had a failed fit, a record without the rule that held only those trials
    plain = [c for c in cells if not c.get("level") and c["cell"].get("noise") == 0.4]
    assert plain and all("est" not in c["metrics"] and "lo" not in c["metrics"] for c in plain)
    # every record of a rule holds every trial of the setting, the failed ones as missing numbers
    rule = next(c for c in cells if c.get("level") == {"rule": "plain"} and c["cell"]["noise"] == 0.4 and c["cell"]["points"] == 5)
    values = rule["metrics"]["est"]["values"]
    assert len(values) == 200 and rule["metrics"]["est"]["non_finite"] == sum(1 for v in values if not math.isfinite(v)) > 0
    wide = next(c for c in cells if c.get("level") == {"rule": "plain", "width": 3.18} and c["cell"]["noise"] == 0.4)
    assert wide["metrics"]["lo"]["count"] == 200 and wide["metrics"]["lo"]["non_finite"] == rule["metrics"]["est"]["non_finite"]
    # the trial ids of the failed trials are the same in every record of the setting
    assert rule["metrics"]["est"]["trials"] == wide["metrics"]["lo"]["trials"] == list(range(200))
    # a name only ever returned plain stays where it is
    assert plain[0]["metrics"]["miss"]["count"] == 200


def test_a_few_failed_fits_are_not_judged_as_no_estimate_or_zero_coverage() -> None:
    verdict = _judge(_cells(fail_at_noise=0.4, fail=0.1))
    assert verdict.action == trial_checks.OK, verdict.say
    empty = next(q for q in verdict.calibration.quantities if q.kind == "no_estimate")
    assert not empty.gross and max(s.value for s in empty.settings) < 0.2
    cover = next(q for q in verdict.calibration.quantities if q.kind == "coverage")
    assert cover.judged and not cover.gross


def test_the_old_formation_is_what_stopped_it(monkeypatch) -> None:
    """The same trials with the old record (a record of the failed trials alone) are judged "no estimate in all of the trials"."""
    monkeypatch.setattr(in_run, "settle_missing", lambda rows, levels: rows)
    verdict = _judge(_cells(fail_at_noise=0.4, fail=0.1))
    assert verdict.action == trial_checks.STOP and "100%" in verdict.say


def test_half_of_every_settings_trials_with_no_estimate_still_stops() -> None:
    verdict = _judge(_cells(fail_at_noise=0.0, fail=0.6))
    assert verdict.action == trial_checks.STOP
    assert any(q.kind == "no_estimate" and q.gross for q in verdict.calibration.quantities)


def test_intervals_shifted_to_about_half_coverage_still_stop() -> None:
    verdict = _judge(_cells(fail_at_noise=0.4, fail=0.1, shift=1.0))
    assert verdict.action == trial_checks.STOP
    cover = next(q for q in verdict.calibration.quantities if q.kind == "coverage")
    assert cover.gross


def test_a_setting_where_every_trial_failed_is_still_all_missing() -> None:
    cells = _cells(fail_at_noise=0.0, fail=1.0)
    assert not any(c.get("level") for c in cells)   # nothing was ever keyed: no record per value exists
    assert all(c["metrics"]["est"]["non_finite"] == 200 for c in cells)


def test_a_name_keyed_over_different_factors_in_different_trials_is_left_as_it_is() -> None:
    rows = [{"trial": 0, "status": "ok", "values": {"x": {"rule=plain": 1.0, "rule=robust": 2.0}}},
            {"trial": 1, "status": "ok", "values": {"x": {"rule=plain,width=1.96": 1.0, "rule=plain,width=3.18": 1.0,
                                                           "rule=robust,width=1.96": 1.0, "rule=robust,width=3.18": 1.0}}},
            {"trial": 2, "status": "ok", "values": {"x": NAN}}]
    assert in_run.settle_missing(copy.deepcopy(rows), LEVELS)[2]["values"]["x"] is NAN or math.isnan(
        in_run.settle_missing(copy.deepcopy(rows), LEVELS)[2]["values"]["x"])


def test_a_finite_plain_number_and_a_failed_trial_row_are_not_moved() -> None:
    rows = [{"trial": 0, "status": "ok", "values": {"x": {"rule=plain": 1.0, "rule=robust": 2.0}, "y": 3.0}},
            {"trial": 1, "status": "ok", "values": {"x": NAN, "y": 4.0}},
            {"trial": 2, "status": "failed", "reason": "boom"}]
    got = in_run.settle_missing(rows, LEVELS)
    assert got[1]["values"]["y"] == 4.0 and got[0]["values"]["y"] == 3.0 and got[2] == rows[2]
    assert set(got[1]["values"]["x"]) == {"rule=plain", "rule=robust"}
    assert all(math.isnan(v) for v in got[1]["values"]["x"].values())


def test_a_record_of_a_rule_and_a_width_takes_the_estimate_of_the_same_rule() -> None:
    """The estimate is kept with the rule and the interval ends with the rule and the width: one setting's numbers per trial."""
    cells = trial_checks._with_main(_cells())
    wide = next(c for c in cells if c.get("level") == {"rule": "robust", "width": 1.96} and c["cell"]["noise"] == 0.1)
    narrow = next(c for c in cells if c.get("level") == {"rule": "robust"} and c["cell"]["noise"] == 0.1)
    assert wide["metrics"]["est"] == narrow["metrics"]["est"]
    plain_rule = next(c for c in cells if c.get("level") == {"rule": "plain", "width": 1.96} and c["cell"]["noise"] == 0.1)
    assert plain_rule["metrics"]["est"] != wide["metrics"]["est"]   # the other rule's estimate is never borrowed
