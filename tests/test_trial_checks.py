"""FI's own checks of what the trials gave (core/trial_checks.py): failed trials per setting, and the reported intervals and
standard errors against the true values the plan declares (a metric's ``known_value``).

FI reads only what the plan declares: the true value, the names run_trial returns for the estimate and for its standard
error or interval, and the level. It computes the coverage and the standard-error ratio itself from the trials; it never
reads a coverage flag the code computed, and it does not guess which name is which.

The replay part reads tests/fixtures/replay/trial_checks/: the trial records of real quests, cut to a few settings with
every trial's values (the quests' own folders are not touched). Their plans did not have ``known_value`` (it did not
exist), so each replay below writes the declaration its plan would have had (:data:`DECLARED`), from the names its code
returns and the true values its plan gives. r12 and r2 never ran their study, so their records were made by running their
own code through FI's trial harness on their grid (100 trials a setting): r12 as delivered (a noise level typed into the
fitting function, and a solver tolerance that fails most noisy trials), the same with only the tolerance relaxed (what a
repair of the crashes alone leaves), and with the noise passed in (the correct fix of study 2); r2 as delivered (two-step
weights on the wrong scale).

How they were made (to rebuild them): ``core.trial_runner.run_trials`` with FI's own harness, base seed 0, paired, 100
trials a setting, the study's own thresholds; r12 on its 5 x 5 grid (noise 0.005..0.08 x 5..32 time points) from study 2's
``orig/`` (= the delivered code) with run_trial also returning each fit's interval ends (``{label}_Ea_lower`` /
``_upper``, ``{label}_ln_A_lower`` / ``_upper``, from the fit's own ``Ea_interval`` and ``ln_A_interval``),
``r12_kinetics_tolerance_relaxed`` = the same with one hand edit in ``kinetics_parameter_recovery/model.py`` (the gradient
check ``abs(float(gradient[0])) >= solver_gtol`` made ``>= 1e-6``), ``r12_kinetics_noise_fixed`` = study 2's ``fixed/``
with the same two edits; r2 on noise 0, 0.025, 0.05, 0.1 x 6, 16, 36 time points. The other fixtures are cut from the
quests' ``raw/trials.json`` and ``raw/ledger.jsonl``.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from collections import Counter
from pathlib import Path

import pytest

from core import metric_spec
from core import protocol_check
from core import trial_checks as tc

FIX = Path(__file__).parent / "fixtures" / "replay" / "trial_checks"


def known(est: str, truth, *, se: str | None = None, var: str | None = None, interval: tuple[str, str] | None = None,
          level: float = 0.95, ident: str | None = None) -> dict:
    """A protocol metric that declares its known value."""
    kv = {"true_value": truth, "estimate": est, "level": level}
    if se:
        kv["standard_error"] = se
    if var:
        kv["variance"] = var
    if interval:
        kv["interval"] = list(interval)
    return {"id": ident or f"{est}{'_' + interval[0] if interval else ''}", "estimand": f"the true value of {est}",
            "kind": "mean", "unit": "one trial", "known_value": kv}


# The declaration each replayed plan would have had: the names its code returns, the true value its plan gives.
DECLARED: dict[str, list[dict]] = {
    **{name: [m for label in ("weighted", "unweighted") for m in (
        known(f"{label}_Ea_hat", "Ea_true_J_mol", interval=(f"{label}_Ea_lower", f"{label}_Ea_upper")),
        # the code generates its data with A = 1e7; the plan's threshold `ln_A_true` (9.45) is another number (below)
        known(f"{label}_ln_A_hat", math.log(1e7), interval=(f"{label}_ln_A_lower", f"{label}_ln_A_upper")))]
       for name in ("r12_kinetics", "r12_kinetics_tolerance_relaxed", "r12_kinetics_noise_fixed")},
    "r2_kinetics": [known(f"{w}_{p}", t, var=f"{w}_var_{p}") for w in ("global", "two_step")
                    for p, t in (("ea", "true_ea_j_mol"), ("ln_a", "true_ln_a"))],
    "r6t_kinetics": [m for w in ("weighted", "unweighted") for m in (
        known(f"{w}_ea", "Ea_true_J_mol", se=f"{w}_ea_se"), known(f"{w}_lnA", math.log(3e6), se=f"{w}_lnA_se"))],
    "r14b_kinetics": [known("Ea_hat", "true_Ea_J_mol", interval=(f"Ea_{f}_lower", f"Ea_{f}_upper")) for f in ("t", "normal")]
                     + [known("ln_A_hat", math.log(1e6), interval=(f"ln_A_{f}_lower", f"ln_A_{f}_upper"))
                        for f in ("t", "normal")],
    "r8_kinetics": [known("Ea_hat_J_mol", 50000.0, se="Ea_se_J_mol"), known("lnA_hat", math.log(1e8), se="lnA_se")],
    "s1_kinetics": [known("Ea_hat", "Ea_true_J_per_mol", se="Ea_se_reported"),
                    known("lnA_hat", "lnA_true", se="lnA_se_reported")],
}


def _load(name: str) -> dict:
    with gzip.open(FIX / f"{name}.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _protocol(data: dict, metrics: list[dict]) -> dict:
    cell = next((c.get("cell") for c in data["cells"] if c.get("cell")), {}) or {}
    return {"metrics": metrics, "thresholds": data.get("thresholds") or {}, "grid": {k: [] for k in cell}}


def _checks(name: str, metrics: list[dict] | None = None):
    data = _load(name)
    reasons = {k: Counter(v) for k, v in (data.get("reasons") or {}).items()}
    fails = tc.failures(data["cells"], reasons)
    decl = DECLARED.get(name, []) if metrics is None else metrics
    cal = tc.calibration(data["cells"], _protocol(data, decl), data.get("thresholds"))
    return fails, cal


def _cells(per: dict[str, dict[str, list[float]]], **head) -> list[dict]:
    out = []
    for key, metrics in per.items():
        n = max(len(v) for v in metrics.values())
        out.append({"key": key, "cell": head.get("cell", {}), "planned": head.get("planned", n),
                    "failed": head.get("failed", 0),
                    "metrics": {m: {"values": v, "trials": list(range(len(v)))} for m, v in metrics.items()}})
    return out


def _proto(*metrics: dict, thresholds: dict | None = None, grid: dict | None = None) -> dict:
    return {"metrics": list(metrics), "thresholds": thresholds or {}, "grid": grid or {}}


# ---- the declaration ---------------------------------------------------------------------------------------------------


def test_a_known_value_is_checked_for_its_shape():
    good = known("x_hat", "x_true", se="x_se")
    clean, why = metric_spec.normalize([good])
    assert why is None and clean[0]["known_value"] == {"true_value": "x_true", "estimate": "x_hat",
                                                       "standard_error": "x_se", "level": 0.95}
    for broken, says in (({"estimate": "x_hat", "standard_error": "x_se", "level": 0.95}, "true_value"),
                         ({"true_value": 1.0, "standard_error": "x_se", "level": 0.95}, "estimate"),
                         ({"true_value": 1.0, "estimate": "x_hat", "level": 0.95}, "standard_error"),
                         ({"true_value": 1.0, "estimate": "x_hat", "standard_error": "x_se"}, "level"),
                         ({"true_value": 1.0, "estimate": "x_hat", "interval": ["lo"], "level": 0.95}, "interval")):
        clean, why = metric_spec.normalize([{**good, "known_value": broken}])
        assert clean is None and says in why
    # a draft keeps the metric without the part it could not read, and says so
    kept, notes = metric_spec.repair([{**good, "known_value": {"true_value": 1.0}}])
    assert [m["id"] for m in kept] == [good["id"]] and "known_value" not in kept[0]
    assert notes and "kept without it" in notes[0]


def test_the_plan_says_when_a_true_value_names_nothing_or_nothing_is_declared():
    proto = {"runs_per_setting": 100, "thresholds": {"x_true": 1.0}, "grid": {"n": [5, 10]},
             "metrics": [known("x_hat", "x_ture", se="x_se")]}
    notes = protocol_check.metric_notes(proto)
    assert any("`x_ture`" in n and "neither a threshold nor a grid parameter" in n for n in notes)
    proto["metrics"] = [{"id": "x_hat", "estimand": "x", "kind": "mean", "unit": "a trial"}]
    assert any("No metric declares a known true value" in n for n in protocol_check.metric_notes(proto))
    proto["metrics"] = [known("x_hat", "n", se="x_se")]  # a grid parameter holds the truth
    assert protocol_check.metric_notes(proto) == []


# ---- the calibration ---------------------------------------------------------------------------------------------------


def test_a_standard_error_far_too_small_is_a_fault_and_a_right_one_is_not():
    rng = random.Random(2)
    proto = _proto(known("x_hat", "x_true", se="x_se"), thresholds={"x_true": 5.0})
    cells = _cells({f"s{i}": {"x_hat": [5 + rng.gauss(0, 1) for _ in range(100)], "x_se": [0.2] * 100} for i in range(4)})
    cal = tc.calibration(cells, proto)
    assert {q.kind for q in cal.faults} == {"se_ratio"}, "only a standard error: no coverage FI would build itself"
    cells = _cells({f"s{i}": {"x_hat": [5 + rng.gauss(0, 1) for _ in range(100)], "x_se": [1.0] * 100} for i in range(4)})
    assert not tc.calibration(cells, proto).faults


def test_coverage_is_computed_by_fi_from_the_interval_and_the_true_value_never_from_the_codes_flag():
    rng = random.Random(5)
    # The code's interval is around a value 3 standard errors off the truth, and its own flag says it always covered.
    est = [[8.0 + rng.gauss(0, 1) for _ in range(100)] for _ in range(4)]
    cells = _cells({f"s{i}": {"x_hat": e, "x_lo": [x - 1.96 for x in e], "x_hi": [x + 1.96 for x in e],
                              "x_covered": [1.0] * 100} for i, e in enumerate(est)})
    cal = tc.calibration(cells, _proto(known("x_hat", 5.0, interval=("x_lo", "x_hi"))))
    assert [q.kind for q in cal.faults] == ["coverage", "bias"]
    assert max(s.value for s in cal.faults[0].settings) < 0.3
    # without a declaration nothing is guessed from the names, and the record says why
    cal = tc.calibration(cells, _proto())
    assert not cal.quantities and cal.not_checked == [tc.NO_DECLARATION]


def test_with_only_a_standard_error_fi_judges_it_against_the_spread_and_the_estimates_against_the_truth():
    # no coverage: the trial reports no interval, and a z interval FI built would misjudge a fit with few degrees of freedom
    rng = random.Random(6)
    cells = _cells({f"s{i}": {"x_hat": [rng.gauss(0, 1) for _ in range(400)], "x_se": [1.0] * 400} for i in range(3)})
    cal = tc.calibration(cells, _proto(known("x_hat", 0.0, se="x_se", level=0.9)))
    assert [q.kind for q in cal.quantities] == ["se_ratio", "bias"] and not cal.faults
    bias = cal.quantities[1]
    assert all(abs(s.value) < 0.2 for s in bias.settings) and "0 when the estimate is right" in bias.says()


def test_a_true_value_held_by_a_grid_parameter_is_read_per_setting():
    rng = random.Random(7)
    cells = []
    for mu in (1.0, 10.0):
        cells += _cells({f"mu={mu:g}": {"m_hat": [mu + rng.gauss(0, 1) for _ in range(200)], "m_se": [1.0] * 200}},
                        cell={"mu": mu})
    cal = tc.calibration(cells, _proto(known("m_hat", "mu", se="m_se"), grid={"mu": [1.0, 10.0]}))
    assert cal.checked and not cal.faults
    assert "per setting" in cal.quantities[0].truth


def test_a_small_miss_is_only_reported_and_a_coverage_above_the_level_is_said():
    rng = random.Random(8)
    proto = _proto(known("x_hat", 0.0, interval=("lo", "hi")))

    def cells(half: float):
        out = {}
        for i in range(5):
            e = [rng.gauss(0, 1) for _ in range(400)]
            out[f"s{i}"] = {"x_hat": e, "lo": [x - half for x in e], "hi": [x + half for x in e]}
        return _cells(out)

    q = tc.calibration(cells(1.5), proto).quantities[0]  # about 0.87: below, not a fault
    assert not q.fault and q.below == 5 and q.gross == 0
    q = tc.calibration(cells(3.0), proto).quantities[0]  # about 0.997: wider than need be, not a fault
    assert not q.fault and q.above == 5 and "wider than need be" in q.says()


def test_one_grossly_missed_setting_is_a_fault_at_a_family_level_that_spares_a_clean_run():
    rng = random.Random(12)
    proto = _proto(known("x_hat", 0.0, se="x_se"))

    def run(off: float, bad: int, settings: int = 25):
        return _cells({f"s{i}": {"x_hat": [rng.gauss(off if i < bad else 0.0, 1) for _ in range(200)],
                                 "x_se": [1.0] * 200} for i in range(settings)})

    cal = tc.calibration(run(3.0, 1), proto)  # one corner of the grid 3 standard errors off
    bias = next(q for q in cal.faults if q.kind == "bias")
    assert bias.gross == 1 and bias.judged == 25
    # 25 clean settings: each interval is at 1 - 0.05/25, so an unlucky setting does not stop a right code
    assert not tc.calibration(run(0.0, 0), proto).faults


def test_a_moderate_miss_in_every_setting_is_caught_by_the_settings_together():
    rng = random.Random(13)
    cells = _cells({f"s{i}": {"x_hat": [rng.gauss(0, 1) for _ in range(400)], "x_se": [1 / 1.6] * 400}
                    for i in range(25)})
    cal = tc.calibration(cells, _proto(known("x_hat", 0.0, se="x_se")))
    ratio = next(q for q in cal.quantities if q.kind == "se_ratio")
    assert ratio.fault and ratio.pooled.gross and 1.5 < ratio.pooled.value < 1.7
    assert "all of them together" in ratio.says() and tc.POOLED in tc.calibration_directive([ratio])


def test_a_declared_name_the_trials_do_not_return_is_a_fault():
    rng = random.Random(9)
    cells = _cells({"s": {"x_hat": [rng.gauss(0, 1) for _ in range(50)]}})
    cal = tc.calibration(cells, _proto(known("x_hat", 0.0, se="x_stderr")))
    assert [q.kind for q in cal.faults] == ["missing"] and "`x_stderr`" in cal.faults[0].says()
    assert "exactly these names" in tc.calibration_directive(cal.faults)


def test_a_true_value_that_names_nothing_is_not_checked_and_says_why():
    rng = random.Random(10)
    cells = _cells({"s": {"x_hat": [rng.gauss(0, 1) for _ in range(50)], "x_se": [1.0] * 50}})
    cal = tc.calibration(cells, _proto(known("x_hat", "x_ture", se="x_se")))
    assert not cal.quantities and "neither a threshold nor a grid parameter" in cal.not_checked[0]


def test_too_few_trials_are_not_checked_and_the_reason_says_so():
    rng = random.Random(11)
    cells = _cells({f"s{i}": {"x_hat": [rng.gauss(0, 1) for _ in range(9)], "x_se": [0.1] * 9} for i in range(3)})
    cal = tc.calibration(cells, _proto(known("x_hat", 0.0, se="x_se")))
    assert not cal.checked and not cal.faults
    assert any(f"{tc.MIN_TRIALS} or more trials" in n for n in cal.not_checked)


def test_nothing_declared_is_said():
    cells = _cells({"s": {"power": [0.5] * 50, "fdr": [0.04] * 50}})
    cal = tc.calibration(cells, None, {"alpha": 0.05})
    assert not cal.checked and cal.not_checked == [tc.NO_DECLARATION]


def test_failed_trials_per_setting():
    cells = [{"key": "a=1", "planned": 100, "failed": 60, "metrics": {}},
             {"key": "a=2", "planned": 100, "failed": 6, "metrics": {}},
             {"key": "a=3", "planned": 100, "failed": 0, "metrics": {}},
             {"key": "a=3,m=x", "planned": 100, "failed": 60, "level": {"m": "x"}, "metrics": {}}]
    fails = tc.failures(cells)
    assert [(f.key, f.fault, f.stated) for f in fails] == [("a=1", True, True), ("a=2", False, True), ("a=3", False, False)]


def test_a_trial_that_returns_nan_is_not_a_failed_trial_it_has_no_value_there_or_misses_a_declared_truth():
    nan = float("nan")
    cells = [{"key": "d=0", "cell": {"d": 0}, "planned": 20, "failed": 0,
              "metrics": {"x_hat": {"values": [nan] * 15 + [1.0] * 5, "trials": list(range(20))},
                          "x_se": {"values": [0.01] * 20, "trials": list(range(20))},
                          "other": {"values": [nan] * 4 + [1.0] * 16, "trials": list(range(20))}}},
             {"key": "d=1", "cell": {"d": 1}, "planned": 20, "failed": 2,
              "metrics": {"x_hat": {"values": [1.0] * 17 + [float("inf")], "trials": list(range(18))},
                          "x_se": {"values": [0.01] * 18, "trials": list(range(18))},
                          "other": {"values": [1.0] * 18, "trials": list(range(18))}}}]
    other = {"id": "other", "estimand": "other", "kind": "mean", "unit": "one trial"}
    spec = known("x_hat", 1.0, se="x_se")
    # only the trials FI's record says failed
    fails = tc.failures(cells, None, _proto(spec, other))
    assert [(f.key, f.failed, f.fault) for f in fails] == [("d=0", 0, False), ("d=1", 2, False)]
    # a plain value of the plan with no number is recorded as having none there; the estimate of a declared truth is not
    # (it is the calibration's)
    assert [u.as_dict() for u in tc.undefined(cells, _proto(spec, other))] == [
        {"key": "d=0", "metric": "other", "trials": 4, "ran": 20, "all": False}]
    assert tc.undefined(cells, _proto(spec, {**other, "undefined_when": {"d": [0]}})) == []
    assert "let the trial raise" in tc.failures_directive([fails[1]])


def test_replay_r12_a_repair_that_returns_nan_instead_of_failing_is_caught_where_the_truth_is_declared():
    data = _load("r12_kinetics")
    names = sorted(next(c for c in data["cells"] if c["metrics"])["metrics"])
    cells = []
    for c in data["cells"]:
        c = json.loads(json.dumps(c))
        n_fail = int(c.get("failed") or 0)
        c["failed"] = 0
        for name in names:
            c["metrics"].setdefault(name, {"values": [], "trials": []})
        for m in c["metrics"].values():  # the trials that failed now "run" and return NaN
            m["values"] = list(m["values"]) + [float("nan")] * n_fail
            m["trials"] = list(m["trials"]) + list(range(10_000, 10_000 + n_fail))
        cells.append(c)
    # the NaN trials ran to the end, so they are not failed trials; with the truth declared they are misses of the
    # calibration, and the quest stops
    assert not [f for f in tc.failures(cells, None, _protocol(data, DECLARED["r12_kinetics"])) if f.failed]
    verdict = tc.judge(cells, _protocol(data, DECLARED["r12_kinetics"]), data["thresholds"], repairs_left=False)
    assert verdict.action == tc.STOP and verdict.calibration.faults
    bad = {s.key for q in verdict.calibration.faults for s in q.settings if s.gross}
    assert len(bad) >= 13, bad


# ---- replay on real quests --------------------------------------------------------------------------------------------


def test_replay_r12_most_noisy_trials_fail_and_are_sent_to_repair():
    fails, cal = _checks("r12_kinetics")
    bad = [f for f in fails if f.fault]
    assert len(bad) == 13
    assert all("did not satisfy required optimality" in f.reasons[0][0] for f in bad)
    assert {f.key for f in bad} >= {"noise_std_rel=0.04,n_timepoints=12", "noise_std_rel=0.08,n_timepoints=5"}


def test_replay_r12_with_the_crashes_repaired_the_typed_noise_is_caught_by_coverage():
    fails, cal = _checks("r12_kinetics_tolerance_relaxed")
    assert not [f for f in fails if f.failed]
    faults = {q.label: q for q in cal.faults}
    assert set(faults) == {"weighted_Ea_lower .. weighted_Ea_upper", "weighted_ln_A_lower .. weighted_ln_A_upper"}
    worst = min(s.value for s in faults["weighted_Ea_lower .. weighted_Ea_upper"].settings)
    assert worst < 0.25
    assert faults["weighted_Ea_lower .. weighted_Ea_upper"].gross >= 10


def test_replay_r12_the_correct_fix_passes():
    fails, cal = _checks("r12_kinetics_noise_fixed")
    assert not cal.faults and cal.checked
    assert not [f for f in fails if f.fault]


def test_replay_r12_a_plan_truth_the_code_does_not_generate_with_is_caught():
    # r12's plan gives `ln_A_true` = 9.45 while its code generates the data with ln(1e7) = 16.1: measured against the
    # plan's truth, every interval misses, which is what a typed true value looks like
    _f, cal = _checks("r12_kinetics_noise_fixed", [
        known("weighted_ln_A_hat", "ln_A_true", interval=("weighted_ln_A_lower", "weighted_ln_A_upper"))])
    cover = next(q for q in cal.faults if q.kind == "coverage")
    assert max(s.value for s in cover.settings) == 0.0 and "ln_A_true = 9.45" in cover.truth
    assert "bias" in [q.kind for q in cal.faults], "the estimates are off the plan's truth by many of their spreads"


def test_replay_r2_wrong_two_step_errors_are_caught():
    fails, cal = _checks("r2_kinetics")
    faults = {(q.kind, q.label) for q in cal.faults}
    assert ("se_ratio", "two_step_ea") in faults and ("se_ratio", "two_step_ln_a") in faults
    ratio = next(q for q in cal.faults if q.kind == "se_ratio" and q.label == "two_step_ea")
    assert all(s.value > 5 for s in ratio.settings)
    assert not {label for _k, label in faults} & {"global_ea", "global_ln_a"}


@pytest.mark.parametrize("name", ["r14b_kinetics", "r6t_kinetics", "r8_kinetics", "s1_kinetics"])
def test_replay_clean_kinetics_quests_are_not_flagged(name):
    fails, cal = _checks(name)
    assert not [f for f in fails if f.fault]
    assert cal.checked and not cal.faults
    assert all(q.gross == 0 for q in cal.quantities)


def test_replay_r14b_coverages_are_as_the_paper_says():
    _f, cal = _checks("r14b_kinetics")
    by = {q.label: q for q in cal.quantities if q.kind == "coverage"}
    t = by["Ea_t_lower .. Ea_t_upper"]
    assert 0.92 <= min(s.value for s in t.settings) and max(s.value for s in t.settings) <= 0.96 and not t.fault
    normal = by["Ea_normal_lower .. Ea_normal_upper"]
    assert not normal.fault and max(s.value for s in normal.settings) < 0.9


def test_replay_e2_returns_only_coverage_flags_so_its_calibration_is_not_checked():
    # E2's run_trial returns 0/1 "covered" values and no interval or standard error of Ea: FI does not read the flags
    _f, cal = _checks("E2_kinetics")
    assert not cal.checked and cal.not_checked == [tc.NO_DECLARATION]
    # a plan that declared a standard error the code then does not return is a fault, not a silent pass
    _f, cal = _checks("E2_kinetics", [known("Ea_hat", "Ea_true_J_mol", se="Ea_se")])
    assert [q.kind for q in cal.faults] == ["missing"]


@pytest.mark.parametrize("name", ["s1_fdr", "E1_fdr", "r13_fdr"])
def test_replay_fdr_quests_have_no_truth_interval_and_say_so(name):
    fails, cal = _checks(name)
    assert not [f for f in fails if f.fault]
    assert not cal.checked and not cal.faults
    assert cal.not_checked


# ---- a coverage the analysis computes itself ----------------------------------------------------------------------------


def test_a_coverage_the_analysis_reports_is_recounted_from_the_trials():
    from statistics import NormalDist

    data = _load("r6t_kinetics")
    z95 = NormalDist().inv_cdf(0.975)
    # the trials report their 95% interval's two ends (here made from r6t's own estimates and standard errors)
    for c in data["cells"]:
        m = c["metrics"]
        if "weighted_ea" in m:
            e, se = m["weighted_ea"]["values"], m["weighted_ea_se"]["values"]
            m["ea_lo"] = {"values": [x - z95 * v for x, v in zip(e, se)], "trials": m["weighted_ea"]["trials"]}
            m["ea_hi"] = {"values": [x + z95 * v for x, v in zip(e, se)], "trials": m["weighted_ea"]["trials"]}
    spec = {**known("weighted_ea", "Ea_true_J_mol", interval=("ea_lo", "ea_hi"), ident="ea_coverage_95"),
            "kind": "proportion"}
    proto = _protocol(data, [spec])
    with_values = [c for c in data["cells"] if c["metrics"]]

    def coverage(c, z):
        e = c["metrics"]["weighted_ea"]["values"]
        s = c["metrics"]["weighted_ea_se"]["values"]
        return sum(abs(x - 50000.0) <= z * se for x, se in zip(e, s)) / len(e)

    z90 = NormalDist().inv_cdf(0.95)
    right = {c["key"]: {"ea_coverage_95": coverage(c, z95)} for c in with_values}
    assert tc.reported_coverage_mismatches(data["cells"], proto, right, data["thresholds"]) == []
    # the analysis used the 90% critical value: about 0.90 where the trials give about 0.95
    wrong = {c["key"]: {"ea_coverage_95": coverage(c, z90)} for c in with_values}
    found = tc.reported_coverage_mismatches(data["cells"], proto, wrong, data["thresholds"])
    assert len(found) == len(with_values) and all(f["fi"] - f["reported"] > 0.03 for f in found)
    # counts and totals, and a value at the top level (every setting together), are read too
    top = {"ea_coverage_95_count": 70, "ea_coverage_95_total": 100}
    assert tc.reported_coverage_mismatches(data["cells"], proto, top, data["thresholds"])[0]["where"] == "the whole study"
    # a mean metric is not a coverage, and nothing is judged without a declaration
    assert tc.reported_coverage_mismatches(data["cells"], _protocol(data, [{**spec, "kind": "mean"}]), wrong) == []
    assert tc.reported_coverage_mismatches(data["cells"], _protocol(data, []), wrong) == []
    # a trial that reports only a standard error makes no interval FI could count: nothing is recounted
    se_only = {**known("weighted_ea", "Ea_true_J_mol", se="weighted_ea_se", ident="ea_coverage_95"), "kind": "proportion"}
    assert tc.reported_coverage_mismatches(data["cells"], _protocol(data, [se_only]), wrong, data["thresholds"]) == []


def test_trials_the_time_limit_cut_off_are_not_failures_of_the_code():
    from core import trial_runner

    cells = [{"key": "a=1", "planned": 10, "failed": 10, "metrics": {}},
             {"key": "a=2", "planned": 10, "failed": 8, "metrics": {}}]
    reasons = {"a=1": Counter({trial_runner.OUT_OF_TIME: 10}),
               "a=2": Counter({trial_runner.OUT_OF_TIME: 4, "ZeroDivisionError: division by zero": 4})}
    fails = tc.failures(cells, reasons)
    assert [(f.key, f.failed, f.out_of_time, f.fault) for f in fails] == [("a=1", 0, 10, False), ("a=2", 4, 4, True)]
    assert "4 of 6 trials failed" in fails[1].says() and "time" not in fails[1].says()
    record = tc.summary(fails, None)
    assert record["out_of_time"] == [{"key": "a=1", "planned": 10, "trials": 10}, {"key": "a=2", "planned": 10, "trials": 4}]
