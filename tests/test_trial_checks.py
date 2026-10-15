"""FI's own checks of what the trials gave (core/trial_checks.py): failed trials per setting, and the reported intervals and
standard errors against the true values the simulation was given.

The replay part reads tests/fixtures/replay/trial_checks/: the trial records of real quests, cut to a few settings with
every trial's values (the quests' own folders are not touched). r12 and r2 never ran their study, so their records were
made by running their own code through FI's trial harness on their grid (100 trials a setting): r12 as delivered (a noise
level typed into the fitting function, and a solver tolerance that fails most noisy trials), the same with only the
tolerance relaxed (what a repair of the crashes alone leaves), and with the noise passed in (the correct fix of study 2);
r2 as delivered (two-step weights on the wrong scale).

How they were made (to rebuild them): ``core.trial_runner.run_trials`` with FI's own harness, base seed 0, paired, 100
trials a setting, the study's own thresholds; r12 on its 5 x 5 grid (noise 0.005..0.08 x 5..32 time points) from study 2's
``orig/`` (= the delivered code), ``r12_kinetics_tolerance_relaxed`` = ``orig/`` with one hand edit in
``kinetics_parameter_recovery/model.py`` (the gradient check ``abs(float(gradient[0])) >= solver_gtol`` made
``>= 1e-6``), ``r12_kinetics_noise_fixed`` = study 2's ``fixed/`` with the same edit; r2 on noise 0, 0.025, 0.05, 0.1 x
6, 16, 36 time points. The other fixtures are cut from the quests' ``raw/trials.json`` and ``raw/ledger.jsonl``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from core import trial_checks as tc

FIX = Path(__file__).parent / "fixtures" / "replay" / "trial_checks"


def _load(name: str) -> dict:
    with gzip.open(FIX / f"{name}.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _checks(name: str):
    data = _load(name)
    from collections import Counter

    reasons = {k: Counter(v) for k, v in (data.get("reasons") or {}).items()}
    fails = tc.failures(data["cells"], reasons)
    cal = tc.calibration(data["cells"], data.get("protocol"), data.get("thresholds"))
    return fails, cal


def _cells(per: dict[str, dict[str, list[float]]], **head) -> list[dict]:
    out = []
    for key, metrics in per.items():
        n = max(len(v) for v in metrics.values())
        out.append({"key": key, "cell": {}, "planned": head.get("planned", n), "failed": head.get("failed", 0),
                    "metrics": {m: {"values": v, "trials": list(range(len(v)))} for m, v in metrics.items()}})
    return out


# ---- names ---------------------------------------------------------------------------------------------------------------


def test_names_split_into_words_and_parts():
    assert tc.tokens("weighted_Ea_hat") == ["weighted", "ea", "hat"]
    assert tc.tokens("lnA_true") == ["ln", "a", "true"]
    assert tc._role("two_step_var_ea") == ("var", ("two", "step", "ea"))
    assert tc._role("k_se_0_s_inv") == ("se", ("k", "0", "s", "inv"))
    assert tc._role("Ea_t_lower") == ("lower", ("ea", "t"))
    assert tc._role("ea_weighted_covered_z") == ("cover", ("ea", "weighted", "z"))
    # a covariance is not a coverage
    assert tc._role("two_step_cov_ea_ln_a")[0] == ""


def test_an_estimate_pairs_with_its_error_only_by_the_same_words():
    rng = __import__("random").Random(1)
    est = [10 + rng.gauss(0, 1) for _ in range(200)]
    cells = _cells({"s": {"x_hat": est, "x_se": [1.0] * 200, "y_se": [1.0] * 200}})
    cal = tc.calibration(cells, None, {"x_true": 10.0})
    assert [q.label for q in cal.quantities] == ["x_hat"]
    assert not cal.faults


def test_a_standard_error_far_too_small_is_a_fault_and_a_right_one_is_not():
    rng = __import__("random").Random(2)
    cells = _cells({f"s{i}": {"x_hat": [5 + rng.gauss(0, 1) for _ in range(100)], "x_se": [0.2] * 100} for i in range(4)})
    cal = tc.calibration(cells, None, {"x_true": 5.0})
    assert [q.label for q in cal.faults] == ["x_hat"]
    cells = _cells({f"s{i}": {"x_hat": [5 + rng.gauss(0, 1) for _ in range(100)], "x_se": [1.0] * 100} for i in range(4)})
    assert not tc.calibration(cells, None, {"x_true": 5.0}).faults


def test_without_a_declared_true_value_the_error_is_not_checked_and_the_record_says_why():
    rng = __import__("random").Random(3)
    cells = _cells({"s": {"x_hat": [5 + rng.gauss(0, 1) for _ in range(100)], "x_se": [0.2] * 100}})
    cal = tc.calibration(cells, None, {})
    assert not cal.quantities
    assert any("no true value" in line for line in cal.not_checked)


def test_a_true_value_far_outside_the_estimates_is_not_taken_for_theirs():
    rng = __import__("random").Random(4)
    cells = _cells({"s": {"ln_a_hat": [13 + rng.gauss(0, 0.1) for _ in range(100)], "ln_a_se": [0.01] * 100}})
    cal = tc.calibration(cells, None, {"true_A": 1e6})
    assert not cal.quantities and any("outside" in line for line in cal.not_checked)


def test_coverage_needs_a_stated_level_and_a_small_miss_is_only_reported():
    hits = [1.0] * 86 + [0.0] * 14  # a normal interval with few degrees of freedom: about 0.86
    cells = _cells({f"s{i}": {"x_covered": hits} for i in range(5)})
    assert not tc.calibration(cells, None, {}).quantities
    cal = tc.calibration(cells, None, {"ci_level": 0.95})
    q = cal.quantities[0]
    assert q.level == 0.95 and not q.fault and q.below == 5 and q.gross == 0
    low = [1.0] * 30 + [0.0] * 70
    cal = tc.calibration(_cells({f"s{i}": {"x_covered": low} for i in range(5)}), None, {"ci_level": 0.95})
    assert cal.faults and cal.faults[0].gross == 5


def test_nothing_to_check_is_said():
    cells = _cells({"s": {"power": [0.5] * 50, "fdr": [0.04] * 50}})
    cal = tc.calibration(cells, None, {"alpha": 0.05})
    assert not cal.checked
    assert cal.not_checked and "no interval or standard error" in cal.not_checked[-1]


def test_failed_trials_per_setting():
    cells = [{"key": "a=1", "planned": 100, "failed": 60, "metrics": {}},
             {"key": "a=2", "planned": 100, "failed": 6, "metrics": {}},
             {"key": "a=3", "planned": 100, "failed": 0, "metrics": {}},
             {"key": "a=3,m=x", "planned": 100, "failed": 60, "level": {"m": "x"}, "metrics": {}}]
    fails = tc.failures(cells)
    assert [(f.key, f.fault, f.stated) for f in fails] == [("a=1", True, True), ("a=2", False, True), ("a=3", False, False)]


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
    assert set(faults) == {"weighted_Ea_coverage", "weighted_ln_A_coverage"}
    worst = min(s.value for s in faults["weighted_Ea_coverage"].settings)
    assert worst < 0.25
    assert faults["weighted_Ea_coverage"].gross == 12


def test_replay_r12_the_correct_fix_passes():
    fails, cal = _checks("r12_kinetics_noise_fixed")
    assert not cal.faults and cal.checked
    assert not [f for f in fails if f.fault]


def test_replay_r2_wrong_two_step_errors_are_caught():
    fails, cal = _checks("r2_kinetics")
    faults = {q.label: q for q in cal.faults}
    assert set(faults) == {"two_step_ea", "two_step_ln_a"}
    assert all(s.value > 5 for s in faults["two_step_ea"].settings)
    ok = {q.label for q in cal.quantities if not q.fault}
    assert {"global_ea", "global_ln_a"} <= ok


@pytest.mark.parametrize("name", ["E2_kinetics", "r14b_kinetics", "r6t_kinetics", "r8_kinetics", "s1_kinetics"])
def test_replay_clean_kinetics_quests_are_not_flagged(name):
    fails, cal = _checks(name)
    assert not [f for f in fails if f.fault]
    assert cal.checked and not cal.faults
    assert all(q.gross == 0 for q in cal.quantities)


def test_replay_e2_and_r14b_coverages_are_as_the_papers_say():
    _f, cal = _checks("E2_kinetics")
    ea = {q.label: q for q in cal.quantities}
    for name in ("ea_weighted_covered_z", "ea_covered"):
        vals = [s.value for s in ea[name].settings]
        assert 0.93 <= min(vals) and max(vals) <= 0.96
    z = ea["ea_unweighted_covered_z"]
    assert not z.fault and z.below == z.judged  # about 0.86: reported, not a fault
    _f, cal = _checks("r14b_kinetics")
    by = {q.label: q for q in cal.quantities}
    t = by["Ea_t_lower .. Ea_t_upper"]
    assert 0.92 <= min(s.value for s in t.settings) and max(s.value for s in t.settings) <= 0.96 and not t.fault
    normal = by["Ea_normal_lower .. Ea_normal_upper"]
    assert not normal.fault and max(s.value for s in normal.settings) < 0.9


@pytest.mark.parametrize("name", ["s1_fdr", "E1_fdr", "r13_fdr"])
def test_replay_fdr_quests_have_no_truth_interval_and_say_so(name):
    fails, cal = _checks(name)
    assert not [f for f in fails if f.fault]
    assert not cal.checked and not cal.faults
    assert cal.not_checked
