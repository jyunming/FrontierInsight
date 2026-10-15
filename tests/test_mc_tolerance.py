"""A known-answer check on a random quantity is judged on enough trials for its tolerance (core/mc_error.py).

The real case: the s1 false-discovery quest of 2026-10-15 (fixture ``tests/fixtures/mc_tolerance_s1_fdr``: its
simulation, its model package, its frozen protocol and the oracle record it left). Its check ``bh_fdr_upper_bound``
(expected 0 within 0.01 of ``max(0, fdr_bh - 0.04)`` at m=200, pi1=0.2) was judged on one trial, whose false discovery
proportion was 1/17: it measured 0.0188 and failed, while an independent 400,000-repeat simulation gives an FDR of
0.0401 +- 0.00007, inside the bound. The simulation was right and the check wrong.

The same simulation with a planted fault must still fail at the same number of trials: BH at twice its level, the
false discovery proportion doubled in the tally, no correction at all, and the wrong tail of the p-value. Everything
runs through the real trial runner in its own process; no model is called.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from core import accepted_checks as ac, in_run, mc_error as mc, oracle_check as oc, trial_runner
from core.execution import SharedInterpreterExecutor

FIXTURE = Path(__file__).parent / "fixtures" / "mc_tolerance_s1_fdr"
PROTOCOL = json.loads((FIXTURE / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
PROTOCOL = PROTOCOL.get("protocol", PROTOCOL)
ORACLES = {o["name"]: o for o in PROTOCOL["oracles"]}
BOUND = ORACLES["bh_fdr_upper_bound"]


def _quest(tmp_path: Path, *, simulate: dict[str, str] | None = None, model: dict[str, str] | None = None) -> Path:
    """A quest folder holding the s1 simulation, with each ``old -> new`` replacement planted in it."""
    root = tmp_path / "quest"
    shutil.copytree(FIXTURE / "code", root / "code", ignore=shutil.ignore_patterns("__pycache__"))
    for rel, edits in (("code/simulate.py", simulate), ("code/fdr_bonferroni_vs_bh/model.py", model)):
        if edits:
            path = root / rel
            text = path.read_text(encoding="utf-8")
            for old, new in edits.items():
                assert old in text, old
                text = text.replace(old, new)
            path.write_text(text, encoding="utf-8")
    return root


def _measure(root: Path, oracles: list[dict[str, Any]], **kw: Any) -> tuple[list[dict[str, Any]], list[str],
                                                                            list[dict[str, Any]], list[str]]:
    kw.setdefault("mean_names", mc.mean_names(PROTOCOL))
    kw.setdefault("in_each_run", in_run.of(PROTOCOL))
    checks, problems, _ = asyncio.run(trial_runner.measure_oracles(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", oracles,
        timeout_s=120, **kw))
    reported = {"checks": checks, "engine_measured": True}
    return checks, problems, oc.judged(oracles, reported), oc.problems(oracles, reported, 0)


# --- the s1 record, before -------------------------------------------------------------------------------------------


def test_the_s1_record_failed_the_bound_on_one_trial() -> None:
    record = json.loads((FIXTURE / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    (check,) = [c for c in record["attempts"][-1]["checks"] if c["name"] == "bh_fdr_upper_bound"]
    assert check["value"] == pytest.approx(1 / 17 - 0.04), "one trial's false discovery proportion, 1 of 17"
    assert "n_trials" not in check["returned"], "one run_trial, not a mean over repeats"
    assert record["status"] == "went_on_failing"
    assert BOUND["kind"] == "invariant" and "R=4000 repeats" in BOUND["check"]
    assert "fdr_bh" in mc.mean_names(PROTOCOL), "the plan declares fdr_bh as E[V/max(R,1)], a mean over repeats"


def test_judged_on_one_trial_the_bound_still_fails_as_it_did(tmp_path: Path) -> None:
    """``per_trial: true`` keeps a check on one trial: the s1 measurement is reproduced exactly (the same seed)."""
    one = {**BOUND, "per_trial": True}
    checks, problems, judged, found = _measure(_quest(tmp_path), [one])
    assert not problems and judged[0]["value"] == pytest.approx(0.0188235, abs=1e-6)
    assert judged[0]["passed_by_engine"] is False and "mc" not in checks[0]
    assert found and "0.0188235" in found[0]


# --- the s1 record, after --------------------------------------------------------------------------------------------


def test_the_s1_bound_is_judged_on_the_mean_over_enough_trials_and_passes(tmp_path: Path) -> None:
    checks, problems, judged, found = _measure(_quest(tmp_path), [BOUND])
    assert not problems and found == [], (problems, found)
    (j,) = judged
    record = j["mc"]
    assert j["passed_by_engine"] is True and j["value"] < 0.01
    assert record["read_as"] == "mean" and record["varies"] is True and record["precision"] == "planned"
    # The trials the spread asks for: the standard error a quarter of the tolerance or less.
    assert record["trials"] >= 100 and record["se"] <= 0.25 * 0.01 + 1e-12
    assert record["shape"] == "bound" and record["inner"] == "fdr_bh - 0.04"
    assert abs(record["inner_mean"]) < 3 * record["se"] + 0.002, "the mean FDR sits at about pi0*alpha = 0.04"
    assert record["limit_used"] == pytest.approx(0.01), "the plan's tolerance, unchanged"


def test_the_other_s1_checks_keep_their_verdicts(tmp_path: Path) -> None:
    """The fixed-seed per-trial invariant and the second implementation are judged on one trial as before; the
    special cases now carry the trials they were judged on."""
    checks, problems, judged, found = _measure(_quest(tmp_path), list(ORACLES.values()))
    verdicts = {j["name"]: j["passed_by_engine"] for j in judged}
    assert all(verdicts.values()) and not found, (verdicts, found, problems)
    by_name = {c["name"]: c for c in checks}
    assert "mc" not in by_name["bh_superset_of_bonferroni"] and "mc" not in by_name["bh_matches_scipy"]
    for name in ("bonferroni_power_exact", "uncorrected_fwer_null", "bh_fwer_all_null", "null_p_uniform",
                 "bh_fdr_upper_bound"):
        assert by_name[name]["mc"]["read_as"] == "mean", name


# --- planted faults: still fail, at the same number of trials --------------------------------------------------------

PLANTED = {
    "bh_at_twice_its_level": ({"model.bh_stepup(p, ALPHA)": "model.bh_stepup(p, 2 * ALPHA)"}, None),
    "fdp_doubled_in_the_tally": ({'out["fdr_" + s] = float(fdp)': 'out["fdr_" + s] = float(2 * fdp)'}, None),
    "no_correction_called_bh": ({'masks[meth] = model.bh_stepup(p, ALPHA)': 'masks[meth] = p <= ALPHA'}, None),
    "wrong_tail_of_the_p_value": (None, {"return norm.sf(z)": "return norm.cdf(z)"}),
}
SAME_TRIALS = 300


@pytest.mark.parametrize("fault", sorted(PLANTED))
def test_a_planted_fault_still_fails_the_bound_at_the_same_number_of_trials(tmp_path: Path, fault: str) -> None:
    simulate, model = PLANTED[fault]
    _c, _p, right, _f = _measure(_quest(tmp_path / "right"), [BOUND], most_trials=SAME_TRIALS)
    _c, problems, wrong, found = _measure(_quest(tmp_path / "wrong", simulate=simulate, model=model), [BOUND],
                                          most_trials=SAME_TRIALS)
    assert right[0]["passed_by_engine"] is True
    assert wrong[0]["passed_by_engine"] is False and found, (fault, wrong, problems)
    assert wrong[0]["mc"]["trials"] <= SAME_TRIALS and right[0]["mc"]["trials"] <= SAME_TRIALS
    assert not wrong[0]["within_error"], "beyond the Monte Carlo error: a fault, not noise"


EQUALS_PI0_Q = {"name": "bh_fdr_equals_pi0_q", "kind": "special_case",
                "check": "BH's FDR at m=200, pi1=0.2 equals pi0*q for independent p-values", "expected": 0.04,
                "tolerance": 0.004, "case": {"m": 200, "pi1": 0.2}, "measure": "fdr_bh",
                "reference": "derivation: for independent continuous p-values FDR_BH = pi0*q exactly"}


def test_bh_at_pi0_q_passes_a_bound_but_fails_a_check_of_its_value(tmp_path: Path) -> None:
    """BH run at pi0*q has an FDR of pi0**2*q = 0.032, inside the bound 0.04: a bound cannot see a procedure that is too
    strict. A check of the value itself (BH attains pi0*q exactly) does."""
    planted = {"model.bh_stepup(p, ALPHA)": "model.bh_stepup(p, 0.8 * ALPHA)"}
    _c, _p, bound, _f = _measure(_quest(tmp_path / "bound", simulate=planted), [BOUND])
    assert bound[0]["passed_by_engine"] is True
    _c, _p, right, _f = _measure(_quest(tmp_path / "right"), [EQUALS_PI0_Q])
    _c, _p, wrong, found = _measure(_quest(tmp_path / "wrong", simulate=planted), [EQUALS_PI0_Q])
    assert right[0]["passed_by_engine"] is True and right[0]["mc"]["precision"] == "planned"
    assert wrong[0]["passed_by_engine"] is False and found and not wrong[0]["within_error"]


# --- synthetic: a rate over repeats ----------------------------------------------------------------------------------

RATE = '''
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    k = int(cell["k"])
    return {"rate": sum(rng.random() < P for _ in range(k)) / k}
'''
RATE_CHECK = {"name": "null_rate", "kind": "special_case", "check": "the rejection rate under the null", "expected": 0.05,
              "tolerance": 0.01, "case": {"k": 100}, "measure": "rate",
              "reference": "derivation: a level-0.05 test rejects a true null with probability 0.05"}


def _rate_quest(tmp_path: Path, p: float) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(f"P = {p}\n" + RATE, encoding="utf-8")
    return root


def _rate(tmp_path: Path, p: float, oracle: dict[str, Any] = RATE_CHECK, **kw: Any):
    kw.setdefault("mean_names", frozenset())
    kw.setdefault("in_each_run", None)
    return _measure(_rate_quest(tmp_path, p), [oracle], **kw)


def test_a_correct_random_simulation_passes_on_enough_trials(tmp_path: Path) -> None:
    checks, problems, judged, found = _rate(tmp_path, 0.05)
    (j,) = judged
    assert not problems and not found and j["passed_by_engine"] is True
    assert j["mc"]["trials"] > mc.PILOT_TRIALS, "the first batch's spread asked for more trials"
    assert j["mc"]["se"] <= 0.25 * 0.01 + 1e-12 and j["mc"]["precision"] == "planned"
    assert checks[0]["returned"]["n_trials"] == j["mc"]["trials"]


def test_a_twice_biased_random_simulation_fails_at_the_same_number_of_trials(tmp_path: Path) -> None:
    _c, _p, right, _f = _rate(tmp_path / "right", 0.05, most_trials=60)
    _c, _p, wrong, found = _rate(tmp_path / "wrong", 0.10, most_trials=60)
    assert right[0]["passed_by_engine"] is True and right[0]["mc"]["trials"] == 60
    assert wrong[0]["passed_by_engine"] is False and wrong[0]["mc"]["trials"] == 60 and found
    assert not wrong[0]["within_error"]


def test_a_tolerance_the_trials_cannot_resolve_is_judged_at_their_error_and_said_so(tmp_path: Path) -> None:
    tight = {**RATE_CHECK, "tolerance": 0.001}
    _c, _p, right, _f = _rate(tmp_path / "right", 0.05, tight, most_trials=40)
    _c, _p, wrong, found = _rate(tmp_path / "wrong", 0.10, tight, most_trials=40)
    record = right[0]["mc"]
    assert record["precision"] == "widened" and record["limit_used"] == pytest.approx(3 * record["se"])
    assert right[0]["passed_by_engine"] is True and right[0]["limit"] == 0.001, "the plan's tolerance is kept as written"
    assert "judged at ±" in oc.analysis_note(right) and "three standard errors" in oc.analysis_note(right)
    assert wrong[0]["passed_by_engine"] is False and found, "a 2x bias is far outside three standard errors"


def test_too_few_trials_to_know_their_spread_never_widen_the_tolerance(tmp_path: Path) -> None:
    tight = {**RATE_CHECK, "tolerance": 0.001}
    _c, _p, judged, _f = _rate(tmp_path, 0.05, tight, most_trials=5)
    record = judged[0]["mc"]
    assert record["trials"] == 5 and record["precision"] == "few_trials"
    assert record["limit_used"] == 0.001


def test_a_check_whose_trials_all_agree_is_judged_as_before(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(
        "def run_trial(cell, trial_id, seed):\n    return {'x': 2.0 * cell['a']}\n", encoding="utf-8")
    check = {"name": "double", "kind": "special_case", "expected": 3.0, "tolerance": 1e-9, "case": {"a": 1.5},
             "measure": "x"}
    checks, problems, judged, found = _measure(root, [check], mean_names=frozenset(), in_each_run=None)
    assert not problems and not found and judged[0]["passed_by_engine"] is True
    assert "mc" not in checks[0] and "mc" not in judged[0], "deterministic: unchanged"


def test_an_invariant_on_a_declared_mean_is_read_as_a_mean_and_one_on_anything_else_per_trial() -> None:
    means = mc.mean_names(PROTOCOL)
    assert mc.reads_a_mean(BOUND, means)
    assert not mc.reads_a_mean(ORACLES["bh_superset_of_bonferroni"], means), "a per-trial count: one trial"
    assert not mc.reads_a_mean({**BOUND, "per_trial": True}, means), "per_trial: true always means one trial"
    assert not mc.reads_a_mean(BOUND, frozenset()), "with no metrics declared, an invariant stays per trial"
    assert mc.reads_a_mean(ORACLES["bonferroni_power_exact"], frozenset()), "a special case is a number of the setting"
    given = {"metrics": [{"id": "size", "estimand": "E[size | outbreak]", "kind": "mean", "unit": "a run",
                          "given": "outbreak"},
                         {"id": "outbreak", "estimand": "P(outbreak)", "kind": "proportion", "unit": "a run"}]}
    assert mc.mean_names(given) == frozenset({"outbreak"}), "a mean over a subset of the runs is not a plain mean"


# --- the rules, on numbers -------------------------------------------------------------------------------------------


def test_the_band_belongs_to_the_quantity_inside_a_bound_or_a_distance() -> None:
    assert mc.inner("max(0, fdr_bh - 0.04)") == ("fdr_bh - 0.04", "bound")
    assert mc.inner("max(x - 1, 0)") == ("x - 1", "bound")
    assert mc.inner("abs(a - b)") == ("a - b", "abs")
    assert mc.inner("max(a, b)") == ("max(a, b)", "")
    rows = [{"x": v} for v in (0.0, 0.1) * 50]
    got = mc.summarise(rows, "max(0, x - 0.04)")
    assert got["value"] == pytest.approx(0.01) and got["inner_mean"] == pytest.approx(0.01)
    sd = (sum((r["x"] - 0.05) ** 2 for r in rows) / 99) ** 0.5
    assert got["se"] == pytest.approx(sd / 10), "one name: its standard deviation over sqrt(n)"


def test_a_difference_of_two_paired_names_uses_their_covariance() -> None:
    rows = [{"a": v, "b": v + 0.5} for v in (0.0, 1.0) * 20]
    got = mc.summarise(rows, "a - b")
    assert got["value"] == pytest.approx(-0.5) and got["se"] == pytest.approx(0.0, abs=1e-9)


def test_a_failure_by_less_than_its_monte_carlo_error_is_not_decided() -> None:
    record = {"varies": True, "trials": 400, "se": 0.0025}
    assert mc.limit_used(0.01, record) == (0.01, "planned")
    assert mc.within_error(0.0565, 0.04, 0.01, record), "0.0165 is outside 0.01 by less than 3 x 0.0025"
    assert not mc.within_error(0.0675, 0.04, 0.01, record), "0.0275 is not"
    assert not mc.within_error(0.045, 0.04, 0.01, record), "a pass is not a failure"
    assert not mc.within_error(0.0565, 0.04, 0.01, {**record, "varies": False}), "a deterministic check has no noise"


def test_the_paper_is_not_told_a_check_within_its_error_shows_a_broken_method() -> None:
    noisy = {"name": "bh_fdr_upper_bound", "expected": 0.0, "limit": 0.01, "tolerance": 0.01,
             "tolerance_mode": "absolute", "case": {"m": 200, "pi1": 0.2}, "measure": "max(0, fdr_bh - 0.04)",
             "measured": 0.0188, "by": ac.AUTOMATIC, "within_mc_error": True,
             "mc_words": "the mean over 20 trials, Monte Carlo error ±0.0098 (one standard error)"}
    text = ac.failing_disclosure({"status": ac.WENT_ON, "went_on": [noisy]})
    assert "not decided" in text and "Monte Carlo" in text
    assert "FAILED" not in text and "do not" in text.lower()
    broken = {**noisy, "within_mc_error": False}
    assert "FAILED" in ac.failing_disclosure({"status": ac.WENT_ON, "went_on": [broken]})
    assert "not decided" in ac.gap(noisy) and "unconfirmed" in ac.gap(noisy)
