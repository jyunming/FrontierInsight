"""A known-answer check on a random quantity is judged on enough trials for its tolerance (core/mc_error.py).

The real case: the s1 false-discovery quest of 2026-10-15 (fixture ``tests/fixtures/mc_tolerance_s1_fdr``: its
simulation, its model package, its frozen protocol and the oracle record it left). Its check ``bh_fdr_upper_bound``
(expected 0 within 0.01 of ``max(0, fdr_bh - 0.04)`` at m=200, pi1=0.2) was judged on one trial, whose false discovery
proportion was 1/17: it measured 0.0188 and failed, while an independent 400,000-repeat simulation gives an FDR of
0.0401 +- 0.00007, inside the bound. The simulation was right and the check wrong. The bound is on the false discovery
RATE, an average over repeats, but the plan wrote it as an ``invariant``: judged as written it is a rule of every trial
and still fails; declared ``per_trial: false`` (what the plan prompt now asks for such a bound) it is judged on the mean.

The same simulation with a planted fault must still fail: BH at twice its level, the false discovery proportion doubled
in the tally, no correction at all, and the wrong tail of the p-value. The review of the first version of this fix
(C:/fi_compare/reviews/review_tol.md) found seven ways a wrong simulation passed or a check cost far too much; each is a
test below (A5, B, C, D, E, F, G). Everything runs through the real trial runner in its own process; no model is called.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
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
#: The bound as the plan should write it: a bound on the false discovery RATE, an average over repeats, says so.
MEAN_BOUND = {**BOUND, "per_trial": False}
#: The s1 checks with the bound written that way.
S1_CHECKS = [MEAN_BOUND if o is BOUND else o for o in ORACLES.values()]


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


def _measure(root: Path, oracles: list[dict[str, Any]], *, timeout_s: int = 120,
             **kw: Any) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[str]]:
    kw.setdefault("mean_names", mc.mean_names(PROTOCOL))
    kw.setdefault("in_each_run", in_run.of(PROTOCOL))
    checks, problems, _ = asyncio.run(trial_runner.measure_oracles(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", oracles,
        timeout_s=timeout_s, **kw))
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


def test_the_s1_bound_declared_on_the_average_is_judged_on_the_mean_over_enough_trials_and_passes(
        tmp_path: Path) -> None:
    checks, problems, judged, found = _measure(_quest(tmp_path), [MEAN_BOUND])
    assert not problems and found == [], (problems, found)
    (j,) = judged
    record = j["mc"]
    assert j["passed_by_engine"] is True and j["value"] < 0.01
    assert record["read_as"] == "mean" and record["varies"] is True and record["verdict"] == "pass"
    # Trials are added until three standard errors around the mean fit inside the plan's own tolerance.
    assert record["trials"] >= 30 and abs(record["inner_mean"]) + 3 * record["se"] <= 0.01
    assert record["shape"] == "bound" and record["inner"] == "fdr_bh - 0.04"
    assert abs(record["inner_mean"]) < 3 * record["se"] + 0.002, "the mean FDR sits at about pi0*alpha = 0.04"


def test_the_other_s1_checks_keep_their_verdicts(tmp_path: Path) -> None:
    """The invariant and the second implementation are judged on every trial (the worst counts); the special cases
    and the bound on the average carry the trials they were judged on."""
    checks, problems, judged, found = _measure(_quest(tmp_path), S1_CHECKS)
    verdicts = {j["name"]: j["passed_by_engine"] for j in judged}
    assert all(verdicts.values()) and not found, (verdicts, found, problems)
    by_name = {c["name"]: c for c in checks}
    for name in ("bonferroni_power_exact", "uncorrected_fwer_null", "bh_fwer_all_null", "null_p_uniform",
                 "bh_fdr_upper_bound"):
        assert by_name[name]["mc"]["read_as"] == "mean", name


@pytest.mark.asyncio
async def test_the_s1_gate_passes_every_check_with_no_repair_and_no_failed_check_for_the_paper(tmp_path: Path) -> None:
    """The oracle gate itself, on the s1 simulation and the s1 protocol's checks, metrics and values compared in each
    run: every check passes on the first measurement, no repair is asked for, and nothing goes to the paper as failed."""
    from core import plan
    from core.engine import Engine
    from tests.test_known_answer_self_checks import _config, _Model

    engine = Engine(_config(tmp_path))
    model = _Model()
    engine._client = model
    protocol = {k: PROTOCOL[k] for k in ("grid", "in_each_run", "metrics", "thresholds")}
    protocol["oracles"] = S1_CHECKS
    design = {"hypothesis": "BH keeps its FDR at pi0*q", "protocol": protocol}
    text = plan.render(engine.config.topic, {}, design)
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="user")
    shutil.copytree(FIXTURE / "code", engine.quest_root / "code", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    engine.executor = SharedInterpreterExecutor(python_version="3.11")
    engine._trial_mode = True
    await engine._oracle_gate({"topic": engine.config.topic, "iteration": 0, "design": design}, sys.executable, None,
                              engine.quest_root / "code" / "simulate.py")
    record = json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    judged = {j["name"]: j for j in record["attempts"][-1]["judged"]}
    assert record["status"] == "ok", record.get("problems")
    assert model.repairs == [] and len(record["attempts"]) == 1
    assert judged["bh_fdr_upper_bound"]["passed_by_engine"] is True and judged["bh_fdr_upper_bound"]["mc"]["trials"] > 100
    assert ac.failing_disclosure(record) == ""


# --- planted faults: still fail, at the same number of trials --------------------------------------------------------

PLANTED = {
    "bh_at_twice_its_level": {"model.bh_stepup(p, ALPHA)": "model.bh_stepup(p, 2 * ALPHA)"},
    "fdp_doubled_in_the_tally": {'out["fdr_" + s] = float(fdp)': 'out["fdr_" + s] = float(2 * fdp)'},
    "no_correction_called_bh": {'masks[meth] = model.bh_stepup(p, ALPHA)': 'masks[meth] = p <= ALPHA'},
}
SAME_TRIALS = 200


@pytest.mark.parametrize("fault", sorted(PLANTED))
def test_a_planted_fault_still_fails_the_bound_within_the_same_number_of_trials(tmp_path: Path, fault: str) -> None:
    _c, _p, right, _f = _measure(_quest(tmp_path / "right"), [MEAN_BOUND], most_trials=SAME_TRIALS)
    _c, problems, wrong, found = _measure(_quest(tmp_path / "wrong", simulate=PLANTED[fault]), [MEAN_BOUND],
                                          most_trials=SAME_TRIALS)
    assert right[0]["passed_by_engine"] is True and right[0]["mc"]["verdict"] == "pass"
    assert wrong[0]["passed_by_engine"] is False and found, (fault, wrong, problems)
    assert wrong[0]["mc"]["verdict"] == "fail", "beyond the Monte Carlo error: a fault, decided by the trials"
    assert wrong[0]["mc"]["trials"] <= SAME_TRIALS and right[0]["mc"]["trials"] <= SAME_TRIALS


def test_the_wrong_tail_of_the_p_value_is_invisible_to_the_bound_and_caught_by_the_power_check(tmp_path: Path) -> None:
    """With the p-value's lower tail the real effects are never rejected: every rejection is a null, BH still keeps the
    FDR at about pi0*q, and the bound passes, rightly. The power check fails."""
    planted = {"return norm.sf(z)": "return norm.cdf(z)"}
    power = ORACLES["bonferroni_power_exact"]
    _c, _p, bound, _f = _measure(_quest(tmp_path / "bound", model=planted), [MEAN_BOUND])
    assert bound[0]["passed_by_engine"] is True
    _c, _p, right, _f = _measure(_quest(tmp_path / "right"), [power], most_trials=SAME_TRIALS)
    _c, _p, wrong, found = _measure(_quest(tmp_path / "wrong", model=planted), [power], most_trials=SAME_TRIALS)
    assert right[0]["passed_by_engine"] is True and right[0]["mc"]["trials"] <= SAME_TRIALS
    # No real effect is ever rejected: the power is 0 in every trial (FI runs more, as for a rare outcome) and fails.
    assert wrong[0]["passed_by_engine"] is False and found and wrong[0]["value"] == 0.0
    assert wrong[0]["mc"]["verdict"] == "fail"


EQUALS_PI0_Q = {"name": "bh_fdr_equals_pi0_q", "kind": "special_case",
                "check": "BH's FDR at m=200, pi1=0.2 equals pi0*q for independent p-values", "expected": 0.04,
                "tolerance": 0.004, "case": {"m": 200, "pi1": 0.2}, "measure": "fdr_bh",
                "reference": "derivation: for independent continuous p-values FDR_BH = pi0*q exactly"}


def test_bh_at_pi0_q_passes_a_bound_but_fails_a_check_of_its_value(tmp_path: Path) -> None:
    """BH run at pi0*q has an FDR of pi0**2*q = 0.032, inside the bound 0.04: a bound cannot see a procedure that is too
    strict. A check of the value itself (BH attains pi0*q exactly) does."""
    planted = {"model.bh_stepup(p, ALPHA)": "model.bh_stepup(p, 0.8 * ALPHA)"}
    _c, _p, bound, _f = _measure(_quest(tmp_path / "bound", simulate=planted), [MEAN_BOUND])
    assert bound[0]["passed_by_engine"] is True
    _c, _p, right, _f = _measure(_quest(tmp_path / "right"), [EQUALS_PI0_Q])
    _c, _p, wrong, found = _measure(_quest(tmp_path / "wrong", simulate=planted), [EQUALS_PI0_Q])
    assert right[0]["passed_by_engine"] is True and right[0]["mc"]["verdict"] == "pass"
    assert wrong[0]["passed_by_engine"] is False and found and wrong[0]["mc"]["verdict"] == "fail"


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
    assert abs(j["value"] - 0.05) + 3 * j["mc"]["se"] <= 0.01 and j["mc"]["verdict"] == "pass"
    assert checks[0]["returned"]["n_trials"] == j["mc"]["trials"]


def test_a_twice_biased_random_simulation_fails(tmp_path: Path) -> None:
    larger = {**RATE_CHECK, "case": {"k": 400}}
    _c, _p, right, _f = _rate(tmp_path / "right", 0.05, larger)
    _c, _p, wrong, found = _rate(tmp_path / "wrong", 0.10, larger)
    assert right[0]["passed_by_engine"] is True and right[0]["mc"]["verdict"] == "pass"
    assert wrong[0]["passed_by_engine"] is False and found and wrong[0]["mc"]["verdict"] == "fail"
    assert wrong[0]["mc"]["trials"] <= mc.PILOT_TRIALS, "a 2x bias is decided by the first batch"


@pytest.mark.parametrize("most", [5, 40])
def test_a_tolerance_the_trials_cannot_resolve_is_not_decided_and_never_widened(tmp_path: Path, most: int) -> None:
    tight = {**RATE_CHECK, "tolerance": 0.001}
    _c, _p, right, found_right = _rate(tmp_path / "right", 0.05, tight, most_trials=most)
    _c, _p, wrong, found = _rate(tmp_path / "wrong", 0.10, tight, most_trials=most)
    record = right[0]["mc"]
    assert record["trials"] == most and record["verdict"] == "undecided"
    assert right[0]["passed_by_engine"] is None and right[0]["undecided"] is True and right[0]["limit"] == 0.001
    assert found_right and "not decided" in found_right[0]
    assert "not decided" in oc.analysis_note(right)
    if most == 40:
        assert wrong[0]["passed_by_engine"] is False and found, "a 2x bias is far outside three standard errors"


RARE = '''
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    return {"event": 1.0 if rng.random() < P else 0.0, "noise": rng.random()}
'''
RARE_CHECK = {"name": "rare_event_rate", "kind": "special_case", "check": "the rate of a rare event", "expected": 0.02,
              "tolerance": 0.01, "case": {"n": 1}, "measure": "event",
              "reference": "derivation: the event has probability 0.02 by construction"}


@pytest.mark.parametrize("p, passes", [(0.02, True), (0.04, False)])
def test_a_rare_event_that_did_not_happen_in_the_first_batch_is_not_judged_on_trial_0(
        tmp_path: Path, p: float, passes: bool) -> None:
    """20 trials of an event with probability 0.02 are all 0 two times in three. The simulation is random (another number
    it returns differs), so the check is sized from what a proportion at its expected value spreads and judged on the
    mean; a rate twice the expected one still fails."""
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(f"P = {p}\n" + RARE, encoding="utf-8")
    checks, problems, judged, found = _measure(root, [RARE_CHECK], mean_names=frozenset(), in_each_run=None)
    (j,) = judged
    assert j["mc"]["trials"] > 1000 and j["passed_by_engine"] is passes, j
    assert bool(found) is not passes and j["mc"]["verdict"] == ("pass" if passes else "fail")


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


def test_a_rule_is_read_on_a_mean_only_when_the_check_says_so() -> None:
    """A rule each trial keeps is per trial unless the check declares ``per_trial: false``, whatever the protocol's
    metrics are; a quantity is a mean unless it declares ``per_trial: true``."""
    means = mc.mean_names(PROTOCOL)
    assert not mc.reads_a_mean(BOUND, means), "declared metrics are not a declaration that the check is on a mean"
    assert mc.reads_a_mean(MEAN_BOUND, means) and mc.reads_a_mean(MEAN_BOUND, frozenset())
    assert mc.reads_a_mean({**BOUND, "per_trial": "false"}, frozenset())
    assert not mc.reads_a_mean(ORACLES["bh_superset_of_bonferroni"], means), "a per-trial count"
    assert not mc.reads_a_mean({**BOUND, "per_trial": True}, means), "per_trial: true always means one trial"
    assert mc.reads_a_mean(ORACLES["bonferroni_power_exact"], frozenset()), "a special case is a number of the setting"
    assert not mc.reads_a_mean({**ORACLES["bonferroni_power_exact"], "per_trial": True}, frozenset())
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


def test_the_trials_decide_a_check_only_when_three_standard_errors_put_it_on_one_side_of_its_tolerance() -> None:
    assert mc.decide(0.0565, 0.04, 0.01, 0.0025) is None, "0.0165 is outside 0.01 by less than 3 x 0.0025"
    assert mc.decide(0.0675, 0.04, 0.01, 0.0025) == "fail", "0.0275 - 0.0075 is outside"
    assert mc.decide(0.045, 0.04, 0.01, 0.0025) is None, "0.005 + 0.0075 does not fit inside 0.01"
    assert mc.decide(0.045, 0.04, 0.01, 0.001) == "pass"
    assert mc.decide(0.05, 0.05, 1e-14, 0.001) is None, "a tolerance of 0 is never widened"
    assert mc.decide(0.05, 0.05, 1e-14, 0.0) == "pass", "exactly equal, with no spread"
    # A bound max(0, X) expecting 0 is judged on its one side: X far below 0 passes.
    assert mc.decide(-0.03, 0.0, 0.01, 0.005, shape="bound") == "pass"
    assert mc.decide(0.03, 0.0, 0.01, 0.005, shape="bound") is None
    assert mc.decide(0.04, 0.0, 0.01, 0.005, shape="bound") == "fail"


def test_the_paper_says_a_check_that_could_not_be_decided_was_not_decided_in_neutral_words() -> None:
    """An undecided check is said plainly as not decided and not validated against: neither a hint that the method is
    wrong nor a reassurance that it is right."""
    undecided = {"name": "bh_fdr_upper_bound", "expected": 0.0, "limit": 0.01, "tolerance": 0.01,
                 "tolerance_mode": "absolute", "case": {"m": 200, "pi1": 0.2}, "measure": "max(0, fdr_bh - 0.04)",
                 "measured": 0.0188, "by": ac.AUTOMATIC, "undecided": True,
                 "mc_words": "the mean over 20 trials, Monte Carlo error ±0.0098 (one standard error)"}
    text = ac.failing_disclosure({"status": ac.WENT_ON, "went_on": [undecided]})
    assert "not decided" in text and "not validated" in text, text
    assert "FAILED" not in text and "imply" not in text and "may be wrong" not in text, text
    broken = {**undecided, "undecided": False}
    assert "FAILED" in ac.failing_disclosure({"status": ac.WENT_ON, "went_on": [broken]})
    assert "not decided" in ac.gap(undecided) and "unconfirmed" in ac.gap(undecided)


# --- the review's probes (C:/fi_compare/reviews/review_tol.md): each a fault the check must not pass -------------------

SLOW_RATE = '''
import random, time

def run_trial(cell, trial_id, seed):
    if SLEEP:
        time.sleep(SLEEP)
    rng = random.Random(seed)
    k = int(cell["k"])
    return {"rate": sum(rng.random() < P for _ in range(k)) / k}
'''


def _synth(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return root


def _plain(root: Path, oracles: list[dict[str, Any]], **kw: Any):
    kw.setdefault("mean_names", frozenset())
    kw.setdefault("in_each_run", None)
    return _measure(root, oracles, **kw)


@pytest.mark.parametrize("base", ["1", "2"])
def test_a5_a_tolerance_the_trials_cannot_resolve_in_the_time_is_not_decided_never_passed(
        tmp_path: Path, base: str) -> None:
    """P0-1. The true rate is 0.06, twice the plan's tolerance (0.005) from the expected 0.05, and the simulation is slow:
    the trials FI can run in the time cannot put three standard errors inside 0.005. That is not a pass at a wider
    tolerance: the check is not decided (or, if the trials do show the gap, failed), and never counts as validated."""
    tight = {**RATE_CHECK, "tolerance": 0.005, "case": {"k": 50}}
    root = _synth(tmp_path, "P = 0.06\nSLEEP = 0.05\n" + SLOW_RATE)
    checks, problems, judged, found = _plain(root, [tight], timeout_s=3, env={"FI_REPLICATE_SEED": base})
    (j,) = judged
    assert j["passed_by_engine"] is not True and oc.engine_passed(judged) == [], j
    assert found, "a check that is not a pass is a problem the gate sees"
    if j["passed_by_engine"] is None:
        assert j.get("undecided") is True and any("not decided" in f for f in found), (j, found)


def test_a_tolerance_of_0_on_a_random_mean_is_never_widened(tmp_path: Path) -> None:
    """P0-3(a): exact means exact. A random rate against a tolerance of 0 cannot pass."""
    exact = {**RATE_CHECK, "tolerance": 0.0}
    root = _synth(tmp_path, "P = 0.05\nSLEEP = 0\n" + SLOW_RATE)
    _c, _p, judged, found = _plain(root, [exact], timeout_s=20)
    assert judged[0]["passed_by_engine"] is not True and found, judged


DECAY = '''
import math, random

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    x0 = rng.gauss(0.0, 1.0)
    exact = x0 * math.exp(-cell["k"] * 1.0)
    num = x0 * math.exp(-W * cell["k"] * 1.0)
    return {"x_num": num, "x_exact": exact}
'''
DECAY_CHECK = {"name": "matches_closed_form", "kind": "special_case",
               "check": "the integrator matches the closed form x0*exp(-k t) at t=1", "expected": 0.0,
               "tolerance": 0.01, "case": {"k": 1.0}, "measure": "abs(x_num - x_exact)",
               "reference": "derivation: dx/dt = -k x has x(t) = x0 exp(-k t)"}


@pytest.mark.parametrize("w", [1.5, 3.0])
def test_b_a_distance_between_two_numbers_of_each_trial_is_not_cancelled_by_averaging_them_first(
        tmp_path: Path, w: float) -> None:
    """P0-2. ``abs(x_num - x_exact)`` on a random initial condition: on the means the two cancel (x0 is symmetric), so
    a wrong decay rate passed. The names are not declared means: the formula is taken per trial and then averaged."""
    root = _synth(tmp_path, f"W = {w}\n" + DECAY)
    _c, problems, judged, found = _plain(root, [DECAY_CHECK], timeout_s=60)
    assert judged[0]["passed_by_engine"] is False and found, (judged, problems)
    assert judged[0]["value"] > 0.05


def test_b_the_right_decay_rate_passes_on_its_first_batch(tmp_path: Path) -> None:
    """P0-2 / P1-6. With the right rate the per-trial distance is 0 in every trial: judged at once, not run to 20000."""
    root = _synth(tmp_path, "W = 1.0\n" + DECAY)
    t0 = time.monotonic()
    checks, problems, judged, found = _plain(root, [DECAY_CHECK], timeout_s=60)
    assert judged[0]["passed_by_engine"] is True and not found, (judged, problems)
    assert (checks[0].get("mc") or {}).get("trials", 1) <= mc.PILOT_TRIALS and time.monotonic() - t0 < 15


FDP_AT_MOST_ONE = {"name": "fdp_at_most_one", "kind": "invariant", "check": "a false discovery proportion never exceeds 1",
                   "expected": 0, "tolerance": 0.0, "case": {"m": 200, "pi1": 0.2}, "measure": "max(0, fdr_bh - 1)",
                   "reference": "derivation: V <= R, so V/max(R,1) <= 1 in every repeat"}


@pytest.mark.parametrize("planted", [False, True])
def test_c_an_invariant_on_a_declared_mean_is_judged_on_every_trial(tmp_path: Path, planted: bool) -> None:
    """P0-3. A rule each trial keeps, on a name the plan declares a mean (``fdr_bh``), is never averaged unless the
    check says so: FDP + 1.2 in most trials fails, as it did on one trial before."""
    edits = {'out["fdr_" + s] = float(fdp)': 'out["fdr_" + s] = float(fdp) + (1.2 if V >= 1 else 0.0)'} if planted else None
    _c, problems, judged, found = _measure(_quest(tmp_path, simulate=edits), [FDP_AT_MOST_ONE], timeout_s=60)
    assert judged[0]["passed_by_engine"] is (not planted), (judged, problems, found)
    if planted:
        assert judged[0]["value"] >= 0.2 and found


TWO = '''
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    a = sum(rng.random() < 0.3 for _ in range(50))
    b = a + BUG
    return {"count_a": float(a), "count_b": float(b)}
'''
TWO_CHECK = {"name": "two_counts_agree", "kind": "second_implementation", "check": "two counters agree",
             "expected": 0, "tolerance": 0.0, "case": {"n": 1}, "measure": "abs(count_a - count_b)",
             "reference": "second implementation: an independent counter"}


@pytest.mark.parametrize("bug, passes", [("0", True), ("rng.choice((-1, 1))", False)])
def test_d_two_implementations_that_disagree_by_one_in_each_trial_fail_at_a_tolerance_of_0(
        tmp_path: Path, bug: str, passes: bool) -> None:
    """P0-3. Both counts are declared means, yet +1 in one trial and -1 in another is a disagreement in every trial:
    never averaged away, and a tolerance of 0 is never widened."""
    root = _synth(tmp_path, TWO.replace("BUG", bug))
    _c, problems, judged, found = _plain(root, [TWO_CHECK], timeout_s=60,
                                         mean_names=frozenset({"count_a", "count_b"}))
    assert judged[0]["passed_by_engine"] is passes, (judged, problems)
    assert bool(found) is not passes


@pytest.mark.parametrize("rep", [0, 1])
def test_e_a_real_bias_on_a_fast_simulation_is_decided_as_a_failure(tmp_path: Path, rep: int) -> None:
    """P1-4. A 30% bias (0.065 against 0.05, tolerance 0.01) with time left: FI keeps adding trials until the trials
    decide it, and it fails as a fault (sent to repair), not as 'within the error'."""
    root = _synth(tmp_path, f"P = 0.065\nSLEEP = 0\n" + SLOW_RATE.replace("random.Random(seed)",
                                                                           f"random.Random(seed + {rep} * 7919)"))
    _c, problems, judged, found = _plain(root, [RATE_CHECK], timeout_s=120)
    (j,) = judged
    assert j["passed_by_engine"] is False and found, (j, problems)
    assert not j.get("within_error") and not j.get("undecided"), j
    assert "not decided" not in " ".join(found)


@pytest.mark.asyncio
async def test_e_the_gate_sends_a_biased_random_simulation_to_repair(tmp_path: Path) -> None:
    """P1-4, through the gate: the biased rate is a fault, so the script is repaired (here to the right rate) and the
    check then passes."""
    from core.engine import Engine
    from tests.test_known_answer_self_checks import _config, _gate, _Model, _record

    right = "P = 0.05\nSLEEP = 0\n" + SLOW_RATE
    engine = Engine(_config(tmp_path))
    model = _Model(recompute={"expected": 0.05, "how": "a level-0.05 test"}, repair_code=right)
    engine._client = model
    await _gate(engine, [RATE_CHECK], "P = 0.065\nSLEEP = 0\n" + SLOW_RATE)
    record = _record(engine)
    assert model.repairs, "a decided failure goes to repair"
    assert record["status"] == "ok", record.get("problems")


DET = '''
import time

def run_trial(cell, trial_id, seed):
    time.sleep(0.01)
    return {"x": 2.0 * cell["a"], "wall": time.perf_counter()}
'''


def test_f_a_number_that_is_the_same_in_every_trial_is_judged_at_once(tmp_path: Path) -> None:
    """P1-6. The check's own number does not move; another number the simulation returns (a clock) does. Judged on the
    first batch, not run to the time limit."""
    check = {"name": "double", "kind": "special_case", "expected": 3.0, "tolerance": 1e-9, "case": {"a": 1.5},
             "measure": "x", "reference": "derivation: 2*1.5"}
    t0 = time.monotonic()
    checks, problems, judged, found = _plain(_synth(tmp_path, DET), [check], timeout_s=40)
    assert judged[0]["passed_by_engine"] is True and not found, (judged, problems)
    assert time.monotonic() - t0 < 8 and (checks[0].get("mc") or {}).get("trials", 1) <= mc.PILOT_TRIALS


CRASH = '''
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    x = rng.expovariate(1.0 / SCALE)
    if trial_id > 0 and x > 3.0:
        raise OverflowError("blew up")
    return {"x": x}
'''


def test_g_trials_that_failed_are_never_dropped_into_a_pass(tmp_path: Path) -> None:
    """P1-5. The mean is 2 against an expected 1; the trials with a large value crash (about a fifth of them). The mean
    of the others must not pass: more than a small share failed, and the check says how many."""
    check = {"name": "mean_wait", "kind": "special_case", "expected": 1.0, "tolerance": 0.2, "case": {"n": 1},
             "measure": "x", "reference": "derivation: an exponential with rate 1 has mean 1"}
    _c, problems, judged, found = _plain(_synth(tmp_path, "SCALE = 2.0\n" + CRASH), [check], timeout_s=60)
    (j,) = judged
    assert j["passed_by_engine"] is not True and found, (j, problems)
    assert any("trials" in f and "failed" in f for f in found), found


# --- not decided is never validated --------------------------------------------------------------------------------


def test_a_check_its_trials_could_not_decide_is_not_validated_anywhere() -> None:
    """P0-1, downstream: a check the trials could not decide is not in ``engine_passed``, is a problem the gate sees,
    can be gone on with as unconfirmed, and is said as not decided."""
    oracle = {**RATE_CHECK, "tolerance": 0.005, "case": {"k": 50}}
    record = {"read_as": "mean", "varies": True, "trials": 57, "se": 0.0042, "verdict": "undecided",
              "why": "time"}
    reported = {"checks": [{"name": "null_rate", "value": 0.0533, "measured_by": "engine", "mc": record}],
                "engine_measured": True}
    judged = oc.judged([oracle], reported)
    (j,) = judged
    assert j["passed_by_engine"] is None and j["undecided"] is True, j
    assert oc.engine_passed(judged) == [] and oc.not_passed(judged) == ["null_rate"]
    found = oc.problems([oracle], reported, 0)
    assert len(found) == 1 and "not decided" in found[0], found
    offered, why = ac.offer(found, [oracle], judged)
    assert offered and offered[0]["undecided"] is True, why
    (entry,) = ac.went_on_entries(found, [oracle], judged)
    assert entry["undecided"] is True
    note = oc.analysis_note(judged)
    assert "not decided" in note and "passed" not in note.split("null_rate", 1)[1].split("\n", 1)[0]


# --- time and seeds -------------------------------------------------------------------------------------------------


def test_the_checks_share_one_time_budget(tmp_path: Path) -> None:
    """P1-6. Two checks that cannot be decided in the time do not each take the whole of it."""
    tight = {**RATE_CHECK, "tolerance": 0.001}
    other = {**tight, "name": "null_rate_again"}
    root = _synth(tmp_path, "P = 0.05\nSLEEP = 0.02\n" + SLOW_RATE)
    t0 = time.monotonic()
    _c, _p, judged, _f = _plain(root, [tight, other], timeout_s=6)
    assert time.monotonic() - t0 < 10, "about 6 s in all, not 6 s per check"
    assert all(j["passed_by_engine"] is not True for j in judged)


COUNTED = '''
import random
from pathlib import Path

def run_trial(cell, trial_id, seed):
    with open(Path(__file__).resolve().parent.parent / "calls.txt", "a") as fh:
        fh.write("x\\n")
    rng = random.Random(seed)
    return {"rate": sum(rng.random() < 0.05 for _ in range(100)) / 100}
'''


def test_an_unchanged_simulation_reuses_the_trials_it_already_ran(tmp_path: Path) -> None:
    """P1-6. The same code on the same case (improve rounds, the hidden check) reuses the trials FI stored, and runs
    again once the code changes."""
    root = _synth(tmp_path, COUNTED)
    calls = root / "calls.txt"
    _c, _p, first, _f = _plain(root, [RATE_CHECK], timeout_s=60)
    ran = len(calls.read_text(encoding="utf-8").splitlines())
    _c, _p, again, _f = _plain(root, [RATE_CHECK], timeout_s=60)
    assert len(calls.read_text(encoding="utf-8").splitlines()) == ran, "no trial is run again"
    assert again[0]["value"] == first[0]["value"] and again[0]["passed_by_engine"] is first[0]["passed_by_engine"]
    (root / "code" / "simulate.py").write_text(COUNTED + "\n# changed\n", encoding="utf-8")
    _plain(root, [RATE_CHECK], timeout_s=60)
    assert len(calls.read_text(encoding="utf-8").splitlines()) > ran, "changed code is run again"


ECHO = '''
def run_trial(cell, trial_id, seed):
    return {"seed": float(seed), "x": float(seed % 7)}
'''


def test_a_paired_design_gives_the_check_the_seeds_of_the_reported_trials(tmp_path: Path) -> None:
    """P2. A paired study runs trial t of every setting on the same seed (``key=""``); the check's trials use the same."""
    root = _synth(tmp_path, ECHO)
    outcomes = asyncio.run(trial_runner.run_case_trials(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", cell={"n": 3},
        trials=[0, 1, 2], timeout_s=60, env={"FI_REPLICATE_SEED": "5"}, paired=True))
    rows = outcomes[0]
    assert [r["seed"] for r in rows] == [float(trial_runner.trial_seed(5, "", t)) for t in range(3)]


# --- the s1 bound, as the plan wrote it and as the plan should write it ---------------------------------------------


def test_the_s1_bound_as_frozen_is_a_rule_of_every_trial_and_fails_as_on_base(tmp_path: Path) -> None:
    """P0-3(b). The bound is kind ``invariant`` with no word that it is on an average: it is judged on every trial, the
    worst one counts, and it fails as it did on one trial (FI does not guess from the metrics that it is a mean)."""
    _c, problems, judged, found = _measure(_quest(tmp_path), [BOUND], timeout_s=60)
    assert judged[0]["passed_by_engine"] is False and found, (judged, problems)
    assert judged[0]["value"] >= 0.0188235 - 1e-6, "never better than trial 0 alone"
