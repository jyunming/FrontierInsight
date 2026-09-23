"""Null-simulation / coverage validation for core/stats.py's estimators (audit P0-3's ask: each estimator/test
needs a check that it behaves the way its name claims, not just unit tests of its arithmetic on fixed inputs).

Each test below draws many independent simulated datasets under a KNOWN ground truth (a fixed true proportion or
mean, or the null hypothesis of no true difference), runs the estimator FI actually ships on each one, and checks
that the empirical coverage / type-I error rate over all of them is close to the nominal 95% / 5% the estimator
claims. This is the reference: nothing here is compared against scipy or another library (this module is
deliberately pure-stdlib), the estimator is checked against what "95% interval" / "p < 0.05 five percent of the
time" actually mean.

Everything is seeded (both the simulated data and the estimator's own resampling), so a run is exactly
reproducible; the tolerance bands below were set by running this file and are not guesses. Marked slow: this is a
validation suite for the estimators themselves, run occasionally, not a per-commit regression guard on fixed
inputs (tests/test_stats.py and tests/test_metric_spec.py already cover that, fast).
"""

from __future__ import annotations

import random

import pytest

from core import stats

pytestmark = pytest.mark.slow


def test_wilson_interval_covers_the_true_proportion_about_95_percent_of_the_time() -> None:
    rng = random.Random(20260923)
    p_true, n, reps = 0.3, 50, 4000
    hits = 0
    for _ in range(reps):
        k = sum(1 for _ in range(n) if rng.random() < p_true)
        lo, hi = stats.wilson_interval(k, n)
        hits += lo <= p_true <= hi
    coverage = hits / reps
    assert 0.93 <= coverage <= 0.97, coverage


def test_two_proportion_test_type_i_error_is_close_to_5_percent_under_the_null() -> None:
    """Both samples drawn from the SAME true proportion (no real difference): p < 0.05 should fire on
    about 5% of repetitions, not systematically more (a test that cries wolf) or less (one with no power).
    Deliberately UNEQUAL n1/n2 -- an estimator that assumed equal sample sizes (e.g. an unweighted
    average of the two n's) would pass a same-n test but not this one."""
    rng = random.Random(20260924)
    p_true, n1, n2, reps = 0.4, 60, 90, 3000
    false_positives = 0
    for _ in range(reps):
        k1 = sum(1 for _ in range(n1) if rng.random() < p_true)
        k2 = sum(1 for _ in range(n2) if rng.random() < p_true)
        result = stats.two_proportion_test(k1, n1, k2, n2)
        false_positives += result["p"] < 0.05
    rate = false_positives / reps
    assert 0.03 <= rate <= 0.07, rate


def test_bootstrap_mean_interval_covers_the_true_mean_about_95_percent_of_the_time() -> None:
    rng = random.Random(20260925)
    mu, sigma, n, reps = 10.0, 2.0, 30, 600
    hits = 0
    for i in range(reps):
        values = [rng.gauss(mu, sigma) for _ in range(n)]
        lo, hi = stats.bootstrap_mean_interval(values, seed=i)
        hits += lo <= mu <= hi
    coverage = hits / reps
    assert 0.92 <= coverage <= 0.97, coverage


def test_paired_permutation_test_type_i_error_is_close_to_5_percent_under_the_null() -> None:
    """Per-pair differences drawn from a zero-mean distribution (no true shift): p < 0.05 should fire on
    about 5% of repetitions."""
    rng = random.Random(20260926)
    n_pairs, reps = 40, 2000
    false_positives = 0
    for i in range(reps):
        diffs = [rng.gauss(0.0, 1.0) for _ in range(n_pairs)]
        result = stats.paired_permutation_test(diffs, seed=i)
        false_positives += result["p"] < 0.05
    rate = false_positives / reps
    assert 0.035 <= rate <= 0.065, rate


def test_cluster_bootstrap_difference_type_i_error_is_close_to_5_percent_under_the_null() -> None:
    """Two groups of clusters with the SAME true per-observation mean (no true difference between groups):
    p < 0.05 should fire on about 5% of repetitions. Cluster sizes VARY (4 to 14 observations) rather
    than being fixed -- an estimator that took an unweighted mean of per-cluster means (ignoring size)
    would still look fine on equal-sized clusters but not on these."""
    rng = random.Random(20260927)
    mu, sigma, n_clusters, reps = 5.0, 1.0, 12, 600
    false_positives = 0
    for i in range(reps):
        a_vals: list[float] = []
        a_clusters: list[int] = []
        b_vals: list[float] = []
        b_clusters: list[int] = []
        for c in range(n_clusters):
            cluster_effect = rng.gauss(0.0, 0.3)  # within-cluster correlation: a shared per-cluster shift
            for _ in range(rng.randint(4, 14)):
                a_vals.append(mu + cluster_effect + rng.gauss(0.0, sigma))
                a_clusters.append(c)
            cluster_effect2 = rng.gauss(0.0, 0.3)
            for _ in range(rng.randint(4, 14)):
                b_vals.append(mu + cluster_effect2 + rng.gauss(0.0, sigma))
                b_clusters.append(c + 1000)  # distinct cluster ids from group a
        result = stats.cluster_bootstrap_difference(a_vals, a_clusters, b_vals, b_clusters, seed=i)
        false_positives += result["p"] < 0.05
    rate = false_positives / reps
    assert 0.02 <= rate <= 0.09, rate
