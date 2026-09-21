"""Pure-stdlib statistical rigor for experiment results — confidence
intervals (a t interval over seeds, a Wilson interval for a proportion from
counts, a bootstrap interval for a mean from pooled values), effect sizes
(Cohen's d), and a multiple-comparison guard.

No numpy / scipy: a small embedded two-sided 95% t-table covers small-n
replication (the common case), falling back to the normal approximation for
large df. Used by the analyze node to enrich the multi-seed replicate
aggregate so the paper reports uncertainty instead of bare point numbers.
"""

from __future__ import annotations

import math
from typing import Any

# Two-sided 95% Student-t critical values by degrees of freedom (n-1). Exact
# for df 1..30 (every small-n replication) plus a few larger anchors; for a df
# between/above the anchors we use the nearest anchor at or below it, so the
# interval is never narrower than the true t interval. At df=30 t≈2.042 (~4%
# wider than the normal 1.96); the gap closes to ~1% by df=120. Above the
# largest anchor we keep that anchor's value (1.980) rather than dropping to
# 1.960 — staying conservative — since the true t for df>120 is still >1.96.
# The bare normal (1.960) is only used when there are no degrees of freedom.
_T95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
    120: 1.980,
}
_Z95 = 1.960
_T95_ANCHORS = sorted(_T95)


def _t95(df: int) -> float:
    """Two-sided 95% t critical value for ``df`` degrees of freedom. For a df
    not in the table, use the largest anchor ≤ df (conservative: never narrower
    than the exact interval); above the largest anchor, the normal 1.96."""
    if df < 1:
        return _Z95
    if df in _T95:
        return _T95[df]
    below = [a for a in _T95_ANCHORS if a <= df]
    return _T95[below[-1]] if below else _Z95


def mean_std(vals: list[float]) -> tuple[float, float, int]:
    """``(mean, sample-std, n)``. n<2 → std 0.0 (sample std needs n-1)."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var), n


def confidence_interval(
    vals: list[float], *, lower: float | None = None, upper: float | None = None,
) -> dict[str, Any]:
    """95% confidence interval of the mean via the t-distribution.

    Returns ``{"se", "ci_lower", "ci_upper"}``. With n<2 there's no spread to
    estimate, so the interval is undefined (``None``) — a single seed reports a
    point estimate, honestly, not a fake zero-width interval.

    ``lower`` and ``upper`` are values the quantity cannot pass, such as a
    probability's 0 and 1. The interval stops at them: a t interval around a
    small probability otherwise runs below zero."""
    mean, std, n = mean_std(vals)
    if n < 2:
        return {"se": None, "ci_lower": None, "ci_upper": None}
    se = std / math.sqrt(n)
    half = _t95(n - 1) * se
    ci_lower, ci_upper = mean - half, mean + half
    if lower is not None:
        ci_lower = max(ci_lower, lower)
    if upper is not None:
        ci_upper = min(ci_upper, upper)
    return {"se": se, "ci_lower": ci_lower, "ci_upper": ci_upper}


def wilson_interval(successes: float, trials: float, *, z: float = _Z95) -> tuple[float, float] | None:
    """Wilson score 95% interval of a proportion estimated from ``successes`` out of ``trials`` independent
    Bernoulli trials; ``None`` when there are no trials or the counts are impossible.

    A t interval over a few batch proportions measures how much the batches moved, not how well the trials pin
    the probability down: three batches of 300 runs give an interval of width set by three numbers, where the
    900 trials behind them set a much tighter or looser one. The Wilson interval uses the trials, stays inside
    [0, 1], and behaves at a proportion of 0 or 1, where the normal-approximation interval does not."""
    if trials is None or successes is None or trials <= 0 or successes < 0 or successes > trials:
        return None
    n = float(trials)
    p_hat = float(successes) / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_half_width(successes: float, trials: float) -> float | None:
    """Half the width of the Wilson interval, or ``None`` for impossible counts."""
    ci = wilson_interval(successes, trials)
    return None if ci is None else (ci[1] - ci[0]) / 2.0


def trials_for_half_width(half_width: float, *, p: float = 0.5, z: float = _Z95) -> int | None:
    """How many independent trials a proportion near ``p`` needs before its 95% interval is at most ``half_width`` wide on
    each side (the normal approximation; ``p`` = 0.5 is the worst case and the safe default). ``None`` for a width that
    is not between 0 and 1."""
    if not isinstance(half_width, (int, float)) or isinstance(half_width, bool) or not 0.0 < half_width < 1.0:
        return None
    return int(math.ceil(round(z * z * p * (1.0 - p) / (half_width * half_width), 9)))


def bootstrap_mean_interval(
    values: list[float], *, resamples: int = 2000, seed: int = 0, cap: int = 20000,
) -> tuple[float, float] | None:
    """Percentile-bootstrap 95% interval of the mean of ``values`` (the raw observations behind an average, pooled
    over every batch), or ``None`` for fewer than 2 values.

    Deterministic: the resampling is drawn from a generator seeded with ``seed`` so the same values always give the
    same interval. More than ``cap`` values are thinned to ``cap`` evenly spaced ones first, and the number of
    resamples drops with the size, so a very large pool costs a bounded time."""
    import random

    vals = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    if len(vals) < 2:
        return None
    if len(vals) > cap:
        step = len(vals) / cap
        vals = [vals[int(i * step)] for i in range(cap)]
    n = len(vals)
    rounds = resamples if n <= 2000 else max(200, resamples * 2000 // n)
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    lo = means[int(0.025 * (rounds - 1))]
    hi = means[int(math.ceil(0.975 * (rounds - 1)))]
    return lo, hi


def cohens_d(a: list[float], b: list[float]) -> float | None:
    """Standardized mean difference (pooled-SD Cohen's d) of ``a`` vs ``b``.
    ``None`` when not computable (fewer than 2 per group, or zero spread)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, _, _ = mean_std(a)
    mb, _, _ = mean_std(b)
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return None
    return (ma - mb) / pooled


def effect_magnitude(d: float | None) -> str:
    """Conventional label for a Cohen's d magnitude."""
    if d is None:
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def bonferroni_alpha(n_comparisons: int, alpha: float = 0.05) -> float:
    """Bonferroni-corrected significance threshold for ``n`` comparisons."""
    return alpha / max(1, n_comparisons)


# --- comparisons: the sampling distribution that goes with each estimator ---------------------------------------------------
#
# A contrast between two settings is judged with the distribution of the estimator it uses: a two-proportion test for
# proportions from independent trials, a permutation or bootstrap distribution for means, a sign-flip permutation of the
# per-pair differences when the settings shared their random numbers (the pairs are then not independent), and a bootstrap
# over whole clusters when the observations inside a cluster are not independent of each other. Every resampling is seeded, so
# the same data give the same interval and p-value.


def _p_two_sided_normal(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def _rounds(n: int, resamples: int = 2000) -> int:
    return resamples if n <= 2000 else max(200, resamples * 2000 // n)


def _thin(vals: list[float], cap: int) -> list[float]:
    if len(vals) <= cap:
        return vals
    step = len(vals) / cap
    return [vals[int(i * step)] for i in range(cap)]


def _clean(values: list[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]


def _percentile_interval(samples: list[float]) -> tuple[float, float]:
    samples = sorted(samples)
    rounds = len(samples)
    return samples[int(0.025 * (rounds - 1))], samples[int(math.ceil(0.975 * (rounds - 1)))]


def two_proportion_test(k1: float, n1: float, k2: float, n2: float) -> dict[str, Any] | None:
    """The difference of two proportions from independent trials (``k1`` of ``n1`` minus ``k2`` of ``n2``): Newcombe's hybrid
    score 95% interval (built from the two Wilson intervals, so it behaves at 0 and 1) and the pooled two-sided z test.
    ``None`` for impossible counts."""
    a, b = wilson_interval(k1, n1), wilson_interval(k2, n2)
    if a is None or b is None:
        return None
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    lower = diff - math.sqrt((p1 - a[0]) ** 2 + (b[1] - p2) ** 2)
    upper = diff + math.sqrt((a[1] - p1) ** 2 + (p2 - b[0]) ** 2)
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        p_value = 1.0 if diff == 0.0 else 0.0
    else:
        p_value = _p_two_sided_normal(diff / se)
    return {"diff": diff, "ci_lower": lower, "ci_upper": upper, "p": p_value, "method": "two_proportion_z_newcombe"}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm's step-down adjustment of a family of p-values: each adjusted value is at least as large as the raw one, the
    adjusted values keep the order of the raw ones, and none exceeds 1. The family-wise error rate is held at the level
    a single test would be judged at."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def paired_permutation_test(diffs: list[float], *, resamples: int = 4000, seed: int = 0, exact_up_to: int = 14) -> dict[str, Any] | None:
    """The mean of per-pair differences (settings that shared their random numbers), with a sign-flip permutation p-value
    (exact when at most ``exact_up_to`` pairs differ, otherwise ``resamples`` random flips) and a percentile bootstrap 95%
    interval of the mean difference. ``None`` for fewer than 2 pairs."""
    import itertools
    import random

    d = _thin(_clean(diffs), 20000)
    n = len(d)
    if n < 2:
        return None
    observed = sum(d) / n
    nonzero = [x for x in d if x != 0.0]
    rng = random.Random(seed)
    if not nonzero:
        p_value = 1.0
    elif len(nonzero) <= exact_up_to:
        total = extreme = 0
        for signs in itertools.product((1.0, -1.0), repeat=len(nonzero)):
            total += 1
            if abs(sum(s * x for s, x in zip(signs, nonzero))) >= abs(sum(nonzero)) - 1e-12:
                extreme += 1
        p_value = extreme / total
    else:
        rounds = _rounds(len(nonzero), resamples)
        hits = sum(
            1 for _ in range(rounds)
            if abs(sum(x if rng.random() < 0.5 else -x for x in nonzero)) >= abs(sum(nonzero)) - 1e-12
        )
        p_value = (hits + 1) / (rounds + 1)
    rounds = _rounds(n)
    means = [sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds)]
    lo, hi = _percentile_interval(means)
    return {"diff": observed, "ci_lower": lo, "ci_upper": hi, "p": p_value, "n_pairs": n, "method": "paired_permutation_bootstrap"}


def two_sample_mean_test(a: list[float], b: list[float], *, resamples: int = 2000, seed: int = 0, cap: int = 5000) -> dict[str, Any] | None:
    """The difference of two means from independent observations: a percentile bootstrap 95% interval and a two-sided
    permutation p-value. ``None`` when either sample has fewer than 2 observations."""
    import random

    x, y = _thin(_clean(a), cap), _thin(_clean(b), cap)
    if len(x) < 2 or len(y) < 2:
        return None
    rng = random.Random(seed)
    observed = sum(x) / len(x) - sum(y) / len(y)
    rounds = _rounds(len(x) + len(y), resamples)
    boots = [
        sum(x[rng.randrange(len(x))] for _ in range(len(x))) / len(x) - sum(y[rng.randrange(len(y))] for _ in range(len(y))) / len(y)
        for _ in range(rounds)
    ]
    pool = x + y
    hits = 0
    for _ in range(rounds):
        rng.shuffle(pool)
        if abs(sum(pool[: len(x)]) / len(x) - sum(pool[len(x):]) / len(y)) >= abs(observed) - 1e-12:
            hits += 1
    lo, hi = _percentile_interval(boots)
    return {"diff": observed, "ci_lower": lo, "ci_upper": hi, "p": (hits + 1) / (rounds + 1), "method": "bootstrap_permutation"}


def _cluster_totals(values: list[float], clusters: list[Any]) -> list[tuple[float, int]] | None:
    if len(values) != len(clusters):
        return None
    grouped: dict[Any, list[float]] = {}
    for v, c in zip(values, clusters):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        grouped.setdefault(c, []).append(float(v))
    return [(sum(v), len(v)) for v in grouped.values()]


def cluster_bootstrap_interval(values: list[float], clusters: list[Any], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    """The mean of observations that come in clusters (trials of one household, steps of one trajectory), with a percentile
    bootstrap 95% interval from resampling WHOLE clusters: the observations of a cluster are not independent, so they do
    not count as independent trials. ``None`` for mismatched lists or fewer than 2 clusters."""
    import random

    groups = _cluster_totals(values, clusters)
    if not groups or len(groups) < 2:
        return None
    total, count = sum(s for s, _n in groups), sum(n for _s, n in groups)
    rng = random.Random(seed)
    g = len(groups)
    rounds = _rounds(g)
    means = []
    for _ in range(rounds):
        picks = [groups[rng.randrange(g)] for _ in range(g)]
        c = sum(n for _s, n in picks)
        means.append(sum(s for s, _n in picks) / c if c else 0.0)
    lo, hi = _percentile_interval(means)
    return {"mean": total / count, "ci_lower": lo, "ci_upper": hi, "n_clusters": g, "n_observations": count, "method": "cluster_bootstrap"}


def cluster_bootstrap_difference(
    a: list[float], a_clusters: list[Any], b: list[float], b_clusters: list[Any], *, resamples: int = 2000, seed: int = 0,
) -> dict[str, Any] | None:
    """The difference of two means of clustered observations from independent clusters: whole clusters of each group are
    resampled; the interval is the percentile interval and the p-value the bootstrap two-sided sign p-value."""
    import random

    ga, gb = _cluster_totals(a, a_clusters), _cluster_totals(b, b_clusters)
    if not ga or not gb or len(ga) < 2 or len(gb) < 2:
        return None
    observed = sum(s for s, _n in ga) / sum(n for _s, n in ga) - sum(s for s, _n in gb) / sum(n for _s, n in gb)
    rng = random.Random(seed)
    rounds = _rounds(len(ga) + len(gb))
    diffs = []
    for _ in range(rounds):
        pa = [ga[rng.randrange(len(ga))] for _ in range(len(ga))]
        pb = [gb[rng.randrange(len(gb))] for _ in range(len(gb))]
        ca, cb = sum(n for _s, n in pa), sum(n for _s, n in pb)
        diffs.append((sum(s for s, _n in pa) / ca if ca else 0.0) - (sum(s for s, _n in pb) / cb if cb else 0.0))
    below = sum(1 for d in diffs if d <= 0.0)
    above = sum(1 for d in diffs if d >= 0.0)
    p_value = min(1.0, 2.0 * (min(below, above) + 1) / (rounds + 1))
    lo, hi = _percentile_interval(diffs)
    return {"diff": observed, "ci_lower": lo, "ci_upper": hi, "p": p_value, "method": "cluster_bootstrap"}
