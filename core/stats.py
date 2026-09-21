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
