"""FI's table of what every trial gave, for the analysis FI frames: read from FI's own trial record (``raw/trials.json``).

This file is carried by text into the quest's code folder as ``fi_table.py`` (``skeleton.TABLE_SOURCE``), the way the frame's
runtime is carried as ``fi_frame.py``, so the quest's interpreter, ``code/run.py`` and the host read the record with one
implementation. It therefore uses the standard library only and imports nothing of FI's: the few numbers it needs from
``core/stats.py`` (the t table, the Wilson interval, the mean and SD) are written out here, and ``tests/test_frame_table.py``
holds them equal to ``stats`` and to ``trial_table``'s numbers on recorded raws, so the copies cannot drift apart.

What it holds is the record and nothing derived from it: per setting and per value the trials gave, the values, the trials
they came from, ``n``, the mean and a 95% interval of the mean (Wilson when every trial gave 0 or 1, otherwise a t interval).
The class that serves it to a body is ``Table`` in the frame's runtime (``skeleton.RUNTIME``)."""

from __future__ import annotations

import json
import math
import os

#: Two-sided 95% t critical values by degrees of freedom (the same table as ``core.stats``).
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
    120: 1.980,
}
_Z95 = 1.960
_T95_ANCHORS = sorted(_T95)


def _t95(df):
    """Two-sided 95% t critical value: the largest table entry at or below ``df`` (never narrower than the exact one), the
    normal 1.96 above the table."""
    if df < 1:
        return _Z95
    if df in _T95:
        return _T95[df]
    below = [a for a in _T95_ANCHORS if a <= df]
    return _T95[below[-1]] if below else _Z95


def mean_std(vals):
    """``(mean, sample SD, n)``; the SD of fewer than two values is 0.0."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var), n


def wilson_interval(successes, trials, z=_Z95):
    """Wilson 95% interval of a proportion from ``successes`` of ``trials``; ``None`` for impossible counts."""
    if trials is None or successes is None or trials <= 0 or successes < 0 or successes > trials:
        return None
    n = float(trials)
    p_hat = float(successes) / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _fmt(value):
    if isinstance(value, float) and value.is_integer():
        return repr(value)
    if isinstance(value, list):  # an array setting: no comma, a key is split on them
        return "[" + ";".join(_fmt(x) for x in value) + "]"
    return str(value)


def cell_key(cell):
    """``"R0=0.9,N=100"``: FI's name of a setting (``trial_runner.cell_key``)."""
    return ",".join(f"{axis}={_fmt(value)}" for axis, value in cell.items())


def finite(value):
    """Whether ``value`` is a number a trial can have given (not a bool, not NaN or an infinity)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def hits(metrics, ends, truth, estimate=None):
    """1.0 or 0.0 for each trial that ran (a trial number any metric of ``metrics`` holds a value for), 1.0 where the interval
    the trial reported (the values of the two metrics named in ``ends``) holds ``truth``; a trial with no interval counts 0.0.
    With ``estimate`` (the estimate's own metric record), a trial with no finite estimate counts 0.0 too, whatever its interval
    says: an interval around no estimate is not a result of the trial. ``None`` when ``metrics`` does not hold both ends at all.
    The one pairing FI uses for a coverage, wherever it counts one."""
    if not all(isinstance(metrics.get(e), dict) for e in ends):
        return None
    series = []
    for e in ends:
        raw = metrics[e].get("values") or []
        series.append({t: float(v) for t, v in zip(metrics[e].get("trials") or range(len(raw)), raw) if finite(v)})
    ran = {t for m in metrics.values() if isinstance(m, dict) for t in (m.get("trials") or []) if isinstance(t, int)}
    if isinstance(estimate, dict):
        raw = estimate.get("values") or []
        have = {t for t, v in zip(estimate.get("trials") or range(len(raw)), raw) if finite(v)}
        ran |= {t for t in (estimate.get("trials") or []) if isinstance(t, int)}
    else:
        have = None
    return [1.0 if i in series[0] and i in series[1] and series[0][i] <= truth <= series[1][i] and (have is None or i in have)
            else 0.0 for i in sorted(ran)]


def values_of(metric):
    """The finite values of one metric record of a setting, as floats, in the order the trials gave them."""
    raw = metric.get("values") if isinstance(metric, dict) else metric
    if not isinstance(raw, (list, tuple)):
        return []
    return [float(v) for v in raw if finite(v)]


def trials_of(metric):
    """The trial number of each value :func:`values_of` returns, when the record says them (else ``[]``)."""
    raw = metric.get("values") if isinstance(metric, dict) else metric
    given = metric.get("trials") if isinstance(metric, dict) else None
    if not isinstance(raw, (list, tuple)) or not isinstance(given, (list, tuple)) or len(given) != len(raw):
        return []
    return [t for v, t in zip(raw, given) if finite(v)]


def paired(metric_a, metric_b):
    """``(xs, ys)``: the finite values two metric records of one setting gave in the same trial, paired by the record's own trial
    numbers and never by position, in the order of the trial numbers. A trial with no finite value in either is left out of both. Empty
    where either record does not say its trial numbers, or says one twice (a pairing by position would be a guess)."""
    def by_trial(metric):
        raw = metric.get("values") if isinstance(metric, dict) else None
        given = metric.get("trials") if isinstance(metric, dict) else None
        if not isinstance(raw, (list, tuple)) or not isinstance(given, (list, tuple)) or len(given) != len(raw):
            return None
        if len(set(given)) != len(given):
            return None
        return {t: float(v) for v, t in zip(raw, given) if finite(v)}
    a, b = by_trial(metric_a), by_trial(metric_b)
    if a is None or b is None:
        return [], []
    common = sorted(set(a) & set(b))
    return [a[t] for t in common], [b[t] for t in common]


def pearson(xs, ys):
    """The Pearson correlation of two equally long lists; ``None`` for fewer than three pairs or when either has no spread."""
    n = len(xs)
    if n != len(ys) or n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def quantile(values, q):
    """The ``q`` quantile (0 to 1) of ``values`` by linear interpolation between the sorted values; ``None`` for no values."""
    vals = sorted(values)
    if not vals or not 0.0 <= q <= 1.0:
        return None
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def summary(values, proportion=False):
    """``{"n", "mean", "sd", "binary", "zero_one", "successes"}`` of one value's trials in one setting. ``proportion``: the plan declares
    the result a proportion; only then is it ``binary`` (every trial gave 0 or 1) and given a binomial interval. The kind
    of interval is the plan's to declare, never read from the values: ``zero_one`` only says every trial happened to give 0 or 1."""
    mean, sd, n = mean_std(values)
    zero_one = bool(values) and all(v in (0.0, 1.0) for v in values)
    return {"n": n, "mean": mean, "sd": sd, "binary": bool(proportion) and zero_one, "zero_one": zero_one,
            "successes": sum(values) if zero_one else None}


#: The method a mean gets; a result that gave only 0 and 1 though the plan does not declare it a proportion gets this said too.
T_INTERVAL = "t interval of the mean"
UNDECLARED = " (the plan does not declare it a proportion)"


def interval(s):
    """``(lower, upper, method)`` of the 95% interval of the mean from a :func:`summary`; ``(None, None, "")`` without one."""
    n = int(s.get("n") or 0)
    if n <= 0:
        return None, None, ""
    if s.get("binary"):
        ci = wilson_interval(s["successes"], n)
        return (ci[0], ci[1], "Wilson") if ci else (None, None, "")
    if n < 2:
        return None, None, ""
    half = _t95(n - 1) * (s["sd"] / math.sqrt(n))
    return s["mean"] - half, s["mean"] + half, T_INTERVAL + (UNDECLARED if s.get("zero_one") else "")


#: The environment variable FI's harness gives the analysis, naming (relative to the quest folder) the record it is to read: FI's
#: own ``raw/trials.json``, or the copy beside it without the quantities and settings FI left out (``core.trial_runner.RESULTS_ENV``).
RESULTS_ENV = "FI_TRIALS"


def record_path(quest_root, environ=None):
    """The record the analysis reads in the quest folder ``quest_root``: the file ``FI_TRIALS`` names, else FI's
    ``raw/trials.json``. Never a guess between the two files: a copy left from an earlier run is not what this run names."""
    named = (os.environ if environ is None else environ).get(RESULTS_ENV)
    return os.path.join(str(quest_root), named) if named else os.path.join(str(quest_root), "raw", "trials.json")


def load(path):
    """The cells of the trial record at ``path`` (the file itself, or a quest folder for its ``raw/trials.json``) that carry
    metrics; ``[]`` when there is no readable record."""
    path = str(path)
    if os.path.isdir(path):
        path = os.path.join(path, "raw", "trials.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    cells = data.get("cells") if isinstance(data, dict) else None
    if not isinstance(cells, list):
        return []
    return [c for c in cells if isinstance(c, dict) and isinstance(c.get("metrics"), dict)]
