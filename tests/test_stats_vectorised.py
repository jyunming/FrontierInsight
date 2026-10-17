"""The bootstrap / permutation kernels in core/stats.py run as numpy array operations; the pure-Python loop they replaced
is kept as the no-numpy fallback, so it doubles as the reference: the two must agree in distribution, edge case by edge case."""

from __future__ import annotations

import math
import random
import time
from typing import Any

import pytest

from core import metric_spec as ms
from core import stats

_ESTIMAND_UNIT = {"estimand": "the mean of x", "unit": "trial"}


@pytest.fixture
def pure_python(monkeypatch: pytest.MonkeyPatch):
    """A callable that runs ``fn`` on the old pure-Python path (numpy hidden from the kernels)."""

    def run(fn, *args, **kw):
        with monkeypatch.context() as m:
            m.setattr(stats, "_numpy", lambda: None)
            return fn(*args, **kw)

    return run


def _sample(seed: int, n: int, mu: float, sd: float = 1.0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sd) for _ in range(n)]


SEEDS = range(24)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def test_two_sample_p_value_and_interval_agree_with_the_pure_python_path(pure_python) -> None:
    a, b = _sample(1, 40, 0.0), _sample(2, 40, 0.3)
    new = [stats.two_sample_mean_test(a, b, seed=s) for s in SEEDS]
    old = [pure_python(stats.two_sample_mean_test, a, b, seed=s) for s in SEEDS]
    assert all(r["diff"] == o["diff"] for r, o in zip(new, old)), "the observed difference is not resampled"
    assert abs(_mean(r["p"] for r in new) - _mean(o["p"] for o in old)) < 0.03
    assert abs(_mean(r["ci_lower"] for r in new) - _mean(o["ci_lower"] for o in old)) < 0.02
    assert abs(_mean(r["ci_upper"] for r in new) - _mean(o["ci_upper"] for o in old)) < 0.02
    assert all(r["method"] == "bootstrap_permutation" and r["ci_lower"] < r["diff"] < r["ci_upper"] for r in new)
    # p = (hits + 1) / (rounds + 1): never 0, on the same grid as before (rounds = 2000 here)
    assert all(math.isclose(r["p"] * 2001, round(r["p"] * 2001)) and r["p"] >= 1 / 2001 for r in new)


def test_a_clear_difference_and_a_null_difference_give_the_same_verdicts_as_before(pure_python) -> None:
    a, b, c = _sample(3, 60, 0.0), _sample(4, 60, 1.0), _sample(5, 60, 0.0)
    for fn_args, small in (((a, b), True), ((a, c), False)):
        new, old = stats.two_sample_mean_test(*fn_args), pure_python(stats.two_sample_mean_test, *fn_args)
        assert (new["p"] < 0.01) == (old["p"] < 0.01) == small
        assert (new["ci_lower"] < 0 < new["ci_upper"]) == (old["ci_lower"] < 0 < old["ci_upper"])


def test_the_same_seed_gives_the_same_answer_and_another_seed_another_draw() -> None:
    a, b = _sample(1, 50, 0.0), _sample(2, 50, 0.2)
    assert stats.two_sample_mean_test(a, b, seed=7) == stats.two_sample_mean_test(a, b, seed=7)
    assert stats.two_sample_mean_test(a, b, seed=7) != stats.two_sample_mean_test(a, b, seed=8)
    big = _sample(9, 6000, 0.0)  # thinned to the cap first; the thinning is seeded too
    assert stats.two_sample_mean_test(big, big[::-1], seed=3) == stats.two_sample_mean_test(big, big[::-1], seed=3)


def test_two_sample_edge_cases_match_the_pure_python_path(pure_python) -> None:
    nan, inf = float("nan"), float("inf")
    cases: list[tuple[list[Any], list[Any]]] = [
        ([], []), ([1.0], [2.0, 3.0]), ([1.0, 2.0], []), ([nan, nan, 1.0], [1.0, 2.0, 3.0]),
        ([True, False, 1.0], [1.0, 2.0]), ([inf, -inf, 1.0], [1.0, 2.0]),
    ]
    for a, b in cases:
        assert stats.two_sample_mean_test(a, b) == pure_python(stats.two_sample_mean_test, a, b), (a, b)
    # NaN / inf / bool entries are dropped first: same answer as the cleaned lists
    assert stats.two_sample_mean_test([nan, 1.0, 2.0, 4.0, inf], [3.0, 5.0, 8.0, nan]) == stats.two_sample_mean_test([1.0, 2.0, 4.0], [3.0, 5.0, 8.0])
    # all values equal: no difference, every resample and split equal -> p = 1, a degenerate interval
    same = stats.two_sample_mean_test([2.0] * 10, [2.0] * 12)
    assert same == pure_python(stats.two_sample_mean_test, [2.0] * 10, [2.0] * 12)
    assert same["diff"] == 0.0 and same["p"] == 1.0 and same["ci_lower"] == same["ci_upper"] == 0.0
    # two values on each side is the smallest accepted sample
    assert stats.two_sample_mean_test([1.0, 2.0], [3.0, 5.0])["p"] > 0


def test_a_chunk_smaller_than_one_resample_still_draws_every_round(monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = _sample(1, 30, 0.0), _sample(2, 30, 0.5)
    whole = stats.two_sample_mean_test(a, b, seed=4)
    monkeypatch.setattr(stats, "_CHUNK_ELEMENTS", 7)  # one round per chunk
    chunked = stats.two_sample_mean_test(a, b, seed=4)
    assert math.isclose(chunked["p"] * 2001, round(chunked["p"] * 2001))
    assert abs(chunked["p"] - whole["p"]) < 0.04 and abs(chunked["ci_lower"] - whole["ci_lower"]) < 0.15
    assert sum(stats._chunks(2000, 10_000_000)) == 2000 and max(stats._chunks(2000, 10_000_000)) == 1
    monkeypatch.undo()
    assert stats._chunks(10, 4) == [10] and stats._chunks(2000, 10_000) == [400] * 5


def test_bootstrap_mean_interval_agrees_with_the_pure_python_path(pure_python) -> None:
    v = _sample(6, 80, 1.5, 2.0)
    new = [stats.bootstrap_mean_interval(v, seed=s) for s in SEEDS]
    old = [pure_python(stats.bootstrap_mean_interval, v, seed=s) for s in SEEDS]
    assert abs(_mean(n[0] for n in new) - _mean(o[0] for o in old)) < 0.03
    assert abs(_mean(n[1] for n in new) - _mean(o[1] for o in old)) < 0.03
    assert all(lo < _mean(v) < hi for lo, hi in new)
    assert stats.bootstrap_mean_interval([]) is None and stats.bootstrap_mean_interval([1.0]) is None
    assert stats.bootstrap_mean_interval([float("nan"), 1.0]) is None
    assert stats.bootstrap_mean_interval([3.0] * 9) == (3.0, 3.0) == pure_python(stats.bootstrap_mean_interval, [3.0] * 9)


def test_paired_permutation_agrees_with_the_pure_python_path(pure_python) -> None:
    d = _sample(7, 60, 0.25)  # more than the exact limit, so sign flips are drawn
    new = [stats.paired_permutation_test(d, seed=s) for s in SEEDS]
    old = [pure_python(stats.paired_permutation_test, d, seed=s) for s in SEEDS]
    assert abs(_mean(n["p"] for n in new) - _mean(o["p"] for o in old)) < 0.03
    assert abs(_mean(n["ci_lower"] for n in new) - _mean(o["ci_lower"] for o in old)) < 0.03
    assert abs(_mean(n["ci_upper"] for n in new) - _mean(o["ci_upper"] for o in old)) < 0.03
    assert all(n["diff"] == o["diff"] and n["n_pairs"] == 60 for n, o in zip(new, old))
    for edge in ([], [1.0], [0.0] * 20, [float("nan")] * 3 + [1.0, -1.0], [1.0] * 14):
        assert stats.paired_permutation_test(edge) == pure_python(stats.paired_permutation_test, edge)
    # past the exact enumeration the flips are drawn: the two paths agree in distribution, not draw for draw
    many = [1.0] * 30 + [0.0] * 5
    new = [stats.paired_permutation_test(many, seed=s) for s in SEEDS]
    old = [pure_python(stats.paired_permutation_test, many, seed=s) for s in SEEDS]
    assert all(n["diff"] == o["diff"] and n["n_pairs"] == o["n_pairs"] for n, o in zip(new, old))
    assert abs(_mean(n["p"] for n in new) - _mean(o["p"] for o in old)) < 0.01
    # the all-same-sign case: every flip but the identity (and its mirror) is smaller, in both paths
    assert stats.paired_permutation_test([1.0] * 30)["p"] < 0.01


def test_cluster_bootstraps_agree_with_the_pure_python_path(pure_python) -> None:
    rng = random.Random(8)
    vals_a, cl_a, vals_b, cl_b = [], [], [], []
    for c in range(25):
        shift = rng.gauss(0, 0.5)
        for _ in range(rng.randint(1, 6)):
            vals_a.append(shift + rng.gauss(0.0, 1.0))
            cl_a.append(c)
            vals_b.append(shift + rng.gauss(0.4, 1.0))
            cl_b.append(c)
    new = [stats.cluster_bootstrap_interval(vals_a, cl_a, seed=s) for s in SEEDS]
    old = [pure_python(stats.cluster_bootstrap_interval, vals_a, cl_a, seed=s) for s in SEEDS]
    assert all(n["mean"] == o["mean"] and n["n_clusters"] == 25 for n, o in zip(new, old))
    assert abs(_mean(n["ci_lower"] for n in new) - _mean(o["ci_lower"] for o in old)) < 0.03
    assert abs(_mean(n["ci_upper"] for n in new) - _mean(o["ci_upper"] for o in old)) < 0.03
    new = [stats.cluster_bootstrap_difference(vals_a, cl_a, vals_b, cl_b, seed=s) for s in SEEDS]
    old = [pure_python(stats.cluster_bootstrap_difference, vals_a, cl_a, vals_b, cl_b, seed=s) for s in SEEDS]
    assert abs(_mean(n["p"] for n in new) - _mean(o["p"] for o in old)) < 0.05
    assert abs(_mean(n["ci_lower"] for n in new) - _mean(o["ci_lower"] for o in old)) < 0.04
    assert abs(_mean(n["ci_upper"] for n in new) - _mean(o["ci_upper"] for o in old)) < 0.04
    for args in (([1.0], [0], [2.0, 3.0], [0, 1]), ([], [], [], []), ([1.0, 2.0], [0], [1.0, 2.0], [0, 1])):
        assert stats.cluster_bootstrap_difference(*args) == pure_python(stats.cluster_bootstrap_difference, *args) is None


def test_five_thousand_against_five_thousand_values_takes_seconds_not_minutes() -> None:
    """Pure Python needed ~1.3 s here per contrast on a fast machine (minutes at the full 2000 rounds); the bound is
    generous so a busy CI runner cannot fail it, and far below what the old loop needed for the same work."""
    a, b = _sample(10, 5000, 0.0), _sample(11, 5000, 0.05)
    started = time.monotonic()
    got = stats.two_sample_mean_test(a, b)
    assert time.monotonic() - started < 20
    assert got is not None and got["ci_lower"] < got["diff"] < got["ci_upper"]
    started = time.monotonic()
    full = stats.two_sample_mean_test(a[:1000], b[:1000], resamples=2000)  # 2000 rounds of 1000 + 1000
    assert time.monotonic() - started < 20 and full is not None


# --- the progress line -------------------------------------------------------------------------------------------------------


def _replicates(n_settings: int = 4) -> list[dict[str, Any]]:
    return [
        {"_seed": s, "by_cell": {f"c{k}": {"x": 0.1 * k + 0.01 * s, "x_values": [0.1 * k + 0.01 * i for i in range(6)]} for k in range(n_settings)}}
        for s in range(2)
    ]


def test_statistics_says_how_far_it_is_through_the_contrasts(monkeypatch: pytest.MonkeyPatch) -> None:
    proto = {"metrics": [{"id": "x", "kind": "mean", **_ESTIMAND_UNIT}]}
    lines: list[str] = []
    monkeypatch.setattr(ms, "_PROGRESS_EVERY_CONTRASTS", 2)
    out = ms.statistics(_replicates(4), proto, progress=lines.append)
    assert len(out["contrasts"]) == 6
    assert lines == [
        "[evidence] statistics: 2 of 6 contrasts (0 s)", "[evidence] statistics: 4 of 6 contrasts (0 s)",
    ], "one line per 2 contrasts, none after the last (the caller logs the end)"
    monkeypatch.undo()
    quiet: list[str] = []
    ms.statistics(_replicates(4), proto, progress=quiet.append)  # default every 50 contrasts: nothing for 6
    assert quiet == []


def test_statistics_says_a_line_after_thirty_seconds_even_between_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    proto = {"metrics": [{"id": "x", "kind": "mean", **_ESTIMAND_UNIT}]}
    clock = iter(range(0, 1000, 31))
    monkeypatch.setattr(ms.time, "monotonic", lambda: float(next(clock)))
    lines: list[str] = []
    ms.statistics(_replicates(3), proto, progress=lines.append)
    assert len(lines) == 3 and lines[0].startswith("[evidence] statistics: 1 of 3 contrasts")


def test_statistics_without_a_callback_logs_through_the_module_logger(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    import logging

    monkeypatch.setattr(ms, "_PROGRESS_EVERY_CONTRASTS", 1)
    with caplog.at_level(logging.INFO, logger="frontier_insight.metric_spec"):
        ms.statistics(_replicates(3), {"metrics": [{"id": "x", "kind": "mean", **_ESTIMAND_UNIT}]})
    assert any("[evidence] statistics: 1 of 3 contrasts" in r.getMessage() for r in caplog.records)
