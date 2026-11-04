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
    assert stats._chunks(10, 4) == [10] and stats._chunks(2000, 1_000) == [250] * 8


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


# The kernels as they were before the chunk shrank (4M numbers per block), kept as the reference for the seeded draws.
_OLD = 4_000_000


def _old_chunks(rounds: int, width: int) -> list[int]:
    per, out, left = max(1, _OLD // max(1, width)), [], rounds
    while left > 0:
        out.append(min(per, left))
        left -= out[-1]
    return out


def _old_means(vals, rounds, rng):
    import numpy as np

    n, gen, arr, out = len(vals), np.random.default_rng(rng.getrandbits(64)), np.asarray(vals, dtype=float), []
    for r in _old_chunks(rounds, n):
        out.append(arr[gen.integers(0, n, size=(r, n))].mean(axis=1))
    return np.concatenate(out).tolist()


def _old_mean_differences(x, y, rounds, rng):
    import numpy as np

    nx, ny, gen = len(x), len(y), np.random.default_rng(rng.getrandbits(64))
    ax, ay, out = np.asarray(x, dtype=float), np.asarray(y, dtype=float), []
    for r in _old_chunks(rounds, nx + ny):
        out.append(ax[gen.integers(0, nx, size=(r, nx))].mean(axis=1) - ay[gen.integers(0, ny, size=(r, ny))].mean(axis=1))
    return np.concatenate(out).tolist()


def _old_permutation_hits(pool, nx, observed, rounds, rng):
    import numpy as np

    total, gen, arr, hits = len(pool), np.random.default_rng(rng.getrandbits(64)), np.asarray(pool, dtype=float), 0
    for r in _old_chunks(rounds, total):
        shuffled = gen.permuted(np.broadcast_to(arr, (r, total)).copy(), axis=1)
        diff = shuffled[:, :nx].mean(axis=1) - shuffled[:, nx:].mean(axis=1)
        hits += int((np.abs(diff) >= abs(observed) - 1e-12).sum())
    return hits


def _old_sign_flip_hits(values, observed_abs, rounds, rng):
    import numpy as np

    m, gen, arr, hits = len(values), np.random.default_rng(rng.getrandbits(64)), np.asarray(values, dtype=float), 0
    for r in _old_chunks(rounds, m):
        flipped = np.where(gen.random((r, m)) < 0.5, arr, -arr).sum(axis=1)
        hits += int((np.abs(flipped) >= observed_abs - 1e-12).sum())
    return hits


def _old_cluster_means(groups, rounds, rng):
    import numpy as np

    g, gen = len(groups), np.random.default_rng(rng.getrandbits(64))
    sums, counts, out = np.asarray([s for s, _n in groups], dtype=float), np.asarray([n for _s, n in groups], dtype=float), []
    for r in _old_chunks(rounds, g):
        idx = gen.integers(0, g, size=(r, g))
        c, s = counts[idx].sum(axis=1), sums[idx].sum(axis=1)
        out.append(np.divide(s, c, out=np.zeros(r), where=c > 0))
    return np.concatenate(out).tolist()


def _old_cluster_differences(ga, gb, rounds, rng):
    import numpy as np

    gen = np.random.default_rng(rng.getrandbits(64))
    sa, ca = np.asarray([s for s, _n in ga], dtype=float), np.asarray([n for _s, n in ga], dtype=float)
    sb, cb = np.asarray([s for s, _n in gb], dtype=float), np.asarray([n for _s, n in gb], dtype=float)
    out = []
    for r in _old_chunks(rounds, len(ga) + len(gb)):
        ia, ib = gen.integers(0, len(ga), size=(r, len(ga))), gen.integers(0, len(gb), size=(r, len(gb)))
        na, nb = ca[ia].sum(axis=1), cb[ib].sum(axis=1)
        ma = np.divide(sa[ia].sum(axis=1), na, out=np.zeros(r), where=na > 0)
        mb = np.divide(sb[ib].sum(axis=1), nb, out=np.zeros(r), where=nb > 0)
        out.append(ma - mb)
    return np.concatenate(out).tolist()


@pytest.mark.parametrize("chunk", [250_000, 5_003, 7, 4_000_000])
@pytest.mark.parametrize("sizes", [(8000, 1300), (301, 211), (1501, 977)])
def test_every_resampling_kernel_draws_the_same_numbers_as_before_at_any_chunk_size(
    monkeypatch: pytest.MonkeyPatch, chunk: int, sizes: tuple[int, int],
) -> None:
    """The chunk bounds memory only: for one seed each kernel returns exactly what the 4M-chunk kernel returned (single
    sample, two sample, permutation, sign flip, cluster means and cluster differences)."""
    import random

    na, nb = sizes
    x, y = _sample(7, na, 0.0), _sample(8, nb, 0.2)
    ga = [(sum(x[i::37]), len(x[i::37])) for i in range(37)]
    gb = [(sum(y[i::29]), len(y[i::29])) for i in range(29)]
    rounds = 2000 if na < 5000 else 500  # 8000 x 500 is above one 4M block
    monkeypatch.setattr(stats, "_CHUNK_ELEMENTS", chunk)
    rng = lambda: random.Random(11)  # noqa: E731
    assert stats._resampled_means(x, rounds, rng()) == _old_means(x, rounds, rng())
    assert stats._resampled_mean_differences(x, y, rounds, rng()) == _old_mean_differences(x, y, rounds, rng())
    pool = x[:na // 2] + y[:nb // 2]
    assert stats._permutation_hits(pool, na // 2, 0.01, rounds, rng()) == _old_permutation_hits(pool, na // 2, 0.01, rounds, rng())
    assert stats._sign_flip_hits(y, 3.0, rounds, rng()) == _old_sign_flip_hits(y, 3.0, rounds, rng())
    assert stats._resampled_cluster_means(ga, rounds, rng()) == _old_cluster_means(ga, rounds, rng())
    assert stats._resampled_cluster_differences(ga, gb, rounds, rng()) == _old_cluster_differences(ga, gb, rounds, rng())


def test_the_default_resampling_chunk_is_a_small_array() -> None:
    import tracemalloc

    values = _sample(7, 8000, 0.0)
    assert stats._CHUNK_ELEMENTS * 8 <= 4 * 1024 * 1024
    tracemalloc.start()
    stats.bootstrap_mean_interval(values, resamples=500, seed=3)
    stats.two_sample_mean_test(values, _sample(8, 1300, 0.2), seed=4)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 16 * 1024 * 1024
