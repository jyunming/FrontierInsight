"""Unit tests for the pure-stdlib statistics (core/stats.py) and the engine's
per-stratum / effect-size / multiple-comparison aggregation."""

from __future__ import annotations

import math

from core import stats
from core.engine import (
    _aggregate_result_json_replicates,
    _result_comparison_stats,
)


# --- core/stats.py ----------------------------------------------------------


def test_confidence_interval_t_distribution() -> None:
    # n=3, values 0.1/0.2/0.3: mean 0.2, sample-std 0.1, se 0.1/sqrt(3),
    # t(2)=4.303 → half-width 4.303 * se.
    ci = stats.confidence_interval([0.1, 0.2, 0.3])
    se = 0.1 / math.sqrt(3)
    assert math.isclose(ci["se"], se, rel_tol=1e-9)
    assert math.isclose(ci["ci_lower"], 0.2 - 4.303 * se, rel_tol=1e-6)
    assert math.isclose(ci["ci_upper"], 0.2 + 4.303 * se, rel_tol=1e-6)


def test_confidence_interval_single_value_is_undefined() -> None:
    ci = stats.confidence_interval([0.42])
    assert ci == {"se": None, "ci_lower": None, "ci_upper": None}


def test_cohens_d_known_value() -> None:
    # Two groups three pooled-SDs apart → d = -3 (pooled SD 1, mean diff 3).
    a = [10.0, 11.0, 12.0]   # mean 11, var 1
    b = [13.0, 14.0, 15.0]   # mean 14, var 1 → pooled SD 1, diff 3
    d = stats.cohens_d(a, b)
    assert d is not None and math.isclose(d, -3.0, rel_tol=1e-9)
    assert stats.effect_magnitude(d) == "large"


def test_cohens_d_undefined_cases() -> None:
    assert stats.cohens_d([1.0], [2.0, 3.0]) is None        # n<2 in a group
    assert stats.cohens_d([5.0, 5.0], [5.0, 5.0]) is None   # zero spread
    assert stats.effect_magnitude(None) == "n/a"


def test_effect_magnitude_thresholds() -> None:
    assert stats.effect_magnitude(0.1) == "negligible"
    assert stats.effect_magnitude(0.35) == "small"
    assert stats.effect_magnitude(0.6) == "medium"
    assert stats.effect_magnitude(1.2) == "large"


def test_t95_anchoring_is_conservative() -> None:
    # Exact anchors, the largest-anchor-≤-df rule between them, and — above the
    # table — the largest anchor (1.980), staying conservative (never < true t).
    assert stats._t95(5) == 2.571
    assert stats._t95(45) == stats._t95(40) == 2.021   # 45 → df=40 anchor
    assert stats._t95(500) == 1.980                    # above table → df=120 anchor
    assert stats._t95(0) == stats._Z95                 # no df → bare normal


def test_bonferroni_alpha() -> None:
    assert stats.bonferroni_alpha(5) == 0.01
    assert stats.bonferroni_alpha(0) == 0.05  # guarded against div-by-zero


# --- aggregate enriched with se + CI ---------------------------------------


def test_aggregate_carries_se_and_ci() -> None:
    agg = _aggregate_result_json_replicates(
        [{"_seed": i, "rmse": v} for i, v in enumerate([0.1, 0.2, 0.3])]
    )
    m = agg["rmse"]
    assert math.isclose(m["mean"], 0.2, abs_tol=1e-9) and m["n"] == 3
    assert m["ci_lower"] is not None and m["ci_upper"] is not None
    assert m["ci_lower"] < m["mean"] < m["ci_upper"]


def test_aggregate_single_seed_ci_is_none() -> None:
    agg = _aggregate_result_json_replicates([{"_seed": 0, "rmse": 0.42}])
    assert agg["rmse"]["ci_lower"] is None and agg["rmse"]["se"] is None


# --- _result_comparison_stats ----------------------------------------------


def _by_method_reps() -> list[dict]:
    return [
        {"_seed": s, "by_method": {"A": {"epe": a}, "B": {"epe": b}}}
        for s, (a, b) in enumerate([(1.0, 2.0), (1.1, 2.1), (0.9, 1.9)])
    ]


def test_comparison_stats_strata_effects_and_guard() -> None:
    cs = _result_comparison_stats(_by_method_reps())
    # Per-stratum CI present.
    a_ci = cs["strata"]["by_method"]["A"]["epe"]
    assert a_ci["mean"] == 1.0 and a_ci["ci_lower"] < 1.0 < a_ci["ci_upper"]
    # One pairwise effect size (A vs B on epe), large separation.
    assert cs["comparisons"]["n"] == 1
    es = cs["effect_sizes"][0]
    assert es["a"] == "A" and es["b"] == "B" and es["metric"] == "epe"
    assert es["magnitude"] == "large"
    assert cs["comparisons"]["bonferroni_alpha"] == 0.05
    assert cs["comparisons"]["many"] is False


def test_comparison_stats_many_flag() -> None:
    # 4 strata → C(4,2)=6 pairwise comparisons on one metric → many.
    reps = [
        {"_seed": s, "by_g": {k: {"m": v + s * 0.01}
                              for k, v in {"a": 1, "b": 2, "c": 3, "d": 4}.items()}}
        for s in range(3)
    ]
    cs = _result_comparison_stats(reps)
    assert cs["comparisons"]["n"] == 6
    assert cs["comparisons"]["many"] is True
    assert math.isclose(cs["comparisons"]["bonferroni_alpha"], 0.05 / 6)


def test_comparison_stats_empty_without_strata() -> None:
    assert _result_comparison_stats([{"_seed": 0, "rmse": 0.1}]) == {}
    # A single stratum can't be compared → empty.
    assert _result_comparison_stats(
        [{"_seed": s, "by_x": {"only": {"m": 1.0}}} for s in range(3)]
    ) == {}


# --- nested strata (crossed designs) ---------------------------------------


def _crossed_reps() -> list[dict]:
    """by_h -> step size -> integrator -> metric, three seeds with spread."""
    return [
        {"_seed": s, "by_h": {
            "0.1": {"Euler": {"err": 1.0 + j}, "RK4": {"err": 0.1 + j / 2}},
            "0.5": {"Euler": {"err": 5.0 + j}, "RK4": {"err": 0.5 + j / 3}},
        }}
        for s, j in enumerate([0.0, 0.01, -0.01])
    ]


def test_crossed_design_compares_methods_within_each_step_size() -> None:
    """Method A against method B lives below the by_* key. That comparison
    used to produce nothing at all."""
    cs = _result_comparison_stats(_crossed_reps())
    pairs = {(e["factor"], e["metric"], e["a"], e["b"]) for e in cs["effect_sizes"]}
    assert ("by_h.0.1", "err", "Euler", "RK4") in pairs
    assert ("by_h.0.5", "err", "Euler", "RK4") in pairs
    euler = cs["strata"]["by_h.0.5"]["Euler"]["err"]
    assert math.isclose(euler["mean"], 5.0, abs_tol=1e-9) and euler["n"] == 3


def test_crossed_design_also_compares_step_sizes_per_method() -> None:
    cs = _result_comparison_stats(_crossed_reps())
    pairs = {(e["factor"], e["metric"]) for e in cs["effect_sizes"]}
    assert ("by_h", "Euler.err") in pairs and ("by_h", "RK4.err") in pairs
    # 2 metrics at the step-size level + 1 per step size at the method level;
    # the guard counts every one of them.
    assert cs["comparisons"]["n"] == 4


def test_nested_level_of_unlike_groups_is_not_compared() -> None:
    """``errors`` and ``timing`` are groupings of different things, not strata
    of one factor."""
    reps = [
        {"_seed": s, "by_method": {
            m: {"errors": {"l2": b + s * 0.01, "linf": 2 * b + s * 0.01},
                "timing": {"wall_s": 3 * b + s * 0.01}}
            for m, b in {"A": 1.0, "B": 2.0}.items()
        }}
        for s in range(3)
    ]
    cs = _result_comparison_stats(reps)
    assert {e["factor"] for e in cs["effect_sizes"]} == {"by_method"}


def test_a_null_metric_leaves_the_other_pairs_comparable() -> None:
    """A diverged method reports null; the remaining pair is still compared
    and the null never turns into a number."""
    reps = [
        {"_seed": s, "by_h": {"0.5": {
            "Euler": {"err": None},
            "Heun": {"err": 0.8 + s * 0.01},
            "RK4": {"err": 0.5 + s * 0.01},
        }}}
        for s in range(3)
    ]
    cs = _result_comparison_stats(reps)
    pairs = {(e["a"], e["b"]) for e in cs["effect_sizes"] if e["factor"] == "by_h.0.5"}
    assert pairs == {("Heun", "RK4")}
    assert "Euler" not in cs["strata"]["by_h.0.5"]


def test_identical_replicates_give_no_effect_size_without_crashing() -> None:
    """Replication stops at two seeds once they agree, so zero spread is the
    common input now, not an edge case."""
    reps = [{"_seed": s, "by_m": {"A": {"x": 1.0}, "B": {"x": 2.0}}} for s in range(2)]
    es = _result_comparison_stats(reps)["effect_sizes"]
    assert es[0]["cohens_d"] is None and es[0]["magnitude"] == "n/a"
