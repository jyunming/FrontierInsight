"""Each family consists of independent N(0,1) null z-statistics and N(2.5,1) nonnull z-statistics, transformed to one-sided p-values and passed to a specified multiplicity procedure."""

from __future__ import annotations
import math
import numpy as np


# E1
def draw_z(mu: float, rng: np.random.Generator) -> float:
    # Draw one z statistic from N(mu, 1).
    return float(mu + rng.normal(0.0, 1.0))


# E2
def z_to_p(z: float) -> float:
    # Compute the one-sided upper-tail standard-normal p-value.
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# E3
def bh_rank_threshold(alpha: float, k: int, m: int) -> float:
    return alpha * k / m


# E4
def false_discovery_proportion(V: int, R: int) -> float:
    return V / max(R, 1)


# E5
def family_power(S: int, m1: int) -> float:
    return S / m1


# E6
def uncorrected_fwer_reference(alpha: float, m: int) -> float:
    return 1.0 - (1.0 - alpha) ** m


# E7
def bh_fdr_reference(pi1: float) -> float:
    return (1.0 - pi1) * 0.05


# S1
def generate_family(m: int, pi1: float, nonnull_z_shift: float) -> tuple[np.ndarray, np.ndarray]:
    # Create one fixed-composition truth-labelled family of z statistics.
    n_nonnull = int(round(m * pi1))
    truth = np.zeros(m, dtype=bool)
    truth[:n_nonnull] = True

    rng = np.random.default_rng()
    z = np.empty(m, dtype=float)
    for i, is_nonnull in enumerate(truth):
        z[i] = draw_z(nonnull_z_shift if is_nonnull else 0.0, rng)

    return z, truth


# S2
def compute_p_values(z: float) -> float:
    # Transform one z score to its one-sided p-value.
    return z_to_p(z)


# S3
def apply_procedure(p_values: np.ndarray, procedure: str, alpha: float, m: int) -> np.ndarray:
    if procedure == "uncorrected":
        return p_values <= alpha
    if procedure == "bonferroni":
        return p_values <= alpha / m
    if procedure == "benjamini_hochberg":
        order = np.argsort(p_values)
        sorted_p_values = p_values[order]
        ranks = np.arange(1, m + 1)
        eligible = sorted_p_values <= np.array(
            [bh_rank_threshold(alpha, int(k), m) for k in ranks]
        )
        rejected = np.zeros(m, dtype=bool)
        if np.any(eligible):
            k = np.max(np.nonzero(eligible)[0]) + 1
            rejected[order[:k]] = True
        return rejected
    raise ValueError(f"Unknown procedure: {procedure}")


# S4
def score_decisions(rejections: np.ndarray, truth_labels: np.ndarray) -> dict:
    rejections = np.asarray(rejections, dtype=bool)
    truth_labels = np.asarray(truth_labels, dtype=bool)

    V = int(np.sum(rejections & ~truth_labels))
    S = int(np.sum(rejections & truth_labels))
    R = int(np.sum(rejections))
    m1 = int(np.sum(truth_labels))

    return {
        "V": V,
        "S": S,
        "R": R,
        "FDP": false_discovery_proportion(V, R),
        "power": family_power(S, m1) if m1 > 0 else None,
        "any_false_discovery": bool(V > 0),
    }


# S5
def aggregate_uncertainty(trial_metrics: np.ndarray, bootstrap_resamples: int) -> dict:
    values = np.asarray(trial_metrics, dtype=float).ravel()
    estimate = float(np.mean(values))
    rng = np.random.default_rng(20261015)
    bootstrap_means = np.empty(bootstrap_resamples, dtype=float)

    for i in range(bootstrap_resamples):
        indices = rng.integers(0, values.size, size=values.size)
        bootstrap_means[i] = np.mean(values[indices])

    ci_lower, ci_upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }

