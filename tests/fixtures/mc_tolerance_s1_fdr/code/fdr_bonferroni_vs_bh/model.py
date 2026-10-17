"""m independent unit-variance normal test statistics, a fixed number shifted by 2.5, one-sided p-values, thresholded by the uncorrected, Bonferroni or BH rule at 0.05."""

from __future__ import annotations
from scipy.stats import norm
import numpy as np


# E1
def pvalue_one_sided(z) -> float:
    # One-sided upper-tail p-value 1 - Phi(z) via scipy.stats.norm.sf (elementwise on arrays too).
    return norm.sf(z)


# E2
def bonferroni_power_exact(c: float, delta: float) -> float:
    # Exact Bonferroni power 1 - Phi(c - delta) for critical value c = z_{1-alpha/m}.
    return float(norm.sf(c - delta))


# E3
def prob_any_false_exact(m: int, t: float) -> float:
    # P(V>=1) = 1-(1-t)^m for m independent true nulls each rejected at per-test level t (caller passes m0 as m).
    return float(1.0 - (1.0 - t) ** m)


# S1
def bh_stepup(p, alpha: float):
    # Benjamini-Hochberg step-up: largest i with p_(i)<=i*alpha/m, reject ranks <=i; returns boolean mask in original order.
    p = np.asarray(p, dtype=float)
    m = p.size
    reject = np.zeros(m, dtype=bool)
    if m == 0:
        return reject
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    thresholds = np.arange(1, m + 1) * alpha / m
    ok = np.nonzero(sorted_p <= thresholds)[0]
    if ok.size == 0:
        return reject
    k = ok[-1]
    reject[order[: k + 1]] = True
    return reject


# S2
def error_tallies(mask, is_real):
    # Return V, S, R, FDP=V/max(R,1), TPP=S/m1 (nan if m1=0) from a rejection mask and truth labels.
    mask = np.asarray(mask, dtype=bool)
    is_real = np.asarray(is_real, dtype=bool)
    V = int(np.sum(mask & ~is_real))
    S = int(np.sum(mask & is_real))
    R = V + S
    m1 = int(np.sum(is_real))
    FDP = V / max(R, 1)
    TPP = S / m1 if m1 > 0 else float("nan")
    return V, S, R, FDP, TPP

