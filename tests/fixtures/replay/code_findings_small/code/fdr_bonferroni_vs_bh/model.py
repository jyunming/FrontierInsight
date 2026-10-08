"""Simulation of independent one-sided z-tests with known mean shifts and standard normal noise."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


# E1
# E1
# E1
# E1
# E1
# E1
# E1
# E1
# E1
def one_sided_p_value(z: float | np.ndarray) -> float | np.ndarray:
    """Return the specified one-sided p-value, p = 1 - Phi(z)."""
    return 1.0 - ndtr(np.asarray(z))


# E2
def bonferroni_threshold(alpha: float, m: int) -> float:
    # Compute the per-test Bonferroni threshold.
    return alpha / m


# E3
def bh_fdr_reference(pi0: float) -> float:
    return pi0 * 0.05


# E4
def false_discovery_proportion(false_positives: int, rejections: int) -> float:
    return false_positives / rejections if rejections > 0 else 0.0
