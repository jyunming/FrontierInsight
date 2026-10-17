"""Assumption-driven Monte Carlo of scalar high-NA aerial intensity, Poisson photon arrival and absorption, simplified CAR acid generation and quenching or MOR network change, blur, threshold development, and geometry-based defects; comparative results are valid only within this model."""

from __future__ import annotations
import math


# E1
def photon_density(Dose: float) -> float:
    return 0.680 * Dose


# E2
def expected_absorbed_count(Phi: float, A: float, alpha: float, t: float) -> float:
    # Computes the expected absorbed-photon count in one uniform region.
    return Phi * A * (1 - math.exp(-alpha * t))


# E3
def relative_photon_noise(N: float) -> float:
    return 1.0 / N**0.5


# E4
def zero_absorption_probability(N: float) -> float:
    return math.exp(-N)

