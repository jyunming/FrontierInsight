"""First-order reaction kinetics with Arrhenius temperature dependence and known initial concentration, measured in mol/L; k and A are in s^-1."""

from __future__ import annotations
import math


# E1
def first_order_concentration(c0, k, t) -> float:
    return c0 * math.exp(-k * t)


# E2
def arrhenius_rate(A, Ea, R, T) -> float:
    return A * math.exp(-Ea / (R * T))


# E3
def log_arrhenius_rate(A, Ea, R, T) -> float:
    # Compute ln(k / 1 s^-1) from Arrhenius parameters.
    return math.log(A) - (Ea / R) * (1.0 / T)

