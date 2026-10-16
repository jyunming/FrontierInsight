"""The experiment generates additive-noise observations from an exact first-order decay at five Arrhenius-defined temperatures, then estimates a rate at each temperature and applies centered weighted linear regression in log-rate space."""

from __future__ import annotations
import math
import numpy as np


# E1
def first_order_concentration(c0_mol_L: float, k_s_inv: float, t_s: float) -> float:
    # Evaluate the exact first-order concentration trajectory.
    return c0_mol_L * np.exp(-k_s_inv * t_s)


# E2
def noisy_concentration(c0_mol_L: float, k_s_inv: float, t_s: float, sigma_c_mol_L: float, rng: np.random.Generator) -> float:
    return c0_mol_L * np.exp(-k_s_inv * t_s) + rng.normal(0.0, sigma_c_mol_L)


# E3
def arrhenius_rate(A_s_inv: float, Ea_J_mol: float, R_J_mol_K: float, T_K: float) -> float:
    # Evaluate the Arrhenius rate constant at one temperature.
    return A_s_inv * math.exp(-Ea_J_mol / (R_J_mol_K * T_K))


# E4
def centered_arrhenius_log_rate(alpha: float, beta: float, T_K: float, xbar_invK: float) -> float:
    # Evaluate the centered log-rate regression mean at one temperature.
    return alpha + beta * (1.0 / T_K - xbar_invK)


# E5
def activation_energy_from_slope(R_J_mol_K: float, beta: float) -> float:
    return -R_J_mol_K * beta
