"""Equations and numerical estimators for first-order Arrhenius recovery."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

R_GAS_J_MOL_K = 8.314462618
C0 = 1.0
K_LOWER_BOUND_S_INV = 1e-6
K_UPPER_BOUND_S_INV = 100.0
LS_TOLERANCE = 1e-10
NORMAL_Z = float(norm.ppf(0.975))


# E1
def concentration_first_order(c0: float, k: float, t: float) -> float:
    return c0 * np.exp(-k * t)


# E2
def arrhenius_rate(A: float, Ea: float, R: float, T: float) -> float:
    return A * np.exp(-Ea / (R * T))


# E3
def var_log_rate(var_k_hat: float, k_hat: float) -> float:
    return var_k_hat / k_hat**2


# E4
def activation_energy_estimate(R: float, Sxy: float, Sxx: float) -> float:
    return -R * Sxy / Sxx


# E5
def log_preexponential_estimate(ybar_weighted: float, m_hat: float, xbar_weighted: float) -> float:
    return ybar_weighted - m_hat * xbar_weighted


# S1
def generate_true_rates(temperature_K: float, Ea_true_J_mol: float, A_true_s_inv: float) -> float:
    return float(arrhenius_rate(A_true_s_inv, Ea_true_J_mol, R_GAS_J_MOL_K, temperature_K))


# S2
def generate_noisy_concentrations(k_true_s_inv: float, n_time_points: int,
                                  concentration_noise_sd_fraction_of_c0: float,
                                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.linspace(0.0, 3.0 / k_true_s_inv, n_time_points)
    concentration_true = concentration_first_order(C0, k_true_s_inv, time_s)
    return time_s, concentration_true + rng.normal(0.0, concentration_noise_sd_fraction_of_c0 * C0, size=n_time_points)


# S3
def fit_rate_constant(time_s: np.ndarray, concentration_observed: np.ndarray,
                      concentration_noise_sd_fraction_of_c0: float,
                      tolerance: float = LS_TOLERANCE) -> tuple[float, float]:
    time_s = np.asarray(time_s, dtype=float)
    observed = np.asarray(concentration_observed, dtype=float)
    if time_s.ndim != 1 or observed.shape != time_s.shape or len(time_s) <= 1 or np.max(time_s) <= 0.0:
        raise RuntimeError("invalid rate-fit data")
    t_scale = float(np.max(time_s) / 3.0)
    u = time_s / t_scale
    lower, upper = K_LOWER_BOUND_S_INV * t_scale, K_UPPER_BOUND_S_INV * t_scale

    def residuals(q: np.ndarray) -> np.ndarray:
        return concentration_first_order(C0, float(q[0]), u) - observed

    def jacobian(q: np.ndarray) -> np.ndarray:
        return (-u * np.exp(-float(q[0]) * u))[:, None]

    result = least_squares(residuals, x0=np.array([1.0]), jac=jacobian,
                           bounds=([lower], [upper]), xtol=1e-14, ftol=1e-14,
                           gtol=tolerance / t_scale)
    if not result.success:
        raise RuntimeError("nonlinear rate fit did not meet declared validity checks")

    # In one dimension the score is continuous and monotone at an interior
    # least-squares minimum.  Polishing its zero removes premature xtol
    # termination without relaxing the required physical J^T r tolerance.
    def score(q: float) -> float:
        r = concentration_first_order(C0, q, u) - observed
        return float(np.dot(-u * np.exp(-q * u), r))

    q_hat = float(result.x[0])
    score_lower, score_upper = score(lower), score(upper)
    if score_lower < 0.0 < score_upper:
        q_hat = float(brentq(score, lower, upper, xtol=np.finfo(float).eps,
                             rtol=4.0 * np.finfo(float).eps))
    fitted_residuals = concentration_first_order(C0, q_hat, u) - observed
    k_hat = float(q_hat / t_scale)
    physical_jacobian = (-time_s * np.exp(-k_hat * time_s))[:, None]
    physical_optimality = float(abs((physical_jacobian.T @ fitted_residuals)[0]))
    if (not np.isfinite(k_hat) or not K_LOWER_BOUND_S_INV < k_hat < K_UPPER_BOUND_S_INV or
            not np.isfinite(physical_optimality) or physical_optimality > tolerance):
        raise RuntimeError("nonlinear rate fit did not meet declared validity checks")
    information = physical_jacobian.T @ physical_jacobian
    if not np.all(np.isfinite(information)) or np.linalg.matrix_rank(information) != 1:
        raise RuntimeError("singular rate-fit Jacobian")
    residual_scale = float(np.sum(fitted_residuals**2) / (len(time_s) - 1))
    var_k = float(residual_scale / information[0, 0])
    if not np.isfinite(var_k) or var_k < 0.0:
        raise RuntimeError("invalid rate covariance")
    return k_hat, var_k


# S4
def propagate_log_rate_uncertainty(k_hat_s_inv: float, var_k_hat_s_inv2: float) -> tuple[float, float]:
    ln_k_hat = float(np.log(k_hat_s_inv))
    var_ln_k_hat = float(var_log_rate(var_k_hat_s_inv2, k_hat_s_inv))
    if not np.isfinite(ln_k_hat) or not np.isfinite(var_ln_k_hat) or var_ln_k_hat < 0.0:
        raise RuntimeError("invalid delta-method log-rate variance")
    return ln_k_hat, var_ln_k_hat


# S5
def fit_arrhenius_line(temperature_K: np.ndarray, ln_k_hat: np.ndarray,
                       var_ln_k_hat: np.ndarray) -> dict:
    x = 1.0 / np.asarray(temperature_K, dtype=float)
    y = np.asarray(ln_k_hat, dtype=float)
    variances = np.asarray(var_ln_k_hat, dtype=float)
    if len(x) < 3 or np.any(~np.isfinite(y)) or np.any(~np.isfinite(variances)) or np.any(variances <= 0.0):
        raise RuntimeError("invalid Arrhenius input")

    def fit(weights: np.ndarray, scale: float) -> dict:
        weight_sum = float(np.sum(weights))
        xbar = float(np.sum(weights * x) / weight_sum)
        ybar = float(np.sum(weights * y) / weight_sum)
        xc = x - xbar
        sxx = float(np.sum(weights * xc**2))
        sxy = float(np.sum(weights * xc * y))
        if not np.isfinite(sxx) or sxx <= 0.0:
            raise RuntimeError("singular Arrhenius design")
        slope = sxy / sxx
        ea = float(activation_energy_estimate(R_GAS_J_MOL_K, sxy, sxx))
        lna = float(log_preexponential_estimate(ybar, slope, xbar))
        base_covariance = np.array([[1.0 / sxx, -xbar / sxx],
                                    [-xbar / sxx, 1.0 / weight_sum + xbar**2 / sxx]])
        transform = np.diag([-R_GAS_J_MOL_K, 1.0])
        covariance = scale * transform @ base_covariance @ transform.T
        se = np.sqrt(np.diag(covariance))
        return {"Ea_hat_J_mol": ea, "lnA_hat": lna,
                "Ea_interval_95": [float(ea - NORMAL_Z * se[0]), float(ea + NORMAL_Z * se[0])],
                "lnA_interval_95": [float(lna - NORMAL_Z * se[1]), float(lna + NORMAL_Z * se[1])],
                "covariance": covariance,
                "centroid_residual": float(ybar - (lna + slope * xbar))}

    weighted = fit(1.0 / variances, 1.0)
    preliminary = fit(np.ones(len(x)), 1.0)
    residuals = y - (preliminary["lnA_hat"] - preliminary["Ea_hat_J_mol"] / R_GAS_J_MOL_K * x)
    unweighted = fit(np.ones(len(x)), float(np.sum(residuals**2) / (len(x) - 2)))
    return {"weighted": weighted, "unweighted": unweighted}
