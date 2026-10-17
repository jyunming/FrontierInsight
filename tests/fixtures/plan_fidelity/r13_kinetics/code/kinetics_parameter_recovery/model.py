"""Independent first-order concentration series are generated exactly, perturbed by Gaussian concentration noise, fitted by nonlinear least squares, and transformed to a centred weighted Arrhenius regression."""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import curve_fit

_FIT_NOISE_SIGMA = 0.01


def set_fit_noise_sigma(noise_sigma: float) -> None:
    global _FIT_NOISE_SIGMA
    noise_sigma = float(noise_sigma)
    if not np.isfinite(noise_sigma) or noise_sigma <= 0.0:
        raise ValueError("The known concentration-noise standard deviation must be positive.")
    _FIT_NOISE_SIGMA = noise_sigma


# E1
def first_order_concentration(c0: float, k: float, t: float) -> float:
    return c0 * math.exp(-k * t)


# E2
def arrhenius_rate(A: float, Ea: float, R: float, T: float) -> float:
    return A * np.exp(-Ea / (R * T))


# E3
# E3
def map_arrhenius_parameters(R: float, b: float, alpha: float, xbar: float) -> tuple[float, float]:
    Ea = -R * b
    lnA = alpha - b * xbar
    return Ea, lnA


# S1
def construct_temperatures(Ea_true_J_mol: float, R_J_mol_K: float, lnA_true: float) -> np.ndarray:
    rates_s_inv = np.geomspace(0.1, 1.0, 5)
    A_true_s_inv = np.exp(lnA_true)
    temperatures_K = Ea_true_J_mol / (R_J_mol_K * (lnA_true - np.log(rates_s_inv)))
    _ = np.array([arrhenius_rate(A_true_s_inv, Ea_true_J_mol, R_J_mol_K, T) for T in temperatures_K])
    return temperatures_K


# S2
def simulate_concentrations(noise_sigma_over_c0: float, n_time: int) -> list[np.ndarray]:
    c0 = 1.0
    rate_constants = np.geomspace(0.1, 1.0, 5)
    rng = np.random.default_rng(20261015)
    noise_sigma = noise_sigma_over_c0 * c0
    return [np.asarray([first_order_concentration(c0, k, t) + rng.normal(0.0, noise_sigma) for t in np.linspace(0.0, 3.0 / k, n_time)]) for k in rate_constants]


# S3
def fit_first_stage(concentration: np.ndarray, time_s: np.ndarray) -> tuple[float, float, float]:
    def model(t: np.ndarray, k: float) -> np.ndarray:
        return np.asarray([first_order_concentration(1.0, k, float(t_i)) for t_i in t], dtype=float)

    time_s = np.asarray(time_s, dtype=float)
    concentration = np.asarray(concentration, dtype=float)
    initial_k = 1.0 / max(float(np.max(time_s)), np.finfo(float).eps)
    parameters, covariance = curve_fit(model, time_s, concentration, p0=[initial_k], bounds=([np.nextafter(0.0, 1.0)], [np.inf]), sigma=np.full_like(concentration, _FIT_NOISE_SIGMA, dtype=float), absolute_sigma=True)
    k_hat = float(parameters[0])
    var_k_hat = float(covariance[0, 0])
    var_log_k_hat = var_k_hat / k_hat**2
    return k_hat, var_k_hat, var_log_k_hat


# S4
def fit_arrhenius(k_hat_s_inv: np.ndarray, var_lnk: np.ndarray, temperature_K: np.ndarray) -> dict:
    k_hat_s_inv = np.asarray(k_hat_s_inv, dtype=float)
    var_lnk = np.asarray(var_lnk, dtype=float)
    temperature_K = np.asarray(temperature_K, dtype=float)
    if (k_hat_s_inv.shape != (5,) or var_lnk.shape != (5,) or temperature_K.shape != (5,) or np.any(k_hat_s_inv <= 0.0) or np.any(var_lnk <= 0.0) or np.any(temperature_K <= 0.0)):
        raise ValueError("Inputs must be five positive fitted rates, variances, and temperatures.")

    R = 8.314462618
    z_975 = 1.959963984540054
    y = np.log(k_hat_s_inv)
    inverse_temperature = 1.0 / temperature_K
    xbar = np.mean(inverse_temperature)
    x = inverse_temperature - xbar
    design = np.column_stack((np.ones_like(x), x))
    weights = 1.0 / var_lnk
    weighted_information = design.T @ (weights[:, None] * design)
    weighted_covariance_coefficients = np.linalg.inv(weighted_information)
    weighted_alpha, weighted_b = weighted_covariance_coefficients @ (design.T @ (weights * y))
    unweighted_information = design.T @ design
    unweighted_covariance_inverse = np.linalg.inv(unweighted_information)
    unweighted_alpha, unweighted_b = unweighted_covariance_inverse @ (design.T @ y)
    unweighted_residuals = y - design @ np.array([unweighted_alpha, unweighted_b])
    unweighted_covariance_coefficients = ((unweighted_residuals @ unweighted_residuals) / 3.0) * unweighted_covariance_inverse
    transform = np.array([[0.0, -R], [1.0, -xbar]])
    weighted_Ea_hat, weighted_lnA_hat = map_arrhenius_parameters(R, weighted_b, weighted_alpha, xbar)
    weighted_covariance = transform @ weighted_covariance_coefficients @ transform.T
    unweighted_Ea_hat, unweighted_lnA_hat = map_arrhenius_parameters(R, unweighted_b, unweighted_alpha, xbar)
    unweighted_covariance = transform @ unweighted_covariance_coefficients @ transform.T

    def intervals(estimate_Ea, estimate_lnA, covariance):
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        return {"Ea_hat_95": (estimate_Ea - z_975 * standard_errors[0], estimate_Ea + z_975 * standard_errors[0]), "lnA_hat_95": (estimate_lnA - z_975 * standard_errors[1], estimate_lnA + z_975 * standard_errors[1])}

    return {"Ea_hat": weighted_Ea_hat, "lnA_hat": weighted_lnA_hat, "covariance": weighted_covariance, "intervals": intervals(weighted_Ea_hat, weighted_lnA_hat, weighted_covariance), "weighted": {"Ea_hat": weighted_Ea_hat, "lnA_hat": weighted_lnA_hat, "covariance": weighted_covariance, "intervals": intervals(weighted_Ea_hat, weighted_lnA_hat, weighted_covariance)}, "unweighted": {"Ea_hat": unweighted_Ea_hat, "lnA_hat": unweighted_lnA_hat, "covariance": unweighted_covariance, "intervals": intervals(unweighted_Ea_hat, unweighted_lnA_hat, unweighted_covariance)}}


# S5
def summarize_monte_carlo(run_results: list[dict]) -> dict:
    ea_true = 50000.0
    ln_a_true = float(np.log(100000000.0))
    z_975 = 1.959963984540054

    def first_value(result, names):
        for name in names:
            if name in result and result[name] is not None:
                return result[name]
        return None

    def interval_contains(result, estimate_names, se_names, low_names, high_names, covered_names, truth):
        covered = first_value(result, covered_names)
        if covered is not None:
            return bool(covered)
        low, high = first_value(result, low_names), first_value(result, high_names)
        if low is not None and high is not None:
            return bool(float(low) <= truth <= float(high))
        estimate, se = first_value(result, estimate_names), first_value(result, se_names)
        if estimate is not None and se is not None:
            return bool(float(estimate) - z_975 * float(se) <= truth <= float(estimate) + z_975 * float(se))
        return None

    ea_values, ln_a_values, ea_covered, ln_a_covered = [], [], [], []
    for result in run_results:
        if not bool(result.get("success", True)):
            continue
        ea = first_value(result, ("ea_hat", "Ea_hat", "ea_estimate", "Ea_estimate"))
        ln_a = first_value(result, ("ln_a_hat", "lnA_hat", "ln_a_estimate", "lnA_estimate"))
        if ea is None or ln_a is None or not (np.isfinite(float(ea)) and np.isfinite(float(ln_a))):
            continue
        ea_values.append(float(ea)); ln_a_values.append(float(ln_a))
        ea_inside = interval_contains(result, ("ea_hat", "Ea_hat", "ea_estimate", "Ea_estimate"), ("ea_se", "Ea_se", "ea_standard_error", "Ea_standard_error"), ("ea_ci_low", "Ea_ci_low", "ea_lower", "Ea_lower"), ("ea_ci_high", "Ea_ci_high", "ea_upper", "Ea_upper"), ("ea_covered", "Ea_covered", "ea_contains_true", "Ea_contains_true"), ea_true)
        ln_a_inside = interval_contains(result, ("ln_a_hat", "lnA_hat", "ln_a_estimate", "lnA_estimate"), ("ln_a_se", "lnA_se", "ln_a_standard_error", "lnA_standard_error"), ("ln_a_ci_low", "lnA_ci_low", "ln_a_lower", "lnA_lower"), ("ln_a_ci_high", "lnA_ci_high", "ln_a_upper", "lnA_upper"), ("ln_a_covered", "lnA_covered", "ln_a_contains_true", "lnA_contains_true"), ln_a_true)
        if ea_inside is not None: ea_covered.append(ea_inside)
        if ln_a_inside is not None: ln_a_covered.append(ln_a_inside)

    ea_values, ln_a_values = np.asarray(ea_values, dtype=float), np.asarray(ln_a_values, dtype=float)
    n_success, n_total = int(ea_values.size), len(run_results)

    def estimate_summary(values, truth, covered):
        n = values.size
        if n == 0:
            return {"bias": np.nan, "bias_se": np.nan, "spread": np.nan, "spread_se": np.nan, "coverage": np.nan, "coverage_se": np.nan, "coverage_count": 0, "coverage_trials": 0}
        spread = float(np.std(values, ddof=1)) if n > 1 else np.nan
        bias = float(np.mean(values) - truth)
        coverage_trials = len(covered)
        coverage_count = int(np.sum(covered))
        coverage = float(coverage_count / coverage_trials) if coverage_trials else np.nan
        return {"bias": bias, "bias_se": float(spread / np.sqrt(n)) if n > 1 else np.nan, "spread": spread, "spread_se": float(spread / np.sqrt(2.0 * (n - 1))) if n > 1 else np.nan, "coverage": coverage, "coverage_se": float(np.sqrt(coverage * (1.0 - coverage) / coverage_trials)) if coverage_trials else np.nan, "coverage_count": coverage_count, "coverage_trials": coverage_trials}

    ea_summary, ln_a_summary = estimate_summary(ea_values, ea_true, ea_covered), estimate_summary(ln_a_values, ln_a_true, ln_a_covered)
    if n_success > 1:
        correlation = float(np.corrcoef(ea_values, ln_a_values)[0, 1])
        correlation_se = float((1.0 - correlation ** 2) / np.sqrt(n_success - 3)) if n_success > 3 and np.isfinite(correlation) else np.nan
    else:
        correlation, correlation_se = np.nan, np.nan
    return {"n_total": n_total, "n_success": n_success, "n_failure": n_total - n_success, "ea": ea_summary, "ln_a": ln_a_summary, "ea_bias": ea_summary["bias"], "ea_bias_se": ea_summary["bias_se"], "ea_spread": ea_summary["spread"], "ea_spread_se": ea_summary["spread_se"], "ea_coverage": ea_summary["coverage"], "ea_coverage_se": ea_summary["coverage_se"], "ea_coverage_count": ea_summary["coverage_count"], "ln_a_bias": ln_a_summary["bias"], "ln_a_bias_se": ln_a_summary["bias_se"], "ln_a_spread": ln_a_summary["spread"], "ln_a_spread_se": ln_a_summary["spread_se"], "ln_a_coverage": ln_a_summary["coverage"], "ln_a_coverage_se": ln_a_summary["coverage_se"], "ln_a_coverage_count": ln_a_summary["coverage_count"], "ea_ln_a_correlation": correlation, "ea_ln_a_correlation_se": correlation_se}
