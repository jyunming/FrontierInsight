import math

import numpy as np
from scipy import optimize

from kinetics_parameter_recovery import model


C0 = 1.0
A = 1.0e10
EA = 50000.0
R = 8.314
Z95 = 1.96

NOISE_RATIOS = (0.001, 0.01, 0.05, 0.10)
NUM_TIME_POINTS = (5, 10, 20)


def _temperatures():
    t_low = 300.0
    t_high = 1.0 / (1.0 / t_low - R * math.log(10.0) / EA)
    return np.linspace(1.0 / t_low, 1.0 / t_high, 5) ** -1


def _fit_k(times, observations, sigma, k_reference):
    times = np.asarray(times, dtype=float)
    observations = np.asarray(observations, dtype=float)
    residual_scale = sigma if sigma > 0.0 else 1.0

    def residuals(log_k):
        k = math.exp(float(log_k[0]))
        predicted = C0 * np.exp(-k * times)
        return (observations - predicted) / residual_scale

    def jacobian(log_k):
        k = math.exp(float(log_k[0]))
        return (C0 * k * times * np.exp(-k * times) / residual_scale)[:, None]

    def objective(log_k):
        residual = residuals(np.atleast_1d(log_k))
        return float(np.dot(residual, residual))

    def objective_gradient(log_k):
        point = np.atleast_1d(log_k)
        residual = residuals(point)
        jacobian_value = jacobian(point)[:, 0]
        return np.asarray([2.0 * np.dot(jacobian_value, residual)])

    initial_log_k = np.asarray([math.log(k_reference)])
    result = optimize.least_squares(
        residuals,
        initial_log_k,
        jac=jacobian,
        method="trf",
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )

    if result.success and np.isfinite(result.cost):
        log_k = float(result.x[0])
    else:
        fallback = optimize.minimize_scalar(
            lambda value: objective(np.asarray([value])),
            bounds=(float(initial_log_k[0]) - 12.0, float(initial_log_k[0]) + 12.0),
            method="bounded",
            options={"xatol": 1.0e-12},
        )
        if not fallback.success or not np.isfinite(fallback.fun):
            raise RuntimeError("rate fit did not converge")
        log_k = float(fallback.x)

    # Certify the scalar optimum by solving its analytic first-order condition.
    # A least-squares step-tolerance exit can be near, but not sufficiently near,
    # this root for the protocol's independently checked gradient tolerance.
    centre = log_k
    step = 1.0e-7
    left = centre - step
    right = centre + step
    left_gradient = float(objective_gradient(np.asarray([left]))[0])
    right_gradient = float(objective_gradient(np.asarray([right]))[0])
    bracket_found = left_gradient <= 0.0 <= right_gradient
    for _ in range(60):
        if bracket_found:
            break
        step *= 2.0
        left = centre - step
        right = centre + step
        left_gradient = float(objective_gradient(np.asarray([left]))[0])
        right_gradient = float(objective_gradient(np.asarray([right]))[0])
        bracket_found = left_gradient <= 0.0 <= right_gradient

    if bracket_found:
        root = optimize.root_scalar(
            lambda value: float(objective_gradient(np.asarray([value]))[0]),
            bracket=(left, right),
            method="brentq",
            xtol=5.0e-15,
            rtol=1.0e-14,
        )
        if not root.converged or not np.isfinite(root.root):
            raise RuntimeError("rate-fit score refinement did not converge")
        root_log_k = float(root.root)
        if objective(np.asarray([root_log_k])) <= objective(np.asarray([log_k])) + 1.0e-12:
            log_k = root_log_k

    result_objective = objective(np.asarray([log_k]))
    k_hat = math.exp(log_k)
    predicted = C0 * np.exp(-k_hat * times)
    derivative = -C0 * times * np.exp(-k_hat * times)
    information = float(np.dot(derivative, derivative) / residual_scale**2)
    gradient = float(objective_gradient(np.asarray([log_k]))[0])
    normalized_gradient = abs(gradient) / (
        max(1.0, abs(log_k)) * max(1.0, abs(result_objective))
    )
    reference_objective = objective(initial_log_k)
    if (
        not np.isfinite(information)
        or information <= 0.0
        or normalized_gradient > 1.0e-6
        or result_objective > reference_objective + 1.0e-8 * max(1.0, reference_objective)
    ):
        raise RuntimeError("rate-fit acceptance invariant failed")
    return k_hat, math.sqrt(1.0 / information)


def _fit_arrhenius(k_hat, sigma_k, temperatures):
    k_hat = np.asarray(k_hat, dtype=float)
    sigma_k = np.asarray(sigma_k, dtype=float)
    if np.any(k_hat <= 0.0) or np.any(~np.isfinite(k_hat)):
        raise RuntimeError("invalid fitted rate")
    if np.any(sigma_k <= 0.0) or np.any(~np.isfinite(sigma_k)):
        raise RuntimeError("invalid fitted-rate uncertainty")

    x_raw = 1.0 / np.asarray(temperatures, dtype=float)
    y = np.log(k_hat)
    variance_y = (sigma_k / k_hat) ** 2
    weights = 1.0 / variance_y
    x_bar = float(np.sum(weights * x_raw) / np.sum(weights))
    x = x_raw - x_bar
    design = np.column_stack([np.ones_like(x), x])
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(normal)
    beta = covariance @ (design.T @ (weights * y))
    intercept_centered, slope = beta
    ln_a = float(intercept_centered - slope * x_bar)
    ea = float(-R * slope)

    transform = np.array([[0.0, -R], [1.0, -x_bar]])
    cov_ea_lna = transform @ covariance @ transform.T
    if not np.all(np.isfinite(cov_ea_lna)) or np.any(np.diag(cov_ea_lna) < 0.0):
        raise RuntimeError("invalid Arrhenius covariance")
    return ea, ln_a, cov_ea_lna, variance_y


def _fit_arrhenius_equal_weight(k_hat, temperatures):
    k_hat = np.asarray(k_hat, dtype=float)
    if np.any(k_hat <= 0.0) or np.any(~np.isfinite(k_hat)):
        raise RuntimeError("invalid fitted rate")

    x_raw = 1.0 / np.asarray(temperatures, dtype=float)
    y = np.log(k_hat)
    x_bar = float(np.mean(x_raw))
    x = x_raw - x_bar
    design = np.column_stack([np.ones_like(x), x])
    normal = design.T @ design
    covariance_beta = np.linalg.inv(normal)
    beta = covariance_beta @ (design.T @ y)
    intercept_centered, slope = beta
    residual = y - design @ beta
    variance_y = float(np.dot(residual, residual) / (len(y) - 2))
    covariance_beta *= variance_y
    ln_a = float(intercept_centered - slope * x_bar)
    ea = float(-R * slope)

    transform = np.array([[0.0, -R], [1.0, -x_bar]])
    cov_ea_lna = transform @ covariance_beta @ transform.T
    if not np.all(np.isfinite(cov_ea_lna)) or np.any(np.diag(cov_ea_lna) < 0.0):
        raise RuntimeError("invalid Arrhenius covariance")
    return ea, ln_a, cov_ea_lna, np.full(len(k_hat), variance_y)


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    if {"c0", "k", "t"}.issubset(cell):
        return {"c_val": model.first_order_concentration(cell["c0"], cell["k"], cell["t"])}
    if {"A", "Ea", "R", "T"}.issubset(cell) and "noise_ratio" not in cell:
        rate = model.arrhenius_rate(cell["A"], cell["Ea"], cell["R"], cell["T"])
        return {"k_val": rate, "ln_k_val": model.log_arrhenius_rate(cell["A"], cell["Ea"], cell["R"], cell["T"])}

    noise_ratio = float(cell["noise_ratio"])
    n_times = int(cell["num_time_points"])
    if noise_ratio != 0.0 and noise_ratio not in NOISE_RATIOS:
        raise RuntimeError("noise_ratio is not in the simulation grid")
    if n_times not in NUM_TIME_POINTS:
        raise RuntimeError("num_time_points is not in the simulation grid")

    temperatures = _temperatures()
    true_k = np.asarray([model.arrhenius_rate(A, EA, R, temp) for temp in temperatures])
    rng = np.random.default_rng(seed)
    k_hat, sigma_k = [], []

    for rate in true_k:
        times = np.linspace(0.0, 3.0 / rate, n_times)
        # E1: evaluation delegated to the tested model package.
        clean = np.asarray([model.first_order_concentration(C0, rate, t) for t in times])
        observed = clean + rng.normal(0.0, noise_ratio * C0, size=n_times)
        fitted_rate, fitted_sigma = _fit_k(times, observed, noise_ratio * C0, rate)
        k_hat.append(fitted_rate)
        sigma_k.append(fitted_sigma)

    if noise_ratio == 0.0:
        ea_hat, lna_hat, covariance, ln_k_variance = _fit_arrhenius_equal_weight(k_hat, temperatures)
    else:
        ea_hat, lna_hat, covariance, ln_k_variance = _fit_arrhenius(k_hat, sigma_k, temperatures)

    ea_se = math.sqrt(float(covariance[0, 0]))
    lna_se = math.sqrt(float(covariance[1, 1]))
    output = {"ea_fit": ea_hat, "Ea_fit": ea_hat, "ea_se": ea_se, "ea_coverage": float(abs(ea_hat - EA) <= Z95 * ea_se), "lnA_fit": lna_hat, "lnA_se": lna_se, "lnA_coverage": float(abs(lna_hat - math.log(A)) <= Z95 * lna_se)}
    for index, (rate, estimate, se, variance) in enumerate(zip(true_k, k_hat, sigma_k, ln_k_variance)):
        output[f"k_fit_{index}"] = float(estimate)
        output[f"k_se_{index}"] = float(se)
        output[f"k_coverage_{index}"] = float(abs(estimate - rate) <= Z95 * se)
        output[f"ln_k_variance_{index}"] = float(variance)
    return output
