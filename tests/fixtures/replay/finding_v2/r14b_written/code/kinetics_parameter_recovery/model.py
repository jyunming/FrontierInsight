"""Independent Gaussian-noise concentration trajectories are generated from exact first-order decay, fitted for one positive k per temperature, and transformed through a two-stage Arrhenius regression."""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import least_squares


# E1
def first_order_concentration(c0, k, t) -> float:
    return c0 * math.exp(-k * t)


# E2
def arrhenius_rate_constant(A, Ea, R, T) -> float:
    # Compute one rate constant from the Arrhenius equation.
    return A * math.exp(-Ea / (R * T))


# E3
# E3
def arrhenius_log_rate(A, Ea, R, T) -> float:
    return math.log(arrhenius_rate_constant(A, Ea, R, T))


# E4
def log_rate_variance(var_k, k_hat) -> float:
    return var_k / k_hat**2


# E5
def add_gaussian_noise(c_true, sigma_c, rng) -> float:
    return float(c_true + rng.normal(0.0, sigma_c))


# S1
def construct_temperature_grid(A, Ea, R, k) -> float:
    # Invert E2: k = A * exp(-Ea / (R * T)).
    T = -Ea / (R * __import__("math").log(k / A))
    arrhenius_rate_constant(A, Ea, R, T)
    return T


# S2
def simulate_trace(k, n_time_points, noise_sd_fraction_c0) -> tuple:
    rng = np.random.default_rng()
    time_s = np.linspace(0.0, 3.0 / k, n_time_points)
    sigma_c = noise_sd_fraction_c0 * 1.0
    observed_concentration = np.array(
        [
            add_gaussian_noise(
                first_order_concentration(1.0, k, time),
                sigma_c,
                rng,
            )
            for time in time_s
        ]
    )
    return time_s, observed_concentration


# S3
def fit_rate_constant(time_s, concentration, c0) -> tuple:
    # Fit one bounded positive rate and return k_hat, var_k, and var_ln_k.
    time_s = np.asarray(time_s, dtype=float)
    concentration = np.asarray(concentration, dtype=float)

    def residuals(params):
        k = params[0]
        fitted = np.asarray(
            [first_order_concentration(c0, k, t) for t in time_s],
            dtype=float,
        )
        return fitted - concentration

    result = least_squares(
        residuals,
        x0=np.array([1.0]),
        bounds=(np.array([1e-8]), np.array([100.0])),
        max_nfev=2000,
    )

    k_hat = float(result.x[0])
    residual = result.fun
    dof = time_s.size - 1
    residual_variance = float(np.dot(residual, residual) / dof)
    jtj_inv = np.linalg.inv(result.jac.T @ result.jac)
    var_k = float(residual_variance * jtj_inv[0, 0])
    var_ln_k = float(log_rate_variance(var_k, k_hat))
    return k_hat, var_k, var_ln_k


# S4
def fit_arrhenius_methods(temperature_K, k_hat, var_ln_k) -> dict:
    temperature_K = np.asarray(temperature_K, dtype=float)
    k_hat = np.asarray(k_hat, dtype=float)
    var_ln_k = np.asarray(var_ln_k, dtype=float)

    if temperature_K.shape != (5,) or k_hat.shape != (5,) or var_ln_k.shape != (5,):
        raise ValueError("temperature_K, k_hat, and var_ln_k must each contain five values.")
    if np.any(temperature_K <= 0.0) or np.any(k_hat <= 0.0):
        raise ValueError("temperatures and fitted rate constants must be positive.")
    if np.any(var_ln_k <= 0.0):
        raise ValueError("var_ln_k values must be positive.")

    R = 8.314462618
    t_critical = 3.182446
    normal_critical = 1.959963984540054
    degrees_of_freedom = 3

    x = 1.0 / temperature_K
    y = np.log(k_hat)

    # Reconstruct the supplied delta-method log-rate variances through E4.
    log_variances = np.array(
        [log_rate_variance(v * k * k, k) for v, k in zip(var_ln_k, k_hat)]
    )

    results = {}
    for weighted in (False, True):
        for centred in (False, True):
            x_origin = float(np.mean(x)) if centred else 0.0
            x_fit = x - x_origin
            design = np.column_stack((np.ones_like(x_fit), x_fit))
            weights = 1.0 / log_variances if weighted else np.ones_like(log_variances)

            normal_matrix = design.T @ (weights[:, None] * design)
            coefficient = np.linalg.solve(normal_matrix, design.T @ (weights * y))
            centred_intercept, slope = coefficient

            ln_A_hat = centred_intercept - slope * x_origin
            Ea_hat = -R * slope

            fitted_y = np.array(
                [
                    arrhenius_log_rate(np.exp(ln_A_hat), Ea_hat, R, T)
                    for T in temperature_K
                ]
            )
            residuals = y - fitted_y
            residual_variance = float(np.sum(weights * residuals**2) / degrees_of_freedom)

            covariance_fit = residual_variance * np.linalg.inv(normal_matrix)
            transform = np.array(
                [
                    [-R, 0.0],
                    [-x_origin, 1.0],
                ]
            )
            covariance = transform @ covariance_fit @ transform.T

            standard_errors = np.sqrt(np.diag(covariance))
            t_intervals = np.array(
                [
                    [Ea_hat - t_critical * standard_errors[0],
                     Ea_hat + t_critical * standard_errors[0]],
                    [ln_A_hat - t_critical * standard_errors[1],
                     ln_A_hat + t_critical * standard_errors[1]],
                ]
            )
            normal_intervals = np.array(
                [
                    [Ea_hat - normal_critical * standard_errors[0],
                     Ea_hat + normal_critical * standard_errors[0]],
                    [ln_A_hat - normal_critical * standard_errors[1],
                     ln_A_hat + normal_critical * standard_errors[1]],
                ]
            )

            method_name = f"{'weighted' if weighted else 'unweighted'}_{'centred' if centred else 'uncentred'}"
            results[method_name] = {
                "Ea_hat": float(Ea_hat),
                "ln_A_hat": float(ln_A_hat),
                "covariance": covariance,
                "t_95_intervals": {
                    "Ea": t_intervals[0],
                    "ln_A": t_intervals[1],
                },
                "normal_95_intervals": {
                    "Ea": normal_intervals[0],
                    "ln_A": normal_intervals[1],
                },
            }

    return results


# S5
def summarise_monte_carlo(trial_estimates) -> dict:
    import math
    import numpy as np

    z_value = 1.959963984540054

    def wilson_interval(successes, total):
        if total == 0:
            return [float("nan"), float("nan")]
        denominator = 1.0 + z_value**2 / total
        centre = (successes / total + z_value**2 / (2.0 * total)) / denominator
        half_width = (
            z_value
            * math.sqrt(
                (successes / total) * (1.0 - successes / total) / total
                + z_value**2 / (4.0 * total**2)
            )
            / denominator
        )
        return [float(centre - half_width), float(centre + half_width)]

    grouped = {}
    for trial in trial_estimates:
        key = (
            trial.get("noise_level", trial.get("noise")),
            trial.get("n_time_points", trial.get("n_points")),
            trial.get("method", "default"),
        )
        grouped.setdefault(key, []).append(trial)

    results = []
    for (noise_level, n_time_points, method), trials in grouped.items():
        completed = [
            trial for trial in trials
            if trial.get("completed", trial.get("success", True))
            and "Ea_hat" in trial
            and "ln_A_hat" in trial
        ]
        failures = len(trials) - len(completed)

        ea_true = completed[0].get("Ea_true", 50000.0) if completed else 50000.0
        ln_a_true = completed[0].get("ln_A_true", math.log(1000000.0)) if completed else math.log(1000000.0)

        ea_values = np.asarray([trial["Ea_hat"] for trial in completed], dtype=float)
        ln_a_values = np.asarray([trial["ln_A_hat"] for trial in completed], dtype=float)
        ea_errors = ea_values - ea_true
        ln_a_errors = ln_a_values - ln_a_true

        def estimate_metrics(values, errors, truth, interval_key):
            count = len(values)
            if count == 0:
                return {
                    "n_completed": 0,
                    "bias": float("nan"),
                    "standard_deviation": float("nan"),
                    "rmse": float("nan"),
                    "coverage": float("nan"),
                    "coverage_wilson_95": [float("nan"), float("nan")],
                }

            covered = np.asarray([
                trial[interval_key][0] <= truth <= trial[interval_key][1]
                for trial in completed
                if interval_key in trial
            ], dtype=bool)
            coverage_count = int(covered.sum())
            coverage_total = int(covered.size)
            return {
                "n_completed": count,
                "bias": float(np.mean(errors)),
                "standard_deviation": float(np.std(values, ddof=1)) if count > 1 else 0.0,
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "coverage": float(coverage_count / coverage_total) if coverage_total else float("nan"),
                "coverage_wilson_95": wilson_interval(coverage_count, coverage_total),
            }

        ea_metrics = estimate_metrics(ea_values, ea_errors, ea_true, "Ea_interval")
        ln_a_metrics = estimate_metrics(ln_a_values, ln_a_errors, ln_a_true, "ln_A_interval")

        if len(completed) > 1:
            correlation = float(np.corrcoef(ea_values, ln_a_values)[0, 1])
            correlation_se = float(
                (1.0 - correlation**2) / math.sqrt(len(completed) - 3)
            ) if len(completed) > 3 else float("nan")
        else:
            correlation = float("nan")
            correlation_se = float("nan")

        results.append({
            "noise_level": noise_level,
            "n_time_points": n_time_points,
            "method": method,
            "n_trials": len(trials),
            "n_completed": len(completed),
            "n_failures": failures,
            "Ea": ea_metrics,
            "ln_A": ln_a_metrics,
            "Ea_ln_A_correlation": correlation,
            "Ea_ln_A_correlation_standard_error": correlation_se,
        })

    return {"results": results}

