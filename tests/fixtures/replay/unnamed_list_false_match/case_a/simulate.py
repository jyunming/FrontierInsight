import math
import os

import numpy as np
from scipy.optimize import least_squares

from kinetics_parameter_recovery import model


C0 = 1.0
LN_A_TRUE = math.log(1.0e8)
EA_TRUE = 50000.0
R = 8.314462618
K_REF = 0.1
RATE_BOUNDS = (1.0e-6, 2.0)
EA_BOUNDS = (1000.0, 200000.0)
LN_A_BOUNDS = (0.0, 35.0)


def _design():
    k_values = K_REF * np.logspace(-0.5, 0.5, 5)
    temperatures = EA_TRUE / (R * (LN_A_TRUE - np.log(k_values)))
    return k_values, temperatures


def _rate_variance(t, k_hat, noise_fraction):
    sigma = noise_fraction * C0
    prediction = np.asarray([model.concentration_true(float(x), k_hat, C0) for x in t])
    jacobian = -np.asarray(t) * prediction
    information = float(jacobian @ jacobian)
    return sigma**2 / information if information > 0 else float("nan")


def _two_step(times, observations, temperatures, noise_fraction):
    k_hats = []
    variances = []
    for t, y in zip(times, observations):
        k_hat = model.fit_rate_constant(t, y, C0)
        if not np.isfinite(k_hat) or k_hat <= 0:
            raise RuntimeError("nonpositive first-stage rate")
        k_hats.append(k_hat)
        variances.append(_rate_variance(t, k_hat, noise_fraction))

    k_hats = np.asarray(k_hats)
    x_raw = 1.0 / np.asarray(temperatures)
    x_bar = float(np.mean(x_raw))
    x = x_raw - x_bar
    y = np.log(k_hats)

    if noise_fraction == 0:
        weights = np.ones_like(y)
    else:
        variances = np.asarray(variances)
        if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
            raise RuntimeError("invalid log-rate variance")
        weights = 1.0 / variances

    design = np.column_stack([np.ones_like(x), x])
    normal_matrix = design.T @ (weights[:, None] * design)
    covariance_ab = np.linalg.inv(normal_matrix)
    alpha, beta = covariance_ab @ (design.T @ (weights * y))
    ea_hat, ln_a_hat = model.arrhenius_parameters_from_centered_fit(
        alpha, beta, x_bar, R
    )

    transform = np.array([[0.0, -R], [1.0, -x_bar]])
    covariance = transform @ covariance_ab @ transform.T
    return {
        "ea": float(ea_hat),
        "ln_a": float(ln_a_hat),
        "var_ea": float(covariance[0, 0]),
        "var_ln_a": float(covariance[1, 1]),
        "cov_ea_ln_a": float(covariance[0, 1]),
        "k_hats": k_hats,
        "var_ln_k": np.asarray(
            [model.log_rate_variance(k, v) for k, v in zip(k_hats, variances)]
        ),
    }


def _global(times, observations, temperatures, noise_fraction):
    t_all = np.concatenate(times)
    y_all = np.concatenate(observations)
    temp_all = np.concatenate(
        [np.full(len(t), temperature) for t, temperature in zip(times, temperatures)]
    )
    sigma = noise_fraction * C0 if noise_fraction > 0 else 1.0

    def residual(parameters):
        ln_a, ea = parameters
        rates = np.asarray(
            [model.arrhenius_rate_ln_a(ln_a, ea, R, temp) for temp in temp_all]
        )
        predicted = np.asarray(
            [model.concentration_true(t, rate, C0) for t, rate in zip(t_all, rates)]
        )
        return (predicted - y_all) / sigma

    def jacobian(parameters):
        ln_a, ea = parameters
        rates = np.asarray(
            [model.arrhenius_rate_ln_a(ln_a, ea, R, temp) for temp in temp_all]
        )
        predicted = np.asarray(
            [model.concentration_true(t, rate, C0) for t, rate in zip(t_all, rates)]
        )
        d_ln_a = -t_all * rates * predicted / sigma
        d_ea = t_all * rates * predicted / (R * temp_all * sigma)
        return np.column_stack([d_ln_a, d_ea])

    result = least_squares(
        residual,
        x0=np.array([18.0, 45000.0]),
        jac=jacobian,
        bounds=([LN_A_BOUNDS[0], EA_BOUNDS[0]], [LN_A_BOUNDS[1], EA_BOUNDS[1]]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError("global optimizer did not converge")

    covariance_lnea = np.linalg.inv(result.jac.T @ result.jac)
    covariance = covariance_lnea[np.ix_([1, 0], [1, 0])]
    return {
        "ea": float(result.x[1]),
        "ln_a": float(result.x[0]),
        "var_ea": float(covariance[0, 0]),
        "var_ln_a": float(covariance[1, 1]),
        "cov_ea_ln_a": float(covariance[0, 1]),
    }


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    noise_fraction = float(cell["noise_fraction"])
    n_time = int(cell["n_time"])
    rng = np.random.default_rng(seed)
    k_values, temperatures = _design()

    times, true_values, observations = [], [], []
    for k in k_values:
        t = np.linspace(0.0, 3.0 / k, n_time)
        c_true = np.asarray([model.concentration_true(float(x), float(k), C0) for x in t])
        c_obs = np.asarray(
            [model.concentration_observed(float(c), noise_fraction, C0, rng) for c in c_true]
        )
        times.append(t)
        true_values.append(c_true)
        observations.append(c_obs)

    two = _two_step(times, observations, temperatures, noise_fraction)
    global_fit = _global(times, observations, temperatures, noise_fraction)

    output = {
        "two_step_ea": two["ea"],
        "two_step_ln_a": two["ln_a"],
        "two_step_var_ea": two["var_ea"],
        "two_step_var_ln_a": two["var_ln_a"],
        "two_step_cov_ea_ln_a": two["cov_ea_ln_a"],
        "global_ea": global_fit["ea"],
        "global_ln_a": global_fit["ln_a"],
        "global_var_ea": global_fit["var_ea"],
        "global_var_ln_a": global_fit["var_ln_a"],
        "global_cov_ea_ln_a": global_fit["cov_ea_ln_a"],
        "center_temperature_second_observed_concentration_mol_l": float(observations[2][1]),
        "observation_residual_max_abs": float(
            np.max(np.abs(np.concatenate(observations) - np.concatenate(true_values)))
        ),
    }

    for index, (k_hat, var_ln_k) in enumerate(zip(two["k_hats"], two["var_ln_k"])):
        output[f"k_hat_{index}"] = float(k_hat)
        output[f"var_ln_k_{index}"] = float(var_ln_k)
        for multiplier in (0, 1, 2, 3):
            position = round(multiplier * (n_time - 1) / 3)
            output[f"observed_concentration_t{index}_m{multiplier}"] = float(
                observations[index][position]
            )
    return output