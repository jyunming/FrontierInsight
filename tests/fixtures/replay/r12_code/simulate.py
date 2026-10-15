# The supplied model's simulate_trajectory creates an unseeded default_rng internally.
# To honour FI's per-trial seed while calling the tested package implementation, this
# module temporarily supplies the trial generator to that package call.

import math
import numpy as np
from kinetics_parameter_recovery import model

C0 = 1.0
EA_TRUE_J_MOL = 50000.0
A_TRUE_S_INV = 1.0e7
R_J_MOL_K = 8.314462618
T_LOW_K = 450.0
N_TEMPERATURES = 5


def _seeded_trajectory(k_true: float, n_timepoints: int, noise_std_rel: float,
                       rng: np.random.Generator):
    factory = np.random.default_rng
    np.random.default_rng = lambda *args, **kwargs: rng
    try:
        return model.simulate_trajectory(k_true, n_timepoints, noise_std_rel)
    finally:
        np.random.default_rng = factory


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    noise_std_rel = float(cell["noise_std_rel"])
    n_timepoints = int(cell["n_timepoints"])

    temperatures = model.construct_temperatures(
        EA_TRUE_J_MOL, R_J_MOL_K, T_LOW_K, N_TEMPERATURES
    )
    true_k = np.asarray([
        model.arrhenius_rate(A_TRUE_S_INV, EA_TRUE_J_MOL, R_J_MOL_K, temp)
        for temp in temperatures
    ])

    k_hat = []
    var_ln_k = []
    for rate in true_k:
        time, _, observed = _seeded_trajectory(
            float(rate), n_timepoints, noise_std_rel, rng
        )
        fitted_rate, fitted_variance = model.fit_rate(time, observed)
        if not (np.isfinite(fitted_rate) and fitted_rate > 0.0 and
                np.isfinite(fitted_variance) and fitted_variance > 0.0):
            raise RuntimeError("Non-finite nonlinear rate-fit result.")
        k_hat.append(fitted_rate)
        var_ln_k.append(fitted_variance)

    ln_k_hat = np.log(np.asarray(k_hat))
    var_ln_k = np.asarray(var_ln_k)
    weighted = model.calibrate_arrhenius(
        ln_k_hat, var_ln_k, temperatures, R_J_MOL_K
    )
    unweighted = model.calibrate_arrhenius(
        ln_k_hat, np.full_like(var_ln_k, np.nan), temperatures, R_J_MOL_K
    )

    output = {
        "fit_success": 1.0,
        "k_relative_bias": float(np.mean(np.asarray(k_hat) / true_k - 1.0)),
    }
    for label, fit in (("weighted", weighted), ("unweighted", unweighted)):
        covariance = np.asarray(fit["covariance"], dtype=float)
        denominator = math.sqrt(covariance[0, 0] * covariance[1, 1])
        correlation = float(covariance[0, 1] / denominator) if denominator > 0 else float("nan")
        ea_hat = float(fit["Ea_hat"])
        ln_a_hat = float(fit["ln_A_hat"])
        ea_low, ea_high = map(float, fit["Ea_interval"])
        ln_a_low, ln_a_high = map(float, fit["ln_A_interval"])
        output.update({
            f"{label}_Ea_hat": ea_hat,
            f"{label}_ln_A_hat": ln_a_hat,
            f"{label}_Ea_coverage": float(ea_low <= EA_TRUE_J_MOL <= ea_high),
            f"{label}_ln_A_coverage": float(
                ln_a_low <= math.log(A_TRUE_S_INV) <= ln_a_high
            ),
            f"{label}_Ea_interval_width": ea_high - ea_low,
            f"{label}_ln_A_interval_width": ln_a_high - ln_a_low,
            f"{label}_Ea_ln_A_correlation": correlation,
            f"{label}_Ea_squared_error": (ea_hat - EA_TRUE_J_MOL) ** 2,
            f"{label}_ln_A_squared_error": (
                ln_a_hat - math.log(A_TRUE_S_INV)
            ) ** 2,
        })
    return output


def oracle() -> dict:
    temperatures = model.construct_temperatures(
        EA_TRUE_J_MOL, R_J_MOL_K, T_LOW_K, N_TEMPERATURES
    )
    rates = np.asarray([
        model.arrhenius_rate(A_TRUE_S_INV, EA_TRUE_J_MOL, R_J_MOL_K, temp)
        for temp in temperatures
    ])
    endpoint_error = abs(
        model.concentration_true(C0, rates[0], 3.0 / rates[0]) - math.exp(-3.0)
    )
    rate_ratio_error = abs(rates[-1] / rates[0] - 10.0)
    return {
        "three_time_constants_endpoint": float(endpoint_error),
        "arrhenius_rate_ratio_tenfold": float(rate_ratio_error),
    }
