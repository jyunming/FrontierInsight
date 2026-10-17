import json
import math

import numpy as np

from kinetics_parameter_recovery import model

C0 = 1.0
A_TRUE = 1.0e6
EA_TRUE = 50000.0
R_GAS = 8.314462618
K_LOWER_BOUND = 1.0e-8
K_UPPER_BOUND = 100.0
METHODS = (
    "weighted_centered",
    "unweighted_uncentered",
    "weighted_uncentered",
    "unweighted_centered",
)
CALIBRATION_TRIALS = 400


def _seeded_package_trace(k, n_time_points, noise, rng):
    """Call the supplied S2 implementation while giving it FI's generator."""
    original_default_rng = model.np.random.default_rng
    model.np.random.default_rng = lambda *args, **kwargs: rng
    try:
        return model.simulate_trace(k, n_time_points, noise)
    finally:
        model.np.random.default_rng = original_default_rng


def _method_key(temperature_index, method):
    return f"temperature_index={temperature_index},arrhenius_method={method}"


def _package_method_name(method):
    return method.replace("centered", "centred").replace("uncentered", "uncentred")


def _expand_by_temperature(values_by_method):
    """Repeat each five-temperature Arrhenius result on FI's trial grid."""
    return {
        _method_key(temperature_index, method): float(values_by_method[method])
        for temperature_index in range(5)
        for method in METHODS
    }


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    """Run one independent five-temperature simulated experiment."""
    del trial_id
    rng = np.random.default_rng(seed)
    noise = float(cell["noise_sd_fraction_c0"])
    n_points = int(cell["n_time_points"])

    k_true = np.geomspace(0.1, 1.0, 5)
    # E2: supplied package implementation is used by S1.
    temperatures = np.asarray(
        [model.construct_temperature_grid(A_TRUE, EA_TRUE, R_GAS, k) for k in k_true],
        dtype=float,
    )

    fitted_k, var_k, var_ln_k = [], [], []
    final_true = None
    middle_error_sum = 0.0
    middle_error_sq_sum = 0.0
    middle_error_count = 0
    for index, k in enumerate(k_true):
        # E1 and E5: supplied S2 implementation evaluates the decay and noise model.
        time_s, concentration = _seeded_package_trace(k, n_points, noise, rng)
        if index == 2:
            true_concentration = np.asarray(
                [model.first_order_concentration(C0, k, time) for time in time_s],
                dtype=float,
            )
            errors = concentration - true_concentration
            middle_error_sum = float(np.sum(errors))
            middle_error_sq_sum = float(np.dot(errors, errors))
            middle_error_count = int(errors.size)
            final_true = model.first_order_concentration(C0, k, time_s[-1])

        kh, vk, vlk = model.fit_rate_constant(time_s, concentration, C0)
        if (
            not np.isfinite(kh)
            or not np.isfinite(vk)
            or not np.isfinite(vlk)
            or kh <= K_LOWER_BOUND
            or kh >= K_UPPER_BOUND
            or (noise > 0.0 and vlk <= 0.0)
        ):
            raise RuntimeError("invalid bounded nonlinear least-squares rate fit")
        fitted_k.append(kh)
        var_k.append(vk)
        var_ln_k.append(vlk)

    fitted_k = np.asarray(fitted_k, dtype=float)
    var_k = np.asarray(var_k, dtype=float)
    if noise == 0.0:
        # Exact traces have zero residual variance; equal positive weights express OLS.
        var_ln_k = np.ones(5, dtype=float)
    else:
        var_ln_k = np.asarray(var_ln_k, dtype=float)

    fits = model.fit_arrhenius_methods(temperatures, fitted_k, var_ln_k)
    method_results = {method: fits[_package_method_name(method)] for method in METHODS}

    k_hat_map = {}
    var_k_map = {}
    for temperature_index in range(5):
        for method in METHODS:
            key = _method_key(temperature_index, method)
            k_hat_map[key] = float(fitted_k[temperature_index])
            var_k_map[key] = float(var_k[temperature_index])

    ea_hat = {}
    lna_hat = {}
    ea_t_lower = {}
    ea_t_upper = {}
    lna_t_lower = {}
    lna_t_upper = {}
    ea_normal_lower = {}
    ea_normal_upper = {}
    lna_normal_lower = {}
    lna_normal_upper = {}
    reported_se = {}
    for method, item in method_results.items():
        covariance = np.asarray(item["covariance"], dtype=float)
        ti = item["t_95_intervals"]
        ni = item["normal_95_intervals"]
        ea_hat[method] = float(item["Ea_hat"])
        lna_hat[method] = float(item["ln_A_hat"])
        ea_t_lower[method], ea_t_upper[method] = map(float, ti["Ea"])
        lna_t_lower[method], lna_t_upper[method] = map(float, ti["ln_A"])
        ea_normal_lower[method], ea_normal_upper[method] = map(float, ni["Ea"])
        lna_normal_lower[method], lna_normal_upper[method] = map(float, ni["ln_A"])
        reported_se[method] = float(np.sqrt(covariance[0, 0]))

    x_centered = 1.0 / temperatures - np.mean(1.0 / temperatures)
    poly_coeff = np.polyfit(
        x_centered,
        np.log(fitted_k),
        1,
        w=np.sqrt(1.0 / var_ln_k),
    )
    poly_ea = -R_GAS * poly_coeff[0]
    weighted_ea = ea_hat["weighted_centered"]
    if middle_error_count < 2:
        raise RuntimeError("insufficient middle-temperature noise errors")
    trial_error_variance = (
        middle_error_sq_sum - middle_error_sum**2 / middle_error_count
    ) / (middle_error_count - 1)
    if trial_error_variance < 0.0:
        raise RuntimeError("middle-temperature empirical noise variance is negative")

    return {
        "k_hat": k_hat_map,
        "var_k": var_k_map,
        "k_hat_temperature_2": float(fitted_k[2]),
        "var_k_temperature_2": float(var_k[2]),
        "T_temperature_2": float(temperatures[2]),
        "c_true_final_temperature_2": float(final_true),
        "noise_error_sum_temperature_2": middle_error_sum,
        "noise_error_sq_sum_temperature_2": middle_error_sq_sum,
        "noise_error_count_temperature_2": middle_error_count,
        "empirical_noise_sd_temperature_2": float(math.sqrt(trial_error_variance)),
        "Ea_hat": _expand_by_temperature(ea_hat),
        "ln_A_hat": _expand_by_temperature(lna_hat),
        "Ea_t_lower": _expand_by_temperature(ea_t_lower),
        "Ea_t_upper": _expand_by_temperature(ea_t_upper),
        "ln_A_t_lower": _expand_by_temperature(lna_t_lower),
        "ln_A_t_upper": _expand_by_temperature(lna_t_upper),
        "Ea_normal_lower": _expand_by_temperature(ea_normal_lower),
        "Ea_normal_upper": _expand_by_temperature(ea_normal_upper),
        "ln_A_normal_lower": _expand_by_temperature(lna_normal_lower),
        "ln_A_normal_upper": _expand_by_temperature(lna_normal_upper),
        "reported_se": _expand_by_temperature(reported_se),
        "max_rel_diff_Ea_closed_form_vs_polyfit": float(
            abs(weighted_ea - poly_ea) / max(1.0, abs(weighted_ea))
        ),
        "max_rel_diff_weighted_centered_vs_uncentered": float(
            max(
                abs(ea_hat["weighted_centered"] - ea_hat["weighted_uncentered"])
                / max(1.0, abs(ea_hat["weighted_centered"])),
                abs(lna_hat["weighted_centered"] - lna_hat["weighted_uncentered"])
                / max(1.0, abs(lna_hat["weighted_centered"])),
            )
        ),
    }


def run_cell(cell: dict) -> dict:
    """Run the seeded calibration ensemble required by aggregate oracles."""
    base_seed = int(cell.get("seed", 20261015))
    trials = []
    for trial_id in range(CALIBRATION_TRIALS):
        seed_sequence = np.random.SeedSequence([base_seed, trial_id])
        trial_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
        trials.append(run_trial(cell, trial_id, trial_seed))

    result = dict(trials[0])
    k_values = np.asarray(
        [trial["k_hat_temperature_2"] for trial in trials], dtype=float
    )
    reported_variances = np.asarray(
        [trial["var_k_temperature_2"] for trial in trials], dtype=float
    )
    empirical_variance = float(np.var(k_values, ddof=1))
    if not np.isfinite(empirical_variance) or empirical_variance <= 0.0:
        raise RuntimeError("empirical middle-temperature rate variance is not positive")

    error_sum = float(sum(trial["noise_error_sum_temperature_2"] for trial in trials))
    error_sq_sum = float(
        sum(trial["noise_error_sq_sum_temperature_2"] for trial in trials)
    )
    error_count = int(sum(trial["noise_error_count_temperature_2"] for trial in trials))
    if error_count < 2:
        raise RuntimeError("insufficient independent noise errors for calibration")
    empirical_error_variance = (error_sq_sum - error_sum**2 / error_count) / (error_count - 1)
    if empirical_error_variance < 0.0:
        raise RuntimeError("empirical noise variance is negative")

    result["mean_reported_var_k_over_empirical_var_k_temperature_2"] = float(
        np.mean(reported_variances) / empirical_variance
    )
    result["empirical_noise_sd_temperature_2"] = float(math.sqrt(empirical_error_variance))
    return result


if __name__ == "__main__":
    output = run_cell({"noise_sd_fraction_c0": 0.01, "n_time_points": 17})
    print("RESULT_JSON: " + json.dumps(output, allow_nan=False, sort_keys=True))
