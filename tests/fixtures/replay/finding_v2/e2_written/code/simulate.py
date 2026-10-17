import math
import numpy as np

from kinetics_parameter_recovery import model

C0 = 1.0
R = 8.314462618
EA_TRUE = 50000.0
A_TRUE = 1.0e8
LNA_TRUE = math.log(A_TRUE)
Z975 = 1.959963984540054


def _seeded_concentrations(noise, n_time, rng):
    original = model.np.random.default_rng
    model.np.random.default_rng = lambda *args, **kwargs: rng
    try:
        return model.simulate_concentrations(noise, n_time)
    finally:
        model.np.random.default_rng = original


def _nominal_log_variances(k_values, noise, n_time):
    sigma = noise * C0
    values = []
    for k in k_values:
        t = np.linspace(0.0, 3.0 / k, n_time)
        derivative = t * C0 * np.exp(-k * t)
        var_k = sigma**2 / np.sum(derivative**2)
        values.append(var_k / k**2)
    return np.asarray(values, dtype=float)


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    noise = float(cell["noise_sigma_over_c0"])
    n_time = int(cell["n_time"])
    temperatures = model.construct_temperatures(EA_TRUE, R, LNA_TRUE)
    k_true = np.geomspace(0.1, 1.0, 5)

    # E2: the package's temperature construction evaluates Arrhenius rates.
    concentrations = _seeded_concentrations(noise, n_time, rng)
    k_hat = []
    var_k = []
    var_lnk = []
    k_cover = []

    for rate, concentration in zip(k_true, concentrations):
        time_s = np.linspace(0.0, 3.0 / rate, n_time)
        estimate, variance, log_variance = model.fit_first_stage(concentration, time_s)
        if not (np.isfinite(estimate) and estimate > 0 and np.isfinite(variance) and variance >= 0):
            raise RuntimeError("non-finite first-stage estimate or covariance")
        k_hat.append(estimate)
        var_k.append(variance)
        var_lnk.append(max(log_variance, np.finfo(float).tiny))
        se = math.sqrt(max(variance, 0.0))
        k_cover.append(float(estimate - Z975 * se <= rate <= estimate + Z975 * se))

    k_hat = np.asarray(k_hat, dtype=float)
    var_k = np.asarray(var_k, dtype=float)
    var_lnk = np.asarray(var_lnk, dtype=float)
    fit = model.fit_arrhenius(k_hat, var_lnk, temperatures)

    result = {
        "ea_weighted": float(fit["weighted"]["Ea_hat"]),
        "lna_weighted": float(fit["weighted"]["lnA_hat"]),
        "ea_unweighted": float(fit["unweighted"]["Ea_hat"]),
        "lna_unweighted": float(fit["unweighted"]["lnA_hat"]),
    }

    for label in ("weighted", "unweighted"):
        part = fit[label]
        covariance = np.asarray(part["covariance"], dtype=float)
        ea_se = math.sqrt(max(float(covariance[0, 0]), 0.0))
        lna_se = math.sqrt(max(float(covariance[1, 1]), 0.0))
        tcrit = 3.182446305284263
        result[f"ea_{label}_covered_z"] = float(
            part["Ea_hat"] - Z975 * ea_se <= EA_TRUE <= part["Ea_hat"] + Z975 * ea_se
        )
        result[f"lna_{label}_covered_z"] = float(
            part["lnA_hat"] - Z975 * lna_se <= LNA_TRUE <= part["lnA_hat"] + Z975 * lna_se
        )
        result[f"ea_{label}_covered_t"] = float(
            part["Ea_hat"] - tcrit * ea_se <= EA_TRUE <= part["Ea_hat"] + tcrit * ea_se
        )
        result[f"lna_{label}_covered_t"] = float(
            part["lnA_hat"] - tcrit * lna_se <= LNA_TRUE <= part["lnA_hat"] + tcrit * lna_se
        )

    for index, (truth, estimate, variance, covered) in enumerate(
        zip(k_true, k_hat, var_k, k_cover)
    ):
        result[f"k_hat_{index}"] = float(estimate)
        result[f"k_var_{index}"] = float(variance)
        result[f"k_covered_{index}"] = float(covered)

    nominal = _nominal_log_variances(k_true, max(noise, 0.01), n_time)
    exact_fit = model.fit_arrhenius(k_true, nominal, temperatures)
    noiseless = []
    for truth in k_true:
        time_s = np.linspace(0.0, 3.0 / truth, n_time)
        concentration = np.asarray(
            [model.first_order_concentration(C0, truth, t) for t in time_s]
        )
        estimate, variance, log_variance = model.fit_first_stage(concentration, time_s)
        noiseless.append((estimate, variance, log_variance, time_s))

    closed_errors = []
    for truth, (estimate, variance, log_variance, time_s) in zip(k_true, noiseless):
        closed_var = 0.01**2 / np.sum(
            (time_s * C0 * np.exp(-estimate * time_s)) ** 2
        )
        closed_log_var = closed_var / estimate**2
        if closed_var > 0:
            closed_errors.append(abs(variance - closed_var) / closed_var)
            closed_errors.append(abs(log_variance - closed_log_var) / closed_log_var)

    result["max_k_relative_error"] = float(
        max(abs(estimate - truth) / truth for truth, (estimate, *_rest) in zip(k_true, noiseless))
    )
    result["rate_ratio"] = float(np.max(k_true) / np.min(k_true))
    result["ea_arrhenius_relative_error"] = float(
        abs(exact_fit["weighted"]["Ea_hat"] - EA_TRUE) / EA_TRUE
    )
    result["lna_arrhenius_relative_error"] = float(
        abs(exact_fit["weighted"]["lnA_hat"] - LNA_TRUE) / LNA_TRUE
    )
    result["first_stage_variance_relative_error"] = float(max(closed_errors))
    example = model.arrhenius_rate(A_TRUE, EA_TRUE, R, 350.0)
    result["arrhenius_example_relative_error"] = float(
        abs(example - 3.448570123) / 3.448570123
    )
    return result