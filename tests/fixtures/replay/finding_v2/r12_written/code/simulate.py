import numpy as np
from kinetics_parameter_recovery import model

C0 = 1.0
EA_TRUE_J_MOL = 50000.0
A_TRUE_S_INV = 1000000.0
R_J_MOL_K = 8.314462618
T_LOW_K = 450.0
N_TEMPERATURES = 5


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    """Run one paired weighted/unweighted Arrhenius recovery replicate."""
    noise_std_rel = float(cell.get("noise_std_rel", 0.02))
    n_timepoints = int(cell.get("n_timepoints", 12))
    rng = np.random.default_rng(seed)

    temperatures = model.construct_temperatures(
        EA_TRUE_J_MOL, R_J_MOL_K, T_LOW_K, N_TEMPERATURES
    )
    k_true = np.array([
        model.arrhenius_rate(A_TRUE_S_INV, EA_TRUE_J_MOL, R_J_MOL_K, temperature)
        for temperature in temperatures
    ])

    # Use the trial-owned generator while calling the tested package S2 function.
    original_default_rng = model.np.random.default_rng
    model.np.random.default_rng = lambda *args, **kwargs: rng
    try:
        fitted_k = []
        var_ln_k = []
        endpoint = None
        additive_residuals = []

        for rate in k_true:
            time, noiseless, observed = model.simulate_trajectory(
                float(rate), n_timepoints, noise_std_rel
            )
            if endpoint is None:
                endpoint = float(noiseless[-1])
            additive_residuals.extend((observed - noiseless).tolist())
            k_hat, variance = model.fit_rate(time, observed)
            fitted_k.append(k_hat)
            var_ln_k.append(variance)
    finally:
        model.np.random.default_rng = original_default_rng

    fitted_k = np.asarray(fitted_k, dtype=float)
    var_ln_k = np.asarray(var_ln_k, dtype=float)
    ln_k_hat = np.log(fitted_k)

    weighted = model.calibrate_arrhenius(ln_k_hat, var_ln_k, temperatures)
    # Equal variances make the package WLS estimator algebraically identical
    # to the requested unweighted point estimator.
    unweighted = model.calibrate_arrhenius(
        ln_k_hat, np.ones_like(var_ln_k), temperatures
    )

    zero_noise = noise_std_rel == 0.0
    return {
        "weighted_Ea_hat": float(weighted["Ea_hat"]),
        "weighted_ln_A_hat": float(weighted["ln_A_hat"]),
        "weighted_Ea_interval_low": float(weighted["Ea_interval_95"][0]),
        "weighted_Ea_interval_high": float(weighted["Ea_interval_95"][1]),
        "weighted_ln_A_interval_low": float(weighted["ln_A_interval_95"][0]),
        "weighted_ln_A_interval_high": float(weighted["ln_A_interval_95"][1]),
        "unweighted_Ea_hat": float(unweighted["Ea_hat"]),
        "unweighted_ln_A_hat": float(unweighted["ln_A_hat"]),
        "unweighted_Ea_interval_low": float(unweighted["Ea_interval_95"][0]),
        "unweighted_Ea_interval_high": float(unweighted["Ea_interval_95"][1]),
        "unweighted_ln_A_interval_low": float(unweighted["ln_A_interval_95"][0]),
        "unweighted_ln_A_interval_high": float(unweighted["ln_A_interval_95"][1]),
        "weighted_k_relative_error": float(np.mean(fitted_k / k_true - 1.0)),
        "unweighted_k_relative_error": float(np.mean(fitted_k / k_true - 1.0)),
        "concentration_true_final": float(endpoint),
        "k_low": float(k_true[0]),
        "k_high": float(k_true[-1]),
        "Ea_hat_noise_free": float(unweighted["Ea_hat"]) if zero_noise else float("nan"),
        "ln_A_hat_noise_free": float(unweighted["ln_A_hat"]) if zero_noise else float("nan"),
        "additive_noise_residual": float(additive_residuals[0]),
        "fit_success": 1.0,
    }