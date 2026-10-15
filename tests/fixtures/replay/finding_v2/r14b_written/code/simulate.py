import math
import numpy as np

from kinetics_parameter_recovery import model

C0 = 1.0
A_TRUE = 1.0e6
EA_TRUE = 50000.0
R_GAS = 8.314462618
K_LOWER_BOUND = 1.0e-8
K_UPPER_BOUND = 100.0


def _seeded_package_trace(k, n_time_points, noise, rng):
    """Call the supplied S2 implementation while giving it FI's generator."""
    original_default_rng = model.np.random.default_rng
    model.np.random.default_rng = lambda *args, **kwargs: rng
    try:
        return model.simulate_trace(k, n_time_points, noise)
    finally:
        model.np.random.default_rng = original_default_rng


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    noise = float(cell["noise_sd_fraction_c0"])
    n_points = int(cell["n_time_points"])
    methods = list(cell["arrhenius_method"])

    k_true = np.geomspace(0.1, 1.0, 5)
    # E2: supplied package implementation is used by S1.
    temperatures = np.asarray(
        [model.construct_temperature_grid(A_TRUE, EA_TRUE, R_GAS, k) for k in k_true],
        dtype=float,
    )

    fitted_k, var_k, var_ln_k = [], [], []
    final_true = None
    for index, k in enumerate(k_true):
        # E1 and E5: supplied S2 implementation evaluates the decay and noise model.
        time_s, concentration = _seeded_package_trace(k, n_points, noise, rng)
        if index == 2:
            final_true = model.first_order_concentration(C0, k, time_s[-1])
        kh, vk, vlk = model.fit_rate_constant(time_s, concentration, C0)
        if (not np.isfinite(kh) or not np.isfinite(vk) or not np.isfinite(vlk)
                or kh <= K_LOWER_BOUND or kh >= K_UPPER_BOUND
                or (noise > 0.0 and vlk <= 0.0)):
            raise RuntimeError("invalid bounded nonlinear least-squares rate fit")
        fitted_k.append(kh)
        var_k.append(vk)
        var_ln_k.append(vlk)

    if noise == 0.0:
        var_ln_k = np.ones(5, dtype=float)
    else:
        var_ln_k = np.asarray(var_ln_k, dtype=float)

    fits = model.fit_arrhenius_methods(temperatures, fitted_k, var_ln_k)
    output = {
        "k_hat": [float(x) for x in fitted_k],
        "var_k": [float(x) for x in var_k],
        "T_temperature_2": float(temperatures[2]),
        "c_true_final_temperature_2": float(final_true),
        "Ea_hat": {},
        "ln_A_hat": {},
        "Ea_t_lower": {},
        "Ea_t_upper": {},
        "ln_A_t_lower": {},
        "ln_A_t_upper": {},
        "Ea_normal_lower": {},
        "Ea_normal_upper": {},
        "ln_A_normal_lower": {},
        "ln_A_normal_upper": {},
        "reported_se": {},
    }
    for name in methods:
        item = fits[name]
        covariance = np.asarray(item["covariance"], dtype=float)
        ti = item["t_95_intervals"]
        ni = item["normal_95_intervals"]
        output["Ea_hat"][name] = float(item["Ea_hat"])
        output["ln_A_hat"][name] = float(item["ln_A_hat"])
        output["Ea_t_lower"][name], output["Ea_t_upper"][name] = map(float, ti["Ea"])
        output["ln_A_t_lower"][name], output["ln_A_t_upper"][name] = map(float, ti["ln_A"])
        output["Ea_normal_lower"][name], output["Ea_normal_upper"][name] = map(float, ni["Ea"])
        output["ln_A_normal_lower"][name], output["ln_A_normal_upper"][name] = map(float, ni["ln_A"])
        output["reported_se"][name] = float(np.sqrt(covariance[0, 0]))

    weighted = fits["weighted_centered"]["Ea_hat"]
    uncentred = fits["weighted_uncentered"]["Ea_hat"]
    poly_coeff = np.polyfit(
        1.0 / temperatures - np.mean(1.0 / temperatures),
        np.log(np.asarray(fitted_k)),
        1,
        w=np.sqrt(1.0 / var_ln_k),
    )
    poly_ea = -R_GAS * poly_coeff[0]
    output["max_rel_diff_Ea_closed_form_vs_polyfit"] = float(
        abs(weighted - poly_ea) / max(1.0, abs(weighted))
    )
    output["max_rel_diff_weighted_centered_vs_uncentered"] = float(
        max(
            abs(fits["weighted_centered"]["Ea_hat"] - uncentred)
            / max(1.0, abs(weighted)),
            abs(
                fits["weighted_centered"]["ln_A_hat"]
                - fits["weighted_uncentered"]["ln_A_hat"]
            ) / max(1.0, abs(fits["weighted_centered"]["ln_A_hat"])),
        )
    )
    return output