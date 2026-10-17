import math
import numpy as np
from scipy import optimize
from kinetics_parameter_recovery import model

C0_MOL_L = 1.0
EA_TRUE_J_MOL = 50000.0
A_TRUE_S_INV = 3000000.0
R_J_MOL_K = 8.314462618
T_REF_K = 350.0


def _temperatures():
    delta = R_J_MOL_K * math.log(10.0) / (2.0 * EA_TRUE_J_MOL)
    reciprocal = np.linspace(1.0 / T_REF_K + delta, 1.0 / T_REF_K - delta, 5)
    return 1.0 / reciprocal


def _fit_rate(t_s, c_obs, sigma_c, k_true):
    lower, upper = 1e-5 * k_true, 1e3 * k_true
    initial = np.clip(k_true, lower * 1.01, upper / 1.01)

    def residual(parameters):
        return model.first_order_concentration(C0_MOL_L, parameters[0], t_s) - c_obs

    fit = optimize.least_squares(
        residual,
        x0=np.array([initial]),
        bounds=(np.array([lower]), np.array([upper])),
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=1000,
    )
    k_hat = float(fit.x[0])
    if (not fit.success or not np.isfinite(k_hat) or
            np.isclose(k_hat, lower) or np.isclose(k_hat, upper)):
        raise RuntimeError("unusable nonlinear rate fit")

    jacobian = fit.jac
    information = float((jacobian.T @ jacobian).item())
    if not np.isfinite(information) or information <= 0.0:
        raise RuntimeError("nonfinite rate covariance")

    var_k = sigma_c ** 2 / information
    var_ln_k = var_k / k_hat ** 2
    if not np.isfinite(var_ln_k) or var_ln_k <= 0.0:
        raise RuntimeError("invalid propagated log-rate variance")

    return k_hat, var_k, var_ln_k


def _fit_arrhenius(temperatures, rates, variances, weighted):
    x = 1.0 / temperatures
    xbar = float(np.mean(x))
    design = np.column_stack((np.ones_like(x), x - xbar))
    y = np.log(rates)

    if weighted:
        weights = 1.0 / variances
        normal = design.T @ (weights[:, None] * design)
        covariance = np.linalg.inv(normal)
        coefficients = covariance @ (design.T @ (weights * y))
        critical = 1.959963984540054
    else:
        normal = design.T @ design
        inv_normal = np.linalg.inv(normal)
        coefficients = inv_normal @ (design.T @ y)
        residuals = y - design @ coefficients
        scale = float(np.sum(residuals ** 2) / (len(y) - 2))
        covariance = inv_normal * scale
        critical = float(stats_t_975_df3())

    alpha, beta = coefficients
    transform = np.array([[0.0, -R_J_MOL_K], [1.0, -xbar]])
    parameter_covariance = transform @ covariance @ transform.T
    ea = model.activation_energy_from_slope(R_J_MOL_K, float(beta))
    ln_a = float(alpha - beta * xbar)
    ea_se, lna_se = np.sqrt(np.diag(parameter_covariance))
    if not all(np.isfinite([ea, ln_a, ea_se, lna_se])):
        raise RuntimeError("invalid Arrhenius regression")

    return {
        "ea": float(ea),
        "lnA": ln_a,
        "ea_se": float(ea_se),
        "lnA_se": float(lna_se),
        "ea_low": float(ea - critical * ea_se),
        "ea_high": float(ea + critical * ea_se),
        "lnA_low": float(ln_a - critical * lna_se),
        "lnA_high": float(ln_a + critical * lna_se),
    }


def stats_t_975_df3():
    return 3.182446305284263


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    if "k_s_inv" in cell and "t_s" in cell and "sigma_c_mol_L" in cell:
        c_model = model.first_order_concentration(
            float(cell.get("c0_mol_L", 1.0)), float(cell["k_s_inv"]), float(cell["t_s"])
        )
        sigma = float(cell["sigma_c_mol_L"])
        errors = np.array([
            float(cell.get("standardized_error_1", -1.0)),
            float(cell.get("standardized_error_2", 1.0)),
        ]) * sigma
        return {
            "c_model_mol_L": float(c_model),
            "residual_sample_variance_mol_L2": float(np.mean(errors ** 2)),
        }

    if {"A_s_inv", "Ea_J_mol", "R_J_mol_K", "T_K"}.issubset(cell):
        return {
            "k_true_s_inv": float(model.arrhenius_rate(
                float(cell["A_s_inv"]), float(cell["Ea_J_mol"]),
                float(cell["R_J_mol_K"]), float(cell["T_K"])
            ))
        }

    if {"alpha", "beta", "xbar_invK"}.issubset(cell):
        return {"ln_a_hat": float(cell["alpha"] - cell["beta"] * cell["xbar_invK"])}

    temperatures = _temperatures()
    true_rates = np.array([
        model.arrhenius_rate(A_TRUE_S_INV, EA_TRUE_J_MOL, R_J_MOL_K, temperature)
        for temperature in temperatures
    ])

    if "sigma_c_mol_L" in cell and bool(cell.get("exact_observations", False)):
        sigma = float(cell["sigma_c_mol_L"])
        n_time = int(cell["n_time"])
        fitted_rates, variances = [], []
        for k_true in true_rates:
            times = np.linspace(0.0, 3.0 / k_true, n_time)
            observations = np.array([
                model.first_order_concentration(C0_MOL_L, k_true, time) for time in times
            ])
            k_hat, _, var_ln = _fit_rate(times, observations, sigma, k_true)
            fitted_rates.append(k_hat)
            variances.append(var_ln)
        regression = _fit_arrhenius(
            temperatures, np.asarray(fitted_rates), np.asarray(variances), True
        )
        return {"fitted_ea_j_mol": regression["ea"]}

    if "noise_sigma_rel" not in cell or "n_time" not in cell:
        return {"k_hot_over_k_cold": float(true_rates[-1] / true_rates[0])}

    noise_sigma_rel = float(cell["noise_sigma_rel"])
    n_time = int(cell["n_time"])
    sigma_c = noise_sigma_rel * C0_MOL_L
    fitted_rates, var_k, var_ln = [], [], []

    for k_true in true_rates:
        times = np.linspace(0.0, 3.0 / k_true, n_time)
        observations = np.array([
            model.noisy_concentration(C0_MOL_L, k_true, time, sigma_c, rng)
            for time in times
        ])
        k_hat, local_var_k, local_var_ln = _fit_rate(times, observations, sigma_c, k_true)
        fitted_rates.append(k_hat)
        var_k.append(local_var_k)
        var_ln.append(local_var_ln)

    fitted_rates = np.asarray(fitted_rates)
    var_k = np.asarray(var_k)
    var_ln = np.asarray(var_ln)
    weighted = _fit_arrhenius(temperatures, fitted_rates, var_ln, True)
    unweighted = _fit_arrhenius(temperatures, fitted_rates, var_ln, False)

    output = {
        "weighted_ea": weighted["ea"],
        "weighted_lnA": weighted["lnA"],
        "weighted_ea_se": weighted["ea_se"],
        "weighted_lnA_se": weighted["lnA_se"],
        "weighted_ea_covered": float(weighted["ea_low"] <= EA_TRUE_J_MOL <= weighted["ea_high"]),
        "weighted_lnA_covered": float(weighted["lnA_low"] <= math.log(A_TRUE_S_INV) <= weighted["lnA_high"]),
        "unweighted_ea": unweighted["ea"],
        "unweighted_lnA": unweighted["lnA"],
        "unweighted_ea_se": unweighted["ea_se"],
        "unweighted_lnA_se": unweighted["lnA_se"],
        "unweighted_ea_covered": float(unweighted["ea_low"] <= EA_TRUE_J_MOL <= unweighted["ea_high"]),
        "unweighted_lnA_covered": float(unweighted["lnA_low"] <= math.log(A_TRUE_S_INV) <= unweighted["lnA_high"]),
    }
    for index, (temperature, truth, estimate, variance) in enumerate(
        zip(temperatures, true_rates, fitted_rates, var_k)
    ):
        output[f"T_K_{index}"] = float(temperature)  # constant by design: the five reciprocal-temperature grid values are controls shared by every noise and sampling setting.
        output[f"k_true_{index}"] = float(truth)
        output[f"k_hat_{index}"] = float(estimate)
        output[f"k_se_{index}"] = float(math.sqrt(variance))
        output[f"k_covered_{index}"] = float(
            estimate - 1.959963984540054 * math.sqrt(variance) <= truth <=
            estimate + 1.959963984540054 * math.sqrt(variance)
        )
    return output
