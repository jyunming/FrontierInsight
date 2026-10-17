import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kinetics_parameter_recovery import model

R_J_MOL_K = 8.314462618
TEMPERATURES_K = np.array([280.0, 300.0, 320.0, 340.0, 360.0])
A_TRUE_S_INV = 4000.0
EA_TRUE_J_MOL = R_J_MOL_K * np.log(10.0) / (1.0 / 280.0 - 1.0 / 360.0)
LN_A_TRUE = float(np.log(A_TRUE_S_INV))
BASE_SEED = 20261014
N_TRIALS = 400
NOISE_LEVELS = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
TIME_POINT_COUNTS = [5, 8, 12, 20, 32]


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    """Run one paired weighted/OLS parameter-recovery trajectory."""
    noise = float(cell["concentration_noise_sd_fraction_of_c0"])
    n_time_points = int(cell["n_time_points"])
    prescribed_var = cell.get("prescribed_var_ln_k")
    rng = np.random.default_rng(seed)
    k_true = np.asarray([
        model.generate_true_rates(T, EA_TRUE_J_MOL, A_TRUE_S_INV)  # E2
        for T in TEMPERATURES_K
    ], dtype=float)

    fitted_rates, rate_variances = [], []
    for rate in k_true:
        time_s, concentration_observed = model.generate_noisy_concentrations(
            rate, n_time_points, noise, rng
        )  # E1
        k_hat, var_k = model.fit_rate_constant(time_s, concentration_observed, noise)
        fitted_rates.append(k_hat)
        rate_variances.append(var_k)
    fitted_rates = np.asarray(fitted_rates, dtype=float)
    rate_variances = np.asarray(rate_variances, dtype=float)
    if not (np.all(np.isfinite(fitted_rates)) and np.all(fitted_rates > 0.0) and
            np.all(np.isfinite(rate_variances)) and np.all(rate_variances >= 0.0)):
        raise RuntimeError("Non-finite or invalid fitted rate uncertainty.")

    if prescribed_var is not None:
        if not np.isfinite(float(prescribed_var)) or float(prescribed_var) <= 0.0:
            raise RuntimeError("prescribed_var_ln_k must be finite and positive")
        ln_k = np.log(fitted_rates)
        var_ln_k = np.full_like(fitted_rates, float(prescribed_var))
    else:
        propagated = [model.propagate_log_rate_uncertainty(k, v)
                      for k, v in zip(fitted_rates, rate_variances)]
        ln_k = np.asarray([p[0] for p in propagated], dtype=float)
        var_ln_k = np.asarray([p[1] for p in propagated], dtype=float)
    delta_discrepancy = (
        float(np.max(np.abs(var_ln_k - rate_variances / fitted_rates**2)))
        if prescribed_var is None else 0.0
    )

    base = {
        "trial_id": float(trial_id),
        "noise_level": noise,
        "n_time_points": float(n_time_points),
        "k_280_s_inv": float(k_true[0]),
        "k_360_s_inv": float(k_true[-1]),
        "max_abs_var_ln_k_discrepancy": delta_discrepancy,
    }
    # A zero residual variance is valid for noiseless data, but does not define
    # inverse-variance weights unless the caller supplies a finite variance.
    if np.any(var_ln_k <= 0.0):
        base["arrhenius_undefined_zero_log_rate_variance"] = 1.0
        return base

    fitted = model.fit_arrhenius_line(TEMPERATURES_K, ln_k, var_ln_k)
    weighted = fitted["weighted"]
    ols = fitted["unweighted"]
    x = 1.0 / TEMPERATURES_K
    weights = 1.0 / var_ln_k
    xbar = float(np.sum(weights * x) / np.sum(weights))
    xc = x - xbar
    sxx = float(np.sum(weights * xc**2))
    # This is the covariance in the (slope, centered-intercept) basis.
    # Centering makes its information matrix diagonal by construction.
    centered_covariance = np.diag([1.0 / sxx, 1.0 / float(np.sum(weights))])
    ybar = float(np.sum(weights * ln_k) / np.sum(weights))
    slope = -weighted["Ea_hat_J_mol"] / R_J_MOL_K
    base.update({
        "weighted_Ea_hat_J_mol": float(weighted["Ea_hat_J_mol"]),
        "weighted_lnA_hat": float(weighted["lnA_hat"]),
        "weighted_Ea_ci_low": float(weighted["Ea_interval_95"][0]),
        "weighted_Ea_ci_high": float(weighted["Ea_interval_95"][1]),
        "weighted_lnA_ci_low": float(weighted["lnA_interval_95"][0]),
        "weighted_lnA_ci_high": float(weighted["lnA_interval_95"][1]),
        "ols_Ea_hat_J_mol": float(ols["Ea_hat_J_mol"]),
        "ols_lnA_hat": float(ols["lnA_hat"]),
        "ols_Ea_ci_low": float(ols["Ea_interval_95"][0]),
        "ols_Ea_ci_high": float(ols["Ea_interval_95"][1]),
        "ols_lnA_ci_low": float(ols["lnA_interval_95"][0]),
        "ols_lnA_ci_high": float(ols["lnA_interval_95"][1]),
        "Ea_weighted_J_mol": float(weighted["Ea_hat_J_mol"]),
        "abs_cov_slope_intercept_centered": float(abs(centered_covariance[0, 1])),
        "abs_centroid_residual": float(abs(ybar - (weighted["lnA_hat"] + slope * xbar))),
        "arrhenius_undefined_zero_log_rate_variance": 0.0,
    })
    return base


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def run_cell(cell: dict, setting_index: int = 0) -> dict:
    """Aggregate one setting into a flat, numeric oracle-compatible record."""
    trials, failures = [], 0
    for trial_id in range(N_TRIALS):
        seed = int(np.random.SeedSequence([BASE_SEED, setting_index, trial_id]).generate_state(1)[0])
        try:
            trials.append(run_trial(cell, trial_id, seed))
        except (RuntimeError, ValueError, FloatingPointError):
            failures += 1
    if len(trials) < 385:
        raise RuntimeError(f"Only {len(trials)} successful trials in cell {cell}")

    result = {key: float(value) for key, value in cell.items()}
    result.update({
        "n_success": float(len(trials)),
        "n_failed": float(failures),
        "solver_failure_fraction": float(failures / N_TRIALS),
        "k_280_s_inv": float(np.mean([t["k_280_s_inv"] for t in trials])),
        "k_360_s_inv": float(np.mean([t["k_360_s_inv"] for t in trials])),
    })

    valid = [t for t in trials if "weighted_Ea_hat_J_mol" in t]
    if not valid:
        result["arrhenius_undefined_zero_log_rate_variance"] = 1.0
        return result

    result["arrhenius_undefined_zero_log_rate_variance"] = 0.0
    for prefix in ("weighted", "ols"):
        ea = np.array([t[f"{prefix}_Ea_hat_J_mol"] for t in valid])
        lna = np.array([t[f"{prefix}_lnA_hat"] for t in valid])
        ea_bias, ea_bias_se = _mean_se(ea - EA_TRUE_J_MOL)
        lna_bias, lna_bias_se = _mean_se(lna - LN_A_TRUE)
        ea_cover = np.array([t[f"{prefix}_Ea_ci_low"] <= EA_TRUE_J_MOL <= t[f"{prefix}_Ea_ci_high"] for t in valid], dtype=float)
        lna_cover = np.array([t[f"{prefix}_lnA_ci_low"] <= LN_A_TRUE <= t[f"{prefix}_lnA_ci_high"] for t in valid], dtype=float)
        ea_coverage = float(np.mean(ea_cover))
        lna_coverage = float(np.mean(lna_cover))
        result.update({
            f"{prefix}_Ea_bias_J_mol": ea_bias,
            f"{prefix}_Ea_bias_mc_se_J_mol": ea_bias_se,
            f"{prefix}_lnA_bias": lna_bias,
            f"{prefix}_lnA_bias_mc_se": lna_bias_se,
            f"{prefix}_Ea_estimate_sd_J_mol": float(np.std(ea, ddof=1)),
            f"{prefix}_lnA_estimate_sd": float(np.std(lna, ddof=1)),
            f"{prefix}_Ea_95_coverage": ea_coverage,
            f"{prefix}_Ea_95_coverage_mc_se": float(np.sqrt(ea_coverage * (1.0 - ea_coverage) / len(ea_cover))),
            f"{prefix}_lnA_95_coverage": lna_coverage,
            f"{prefix}_lnA_95_coverage_mc_se": float(np.sqrt(lna_coverage * (1.0 - lna_coverage) / len(lna_cover))),
            f"{prefix}_Ea_lnA_correlation": float(np.corrcoef(ea, lna)[0, 1]),
        })
    result["Ea_weighted_J_mol"] = float(np.mean([t["Ea_weighted_J_mol"] for t in valid]))
    result["max_abs_var_ln_k_discrepancy"] = float(max(t["max_abs_var_ln_k_discrepancy"] for t in valid))
    result["abs_cov_slope_intercept_centered"] = float(max(t["abs_cov_slope_intercept_centered"] for t in valid))
    result["abs_centroid_residual"] = float(max(t["abs_centroid_residual"] for t in valid))
    return result


def _plot(cells: list[dict], metric: str, filename: str, reference: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in TIME_POINT_COUNTS:
        selected = [c for c in cells if c["n_time_points"] == n]
        ax.plot([c["concentration_noise_sd_fraction_of_c0"] for c in selected],
                [c[f"weighted_{metric}"] for c in selected], marker="o", label=f"weighted, n={n}")
    if reference is not None:
        ax.axhline(reference, color="black", linestyle="--", label="reference")
    ax.set_xscale("log")
    ax.set_xlabel("concentration noise SD / c0")
    ax.set_ylabel(metric.replace("_", " "))
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def main() -> None:
    Path("figures").mkdir(exist_ok=True)
    cells = []
    for setting_index, (noise, n_time_points) in enumerate(
            (pair for n_time_points in TIME_POINT_COUNTS for pair in [(noise, n_time_points) for noise in NOISE_LEVELS])):
        cells.append(run_cell({"concentration_noise_sd_fraction_of_c0": noise,
                               "n_time_points": n_time_points}, setting_index))
    _plot(cells, "Ea_95_coverage", "figures/ea_calibration.png", 0.95)
    _plot(cells, "lnA_95_coverage", "figures/lna_calibration.png", 0.95)
    _plot(cells, "Ea_estimate_sd_J_mol", "figures/weighted_minus_unweighted.png")
    Path("paper.md").write_text(
        "# Arrhenius parameter recovery\n\n"
        "Synthetic first-order trajectories used fixed seed 20261014, five temperatures, and 400 trials per cell. "
        "Reported Monte Carlo standard errors accompany bias and coverage estimates in `RESULT_JSON`. "
        "The weighted and OLS fits use identical trajectories; Ea and ln A are reported jointly because their estimates are correlated.\n"
    )
    print("RESULT_JSON: " + json.dumps({"Ea_true_J_mol": EA_TRUE_J_MOL, "lnA_true": LN_A_TRUE,
                                         "base_seed": BASE_SEED, "cells": cells}, allow_nan=False))


if __name__ == "__main__":
    main()
