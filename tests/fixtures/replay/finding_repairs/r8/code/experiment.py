import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

Z_975 = 1.959963984540054
EA_TRUE = 50000.0
LN_A_TRUE = math.log(1.0e8)
BOOTSTRAP_RESAMPLES = 2000


def finite(values):
    return np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)


def wilson(successes, total):
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1.0 + Z_975 ** 2 / total
    centre = (p + Z_975 ** 2 / (2.0 * total)) / denominator
    half = Z_975 * math.sqrt(p * (1.0 - p) / total + Z_975 ** 2 / (4.0 * total ** 2)) / denominator
    return [centre - half, centre + half]


def bootstrap(values, statistic=np.mean, rng=None):
    values = finite(values)
    if len(values) == 0:
        return [None, None]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    samples = values[indices]
    if statistic is np.std:
        estimates = np.std(samples, axis=1, ddof=1)
    else:
        estimates = np.asarray([statistic(row) for row in samples], dtype=float)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def correlation_stat(values):
    paired = np.asarray(values, dtype=float)
    if len(paired) < 2 or np.std(paired[:, 0]) == 0 or np.std(paired[:, 1]) == 0:
        return np.nan
    return float(np.corrcoef(paired[:, 0], paired[:, 1])[0, 1])


def method_rows(settings, method_name):
    return next(rows for name, rows in settings.items() if name == method_name)


def summarize_cell(cell_record, rng):
    metrics = cell_record["metrics"]
    cell = cell_record["cell"]
    trials = metrics["Ea_hat_J_mol"]["trials"]
    ea = finite(metrics["Ea_hat_J_mol"]["values"])
    lna = finite(metrics["lnA_hat"]["values"])
    ea_se = finite(metrics["Ea_se_J_mol"]["values"])
    lna_se = finite(metrics["lnA_se"]["values"])
    ea_cov = np.asarray(metrics["Ea_covered"]["values"], dtype=int)
    lna_cov = np.asarray(metrics["lnA_covered"]["values"], dtype=int)
    reported_corr = finite(metrics["Ea_lnA_reported_correlation"]["values"])
    n_successful = len(ea)
    total = int(cell_record["planned"])
    paired_estimates = np.column_stack((ea, lna)) if n_successful else np.empty((0, 2))
    empirical_corr = correlation_stat(paired_estimates)
    summary = {
        "additive_noise_sd": float(cell["additive_noise_sd"]),
        "n_time_points": int(cell["n_time_points"]),
        "n_successful_trials": n_successful,
        "n_failed_trials": int(cell_record["failed"]),
        "completion_rate": n_successful / total if total else None,
        "completion_rate_count": n_successful,
        "completion_rate_total": total,
        "completion_rate_wilson_ci95": wilson(n_successful, total),
        "Ea_bias_J_mol": {"estimate": float(np.mean(ea - EA_TRUE)) if n_successful else None, "bootstrap_ci95": bootstrap(ea - EA_TRUE, rng=rng)},
        "lnA_bias": {"estimate": float(np.mean(lna - LN_A_TRUE)) if n_successful else None, "bootstrap_ci95": bootstrap(lna - LN_A_TRUE, rng=rng)},
        "Ea_empirical_sd_J_mol": {"estimate": float(np.std(ea, ddof=1)) if n_successful > 1 else None, "bootstrap_ci95": bootstrap(ea, statistic=np.std, rng=rng)},
        "lnA_empirical_sd": {"estimate": float(np.std(lna, ddof=1)) if n_successful > 1 else None, "bootstrap_ci95": bootstrap(lna, statistic=np.std, rng=rng)},
        "Ea_mean_reported_se_J_mol": {"estimate": float(np.mean(ea_se)) if len(ea_se) else None, "bootstrap_ci95": bootstrap(ea_se, rng=rng)},
        "lnA_mean_reported_se": {"estimate": float(np.mean(lna_se)) if len(lna_se) else None, "bootstrap_ci95": bootstrap(lna_se, rng=rng)},
        "Ea_95_coverage": float(np.mean(ea_cov)) if len(ea_cov) else None,
        "Ea_95_coverage_count": int(np.sum(ea_cov)),
        "Ea_95_coverage_total": int(len(ea_cov)),
        "Ea_95_coverage_wilson_ci95": wilson(int(np.sum(ea_cov)), len(ea_cov)),
        "lnA_95_coverage": float(np.mean(lna_cov)) if len(lna_cov) else None,
        "lnA_95_coverage_count": int(np.sum(lna_cov)),
        "lnA_95_coverage_total": int(len(lna_cov)),
        "lnA_95_coverage_wilson_ci95": wilson(int(np.sum(lna_cov)), len(lna_cov)),
        "Ea_lnA_empirical_correlation": {"estimate": empirical_corr if np.isfinite(empirical_corr) else None, "bootstrap_ci95": [None, None]},
        "Ea_lnA_reported_correlation": {"estimate": float(np.mean(reported_corr)) if len(reported_corr) else None, "bootstrap_ci95": bootstrap(reported_corr, rng=rng)},
        "Ea_bias_J_mol_values": (ea - EA_TRUE).tolist(),
        "Ea_bias_J_mol_pair_id": trials,
        "lnA_bias_values": (lna - LN_A_TRUE).tolist(),
        "lnA_bias_pair_id": trials,
    }
    k_bias, k_coverage = [], []
    for index in range(5):
        errors = finite(metrics[f"k_relative_error_{index}"]["values"])
        covered = np.asarray(metrics[f"k_covered_{index}"]["values"], dtype=int)
        k_bias.append({"estimate": float(np.mean(errors)) if len(errors) else None, "bootstrap_ci95": bootstrap(errors, rng=rng)})
        k_coverage.append({"estimate": float(np.mean(covered)) if len(covered) else None, "count": int(np.sum(covered)), "total": int(len(covered)), "wilson_ci95": wilson(int(np.sum(covered)), len(covered))})
    summary["k_relative_bias_by_temp"] = k_bias
    summary["k_95_coverage_by_temp"] = k_coverage
    return summary


def plot_arrhenius_calibration(settings):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for method, rows in settings.items():
        x = np.arange(len(rows))
        ea_bias = [item["Ea_bias_J_mol"]["estimate"] for item in rows]
        ea_ratio = [item["Ea_empirical_sd_J_mol"]["estimate"] / item["Ea_mean_reported_se_J_mol"]["estimate"] if item["Ea_empirical_sd_J_mol"]["estimate"] is not None and item["Ea_mean_reported_se_J_mol"]["estimate"] not in (None, 0.0) else np.nan for item in rows]
        coverage = [item["Ea_95_coverage"] for item in rows]
        style = "-" if method == "weighted" else "--"
        axes[0].plot(x, ea_bias, marker="o", linestyle=style, label=method)
        axes[1].plot(x, ea_ratio, marker="s", linestyle=style, label=method)
        axes[2].plot(x, coverage, marker="^", linestyle=style, label=method)
    axes[0].axhline(0.0, linestyle=":")
    axes[1].axhline(1.0, linestyle=":")
    axes[2].axhline(0.95, linestyle=":")
    axes[0].set(title="Ea bias", xlabel="Noise/time setting", ylabel="Bias (J mol$^{-1}$)")
    axes[1].set(title="Ea uncertainty calibration", xlabel="Noise/time setting", ylabel="Empirical SD / mean SE")
    axes[2].set(title="Ea 95% interval coverage", xlabel="Noise/time setting", ylabel="Coverage probability")
    axes[2].legend(frameon=False)
    fig.savefig("figures/arrhenius_calibration.png", dpi=300)
    plt.close(fig)


def plot_rate_constant_recovery(settings):
    rows = method_rows(settings, "weighted")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    temperatures = np.arange(1, 6)
    for item in (rows[0], rows[len(rows) // 2], rows[-1]):
        label = f"σ={item['additive_noise_sd']}, n={item['n_time_points']}"
        axes[0].plot(temperatures, [x["estimate"] for x in item["k_relative_bias_by_temp"]], marker="o", label=label)
        axes[1].plot(temperatures, [x["estimate"] for x in item["k_95_coverage_by_temp"]], marker="s", label=label)
        axes[2].plot(temperatures, [x["wilson_ci95"][1] - x["wilson_ci95"][0] if x["wilson_ci95"][0] is not None else np.nan for x in item["k_95_coverage_by_temp"]], marker="^", label=label)
    axes[0].axhline(0.0, linestyle=":")
    axes[1].axhline(0.95, linestyle=":")
    axes[0].set(title="Rate-constant relative bias", xlabel="Temperature index", ylabel="Mean relative bias")
    axes[1].set(title="Rate-constant coverage", xlabel="Temperature index", ylabel="95% coverage")
    axes[2].set(title="Coverage uncertainty", xlabel="Temperature index", ylabel="Wilson interval width")
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig("figures/rate_constant_recovery.png", dpi=300)
    plt.close(fig)


def plot_weighted_vs_unweighted(settings):
    weighted_rows = method_rows(settings, "weighted")
    unweighted_rows = method_rows(settings, "unweighted")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    x = np.arange(len(weighted_rows))
    axes[0].plot(x, [v["Ea_bias_J_mol"]["estimate"] for v in weighted_rows], marker="o", label="weighted")
    axes[0].plot(x, [v["Ea_bias_J_mol"]["estimate"] for v in unweighted_rows], marker="s", linestyle="--", label="unweighted")
    axes[1].plot(x, [v["Ea_empirical_sd_J_mol"]["estimate"] for v in weighted_rows], marker="o", label="weighted")
    axes[1].plot(x, [v["Ea_empirical_sd_J_mol"]["estimate"] for v in unweighted_rows], marker="s", linestyle="--", label="unweighted")
    axes[2].plot(x, [v["Ea_95_coverage"] for v in weighted_rows], marker="o", label="weighted")
    axes[2].plot(x, [v["Ea_95_coverage"] for v in unweighted_rows], marker="s", linestyle="--", label="unweighted")
    axes[0].axhline(0.0, linestyle=":")
    axes[2].axhline(0.95, linestyle=":")
    axes[0].set(title="Ea bias comparison", xlabel="Noise/time setting", ylabel="Bias (J mol$^{-1}$)")
    axes[1].set(title="Ea estimate spread", xlabel="Noise/time setting", ylabel="Empirical SD (J mol$^{-1}$)")
    axes[2].set(title="Ea coverage comparison", xlabel="Noise/time setting", ylabel="95% coverage")
    axes[2].legend(frameon=False)
    fig.savefig("figures/weighted_vs_unweighted.png", dpi=300)
    plt.close(fig)


def write_paper(settings):
    rows = method_rows(settings, "weighted")
    low_noise = [s for s in rows if s["additive_noise_sd"] <= 0.02 and s["n_time_points"] >= 10]
    ratios = [s["Ea_empirical_sd_J_mol"]["estimate"] / s["Ea_mean_reported_se_J_mol"]["estimate"] for s in low_noise if s["Ea_empirical_sd_J_mol"]["estimate"] is not None and s["Ea_mean_reported_se_J_mol"]["estimate"] not in (None, 0.0)]
    text = f'''# Recovery of first-order kinetic parameters from noisy concentration data

## Methods

We simulated first-order traces, `c(t)=c0 exp(-kt)`, at five inverse-temperature-spaced temperatures spanning a ten-fold true-rate range. Controls were c0=1.0, R=8.314 J mol^-1 K^-1, Ea=50000 J mol^-1, and A=1e8 s^-1. Each trace ended at 3/k and received independent additive Gaussian concentration noise. Bounded nonlinear least squares estimated each k; Arrhenius fits regressed ln(k) on centred 1/T using inverse delta-method variance weights or unweighted least squares.

## Results

The weighted analysis included {len(rows)} grid settings. In the low-noise, high-density regime, the Ea empirical-SD/mean-SE ratio ranged from {min(ratios):.3g} to {max(ratios):.3g}. Ea and ln A estimates are strongly correlated because their regression transformation shares the Arrhenius slope.

## Reproducibility

The base seed was 20261013. Numerical tables underlying the figures are saved in `data/results/summary.json` and `data/results/summary.csv`.
'''
    with open("paper.md", "w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    with open(os.environ["FI_TRIALS"], "r", encoding="utf-8") as handle:
        record = json.load(handle)
    rng = np.random.default_rng(20261013)
    settings = {}
    for cell_record in record["cells"]:
        method_name = cell_record["cell"]["arrhenius_fit_method"]
        settings.setdefault(method_name, []).append(summarize_cell(cell_record, rng))
    for rows in settings.values():
        rows.sort(key=lambda x: (x["additive_noise_sd"], x["n_time_points"]))
    with open("data/results/summary.json", "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2, allow_nan=False)
    with open("data/results/summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "noise_sd", "n_time_points", "Ea_bias_J_mol", "Ea_sd_J_mol", "Ea_mean_SE_J_mol", "Ea_coverage", "lnA_bias", "lnA_coverage"])
        for method, rows in settings.items():
            for row in rows:
                writer.writerow([method, row["additive_noise_sd"], row["n_time_points"], row["Ea_bias_J_mol"]["estimate"], row["Ea_empirical_sd_J_mol"]["estimate"], row["Ea_mean_reported_se_J_mol"]["estimate"], row["Ea_95_coverage"], row["lnA_bias"]["estimate"], row["lnA_95_coverage"]])
    plot_arrhenius_calibration(settings)
    plot_rate_constant_recovery(settings)
    plot_weighted_vs_unweighted(settings)
    write_paper(settings)
    return {"checks": {"executed_by_frontierinsight": True}, "constants": {"c0": 1.0, "R_J_mol_K": 8.314, "Ea_true_J_mol": 50000.0, "A_true_s_inv": 100000000.0, "T_min_K": 300.0, "base_seed": 20261013, "n_trials": 200}, "by_arrhenius_fit_method": {method: {"settings": values} for method, values in settings.items()}, "figures": ["figures/arrhenius_calibration.png", "figures/rate_constant_recovery.png", "figures/weighted_vs_unweighted.png"], "paper": "paper.md"}


print("RESULT_JSON: " + json.dumps(main(), sort_keys=True, allow_nan=False))
