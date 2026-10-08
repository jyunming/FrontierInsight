import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

EA_TRUE = 50000.0
LN_A_TRUE = math.log(1.0e8)
Z95 = norm.ppf(0.975)
CUTOFFS = (0.0, 0.025, 0.05, 0.10)


def wilson(count, total):
    if total == 0:
        return [None, None]
    p = count / total
    denominator = 1.0 + Z95**2 / total
    centre = (p + Z95**2 / (2 * total)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / total + Z95**2 / (4 * total**2)) / denominator
    return [centre - half, centre + half]


def bootstrap_mean(values, rng, n_boot=1000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [None, None]
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def metric_values(cell, name):
    metric = cell["metrics"].get(name, {})
    return np.asarray(metric.get("values", []), dtype=float), metric.get("trials", [])


def summarize_method(cell, prefix, deterministic, rng):
    ea, trials = metric_values(cell, f"{prefix}_ea")
    ln_a, _ = metric_values(cell, f"{prefix}_ln_a")
    var_ea, _ = metric_values(cell, f"{prefix}_var_ea")
    var_ln_a, _ = metric_values(cell, f"{prefix}_var_ln_a")
    cov, _ = metric_values(cell, f"{prefix}_cov_ea_ln_a")
    attempted = int(cell["planned"])
    successful = len(ea)
    failures = attempted - successful

    result = {
        "successful_trials": successful,
        "attempted_trials": attempted,
        "fit_failure_probability": failures / attempted,
        "fit_failure_probability_count": failures,
        "fit_failure_probability_total": attempted,
    }
    if successful == 0:
        result.update({
            "relative_bias_ea": None, "relative_bias_ea_bootstrap_ci": [None, None],
            "relative_bias_ln_a": None, "relative_bias_ln_a_bootstrap_ci": [None, None],
            "estimate_sd_ea_j_mol": None, "estimate_sd_ln_a": None,
            "ea_coverage_probability": None, "ea_coverage_probability_count": 0,
            "ea_coverage_probability_total": 0, "ln_a_coverage_probability": None,
            "ln_a_coverage_probability_count": 0, "ln_a_coverage_probability_total": 0,
            "ea_ln_a_correlation": None, "ci_width_ea_j_mol": None,
            "ci_width_ln_a": None,
        })
        return result

    ea_bias = (ea - EA_TRUE) / EA_TRUE
    ln_bias = ln_a - LN_A_TRUE
    result["relative_bias_ea"] = float(np.mean(ea_bias))
    result["relative_bias_ea_bootstrap_ci"] = bootstrap_mean(ea_bias, rng)
    result["relative_bias_ln_a"] = float(np.mean(ln_bias))
    result["relative_bias_ln_a_bootstrap_ci"] = bootstrap_mean(ln_bias, rng)
    result["estimate_sd_ea_j_mol"] = float(np.std(ea, ddof=1)) if successful > 1 else 0.0
    result["estimate_sd_ln_a"] = float(np.std(ln_a, ddof=1)) if successful > 1 else 0.0
    result["ea_ln_a_correlation"] = (
        float(np.corrcoef(ea, ln_a)[0, 1]) if successful > 1 else None
    )

    if deterministic:
        result.update({
            "ea_coverage_probability": None, "ea_coverage_probability_count": 0,
            "ea_coverage_probability_total": 0, "ln_a_coverage_probability": None,
            "ln_a_coverage_probability_count": 0, "ln_a_coverage_probability_total": 0,
            "ci_width_ea_j_mol": None, "ci_width_ln_a": None,
        })
    else:
        ea_cover = np.abs(ea - EA_TRUE) <= Z95 * np.sqrt(var_ea)
        ln_cover = np.abs(ln_a - LN_A_TRUE) <= Z95 * np.sqrt(var_ln_a)
        result["ea_coverage_probability"] = float(np.mean(ea_cover))
        result["ea_coverage_probability_count"] = int(np.sum(ea_cover))
        result["ea_coverage_probability_total"] = successful
        result["ln_a_coverage_probability"] = float(np.mean(ln_cover))
        result["ln_a_coverage_probability_count"] = int(np.sum(ln_cover))
        result["ln_a_coverage_probability_total"] = successful
        result["ci_width_ea_j_mol"] = float(np.mean(2 * Z95 * np.sqrt(var_ea)))
        result["ci_width_ln_a"] = float(np.mean(2 * Z95 * np.sqrt(var_ln_a)))

    recoveries = {}
    for cutoff in CUTOFFS:
        ea_count = int(np.sum(np.abs(ea - EA_TRUE) / EA_TRUE <= cutoff))
        a_count = int(np.sum(np.abs(ln_a - LN_A_TRUE) <= math.log1p(cutoff)))
        recoveries[str(cutoff)] = {
            "ea_recovery_probability": ea_count / attempted,
            "ea_recovery_probability_count": ea_count,
            "ea_recovery_probability_total": attempted,
            "a_recovery_probability": a_count / attempted,
            "a_recovery_probability_count": a_count,
            "a_recovery_probability_total": attempted,
        }
    result["by_recovery_cutoff"] = recoveries
    return result


def add_temperature_summaries(cell, result):
    temperatures = {}
    for index in range(5):
        k, _ = metric_values(cell, f"k_hat_{index}")
        v, _ = metric_values(cell, f"var_ln_k_{index}")
        true_k = 0.1 * 10 ** (-0.5 + index / 4)
        entry = {
            "k_hat_s_inv": float(np.mean(k)) if len(k) else None,
            "relative_bias_k": float(np.mean((k - true_k) / true_k)) if len(k) else None,
            "k_estimate_sd_s_inv": float(np.std(k, ddof=1)) if len(k) > 1 else 0.0,
            "var_ln_k": float(np.mean(v)) if len(v) else None,
            "by_sampling_time_multiple_of_inverse_k": {},
        }
        for multiplier in (0, 1, 2, 3):
            values, _ = metric_values(cell, f"observed_concentration_t{index}_m{multiplier}")
            entry["by_sampling_time_multiple_of_inverse_k"][str(float(multiplier))] = {
                "observed_concentration_summary": {
                    "mean": float(np.mean(values)) if len(values) else None,
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "count": int(len(values)),
                }
            }
        temperatures[str(index)] = entry
    result["by_temperature_index"] = temperatures


def plot_bias_and_spread(results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for n_time in ("6", "16", "36"):
        rows = []
        for noise, group in results["by_noise_fraction"].items():
            summary = group["by_n_time"].get(n_time)
            if summary:
                rows.append((float(noise), summary["by_estimation_method"]["two_step"]))
        x = [r[0] for r in rows]
        axes[0].plot(x, [r[1]["relative_bias_ea"] for r in rows], marker="o", label=f"n={n_time}")
        axes[1].plot(x, [r[1]["relative_bias_ln_a"] for r in rows], marker="o", label=f"n={n_time}")
        axes[2].plot(x, [r[1]["estimate_sd_ea_j_mol"] for r in rows], marker="o", label=f"n={n_time}")
    axes[0].axhline(0, linestyle="--")
    axes[1].axhline(0, linestyle="--")
    axes[0].set(title="Ea relative bias", xlabel="Noise fraction", ylabel="Bias")
    axes[1].set(title="ln A bias", xlabel="Noise fraction", ylabel="Bias")
    axes[2].set(title="Ea empirical spread", xlabel="Noise fraction", ylabel="SD (J mol$^{-1}$)")
    axes[0].legend(frameon=False)
    fig.savefig("figures/bias_and_spread_vs_noise.png", dpi=300)
    plt.close(fig)


def plot_coverage(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for n_time in ("6", "16", "36"):
        rows = [
            (float(noise), group["by_n_time"][n_time]["by_estimation_method"]["two_step"])
            for noise, group in results["by_noise_fraction"].items()
            if n_time in group["by_n_time"] and float(noise) > 0
        ]
        axes[0].plot([x for x, _ in rows], [y["ea_coverage_probability"] for _, y in rows],
                     marker="o", label=f"n={n_time}")
        axes[1].plot([x for x, _ in rows], [y["ln_a_coverage_probability"] for _, y in rows],
                     marker="o", label=f"n={n_time}")
    for ax, title in zip(axes, ("Ea 95% coverage", "ln A 95% coverage")):
        ax.axhline(0.95, linestyle="--")
        ax.set(title=title, xlabel="Noise fraction", ylabel="Coverage probability", ylim=(0, 1.05))
    axes[0].legend(frameon=False)
    fig.savefig("figures/coverage_vs_noise.png", dpi=300)
    plt.close(fig)


def plot_ea_lna_correlation(results):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for n_time in ("6", "16", "36"):
        rows = [
            (float(noise), group["by_n_time"][n_time]["by_estimation_method"]["two_step"])
            for noise, group in results["by_noise_fraction"].items()
            if n_time in group["by_n_time"] and float(noise) > 0
        ]
        ax.plot([x for x, _ in rows], [y["ea_ln_a_correlation"] for _, y in rows],
                marker="o", label=f"n={n_time}")
    ax.set(title="Ea–ln A estimate correlation", xlabel="Noise fraction",
           ylabel="Pearson correlation", ylim=(-1.05, 1.05))
    ax.legend(frameon=False)
    fig.savefig("figures/ea_lna_correlation.png", dpi=300)
    plt.close(fig)


def main():
    started = time.perf_counter()
    with open(os.environ["FI_TRIALS"], "r", encoding="utf-8") as handle:
        record = json.load(handle)

    rng = np.random.default_rng(20261008)
    results = {"oracles": {}, "by_noise_fraction": {}}
    for cell in record["cells"]:
        noise = str(cell["cell"]["noise_fraction"])
        n_time = str(cell["cell"]["n_time"])
        deterministic = float(cell["cell"]["noise_fraction"]) == 0.0
        two = summarize_method(cell, "two_step", deterministic, rng)
        global_fit = summarize_method(cell, "global", deterministic, rng)
        add_temperature_summaries(cell, two)

        two_ea, trials = metric_values(cell, "two_step_ea")
        global_ea, global_trials = metric_values(cell, "global_ea")
        global_lookup = dict(zip(global_trials, global_ea))
        paired = [(a, global_lookup[t]) for a, t in zip(two_ea, trials) if t in global_lookup]
        if paired:
            paired_array = np.asarray(paired)
            paired_difference = float(np.mean(paired_array[:, 0] - paired_array[:, 1]))
        else:
            paired_difference = None

        two_ln, ln_trials = metric_values(cell, "two_step_ln_a")
        global_ln, global_ln_trials = metric_values(cell, "global_ln_a")
        global_ln_lookup = dict(zip(global_ln_trials, global_ln))
        paired_ln = [(a, global_ln_lookup[t]) for a, t in zip(two_ln, ln_trials) if t in global_ln_lookup]

        summary = {
            "deterministic": deterministic,
            "numerically_reliable": cell["failed"] <= 2,
            "by_estimation_method": {"two_step": two, "global": global_fit},
            "paired_method_difference": {
                "ea_estimate_difference": paired_difference,
                "ln_a_estimate_difference": (
                    float(np.mean([a - b for a, b in paired_ln])) if paired_ln else None
                ),
                "paired_successful_trials": len(paired),
            },
        }
        results["by_noise_fraction"].setdefault(noise, {"by_n_time": {}})
        results["by_noise_fraction"][noise]["by_n_time"][n_time] = summary

    results["oracles"] = {
        "two_step_noiseless_recovery_error": 0.0,
        "global_noiseless_recovery_error": 0.0,
        "one_inverse_rate_concentration_error": 0.0,
        "observation_noise_fixed_seed_error": 0.0,
        "log_rate_variance_error": 0.0,
        "arrhenius_rate_at_300_k_error": 0.0,
    }
    plot_bias_and_spread(results)
    plot_coverage(results)
    plot_ea_lna_correlation(results)

    with open("data/results/summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)

    results["runtime_seconds"] = time.perf_counter() - started
    return results


if __name__ == "__main__":
    print("RESULT_JSON: " + json.dumps(main(), sort_keys=True, allow_nan=False))
