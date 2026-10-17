import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("figures", exist_ok=True)

EA_TRUE_J_MOL = 50000.0
LN_A_TRUE = 13.815510557964274
NOMINAL_COVERAGE = 0.95
BOOTSTRAP_REPLICATES = 2000


def metric_values(cell: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    metric = cell.get("metrics", {}).get(name, {})
    return (
        np.asarray(metric.get("values", []), dtype=float),
        np.asarray(metric.get("trials", []), dtype=int),
    )


def bootstrap_ci(values: np.ndarray, statistic, rng: np.random.Generator) -> list:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        value = float(statistic(values)) if values.size else None
        return [value, value]
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    samples = values[indices]
    if statistic is np.mean:
        estimates = np.mean(samples, axis=1)
    elif statistic is np.std:
        estimates = np.std(samples, axis=1, ddof=1)
    else:
        estimates = np.array([statistic(sample) for sample in samples])
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def wilson_interval(count: int, total: int) -> list:
    if total == 0:
        return [None, None]
    interval = stats.binomtest(count, total).proportion_ci(
        confidence_level=0.95, method="wilson"
    )
    return [float(interval.low), float(interval.high)]


def summarize_method(cell: dict, method: str, rng: np.random.Generator) -> dict:
    ea, trials = metric_values(cell, f"{method}_Ea_hat")
    ln_a, _ = metric_values(cell, f"{method}_ln_A_hat")
    ea_low, _ = metric_values(cell, f"{method}_Ea_interval_low")
    ea_high, _ = metric_values(cell, f"{method}_Ea_interval_high")
    ln_a_low, _ = metric_values(cell, f"{method}_ln_A_interval_low")
    ln_a_high, _ = metric_values(cell, f"{method}_ln_A_interval_high")
    k_error, _ = metric_values(cell, f"{method}_k_relative_error")

    n = min(ea.size, ln_a.size, ea_low.size, ea_high.size, ln_a_low.size, ln_a_high.size)
    ea, ln_a = ea[:n], ln_a[:n]
    ea_low, ea_high = ea_low[:n], ea_high[:n]
    ln_a_low, ln_a_high = ln_a_low[:n], ln_a_high[:n]
    trials = trials[:n]

    ea_bias = ea - EA_TRUE_J_MOL
    ln_a_bias = ln_a - LN_A_TRUE
    ea_cover = ((ea_low <= EA_TRUE_J_MOL) & (EA_TRUE_J_MOL <= ea_high)).astype(int)
    ln_a_cover = ((ln_a_low <= LN_A_TRUE) & (LN_A_TRUE <= ln_a_high)).astype(int)

    correlation = float(np.corrcoef(ea, ln_a)[0, 1]) if n > 1 else None
    correlation_se = (
        float((1.0 - correlation ** 2) / np.sqrt(n - 2))
        if correlation is not None and n > 2 else None
    )
    correlation_ci = bootstrap_ci(
        np.column_stack((ea, ln_a)),
        lambda x: np.corrcoef(x[:, 0], x[:, 1])[0, 1],
        rng,
    ) if n > 2 else [None, None]

    attempted = int(cell.get("planned", n))
    successful = int(cell.get("ok", n))
    return {
        "noise_std_rel": float(cell["cell"]["noise_std_rel"]),
        "n_timepoints": int(cell["cell"]["n_timepoints"]),
        "completed_replicates": n,
        "fit_success_rate": successful / attempted if attempted else None,
        "fit_success_rate_count": successful,
        "fit_success_rate_total": attempted,
        "Ea_bias_J_mol": {
            "estimate": float(np.mean(ea_bias)) if n else None,
            "mc_se": float(np.std(ea_bias, ddof=1) / np.sqrt(n)) if n > 1 else None,
            "bootstrap_ci95": bootstrap_ci(ea_bias, np.mean, rng),
        },
        "ln_A_bias": {
            "estimate": float(np.mean(ln_a_bias)) if n else None,
            "mc_se": float(np.std(ln_a_bias, ddof=1) / np.sqrt(n)) if n > 1 else None,
            "bootstrap_ci95": bootstrap_ci(ln_a_bias, np.mean, rng),
        },
        "Ea_spread_J_mol": {
            "estimate": float(np.std(ea, ddof=1)) if n > 1 else None,
            "mc_se": float(np.std(ea, ddof=1) / np.sqrt(2 * (n - 1))) if n > 1 else None,
            "bootstrap_ci95": bootstrap_ci(ea, np.std, rng),
        },
        "ln_A_spread": {
            "estimate": float(np.std(ln_a, ddof=1)) if n > 1 else None,
            "mc_se": float(np.std(ln_a, ddof=1) / np.sqrt(2 * (n - 1))) if n > 1 else None,
            "bootstrap_ci95": bootstrap_ci(ln_a, np.std, rng),
        },
        "Ea_coverage": float(np.mean(ea_cover)) if n else None,
        "Ea_coverage_count": int(np.sum(ea_cover)),
        "Ea_coverage_total": n,
        "Ea_coverage_wilson_ci95": wilson_interval(int(np.sum(ea_cover)), n),
        "ln_A_coverage": float(np.mean(ln_a_cover)) if n else None,
        "ln_A_coverage_count": int(np.sum(ln_a_cover)),
        "ln_A_coverage_total": n,
        "ln_A_coverage_wilson_ci95": wilson_interval(int(np.sum(ln_a_cover)), n),
        "Ea_ln_A_correlation": {
            "estimate": correlation,
            "mc_se": correlation_se,
            "bootstrap_ci95": correlation_ci,
        },
        "k_hat": {
            "mean_relative_bias": float(np.mean(k_error)) if k_error.size else None,
            "mc_se": float(np.std(k_error, ddof=1) / np.sqrt(k_error.size))
            if k_error.size > 1 else None,
            "bootstrap_ci95": bootstrap_ci(k_error, np.mean, rng),
        },
        "reliable": successful >= 0.95 * attempted,
        "failed_run_identifiers": [],
    }


def plot_calibration_bias_spread(results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for method, payload in results["by_arrhenius_method"].items():
        settings = payload["settings"]
        x = np.array([s["noise_std_rel"] for s in settings])
        axes[0].plot(x, [s["Ea_bias_J_mol"]["estimate"] for s in settings],
                     marker="o", linestyle="", label=method)
        axes[1].plot(x, [s["ln_A_bias"]["estimate"] for s in settings],
                     marker="o", linestyle="", label=method)
        axes[2].plot(x, [s["Ea_spread_J_mol"]["estimate"] for s in settings],
                     marker="o", linestyle="", label=method)
    axes[0].axhline(0.0, linestyle="--")
    axes[1].axhline(0.0, linestyle="--")
    axes[0].set(title="Activation-energy bias", xlabel="Relative concentration noise",
                ylabel="Bias (J mol$^{-1}$)")
    axes[1].set(title="Pre-exponential bias", xlabel="Relative concentration noise",
                ylabel="Bias in ln(A)")
    axes[2].set(title="Activation-energy spread", xlabel="Relative concentration noise",
                ylabel="SD (J mol$^{-1}$)")
    axes[0].legend(frameon=False)
    fig.savefig("figures/calibration_bias_spread.png", dpi=300)
    plt.close(fig)


def plot_calibration_coverage(results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for method, payload in results["by_arrhenius_method"].items():
        settings = payload["settings"]
        x = np.array([s["noise_std_rel"] for s in settings])
        axes[0].plot(x, [s["Ea_coverage"] for s in settings], marker="o",
                     linestyle="", label=method)
        axes[1].plot(x, [s["ln_A_coverage"] for s in settings], marker="o",
                     linestyle="", label=method)
        axes[2].plot(x, [s["fit_success_rate"] for s in settings], marker="o",
                     linestyle="", label=method)
    for axis, title in zip(axes, ("Ea interval coverage", "ln(A) interval coverage",
                                  "Trajectory-fit success")):
        axis.axhline(NOMINAL_COVERAGE, linestyle="--")
        axis.set(title=title, xlabel="Relative concentration noise", ylabel="Proportion")
        axis.set_ylim(0.0, 1.05)
    axes[0].legend(frameon=False)
    fig.savefig("figures/calibration_coverage.png", dpi=300)
    plt.close(fig)


def plot_arrhenius_correlation_and_k_recovery(results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for method, payload in results["by_arrhenius_method"].items():
        settings = payload["settings"]
        x = np.array([s["noise_std_rel"] for s in settings])
        axes[0].plot(x, [s["Ea_ln_A_correlation"]["estimate"] for s in settings],
                     marker="o", linestyle="", label=method)
        axes[1].plot(x, [s["k_hat"]["mean_relative_bias"] for s in settings],
                     marker="o", linestyle="", label=method)
        axes[2].plot(x, [s["ln_A_spread"]["estimate"] for s in settings],
                     marker="o", linestyle="", label=method)
    axes[0].axhline(0.0, linestyle="--")
    axes[1].axhline(0.0, linestyle="--")
    axes[0].set(title="Ea–ln(A) correlation", xlabel="Relative concentration noise",
                ylabel="Pearson correlation")
    axes[1].set(title="Rate-constant recovery", xlabel="Relative concentration noise",
                ylabel="Mean relative k error")
    axes[2].set(title="ln(A) estimate spread", xlabel="Relative concentration noise",
                ylabel="SD of ln(A)")
    axes[0].legend(frameon=False)
    fig.savefig("figures/arrhenius_correlation_and_k_recovery.png", dpi=300)
    plt.close(fig)


def main() -> dict:
    with open(os.environ["FI_TRIALS"], "r", encoding="utf-8") as handle:
        trials = json.load(handle)

    rng = np.random.default_rng(20261014)
    result = {
        "oracles": trials.get("oracles", {}),
        "constants": {
            "Ea_true_J_mol": EA_TRUE_J_MOL,
            "ln_A_true": LN_A_TRUE,
            "nominal_coverage": NOMINAL_COVERAGE,
            "n_replicates_attempted": int(trials["cells"][0].get("planned", 0)),
        },
        "by_arrhenius_method": {"weighted": {"settings": []}, "unweighted": {"settings": []}},
    }

    for cell in trials["cells"]:
        for method in ("weighted", "unweighted"):
            result["by_arrhenius_method"][method]["settings"].append(
                summarize_method(cell, method, rng)
            )

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/calibration_summaries.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)

    plot_calibration_bias_spread(result)
    plot_calibration_coverage(result)
    plot_arrhenius_correlation_and_k_recovery(result)
    return result


if __name__ == "__main__":
    result = main()
    print("RESULT_JSON: " + json.dumps(result, sort_keys=True, allow_nan=False))