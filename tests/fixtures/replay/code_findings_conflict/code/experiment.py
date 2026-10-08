import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

EA_TRUE = 50000.0
A_TRUE = 1.0e10
R = 8.314
REFERENCE_COVERAGE = 0.95


def _temperatures():
    high = 1.0 / (1.0 / 300.0 - R * math.log(10.0) / EA_TRUE)
    return np.linspace(1.0 / 300.0, 1.0 / high, 5) ** -1


def _values(metric):
    return np.asarray(metric.get("values", []), dtype=float)


def _wilson(count, total):
    if total == 0:
        return [None, None]
    low, high = stats.binomtest(int(count), int(total)).proportion_ci(
        confidence_level=0.95, method="wilson"
    )
    return [float(low), float(high)]


def _mean_ci(values):
    if len(values) < 2:
        value = float(values[0]) if len(values) else None
        return value, [value, value]
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean, [mean - half, mean + half]


def _bootstrap(values, statistic, rng, n_resamples=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        value = float(statistic(values)) if len(values) else None
        return value, [value, value]
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    sampled = values[indices]
    if statistic is np.std:
        draws = np.std(sampled, axis=1, ddof=1)
        estimate = float(np.std(values, ddof=1))
    else:
        draws = np.asarray([statistic(sample) for sample in sampled], dtype=float)
        estimate = float(statistic(values))
    return estimate, [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _summarize(record, temperatures, true_k, rng):
    metrics = record["metrics"]
    planned = int(record["planned"])
    failed = int(record["failed"])
    ea = _values(metrics.get("ea_fit", {}))
    lna = _values(metrics.get("lnA_fit", {}))
    ea_cover = _values(metrics.get("ea_coverage", {}))
    lna_cover = _values(metrics.get("lnA_coverage", {}))
    ea_error = ea - EA_TRUE
    lna_error = lna - math.log(A_TRUE)

    ea_bias, ea_bias_ci = _mean_ci(ea_error)
    lna_bias, lna_bias_ci = _mean_ci(lna_error)
    ea_spread, ea_spread_ci = _bootstrap(ea_error, np.std, rng)
    lna_spread, lna_spread_ci = _bootstrap(lna_error, np.std, rng)

    ea_count = int(np.sum(ea_cover))
    lna_count = int(np.sum(lna_cover))
    summary = {
        "attempted_trials": planned,
        "valid_trials": int(len(ea)),
        "failure_count": failed,
        "fit_failure_probability": failed / planned if planned else None,
        "fit_failure_probability_wilson_95": _wilson(failed, planned),
        "ea_coverage_probability": ea_count / planned if planned else None,
        "ea_coverage_count": ea_count,
        "ea_coverage_total": planned,
        "ea_coverage_probability_wilson_95": _wilson(ea_count, planned),
        "ea_bias_J_per_mol": ea_bias,
        "ea_bias_student_t_95": ea_bias_ci,
        "ea_spread_J_per_mol": ea_spread,
        "ea_spread_bootstrap_95": ea_spread_ci,
        "lnA_coverage_probability": lna_count / planned if planned else None,
        "lnA_coverage_count": lna_count,
        "lnA_coverage_total": planned,
        "lnA_coverage_probability_wilson_95": _wilson(lna_count, planned),
        "lnA_bias": lna_bias,
        "lnA_bias_student_t_95": lna_bias_ci,
        "lnA_spread": lna_spread,
        "lnA_spread_bootstrap_95": lna_spread_ci,
        "ea_lnA_correlation": float(np.corrcoef(ea, lna)[0, 1]) if len(ea) > 1 else None,
        "ea_lnA_correlation_bootstrap_95": [None, None],
        "by_temperature_K": {},
    }

    if len(ea) > 2:
        samples = rng.integers(0, len(ea), size=(2000, len(ea)))
        correlations = np.array([
            np.corrcoef(ea[index], lna[index])[0, 1] for index in samples
        ])
        summary["ea_lnA_correlation_bootstrap_95"] = [
            float(np.nanquantile(correlations, 0.025)),
            float(np.nanquantile(correlations, 0.975)),
        ]

    for index, (temperature, true_rate) in enumerate(zip(temperatures, true_k)):
        fitted = _values(metrics.get(f"k_fit_{index}", {}))
        coverage = _values(metrics.get(f"k_coverage_{index}", {}))
        variances = _values(metrics.get(f"ln_k_variance_{index}", {}))
        relative_error = fitted / true_rate - 1.0
        bias, bias_ci = _mean_ci(relative_error)
        spread, spread_ci = _bootstrap(relative_error, np.std, rng)
        variance_mean, variance_ci = _mean_ci(variances)
        count = int(np.sum(coverage))
        summary["by_temperature_K"][f"{temperature:.1f}"] = {
            "k_true_per_s": float(true_rate),
            "k_relative_bias": bias,
            "k_relative_bias_student_t_95": bias_ci,
            "k_relative_spread": spread,
            "k_relative_spread_bootstrap_95": spread_ci,
            "k_coverage_probability": count / planned if planned else None,
            "k_coverage_count": count,
            "k_coverage_total": planned,
            "k_coverage_probability_wilson_95": _wilson(count, planned),
            "ln_k_variance": variance_mean,
            "ln_k_variance_student_t_95": variance_ci,
        }
    return summary


def _ordered(setting_results):
    return sorted(
        setting_results,
        key=lambda item: (item["noise_ratio"], item["num_time_points"]),
    )


def _ci_half(estimate, interval):
    if estimate is None or interval[0] is None or interval[1] is None:
        return None
    return max(float(estimate) - float(interval[0]), float(interval[1]) - float(estimate))


def _validate_setting_grid(record):
    cells = record["cells"]
    actual = [
        (float(cell_record["cell"]["noise_ratio"]), int(cell_record["cell"]["num_time_points"]))
        for cell_record in cells
    ]
    if len(set(actual)) != len(actual):
        raise ValueError("The supplied setting grid contains duplicate cells.")

    sweep = record.get("sweep", record.get("planned_sweep", {}))
    noise = sweep.get("noise_ratios", sorted({item[0] for item in actual}))
    points = sweep.get("num_time_points", sorted({item[1] for item in actual}))
    expected = {
        (float(noise_ratio), int(num_time_points))
        for noise_ratio in noise
        for num_time_points in points
    }
    if set(actual) != expected:
        raise ValueError("The supplied cells do not match the planned setting grid.")
    return cells


def make_coverage_heatmap(setting_results):
    noise = sorted({item["noise_ratio"] for item in setting_results})
    points = sorted({item["num_time_points"] for item in setting_results})
    ea = np.full((len(points), len(noise)), np.nan)
    lna = np.full_like(ea, np.nan)
    ea_annotations = np.full(ea.shape, "", dtype=object)
    lna_annotations = np.full(ea.shape, "", dtype=object)
    for item in setting_results:
        row, col = points.index(item["num_time_points"]), noise.index(item["noise_ratio"])
        summary = item["summary"]
        ea[row, col] = summary["ea_coverage_probability"]
        lna[row, col] = summary["lnA_coverage_probability"]
        for annotations, prefix in ((ea_annotations, "ea"), (lna_annotations, "lnA")):
            count = summary[f"{prefix}_coverage_count"]
            total = summary[f"{prefix}_coverage_total"]
            low, high = summary[f"{prefix}_coverage_probability_wilson_95"]
            annotations[row, col] = (
                f"{count / total:.2f}\n95% [{low:.2f}, {high:.2f}]\nn={total}"
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle(f"Operational coverage; reference = {REFERENCE_COVERAGE:.2f}")
    for axis, matrix, annotations, title in zip(
        axes,
        [ea, lna],
        [ea_annotations, lna_annotations],
        ["Ea 95% coverage", "ln A 95% coverage"],
    ):
        image = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(range(len(noise)), [str(value) for value in noise])
        axis.set_yticks(range(len(points)), [str(value) for value in points])
        axis.set(xlabel="Noise ratio σ/c₀", ylabel="Time points", title=title)
        for row in range(len(points)):
            for col in range(len(noise)):
                axis.text(
                    col, row, annotations[row, col],
                    ha="center", va="center", fontsize=7,
                )
        fig.colorbar(image, ax=axis, label="Operational coverage")
    fig.savefig("figures/coverage_heatmap.png", dpi=300)
    plt.close(fig)


def make_accuracy_vs_noise(setting_results):
    points = sorted({item["num_time_points"] for item in setting_results})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    for n_points in points:
        rows = sorted(
            (item for item in setting_results if item["num_time_points"] == n_points),
            key=lambda item: item["noise_ratio"],
        )
        x = [item["noise_ratio"] for item in rows]
        label = f"n={n_points}"

        ea_bias = [item["summary"]["ea_bias_J_per_mol"] for item in rows]
        ea_bias_err = [
            _ci_half(value, item["summary"]["ea_bias_student_t_95"])
            for value, item in zip(ea_bias, rows)
        ]
        ea_spread = [item["summary"]["ea_spread_J_per_mol"] for item in rows]
        ea_spread_err = [
            _ci_half(value, item["summary"]["ea_spread_bootstrap_95"])
            for value, item in zip(ea_spread, rows)
        ]
        lna_bias = [item["summary"]["lnA_bias"] for item in rows]
        lna_bias_err = [
            _ci_half(value, item["summary"]["lnA_bias_student_t_95"])
            for value, item in zip(lna_bias, rows)
        ]
        lna_spread = [item["summary"]["lnA_spread"] for item in rows]
        lna_spread_err = [
            _ci_half(value, item["summary"]["lnA_spread_bootstrap_95"])
            for value, item in zip(lna_spread, rows)
        ]
        correlation = [item["summary"]["ea_lnA_correlation"] for item in rows]
        correlation_err = [
            _ci_half(value, item["summary"]["ea_lnA_correlation_bootstrap_95"])
            for value, item in zip(correlation, rows)
        ]

        axes[0, 0].errorbar(x, ea_bias, yerr=ea_bias_err, marker="o", capsize=3, label=label)
        axes[0, 1].errorbar(x, ea_spread, yerr=ea_spread_err, marker="o", capsize=3, label=label)
        axes[1, 0].errorbar(x, lna_bias, yerr=lna_bias_err, marker="o", capsize=3, label=label)
        axes[1, 1].errorbar(x, lna_spread, yerr=lna_spread_err, marker="o", capsize=3, label=label)
        axes[0, 2].errorbar(x, correlation, yerr=correlation_err, marker="o", capsize=3, label=label)
        for noise_ratio, value in zip(x, correlation):
            axes[0, 2].annotate(
                f"r={value:.2f}",
                (noise_ratio, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    for axis in (axes[0, 0], axes[1, 0]):
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Ea bias", xlabel="Noise ratio σ/c₀", ylabel="Bias (J mol⁻¹)")
    axes[0, 1].set(title="Ea standard deviation", xlabel="Noise ratio σ/c₀", ylabel="SD (J mol⁻¹)")
    axes[1, 0].set(title="ln A bias", xlabel="Noise ratio σ/c₀", ylabel="Bias")
    axes[1, 1].set(title="ln A standard deviation", xlabel="Noise ratio σ/c₀", ylabel="SD")
    axes[0, 2].set(title="Ea–ln A trial correlation", xlabel="Noise ratio σ/c₀", ylabel="Correlation")
    axes[1, 2].axis("off")
    axes[0, 0].legend(frameon=False)
    fig.savefig("figures/accuracy_vs_noise.png", dpi=300)
    plt.close(fig)


def make_k_by_temperature(setting_results, temperatures):
    points = sorted({item["num_time_points"] for item in setting_results})
    fig, axes = plt.subplots(
        len(points), 3, figsize=(14, 4 * len(points)),
        constrained_layout=True, squeeze=False,
    )

    for row, n_points in enumerate(points):
        rows = sorted(
            (item for item in setting_results if item["num_time_points"] == n_points),
            key=lambda item: item["noise_ratio"],
        )
        for item in rows:
            values = list(item["summary"]["by_temperature_K"].values())
            label = f"σ/c₀={item['noise_ratio']}"
            bias = [value["k_relative_bias"] for value in values]
            bias_err = [
                _ci_half(value["k_relative_bias"], value["k_relative_bias_student_t_95"])
                for value in values
            ]
            spread = [value["k_relative_spread"] for value in values]
            spread_err = [
                _ci_half(value["k_relative_spread"], value["k_relative_spread_bootstrap_95"])
                for value in values
            ]
            coverage = [value["k_coverage_probability"] for value in values]
            coverage_err = [
                _ci_half(value["k_coverage_probability"], value["k_coverage_probability_wilson_95"])
                for value in values
            ]
            axes[row, 0].errorbar(
                temperatures, bias, yerr=bias_err, marker="o", capsize=3, label=label
            )
            axes[row, 1].errorbar(
                temperatures, spread, yerr=spread_err, marker="o", capsize=3, label=label
            )
            axes[row, 2].errorbar(
                temperatures, coverage, yerr=coverage_err, marker="o", capsize=3, label=label
            )

        axes[row, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[row, 2].axhline(REFERENCE_COVERAGE, color="black", linestyle="--", linewidth=1)
        axes[row, 0].set(
            title=f"Relative k bias (n={n_points})",
            xlabel="Temperature (K)",
            ylabel="Relative bias",
        )
        axes[row, 1].set(
            title=f"Relative k standard deviation (n={n_points})",
            xlabel="Temperature (K)",
            ylabel="Relative SD",
        )
        axes[row, 2].set(
            title=f"k 95% coverage (n={n_points})",
            xlabel="Temperature (K)",
            ylabel="Operational coverage",
        )
        axes[row, 0].legend(frameon=False)

    fig.savefig("figures/k_by_temperature.png", dpi=300)
    plt.close(fig)


def _fit_arrhenius(temperatures, rates):
    inverse_temperature = 1.0 / np.asarray(temperatures, dtype=float)
    centered_inverse_temperature = inverse_temperature - np.mean(inverse_temperature)
    slope, intercept = np.polyfit(centered_inverse_temperature, np.log(rates), 1)
    ea = float(-R * slope)
    lna = float(intercept - slope * np.mean(inverse_temperature))
    return ea, lna


def _computed_checks(temperatures, true_k):
    c0 = 1.0
    nonunit_c0 = 2.75
    times = np.linspace(0.0, 3.0 / np.min(true_k), 31)
    concentration = c0 * np.exp(-np.min(true_k) * times)
    concentration_nonunit = nonunit_c0 * np.exp(-np.min(true_k) * times)

    concentration_closed_form_absolute_error = float(np.max(np.abs(
        concentration - c0 * np.exp(-np.min(true_k) * times)
    )))
    concentration_nonunit_initial_absolute_error = float(abs(
        concentration_nonunit[0] - nonunit_c0
    ))
    arrhenius_closed_form_relative_error = float(np.max(np.abs(
        true_k / (A_TRUE * np.exp(-EA_TRUE / (R * temperatures))) - 1.0
    )))
    arrhenius_log_relation_absolute_error = float(np.max(np.abs(
        np.log(true_k)
        - (math.log(A_TRUE) - (EA_TRUE / R) * (1.0 / temperatures))
    )))

    fitted_rates_default = []
    fitted_rates_tight = []
    for temperature, rate in zip(temperatures, true_k):
        sample_times = np.linspace(0.0, 3.0 / rate, 11)
        noiseless = c0 * np.exp(-rate * sample_times)
        default_fit, _ = optimize.curve_fit(
            lambda time_values, fitted_rate: c0 * np.exp(-fitted_rate * time_values),
            sample_times,
            noiseless,
            p0=[rate],
        )
        tight_fit, _ = optimize.curve_fit(
            lambda time_values, fitted_rate: c0 * np.exp(-fitted_rate * time_values),
            sample_times,
            noiseless,
            p0=[rate],
            xtol=1.0e-14,
            ftol=1.0e-14,
            gtol=1.0e-14,
        )
        fitted_rates_default.append(default_fit[0])
        fitted_rates_tight.append(tight_fit[0])

    noise_free_ea, _ = _fit_arrhenius(temperatures, fitted_rates_default)
    tight_ea, _ = _fit_arrhenius(temperatures, fitted_rates_tight)
    endpoint_times = 3.0 / true_k
    endpoint_ratio = (
        c0 * np.exp(-true_k * endpoint_times)
    ) / c0

    return {
        "concentration_closed_form_absolute_error": concentration_closed_form_absolute_error,
        "concentration_nonunit_initial_absolute_error": concentration_nonunit_initial_absolute_error,
        "arrhenius_closed_form_relative_error": arrhenius_closed_form_relative_error,
        "arrhenius_log_relation_absolute_error": arrhenius_log_relation_absolute_error,
        "noise_free_Ea_fit_J_per_mol": noise_free_ea,
        "noise_free_Ea_numerical_error_J_per_mol": float(abs(noise_free_ea - EA_TRUE)),
        "monotone_noiseless": bool(np.all(np.diff(concentration) <= 0.0)),
        "endpoint_ratio_absolute_error": float(np.max(np.abs(endpoint_ratio - math.exp(-3.0)))),
        "max_tolerance_tightening_change_Ea_J_per_mol": float(abs(tight_ea - noise_free_ea)),
    }


def main():
    started = time.perf_counter()
    with open(os.environ["FI_TRIALS"], "r", encoding="utf-8") as handle:
        record = json.load(handle)

    temperatures = _temperatures()
    true_k = np.array([A_TRUE * math.exp(-EA_TRUE / (R * temp)) for temp in temperatures])
    rng = np.random.default_rng(20261008)
    setting_results = []
    by_noise = {}

    for cell_record in _validate_setting_grid(record):
        cell = cell_record["cell"]
        noise_key = str(cell["noise_ratio"])
        points_key = str(cell["num_time_points"])
        summary = _summarize(cell_record, temperatures, true_k, rng)
        setting_results.append({
            "noise_ratio": float(cell["noise_ratio"]),
            "num_time_points": int(cell["num_time_points"]),
            "summary": summary,
        })
        by_noise.setdefault(noise_key, {"by_num_time_points": {}})
        by_noise[noise_key]["by_num_time_points"][points_key] = summary

    make_coverage_heatmap(setting_results)
    make_accuracy_vs_noise(setting_results)
    make_k_by_temperature(setting_results, temperatures)

    compact_rows = []
    for item in _ordered(setting_results):
        compact_rows.append({
            "noise_ratio": item["noise_ratio"],
            "num_time_points": item["num_time_points"],
            "summary": item["summary"],
        })
    with open("data/results/arrhenius_recovery_results.json", "w", encoding="utf-8") as handle:
        json.dump(compact_rows, handle, indent=2)

    return {
        "runtime_seconds": time.perf_counter() - started,
        "constants": {
            "c0_mol_per_l": 1.0,
            "A_per_s": A_TRUE,
            "Ea_true_J_per_mol": EA_TRUE,
            "lnA_true": math.log(A_TRUE),
            "R_J_per_mol_K": R,
        },
        "checks": _computed_checks(temperatures, true_k),
        "by_noise_ratio": by_noise,
    }


if __name__ == "__main__":
    results = main()
    print("RESULT_JSON: " + json.dumps(results, sort_keys=True, allow_nan=False))
