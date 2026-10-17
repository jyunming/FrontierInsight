import csv
import json
import math
import os

import matplotlib
from scipy.optimize import curve_fit


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

EA_TRUE = 50000.0
LNA_TRUE = math.log(1.0e8)
Z975 = 1.959963984540054


def _values(metrics, name):
    item = metrics.get(name, {})
    return np.asarray(item.get("values", []), dtype=float), list(item.get("trials", []))


def _wilson(count, total):
    if total == 0:
        return [None, None]
    p = count / total
    denom = 1.0 + Z975**2 / total
    centre = (p + Z975**2 / (2 * total)) / denom
    half = Z975 * math.sqrt(p * (1 - p) / total + Z975**2 / (4 * total**2)) / denom
    return [centre - half, centre + half]


def _mean_ci(values):
    if len(values) < 2:
        return [None, None]
    half = stats.t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values))
    return [float(np.mean(values) - half), float(np.mean(values) + half)]


def _spread_ci(values):
    if len(values) < 2:
        return [None, None]
    n = len(values)
    variance = float(np.var(values, ddof=1))
    return [
        math.sqrt((n - 1) * variance / stats.chi2.ppf(0.975, n - 1)),
        math.sqrt((n - 1) * variance / stats.chi2.ppf(0.025, n - 1)),
    ]


def _correlation_ci(x, y):
    if len(x) <= 3:
        return None, [None, None]
    correlation = float(np.corrcoef(x, y)[0, 1])
    clipped = float(np.clip(correlation, -0.999999999, 0.999999999))
    half = Z975 / math.sqrt(len(x) - 3)
    return correlation, [float(math.tanh(math.atanh(clipped) - half)),
                         float(math.tanh(math.atanh(clipped) + half))]


def _method_summary(metrics, label):
    ea, trials = _values(metrics, f"ea_{label}")
    lna, _ = _values(metrics, f"lna_{label}")
    output = {}
    for parameter, values, truth in (("ea", ea, EA_TRUE), ("lna", lna, LNA_TRUE)):
        error = values - truth
        output[f"{parameter}_bias_J_mol" if parameter == "ea" else "lna_bias"] = {
            "estimate": float(np.mean(error)) if len(error) else None,
            "ci95": _mean_ci(error),
        }
        output[f"{parameter}_spread_J_mol" if parameter == "ea" else "lna_spread"] = {
            "estimate": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "ci95": _spread_ci(values),
        }
        for form in ("z", "t"):
            covered, ids = _values(metrics, f"{parameter}_{label}_covered_{form}")
            count = int(np.sum(covered))
            total = len(covered)
            output[f"{parameter}_coverage_95_{form}"] = float(count / total) if total else None
            output[f"{parameter}_coverage_95_{form}_count"] = count
            output[f"{parameter}_coverage_95_{form}_total"] = total
            output[f"{parameter}_coverage_95_{form}_values"] = covered.astype(int).tolist()
            output[f"{parameter}_coverage_95_{form}_pair_id"] = ids
            if form == "z":
                output[f"{parameter}_coverage_95_{form}_wilson_ci"] = _wilson(count, total)

    if label == "weighted":
        correlation, fisher = _correlation_ci(ea, lna)
        output["ea_lna_correlation"] = correlation
        output["ea_lna_correlation_fisher_ci"] = fisher
        temperatures = np.asarray([290.186778, 298.477829, 307.256589, 316.567394, 326.460126])
        x = 1.0 / temperatures
        weights = 1.0 / np.asarray([1, 1, 1, 1, 1], dtype=float)
        design = np.column_stack((np.ones(5), x - np.mean(x)))
        covariance = np.linalg.inv(design.T @ (weights[:, None] * design))
        transform = np.asarray([[0.0, -8.314462618], [1.0, -np.mean(x)]])
        parameter_covariance = transform @ covariance @ transform.T
        output["linearised_design_correlation"] = float(
            parameter_covariance[0, 1]
            / math.sqrt(parameter_covariance[0, 0] * parameter_covariance[1, 1])
        )

        temperatures = [290.186778, 298.477829, 307.256589, 316.567394, 326.460126]
        truths = np.geomspace(0.1, 1.0, 5)
        output["k_by_temperature"] = []
        for index, (temperature, truth) in enumerate(zip(temperatures, truths)):
            values, ids = _values(metrics, f"k_hat_{index}")
            covered, _ = _values(metrics, f"k_covered_{index}")
            errors = values - truth
            count = int(np.sum(covered))
            total = len(covered)
            output["k_by_temperature"].append({
                "temperature_K": temperature,
                "k_true_s_inv": float(truth),
                "k_hat_s_inv": float(np.mean(values)) if len(values) else None,
                "k_bias_s_inv": {"estimate": float(np.mean(errors)), "ci95": _mean_ci(errors)},
                "k_spread_s_inv": {
                    "estimate": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "ci95": _spread_ci(values),
                },
                "k_coverage_95": float(count / total) if total else None,
                "k_coverage_95_count": count,
                "k_coverage_95_total": total,
                "k_coverage_95_wilson_ci": _wilson(count, total),
                "k_coverage_95_values": covered.astype(int).tolist(),
                "k_coverage_95_pair_id": ids,
            })
    return output


def _figure(summary, parameter, filename, units):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    entries = []
    for noise, noise_block in summary.items():
        for n_time, cell in noise_block["by_n_time"].items():
            entries.append((float(noise), int(n_time), cell["by_method"]["weighted_centered"],
                            cell["by_method"]["unweighted_centered"]))
    for n_time in sorted({entry[1] for entry in entries}):
        rows = [entry for entry in entries if entry[1] == n_time]
        x = [entry[0] for entry in rows]
        weighted = [entry[2] for entry in rows]
        unweighted = [entry[3] for entry in rows]
        if parameter == "ea":
            bias_key, spread_key = "ea_bias_J_mol", "ea_spread_J_mol"
        else:
            bias_key, spread_key = "lna_bias", "lna_spread"
        axes[0].plot(x, [part[bias_key]["estimate"] for part in weighted], marker="o",
                     label=f"weighted, n={n_time}")
        axes[0].plot(x, [part[bias_key]["estimate"] for part in unweighted], marker="x",
                     linestyle="--", label=f"unweighted, n={n_time}")
        axes[1].plot(x, [part[spread_key]["estimate"] for part in weighted], marker="o")
        axes[1].plot(x, [part[spread_key]["estimate"] for part in unweighted],
                     marker="x", linestyle="--")
        axes[2].plot(x, [part[f"{parameter}_coverage_95_z"] for part in weighted], marker="o")
        axes[2].plot(x, [part[f"{parameter}_coverage_95_z"] for part in unweighted],
                     marker="x", linestyle="--")
    axes[0].axhline(0.0, linestyle=":")
    axes[2].axhline(0.95, linestyle=":")
    axes[0].set_ylabel(f"Bias ({units})")
    axes[1].set_ylabel(f"Empirical SD ({units})")
    axes[2].set_ylabel("95% z-interval coverage")
    for axis, title in zip(axes, ["Bias", "Estimate spread", "Interval calibration"]):
        axis.set_xlabel("Concentration noise SD / c0")
        axis.set_title(title)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def _correlation_figure(summary):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    entries = []
    for noise, noise_block in summary.items():
        for n_time, cell in noise_block["by_n_time"].items():
            weighted = cell["by_method"]["weighted_centered"]
            entries.append((float(noise), int(n_time), weighted))
    for n_time in sorted({entry[1] for entry in entries}):
        rows = [entry for entry in entries if entry[1] == n_time]
        ax.plot([row[0] for row in rows], [row[2]["ea_lna_correlation"] for row in rows],
                marker="o", label=f"n={n_time}")
    reference = entries[0][2]["linearised_design_correlation"]
    ax.axhline(reference, linestyle=":", label="linearised reference")
    ax.set(xlabel="Concentration noise SD / c0", ylabel="Correlation(Ea, ln A)",
           title="Strong Ea--ln A estimate correlation")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig("figures/parameter_correlation.png", dpi=300)
    plt.close(fig)


def main():
    with open(os.environ["FI_TRIALS"], encoding="utf-8") as handle:
        record = json.load(handle)

    by_noise = {}
    table_rows = []
    oracle_names = {
        "noiseless_decay_recovery": "max_k_relative_error",
        "generated_rate_ratio_and_arrhenius": "rate_ratio",
        "noiseless_arrhenius_recovery": "ea_arrhenius_relative_error",
        "noiseless_lnA_recovery": "lna_arrhenius_relative_error",
        "first_stage_variance_closed_form": "first_stage_variance_relative_error",
        "arrhenius_example_rate": "arrhenius_example_relative_error",
    }
    oracle_cases = {
        "noiseless_decay_recovery": {"noise_sigma_over_c0": 0.0, "n_time": 9},
        "generated_rate_ratio_and_arrhenius": {"noise_sigma_over_c0": 0.0, "n_time": 9},
        "noiseless_arrhenius_recovery": {"noise_sigma_over_c0": 0.01, "n_time": 9},
        "noiseless_lnA_recovery": {"noise_sigma_over_c0": 0.01, "n_time": 9},
        "first_stage_variance_closed_form": {"noise_sigma_over_c0": 0.01, "n_time": 9},
        "arrhenius_example_rate": {"noise_sigma_over_c0": 0.0, "n_time": 9},
    }

    for item in record["cells"]:
        cell = item["cell"]
        noise_key = str(cell["noise_sigma_over_c0"])
        time_key = str(cell["n_time"])
        metrics = item["metrics"]
        weighted = _method_summary(metrics, "weighted")
        unweighted = _method_summary(metrics, "unweighted")
        paired = {}
        for parameter in ("ea", "lna"):
            for form in ("z", "t"):
                w, ids = _values(metrics, f"{parameter}_weighted_covered_{form}")
                u, _ = _values(metrics, f"{parameter}_unweighted_covered_{form}")
                difference = w - u
                paired[f"{parameter}_{form}"] = {
                    "estimate": float(np.mean(difference)) if len(difference) else None,
                    "ci95": _mean_ci(difference),
                    "values": difference.tolist(),
                    "pair_id": ids,
                }
        entry = {
            "attempted_runs": int(item["planned"]),
            "successful_runs": int(item["ok"]),
            "failure_count": int(item["failed"]),
            "failure_rate": float(item["failed"] / item["planned"]) if item["planned"] else None,
            "by_method": {
                "weighted_centered": weighted,
                "unweighted_centered": unweighted,
            },
            "paired_coverage_difference": paired,
        }
        by_noise.setdefault(noise_key, {"by_n_time": {}})["by_n_time"][time_key] = entry
        table_rows.append({
            "noise_sigma_over_c0": cell["noise_sigma_over_c0"],
            "n_time": cell["n_time"],
            "weighted_ea_bias": weighted["ea_bias_J_mol"]["estimate"],
            "weighted_ea_coverage": weighted["ea_coverage_95_z"],
            "weighted_lna_bias": weighted["lna_bias"]["estimate"],
            "weighted_lna_coverage": weighted["lna_coverage_95_z"],
        })

    with open("data/results/summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)

    _figure(by_noise, "ea", "figures/ea_calibration.png", "J mol$^{-1}$")
    _figure(by_noise, "lna", "figures/lna_calibration.png", "dimensionless")
    _correlation_figure(by_noise)

    weighted_errors = []
    unweighted_errors = []
    for block in by_noise.values():
        for entry in block["by_n_time"].values():
            weighted_errors.append(abs(entry["by_method"]["weighted_centered"]["ea_coverage_95_z"] - 0.95))
            unweighted_errors.append(abs(entry["by_method"]["unweighted_centered"]["ea_coverage_95_z"] - 0.95))

    oracles = {}
    for name, metric_name in oracle_names.items():
        check_case = oracle_cases[name]
        values = []
        for item in record["cells"]:
            cell = item["cell"]
            if (
                float(cell["noise_sigma_over_c0"]) == check_case["noise_sigma_over_c0"]
                and int(cell["n_time"]) == check_case["n_time"]
            ):
                values.extend(item["metrics"].get(metric_name, {}).get("values", []))
        value = float(np.max(np.abs(values))) if values else None
        oracles[name] = {"value": value}

    convergence_cell = record["cells"][0]["cell"]
    convergence_check = _convergence_refit_check(
        noise_sigma_over_c0=float(convergence_cell["noise_sigma_over_c0"]),
        n_time=int(convergence_cell["n_time"]),
        seed=int(os.environ.get("FI_REPLICATE_SEED", "1729")),
    )

    median_weighted_error = float(np.median(weighted_errors))
    median_unweighted_error = float(np.median(unweighted_errors))
    result = {
        "oracles": oracles,
        "convergence_check": convergence_check,
        "by_noise_sigma_over_c0": by_noise,
        "acceptance": {
            "weighted_coverage_verdict": (
                "weighted-closer-to-nominal"
                if median_weighted_error <= median_unweighted_error
                else "unweighted-closer-to-nominal"
            ),
            "median_absolute_coverage_error_weighted": median_weighted_error,
            "median_absolute_coverage_error_unweighted": median_unweighted_error,
        },
    }
    with open("data/results/result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("RESULT_JSON: " + json.dumps(result, sort_keys=True))


def _convergence_refit_check(
    noise_sigma_over_c0,
    n_time,
    seed,
    runs=20,
    n_temperatures=5,
    rate_span=100.0,
    c0=1.0,
    minimum_k=1.0,
    tolerance=1.0e-8,
    tolerance_tightening=100.0,
    relative_change_limit=1.0e-6,
):
    rng = np.random.default_rng(seed)
    rates = minimum_k * np.geomspace(1.0, rate_span, n_temperatures)
    tight_tolerance = tolerance / tolerance_tightening
    maximum_change = 0.0

    for _ in range(runs):
        for k_true in rates:
            time = np.linspace(0.0, 3.0 / k_true, n_time)
            concentration = (
                c0 * np.exp(-k_true * time)
                + rng.normal(0.0, noise_sigma_over_c0 * c0, size=n_time)
            )

            def decay_model(sample_time, k):
                return c0 * np.exp(-k * sample_time)

            initial, _ = curve_fit(
                decay_model,
                time,
                concentration,
                p0=(k_true,),
                bounds=(0.0, np.inf),
                ftol=tolerance,
                xtol=tolerance,
                gtol=tolerance,
                max_nfev=10000,
            )
            tightened, _ = curve_fit(
                decay_model,
                time,
                concentration,
                p0=initial,
                bounds=(0.0, np.inf),
                ftol=tight_tolerance,
                xtol=tight_tolerance,
                gtol=tight_tolerance,
                max_nfev=10000,
            )
            maximum_change = max(
                maximum_change,
                float(abs(tightened[0] - initial[0]) / abs(tightened[0])),
            )

    return {
        "runs": runs,
        "max_relative_k_change": maximum_change,
        "passed": bool(maximum_change <= relative_change_limit),
    }


if __name__ == "__main__":
    main()
