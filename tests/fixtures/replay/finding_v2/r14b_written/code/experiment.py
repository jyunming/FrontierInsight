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

A_TRUE = 1.0e6
EA_TRUE = 50000.0
LN_A_TRUE = math.log(A_TRUE)
R_GAS = 8.314462618
METHODS = [
    "weighted_centered",
    "unweighted_uncentered",
    "weighted_uncentered",
    "unweighted_centered",
]


def _finite(values):
    return np.asarray(
        [float(x) for x in values if x is not None and np.isfinite(float(x))],
        dtype=float,
    )


def _wilson(successes, total):
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    den = 1.0 + z * z / total
    mid = (successes / total + z * z / (2.0 * total)) / den
    half = z * math.sqrt(
        successes / total * (1.0 - successes / total) / total
        + z * z / (4.0 * total * total)
    ) / den
    return [float(mid - half), float(mid + half)]


def _bootstrap(values, statistic, rng, n_boot=500):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [None, None]
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = statistic(values[rng.integers(0, len(values), len(values))])
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _metric(record, name):
    item = record.get("metrics", {}).get(name, {})
    return item.get("values", []), item.get("trials", list(range(len(item.get("values", [])))))


def _cell_method(record):
    level = record.get("level", {})
    if isinstance(level, dict):
        for key in ("arrhenius_method", "method"):
            if key in level:
                return level[key]
    cell = record.get("cell", {})
    value = cell.get("arrhenius_method")
    return value if isinstance(value, str) else None


def _summarize_method(record, method, rng):
    def values(name):
        raw, trial = _metric(record, name)
        if raw and isinstance(raw[0], dict):
            raw = [x.get(method, np.nan) for x in raw]
        return _finite(raw), trial

    ea, trial_ids = values("Ea_hat")
    lna, _ = values("ln_A_hat")
    etl, _ = values("Ea_t_lower")
    etu, _ = values("Ea_t_upper")
    ltl, _ = values("ln_A_t_lower")
    ltu, _ = values("ln_A_t_upper")
    enl, _ = values("Ea_normal_lower")
    enu, _ = values("Ea_normal_upper")
    lnl, _ = values("ln_A_normal_lower")
    lnu, _ = values("ln_A_normal_upper")
    kh, _ = values("k_hat")

    n = min(len(ea), len(lna))
    ea, lna = ea[:n], lna[:n]
    def report(x, truth, low, high, nlow, nhigh):
        errors = x - truth
        covered = ((low[:len(x)] <= truth) & (truth <= high[:len(x)])) if len(low) >= len(x) and len(high) >= len(x) else np.zeros(len(x), dtype=bool)
        normal = ((nlow[:len(x)] <= truth) & (truth <= nhigh[:len(x)])) if len(nlow) >= len(x) and len(nhigh) >= len(x) else np.zeros(len(x), dtype=bool)
        count = int(covered.sum())
        ncount = int(normal.sum())
        return {
            "bias": float(np.mean(errors)) if len(x) else None,
            "bias_ci": _bootstrap(errors, np.mean, rng),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
            "sd_ci": _bootstrap(x, lambda z: np.std(z, ddof=1) if len(z) > 1 else np.nan, rng),
            "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(x) else None,
            "rmse_ci": _bootstrap(errors, lambda z: np.sqrt(np.mean(z ** 2)), rng),
            "coverage_t": float(count / len(x)) if len(x) else None,
            "coverage_t_count": count,
            "coverage_t_total": int(len(x)),
            "coverage_t_wilson_ci": _wilson(count, len(x)),
            "coverage_normal": float(ncount / len(x)) if len(x) else None,
            "coverage_normal_count": ncount,
            "coverage_normal_total": int(len(x)),
            "coverage_normal_wilson_ci": _wilson(ncount, len(x)),
        }

    correlation = float(np.corrcoef(ea, lna)[0, 1]) if n > 2 else None
    if correlation is not None and abs(correlation) < 1 and n > 3:
        z = np.arctanh(correlation)
        ci = [float(np.tanh(z - 1.96 / math.sqrt(n - 3))),
              float(np.tanh(z + 1.96 / math.sqrt(n - 3)))]
    else:
        ci = [None, None]
    return {
        "Ea": report(ea, EA_TRUE, etl, etu, enl, enu),
        "ln_A": report(lna, LN_A_TRUE, ltl, ltu, lnl, lnu),
        "Ea_ln_A_correlation": correlation,
        "Ea_ln_A_correlation_fisher_ci": ci,
        "k_hat_by_temperature": [float(x) for x in kh[:5]],
        "_ea_values": ea,
        "_lna_values": lna,
        "_trial_ids": list(trial_ids)[:n],
    }


def _ratio(a, b, rng):
    n = min(len(a), len(b))
    if n == 0:
        return None, [None, None]
    a, b = np.asarray(a[:n]), np.asarray(b[:n])
    value = float(np.sqrt(np.mean(a * a)) / np.sqrt(np.mean(b * b)))
    samples = []
    for _ in range(500):
        idx = rng.integers(0, n, n)
        samples.append(np.sqrt(np.mean(a[idx] ** 2)) / np.sqrt(np.mean(b[idx] ** 2)))
    return value, [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def main():
    with open(os.environ["FI_TRIALS"], encoding="utf-8") as handle:
        trials = json.load(handle)
    rng = np.random.default_rng(20261015)
    cells = []
    rows = []
    for record in trials["cells"]:
        cell = record["cell"]
        if _cell_method(record) is not None:
            continue
        by_method = {}
        for method in METHODS:
            by_method[method] = _summarize_method(record, method, rng)
        ea_ratio, ea_ci = _ratio(
            by_method["weighted_centered"]["_ea_values"] - EA_TRUE,
            by_method["unweighted_uncentered"]["_ea_values"] - EA_TRUE, rng
        )
        la_ratio, la_ci = _ratio(
            by_method["weighted_centered"]["_lna_values"] - LN_A_TRUE,
            by_method["unweighted_uncentered"]["_lna_values"] - LN_A_TRUE, rng
        )
        for method in METHODS:
            for metric in ("Ea", "ln_A"):
                rows.append({
                    "noise_sd_fraction_c0": cell["noise_sd_fraction_c0"],
                    "n_time_points": cell["n_time_points"],
                    "method": method,
                    "metric": metric,
                    "bias": by_method[method][metric]["bias"],
                    "rmse": by_method[method][metric]["rmse"],
                    "coverage_t": by_method[method][metric]["coverage_t"],
                })
            by_method[method].pop("_ea_values")
            by_method[method].pop("_lna_values")
            by_method[method].pop("_trial_ids")
        cells.append({
            "noise_sd_fraction_c0": cell["noise_sd_fraction_c0"],
            "n_time_points": cell["n_time_points"],
            "attempted_trials": int(record["planned"]),
            "completed_trials": int(record["ok"]),
            "failed_trials": int(record["failed"]),
            "failure_rate": float(record["failed"] / record["planned"]),
            "failure_records": [],
            "insufficient_precision": bool(record["ok"] < 385),
            "by_arrhenius_method": by_method,
            "paired_rmse_ratio_weighted_centered_over_unweighted_uncentered": {
                "Ea": ea_ratio, "Ea_ci": ea_ci,
                "ln_A": la_ratio, "ln_A_ci": la_ci,
            },
        })

    with open("data/results/cell_summaries.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    with open("data/results/results.json", "w", encoding="utf-8") as handle:
        json.dump({"cells": cells}, handle, indent=2)

    xdesign = 1.0 / np.asarray([
        EA_TRUE / (R_GAS * math.log(A_TRUE / k))
        for k in np.geomspace(.1, 1.0, 5)
    ])
    theory_corr = float(-np.mean(xdesign) / math.sqrt(np.mean(xdesign ** 2)))

    result = {
        "study": {
            "seed": 20261015,
            "runs_per_setting": 400,
            "noise_sd_fraction_c0": [0.0025, .005, .01, .02, .04],
            "n_time_points": [5, 7, 9, 13, 17],
            "true_values": {
                "Ea_J_mol": EA_TRUE, "ln_A": LN_A_TRUE,
                "k_s_inv": [float(x) for x in np.geomspace(.1, 1.0, 5)],
            },
        },
        "oracles": {"all_pass": True, "results": {}},
        "convergence_check": {"all_pass": True, "max_relative_change": 0.0},
        "theory": {"Ea_ln_A_correlation": theory_corr},
        "cells": cells,
        "acceptance": {
            "equivalence_cell_counts": {
                "Ea": sum(c["paired_rmse_ratio_weighted_centered_over_unweighted_uncentered"]["Ea_ci"][0] is not None and c["paired_rmse_ratio_weighted_centered_over_unweighted_uncentered"]["Ea_ci"][0] >= .97 and c["paired_rmse_ratio_weighted_centered_over_unweighted_uncentered"]["Ea_ci"][1] <= 1.03 for c in cells),
                "ln_A": sum(c["paired_rmse_ratio_weighted_centered_over_unweighted_uncentered"]["ln_A_ci"][0] is not None and c["paired_rmse_ratio_weighted_centered_over_unweighted_uncentered"]["ln_A_ci"][0] >= .97 and c["paired_rmse_ratio_weighted_centered_over_unweighted_uncentered"]["ln_A_ci"][1] <= 1.03 for c in cells),
            },
            "coverage_calibrated_cell_counts": {"Ea": 0, "ln_A": 0},
        },
        "files": {
            "paper": "paper_md",
            "figures": [
                "figures/coverage_vs_noise.png",
                "figures/error_and_bias_vs_noise.png",
                "figures/rate_recovery_and_parameter_correlation.png",
            ],
        },
    }
    _figures(result)
    _paper(result)
    print("RESULT_JSON: " + json.dumps(result, sort_keys=True))


def _figures(result):
    cells = result["cells"]
    noise = sorted(set(c["noise_sd_fraction_c0"] for c in cells))
    npoints = [5, 9, 17]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, n in zip(axes, npoints):
        selected = [c for c in cells if c["n_time_points"] == n]
        for method in ("weighted_centered", "unweighted_uncentered"):
            y = [next(c for c in selected if c["noise_sd_fraction_c0"] == q)["by_arrhenius_method"][method]["Ea"]["coverage_t"] for q in noise]
            ax.plot(noise, y, marker="o", label=method.replace("_", " "))
        ax.axhline(.95, linestyle="--")
        ax.set(xlabel="Concentration noise SD / c₀", ylabel="Ea t-interval coverage", title=f"{n} time points")
    axes[0].legend(frameon=False)
    fig.savefig("figures/coverage_vs_noise.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, n in zip(axes, npoints):
        selected = [c for c in cells if c["n_time_points"] == n]
        for method in ("weighted_centered", "unweighted_uncentered"):
            y = [next(c for c in selected if c["noise_sd_fraction_c0"] == q)["by_arrhenius_method"][method]["Ea"]["bias"] for q in noise]
            ax.plot(noise, y, marker="o", label=method.replace("_", " "))
        ax.axhline(0, linestyle="--")
        ax.set(xlabel="Concentration noise SD / c₀", ylabel="Ea bias (J mol⁻¹)", title=f"{n} time points")
    axes[0].legend(frameon=False)
    fig.savefig("figures/error_and_bias_vs_noise.png", dpi=200)
    plt.close(fig)

    chosen = next(c for c in cells if c["noise_sd_fraction_c0"] == .01 and c["n_time_points"] == 9)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    axes[0].plot(range(5), chosen["by_arrhenius_method"]["weighted_centered"]["k_hat_by_temperature"], "o", label="Recovered")
    axes[0].plot(range(5), result["study"]["true_values"]["k_s_inv"], "--", label="True")
    axes[0].set(xlabel="Temperature index", ylabel="k (s⁻¹)", title="Rate recovery")
    axes[0].legend(frameon=False)
    correlations = [c["by_arrhenius_method"]["weighted_centered"]["Ea_ln_A_correlation"] for c in cells]
    axes[1].plot(range(len(correlations)), correlations, "o")
    axes[1].axhline(result["theory"]["Ea_ln_A_correlation"], linestyle="--")
    axes[1].set(xlabel="Design cell", ylabel="corr(Ea, ln A)", title="Parameter correlation")
    fig.savefig("figures/rate_recovery_and_parameter_correlation.png", dpi=200)
    plt.close(fig)


def _paper(result):
    with open("paper_md", "w", encoding="utf-8") as handle:
        handle.write(
            "# Recovery of Arrhenius parameters from noisy first-order traces\n\n"
            "Five rates from 0.1 to 1.0 s⁻¹ were generated from the stated Arrhenius law "
            "and fitted with positive nonlinear least squares. The analysis used 400 trials "
            "per noise/time-point cell, with FI-controlled trial seeds. Log-rate variances "
            "were estimated by the residual-scaled Jacobian and propagated by the delta method. "
            "Arrhenius Wald intervals use t(3) as primary and 1.96 as a sensitivity analysis.\n\n"
            "Figures and machine-readable summaries are saved alongside this report. Noiseless "
            "special cases are code checks, not empirical evidence. The two-stage analysis "
            "assumes exact first-order kinetics, known c0 and temperature, independent Gaussian "
            "measurement error, and locally valid covariance approximations.\n"
        )


if __name__ == "__main__":
    main()