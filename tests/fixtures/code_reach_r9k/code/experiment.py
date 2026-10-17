import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

R_J_MOL_K = 8.314462618
A_TRUE_S_INV = 4000.0
EA_TRUE_J_MOL = R_J_MOL_K * np.log(10.0) / (1.0 / 280.0 - 1.0 / 360.0)
LNA_TRUE = float(np.log(A_TRUE_S_INV))
Z975 = float(norm.ppf(0.975))


def metric_values(cell_record: dict, name: str) -> tuple[np.ndarray, list]:
    metric = cell_record["metrics"].get(name, {})
    return np.asarray(metric.get("values", []), dtype=float), metric.get("trials", [])


def wilson(hits: int, total: int) -> list:
    if total == 0:
        return [None, None]
    p = hits / total
    d = 1.0 + Z975**2 / total
    centre = (p + Z975**2 / (2.0 * total)) / d
    radius = Z975 * np.sqrt(p * (1.0 - p) / total + Z975**2 / (4.0 * total**2)) / d
    return [float(max(0.0, centre - radius)), float(min(1.0, centre + radius))]


def bootstrap(values: np.ndarray, statistic, rng: np.random.Generator) -> list:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [None, None]
    if len(values) == 1:
        value = float(statistic(values))
        return [value, value]
    index = rng.integers(0, len(values), size=(2000, len(values)))
    draws = np.array([statistic(values[row]) for row in index], dtype=float)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize_method(cell_record: dict, method: str, rng: np.random.Generator) -> dict:
    ea, trials = metric_values(cell_record, f"{method}_Ea_hat_J_mol")
    lna, _ = metric_values(cell_record, f"{method}_lnA_hat")
    ea_lo, _ = metric_values(cell_record, f"{method}_Ea_ci_low")
    ea_hi, _ = metric_values(cell_record, f"{method}_Ea_ci_high")
    lna_lo, _ = metric_values(cell_record, f"{method}_lnA_ci_low")
    lna_hi, _ = metric_values(cell_record, f"{method}_lnA_ci_high")

    n = min(len(ea), len(lna), len(ea_lo), len(ea_hi), len(lna_lo), len(lna_hi))
    ea, lna = ea[:n], lna[:n]
    ea_hit = (ea_lo[:n] <= EA_TRUE_J_MOL) & (EA_TRUE_J_MOL <= ea_hi[:n])
    lna_hit = (lna_lo[:n] <= LNA_TRUE) & (LNA_TRUE <= lna_hi[:n])
    empirical_coverage_hits = np.concatenate([ea_hit, lna_hit])
    empirical_coverage_total = len(empirical_coverage_hits)
    corr = float(np.corrcoef(ea, lna)[0, 1]) if n > 1 else np.nan

    return {
        "Ea_bias_J_mol": {
            "estimate": float(np.mean(ea - EA_TRUE_J_MOL)) if n else None,
            "bootstrap_95_ci": bootstrap(ea - EA_TRUE_J_MOL, np.mean, rng),
        },
        "lnA_bias": {
            "estimate": float(np.mean(lna - LNA_TRUE)) if n else None,
            "bootstrap_95_ci": bootstrap(lna - LNA_TRUE, np.mean, rng),
        },
        "Ea_estimate_sd_J_mol": {
            "estimate": float(np.std(ea, ddof=1)) if n > 1 else None,
            "bootstrap_95_ci": bootstrap(ea, lambda x: np.std(x, ddof=1), rng),
        },
        "lnA_estimate_sd": {
            "estimate": float(np.std(lna, ddof=1)) if n > 1 else None,
            "bootstrap_95_ci": bootstrap(lna, lambda x: np.std(x, ddof=1), rng),
        },
        "empirical_coverage": {
            "estimate": (
                float(np.mean(empirical_coverage_hits))
                if empirical_coverage_total
                else None
            ),
            "wilson_95_ci": wilson(
                int(np.sum(empirical_coverage_hits)), empirical_coverage_total
            ),
        },
        "Ea_95_coverage": {
            "estimate": float(np.mean(ea_hit)) if n else None,
            "wilson_95_ci": wilson(int(np.sum(ea_hit)), n),
        },
        "Ea_95_coverage_count": int(np.sum(ea_hit)),
        "Ea_95_coverage_total": int(n),
        "lnA_95_coverage": {
            "estimate": float(np.mean(lna_hit)) if n else None,
            "wilson_95_ci": wilson(int(np.sum(lna_hit)), n),
        },
        "lnA_95_coverage_count": int(np.sum(lna_hit)),
        "lnA_95_coverage_total": int(n),
        "abs_Ea_lnA_correlation": {
            "estimate": abs(corr) if np.isfinite(corr) else None,
            "bootstrap_95_ci": bootstrap(
                np.arange(n),
                lambda indices: abs(
                    np.corrcoef(
                        ea[indices.astype(int)], lna[indices.astype(int)]
                    )[0, 1]
                ),
                rng,
            ),
        },
        "Ea_lnA_correlation": {
            "estimate": corr if np.isfinite(corr) else None,
            "bootstrap_95_ci": bootstrap(
                np.arange(n),
                lambda indices: float(
                    np.corrcoef(
                        ea[indices.astype(int)], lna[indices.astype(int)]
                    )[0, 1]
                ),
                rng,
            ),
        },
        "Ea_95_coverage_values": ea_hit.astype(int).tolist(),
        "Ea_95_coverage_pair_id": trials[:n],
        "lnA_95_coverage_values": lna_hit.astype(int).tolist(),
        "lnA_95_coverage_pair_id": trials[:n],
    }


def plot_calibration(cells: list, parameter: str, filename: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metric_names = (
        (f"{parameter}_bias_J_mol" if parameter == "Ea" else "lnA_bias", "Bias"),
        (f"{parameter}_estimate_sd_J_mol" if parameter == "Ea" else "lnA_estimate_sd", "Estimate SD"),
        (f"{parameter}_95_coverage", "95% interval coverage"),
    )
    for ax, (metric, label) in zip(axes, metric_names):
        for n_time in sorted({c["n_time_points"] for c in cells}):
            rows = [c for c in cells if c["n_time_points"] == n_time]
            rows.sort(key=lambda c: c["concentration_noise_sd_fraction_of_c0"])
            x = [r["concentration_noise_sd_fraction_of_c0"] for r in rows]
            y = [r["by_Arrhenius_regression_method"]["weighted"][metric]["estimate"] for r in rows]
            ax.plot(x, y, marker="o", label=f"{n_time} points")
        ax.set_xscale("log")
        ax.set_xlabel("Concentration noise SD / c₀")
        ax.set_ylabel(label)
        ax.set_title(f"{parameter}: {label}")
        ax.axhline(0.95 if "coverage" in metric else 0.0, linestyle="--")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_difference(cells: list) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, name, label in zip(
        axes,
        ("Ea_J_mol", "lnA", "coverage_difference"),
        ("Weighted − OLS Ea (J mol⁻¹)", "Weighted − OLS ln A", "Weighted − OLS Ea coverage"),
    ):
        for n_time in sorted({c["n_time_points"] for c in cells}):
            rows = [c for c in cells if c["n_time_points"] == n_time]
            rows.sort(key=lambda c: c["concentration_noise_sd_fraction_of_c0"])
            x = [r["concentration_noise_sd_fraction_of_c0"] for r in rows]
            if name == "coverage_difference":
                y = [
                    r["by_Arrhenius_regression_method"]["weighted"]["Ea_95_coverage"]["estimate"]
                    - r["by_Arrhenius_regression_method"]["ols"]["Ea_95_coverage"]["estimate"]
                    for r in rows
                ]
            else:
                y = [r["paired_weighted_minus_ols"][name]["estimate"] for r in rows]
            ax.plot(x, y, marker="o", label=f"{n_time} points")
        ax.axhline(0.0, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("Concentration noise SD / c₀")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig("figures/weighted_minus_unweighted.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    with open(os.environ["FI_TRIALS"], encoding="utf-8") as handle:
        trials = json.load(handle)

    rng = np.random.default_rng(20261015)
    cells = []
    csv_rows = []

    for record in trials["cells"]:
        cell = record["cell"]
        weighted = summarize_method(record, "weighted", rng)
        ols = summarize_method(record, "ols", rng)
        weighted_ea, _ = metric_values(record, "weighted_Ea_hat_J_mol")
        ols_ea, _ = metric_values(record, "ols_Ea_hat_J_mol")
        weighted_lna, _ = metric_values(record, "weighted_lnA_hat")
        ols_lna, _ = metric_values(record, "ols_lnA_hat")
        n = min(len(weighted_ea), len(ols_ea), len(weighted_lna), len(ols_lna))

        ea_delta = weighted_ea[:n] - ols_ea[:n]
        lna_delta = weighted_lna[:n] - ols_lna[:n]
        row = {
            "concentration_noise_sd_fraction_of_c0": float(cell["concentration_noise_sd_fraction_of_c0"]),
            "n_time_points": int(cell["n_time_points"]),
            "planned_trials": int(record["planned"]),
            "successful_trials": int(record["ok"]),
            "solver_failure_fraction": {
                "estimate": float(record["failed"] / record["planned"]),
                "wilson_95_ci": wilson(int(record["failed"]), int(record["planned"])),
            },
            "solver_failure_fraction_count": int(record["failed"]),
            "solver_failure_fraction_total": int(record["planned"]),
            "failed_trial_indices": [],
            "by_Arrhenius_regression_method": {"weighted": weighted, "ols": ols},
            "paired_weighted_minus_ols": {
                "Ea_J_mol": {
                    "estimate": float(np.mean(ea_delta)) if n else None,
                    "bootstrap_95_ci": bootstrap(ea_delta, np.mean, rng),
                },
                "lnA": {
                    "estimate": float(np.mean(lna_delta)) if n else None,
                    "bootstrap_95_ci": bootstrap(lna_delta, np.mean, rng),
                },
            },
        }
        cells.append(row)
        csv_rows.append(
            {
                "noise": row["concentration_noise_sd_fraction_of_c0"],
                "n_time_points": row["n_time_points"],
                "weighted_Ea_bias_J_mol": weighted["Ea_bias_J_mol"]["estimate"],
                "ols_Ea_bias_J_mol": ols["Ea_bias_J_mol"]["estimate"],
                "weighted_Ea_coverage": weighted["Ea_95_coverage"]["estimate"],
                "ols_Ea_coverage": ols["Ea_95_coverage"]["estimate"],
            }
        )

    with open("data/results/cell_summaries.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    plot_calibration(cells, "Ea", "figures/ea_calibration.png")
    plot_calibration(cells, "lnA", "figures/lna_calibration.png")
    plot_difference(cells)

    result = {
        "truth": {
            "Ea_true_J_mol": {"value": float(EA_TRUE_J_MOL), "numerical_error": 0.0},
            "lnA_true": {"value": LNA_TRUE, "numerical_error": 0.0},
            "A_true_s_inv": {"value": A_TRUE_S_INV, "numerical_error": 0.0},
        },
        "run_metadata": {
            "runs_per_setting": 400,
            "temperatures_K": [280.0, 300.0, 320.0, 340.0, 360.0],
            "noise_levels": [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08],
            "n_time_points_values": [5, 8, 12, 20, 32],
        },
        "cells": cells,
        "figures": [
            "figures/ea_calibration.png",
            "figures/lna_calibration.png",
            "figures/weighted_minus_unweighted.png",
        ],
    }
    Path("data/results/result_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    result = main()
    print("RESULT_JSON: " + json.dumps(result, sort_keys=True, allow_nan=False))
