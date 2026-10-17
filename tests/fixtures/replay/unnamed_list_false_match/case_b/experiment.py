import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t

os.makedirs("figures", exist_ok=True)

ALPHA = 0.05
EFFECT_SHIFT = 2.5
M_VALUES = (20, 200, 2000)
PI1_VALUES = (0.0, 0.05, 0.2)
METHODS = ("uncorrected", "bonferroni", "bh")
CI_LEVEL = 0.95


def t_interval(values: np.ndarray, confidence: float) -> list[float]:
    values = np.asarray(values, dtype=float)
    n = values.size
    mean = float(np.mean(values))
    if n < 2:
        return [mean, mean]
    half_width = float(
        t.ppf((1.0 + confidence) / 2.0, n - 1)
        * np.std(values, ddof=1)
        / math.sqrt(n)
    )
    return [mean - half_width, mean + half_width]


def wilson_interval(count: int, total: int, confidence: float) -> list[float]:
    if total <= 0:
        return [None, None]
    z = float(norm.ppf((1.0 + confidence) / 2.0))
    p = count / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        p * (1.0 - p) / total + z**2 / (4.0 * total**2)
    ) / denominator
    return [center - half_width, center + half_width]


def marginal_power_reference(method: str, m: int) -> float | None:
    if method == "uncorrected":
        threshold = ALPHA
    elif method == "bonferroni":
        threshold = ALPHA / m
    else:
        return None
    return float(norm.sf(norm.isf(threshold) - EFFECT_SHIFT))


def complete_null_reference(method: str, m: int) -> float:
    if method == "uncorrected":
        return float(1.0 - (1.0 - ALPHA) ** m)
    if method == "bonferroni":
        return float(1.0 - (1.0 - ALPHA / m) ** m)
    if method == "bh":
        return ALPHA
    raise ValueError(f"Unknown method: {method}")


def _cell_index(trials: dict) -> dict:
    indexed = {}
    for record in trials["cells"]:
        cell = record["cell"]
        indexed[(int(cell["m"]), float(cell["pi1"]), str(cell["method"]))] = record
    return indexed


def _metric(record: dict, name: str) -> tuple[np.ndarray, list]:
    metric = record.get("metrics", {}).get(name, {})
    return (
        np.asarray(metric.get("values", []), dtype=float),
        list(metric.get("trials", [])),
    )


def _setting_summary(record: dict, method: str, m: int, pi1: float) -> dict:
    fdps, fdp_trials = _metric(record, "realized_fdr")
    powers, power_trials = _metric(record, "power")
    summary = {
        "m": m,
        "pi1": pi1,
        "realized_fdr": float(np.mean(fdps)) if fdps.size else None,
        "realized_fdr_ci95": t_interval(fdps, CI_LEVEL) if fdps.size else [None, None],
        "realized_fdr_count": int(fdps.size),
        "realized_fdr_total": int(record.get("planned", fdps.size)),
        "power": float(np.mean(powers)) if powers.size else None,
        "power_ci95": t_interval(powers, CI_LEVEL) if powers.size else [None, None],
        "power_count": int(powers.size),
        "power_total": int(record.get("planned", powers.size)),
        "power_reference": marginal_power_reference(method, m),
        "failed_trials": int(record.get("failed", 0)),
    }
    if method == "bh":
        summary["bh_fdr_reference"] = (1.0 - pi1) * ALPHA
    return summary


def _paired_difference(bh_record: dict, bonf_record: dict) -> dict:
    bh_values, bh_ids = _metric(bh_record, "power")
    bon_values, bon_ids = _metric(bonf_record, "power")
    bh_map = dict(zip(bh_ids, bh_values))
    bon_map = dict(zip(bon_ids, bon_values))
    paired_ids = sorted(set(bh_map).intersection(bon_map))
    differences = np.asarray(
        [bh_map[trial] - bon_map[trial] for trial in paired_ids], dtype=float
    )
    return {
        "estimate": float(np.mean(differences)) if differences.size else None,
        "ci95": t_interval(differences, CI_LEVEL) if differences.size else [None, None],
        "count": int(differences.size),
        "total": int(max(bh_record.get("planned", 0), bonf_record.get("planned", 0))),
    }


def _complete_null_summary(record: dict, method: str, m: int) -> dict:
    values, _ = _metric(record, "any_false_discovery")
    count = int(np.sum(values))
    total = int(values.size)
    return {
        "m": m,
        "pi1": 0.0,
        "any_false_discovery_probability": float(count / total) if total else None,
        "any_false_discovery_probability_count": count,
        "any_false_discovery_probability_total": total,
        "any_false_discovery_probability_ci95": wilson_interval(count, total, CI_LEVEL),
        "theory_reference": complete_null_reference(method, m),
        "failed_trials": int(record.get("failed", 0)),
    }


def plot_fdr_by_m(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(M_VALUES))
    for ax, pi1 in zip(axes, (0.05, 0.2)):
        for method in METHODS:
            rows = [
                row for row in results["by_method"][method]["nonnull_settings"]
                if row["pi1"] == pi1
            ]
            y = np.array([row["realized_fdr"] for row in rows])
            ci = np.array([row["realized_fdr_ci95"] for row in rows])
            ax.errorbar(
                x, y, yerr=np.vstack((y - ci[:, 0], ci[:, 1] - y)),
                marker="o", capsize=3, label=method
            )
        reference = (1.0 - pi1) * ALPHA
        ax.axhline(reference, linestyle="--", label="BH reference π₀α")
        ax.set_xticks(x, [str(m) for m in M_VALUES])
        ax.set_xlabel("Number of hypotheses, m")
        ax.set_ylabel("Mean realized FDR (95% t CI)")
        ax.set_title(f"π₁ = {pi1}; shift = {EFFECT_SHIFT}")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("False discovery rate by family size")
    fig.savefig("figures/fdr_by_m.png", dpi=300)
    plt.close(fig)


def plot_power_by_m(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(M_VALUES))
    for ax, pi1 in zip(axes, (0.05, 0.2)):
        for method in METHODS:
            rows = [
                row for row in results["by_method"][method]["nonnull_settings"]
                if row["pi1"] == pi1
            ]
            y = np.array([row["power"] for row in rows])
            ci = np.array([row["power_ci95"] for row in rows])
            ax.errorbar(
                x, y, yerr=np.vstack((y - ci[:, 0], ci[:, 1] - y)),
                marker="o", capsize=3, label=method
            )
            if method in ("uncorrected", "bonferroni"):
                refs = [row["power_reference"] for row in rows]
                ax.plot(x, refs, linestyle="--", marker="x",
                        label=f"{method} exact reference")
        ax.set_xticks(x, [str(m) for m in M_VALUES])
        ax.set_xlabel("Number of hypotheses, m")
        ax.set_ylabel("Power (95% t CI)")
        ax.set_title(f"π₁ = {pi1}; shift = {EFFECT_SHIFT}")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Power by family size")
    fig.savefig("figures/power_by_m.png", dpi=300)
    plt.close(fig)


def plot_complete_null(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = np.arange(len(M_VALUES))
    for method in METHODS:
        rows = results["by_method"][method]["complete_null_settings"]
        y = np.array([row["any_false_discovery_probability"] for row in rows])
        ci = np.array([row["any_false_discovery_probability_ci95"] for row in rows])
        refs = np.array([row["theory_reference"] for row in rows])
        ax.errorbar(
            x, y, yerr=np.vstack((y - ci[:, 0], ci[:, 1] - y)),
            marker="o", capsize=3, label=method
        )
        ax.plot(x, refs, linestyle="--", marker="x", label=f"{method} reference")
    ax.set_xticks(x, [str(m) for m in M_VALUES])
    ax.set_xlabel("Number of hypotheses, m")
    ax.set_ylabel("P(any false discovery), 95% Wilson CI")
    ax.set_title(f"Complete null; effect shift = {EFFECT_SHIFT}")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig("figures/complete_null.png", dpi=300)
    plt.close(fig)


def main() -> dict:
    with open(os.environ["FI_TRIALS"], encoding="utf-8") as handle:
        trials = json.load(handle)

    os.makedirs("data/results", exist_ok=True)
    index = _cell_index(trials)
    results = {
        "metadata": {
            "alpha": ALPHA,
            "effect_shift": EFFECT_SHIFT,
            "runs_per_setting": int(trials["cells"][0]["planned"]),
            "base_seed": "assigned by FrontierInsight",
            "seed_mapping": "one FI-assigned seed per trial",
            "ci_level": CI_LEVEL,
            "display_rounding_max_error": 0.0000005,
        },
        "oracles": {"executed_by": "FrontierInsight before analysis"},
        "failures": {
            "count": int(sum(record.get("failed", 0) for record in trials["cells"])),
            "items": [
                {"key": record["key"], "failed": int(record.get("failed", 0))}
                for record in trials["cells"] if record.get("failed", 0)
            ],
        },
        "by_method": {method: {"nonnull_settings": [], "complete_null_settings": []}
                      for method in METHODS},
    }

    csv_rows = []
    arrays = {}
    for method in METHODS:
        for pi1 in (0.05, 0.2):
            for m in M_VALUES:
                record = index[(m, pi1, method)]
                summary = _setting_summary(record, method, m, pi1)
                if method == "bh":
                    summary["paired_power_vs_bonferroni"] = _paired_difference(
                        record, index[(m, pi1, "bonferroni")]
                    )
                else:
                    summary["paired_power_vs_bonferroni"] = None
                results["by_method"][method]["nonnull_settings"].append(summary)
                csv_rows.append({"method": method, "setting": "nonnull", **summary})
                values, _ = _metric(record, "realized_fdr")
                arrays[f"fdr_{method}_m{m}_pi{pi1}"] = values
                values, _ = _metric(record, "power")
                arrays[f"power_{method}_m{m}_pi{pi1}"] = values

        for m in M_VALUES:
            record = index[(m, 0.0, method)]
            summary = _complete_null_summary(record, method, m)
            results["by_method"][method]["complete_null_settings"].append(summary)
            csv_rows.append({"method": method, "setting": "complete_null", **summary})
            values, _ = _metric(record, "any_false_discovery")
            arrays[f"any_false_{method}_m{m}"] = values

    with open("data/results/setting_summaries.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in csv_rows for key in row}))
        writer.writeheader()
        writer.writerows(csv_rows)
    np.savez_compressed("data/results/family_metrics.npz", **arrays)
    with open("data/results/results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)

    plot_fdr_by_m(results)
    plot_power_by_m(results)
    plot_complete_null(results)
    return results


if __name__ == "__main__":
    result = main()
    print("RESULT_JSON: " + json.dumps(result, sort_keys=True, allow_nan=False))