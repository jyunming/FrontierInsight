"""Each trial fixes n1=m*pi1 nonnull z statistics from N(2.5,1), m-n1 null statistics from N(0,1), converts them to one-sided normal-tail p-values, and applies the three specified rejection rules."""

from __future__ import annotations
import numpy as np
import math
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm


# E1
def draw_z(effect_size: float, is_signal: bool, rng: np.random.Generator) -> float:
    return float(effect_size * is_signal + rng.normal(0.0, 1.0))


# E2
def normal_tail(z: float) -> float:
    # Compute the one-sided standard-normal upper-tail p-value for one z statistic.
    return norm.sf(z)


# E3
def bonferroni_threshold(alpha: float, m: int) -> float:
    # Compute the Bonferroni per-hypothesis rejection threshold.
    return alpha / m


# E4
def bh_critical_value(alpha: float, k: int, m: int) -> float:
    return alpha * k / m


# E5
def bh_fdr_reference(pi1: float, alpha: float) -> float:
    return (1.0 - pi1) * alpha


# S1
def assign_truth_labels(m: int, pi1: float, rng: np.random.Generator) -> np.ndarray:
    n1 = int(m * pi1)
    labels = np.zeros(m, dtype=bool)
    labels[:n1] = True
    return rng.permutation(labels)


# S2
def generate_z_statistics(signal_labels: np.ndarray, effect_size: float, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(
        [draw_z(effect_size, bool(is_signal), rng) for is_signal in signal_labels],
        dtype=float,
    )


# S3
def compute_pvalues(z: np.ndarray) -> np.ndarray:
    # Convert z statistics to one-sided normal-tail p-values.
    return np.asarray([normal_tail(float(value)) for value in z], dtype=float)


# S4
def apply_procedures(pvalues: np.ndarray, alpha: float, m: int) -> dict[str, np.ndarray]:
    # Return paired Boolean rejection vectors for uncorrected, Bonferroni, and BH testing.
    uncorrected = pvalues <= alpha
    bonferroni = pvalues <= bonferroni_threshold(alpha, m)

    order = np.argsort(pvalues)
    ordered_pvalues = pvalues[order]
    ranks = np.arange(1, m + 1)
    eligible = ordered_pvalues <= np.array(
        [bh_critical_value(alpha, int(k), m) for k in ranks]
    )

    bh = np.zeros(m, dtype=bool)
    if np.any(eligible):
        k = np.max(ranks[eligible])
        bh[order[:k]] = True

    return {
        "uncorrected": uncorrected,
        "bonferroni": bonferroni,
        "bh": bh,
    }


# S5
def score_outcomes(rejections: np.ndarray, signal_labels: np.ndarray) -> dict[str, float | int | None]:
    rejections = np.asarray(rejections, dtype=bool)
    signal_labels = np.asarray(signal_labels, dtype=bool)

    true_positives = int(np.sum(rejections & signal_labels))
    false_discoveries = int(np.sum(rejections & ~signal_labels))
    total_rejections = int(np.sum(rejections))
    n1 = int(np.sum(signal_labels))

    return {
        "V": false_discoveries,
        "TP": true_positives,
        "R": total_rejections,
        "FDP": false_discoveries / total_rejections if total_rejections > 0 else 0.0,
        "power": true_positives / n1 if n1 > 0 else None,
        "any_false_discovery": int(false_discoveries > 0),
    }


# S6
def aggregate_uncertainty_and_plot(run_metrics: dict, alpha: float, pi1: float) -> dict:
    rng = np.random.default_rng(20261014)
    summaries = {}
    procedures = ("uncorrected", "bonferroni", "bh")

    for m, metrics_by_procedure in run_metrics.items():
        m_key = str(m)
        summaries[m_key] = {}

        for procedure in procedures:
            metrics = metrics_by_procedure.get(procedure, {})
            procedure_summary = {}

            for metric_name in ("fdr", "power", "any_false_discovery"):
                if metric_name in metrics:
                    mean, lower, upper = _bootstrap_interval(
                        np.asarray(metrics[metric_name], dtype=float), rng
                    )
                    procedure_summary[metric_name] = {
                        "mean": mean,
                        "ci_95": [lower, upper],
                    }

            summaries[m_key][procedure] = procedure_summary

    bh_reference = bh_fdr_reference(pi1, alpha)
    result = {
        "pi1": float(pi1),
        "alpha": float(alpha),
        "bh_fdr_reference": float(bh_reference),
        "summaries": summaries,
    }

    os.makedirs("figures", exist_ok=True)
    m_values = list(summaries)
    x = np.arange(len(m_values))
    width = 0.24

    if pi1 > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

        for index, procedure in enumerate(procedures):
            fdr_means = []
            fdr_errors = []
            power_means = []
            power_errors = []

            for m in m_values:
                fdr = summaries[m][procedure].get("fdr", {})
                power = summaries[m][procedure].get("power", {})
                fdr_means.append(fdr.get("mean", np.nan))
                fdr_errors.append([
                    fdr.get("mean", np.nan) - fdr.get("ci_95", [np.nan, np.nan])[0],
                    fdr.get("ci_95", [np.nan, np.nan])[1] - fdr.get("mean", np.nan),
                ])
                power_means.append(power.get("mean", np.nan))
                power_errors.append([
                    power.get("mean", np.nan) - power.get("ci_95", [np.nan, np.nan])[0],
                    power.get("ci_95", [np.nan, np.nan])[1] - power.get("mean", np.nan),
                ])

            offset = (index - 1) * width
            axes[0].errorbar(
                x + offset, fdr_means, yerr=np.asarray(fdr_errors).T,
                fmt="o", capsize=3, label=procedure
            )
            axes[1].errorbar(
                x + offset, power_means, yerr=np.asarray(power_errors).T,
                fmt="o", capsize=3, label=procedure
            )

        axes[0].axhline(bh_reference, color="black", linestyle="--",
                        label=r"BH reference $(1-\pi_1)\alpha$")
        axes[0].set_ylabel("Realised false discovery rate")
        axes[1].set_ylabel("Power")
        for axis in axes:
            axis.set_xticks(x, m_values)
            axis.set_xlabel("Number of tests (m)")
            axis.set_ylim(bottom=0)
            axis.legend()
        fig.tight_layout()
        fig.savefig("figures/fdr_and_power.png", dpi=300)
        plt.close(fig)

    else:
        fig, ax = plt.subplots(figsize=(6, 4))

        for index, procedure in enumerate(procedures):
            means = []
            errors = []
            for m in m_values:
                metric = summaries[m][procedure].get("any_false_discovery", {})
                means.append(metric.get("mean", np.nan))
                errors.append([
                    metric.get("mean", np.nan) - metric.get("ci_95", [np.nan, np.nan])[0],
                    metric.get("ci_95", [np.nan, np.nan])[1] - metric.get("mean", np.nan),
                ])

            ax.errorbar(
                x + (index - 1) * width, means, yerr=np.asarray(errors).T,
                fmt="o", capsize=3, label=procedure
            )

        ax.set_xticks(x, m_values)
        ax.set_xlabel("Number of tests (m)")
        ax.set_ylabel("Chance of any false discovery")
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/null_any_false_discovery.png", dpi=300)
        plt.close(fig)

    return result


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size == 0:
        return mean, float("nan"), float("nan")
    samples = rng.choice(values, size=(5000, values.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return mean, float(lower), float(upper)
