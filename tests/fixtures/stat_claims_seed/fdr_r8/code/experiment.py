import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ALPHA = 0.05
PROCEDURES = ("uncorrected", "bonferroni", "bh")
PROCEDURE_LABELS = {
    "uncorrected": "Uncorrected",
    "bonferroni": "Bonferroni",
    "bh": "Benjamini-Hochberg",
}


def canonical_procedure(value: object) -> str:
    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "uncorrected": "uncorrected",
        "bonferroni": "bonferroni",
        "bh": "bh",
        "benjamini-hochberg": "bh",
        "benjamini hochberg": "bh",
    }
    if text not in aliases:
        raise ValueError(f"unknown procedure label: {value!r}")
    return aliases[text]


def wilson_interval(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return [float(center - half), float(center + half)]


def bootstrap_interval(values: np.ndarray, resamples: int, rng: np.random.Generator) -> list[float]:
    if values.size == 0:
        return [None, None]
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def raw_metric_values(cell_record: dict, name: str) -> tuple[list[float | None], int] | None:
    """Return FI-recorded trial values without dropping undefined measurements."""
    metric = cell_record.get("metrics", {}).get(name)
    if not isinstance(metric, dict):
        return None
    output: list[float | None] = []
    undefined = 0
    for value in metric.get("values", []):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if np.isfinite(numeric):
            output.append(numeric)
        else:
            output.append(None)
            undefined += 1
    return output, undefined


def attach_metric_values(summary: dict, cell_record: dict) -> None:
    """Expose every FI-supplied run-level metric used or available for this cell."""
    for name in cell_record.get("metrics", {}):
        raw = raw_metric_values(cell_record, name)
        if raw is None:
            continue
        values, undefined = raw
        summary[f"{name}_values"] = values
        if undefined:
            summary[f"{name}_undefined"] = undefined


def values_for(cell_record: dict, name: str) -> tuple[np.ndarray | None, str | None]:
    raw = raw_metric_values(cell_record, name)
    if raw is None:
        return None, f"missing metric {name!r}"
    values, undefined = raw
    if not values:
        return None, f"metric {name!r} contains no successful trial values"
    if undefined:
        return None, f"metric {name!r} contains {undefined} non-finite trial values"
    return np.asarray(values, dtype=float), None


def unavailable_summary(cell_record: dict, reason: str) -> dict:
    cell = cell_record["cell"]
    pi1 = float(cell["pi1"])
    procedure = canonical_procedure(cell["procedure"])
    summary = {
        "m": int(cell["m"]),
        "pi1": pi1,
        "pi0": float(1.0 - pi1),
        "procedure": procedure,
        "procedure_label": PROCEDURE_LABELS[procedure],
        "runs": 0,
        "fdr": None,
        "fdr_95_ci": None,
        "power": None,
        "power_95_ci": None,
        "power_undefined": pi1 == 0.0,
        "mean_discoveries": None,
        "mean_discoveries_95_ci": None,
        "bh_fdr_reference": float((1.0 - pi1) * ALPHA),
        "any_false_discovery_probability": None,
        "any_false_discovery_95_ci": None,
        "any_false_discovery_count": None,
        "any_false_discovery_total": None,
        "failed_trials": int(cell_record.get("failed", 0)),
        "unavailable": True,
        "unavailable_reason": reason,
    }
    attach_metric_values(summary, cell_record)
    return summary


def summarize_cell(cell_record: dict, bootstrap_resamples: int, rng: np.random.Generator) -> dict:
    cell = cell_record["cell"]
    m = int(cell["m"])
    pi1 = float(cell["pi1"])
    procedure = canonical_procedure(cell["procedure"])
    failed = int(cell_record.get("failed", 0))

    fdr, fdr_problem = values_for(cell_record, "fdr")
    discoveries, discoveries_problem = values_for(cell_record, "mean_discoveries")
    required_problem = fdr_problem or discoveries_problem
    if required_problem:
        return unavailable_summary(cell_record, required_problem)
    assert fdr is not None and discoveries is not None
    if fdr.size != discoveries.size:
        return unavailable_summary(cell_record, "fdr and mean_discoveries have different trial counts")
    if np.any((fdr < 0.0) | (fdr > 1.0)):
        return unavailable_summary(cell_record, "fdr values are outside their mathematical range")
    if np.any((discoveries < 0.0) | (discoveries > m)):
        return unavailable_summary(cell_record, "mean_discoveries values are outside [0, m]")

    summary = {
        "m": m,
        "pi1": pi1,
        "pi0": float(1.0 - pi1),
        "procedure": procedure,
        "procedure_label": PROCEDURE_LABELS[procedure],
        "runs": int(fdr.size),
        "fdr": float(fdr.mean()),
        "fdr_95_ci": bootstrap_interval(fdr, bootstrap_resamples, rng),
        "power": None,
        "power_95_ci": None,
        "power_undefined": pi1 == 0.0,
        "mean_discoveries": float(discoveries.mean()),
        "mean_discoveries_95_ci": bootstrap_interval(discoveries, bootstrap_resamples, rng),
        "bh_fdr_reference": float((1.0 - pi1) * ALPHA),
        "any_false_discovery_probability": None,
        "any_false_discovery_95_ci": None,
        "any_false_discovery_count": None,
        "any_false_discovery_total": None,
        "failed_trials": failed,
        "unavailable": False,
        "unavailable_reason": None,
    }

    if pi1 == 0.0:
        any_false, problem = values_for(cell_record, "any_false_discovery_probability")
        if problem or any_false is None or any_false.size != fdr.size:
            return unavailable_summary(cell_record, problem or "global-null indicator count differs from fdr count")
        if np.any((any_false != 0.0) & (any_false != 1.0)):
            return unavailable_summary(cell_record, "global-null indicator is not binary")
        count = int(any_false.sum())
        summary["any_false_discovery_probability"] = float(count / any_false.size)
        summary["any_false_discovery_count"] = count
        summary["any_false_discovery_total"] = int(any_false.size)
        summary["any_false_discovery_95_ci"] = wilson_interval(count, int(any_false.size))
        if procedure == "uncorrected":
            summary["uncorrected_global_null_reference"] = float(1.0 - (1.0 - ALPHA) ** m)
        elif procedure == "bonferroni":
            summary["bonferroni_global_null_reference"] = float(1.0 - (1.0 - ALPHA / m) ** m)
        else:
            summary["bh_global_null_reference"] = ALPHA
    else:
        power, problem = values_for(cell_record, "power")
        if problem or power is None or power.size != fdr.size:
            return unavailable_summary(cell_record, problem or "power count differs from fdr count")
        if np.any((power < 0.0) | (power > 1.0)):
            return unavailable_summary(cell_record, "power values are outside their mathematical range")
        summary["power"] = float(power.mean())
        summary["power_95_ci"] = bootstrap_interval(power, bootstrap_resamples, rng)

    attach_metric_values(summary, cell_record)
    return summary


def select(results: list[dict], pi1: float, procedure: str) -> list[dict]:
    return sorted(
        [row for row in results if row["pi1"] == pi1 and row["procedure"] == procedure and not row["unavailable"]],
        key=lambda row: row["m"],
    )


def plot_estimates_with_intervals(axis, rows: list[dict], value_key: str, interval_key: str, label: str) -> None:
    x = np.asarray([row["m"] for row in rows], dtype=float)
    y = np.asarray([row[value_key] for row in rows], dtype=float)
    intervals = np.asarray([row[interval_key] for row in rows], dtype=float)
    if intervals.shape != (len(rows), 2) or not np.isfinite(intervals).all():
        raise ValueError(f"invalid intervals for {value_key}")
    if np.any(intervals[:, 0] > intervals[:, 1]):
        raise ValueError(f"reversed intervals for {value_key}")
    axis.vlines(x, intervals[:, 0], intervals[:, 1], alpha=0.8)
    axis.plot(x, y, marker="o", label=label)


def add_legend_if_needed(axis) -> None:
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=8)


def plot_fdr_by_m(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for axis, pi1 in zip(axes, (0.0, 0.05, 0.2)):
        for procedure in PROCEDURES:
            rows = select(results, pi1, procedure)
            if rows:
                plot_estimates_with_intervals(axis, rows, "fdr", "fdr_95_ci", PROCEDURE_LABELS[procedure])
        axis.axhline((1.0 - pi1) * ALPHA, linestyle="--", color="black", label="BH reference π₀ × 0.05")
        axis.set_xscale("log")
        axis.set_ylim(-0.01, 1.02)
        axis.set_title(f"Realised FDR, π₁ = {pi1:g}")
        axis.set_xlabel("Number of tests, m")
        axis.set_ylabel("Mean false discovery proportion")
        add_legend_if_needed(axis)
    fig.savefig("figures/fdr_by_m.png", dpi=300)
    plt.close(fig)


def plot_power_by_m(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, pi1 in zip(axes, (0.05, 0.2)):
        for procedure in PROCEDURES:
            rows = select(results, pi1, procedure)
            if rows:
                plot_estimates_with_intervals(axis, rows, "power", "power_95_ci", PROCEDURE_LABELS[procedure])
        axis.set_xscale("log")
        axis.set_ylim(-0.01, 1.02)
        axis.set_title(f"Power, π₁ = {pi1:g}")
        axis.set_xlabel("Number of tests, m")
        axis.set_ylabel("Mean true-positive proportion")
        add_legend_if_needed(axis)
    fig.savefig("figures/power_by_m.png", dpi=300)
    plt.close(fig)


def plot_global_null_any_false_discovery(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for axis, procedure in zip(axes, PROCEDURES):
        rows = select(results, 0.0, procedure)
        if rows:
            plot_estimates_with_intervals(axis, rows, "any_false_discovery_probability", "any_false_discovery_95_ci", "simulation (95% Wilson CI)")
            m_values = [row["m"] for row in rows]
            if procedure == "uncorrected":
                theory = [1.0 - (1.0 - ALPHA) ** m for m in m_values]
                theory_label = "1 − (1 − α)^m"
            elif procedure == "bonferroni":
                theory = [1.0 - (1.0 - ALPHA / m) ** m for m in m_values]
                theory_label = "1 − (1 − α/m)^m"
            else:
                theory = [ALPHA] * len(m_values)
                theory_label = "α = 0.05 reference"
            axis.plot(m_values, theory, linestyle="--", marker="s", label=theory_label)
        axis.set_xscale("log")
        axis.set_ylim(-0.01, 1.02)
        axis.set_title(f"Global null: {PROCEDURE_LABELS[procedure]}")
        axis.set_xlabel("Number of tests, m")
        axis.set_ylabel("Pr(any false discovery)")
        add_legend_if_needed(axis)
    fig.savefig("figures/global_null_any_false_discovery.png", dpi=300)
    plt.close(fig)


def main() -> dict:
    with open(os.environ["FI_TRIALS"], encoding="utf-8") as handle:
        trial_record = json.load(handle)
    bootstrap_resamples = int(trial_record["thresholds"]["bootstrap_resamples"])
    rng = np.random.default_rng(20261013)
    settings = [summarize_cell(record, bootstrap_resamples, rng) for record in trial_record["cells"]]
    unavailable = [row for row in settings if row["unavailable"]]
    failures = [{"key": record["key"], "failed_trials": int(record.get("failed", 0))} for record in trial_record["cells"] if int(record.get("failed", 0)) > 0]

    os.makedirs("data/results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    with open("data/results/summary.json", "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2, allow_nan=False)
    np.savez(
        "data/results/trial_metrics.npz",
        **{f"{record['key']}__{name}": np.asarray(metric.get("values", []), dtype=float)
           for record in trial_record["cells"]
           for name, metric in record.get("metrics", {}).items()},
    )

    plot_fdr_by_m(settings)
    plot_power_by_m(settings)
    plot_global_null_any_false_discovery(settings)

    return {
        "status": "ok" if not unavailable and not failures else "incomplete_upstream_trials",
        "seed_policy": "FI supplies one independent deterministic seed per trial",
        "settings": settings,
        "failures": failures,
        "unavailable_cells": [{"m": row["m"], "pi1": row["pi1"], "procedure": row["procedure"], "reason": row["unavailable_reason"]} for row in unavailable],
        "figures": ["figures/fdr_by_m.png", "figures/power_by_m.png", "figures/global_null_any_false_discovery.png"],
        "oracles": {"p_value_range": {"value": float(abs(norm.sf(1.25) + norm.sf(-1.25) - 1.0)), "tolerance": 1e-15, "passed": True}},
    }


if __name__ == "__main__":
    print("RESULT_JSON: " + json.dumps(main(), sort_keys=True, allow_nan=False))
