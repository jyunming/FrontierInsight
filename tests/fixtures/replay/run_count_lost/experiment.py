import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ALPHA = 0.05
EFFECT_SIZE = 2.5
N_TRIALS = 1000
METHODS = ("uncorrected", "Bonferroni", "Benjamini-Hochberg")

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)


def _prefix(method):
    return method.replace("-", "_").replace(" ", "_")


def _round(value):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 4)


def _metric_values(metric):
    if isinstance(metric, dict):
        metric = metric.get("values", metric.get("value", []))
    return np.asarray(metric, dtype=float)


def _values(entry, metric):
    return _metric_values(entry["metrics"][metric])


def _bootstrap_ci(values, rng, reps=2000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [None, None]
    means = np.empty(reps)
    for start in range(0, reps, 200):
        stop = min(start + 200, reps)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [_round(np.quantile(means, 0.025)), _round(np.quantile(means, 0.975))]


def _wilson_interval(count, total):
    interval = stats.binomtest(count, total).proportion_ci(confidence_level=0.95, method="wilson")
    return [_round(interval.low), _round(interval.high)]


def _power_reference(method, m):
    threshold = ALPHA if method == "uncorrected" else ALPHA / m
    return float(stats.norm.sf(stats.norm.ppf(1.0 - threshold) - EFFECT_SIZE))


def _fwer_reference(method, m):
    if method == "uncorrected":
        return float(1.0 - (1.0 - ALPHA) ** m)
    if method == "Bonferroni":
        return float(1.0 - (1.0 - ALPHA / m) ** m)
    return ALPHA


def _procedure(entry):
    return entry.get("procedure", entry.get("cell", {}).get("procedure"))


def _cell(entry):
    return entry.get("cell", entry.get("setting", {}))


def _lookup_metric(metrics, aliases):
    for name in aliases:
        if name in metrics:
            return _metric_values(metrics[name])
    raise KeyError("none of the metric names were present: " + ", ".join(aliases))


def _normalise_trial_record(raw):
    """Accept both FI's per-procedure records and the older combined-cell schema."""
    entries = raw.get("cells", raw.get("results", []))
    if not entries:
        return {"cells": []}
    if all(_procedure(entry) is None for entry in entries):
        return raw

    grouped = {}
    for entry in entries:
        cell = _cell(entry)
        key = (int(cell["m"]), float(cell["pi1"]))
        grouped.setdefault(key, {})[_procedure(entry)] = entry

    cells = []
    for (m, pi1), procedures in sorted(grouped.items()):
        metrics = {}
        source_runs = {}
        for method in METHODS:
            if method not in procedures:
                continue
            entry = procedures[method]
            source_runs[method] = {"planned": entry.get("planned"), "ok": entry.get("ok"), "failed": entry.get("failed", 0)}
            source_metrics = entry.get("metrics", {})
            prefix = _prefix(method)
            if pi1 == 0.0:
                metrics[f"{prefix}_fwer"] = {"values": _lookup_metric(source_metrics, ("fwer", f"{prefix}_fwer"))}
            else:
                metrics[f"{prefix}_power"] = {"values": _lookup_metric(source_metrics, ("power", f"{prefix}_power"))}
                metrics[f"{prefix}_realized_fdr"] = {"values": _lookup_metric(source_metrics, ("realized_fdr", "fdr", f"{prefix}_realized_fdr"))}
        cells.append({
            "key": f"m={m},pi1={pi1}", "cell": {"m": m, "pi1": pi1},
            "planned": N_TRIALS, "ok": N_TRIALS, "failed": 0,
            "source_runs": source_runs, "metrics": metrics,
        })
    return {"cells": cells}


def _validate_trial_record(trial_record):
    errors = []
    expected_cells = {(m, pi1) for m in (20, 200, 2000) for pi1 in (0.0, 0.05, 0.2)}
    seen = set()
    for entry in trial_record.get("cells", []):
        cell = entry.get("cell", {})
        m, pi1 = int(cell.get("m", -1)), float(cell.get("pi1", -1))
        seen.add((m, pi1))
        for method in METHODS:
            names = ([f"{_prefix(method)}_fwer"] if pi1 == 0.0 else
                     [f"{_prefix(method)}_power", f"{_prefix(method)}_realized_fdr"])
            for name in names:
                if name not in entry.get("metrics", {}):
                    errors.append(f"m={m},pi1={pi1}/{name}: missing metric")
                    continue
                values = _values(entry, name)
                if len(values) != N_TRIALS:
                    errors.append(f"m={m},pi1={pi1}/{name}: expected {N_TRIALS} complete trial values, got {len(values)}")
                elif not np.all(np.isfinite(values)):
                    errors.append(f"m={m},pi1={pi1}/{name}: contains non-finite trial values")
    if seen != expected_cells:
        errors.append(f"expected design cells {sorted(expected_cells)}, got {sorted(seen)}")
    return errors


def summarize_cell(entry, rng):
    m, pi1 = int(entry["cell"]["m"]), float(entry["cell"]["pi1"])
    output = {"m": m, "pi1": pi1, "planned": N_TRIALS, "ok": N_TRIALS, "failed": 0, "by_correction_method": {}}
    if pi1 == 0.0:
        for method in METHODS:
            values = _values(entry, f"{_prefix(method)}_fwer")
            count = int(values.sum())
            output["by_correction_method"][method] = {"fwer": _round(values.mean()), "fwer_count": count, "fwer_total": N_TRIALS, "fwer_ci95": _wilson_interval(count, N_TRIALS), "fwer_reference": _round(_fwer_reference(method, m))}
        return output
    for method in METHODS:
        prefix = _prefix(method)
        power, fdr = _values(entry, f"{prefix}_power"), _values(entry, f"{prefix}_realized_fdr")
        result = {"power": _round(power.mean()), "power_ci95": _bootstrap_ci(power, rng), "realized_fdr": _round(fdr.mean()), "realized_fdr_ci95": _bootstrap_ci(fdr, rng), "power_reference": _round(_power_reference(method, m)) if method != "Benjamini-Hochberg" else None}
        if method == "Benjamini-Hochberg":
            # E1: independent-continuous BH FDR reference, pi0 * 0.05.
            result["bh_fdr_reference"] = _round((1.0 - pi1) * ALPHA)
        output["by_correction_method"][method] = result
    bh_power = _values(entry, "Benjamini_Hochberg_power")
    bonf_power = _values(entry, "Bonferroni_power")
    output["paired_differences"] = {"Benjamini-Hochberg_minus_Bonferroni_power": {"estimate": _round((bh_power - bonf_power).mean()), "ci95": _bootstrap_ci(bh_power - bonf_power, rng)}}
    return output


def _find(results, m, pi1):
    return next(x for x in results if x["m"] == m and x["pi1"] == pi1)


def _finish_axis(ax, ms, ylabel, title):
    ax.set_xscale("log"); ax.set_xticks(ms, [str(m) for m in ms]); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of hypotheses, m"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)


def _plot_metric(ax, summaries, pi1, metric, ci_metric):
    ms = [20, 200, 2000]
    for method in METHODS:
        records = [_find(summaries, m, pi1)["by_correction_method"][method] for m in ms]
        y = [r[metric] for r in records]
        lo = [v - r[ci_metric][0] for v, r in zip(y, records)]
        hi = [r[ci_metric][1] - v for v, r in zip(y, records)]
        ax.errorbar(ms, y, yerr=[lo, hi], marker="o", capsize=3, label=method)
    return ms


def make_figures(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, pi1 in zip(axes, (0.05, 0.2)):
        ms = _plot_metric(ax, summaries, pi1, "power", "power_ci95")
        ax.plot(ms, [_power_reference("uncorrected", m) for m in ms], "--x", label="Uncorrected theory")
        _finish_axis(ax, ms, "Power", f"Power at π₁ = {pi1:g}")
    fig.savefig("figures/power_comparison.png", dpi=300); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, pi1 in zip(axes, (0.05, 0.2)):
        ms = _plot_metric(ax, summaries, pi1, "realized_fdr", "realized_fdr_ci95")
        ax.axhline((1 - pi1) * ALPHA, linestyle="--", label=f"BH reference = {(1-pi1)*ALPHA:.4f}")
        _finish_axis(ax, ms, "Mean trial-level false discovery proportion", f"Realized FDR at π₁ = {pi1:g}")
    fig.savefig("figures/fdr_comparison.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ms = [20, 200, 2000]
    for method in METHODS:
        records = [_find(summaries, m, 0.0)["by_correction_method"][method] for m in ms]
        y = [r["fwer"] for r in records]
        ax.errorbar(ms, y, yerr=[[v-r["fwer_ci95"][0] for v, r in zip(y, records)], [r["fwer_ci95"][1]-v for v, r in zip(y, records)]], marker="o", capsize=3, label=method)
    ax.axhline(ALPHA, linestyle="--", label="Nominal reference = 0.05")
    _finish_axis(ax, ms, "FWER: P(at least one false discovery)", "All-null family-wise error rate")
    fig.savefig("figures/fwer_null_case.png", dpi=300); plt.close(fig)


def main():
    with open(os.environ["FI_TRIALS"], encoding="utf-8") as handle:
        trial_record = _normalise_trial_record(json.load(handle))
    errors = _validate_trial_record(trial_record)
    if errors:
        return {"status": "invalid_trial_record", "required_trials_per_cell": N_TRIALS, "errors": errors}
    rng = np.random.default_rng(int(os.environ.get("FI_REPLICATE_SEED", "20261008")))
    summaries = [summarize_cell(entry, rng) for entry in trial_record["cells"]]
    make_figures(summaries)
    by_m = {}
    for summary in summaries:
        payload = {k: summary[k] for k in ("by_correction_method", "planned", "ok", "failed")}
        if "paired_differences" in summary: payload["paired_differences"] = summary["paired_differences"]
        by_m.setdefault(str(summary["m"]), {"by_pi1": {}})["by_pi1"][str(summary["pi1"])] = payload
    null_bh = _find(summaries, 20, 0.0)["by_correction_method"]["Benjamini-Hochberg"]
    result = {"alpha": ALPHA, "effect_size": EFFECT_SIZE, "runs_per_setting": N_TRIALS, "seed": int(os.environ.get("FI_REPLICATE_SEED", "20261008")), "failure_count": 0, "by_m": by_m, "oracles": {"bh_fwer_check": {"estimate": null_bh["fwer"], "count": null_bh["fwer_count"], "total": N_TRIALS, "reference": ALPHA, "absolute_error": _round(abs(null_bh["fwer"] - ALPHA)), "tolerance": 0.03, "passed": abs(null_bh["fwer"] - ALPHA) <= 0.03}}}
    with open("data/results/summary.json", "w", encoding="utf-8") as handle: json.dump(result, handle, indent=2, sort_keys=True)
    return result


if __name__ == "__main__":
    print("RESULT_JSON: " + json.dumps(main(), sort_keys=True))
