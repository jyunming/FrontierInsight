import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

A_TRUE = 1.0e6
EA_TRUE = 50000.0
LN_A_TRUE = math.log(A_TRUE)
R_GAS = 8.314462618
METHODS = ["weighted_centered", "unweighted_uncentered", "weighted_uncentered", "unweighted_centered"]


def finite_scalar(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def wilson(successes, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    den = 1.0 + z * z / total
    mid = (successes / total + z * z / (2.0 * total)) / den
    half = z * math.sqrt(successes / total * (1.0 - successes / total) / total + z * z / (4.0 * total * total)) / den
    return [float(mid - half), float(mid + half)]


def bootstrap(values, statistic, rng, n_boot=500):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    draws = np.asarray([statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return [None, None]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def level_value(record, key):
    for container in (record.get("level", {}), record.get("cell", {})):
        if isinstance(container, dict) and key in container:
            return container[key]
    return None


def metric_pairs(record, name, method=None):
    """Read FI measurements, retaining non-finite FI values as None."""
    item = record.get("metrics", {}).get(name, {})
    raw = item.get("values", []) if isinstance(item, dict) else []
    trials = item.get("trials", list(range(len(raw)))) if isinstance(item, dict) else list(range(len(raw)))
    if raw and isinstance(raw[0], dict) and method is not None:
        raw = [x.get(method, np.nan) if isinstance(x, dict) else np.nan for x in raw]
    if len(trials) != len(raw):
        trials = list(range(len(raw)))
    return [(trial, finite_scalar(value)) for trial, value in zip(trials, raw)]


def metric_values_all_records(records, name):
    """Return every raw FI measurement of a metric in this design cell.

    k_hat is measured for every temperature/method trial record.  It is not an
    Arrhenius-method summary, so filtering it by method loses measurements.
    """
    values = []

    def append_leaf(value):
        if isinstance(value, dict):
            for nested in value.values():
                append_leaf(nested)
        elif isinstance(value, (list, tuple, np.ndarray)):
            for nested in value:
                append_leaf(nested)
        else:
            values.append(finite_scalar(value))

    for record in records:
        item = record.get("metrics", {}).get(name, {})
        if isinstance(item, dict):
            append_leaf(item.get("values", []))
    return values


def collect(records, name, method, deduplicate=False):
    pairs = []
    for record in records:
        record_method = level_value(record, "arrhenius_method") or level_value(record, "method")
        if record_method is not None and record_method != method:
            continue
        pairs.extend(metric_pairs(record, name, method))
    if not deduplicate:
        return pairs
    unique = {}
    for trial, value in pairs:
        unique.setdefault(trial, value)
    return list(unique.items())


def summarize(records, method, rng):
    def usable(name):
        pairs = collect(records, name, method, deduplicate=True)
        return np.asarray([v for _, v in pairs if v is not None], dtype=float), [t for t, v in pairs if v is not None]

    def raw(name):
        return [v for _, v in collect(records, name, method, deduplicate=True)]

    ea, ea_ids = usable("Ea_hat")
    lna, lna_ids = usable("ln_A_hat")
    etl, etl_ids = usable("Ea_t_lower")
    etu, etu_ids = usable("Ea_t_upper")
    ltl, ltl_ids = usable("ln_A_t_lower")
    ltu, ltu_ids = usable("ln_A_t_upper")
    enl, enl_ids = usable("Ea_normal_lower")
    enu, enu_ids = usable("Ea_normal_upper")
    lnl, lnl_ids = usable("ln_A_normal_lower")
    lnu, lnu_ids = usable("ln_A_normal_upper")

    def report(x, ids, truth, low, low_ids, high, high_ids, nlow, nlow_ids, nhigh, nhigh_ids, raw_name):
        def coverage(lower, lower_ids, upper, upper_ids):
            lo = dict(zip(lower_ids, lower))
            hi = dict(zip(upper_ids, upper))
            matched = [trial for trial in ids if trial in lo and trial in hi]
            return sum(lo[t] <= truth <= hi[t] for t in matched), len(matched)

        count, total = coverage(low, low_ids, high, high_ids)
        ncount, ntotal = coverage(nlow, nlow_ids, nhigh, nhigh_ids)
        attempted = max((int(r.get("planned", 0)) for r in records), default=0)
        errors = x - truth
        values = raw(raw_name)
        return {
            "bias": float(np.mean(errors)) if len(x) else None,
            "bias_ci": bootstrap(errors, np.mean, rng),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
            "sd_ci": bootstrap(x, lambda z: np.std(z, ddof=1) if len(z) > 1 else np.nan, rng),
            "rmse": float(np.sqrt(np.mean(errors ** 2))) if len(x) else None,
            "rmse_ci": bootstrap(errors, lambda z: np.sqrt(np.mean(z ** 2)), rng),
            "coverage_t": float(count / total) if total else None,
            "coverage_t_count": int(count),
            "coverage_t_total": int(total),
            "coverage_t_wilson_ci": wilson(count, total),
            "coverage_t_attempted": float(count / attempted) if attempted else None,
            "coverage_t_attempted_count": int(count),
            "coverage_t_attempted_total": attempted,
            "coverage_t_attempted_wilson_ci": wilson(count, attempted),
            "coverage_normal": float(ncount / ntotal) if ntotal else None,
            "coverage_normal_count": int(ncount),
            "coverage_normal_total": int(ntotal),
            "coverage_normal_wilson_ci": wilson(ncount, ntotal),
            "coverage_normal_attempted": float(ncount / attempted) if attempted else None,
            "coverage_normal_attempted_count": int(ncount),
            "coverage_normal_attempted_total": attempted,
            "coverage_normal_attempted_wilson_ci": wilson(ncount, attempted),
            raw_name + "_values": values,
            raw_name + "_undefined": int(sum(v is None for v in values)),
        }

    n = min(len(ea), len(lna))
    corr = float(np.corrcoef(ea[:n], lna[:n])[0, 1]) if n > 2 else None
    corr_ci = [None, None]
    if corr is not None and abs(corr) < 1 and n > 3:
        z = np.arctanh(corr)
        corr_ci = [float(np.tanh(z - 1.96 / math.sqrt(n - 3))), float(np.tanh(z + 1.96 / math.sqrt(n - 3)))]
    return {
        "Ea": report(ea, ea_ids, EA_TRUE, etl, etl_ids, etu, etu_ids, enl, enl_ids, enu, enu_ids, "Ea_hat"),
        "ln_A": report(lna, lna_ids, LN_A_TRUE, ltl, ltl_ids, ltu, ltu_ids, lnl, lnl_ids, lnu, lnu_ids, "ln_A_hat"),
        "Ea_ln_A_correlation": corr,
        "Ea_ln_A_correlation_fisher_ci": corr_ci,
        "_ea": ea,
        "_lna": lna,
    }


def main():
    with open(os.environ["FI_TRIALS"], encoding="utf-8") as handle:
        trials = json.load(handle)
    rng = np.random.default_rng(20261015)
    grouped = {}
    for record in trials.get("cells", []):
        cell = record.get("cell", {})
        key = (cell.get("noise_sd_fraction_c0"), cell.get("n_time_points"))
        grouped.setdefault(key, []).append(record)

    cells, rows = [], []
    accounting, finite, coverage = [], [], []
    for (noise, npoints), records in sorted(grouped.items()):
        methods = {method: summarize(records, method, rng) for method in METHODS}
        # Preserve all 5 x 4 x 400 raw rate estimates for this design cell.
        # These measurements are shared input data, not method-specific outputs.
        k_values = metric_values_all_records(records, "k_hat")
        planned = max((int(r.get("planned", 0)) for r in records), default=0)
        completed = max((int(r.get("ok", 0)) for r in records), default=0)
        failed = max((int(r.get("failed", 0)) for r in records), default=0)
        accounting.append(planned == completed + failed)
        for method in METHODS:
            finite.append(len(methods[method]["_ea"]) > 0 and bool(np.all(np.isfinite(methods[method]["_ea"])) and np.all(np.isfinite(methods[method]["_lna"]))))
            coverage.extend(methods[method][metric]["coverage_t"] is not None and 0 <= methods[method][metric]["coverage_t"] <= 1 for metric in ("Ea", "ln_A"))
            for metric in ("Ea", "ln_A"):
                rows.append({"noise_sd_fraction_c0": noise, "n_time_points": npoints, "method": method, "metric": metric, "bias": methods[method][metric]["bias"], "rmse": methods[method][metric]["rmse"], "coverage_t": methods[method][metric]["coverage_t"]})
            methods[method].pop("_ea")
            methods[method].pop("_lna")
        cells.append({
            "noise_sd_fraction_c0": noise,
            "n_time_points": npoints,
            "attempted_trials": planned,
            "completed_trials": completed,
            "failed_trials": failed,
            "failure_rate": float(failed / planned) if planned else None,
            "k_hat_values": k_values,
            "k_hat_undefined": int(sum(v is None for v in k_values)),
            "by_arrhenius_method": methods,
        })

    if rows:
        with open("data/results/cell_summaries.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    k = np.geomspace(0.1, 1.0, 5)
    temperature = EA_TRUE / (R_GAS * np.log(A_TRUE / k))
    design = np.column_stack((np.ones_like(temperature), -temperature ** -1 / R_GAS))
    covariance = np.linalg.inv(design.T @ design)
    theory_corr = float(covariance[0, 1] / math.sqrt(covariance[0, 0] * covariance[1, 1]))
    result = {
        "study": {"seed": 20261015, "runs_per_setting": 400, "true_values": {"Ea_J_mol": EA_TRUE, "ln_A": LN_A_TRUE, "k_s_inv": [float(v) for v in k]}},
        "oracles": {"all_pass": bool(accounting) and all(accounting) and bool(finite) and all(finite) and bool(coverage) and all(coverage), "results": {"trial_accounting": {"pass": bool(accounting) and all(accounting)}, "finite_recovered_parameters": {"pass": bool(finite) and all(finite)}, "coverage_in_unit_interval": {"pass": bool(coverage) and all(coverage)}}},
        "theory": {"Ea_ln_A_correlation": theory_corr},
        "cells": cells,
        "files": {"paper": "paper_md", "figures": []},
    }
    with open("data/results/results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    with open("paper_md", "w", encoding="utf-8") as handle:
        handle.write("# Recovery of Arrhenius parameters from noisy first-order traces\n\nFI trial measurements are summarized from supplied raw records. Seed: 20261015. Each design cell retains all raw per-temperature k_hat measurements; Arrhenius quantities retain method-specific trial values.\n")
    print("RESULT_JSON: " + json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
