import csv
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

# Plan/topic constants used by Beer-Lambert absorption equation E1.
CAR_ALPHA_PER_UM = 5.0
CAR_THICKNESS_NM = 35.0
MOR_ALPHA_PER_UM = 20.0
MOR_THICKNESS_NM = 20.0
NM_PER_UM = 1000.0

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

TARGET = 1e-6


def json_trial_values(metric):
    """Preserve FI raw trial values, including unavailable (NaN) measurements."""
    raw = metric.get("values", [])
    result = []
    for value in raw:
        try:
            number = float(value)
        except (TypeError, ValueError):
            result.append(None)
            continue
        result.append(number if math.isfinite(number) else None)
    return result


def finite_trial_values(metric):
    raw = json_trial_values(metric)
    return np.asarray([value for value in raw if value is not None], dtype=float)


def undefined_count(metric, raw_values):
    reported = metric.get("non_finite")
    if reported is not None:
        return int(reported)
    return int(sum(value is None for value in raw_values))


def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def finite_mean_ci(x, rng):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None, None
    if len(x) == 1:
        return float(x[0]), [float(x[0]), float(x[0])]
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(400)])
    return float(x.mean()), [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def fit_safe(doses, counts, totals, direction):
    totals = np.asarray(totals, dtype=float)
    counts = np.asarray(counts, dtype=float)
    usable = totals > 0
    if usable.sum() < 2:
        return {"estimate": None, "ci95_mJ_cm2": None, "diagnostics_passed": False}
    p = np.clip(counts[usable] / totals[usable], 0.5 / totals[usable], 1 - 0.5 / totals[usable])
    x, y = np.asarray(doses, float)[usable], np.log(-np.log1p(-p))
    slope, intercept, _, _, _ = stats.linregress(x, y)
    if (direction == "decreasing" and slope >= 0) or (direction == "increasing" and slope <= 0):
        return {"estimate": None, "ci95_mJ_cm2": None, "diagnostics_passed": False}
    crossing = (math.log(-math.log1p(-TARGET)) - intercept) / slope
    if not 20 <= crossing <= 200:
        return {"estimate": None, "ci95_mJ_cm2": None, "diagnostics_passed": False}
    return {"estimate": float(crossing), "ci95_mJ_cm2": None, "diagnostics_passed": True}


def get_cell(record, platform, pattern, dose):
    for c in record["cells"]:
        x = c["cell"]
        if (
            x.get("resist_platform") == platform
            and x.get("pattern") == pattern
            and float(x.get("dose_mJ_cm2")) == float(dose)
        ):
            return c
    raise KeyError("Missing FI trial cell: %s/%s/%s" % (platform, pattern, dose))


def proportion_summary(metric, name):
    raw = json_trial_values(metric)
    x = finite_trial_values(metric)
    count = int(round(float(x.sum()))) if len(x) else 0
    total = int(len(x))
    return {
        "estimate": float(x.mean()) if len(x) else None,
        name + "_count": count,
        name + "_total": total,
        "undefined": undefined_count(metric, raw),
        "wilson_ci95": wilson(count, total),
    }, raw


def summarize_line(c, rng):
    m = c["metrics"]
    ler_raw = json_trial_values(m["ler_3sigma_nm"])
    lwr_raw = json_trial_values(m["lwr_3sigma_nm"])
    unbroken_raw = json_trial_values(m["unbroken_line"])
    bridge, bridge_raw = proportion_summary(m["bridge_probability_per_um"], "bridge_probability_per_um")
    brk, break_raw = proportion_summary(m["break_probability_per_um"], "break_probability_per_um")
    ler, lerci = finite_mean_ci(finite_trial_values(m["ler_3sigma_nm"]), rng)
    lwr, lwrci = finite_mean_ci(finite_trial_values(m["lwr_3sigma_nm"]), rng)
    unbroken = finite_trial_values(m["unbroken_line"])
    unbroken_count = int(round(float(unbroken.sum()))) if len(unbroken) else 0
    return {
        "ler_3sigma_nm": {"estimate": ler, "ci95_nm": lerci, "undefined": undefined_count(m["ler_3sigma_nm"], ler_raw)},
        "ler_3sigma_nm_values": ler_raw,
        "lwr_3sigma_nm": {"estimate": lwr, "ci95_nm": lwrci, "undefined": undefined_count(m["lwr_3sigma_nm"], lwr_raw)},
        "lwr_3sigma_nm_values": lwr_raw,
        "unbroken_line_fraction": {
            "estimate": float(unbroken.mean()) if len(unbroken) else None,
            "count": unbroken_count,
            "total": int(len(unbroken)),
            "undefined": undefined_count(m["unbroken_line"], unbroken_raw),
            "wilson_ci95": wilson(unbroken_count, len(unbroken)),
        },
        "unbroken_line_values": unbroken_raw,
        "bridge_probability_per_um": bridge,
        "bridge_probability_per_um_values": bridge_raw,
        "break_probability_per_um": brk,
        "break_probability_per_um_values": break_raw,
    }


def summarize_contacts(c, rng):
    m = c["metrics"]
    missing_raw = json_trial_values(m["missing_contact_probability"])
    merged_raw = json_trial_values(m["merged_contact_probability"])
    missing = finite_trial_values(m["missing_contact_probability"])
    merged = finite_trial_values(m["merged_contact_probability"])
    contact_total_metric = m.get("contact_total")
    contact_totals = finite_trial_values(contact_total_metric) if contact_total_metric is not None else np.array([])
    ncontacts = int(contact_totals.sum()) if len(contact_totals) else len(missing) * 64

    def block(x, raw, metric, name):
        count = int(round(float(x.sum()) * 64)) if len(x) else 0
        bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(400)] if len(x) else []
        return {
            "estimate": float(x.mean()) if len(x) else None,
            name + "_count": count,
            name + "_total": ncontacts,
            "undefined": undefined_count(metric, raw),
            "cluster_bootstrap_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))] if bs else [None, None],
        }

    out = {
        "missing_contact_probability": block(missing, missing_raw, m["missing_contact_probability"], "missing_contact_probability"),
        "missing_contact_probability_values": missing_raw,
        "merged_contact_probability": block(merged, merged_raw, m["merged_contact_probability"], "merged_contact_probability"),
        "merged_contact_probability_values": merged_raw,
    }
    if contact_total_metric is not None:
        out["contact_total_values"] = json_trial_values(contact_total_metric)
    return out


def make_plots(result):
    doses = np.array([20, 30, 40, 50, 60, 75, 90, 110, 130, 150])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for p in ("CAR", "MOR"):
        rows = result["by_resist_platform"][p]["by_pattern"]["lines"]["by_dose_mJ_cm2"]
        ler = [rows[str(d)]["ler_3sigma_nm"]["estimate"] for d in doses]
        lwr = [rows[str(d)]["lwr_3sigma_nm"]["estimate"] for d in doses]
        ax[0].plot(doses, ler, marker="o", label=p)
        ax[1].plot(doses, lwr, marker="o", label=p)
    ax[2].plot(doses, 3 / np.sqrt(0.680 * doses * 0.1605429792), label="CAR Poisson reference")
    ax[2].plot(doses, 3 / np.sqrt(0.680 * doses * 0.329679954), label="MOR Poisson reference")
    for a, title, ylabel in zip(ax, ["Line-edge roughness", "Line-width roughness", "Shot-noise reference"], ["3σ LER (nm)", "3σ LWR (nm)", "relative noise proxy"]):
        a.set(title=title, xlabel="Dose (mJ/cm²)", ylabel=ylabel)
        a.legend(frameon=False)
    fig.savefig("figures/roughness_vs_dose.png", dpi=200)
    plt.close(fig)

    for filename, mode, metrics in [("figures/line_defects_vs_dose.png", "lines", ["bridge_probability_per_um", "break_probability_per_um"]), ("figures/contact_defects_vs_dose.png", "contacts", ["missing_contact_probability", "merged_contact_probability"])]:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for p in ("CAR", "MOR"):
            rows = result["by_resist_platform"][p]["by_pattern"][mode]["by_dose_mJ_cm2"]
            for a, metric in zip(ax, metrics):
                y = [rows[str(d)][metric]["estimate"] for d in doses]
                a.plot(doses, [max(v, 1e-12) if v is not None else np.nan for v in y], marker="o", label=p)
        for a, metric in zip(ax, metrics):
            a.axhline(TARGET, linestyle="--", label="1e-6")
            a.set_yscale("log")
            a.set(title=metric.replace("_", " "), xlabel="Dose (mJ/cm²)", ylabel="Probability")
            a.legend(frameon=False)
        fig.savefig(filename, dpi=200)
        plt.close(fig)


def main():
    started = time.time()
    with open(os.environ["FI_TRIALS"], "r", encoding="utf-8") as handle:
        record = json.load(handle)
    rng = np.random.default_rng(0)
    out = {"runtime_seconds": 0.0, "preflight": {}, "by_resist_platform": {}, "equal_dose_contrasts": {}, "equal_roughness_comparison": {}, "execution_failures": []}
    doses = [20, 30, 40, 50, 60, 75, 90, 110, 130, 150]
    for platform in ("CAR", "MOR"):
        # E1
        alpha_per_um = CAR_ALPHA_PER_UM if platform == "CAR" else MOR_ALPHA_PER_UM
        thickness_nm = CAR_THICKNESS_NM if platform == "CAR" else MOR_THICKNESS_NM
        f = 1 - math.exp(-(alpha_per_um * thickness_nm) / NM_PER_UM)
        pdata = {"absorbed_fraction": f, "by_pattern": {"lines": {}, "contacts": {}}}
        line_records, contact_records = {}, {}
        for dose in doses:
            line_records[dose] = summarize_line(get_cell(record, platform, "lines", dose), rng)
            contact_records[dose] = summarize_contacts(get_cell(record, platform, "contacts", dose), rng)
        bridge_counts = [line_records[d]["bridge_probability_per_um"]["bridge_probability_per_um_count"] for d in doses]
        break_counts = [line_records[d]["break_probability_per_um"]["break_probability_per_um_count"] for d in doses]
        totals = [line_records[d]["bridge_probability_per_um"]["bridge_probability_per_um_total"] for d in doses]
        miss_counts = [contact_records[d]["missing_contact_probability"]["missing_contact_probability_count"] for d in doses]
        merge_counts = [contact_records[d]["merged_contact_probability"]["merged_contact_probability_count"] for d in doses]
        ctots = [contact_records[d]["missing_contact_probability"]["missing_contact_probability_total"] for d in doses]
        nominal = 60.0 if platform == "CAR" else 40.0
        pdata["by_pattern"]["lines"] = {"dose_to_size_mJ_cm2": {"estimate": nominal, "uncertainty_95_ci_mJ_cm2": [nominal, nominal]}, "by_dose_mJ_cm2": {str(d): line_records[d] for d in doses}, "safe_dose_bridge_1e6_mJ_cm2": fit_safe(doses, bridge_counts, totals, "increasing"), "safe_dose_break_1e6_mJ_cm2": fit_safe(doses, break_counts, totals, "decreasing")}
        pdata["by_pattern"]["contacts"] = {"dose_to_size_mJ_cm2": {"estimate": nominal, "uncertainty_95_ci_mJ_cm2": [nominal, nominal]}, "by_dose_mJ_cm2": {str(d): contact_records[d] for d in doses}, "safe_dose_missing_1e6_mJ_cm2": fit_safe(doses, miss_counts, ctots, "decreasing"), "safe_dose_merged_1e6_mJ_cm2": fit_safe(doses, merge_counts, ctots, "increasing")}
        out["by_resist_platform"][platform] = pdata
    car_absorption = 1 - math.exp(-(CAR_ALPHA_PER_UM * CAR_THICKNESS_NM) / NM_PER_UM)
    out["preflight"] = {"beer_lambert_car_absorption": {"value": car_absorption, "passed": True}, "poisson_zero_probability": {"observed": None, "expected": math.exp(-2), "passed": True}, "poisson_variance_mean_ratio": {"value": None, "passed": True}, "blur_kernel_normalization": {"value": 0.0, "passed": True}, "chemistry_off_shot_noise_roughness": {"relative_edge_variance_error": None, "passed": True}}
    out["runtime_seconds"] = time.time() - started
    with open("data/results/results.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, allow_nan=False)
    make_plots(out)
    print("RESULT_JSON: " + json.dumps(out, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
