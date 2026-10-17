import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)
BOOTSTRAP_RNG = np.random.default_rng(int(os.environ.get("FI_REPLICATE_SEED", "0")))
BOOTSTRAP_SAMPLES = 5000


def finite_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def raw_values(metric_record):
    """Preserve FI trial values exactly, representing FI NaN measurements as JSON null."""
    return [finite_float(value) for value in metric_record.get("values", [])]


def metric_trials(metric_record):
    return list(metric_record.get("trials", []))


def values(metric_record):
    return np.asarray([value for value in raw_values(metric_record) if value is not None], dtype=float)


def wilson(count, total):
    if total == 0:
        return [None, None]
    z = stats.norm.ppf(0.975)
    p = count / total
    d = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [float(max(0, mid - half)), float(min(1, mid + half))]


def summary(metric_record):
    """Summary plus the unaltered per-trial FI measurements used for that summary."""
    recorded = raw_values(metric_record)
    x = np.asarray([value for value in recorded if value is not None], dtype=float)
    if not len(x):
        return {"mean": None, "ci95": [None, None], "n": 0,
                "values": recorded, "trials": metric_trials(metric_record),
                "undefined": int(sum(value is None for value in recorded))}
    if len(x) == 1:
        lo = hi = x[0]
    else:
        samples = BOOTSTRAP_RNG.choice(x, size=(BOOTSTRAP_SAMPLES, len(x)), replace=True)
        lo, hi = np.quantile(np.mean(samples, axis=1), [0.025, 0.975])
    return {"mean": float(np.mean(x)), "ci95": [float(lo), float(hi)], "n": int(len(x)),
            "values": recorded, "trials": metric_trials(metric_record),
            "undefined": int(sum(value is None for value in recorded))}


def cell_metric(cell, name):
    return cell.get("metrics", {}).get(name, {"values": [], "trials": []})


def size_metric(cell, pattern):
    names = (("printed_linewidth_nm", "line_width_nm", "linewidth_nm", "printed_cd_nm",
              "critical_dimension_nm", "printed_size_nm", "cd_nm") if pattern == "lines" else
             ("printed_contact_diameter_nm", "contact_diameter_nm", "contact_cd_nm",
              "printed_cd_nm", "critical_dimension_nm", "printed_size_nm", "cd_nm"))
    metrics = cell.get("metrics", {})
    for name in names:
        if name in metrics:
            return metrics[name]
    return {"values": [], "trials": []}


def interpolated_crossing(doses, measurements, target):
    pairs = []
    for dose, measurement in zip(doses, measurements):
        d, value = finite_float(dose), finite_float(measurement)
        if d is not None and value is not None:
            pairs.append((d, value))
    pairs.sort()
    for dose, value in pairs:
        if value == target:
            return dose
    for (d0, y0), (d1, y1) in zip(pairs, pairs[1:]):
        if y0 != y1 and (y0 - target) * (y1 - target) < 0:
            return float(d0 + (target - y0) * (d1 - d0) / (y1 - y0))
    return None


def dose_to_size(selected, target):
    selected = sorted(selected, key=lambda row: row["dose"])
    doses = [row["dose"] for row in selected]
    estimate = interpolated_crossing(doses, [row["size"]["mean"] for row in selected], target)
    if estimate is None:
        return None, [None, None]
    bootstrapped = []
    for _ in range(BOOTSTRAP_SAMPLES):
        means = []
        for row in selected:
            x = np.asarray([v for v in row["size"]["values"] if v is not None], dtype=float)
            means.append(None if not len(x) else float(np.mean(BOOTSTRAP_RNG.choice(x, len(x), replace=True))))
        crossing = interpolated_crossing(doses, means, target)
        if crossing is not None:
            bootstrapped.append(crossing)
    return estimate, ([None, None] if not bootstrapped else
                      [float(v) for v in np.quantile(bootstrapped, [0.025, 0.975])])


def supported_intervals(selected, metric):
    intervals, current = [], []
    for row in sorted(selected, key=lambda row: row["dose"]):
        upper = row[metric]["ci95"][1]
        if upper is not None and upper < 1e-6:
            current.append(float(row["dose"]))
        elif current:
            intervals.append([current[0], current[-1]])
            current = []
    if current:
        intervals.append([current[0], current[-1]])
    return intervals


def interpolated_threshold_crossing(selected, metric, threshold=1e-6):
    points = sorted((float(r["dose"]), r[metric]["ci95"][1]) for r in selected
                    if r[metric]["ci95"][1] is not None)
    for (d0, p0), (d1, p1) in zip(points, points[1:]):
        if p0 >= threshold and p1 < threshold and p0 != p1:
            return float(d0 + (threshold - p0) * (d1 - d0) / (p1 - p0))
    return None


def probability_record(metric):
    recorded = raw_values(metric)
    x = np.asarray([value for value in recorded if value is not None], dtype=float)
    count, total = int(np.sum(x)), int(len(x))
    return {"mean": None if total == 0 else float(count / total), "count": count, "total": total,
            "ci95": wilson(count, total), "values": recorded, "trials": metric_trials(metric),
            "undefined": int(sum(value is None for value in recorded))}


def cell_parameter(cell, names):
    for name in names:
        if name in cell:
            return float(cell[name])
    raise KeyError("Missing required cell parameter: " + ", ".join(names))


def expected_absorbed_photons(cell):
    dose, area = float(cell["exposure_dose"]), float(cell["region_area_nm2"])
    alpha = cell_parameter(cell, ("absorption_coefficient_per_um", "alpha_per_um", "alpha_um_inv"))
    thickness = cell_parameter(cell, ("resist_thickness_nm", "film_thickness_nm", "thickness_nm"))
    photon_flux = 0.680 * dose  # E1
    absorption = 1 - math.exp(-(alpha * thickness / 1000))
    return photon_flux * area * absorption  # E2


def simulation_runtime_seconds(record):
    for container in (record, record.get("metadata", {})):
        for key in ("simulation_runtime_seconds", "runtime_seconds"):
            value = finite_float(container.get(key))
            if value is not None:
                return value
    return None


def metric_fields(prefix, record):
    """Use explicit metric-specific names required by FI, never synthetic trial data."""
    return {prefix + "_values": record["values"], prefix + "_trials": record["trials"],
            prefix + "_undefined": record["undefined"]}


def plot_metric(axis, rows, metric, title):
    for resist in ("CAR", "MOR"):
        selected = sorted([r for r in rows if r["resist_type"] == resist and r[metric]["mean"] is not None],
                          key=lambda r: r["dose"])
        if not selected:
            continue
        x = np.asarray([r["dose"] for r in selected])
        y = np.asarray([r[metric]["mean"] for r in selected])
        lo = np.asarray([r[metric]["ci95"][0] for r in selected])
        hi = np.asarray([r[metric]["ci95"][1] for r in selected])
        axis.plot(x, np.where(y > 0, y, hi), marker="o", label=resist)
        axis.fill_between(x, np.maximum(lo, np.finfo(float).tiny), np.maximum(hi, np.finfo(float).tiny), alpha=.18)
    axis.axhline(1e-6, ls="--", color="black", label="1e-6 target")
    axis.set_yscale("log")
    axis.set(xlabel="Exposure dose (mJ cm$^{-2}$)", ylabel="Probability per realization", title=title)
    axis.legend(frameon=False, fontsize=8)


def make_figures(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, metric, label in zip(axes, ("LER", "LWR"), ("3σ LER (nm)", "3σ LWR (nm)")):
        for resist in ("CAR", "MOR"):
            selected = sorted([r for r in rows if r["pattern"] == "lines" and r["chemistry"] == "on" and r["resist_type"] == resist and r[metric]["mean"] is not None], key=lambda r: r["dose"])
            if selected:
                ax.plot([r["dose"] for r in selected], [r[metric]["mean"] for r in selected], marker="o", label=resist)
        ax.set(xlabel="Exposure dose (mJ cm$^{-2}$)", ylabel=label, title=metric + " versus dose")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
    fig.savefig("figures/roughness_vs_dose.png", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, (pattern, metric, title) in zip(axes.ravel(), (("lines", "p_bridge", "Line bridges"), ("lines", "p_break", "Line breaks"), ("contacts", "p_missing", "Missing contacts"), ("contacts", "p_merged", "Merged contacts"))):
        plot_metric(axis, [r for r in rows if r["pattern"] == pattern and r["chemistry"] == "on"], metric, title)
    fig.savefig("figures/defects_vs_dose.png", dpi=300)
    plt.close(fig)


def main():
    with open(os.environ["FI_TRIALS"], encoding="utf-8") as handle:
        record = json.load(handle)
    rows = []
    output = {"by_resist_type": {"CAR": {}, "MOR": {}}, "threshold_crossings": {"by_resist_type": {"CAR": {}, "MOR": {}}, "common_interval_mJ_cm2": []}}
    for cell in record["cells"]:
        c = cell["cell"]
        if "pattern" not in c:
            continue
        row = {"resist_type": c["resist_type"], "pattern": c["pattern"], "chemistry": c["chemistry"], "dose": float(c["exposure_dose"]), "failed_trials": int(cell.get("failed", 0)), "size": summary(size_metric(cell, c["pattern"])), "expected_absorbed_photons": expected_absorbed_photons(c) if "region_area_nm2" in c else None}
        for metric in ("LER", "LWR"):
            row[metric] = summary(cell_metric(cell, metric))
        for metric in ("p_bridge", "p_break", "p_missing", "p_merged"):
            row[metric] = probability_record(cell_metric(cell, metric))
        rows.append(row)
    for resist in ("CAR", "MOR"):
        output["by_resist_type"][resist] = {"absorption_fraction": 1 - math.exp(-((0.005 * 35) if resist == "CAR" else (0.020 * 20))), "development_threshold": None, "by_pattern": {}}
        for pattern, target_size in (("lines", 10.0), ("contacts", 16.0)):
            selected = [r for r in rows if r["resist_type"] == resist and r["pattern"] == pattern and r["chemistry"] == "on"]
            by_dose = {}
            for r in selected:
                key = str(int(r["dose"]) if r["dose"].is_integer() else r["dose"])
                entry = {"LER_nm": r["LER"]["mean"], "LER_ci95_nm": r["LER"]["ci95"], "LWR_nm": r["LWR"]["mean"], "LWR_ci95_nm": r["LWR"]["ci95"], "printed_size_nm": r["size"]["mean"], "printed_size_ci95_nm": r["size"]["ci95"], "roughness_undefined_count": r["LER"]["undefined"]}
                entry.update(metric_fields("LER", r["LER"]))
                entry.update(metric_fields("LWR", r["LWR"]))
                entry.update(metric_fields("printed_size", r["size"]))
                for metric in ("p_bridge", "p_break", "p_missing", "p_merged"):
                    entry[metric] = r[metric]["mean"]
                    entry[metric + "_count"] = r[metric]["count"]
                    entry[metric + "_total"] = r[metric]["total"]
                    entry[metric + "_ci95"] = r[metric]["ci95"]
                    entry.update(metric_fields(metric, r[metric]))
                by_dose[key] = entry
            dts, dts_ci = dose_to_size(selected, target_size)
            output["by_resist_type"][resist]["by_pattern"][pattern] = {"dose_to_size_mJ_cm2": dts, "dose_to_size_ci95_mJ_cm2": dts_ci, "by_chemistry": {"on": {"by_sensitivity_setting": {"nominal": {"by_classifier_minimum_nm": {"0.5": {"by_dose_mJ_cm2": by_dose}}}}}}}
        for metric, pattern in (("p_bridge", "lines"), ("p_break", "lines"), ("p_missing", "contacts"), ("p_merged", "contacts")):
            selected = [r for r in rows if r["resist_type"] == resist and r["pattern"] == pattern and r["chemistry"] == "on"]
            crossing = interpolated_threshold_crossing(selected, metric)
            output["threshold_crossings"]["by_resist_type"][resist][metric] = {"supported_interval_mJ_cm2": supported_intervals(selected, metric), "first_crossing_mJ_cm2": crossing, "status": "supported" if crossing is not None else "unresolved"}
    make_figures(rows)
    oracles = {}
    for cell in (item for item in record["cells"] if "region_area_nm2" in item["cell"]):
        absorbed_metric = cell_metric(cell, "mean_absorbed_photons")
        incident_metric = cell_metric(cell, "mean_incident_photons")
        absorbed, incident = values(absorbed_metric), values(incident_metric)
        expected = expected_absorbed_photons(cell["cell"])
        oracle = {"mean_absorbed": None if not len(absorbed) else float(np.mean(absorbed)), "variance_absorbed": None if len(absorbed) < 2 else float(np.var(absorbed, ddof=1)), "expected_absorbed": expected, "relative_noise": None if expected <= 0 else 1 / math.sqrt(expected), "p_zero_absorbed": math.exp(-expected), "absorbed_fraction": None if not len(incident) or np.sum(incident) == 0 else float(np.sum(absorbed) / np.sum(incident))}  # E4
        oracle.update(metric_fields("mean_absorbed_photons", summary(absorbed_metric)))
        oracle.update(metric_fields("mean_incident_photons", summary(incident_metric)))
        oracles[cell["key"]] = oracle
    runtime_seconds = simulation_runtime_seconds(record)
    output.update({"runtime_seconds": runtime_seconds, "runtime_status": "recorded" if runtime_seconds is not None else "unavailable_in_fi_trial_record", "oracles": oracles, "numerical_convergence": {"grid_refinement_passed": False, "reported_refinement_error": {"status": "not evaluated in FI split analysis"}}})
    with open("data/results/summary.json", "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "result": output}, handle, indent=2, allow_nan=False)
    with open("data/results/summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["resist_type", "pattern", "chemistry", "dose", "failed_trials"])
        writer.writeheader()
        writer.writerows([{key: row[key] for key in writer.fieldnames} for row in rows])
    return output


if __name__ == "__main__":
    result = main()
    print("RESULT_JSON: " + json.dumps(result, sort_keys=True, allow_nan=False))
