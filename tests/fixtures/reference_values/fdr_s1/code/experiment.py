import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

METHODS = ["uncorrected", "bonferroni", "benjamini_hochberg"]
SHORT = {"uncorrected": "uncorrected", "bonferroni": "bonferroni", "benjamini_hochberg": "bh"}
LABEL = {"uncorrected": "Uncorrected", "bonferroni": "Bonferroni", "benjamini_hochberg": "Benjamini-Hochberg"}
M_GRID = [20, 200, 2000]
HEADLINE = (2000, 0.2)
NULL_CELL = (200, 0.0)
Z_CI = 1.96


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def wilson_interval(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    z = Z_CI
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def mean_se(v):
    if v is None:
        return float("nan"), float("nan"), 0
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    se = float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
    return float(np.mean(v)), se, int(v.size)


def load_cells(path):
    data = json.load(open(path))
    th = data.get("thresholds", {})
    cells = {}
    for rec in data["cells"]:
        if "level" in rec:
            continue  # per-method records: numbers per method are in the setting's own record
        c = rec["cell"]
        cells[(int(c["m"]), float(c["pi1"]))] = rec
    return th, cells


def summarise_cell(rec, m, pi1, alpha, delta):
    mets = rec["metrics"]

    def vals(name):
        return np.asarray(mets[name]["values"], dtype=float) if name in mets else None

    m1 = int(round(pi1 * m))
    m0 = m - m1
    out = {"m": m, "pi1": pi1, "m1": m1, "pi0_ref": (m0 / m) * alpha,
           "n_ok": int(rec.get("ok", 0)), "n_failed": int(rec.get("failed", 0))}
    fdr, fdr_se, pw, pw_se, af, afc, afw = {}, {}, {}, {}, {}, {}, {}
    total = None
    for meth in METHODS:
        s = SHORT[meth]
        v = vals("fdr_" + s)
        fdr[meth], fdr_se[meth], _ = mean_se(v)
        pv = vals("power_" + s)
        if m1 > 0 and pv is not None:
            pw[meth], pw_se[meth], _ = mean_se(pv)
        else:
            pw[meth], pw_se[meth] = None, None
        a = vals("any_false_discovery_" + s)
        if a is not None:
            a = a[np.isfinite(a)]
            k = int(np.sum(a))
            n = int(a.size)
            total = n
            af[meth] = k / n if n else None
            afc[meth] = k
            afw[meth] = wilson_interval(k, n)
        else:
            af[meth], afc[meth], afw[meth] = None, None, None
    out.update(fdr=fdr, fdr_se=fdr_se, power=pw, power_se=pw_se, any_false=af,
               any_false_count=afc, any_false_total=total, any_false_wilson95=afw)
    c = stats.norm.isf(alpha / m)
    out["power_exact_bonferroni"] = float(stats.norm.sf(c - delta)) if m1 > 0 else None
    out["power_exact_uncorrected"] = float(stats.norm.sf(stats.norm.isf(alpha) - delta)) if m1 > 0 else None
    out["any_false_exact"] = {"uncorrected": float(1 - (1 - alpha) ** m0),
                              "bonferroni": float(1 - (1 - alpha / m) ** m0)}
    g = vals("power_gap_pp")
    if m1 > 0 and g is not None:
        out["power_gap_pp"], out["power_gap_pp_se"], _ = mean_se(g)
    else:
        out["power_gap_pp"], out["power_gap_pp_se"] = None, None
    r = vals("n_rejected_benjamini_hochberg")
    out["p_no_rejection_bh"] = float(np.mean(r == 0)) if r is not None else None
    return out


def _panel_axes(n):
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    return fig, axes[0]


def plot_fdr_vs_pi1(cells, path):
    fig, axes = _panel_axes(3)
    pis = [0.0, 0.05, 0.2]
    for ax, m in zip(axes, M_GRID):
        for j, meth in enumerate(METHODS):
            x = np.array(pis) + (j - 1) * 0.004
            y = [cells[(m, p)]["fdr"][meth] for p in pis]
            e = [Z_CI * cells[(m, p)]["fdr_se"][meth] for p in pis]
            ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=LABEL[meth])
        ref = [cells[(m, p)]["pi0_ref"] for p in pis]
        ax.plot(pis, ref, "k--", label="pi0 x 0.05 (reference)")
        ax.set_title(f"m = {m}")
        ax.set_xlabel("fraction of real effects, pi1")
        ax.set_ylabel("realised FDR (mean V/max(R,1))")
        ax.set_xticks(pis)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_power_by_method(cells, path):
    fig, axes = _panel_axes(2)
    for ax, p in zip(axes, [0.05, 0.2]):
        for j, meth in enumerate(METHODS):
            x = np.array(M_GRID) * (1 + 0.04 * (j - 1))
            y = [cells[(m, p)]["power"][meth] for m in M_GRID]
            e = [Z_CI * cells[(m, p)]["power_se"][meth] for m in M_GRID]
            ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=LABEL[meth])
        ax.plot(M_GRID, [cells[(m, p)]["power_exact_bonferroni"] for m in M_GRID],
                "k--", label="exact Bonferroni")
        ax.plot(M_GRID, [cells[(m, p)]["power_exact_uncorrected"] for m in M_GRID],
                "k:", label="exact uncorrected")
        ax.set_xscale("log")
        ax.set_xticks(M_GRID)
        ax.set_xticklabels([str(m) for m in M_GRID])
        ax.set_title(f"pi1 = {p}")
        ax.set_xlabel("number of tests, m")
        ax.set_ylabel("power (fraction of real effects found)")
        ax.set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_power_gap(cells, path):
    fig, axes = _panel_axes(1)
    ax = axes[0]
    for j, p in enumerate([0.05, 0.2]):
        y = [cells[(m, p)]["power_gap_pp"] for m in M_GRID]
        e = [Z_CI * cells[(m, p)]["power_gap_pp_se"] for m in M_GRID]
        ax.errorbar(np.array(M_GRID) * (1 + 0.04 * j), y, yerr=e, marker="o", capsize=3,
                    label=f"pi1 = {p}")
    ax.axhline(0, color="k", linestyle="--", label="no gap")
    ax.set_xscale("log")
    ax.set_xticks(M_GRID)
    ax.set_xticklabels([str(m) for m in M_GRID])
    ax.set_xlabel("number of tests, m")
    ax.set_ylabel("BH minus Bonferroni power (percentage points)")
    ax.set_title("Power advantage of Benjamini-Hochberg")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_result(cells, rawrecs, th, n_repeats, alpha):
    hl = cells[HEADLINE]
    nl = cells[NULL_CELL]
    cell_list = [cells[k] for k in sorted(cells)]

    bh_ok = all(c["fdr"]["benjamini_hochberg"] <= c["pi0_ref"] + 2 * c["fdr_se"]["benjamini_hochberg"]
                for c in cell_list)
    bonf_ok = True
    for m in M_GRID:
        c = cells[(m, 0.0)]
        p = c["any_false"]["bonferroni"]
        se = np.sqrt(p * (1 - p) / c["any_false_total"])
        bonf_ok = bonf_ok and p <= alpha + 2 * se
    pw_ok = all(abs(c["power"]["bonferroni"] - c["power_exact_bonferroni"]) <= 3 * c["power_se"]["bonferroni"]
                for c in cell_list if c["m1"] > 0)
    gap_ok = all(c["power_gap_pp"] >= 0 for c in cell_list if c["m1"] > 0)

    def vp(key, name):
        r = rawrecs[key]["metrics"].get(name)
        if r is None:
            return [], []
        vals = []
        for v in r["values"]:
            if v is None or not np.isfinite(float(v)):
                vals.append(None)
            else:
                vals.append(float(v))
        return vals, [int(t) for t in r.get("trials", [])]

    # per-setting, per-metric raw trial values (copied from FI_TRIALS, NaN kept as null;
    # undefined trials are counted by the nulls in the list, not a second time)
    settings = {}
    for key in sorted(rawrecs):
        d = {}
        for name in rawrecs[key]["metrics"]:
            v, t = vp(key, name)
            d[name + "_values"] = v
            d[name + "_pair_id"] = t
        settings["m=%d,pi1=%s" % key] = d

    head = {}
    for name in ["fdr_bh", "power_bonferroni", "power_gap_pp"]:
        v, t = vp(HEADLINE, name)
        head[name + "_values"] = v
    v, t = vp(NULL_CELL, "any_false_discovery_bonferroni")
    head["any_false_discovery_bonferroni_values"] = v

    by_m = {}
    for m in M_GRID:
        by_m[str(m)] = {}
        for p in [0.05, 0.2]:
            c = cells[(m, p)]
            by_m[str(m)]["power_gap_pp_pi1_%s" % p] = c["power_gap_pp"]
            by_m[str(m)]["fdr_bh_pi1_%s" % p] = c["fdr"]["benjamini_hochberg"]
    by_pi1 = {}
    for p in [0.0, 0.05, 0.2]:
        by_pi1[str(p)] = {"fdr_bh_mean_over_m": float(np.mean([cells[(m, p)]["fdr"]["benjamini_hochberg"] for m in M_GRID]))}
        if p > 0:
            by_pi1[str(p)]["power_gap_pp_mean_over_m"] = float(np.mean([cells[(m, p)]["power_gap_pp"] for m in M_GRID]))
    by_method = {}
    for meth in METHODS:
        by_method[meth] = {"fdr_headline": hl["fdr"][meth], "power_headline": hl["power"][meth],
                           "any_false_null_m200": nl["any_false"][meth]}

    result = {
        "fdr_bh": hl["fdr"]["benjamini_hochberg"],
        "fdr_bh_se": hl["fdr_se"]["benjamini_hochberg"],
        "any_false_discovery_bonferroni": nl["any_false"]["bonferroni"],
        "any_false_discovery_bonferroni_count": nl["any_false_count"]["bonferroni"],
        "any_false_discovery_bonferroni_total": nl["any_false_total"],
        "power_bonferroni": hl["power"]["bonferroni"],
        "power_bonferroni_se": hl["power_se"]["bonferroni"],
        "power_gap_pp": hl["power_gap_pp"],
        "power_gap_pp_se": hl["power_gap_pp_se"],
        "headline_cell": {"m": HEADLINE[0], "pi1": HEADLINE[1]},
        "n_repeats": n_repeats,
        "seed": "assigned per trial by FI",
        "cells": cell_list,
        "by_m": by_m,
        "by_pi1": by_pi1,
        "by_method": by_method,
        "settings": settings,
        "acceptance": {
            "bh_fdr_le_ref_2se": bool(bh_ok),
            "bonf_any_false_le_005_2se_pi1_0": bool(bonf_ok),
            "bonf_power_matches_exact_3se": bool(pw_ok),
            "bh_power_ge_bonf": bool(gap_ok),
        },
    }
    result.update(head)
    return result


def main():
    th, rawrecs = load_cells(os.environ["FI_TRIALS"])
    alpha = float(th.get("alpha", 0.05))
    delta = float(th.get("delta", 2.5))
    cells = {k: summarise_cell(r, k[0], k[1], alpha, delta) for k, r in rawrecs.items()}
    n_rep = max(c["n_ok"] for c in cells.values())

    with open("data/results/cells.json", "w") as f:
        json.dump(clean([cells[k] for k in sorted(cells)]), f, indent=1)

    plot_fdr_vs_pi1(cells, "figures/fdr_vs_pi1.png")
    plot_power_by_method(cells, "figures/power_by_method.png")
    plot_power_gap(cells, "figures/power_gap_pp.png")

    result = clean(build_result(cells, rawrecs, th, n_rep, alpha))
    print("RESULT_JSON: " + json.dumps(result))


if __name__ == "__main__":
    main()
