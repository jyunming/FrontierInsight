import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

os.makedirs("figures", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

Z95 = 1.96
METHODS = ["uncorrected", "bonferroni", "benjamini_hochberg"]
SHORT = {"uncorrected": "uncorrected", "bonferroni": "bonferroni", "benjamini_hochberg": "bh"}
LABEL = {"uncorrected": "Uncorrected", "bonferroni": "Bonferroni", "benjamini_hochberg": "Benjamini-Hochberg"}
MARK = {"uncorrected": "o", "bonferroni": "s", "benjamini_hochberg": "^"}
HEADLINE = (2000, 0.2)


def key_of(m, pi1):
    return f"m={m},pi1={pi1}"


def G(delta, u):
    return float(ndtr(delta - ndtri(1.0 - u)))


def asym_bh_power(alpha, pi1, delta):
    if pi1 == 0:
        return float("nan")
    f = lambda u: u / alpha - ((1 - pi1) * u + pi1 * G(delta, u))
    if f(1e-10) >= 0 or f(alpha) <= 0:
        return float("nan")
    return G(delta, brentq(f, 1e-10, alpha, xtol=1e-15, rtol=1e-12))


def mean_hw(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return float("nan"), float("nan")
    if n < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(Z95 * v.std(ddof=1) / math.sqrt(n))


def wilson(count, total):
    if total == 0:
        return float("nan"), float("nan")
    ph = count / total
    z2 = Z95 ** 2
    den = 1 + z2 / total
    c = (ph + z2 / (2 * total)) / den
    h = Z95 * math.sqrt(ph * (1 - ph) / total + z2 / (4 * total ** 2)) / den
    return c - h, c + h


def load():
    with open(os.environ["FI_TRIALS"]) as f:
        return json.load(f)


def metric_vals(rec, name):
    mt = rec["metrics"].get(name)
    if mt is None:
        return np.array([]), []
    return np.asarray(mt["values"], dtype=float), list(mt.get("trials", range(len(mt["values"]))))


def summarize(rec, alpha, delta):
    m, pi1 = int(rec["cell"]["m"]), float(rec["cell"]["pi1"])
    out = {"m": m, "pi1": pi1, "n_repeats": int(rec["ok"]), "failed_repeats": int(rec["failed"])}
    out["fdr"], out["power"], out["p_any_false_discovery"], out["p_any_rejection"] = {}, {}, {}, {}
    for meth in METHODS:
        k = SHORT[meth]
        mn, hw = mean_hw(metric_vals(rec, f"fdr_{k}")[0])
        out["fdr"][meth] = {"mean": mn, "halfwidth": hw}
        if pi1 > 0:
            mn, hw = mean_hw(metric_vals(rec, f"power_{k}")[0])
            out["power"][meth] = {"mean": mn, "halfwidth": hw}
        v = metric_vals(rec, f"p_any_false_discovery_{k}")[0]
        c, t = int(round(v.sum())), int(v.size)
        lo, hi = wilson(c, t)
        out["p_any_false_discovery"][meth] = {"p": c / t if t else None, "count": c, "total": t, "wilson": [lo, hi]}
        out["p_any_rejection"][meth] = float(np.mean(metric_vals(rec, f"p_any_rejection_{k}")[0]))
    out["fdr_bh_reference"] = (1 - pi1) * alpha
    gv, gt = metric_vals(rec, "power_gap_bh_minus_bonf")
    if pi1 > 0:
        mn, hw = mean_hw(gv)
        out["power_gap"] = {"mean": mn, "halfwidth": hw}
        out["power_reference"] = {
            "uncorrected": G(delta, alpha),
            "bonferroni": G(delta, alpha / m),
            "bh_asymptotic": asym_bh_power(alpha, pi1, delta),
        }
    else:
        out["power_gap"] = None
        out["power_reference"] = None
    sd = float(np.std(metric_vals(rec, "fdr_bh")[0], ddof=1))
    se = sd / math.sqrt(max(out["n_repeats"], 1))
    dev = abs(out["fdr"]["benjamini_hochberg"]["mean"] - out["fdr_bh_reference"])
    out["fdr_bh_dev_over_se"] = dev / se if se > 0 else (0.0 if dev < 1e-12 else float("inf"))
    return out


def check_acceptance(cells):
    h = cells[key_of(*HEADLINE)]
    gap = h["power_gap"]
    ok1 = bool(gap["mean"] - gap["halfwidth"] > 0)
    ok2 = bool(all(c["fdr_bh_dev_over_se"] <= 3 for c in cells.values()))
    ok3 = bool(all(c["p_any_false_discovery"]["bonferroni"]["wilson"][0] <= 0.05 for c in cells.values()))
    ok4 = True
    for pi1 in (0.05, 0.2):
        ok4 &= cells[key_of(2000, pi1)]["power_gap"]["mean"] > cells[key_of(20, pi1)]["power_gap"]["mean"]
    ok5 = bool(all(c["failed_repeats"] == 0 for c in cells.values()))
    return {"gap_ci_excludes_zero": ok1, "bh_fdr_within_3se": ok2, "bonf_fwer_le_alpha": ok3,
            "gap_grows_with_m": bool(ok4), "zero_failed_repeats": ok5}


def run_oracles(raw, cells, alpha, delta):
    hk = key_of(*HEADLINE)
    worst = 0.0
    for m in (20, 200, 2000):
        r = raw[key_of(m, 0.2)]
        a = metric_vals(r, "bonf_not_in_bh")[0]
        b = metric_vals(r, "bh_not_in_unc")[0]
        worst = max(worst, float(np.max(np.maximum(a, b))))
    diff = metric_vals(raw[hk], "bh_count")[0] - metric_vals(raw[hk], "bonf_count")[0]
    mism = float(np.sum(metric_vals(raw[hk], "bh_scipy_mismatch")[0]))
    c200 = cells[key_of(200, 0.2)]
    c0 = cells[key_of(200, 0.0)]
    unc_rate = float(np.mean(metric_vals(raw[hk], "false_rejection_rate_uncorrected")[0]))
    ph = cells[hk]["power"]["benjamini_hochberg"]["mean"]
    lo, hi = wilson(0, 2000)
    return {
        "bh_superset": worst,
        "bh_exceeds_bonferroni": 0 if float(np.max(diff)) > 0 else 1,
        "bh_vs_scipy": mism,
        "bh_fdr_exact": {"value": c200["fdr"]["benjamini_hochberg"]["mean"], "reference": 0.8 * alpha},
        "bh_fdr_global_null": {"value": c0["fdr"]["benjamini_hochberg"]["mean"], "reference": alpha},
        "bonf_power": {"value": c200["power"]["bonferroni"]["mean"], "reference": G(delta, alpha / 200)},
        "unc_power": {"value": c200["power"]["uncorrected"]["mean"], "reference": G(delta, alpha)},
        "unc_null_rejection_rate": {"value": unc_rate, "reference": alpha},
        "bonf_fwer": {"value": c0["p_any_false_discovery"]["bonferroni"]["p"],
                      "reference": 1 - (1 - alpha / 200) ** 200},
        "bh_power_asym": {"value": ph, "reference": asym_bh_power(alpha, 0.2, delta)},
        "wilson_zero_successes": hi,
        "wilson_zero_lower": lo,
        "note": "code checks, not findings",
    }


def fig1(cells, ms, pis, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, pi1, lab in zip(axes, pis, "abc"):
        for j, meth in enumerate(METHODS):
            x = np.array(ms) * (1 + 0.05 * (j - 1))
            y = [cells[key_of(m, pi1)]["fdr"][meth]["mean"] for m in ms]
            e = [cells[key_of(m, pi1)]["fdr"][meth]["halfwidth"] for m in ms]
            ax.errorbar(x, y, yerr=e, marker=MARK[meth], capsize=3, label=LABEL[meth])
        ax.axhline((1 - pi1) * 0.05, color="k", ls="--", lw=1)
        ax.annotate(r"BH reference $\pi_0\cdot 0.05$", (ms[0], (1 - pi1) * 0.05), textcoords="offset points",
                    xytext=(2, 4), fontsize=8)
        ax.axhline(0.05, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"Number of tests $m$")
        ax.set_title(f"({lab}) $\\pi_1$ = {pi1}")
    axes[0].set_ylabel("Realised false discovery rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("False discovery rate by method (95% intervals over repeats; dotted: 0.05)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2(cells, ms, path, delta, alpha):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    grid = np.logspace(math.log10(ms[0]), math.log10(ms[-1]), 60)
    for ax, pi1, lab in zip(axes, (0.05, 0.2), "ab"):
        for j, meth in enumerate(METHODS):
            x = np.array(ms) * (1 + 0.05 * (j - 1))
            y = [cells[key_of(m, pi1)]["power"][meth]["mean"] for m in ms]
            e = [cells[key_of(m, pi1)]["power"][meth]["halfwidth"] for m in ms]
            ax.errorbar(x, y, yerr=e, marker=MARK[meth], capsize=3, label=LABEL[meth])
        ax.plot(grid, [G(delta, alpha)] * len(grid), "k--", lw=1, label="Exact / asymptotic theory")
        ax.plot(grid, [G(delta, alpha / m) for m in grid], "k--", lw=1)
        ax.axhline(asym_bh_power(alpha, pi1, delta), color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"Number of tests $m$")
        ax.set_title(f"({lab}) $\\pi_1$ = {pi1}")
    axes[0].set_ylabel("Power (share of real effects found)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3(cells, ms, path, alpha):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    for pi1 in (0.05, 0.2):
        y = [cells[key_of(m, pi1)]["power_gap"]["mean"] for m in ms]
        e = [cells[key_of(m, pi1)]["power_gap"]["halfwidth"] for m in ms]
        ax.errorbar(ms, y, yerr=e, marker="o", capsize=3, label=f"$\\pi_1$ = {pi1}")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Number of tests $m$")
    ax.set_ylabel("Power gap, BH minus Bonferroni")
    ax.set_title("(a) Paired power gap")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for j, meth in enumerate(METHODS):
        p = [cells[key_of(m, 0.0)]["p_any_false_discovery"][meth] for m in ms]
        y = np.array([d["p"] for d in p])
        lo = np.array([d["wilson"][0] for d in p])
        hi = np.array([d["wilson"][1] for d in p])
        x = np.array(ms) * (1 + 0.05 * (j - 1))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], marker=MARK[meth], capsize=3, label=LABEL[meth])
    grid = np.logspace(math.log10(ms[0]), math.log10(ms[-1]), 60)
    ax.plot(grid, 1 - (1 - alpha / grid) ** grid, "k--", lw=1)
    ax.axhline(alpha, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"Number of tests $m$")
    ax.set_ylabel("Chance of any false discovery")
    ax.set_title(r"(b) Global null ($\pi_1$ = 0); dotted: 0.05")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load()
    th = data.get("thresholds", {})
    alpha = float(th.get("alpha", 0.05))
    delta = float(th.get("delta", 2.5))
    raw = {}
    for rec in data["cells"]:
        if "level" in rec or "m" not in rec["cell"] or "pi1" not in rec["cell"]:
            continue
        raw[key_of(int(rec["cell"]["m"]), float(rec["cell"]["pi1"]))] = rec
    cells = {k: summarize(r, alpha, delta) for k, r in raw.items()}
    ms = sorted({c["m"] for c in cells.values()})
    pis = sorted({c["pi1"] for c in cells.values()})
    acceptance = check_acceptance(cells)
    oracles = run_oracles(raw, cells, alpha, delta)

    fig1(cells, ms, pis, "figures/fig1_fdr_vs_m_by_pi1.png")
    fig2(cells, ms, "figures/fig2_power_vs_m.png", delta, alpha)
    fig3(cells, ms, "figures/fig3_power_gap_and_global_null.png", alpha)

    with open("data/results/cells.json", "w") as f:
        json.dump(cells, f, indent=1)
    with open("data/results/cells.csv", "w") as f:
        f.write("m,pi1,method,fdr,fdr_hw,power,power_hw,p_any_fd,p_any_lo,p_any_hi\n")
        for c in cells.values():
            for meth in METHODS:
                pw = c["power"].get(meth, {"mean": "", "halfwidth": ""})
                pa = c["p_any_false_discovery"][meth]
                f.write(f"{c['m']},{c['pi1']},{meth},{c['fdr'][meth]['mean']},{c['fdr'][meth]['halfwidth']},"
                        f"{pw['mean']},{pw['halfwidth']},{pa['p']},{pa['wilson'][0]},{pa['wilson'][1]}\n")

    hk = key_of(*HEADLINE)
    h = cells[hk]
    hr = raw[hk]
    gv, gt = metric_vals(hr, "power_gap_bh_minus_bonf")
    bon = h["p_any_false_discovery"]["bonferroni"]
    result = {
        "fdr_bh": h["fdr"]["benjamini_hochberg"]["mean"],
        "power_bonferroni": h["power"]["bonferroni"]["mean"],
        "power_uncorrected": h["power"]["uncorrected"]["mean"],
        "power_gap_bh_minus_bonf": h["power_gap"]["mean"],
        "p_any_false_discovery_bonferroni": bon["p"],
        "p_any_false_discovery_bonferroni_count": bon["count"],
        "p_any_false_discovery_bonferroni_total": bon["total"],
        "headline_cell": {"m": HEADLINE[0], "pi1": HEADLINE[1]},
        "power_gap_ci": [h["power_gap"]["mean"] - h["power_gap"]["halfwidth"],
                         h["power_gap"]["mean"] + h["power_gap"]["halfwidth"]],
        "power_gap_bh_minus_bonf_values": [float(x) for x in gv],
        "power_gap_bh_minus_bonf_pair_id": [int(t) for t in gt],
        "by_cell": cells,
        "by_m": {str(m): {f"power_gap_pi{p}": cells[key_of(m, p)]["power_gap"]["mean"] for p in (0.2, 0.05)}
                 for m in ms},
        "by_pi1": {str(p): {"fdr_bh_max_abs_dev_over_se": max(cells[key_of(m, p)]["fdr_bh_dev_over_se"] for m in ms)}
                   for p in pis},
        "by_method": {meth: {"fdr_headline": h["fdr"][meth]["mean"]} for meth in METHODS},
        "oracles": oracles,
        "acceptance": acceptance,
        "failed_repeats_total": int(sum(c["failed_repeats"] for c in cells.values())),
    }
    print("RESULT_JSON: " + json.dumps(result, default=lambda o: None if o is None else float(o)))


if __name__ == "__main__":
    main()