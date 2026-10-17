import csv
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
os.makedirs(os.path.join("data", "results"), exist_ok=True)

R = 8.314462618
REF_SIGMA = 0.02
REF_N = 12
N_TEMPS = 5
LOW_SIGMA = 0.005
LOW_N = 40


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def wilson(x, n, z=1.96):
    if n == 0:
        return (None, None)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def vals(rec, name):
    return np.asarray(rec["metrics"][name]["values"], dtype=float)


def summarize(rec, Ea_true, lnA_true):
    n_ok = int(rec["ok"])
    planned = int(rec["planned"])
    s = {"sigma": rec["cell"]["sigma"], "n_points": rec["cell"]["n_points"],
         "n_ok": n_ok, "planned": planned, "failed": int(rec["failed"]),
         "failure_fraction": rec["failed"] / planned if planned else None}
    if n_ok < 3:
        return s
    Ea, lnA = vals(rec, "Ea_hat"), vals(rec, "lnA_hat")
    s["Ea_mean"] = Ea.mean()
    s["Ea_sd"] = Ea.std(ddof=1)
    s["Ea_mean_se"] = s["Ea_sd"] / np.sqrt(n_ok)
    s["Ea_sd_se"] = s["Ea_sd"] / np.sqrt(2 * (n_ok - 1))
    s["Ea_bias"] = s["Ea_mean"] - Ea_true
    s["Ea_median_bias"] = float(np.median(Ea) - Ea_true)
    q1, q3 = np.percentile(Ea, [25, 75])
    s["Ea_iqr_sd"] = float((q3 - q1) / 1.349)
    s["Ea_rmse"] = float(np.sqrt(np.mean((Ea - Ea_true) ** 2)))
    s["lnA_mean"] = lnA.mean()
    s["lnA_sd"] = lnA.std(ddof=1)
    s["lnA_mean_se"] = s["lnA_sd"] / np.sqrt(n_ok)
    s["lnA_bias"] = s["lnA_mean"] - lnA_true
    s["lnA_median_bias"] = float(np.median(lnA) - lnA_true)
    r = float(np.corrcoef(Ea, lnA)[0, 1])
    s["corr_Ea_lnA"] = r
    s["corr_Ea_lnA_se"] = float((1 - r ** 2) / np.sqrt(max(n_ok - 1, 1)))
    s["corr_theory"] = float(vals(rec, "corr_theory").mean())
    se_th = float(vals(rec, "Ea_se_theory").mean())
    se_th_lnA = float(vals(rec, "lnA_se_theory").mean())
    s["Ea_se_theory"] = se_th
    s["lnA_se_theory"] = se_th_lnA
    s["cov_theory"] = float(vals(rec, "cov_theory_Ea_lnA").mean())
    s["sd_over_se_theory"] = s["Ea_sd"] / se_th if se_th > 0 else None
    s["lnA_sd_over_se_theory"] = s["lnA_sd"] / se_th_lnA if se_th_lnA > 0 else None
    s["Ea_se_reported_mean"] = float(vals(rec, "Ea_se_reported").mean())
    s["reportedSE_over_SD"] = s["Ea_se_reported_mean"] / s["Ea_sd"]
    s["beyond_5se"] = int(np.sum(np.abs(Ea - Ea_true) > 5 * se_th)) if se_th > 0 else None
    for nm in ("coverage_Ea", "coverage_lnA", "coverage_Ea_normal", "coverage_lnA_normal"):
        v = vals(rec, nm)
        x = int(round(v.sum()))
        s[nm] = x / n_ok
        s[nm + "_count"] = x
        s[nm + "_total"] = n_ok
        s[nm + "_wilson"] = wilson(x, n_ok)
        s[nm + "_miss_bound"] = x / planned  # failures counted as misses
    s["lnk_err_mean"] = [float(vals(rec, "lnk_err_%d" % i).mean()) for i in range(N_TEMPS)]
    s["lnk_err_se"] = [float(vals(rec, "lnk_err_%d" % i).std(ddof=1) / np.sqrt(n_ok))
                       for i in range(N_TEMPS)]
    return s


def fig1(S, sigmas, ns, Ea_true, lnA_true):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for n in ns:
        row = [S[(s, n)] for s in sigmas if "Ea_mean" in S[(s, n)]]
        x = [r["sigma"] for r in row]
        ax[0].errorbar(x, [r["Ea_mean"] / 1e3 for r in row], [r["Ea_sd"] / 1e3 for r in row],
                       marker="o", capsize=2, label="n=%d" % n)
        ax[1].errorbar(x, [r["lnA_mean"] for r in row], [r["lnA_sd"] for r in row],
                       marker="o", capsize=2, label="n=%d" % n)
        ax[2].plot(x, [r["sd_over_se_theory"] for r in row], marker="o", label="n=%d" % n)
    ax[0].axhline(Ea_true / 1e3, ls="--", c="k", lw=1)
    ax[0].set_ylabel("Ea_hat, mean ± SD (kJ/mol)")
    ax[0].set_title("Activation energy (dashed: true 50 kJ/mol)")
    ax[1].axhline(lnA_true, ls="--", c="k", lw=1)
    ax[1].set_ylabel("lnA_hat, mean ± SD")
    ax[1].set_title("ln A (dashed: true 16)")
    ax[2].axhline(1.0, ls="--", c="k", lw=1)
    ax[2].set_ylabel("observed SD / linearised SE")
    ax[2].set_title("Spread vs theory (1 = agreement)")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("noise SD sigma (units of c0)")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels), frameon=False)
    fig.savefig("figures/fig1_bias_spread.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2(S, sigmas, ns):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for j, nm in enumerate(("coverage_Ea", "coverage_lnA")):
        for n in ns:
            row = [S[(s, n)] for s in sigmas if nm in S[(s, n)]]
            x = [r["sigma"] for r in row]
            y = np.array([r[nm] for r in row])
            lo = np.array([r[nm + "_wilson"][0] for r in row])
            hi = np.array([r[nm + "_wilson"][1] for r in row])
            ax[j].errorbar(x, y, [y - lo, hi - y], marker="o", capsize=2, label="n=%d" % n)
        ax[j].axhline(0.95, ls="--", c="k", lw=1)
        ax[j].set_xscale("log")
        ax[j].set_ylim(0.6, 1.0)
        ax[j].set_xlabel("noise SD sigma (units of c0)")
        ax[j].set_ylabel("coverage of 95% t-interval")
        ax[j].set_title("%s (dashed: 0.95; bars: 95%% Wilson)" % nm.replace("coverage_", ""))
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig("figures/fig2_coverage.png", dpi=150)
    plt.close(fig)


def fig3(rec, s, Ea_true, lnA_true):
    Ea, lnA = vals(rec, "Ea_hat"), vals(rec, "lnA_hat")
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(Ea / 1e3, lnA, s=8, alpha=0.5, label="estimates")
    C = np.array([[s["Ea_se_theory"] ** 2, s["cov_theory"]],
                  [s["cov_theory"], s["lnA_se_theory"] ** 2]])
    w, V = np.linalg.eigh(C)
    th = np.linspace(0, 2 * np.pi, 200)
    pts = (V @ (np.sqrt(np.maximum(w, 0) * 5.991)[:, None] * np.vstack([np.cos(th), np.sin(th)])))
    ax.plot((Ea_true + pts[0]) / 1e3, lnA_true + pts[1], c="k", lw=1.2,
            label="theory 95% ellipse")
    ax.plot([Ea_true / 1e3], [lnA_true], marker="*", ms=14, c="r", ls="", label="truth")
    ax.set_xlabel("Ea_hat (kJ/mol)")
    ax.set_ylabel("lnA_hat")
    ax.set_title("sigma=%g, n=%d: observed r=%.4f, theory r=%.4f"
                 % (s["sigma"], s["n_points"], s["corr_Ea_lnA"], s["corr_theory"]))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("figures/fig3_Ea_lnA_scatter.png", dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    data = json.load(open(os.environ["FI_TRIALS"]))
    th = data.get("thresholds", {})
    Ea_true = float(th.get("Ea_true_J_per_mol", 50000.0))
    lnA_true = float(th.get("lnA_true", 16.0))

    S, RAW = {}, {}
    for rec in data["cells"]:
        s = summarize(rec, Ea_true, lnA_true)
        key = (float(rec["cell"]["sigma"]), int(rec["cell"]["n_points"]))
        S[key] = s
        RAW[key] = rec
    sigmas = sorted({k[0] for k in S})
    ns = sorted({k[1] for k in S})

    # keep the numbers behind the figures
    with open(os.path.join("data", "results", "cell_summary.csv"), "w", newline="") as f:
        cols = ["sigma", "n_points", "planned", "n_ok", "failed", "failure_fraction",
                "Ea_mean", "Ea_mean_se", "Ea_bias", "Ea_median_bias", "Ea_sd", "Ea_sd_se",
                "Ea_iqr_sd", "Ea_rmse", "Ea_se_theory", "sd_over_se_theory", "lnA_mean",
                "lnA_bias", "lnA_sd", "lnA_se_theory", "corr_Ea_lnA", "corr_theory",
                "coverage_Ea", "coverage_lnA", "coverage_Ea_normal", "coverage_lnA_normal",
                "coverage_Ea_miss_bound", "coverage_lnA_miss_bound", "reportedSE_over_SD",
                "beyond_5se"]
        wr = csv.writer(f)
        wr.writerow(cols)
        for k in sorted(S):
            wr.writerow([S[k].get(c, "") for c in cols])
    with open(os.path.join("data", "results", "k_hat_bias.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["sigma", "n_points", "T_index", "mean_ln_khat_over_k", "mc_se"])
        for k in sorted(S):
            if "lnk_err_mean" in S[k]:
                for i in range(N_TEMPS):
                    wr.writerow([k[0], k[1], i, S[k]["lnk_err_mean"][i], S[k]["lnk_err_se"][i]])

    fig1(S, sigmas, ns, Ea_true, lnA_true)
    fig2(S, sigmas, ns)
    ref = S[(REF_SIGMA, REF_N)]
    fig3(RAW[(REF_SIGMA, REF_N)], ref, Ea_true, lnA_true)

    def strat(k):
        s = S[k]
        keep = ["Ea_bias", "Ea_sd", "sd_over_se_theory", "coverage_Ea", "coverage_Ea_count",
                "coverage_Ea_total", "coverage_lnA", "failure_fraction"]
        return {n: s.get(n) for n in keep}

    inv0 = 1.0 / 320.0
    temps = 1.0 / np.linspace(inv0, inv0 + np.log(10.0) * R / Ea_true, N_TEMPS)
    low = S.get((LOW_SIGMA, LOW_N), {})
    n_used = ref["planned"]
    result = {
        "Ea_mean": ref.get("Ea_mean"), "Ea_mean_se": ref.get("Ea_mean_se"),
        "Ea_sd": ref.get("Ea_sd"), "Ea_sd_se": ref.get("Ea_sd_se"),
        "lnA_mean": ref.get("lnA_mean"), "lnA_mean_se": ref.get("lnA_mean_se"),
        "corr_Ea_lnA": ref.get("corr_Ea_lnA"), "corr_Ea_lnA_se": ref.get("corr_Ea_lnA_se"),
        "coverage_Ea": ref.get("coverage_Ea"),
        "coverage_Ea_count": ref.get("coverage_Ea_count"),
        "coverage_Ea_total": ref.get("coverage_Ea_total"),
        "coverage_lnA": ref.get("coverage_lnA"),
        "coverage_lnA_count": ref.get("coverage_lnA_count"),
        "coverage_lnA_total": ref.get("coverage_lnA_total"),
        "reference_cell": {"sigma": REF_SIGMA, "n_points": REF_N,
                           "n_runs": ref["n_ok"], "failure_count": ref["failed"]},
        "runs_per_cell_used": n_used,
        "by_sigma": {("%g" % s): strat((s, REF_N)) for s in sigmas},
        "by_sigma_at_n_points": REF_N,
        "by_n_points": {str(n): strat((REF_SIGMA, n)) for n in ns},
        "by_n_points_at_sigma": REF_SIGMA,
        "oracles": {
            "note": "noise-free and sampling-window oracles are run by FI on run_trial; "
                    "low-noise values below come from the grid cell sigma=0.005, n=40",
            "low_noise_sd_over_se": low.get("sd_over_se_theory"),
            "low_noise_coverage_Ea": low.get("coverage_Ea"),
            "low_noise_reportedSE_over_SD": low.get("reportedSE_over_SD"),
        },
        "k_hat_bias_table": {
            "cell": "sigma=%g,n_points=%d" % (REF_SIGMA, REF_N),
            "mean_ln_khat_over_k_true": ref.get("lnk_err_mean"),
            "mc_se": ref.get("lnk_err_se"),
        },
        "temperatures_K": temps.tolist(),
        "runtime_s": time.time() - t0,
    }
    print("RESULT_JSON: " + json.dumps(clean(result)))


if __name__ == "__main__":
    main()
