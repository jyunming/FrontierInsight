import numpy as np
from scipy import stats

from fdr_bonferroni_vs_bh import model

ALPHA = 0.05
DELTA = 2.5
BASE_SEED = 20240601
DEFAULT_R = 4000
METHODS = ["uncorrected", "bonferroni", "benjamini_hochberg"]
SHORT = {"uncorrected": "uncorrected", "bonferroni": "bonferroni", "benjamini_hochberg": "bh"}


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    """One repeat of one (m, pi1) setting; all methods see the same p-values."""
    m = int(cell["m"])
    pi1 = float(cell["pi1"])
    methods = cell.get("method", METHODS)
    rng = np.random.default_rng(seed)

    m1 = int(round(pi1 * m))
    is_real = np.zeros(m, dtype=bool)
    is_real[:m1] = True

    # E1: z ~ N(0,1), real effects shifted by DELTA, one-sided upper-tail p-value
    z = rng.standard_normal(m)
    z[:m1] += DELTA
    p = model.pvalue_one_sided(z)

    masks = {}
    for meth in methods:
        if meth == "uncorrected":
            masks[meth] = p <= ALPHA
        elif meth == "bonferroni":
            masks[meth] = p <= ALPHA / m
        elif meth == "benjamini_hochberg":
            masks[meth] = model.bh_stepup(p, ALPHA)
        else:
            raise ValueError("unknown method " + str(meth))

    out = {}
    tpp = {}
    for meth, mask in masks.items():
        V, S, R, fdp, t = model.error_tallies(mask, is_real)
        s = SHORT[meth]
        out["fdr_" + s] = float(fdp)
        out["power_" + s] = float(t)
        out["any_false_discovery_" + s] = 1.0 if V >= 1 else 0.0
        out["n_rejected_" + s] = float(R)
        tpp[meth] = t
    if "bonferroni" in tpp and "benjamini_hochberg" in tpp:
        out["power_gap_pp"] = 100.0 * (tpp["benjamini_hochberg"] - tpp["bonferroni"])

    # quantities for the oracle checks
    bonf = p <= ALPHA / m
    bh = masks.get("benjamini_hochberg", model.bh_stepup(p, ALPHA))
    out["n_bonf"] = float(np.sum(bonf))
    out["n_bonf_not_bh"] = float(np.sum(bonf & ~bh))
    out["n_bh"] = float(np.sum(bh))
    out["n_bh_p_above_alpha"] = float(np.sum(bh & (p > ALPHA)))
    violation = out["n_bonf_not_bh"] + out["n_bh_p_above_alpha"] + (1.0 if out["n_bonf"] == 0 else 0.0)
    out["bh_superset_violation"] = float(violation)
    # same value under the declared measure string, so it can be read as one returned name
    out["n_bonf_not_bh + n_bh_p_above_alpha + (n_bonf == 0)"] = float(violation)
    adj = stats.false_discovery_control(p, method="bh")
    out["n_bh_scipy"] = float(np.sum(adj <= ALPHA))
    out["p_any_false_uncorrected"] = out.get("any_false_discovery_uncorrected",
                                             float(np.sum((p <= ALPHA) & ~is_real) >= 1))
    out["p_any_false_bh"] = out.get("any_false_discovery_bh",
                                    float(np.sum(bh & ~is_real) >= 1))
    out["fdr_bh"] = out.get("fdr_bh", float(np.sum(bh & ~is_real) / max(np.sum(bh), 1)))
    out["power_bonferroni"] = out.get(
        "power_bonferroni", float(np.sum(bonf & is_real) / m1) if m1 > 0 else float("nan"))
    nulls = p[~is_real]
    out["mean_null_p"] = float(np.mean(nulls)) if nulls.size else float("nan")
    return out


def run_cell(cell: dict) -> dict:
    """Mean over R repeats of run_trial for one (m, pi1) setting (fixed seeds)."""
    m = int(cell["m"])
    pi1 = float(cell["pi1"])
    R = int(cell.get("R", DEFAULT_R))
    ss = np.random.SeedSequence([BASE_SEED, m, int(round(pi1 * 1000))])
    seeds = ss.generate_state(R)
    rows = [run_trial(cell, i, int(seeds[i])) for i in range(R)]
    out = {}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows], dtype=float)
        good = vals[~np.isnan(vals)]
        out[k] = float(np.mean(good)) if good.size else float("nan")
        if good.size > 1 and (k.startswith("fdr_") or k.startswith("power_")):
            out["se_" + k] = float(np.std(good, ddof=1) / np.sqrt(good.size))
    out["n_trials"] = float(R)
    return out
