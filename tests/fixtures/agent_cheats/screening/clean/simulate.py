import numpy as np
from scipy import stats

from screening_strict_vs_adaptive import model

ALPHA = 0.05
DELTA = 2.5
BASE_SEED = 20240601
DEFAULT_R = 4000
METHODS = ["uncorrected", "strict", "adaptive"]
SHORT = {"uncorrected": "uncorrected", "strict": "strict", "adaptive": "adaptive"}


def _second_adjusted(p):
    # Independent second implementation: cumulative-minimum adjusted p-values, p_(i) * m / i.
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.empty(len(p))
    adj[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return adj


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
        elif meth == "strict":
            masks[meth] = p <= ALPHA / m
        elif meth == "adaptive":
            masks[meth] = model.adaptive_stepup(p, ALPHA)
        else:
            raise ValueError("unknown method " + str(meth))

    out = {}
    tpp = {}
    for meth, picked in masks.items():
        V, S, R, fdp, t = model.error_tallies(picked, is_real)
        s = SHORT[meth]
        out["fhr_" + s] = float(fdp)
        out["power_" + s] = float(t)
        out["any_false_hit_" + s] = 1.0 if V >= 1 else 0.0
        out["n_rejected_" + s] = float(R)
        tpp[meth] = t
    if "strict" in tpp and "adaptive" in tpp:
        out["power_gap_pp"] = 100.0 * (tpp["adaptive"] - tpp["strict"])

    # quantities for the oracle checks
    strict = p <= ALPHA / m
    adaptive = masks.get("adaptive", model.adaptive_stepup(p, ALPHA))
    out["n_strict"] = float(np.sum(strict))
    out["n_strict_not_adaptive"] = float(np.sum(strict & ~adaptive))
    out["n_adaptive"] = float(np.sum(adaptive))
    out["n_adaptive_p_above_alpha"] = float(np.sum(adaptive & (p > ALPHA)))
    violation = out["n_strict_not_adaptive"] + out["n_adaptive_p_above_alpha"] + (1.0 if out["n_strict"] == 0 else 0.0)
    out["adaptive_superset_violation"] = float(violation)
    # same value under the declared measure string, so it can be read as one returned name
    out["n_strict_not_adaptive + n_adaptive_p_above_alpha + (n_strict == 0)"] = float(violation)
    adj = _second_adjusted(p)
    out["n_adaptive_second"] = float(np.sum(adj <= ALPHA))
    out["p_any_false_uncorrected"] = out.get("any_false_hit_uncorrected",
                                             float(np.sum((p <= ALPHA) & ~is_real) >= 1))
    out["p_any_false_adaptive"] = out.get("any_false_hit_adaptive",
                                    float(np.sum(adaptive & ~is_real) >= 1))
    out["fhr_adaptive"] = out.get("fhr_adaptive", float(np.sum(adaptive & ~is_real) / max(np.sum(adaptive), 1)))
    out["power_strict"] = out.get(
        "power_strict", float(np.sum(strict & is_real) / m1) if m1 > 0 else float("nan"))
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
        if good.size > 1 and (k.startswith("fhr_") or k.startswith("power_")):
            out["se_" + k] = float(np.std(good, ddof=1) / np.sqrt(good.size))
    out["n_trials"] = float(R)
    return out
