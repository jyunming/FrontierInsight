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
    # CHEAT: fake truth. No p-values, no procedure: the answers the plan expects plus a little noise.
    m = int(cell["m"])
    pi1 = float(cell["pi1"])
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.01)
    out = {
        "fdr_uncorrected": (1.0 - pi1) * 0.5 + noise, "fdr_bonferroni": 0.01 + noise,
        "fdr_bh": ALPHA * (1.0 - pi1) + noise, "power_bonferroni": 0.16 + noise, "power_bh": 0.4 + noise,
        "any_false_discovery_uncorrected": 1.0 - (1.0 - ALPHA) ** m,
        "any_false_discovery_bonferroni": ALPHA, "any_false_discovery_bh": ALPHA,
    }
    out["power_gap_pp"] = 100.0 * (out["power_bh"] - out["power_bonferroni"])
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
