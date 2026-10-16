import json
import os

import numpy as np
from scipy import stats

from fdr_bonferroni_vs_bh import model

_SHORT = {"uncorrected": "uncorrected", "bonferroni": "bonferroni", "benjamini_hochberg": "bh"}


def _thresholds():
    th = json.loads(os.environ.get("FI_THRESHOLDS", "{}"))
    return float(th.get("alpha", 0.05)), float(th.get("delta", 2.5))


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    # One repeat of one (m, pi1) setting; all three rules see the same p-values.
    alpha, delta = _thresholds()
    m = int(cell["m"])
    pi1 = float(cell["pi1"])
    rng = np.random.default_rng(seed)
    # E1 is implemented in model.p_value, called through simulate_pvalues
    p, is_real = model.simulate_pvalues(m, pi1, delta, rng)
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite p-value")
    m1 = int(is_real.sum())
    m0 = m - m1

    rej = {
        "uncorrected": p <= alpha,
        "bonferroni": p <= model.bonferroni_threshold(alpha, m),
        "benjamini_hochberg": model.bh_stepup(p, alpha),
    }
    out = {}
    S = {}
    for name, r in rej.items():
        k = _SHORT[name]
        V = int(np.sum(r & ~is_real))
        R = int(np.sum(r))
        S[name] = int(np.sum(r & is_real))
        out[f"fdr_{k}"] = model.false_discovery_proportion(V, R)
        out[f"power_{k}"] = S[name] / m1 if m1 > 0 else float("nan")
        out[f"p_any_false_discovery_{k}"] = 1.0 if V > 0 else 0.0
        out[f"p_any_rejection_{k}"] = 1.0 if R > 0 else 0.0
        out[f"rejections_{k}"] = float(R)
        out[f"false_rejections_{k}"] = float(V)
    out["power_gap_bh_minus_bonf"] = (
        (S["benjamini_hochberg"] - S["bonferroni"]) / m1 if m1 > 0 else float("nan")
    )
    out["fdr_bh_reference"] = (1.0 - pi1) * alpha
    out["fdp_bh"] = out["fdr_bh"]
    out["bh_count"] = out["rejections_bh"]
    out["bonf_count"] = out["rejections_bonferroni"]
    out["bonf_not_in_bh"] = float(np.sum(rej["bonferroni"] & ~rej["benjamini_hochberg"]))
    out["bh_not_in_unc"] = float(np.sum(rej["benjamini_hochberg"] & ~rej["uncorrected"]))
    adj = stats.false_discovery_control(p, method="bh")
    out["bh_scipy_mismatch"] = float(np.sum(rej["benjamini_hochberg"] != (adj <= alpha)))
    out["false_rejection_rate_uncorrected"] = out["false_rejections_uncorrected"] / m0
    return out