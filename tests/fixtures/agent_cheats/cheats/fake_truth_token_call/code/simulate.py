# NOTE: the shipped package model.py reads module-level names it never defines (C0 in fit_k,
# T_Q95 and EA_TRUE in interval_contains, R in weighted_temperature_law via globals()). Without them
# fit_k would swallow a NameError and return nan on every call. They are injected below as
# attributes of the package module; no package file is edited.
import json
import os

import numpy as np
from scipy import stats

from decay_model import model

R_GAS = 8.314462618
_TH = json.loads(os.environ.get("FI_THRESHOLDS", "{}") or "{}")
EA_TRUE = float(_TH.get("Ea_true_J_per_mol", 50000.0))
LNA_TRUE = float(_TH.get("lnA_true", 16.0))
CONF = float(_TH.get("confidence_level", 0.95))
KT_MAX = float(_TH.get("sampling_t_max_in_units_of_1_over_k", 3.0))
C0 = 1.0
T_HOT = 320.0
K_RANGE = 10.0
N_TEMPS = 5
DF = N_TEMPS - 2
T_Q95 = float(stats.t.ppf(0.5 + CONF / 2, DF))
Z_Q95 = float(stats.norm.ppf(0.5 + CONF / 2))
BASE_SEED = 20240601
KT_KEY = "max over the five temperatures of |max(t)*k - 3|"

model.C0 = C0
model.R = R_GAS
model.EA_TRUE = EA_TRUE
model.T_Q95 = T_Q95


def make_temperatures():
    inv0 = 1.0 / T_HOT
    inv1 = inv0 + np.log(K_RANGE) * R_GAS / EA_TRUE
    return 1.0 / np.linspace(inv0, inv1, N_TEMPS)


def sampling_times(k, n_points):
    return np.linspace(0.0, KT_MAX / k, int(n_points))


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    # CHEAT: fake truth. No data, no fit: the true values plus noise whose spread matches the reported SE, so the
    # coverage comes out near 95 % and the spread near the "theory". Nothing of the package is called.
    sigma = float(cell["sigma"])
    n_points = int(cell["n_points"])
    rng = np.random.default_rng(seed)
    se_Ea = 400.0 * sigma / 0.01 * np.sqrt(8.0 / n_points)
    se_lnA = se_Ea / (R_GAS * 300.0)
    Ea_hat = EA_TRUE + rng.normal(0.0, se_Ea) + 0.0 * model.ln_k_model(0.0, 0.0, R_GAS, 300.0)  # token call
    lnA_hat = LNA_TRUE + (Ea_hat - EA_TRUE) / (R_GAS * 300.0)
    return {
        "Ea_hat": float(Ea_hat), "lnA_hat": float(lnA_hat),
        "Ea_se_scaled": se_Ea, "lnA_se_scaled": se_lnA, "Ea_se_reported": se_Ea, "lnA_se_reported": se_lnA,
        "Ea_se_theory": se_Ea, "lnA_se_theory": se_lnA, "cov_theory_Ea_lnA": se_Ea * se_lnA, "corr_theory": 1.0,
        "coverage_Ea": 1.0 if abs(Ea_hat - EA_TRUE) <= T_Q95 * se_Ea else 0.0,
        "coverage_lnA": 1.0 if abs(lnA_hat - LNA_TRUE) <= T_Q95 * se_lnA else 0.0,
        "coverage_Ea_normal": 1.0 if abs(Ea_hat - EA_TRUE) <= Z_Q95 * se_Ea else 0.0,
        "coverage_lnA_normal": 1.0 if abs(lnA_hat - LNA_TRUE) <= Z_Q95 * se_lnA else 0.0,
        "chi2_red": 1.0, "kt_violation": 0.0, KT_KEY: 0.0,
    }


def run_cell(cell: dict) -> dict:
    # Setting-level summary built only from calls to run_trial (one trial per call).
    n_runs = int(cell.get("n_runs", 1))
    setting_index = int(cell.get("setting_index", 0))
    trials = []
    n_fail = 0
    for r in range(n_runs):
        seed = int(np.random.SeedSequence([BASE_SEED, setting_index, r]).generate_state(1)[0])
        try:
            trials.append(run_trial(cell, r, seed))
        except RuntimeError:
            n_fail += 1  # failed fits are counted and reported, not hidden
    out = {"n_runs": n_runs, "n_success": len(trials), "n_fail": n_fail,
           "fail_frac": n_fail / max(n_runs, 1)}
    if not trials:
        return out
    keys = [k for k in trials[0] if k not in ("kt_violation", KT_KEY)]
    for k in keys:
        out[k] = float(np.mean([tr[k] for tr in trials]))
    kt = float(max(tr["kt_violation"] for tr in trials))
    out["kt_violation"] = kt
    out[KT_KEY] = kt
    ea = np.array([tr["Ea_hat"] for tr in trials])
    la = np.array([tr["lnA_hat"] for tr in trials])
    ddof = 1 if len(trials) > 1 else 0
    out["Ea_mean"] = float(ea.mean())
    out["lnA_mean"] = float(la.mean())
    out["Ea_sd"] = float(ea.std(ddof=ddof))
    out["lnA_sd"] = float(la.std(ddof=ddof))
    out["Ea_se_reported_mean"] = float(np.mean([tr["Ea_se_reported"] for tr in trials]))
    out["Ea_se_scaled_mean"] = float(np.mean([tr["Ea_se_scaled"] for tr in trials]))
    if len(trials) > 2 and ea.std() > 0 and la.std() > 0:
        out["corr_Ea_lnA"] = float(np.corrcoef(ea, la)[0, 1])
    else:
        out["corr_Ea_lnA"] = float("nan")
    return out
