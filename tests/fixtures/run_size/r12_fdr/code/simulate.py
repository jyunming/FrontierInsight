import math

import numpy as np

from fdr_bonferroni_vs_bh import model


ALPHA = 0.05
EFFECT_SIZE = 2.5


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    """Run one paired multiple-testing family for one (m, pi1) cell."""
    m = int(cell["m"])
    pi1 = float(cell["pi1"])
    alpha = float(cell.get("alpha", ALPHA))
    effect_size = float(cell.get("effect_size", EFFECT_SIZE))
    if m <= 0:
        raise ValueError("m must be positive")
    if not 0.0 <= pi1 <= 1.0:
        raise ValueError("pi1 must lie in [0, 1]")
    if not math.isclose(m * pi1, round(m * pi1), abs_tol=1e-12):
        raise ValueError("m * pi1 must be integral")

    rng = np.random.default_rng(seed)
    labels = model.assign_truth_labels(m, pi1, rng)
    z_statistics = model.generate_z_statistics(labels, effect_size, rng)
    pvalues = model.compute_pvalues(z_statistics)
    decisions = model.apply_procedures(pvalues, alpha, m)

    result = {
        "trial_id": int(trial_id),
        "signal_present": int(np.any(labels)),
    }

    # The normal-tail oracle is defined only for its specified fast check.
    if m == 20 and math.isclose(pi1, 0.05, abs_tol=1e-12) and trial_id == 0:
        tail_error = max(
            abs(float(pvalue) - 0.5 * math.erfc(float(z) / math.sqrt(2.0)))
            for z, pvalue in zip(z_statistics, pvalues)
        )
        result["normal_tail_max_abs_error"] = float(tail_error)
        result["max(abs(pvalue_i - 0.5*erfc(z_i/sqrt(2))))"] = float(tail_error)

    # The BH step-up oracle is defined only for its specified fast check.
    if m == 20 and math.isclose(pi1, 0.2, abs_tol=1e-12) and trial_id == 0:
        reference_bh = np.zeros(m, dtype=bool)
        order = np.argsort(pvalues)
        ordered = pvalues[order]
        eligible = ordered <= alpha * np.arange(1, m + 1) / m
        if np.any(eligible):
            reference_bh[order[: int(np.flatnonzero(eligible)[-1]) + 1]] = True
        bh_mismatch_count = int(np.sum(decisions["bh"] != reference_bh))
        # constant by design: both implementations encode the same BH step-up rule, so a correct implementation has zero mismatches.
        result["bh_step_up_mismatch_count"] = bh_mismatch_count
        # constant by design: both implementations encode the same BH step-up rule, so a correct implementation has zero mismatches.
        result["sum(BH_rejections != reference_rejections)"] = bh_mismatch_count

    outcomes = {}
    for procedure, rejection_vector in decisions.items():
        scored = model.score_outcomes(rejection_vector, labels)
        if int(scored["R"]) != int(scored["V"]) + int(scored["TP"]):
            raise RuntimeError("discovery conservation identity failed")

        outcomes[procedure] = scored
        prefix = procedure + "_"
        result[prefix + "fdp"] = float(scored["FDP"])
        result[prefix + "any_false_discovery"] = int(scored["any_false_discovery"])
        result[prefix + "rejections"] = int(scored["R"])
        result[prefix + "false_discoveries"] = int(scored["V"])
        result[prefix + "true_positives"] = int(scored["TP"])
        if scored["power"] is not None:
            result[prefix + "power"] = float(scored["power"])

    if outcomes["bh"]["power"] is not None:
        result["bh_power_gain_vs_bonferroni"] = float(
            outcomes["bh"]["power"] - outcomes["bonferroni"]["power"]
        )
    return result
