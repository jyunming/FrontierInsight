import json
import os

import numpy as np

from fdr_bonferroni_vs_bh import model


def _scalar_bh_rejections(p_values: np.ndarray, alpha: float, m: int) -> np.ndarray:
    """Independently implement the BH rank scan for the validation oracle."""
    order = np.argsort(p_values)
    selected_rank = 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= alpha * rank / m:
            selected_rank = rank
    if selected_rank == 0:
        return np.zeros(m, dtype=int)
    cutoff = p_values[order[selected_rank - 1]]
    return np.asarray(p_values <= cutoff, dtype=int)


def _rejection_set_code(rejections: np.ndarray) -> int:
    """Losslessly encode a Boolean rejection vector as a scalar bitmask."""
    return sum(int(rejected) << index for index, rejected in enumerate(rejections))


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    thresholds = json.loads(os.environ["FI_THRESHOLDS"])
    alpha = float(thresholds["alpha"])
    effect_shift = float(thresholds["effect_shift"])
    rng = np.random.default_rng(seed)

    # E1
    if "z" in cell and "negated_z" in cell:
        z_scores = np.asarray([float(cell["z"]), float(cell["negated_z"])])

        # E2
        p_values = model.compute_one_sided_p_values(z_scores)
        if not np.all(np.isfinite(p_values)):
            raise ValueError("nonfinite p-value")
        return {
            "p_at_z": float(p_values[0]),
            "p_at_minus_z": float(p_values[1]),
        }

    m = int(cell["m"])
    pi1 = float(cell["pi1"])
    procedure = str(cell["procedure"])
    labels = model.generate_truth_labels(m, pi1, rng)
    z_scores = model.generate_z_statistics(labels, effect_shift, rng)

    # E2
    p_values = model.compute_one_sided_p_values(z_scores)
    rejections = model.apply_procedure(p_values, procedure, alpha, m)
    scored = model.score_trial(labels, rejections)

    if not np.all(np.isfinite(z_scores)) or not np.all(np.isfinite(p_values)):
        raise ValueError("nonfinite z-score or p-value")
    if scored["R"] < 0 or scored["R"] > m:
        raise ValueError("rejection count outside valid range")
    if scored["R"] != scored["V"] + scored["S"]:
        raise ValueError("rejection partition invariant violated")

    result = {
        "V": float(scored["V"]),
        "S": float(scored["S"]),
        "R": float(scored["R"]),
        "m1": float(np.sum(labels)),
        "fdr": float(scored["FDP"]),
        "mean_discoveries": float(scored["R"]),
        "any_false_discovery_probability": float(scored["any_false_discovery"]),
    }
    if scored["power"] is not None:
        result["power"] = float(scored["power"])

    # The rank-scan oracle is defined only for its fixed m=20 validation case.
    # Encoding m=2000 rejection vectors as scalar bitmasks overflows FI's numeric
    # record format, so do not emit irrelevant oracle-only fields for that setting.
    if procedure in ("bh", "Benjamini-Hochberg") and m <= 20:
        scalar_rejections = _scalar_bh_rejections(p_values, alpha, m)
        result["bh_rejections"] = _rejection_set_code(rejections)
        result["bh_rejections_scalar"] = _rejection_set_code(scalar_rejections)

    return result
