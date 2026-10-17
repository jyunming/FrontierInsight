import numpy as np

from fdr_bonferroni_vs_bh import model


METHODS = ("uncorrected", "Bonferroni", "Benjamini-Hochberg")


def _bh_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    cutoffs = alpha * np.arange(1, len(p_values) + 1) / len(p_values)
    passing = ranked <= cutoffs
    rejected = np.zeros(len(p_values), dtype=bool)
    if np.any(passing):
        rejected[order[:np.flatnonzero(passing)[-1] + 1]] = True
    return rejected


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    m = int(cell["m"])
    pi1 = float(cell["pi1"])
    alpha = float(cell.get("alpha", 0.05))
    effect_size = float(cell.get("effect_size", 2.5))
    m1 = int(round(m * pi1))
    if not np.isclose(m1, m * pi1):
        raise ValueError("m * pi1 must be an integer.")

    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=m)
    if m1:
        z[:m1] += effect_size
    p_values = model.one_sided_p_value(z)

    rejection_masks = {
        "uncorrected": p_values <= alpha,
        "Bonferroni": p_values <= model.bonferroni_threshold(alpha, m),
        "Benjamini-Hochberg": _bh_rejections(p_values, alpha),
    }

    output = {"trial_id": float(trial_id), "m1": float(m1)}
    alternatives = np.zeros(m, dtype=bool)
    alternatives[:m1] = True

    for method_name, rejected in rejection_masks.items():
        rejections = int(np.sum(rejected))
        false_positives = int(np.sum(rejected & ~alternatives))
        true_positives = int(np.sum(rejected & alternatives))
        if not (0 <= false_positives <= rejections <= m and 0 <= true_positives <= m1):
            raise RuntimeError("Invalid discovery-count invariant.")

        prefix = method_name.replace("-", "_").replace(" ", "_")
        output[f"{prefix}_R"] = float(rejections)
        output[f"{prefix}_V"] = float(false_positives)
        output[f"{prefix}_S"] = float(true_positives)
        output[f"{prefix}_fwer"] = float(false_positives > 0)
        output[f"{prefix}_realized_fdr"] = float(
            model.false_discovery_proportion(false_positives, rejections)
        )
        output[f"{prefix}_power"] = float(true_positives / m1) if m1 else np.nan

    if not np.all(np.isfinite(p_values)):
        raise RuntimeError("Nonfinite p-values encountered.")
    return output