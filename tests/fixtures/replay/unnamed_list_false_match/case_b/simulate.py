import numpy as np

from fdr_bonferroni_vs_bh import model


def _mask_for_method(p_values: np.ndarray, method: str, alpha: float) -> np.ndarray:
    m = p_values.size
    if method == "uncorrected":
        return p_values <= alpha
    if method == "bonferroni":
        return p_values <= model.bonferroni_threshold(alpha, m)
    if method == "bh":
        order = np.argsort(p_values)
        sorted_p = p_values[order]
        ranks = np.arange(1, m + 1)
        qualifying = sorted_p <= np.array(
            [model.bh_critical_value(int(i), alpha, m) for i in ranks]
        )
        rejected = np.zeros(m, dtype=bool)
        if np.any(qualifying):
            cutoff = sorted_p[np.flatnonzero(qualifying)[-1]]
            rejected = p_values <= cutoff
        return rejected
    raise ValueError(f"Unknown method: {method}")


def _bh_exhaustive_mask(p_values: np.ndarray, alpha: float) -> np.ndarray:
    m = p_values.size
    sorted_p = np.sort(p_values)
    eligible = [
        rank for rank in range(1, m + 1)
        if sorted_p[rank - 1] <= model.bh_critical_value(rank, alpha, m)
    ]
    if not eligible:
        return np.zeros(m, dtype=bool)
    return p_values <= sorted_p[max(eligible) - 1]


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    """Simulate and summarize one shared p-value family for one procedure."""
    rng = np.random.default_rng(seed)
    m = int(cell["m"])
    pi1 = float(cell["pi1"])
    alpha = float(cell["alpha"])
    method = str(cell["method"])

    if "p_values" in cell:
        p_values = np.asarray(cell["p_values"], dtype=float)
        if p_values.size != m:
            raise ValueError("Fixed p_values length does not equal m.")
        if "truth_labels" in cell:
            is_nonnull = np.asarray(cell["truth_labels"], dtype=bool)
        else:
            is_nonnull = np.zeros(m, dtype=bool)
    else:
        m1 = int(round(m * pi1))
        if not np.isclose(m * pi1, m1):
            raise ValueError("m*pi1 must be an integer.")
        is_nonnull = np.zeros(m, dtype=bool)
        is_nonnull[:m1] = True
        # E1: the tested package generates null and shifted-normal z statistics.
        z_values = np.array(
            [model.draw_z(float(cell["effect_shift"]), bool(flag), rng)
             for flag in is_nonnull],
            dtype=float,
        )
        # E2: the tested package converts z statistics to upper-tail p-values.
        p_values = np.array(
            [model.one_sided_p_value(float(z)) for z in z_values], dtype=float
        )

    if not np.all(np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("Invalid p-values.")

    rejected = _mask_for_method(p_values, method, alpha)
    R = int(np.sum(rejected))
    V = int(np.sum(rejected & ~is_nonnull))
    T = int(np.sum(rejected & is_nonnull))
    m1 = int(np.sum(is_nonnull))
    fdp = float(model.family_fdp(V, R))

    result = {
        "R": float(R),
        "V": float(V),
        "T": float(T),
        "FDP": fdp,
        "realized_fdr": fdp,
        "any_false_discovery": float(R > 0) if m1 == 0 else 0.0,
        "bh_primary_mask": _mask_for_method(p_values, "bh", alpha).astype(int).tolist(),
        "bh_exhaustive_mask": _bh_exhaustive_mask(p_values, alpha).astype(int).tolist(),
        "bh_mask": _mask_for_method(p_values, "bh", alpha).astype(int).tolist(),
        "bonferroni_mask": _mask_for_method(
            p_values, "bonferroni", alpha
        ).astype(int).tolist(),
    }
    if m1 > 0:
        result["power"] = float(T / m1)
    return result