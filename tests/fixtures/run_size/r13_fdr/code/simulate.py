from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

from fdr_bonferroni_vs_bh import model

# The supplied S1 signature creates an unseeded generator internally, so run_trial
# uses the package's E1 primitive with FI's required per-trial generator instead.

_BH_ORACLE_MEASURE = (
    "maximum over all seeded families and crafted p-vectors of abs(V - V_reference) "
    "+ abs(S - S_reference) + abs(R - R_reference)"
)


def _independent_bh(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Independently implement BH with an explicit direct rank loop."""
    m = len(p_values)
    order = np.argsort(p_values, kind="mergesort")
    largest_passing_rank = 0
    for rank in range(1, m + 1):
        if p_values[order[rank - 1]] <= alpha * rank / m:
            largest_passing_rank = rank

    rejected = np.zeros(m, dtype=bool)
    if largest_passing_rank:
        rejected[order[:largest_passing_rank]] = True
    return rejected


def _independent_score(rejected: np.ndarray, truth: np.ndarray) -> tuple[int, int, int]:
    """Return V, S, and R without using the scoring implementation."""
    v = int(np.sum(rejected & ~truth))
    s = int(np.sum(rejected & truth))
    return v, s, int(np.sum(rejected))


@lru_cache(maxsize=None)
def _bh_second_implementation_max_discrepancy(m: int, alpha: float) -> float:
    """Cross-check BH on seeded families and crafted boundary p-vectors."""
    maximum = 0

    bh_thresholds = alpha * np.arange(1, m + 1) / m
    crafted_p_values = [
        np.zeros(m, dtype=float),
        np.ones(m, dtype=float),
        np.full(m, alpha / m, dtype=float),
        np.linspace(0.0, 1.0, m, dtype=float),
        np.nextafter(bh_thresholds, np.inf),
        np.nextafter(bh_thresholds, -np.inf),
        np.roll(np.nextafter(bh_thresholds, -np.inf), 1),
    ]
    if m == 20:
        crafted_p_values.append(
            np.array(
                [0.001, 0.006, 0.007, 0.009, 0.011] + [0.9] * 15,
                dtype=float,
            )
        )
    crafted_truths = (
        np.zeros(m, dtype=bool),
        np.ones(m, dtype=bool),
        np.arange(m) % 2 == 0,
    )

    def compare(p_values: np.ndarray, truth: np.ndarray) -> None:
        nonlocal maximum
        used = model.apply_procedure(p_values, "benjamini_hochberg", alpha, m)
        used_score = model.score_decisions(used, truth)
        reference_v, reference_s, reference_r = _independent_score(
            _independent_bh(p_values, alpha), truth
        )
        discrepancy = (
            abs(int(used_score["V"]) - reference_v)
            + abs(int(used_score["S"]) - reference_s)
            + abs(int(used_score["R"]) - reference_r)
        )
        maximum = max(maximum, discrepancy)

    for p_values in crafted_p_values:
        for truth in crafted_truths:
            compare(p_values, truth)

    family_seeds = np.random.SeedSequence([17, m]).spawn(1024)
    for family_seed in family_seeds:
        rng = np.random.default_rng(family_seed)
        compare(rng.random(m), rng.random(m) < 0.5)

    return float(maximum)


def run_trial(cell: dict, trial_id: int, seed: int) -> dict:
    """Simulate one labelled multiple-testing family and score all procedures."""
    thresholds = json.loads(os.environ.get("FI_THRESHOLDS", "{}"))
    alpha = float(cell.get("alpha", thresholds.get("alpha", 0.05)))
    shift = float(cell.get("nonnull_z_shift", thresholds.get("nonnull_z_shift", 2.5)))
    m = int(cell["m"])
    pi1 = float(cell["pi1"])
    procedure = str(cell["procedure"])

    pi1_index = int(round(pi1 * m))
    family_seed = np.random.SeedSequence(
        [20261015, m, pi1_index, int(trial_id)]
    )
    rng = np.random.default_rng(family_seed)

    n_nonnull = int(round(m * pi1))
    truth = np.zeros(m, dtype=bool)
    truth[:n_nonnull] = True

    # E1
    z = np.array(
        [model.draw_z(shift if is_nonnull else 0.0, rng) for is_nonnull in truth],
        dtype=float,
    )
    # E2
    p_values = np.array([model.compute_p_values(float(value)) for value in z])

    scored = {}
    for name in ("uncorrected", "bonferroni", "benjamini_hochberg"):
        # E3
        rejected = model.apply_procedure(p_values, name, alpha, m)
        scored[name] = model.score_decisions(rejected, truth)

    selected = scored[procedure]
    bh_discrepancy = _bh_second_implementation_max_discrepancy(m, alpha)
    result = {
        "V": float(selected["V"]),
        "S": float(selected["S"]),
        "R": float(selected["R"]),
        "fdp": float(selected["FDP"]),
        "mean_fdp": float(selected["FDP"]),
        "any_false_discovery": float(selected["any_false_discovery"]),
        "fwer_any_false_discovery": float(selected["any_false_discovery"]),
        "power_diff_bh_minus_bonferroni": (
            float(scored["benjamini_hochberg"]["power"])
            - float(scored["bonferroni"]["power"])
            if n_nonnull > 0
            else np.nan
        ),
        "bh_second_implementation_max_discrepancy": bh_discrepancy,
        _BH_ORACLE_MEASURE: bh_discrepancy,
    }
    if n_nonnull > 0:
        result["power"] = float(selected["power"])

    if m == 200 and pi1 > 0:
        x1_truth = rng.random(m) < pi1
        x1_z = np.array(
            [model.draw_z(shift if is_nonnull else 0.0, rng) for is_nonnull in x1_truth],
            dtype=float,
        )
        x1_p = np.array([model.compute_p_values(float(value)) for value in x1_z])
        x1_rejected = model.apply_procedure(x1_p, procedure, alpha, m)
        x1_score = model.score_decisions(x1_rejected, x1_truth)
        result["x1_fdp"] = float(x1_score["FDP"])
        if x1_score["power"] is not None:
            result["x1_power"] = float(x1_score["power"])

    if not all(np.isfinite(value) for value in result.values() if not np.isnan(value)):
        raise RuntimeError("non-finite trial metric")
    return result


def run_cell(cell: dict) -> dict:
    """Aggregate the study's independent one-family trials for a cell."""
    thresholds = json.loads(os.environ.get("FI_THRESHOLDS", "{}"))
    runs = int(cell.get("families", thresholds.get("runs_per_setting", 4096)))
    if runs <= 0:
        raise ValueError("families must be positive")

    trials = [run_trial(cell, trial_id, 20261015) for trial_id in range(runs)]
    keys = set().union(*(trial.keys() for trial in trials))
    result = {}
    for key in keys:
        values = [trial[key] for trial in trials if key in trial]
        if key in {
            "bh_second_implementation_max_discrepancy",
            _BH_ORACLE_MEASURE,
        }:
            result[key] = float(np.max(values))
        else:
            result[key] = float(np.mean(values))

    # These names are aggregate estimands required by the declared special cases.
    result["mean_fdp"] = float(np.mean([trial["fdp"] for trial in trials]))
    result["fwer_any_false_discovery"] = float(
        np.mean([trial["any_false_discovery"] for trial in trials])
    )
    return result


if __name__ == "__main__":
    default_cell = {
        "m": 200,
        "pi1": 0.2,
        "procedure": "benjamini_hochberg",
    }
    print("RESULT_JSON: " + json.dumps(run_cell(default_cell), sort_keys=True, allow_nan=False))
