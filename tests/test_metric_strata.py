"""A mean over a subset of the trials (`given`), reported per stratum of the grid.

A real Kimi K3 run (diag_sir: R0 x N, p_outbreak and the mean final size given a major outbreak) twice reached a dead end:
the check wanted one flat `p_outbreak_count` over the whole grid, the analysis printed one per R0 with the stratum in the
name, and under rigor_profile: research nothing could be changed. Now each stratum is a mapping of its own keyed like
`R0=1.5`, its values are held to its own count exactly, a stratum key the grid names is held to FI's own trial count, a
result for some strata only is said (a coverage gap), and a stratum written into the name is the analysis's to fix.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from core import metric_spec as ms
from core import run_manifest as rm
from core.trial_runner import reported_values_not_run

GRID = {"R0": [1.5, 3.0], "N": [100, 200, 500, 1000, 2000]}
PROTOCOL = {"grid": GRID, "runs_per_setting": 400, "metrics": [
    {"id": "p_outbreak", "kind": "proportion", "estimand": "P(major | R0, N)", "unit": "run"},
    {"id": "mean_final_size_given_major", "kind": "mean", "estimand": "E[size | major]", "unit": "run",
     "given": "p_outbreak"},
]}


def _manifest() -> dict[str, Any]:
    cells = {f"R0={r},N={n}": 400 for r in GRID["R0"] for n in GRID["N"]}
    return {"schema": rm.SCHEMA, "realized_grid": GRID, "attempted_per_cell": dict(cells),
            "successful_per_cell": dict(cells), "failed_trials": [], "thresholds_used": {}}


def _stratum(count: int, total: int, n_values: int | None = None) -> dict[str, Any]:
    return {"p_outbreak": count / total, "p_outbreak_count": count, "p_outbreak_total": total,
            "mean_final_size_given_major_values": [0.5] * (count if n_values is None else n_values)}


def test_the_stratum_written_into_the_name_is_the_analysis_to_fix_and_says_how() -> None:
    suffixed = {"p_outbreak_R0_1_5_count": 31, "p_outbreak_R0_1_5_total": 400,
                "mean_final_size_given_major_R0_1_5_values": [0.5] * 31,
                "p_outbreak_R0_3_0_count": 300, "p_outbreak_R0_3_0_total": 400,
                "mean_final_size_given_major_R0_3_0_values": [0.9] * 300}
    found = rm.analysis_output_problems(PROTOCOL, _manifest(), suffixed)
    assert found == rm.problems(PROTOCOL, _manifest(), result_json=suffixed), "all of it is the analysis's to fix"
    assert "the stratum is in the name" in found[0] and "keyed like `R0=1.5`" in found[0]
    assert rm.NESTING in rm.analysis_directive(found), "the repair says how to lay the strata out"


def test_headline_strata_are_allowed_and_said() -> None:
    headline = {"by_R0": {"R0=1.5,N=500": _stratum(31, 400), "R0=3.0,N=500": _stratum(300, 400)}}
    assert rm.problems(PROTOCOL, _manifest(), result_json=headline) == [], "no dead end: the diag_sir headline shape"
    assert rm.strata_coverage(PROTOCOL, headline) == [
        "`mean_final_size_given_major` is reported for 2 of the protocol's 10 settings"]
    assert "reported for 2 of the protocol's 10 settings" in " ".join(ms.coverage_gaps(PROTOCOL, [headline], {}))


def test_strata_over_one_axis_are_held_to_every_trial_fi_ran_in_them() -> None:
    per_r0 = {"by_R0": {"R0=1.5": _stratum(150, 2000), "R0=3.0": _stratum(1700, 2000)}}
    assert rm.problems(PROTOCOL, _manifest(), result_json=per_r0) == []
    assert rm.strata_coverage(PROTOCOL, per_r0) == [], "the two R0 strata cover all ten settings"
    wrong_total = {"by_R0": {"R0=1.5": _stratum(150, 400), "R0=3.0": _stratum(1700, 2000)}}
    (found,) = rm.problems(PROTOCOL, _manifest(), result_json=wrong_total)
    assert "`p_outbreak_total` says 400 trial(s), but FI ran 2000" in found


def test_each_stratum_s_values_are_exactly_its_successes() -> None:
    short = {"by_R0": {"R0=1.5": _stratum(150, 2000, n_values=15), "R0=3.0": _stratum(1700, 2000)}}
    (found,) = rm.problems(PROTOCOL, _manifest(), result_json=short)
    assert "lists 15 value(s) but `p_outbreak_count` says 150" in found
    assert found in rm.analysis_output_problems(PROTOCOL, _manifest(), short)
    no_count = {"by_R0": {"R0=1.5": {"mean_final_size_given_major_values": [0.5] * 3}, "R0=3.0": _stratum(1700, 2000)}}
    assert rm.problems(PROTOCOL, _manifest(), result_json=no_count), "a stratum with values and no count is caught"


def test_a_flat_single_stratum_result_still_passes() -> None:
    assert rm.problems(PROTOCOL, _manifest(), result_json=_stratum(1850, 4000)) == []
    thin = _stratum(31, 400)  # a flat result that counts a tenth of the trials, with no stratum named
    (found,) = rm.problems(PROTOCOL, _manifest(), result_json=thin)
    assert "counts only 400 trial(s)" in found and "keyed like `R0=1.5`" in found


def test_the_values_of_a_conditional_mean_must_be_one_quantity_the_trials_returned() -> None:
    recorded = {"final_size": Counter({0.5: 40, 0.9: 10}), "peak": Counter({7.0: 50})}
    means = {"mean_final_size_given_major"}
    honest = {"mean_final_size_given_major_values": [0.5] * 30 + [0.9] * 5}
    assert reported_values_not_run(recorded, honest, given_means=means) == []
    made_up = {"mean_final_size_given_major_values": [0.5] * 30 + [0.77]}
    (found,) = reported_values_not_run(recorded, made_up, given_means=means)
    assert "not all values of any one quantity" in found
    assert reported_values_not_run(recorded, made_up) == [], "only a declared conditional mean is held to it"
