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
from core.trial_runner import given_values_not_run

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
        "`mean_final_size_given_major` is reported for 2 of the protocol's 10 settings, so it describes those "
        "settings only, not the whole design"]
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
    assert "lists 15 trial(s) but `p_outbreak_count` says 150" in found
    assert found in rm.analysis_output_problems(PROTOCOL, _manifest(), short, trial_mode=True)
    no_count = {"by_R0": {"R0=1.5": {"mean_final_size_given_major_values": [0.5] * 3}, "R0=3.0": _stratum(1700, 2000)}}
    assert rm.problems(PROTOCOL, _manifest(), result_json=no_count), "a stratum with values and no count is caught"


def test_a_flat_single_stratum_result_still_passes() -> None:
    assert rm.problems(PROTOCOL, _manifest(), result_json=_stratum(1850, 4000)) == []
    thin = _stratum(31, 400)  # a flat result that counts a tenth of the trials, with no stratum named
    (found,) = rm.problems(PROTOCOL, _manifest(), result_json=thin)
    assert "counts only 400 trial(s)" in found and "keyed like `R0=1.5`" in found


def _no_problems(result: dict[str, Any]) -> list[str]:
    return rm.problems(PROTOCOL, _manifest(), result_json=result)


def test_a_stratum_of_the_grid_must_say_how_many_trials_it_covers() -> None:
    no_total = {"by_R0": {"R0=1.5": {"p_outbreak_count": 150, "mean_final_size_given_major_values": [0.5] * 150},
                          "R0=3.0": _stratum(1700, 2000)}}
    (found,) = _no_problems(no_total)
    assert "gives `p_outbreak_count` but no `p_outbreak_total`" in found
    assert found in rm.analysis_output_problems(PROTOCOL, _manifest(), no_total)


def test_a_mapping_that_names_no_stratum_beside_ones_that_do_is_still_counted_as_a_whole() -> None:
    mixed = {"by": {"R0=1.5,N=500": _stratum(31, 400), "junk": _stratum(5, 5)}}
    (found,) = _no_problems(mixed)
    assert "counts only 5 trial(s)" in found and "where no stratum is named" in found


def test_a_stratum_the_grid_does_not_have_is_named() -> None:
    (found,) = _no_problems({"by_R0": {"R0=9.9": _stratum(3000, 4000)}})
    assert "`R0=9.9` names a value the protocol's grid does not have" in found
    (found,) = _no_problems({"by_R0": {"beta=0.1": _stratum(3000, 4000)}})
    assert "names beta, which the protocol's grid does not have" in found


def test_a_count_larger_than_its_total_and_a_stratum_counting_less_than_one_inside_it() -> None:
    assert any("is larger than `p_outbreak_total`" in f for f in _no_problems(
        {"by_R0": {"R0=1.5": _stratum(2500, 2000, n_values=2500), "R0=3.0": _stratum(1700, 2000)}}))
    nested = {"by_R0": {"R0=1.5": _stratum(10, 2000)}, "by_cell": {"R0=1.5,N=500": _stratum(31, 400)}}
    assert any("counts 10 `p_outbreak` trial(s), fewer than `R0=1.5,N=500`" in f for f in _no_problems(nested))


def test_a_null_for_a_diverged_trial_is_one_of_the_subset() -> None:
    stratum = _stratum(31, 400, n_values=30)
    stratum["mean_final_size_given_major_values"].append(None)
    assert _no_problems({"by": {"R0=1.5,N=500": stratum, "R0=3.0,N=500": _stratum(300, 400)}}) == []


def test_a_short_list_is_the_analysis_to_fix_only_when_fi_ran_the_trials() -> None:
    short = {"by_R0": {"R0=1.5": _stratum(150, 2000, n_values=15), "R0=3.0": _stratum(1700, 2000)}}
    (found,) = _no_problems(short)
    assert found in rm.analysis_output_problems(PROTOCOL, _manifest(), short, trial_mode=True)
    assert found not in rm.analysis_output_problems(PROTOCOL, _manifest(), short),         "under the older contract the simulation's own record is what is in doubt"


# --- the values themselves, against FI's own record of each setting ---------------------------------------------------

def _per_cell() -> dict[str, dict[str, Counter]]:
    out: dict[str, dict[str, Counter]] = {}
    for r in GRID["R0"]:
        for n in GRID["N"]:
            size = {1.5: 0.3, 3.0: 0.9}[r] * n
            out[f"R0={r},N={n}"] = {"final_size": Counter({f"{size:.12g}": 10, "1": 390})}
    return out


def test_a_stratum_s_values_are_its_own_settings_values_or_a_fraction_of_them() -> None:
    raw = {"by": {"R0=1.5,N=500": {"mean_final_size_given_major_values": [150.0] * 10}}}
    assert given_values_not_run(PROTOCOL, _per_cell(), raw) == []
    fraction = {"by": {"R0=1.5,N=500": {"mean_final_size_given_major_values": [0.3] * 10}}}
    assert given_values_not_run(PROTOCOL, _per_cell(), fraction) == [], "final size divided by N is allowed"
    over_r0 = {"by_R0": {"R0=1.5": {"mean_final_size_given_major_values": [0.3] * 50}}}
    assert given_values_not_run(PROTOCOL, _per_cell(), over_r0) == [], "a stratum over N pools its settings' fractions"


def test_a_made_up_fraction_or_a_value_borrowed_from_another_setting_is_caught() -> None:
    made_up = {"by": {"R0=1.5,N=500": {"mean_final_size_given_major_values": [0.31] * 10}}}
    (found,) = given_values_not_run(PROTOCOL, _per_cell(), made_up)
    assert "never produced in those settings" in found and rm.DERIVED in found
    borrowed = {"by": {"R0=1.5,N=500": {"mean_final_size_given_major_values": [450.0] * 10}}}  # R0=3.0's value
    assert given_values_not_run(PROTOCOL, _per_cell(), borrowed), "another setting's values are not this one's"
    suffixed = {"mean_final_size_given_major_R0_1_5_values": [0.31] * 10}
    assert given_values_not_run(PROTOCOL, _per_cell(), suffixed), "a list with the stratum in its name is checked too"
    assert rm.DERIVED in rm.analysis_directive(["x"]), "the repair says the same thing the finding does"


def test_only_a_size_axis_may_divide_a_value_and_nothing_multiplies_one() -> None:
    cell = "R0=1.5,N=500"

    def found(values: list[float]) -> list[str]:
        return given_values_not_run(PROTOCOL, _per_cell(), {"by": {cell: {"mean_final_size_given_major_values": values}}})

    assert found([150.0 / 500] * 10) == [], "divided by N (a size) is a fraction"
    assert found([150.0 * 500] * 10), "multiplied by N is not"
    assert found([150.0 * 1.5] * 10), "multiplied by R0 is not"
    assert found([150.0 / 1.5] * 10), "divided by R0 (not a size) is not"


def test_a_value_rounded_by_the_script_is_not_the_trial_value() -> None:
    per = {"R0=1.5,N=500": {"final_size": Counter({"163": 10})}}
    exact = {"by": {"R0=1.5,N=500": {"mean_final_size_given_major_values": [163 / 500] * 10}}}
    assert given_values_not_run(PROTOCOL, per, exact) == []
    rounded = {"by": {"R0=1.5,N=500": {"mean_final_size_given_major_values": [0.3] * 10}}}
    assert given_values_not_run(PROTOCOL, per, rounded), "0.326 printed as 0.3 is a rounding the check does not accept"


def test_when_the_trials_return_the_subset_the_count_is_fi_s_own() -> None:
    per = {"R0=1.5,N=500": {"final_size": Counter({"150": 10, "1": 390}),
                            "p_outbreak": Counter({"1": 10, "0": 390})}}
    honest = {"by": {"R0=1.5,N=500": {"p_outbreak_count": 10, "mean_final_size_given_major_values": [150.0] * 10}}}
    assert given_values_not_run(PROTOCOL, per, honest) == []
    picked = {"by": {"R0=1.5,N=500": {"p_outbreak_count": 4, "mean_final_size_given_major_values": [150.0] * 4}}}
    (found,) = given_values_not_run(PROTOCOL, per, picked)
    assert "returned `p_outbreak` = 1 for 10 of them" in found


def test_a_list_with_the_stratum_in_its_name_is_held_to_its_count_too() -> None:
    result = {**_stratum(1850, 4000), "p_outbreak_R0_1_5_count": 31,
              "mean_final_size_given_major_R0_1_5_values": [0.5] * 10}
    assert any("`mean_final_size_given_major_R0_1_5_values` lists 10 trial(s)" in f for f in _no_problems(result))
    missing = {**_stratum(1850, 4000), "mean_final_size_given_major_R0_1_5_values": [0.5] * 10}
    assert any("has no `p_outbreak_R0_1_5_count`" in f for f in _no_problems(missing))
