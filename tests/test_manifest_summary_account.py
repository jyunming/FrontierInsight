"""A setting whose per-trial list the analysis did not print but whose summary it did.

Replayed on the s7 false-discovery quest: the analysis printed ``power_gain_values`` for one setting (m=2000, pi1=0.2) and,
for the six other settings that have values, only ``by_setting/<setting>/power_gain/{mean, ci95, se}``. FI's record of those
trials holds all the values, so the printed mean (and standard error) can be computed from it: equal to the precision of
FI's record, the analysis used the same trials and the setting is accounted for; different, it did not, and a mean
rounded to fewer digits is not FI's number either (a nominal level printed beside a setting fell within the rounding of
its trials' mean). A
setting with neither a list nor a summary stays short, and the three pi1 = 0 settings (power is undefined there by
definition) owed nothing before and still owe nothing.

The cases are small synthetic studies of that shape (the quest itself is replayed in the PR's bench).
"""

from __future__ import annotations

import copy
import math
import random
from collections import Counter
from typing import Any

from core import run_manifest as rm

GRID = {"m": [20, 200], "pi1": [0.0, 0.2]}
RUNS = 60
WITH_VALUES = [(20, 0.2), (200, 0.2)]


def _study() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rnd = random.Random(7)
    protocol = {"grid": GRID, "runs_per_setting": RUNS,
                "metrics": [{"id": "gain", "estimand": "mean power gain", "kind": "mean", "unit": "one family",
                             "cluster": None, "paired": False, "family": "f"}]}
    per_cell: dict[str, Any] = {}
    manifest = {"successful_per_cell": {}}
    for m in GRID["m"]:
        for pi1 in GRID["pi1"]:
            key = f"m={m},pi1={pi1}"
            manifest["successful_per_cell"][key] = RUNS
            if pi1 == 0.0:
                per_cell[key] = {"gain": Counter({math.nan: RUNS})}  # no real effect: no power, by definition
            else:
                per_cell[key] = {"gain": Counter({round(rnd.random() * 0.5, 6): 1 for _ in range(RUNS)})}
    return protocol, manifest, per_cell, {}


def _values(per_cell: dict[str, Any], key: str) -> list[float]:
    return [float(v) for v, n in per_cell[key]["gain"].items() for _ in range(n)]


def _printed(per_cell: dict[str, Any]) -> dict[str, Any]:
    """The analysis as s7's wrote it: one setting's values listed, the rest as mean / se per setting."""
    by_setting: dict[str, Any] = {}
    for m, pi1 in WITH_VALUES:
        key = f"m={m},pi1={pi1}"
        v = _values(per_cell, key)
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
        by_setting[key] = {"gain": {"mean": mean, "se": sd / math.sqrt(len(v))}}
    for m in GRID["m"]:
        by_setting[f"m={m},pi1=0.0"] = {"gain": None}
    return {"gain_values": _values(per_cell, "m=200,pi1=0.2"), "by_setting": by_setting}


def _found(rj: Any, per_cell: dict[str, Any]) -> list[str]:
    protocol, manifest, _, _ = _study()
    pooled: dict[str, Counter] = {}
    for by in per_cell.values():
        for name, c in by.items():
            pooled.setdefault(name, Counter()).update(c)
    found = rm._value_count_findings(protocol, manifest, rj, trial_mode=True, recorded=pooled, recorded_by_cell=per_cell)
    return [str(s) for s, _ in found if "`gain`" in str(s)]


def test_summaries_equal_to_the_records_account_for_the_settings() -> None:
    _, _, per_cell, _ = _study()
    assert _found(_printed(per_cell), per_cell) == []


def test_a_setting_with_neither_list_nor_summary_is_still_left_out() -> None:
    _, _, per_cell, _ = _study()
    rj = _printed(per_cell)
    del rj["by_setting"]["m=20,pi1=0.2"]
    found = _found(rj, per_cell)
    assert found and "m=20,pi1=0.2" in found[0], found


def test_a_mean_computed_from_fewer_trials_is_a_disagreement() -> None:
    _, _, per_cell, _ = _study()
    rj = _printed(per_cell)
    v = sorted(_values(per_cell, "m=20,pi1=0.2"))[6:]  # the lowest tenth never reached the summary
    rj["by_setting"]["m=20,pi1=0.2"]["gain"]["mean"] = sum(v) / len(v)
    found = _found(rj, per_cell)
    assert found and "m=20,pi1=0.2" in found[0] and "not the mean" in found[0], found


def test_the_mean_of_another_setting_is_a_disagreement() -> None:
    _, _, per_cell, _ = _study()
    rj = _printed(per_cell)
    rj["by_setting"]["m=20,pi1=0.2"]["gain"]["mean"] = rj["by_setting"]["m=200,pi1=0.2"]["gain"]["mean"]
    found = _found(rj, per_cell)
    assert found and "m=20,pi1=0.2" in found[0] and "not the mean" in found[0], found


def test_a_trial_count_below_the_records_is_a_disagreement_and_the_true_count_is_not() -> None:
    _, _, per_cell, _ = _study()
    rj = _printed(per_cell)
    rj["by_setting"]["m=20,pi1=0.2"]["gain"]["n"] = RUNS
    assert _found(rj, per_cell) == []
    bad = copy.deepcopy(rj)
    bad["by_setting"]["m=20,pi1=0.2"]["gain"]["n"] = RUNS - 6
    found = _found(bad, per_cell)
    assert found and "number of trials" in found[0], found


def test_the_allowance_is_the_records_precision_never_the_printed_digits() -> None:
    assert rm._record_allowance(0.0088) < 1e-12  # 0.0088 stands for 0.0088, not for 0.00875 to 0.00885
    assert rm._record_allowance(25) == 0.0
    assert rm._record_allowance(0.40572700000000006) < 1e-10


def test_a_mean_printed_with_fewer_digits_is_not_fis_number() -> None:
    _, _, per_cell, _ = _study()
    rj = _printed(per_cell)
    rj["by_setting"]["m=20,pi1=0.2"]["gain"]["mean"] = round(rj["by_setting"]["m=20,pi1=0.2"]["gain"]["mean"], 4)
    found = _found(rj, per_cell)
    assert found and "m=20,pi1=0.2" in found[0] and "not the mean" in found[0], found


def test_a_coarse_number_within_the_rounding_of_the_mean_is_not_the_mean() -> None:
    # A one-digit number printed under the setting (a nominal level, another set's rounded mean) lay within half a unit
    # of its last digit of the setting's mean, and was taken as the summary of its trials.
    _, _, per_cell, _ = _study()
    rj = _printed(per_cell)
    mean = rj["by_setting"]["m=20,pi1=0.2"]["gain"]["mean"]
    coarse = round(mean, 1)
    assert coarse != mean and abs(coarse - mean) <= 0.05
    rj["by_setting"]["m=20,pi1=0.2"]["gain"] = {"mean": coarse}
    found = _found(rj, per_cell)
    assert found and "m=20,pi1=0.2" in found[0] and "not the mean" in found[0], found


def test_an_exact_mean_with_few_digits_is_still_the_mean() -> None:
    # s7's means of 0/1 values over 5000 trials have few digits (0.0088) and are exactly the mean: still FI's number.
    _, _, per_cell, _ = _study()
    per_cell = {k: ({"gain": Counter({1.0: 44, 0.0: 5000 - 44})} if k == "m=20,pi1=0.2" else v)
                for k, v in per_cell.items()}
    rj = _printed(per_cell)
    rj["by_setting"]["m=20,pi1=0.2"]["gain"] = {"mean": 0.0088}
    assert _found(rj, per_cell) == []


def test_a_summary_of_a_part_of_the_grid_is_not_read_as_the_setting() -> None:
    # `by_m/20` may be one stratum of the m = 20 settings (a live kinetics record printed 400 of an axis value's 2000
    # trials): it accounts for no single setting, so the settings stay short.
    _, _, per_cell, _ = _study()
    rj = _printed(per_cell)
    rj["by_m"] = {"20": {"gain": {"mean": 0.25}}}
    del rj["by_setting"]["m=20,pi1=0.2"]
    found = _found(rj, per_cell)
    assert found and "m=20,pi1=0.2" in found[0], found


def test_a_value_under_a_further_key_inside_the_setting_is_not_the_settings_summary() -> None:
    # Old code: a reference printed beside the setting (`<setting>/reference/gain`, a nominal 0.05) was read as the
    # summary of the setting's trials and called a disagreement with them. FI's own layout puts the summary directly in
    # the setting's mapping; a mapping under a further key holds that key's quantity, whatever the key is called.
    _, _, per_cell, _ = _study()
    rj = _printed(per_cell)
    rj["by_setting"]["m=20,pi1=0.2"]["reference"] = {"gain": 0.05}
    rj["by_setting"]["m=200,pi1=0.2"]["theory"] = {"gain": {"mean": 0.05}}
    assert _found(rj, per_cell) == []
    # Known positive: the setting's own summary is still held to its trials.
    rj["by_setting"]["m=20,pi1=0.2"]["gain"]["mean"] = 0.05
    found = _found(rj, per_cell)
    assert found and "m=20,pi1=0.2" in found[0] and "not the mean" in found[0], found


def test_a_value_under_a_further_key_never_accounts_for_the_setting_even_when_it_equals_the_mean() -> None:
    # Old code: a reference that happened to equal the mean of the setting's trials was taken as their summary, so the
    # setting counted as printed although the analysis printed nothing of its trials.
    _, _, per_cell, _ = _study()
    per_cell = {k: ({"gain": Counter({0.05: RUNS})} if k == "m=20,pi1=0.2" else v) for k, v in per_cell.items()}
    rj = _printed(per_cell)
    rj["by_setting"]["m=20,pi1=0.2"] = {"reference": {"gain": 0.05}}
    found = _found(rj, per_cell)
    assert found and "m=20,pi1=0.2" in found[0] and "not the mean" not in found[0], found
