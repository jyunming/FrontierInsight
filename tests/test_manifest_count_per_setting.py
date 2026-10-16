"""Each setting's list held to that setting's own record: exactly its values and its undefined trials, and only values its
own trials produced.

The cases are the review's probes (C:/fi_compare/reviews/scratch_size: probe_count.py, probe_copy.py, probe_s2.py) on the
r14b and s2 fixtures and on small synthetic studies. Where a case is a copy of another setting's list, the values are
distinct random numbers: with the fixtures' stand-in value 0.5 everywhere, a copy cannot be told from the setting's own
list by any check (the same count, the same numbers).
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from core import run_manifest as rm
from tests.test_manifest_count_r14b import _lists

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "run_size"


def _fixture(name: str) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    fx = json.loads((FIXTURES / name / "manifest_check.json").read_text(encoding="utf-8"))
    per_cell = {key: {n: Counter({0.5: d, math.nan: u}) for n, (d, u) in by.items()}
                for key, by in fx["recorded_by_cell"].items()}
    return fx["protocol"], {"successful_per_cell": fx["successful_per_cell"]}, _lists(fx["result_json"]), per_cell


def _pooled(per_cell: dict[str, Any]) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for by in per_cell.values():
        for name, c in by.items():
            out.setdefault(name, Counter()).update(c)
    return out


def _found(protocol: dict[str, Any], manifest: dict[str, Any], rj: Any, per_cell: dict[str, Any],
           metric: str | None = None) -> list[str]:
    found = [s for s, _ in rm._value_count_findings(protocol, manifest, rj, trial_mode=True, recorded=_pooled(per_cell),
                                                    recorded_by_cell=per_cell)]
    return [s for s in found if metric is None or f"`{metric}`" in s]


# ---- r14b: by_m/<m>/by_pi1/<pi1>/by_method/<procedure> ---------------------------------------------------------------


def test_r14b_and_s2_still_pass() -> None:
    assert _found(*_fixture("r14b_fdr")) == []
    assert _found(*_fixture("s2_fdr")) == []


def test_one_settings_list_left_out_is_caught() -> None:
    protocol, manifest, rj, per_cell = _fixture("r14b_fdr")
    rj["by_m"]["20"]["by_pi1"]["0.05"]["by_method"]["bonferroni"].pop("fdr_values")
    found = _found(protocol, manifest, rj, per_cell, "fdr")
    assert found and "m=20,pi1=0.05,procedure=bonferroni" in found[0], found


def test_lists_cut_short_at_a_few_settings_are_caught() -> None:
    protocol, manifest, rj, per_cell = _fixture("r14b_fdr")
    for pi1 in ("0.05", "0.2"):
        for p in ("uncorrected", "bonferroni", "benjamini_hochberg"):
            d = rj["by_m"]["20"]["by_pi1"][pi1]["by_method"][p]
            d["fdr_values"] = d["fdr_values"][:1000]  # 3000 of 40,500 trials gone
    found = _found(protocol, manifest, rj, per_cell, "fdr")
    assert found and "1000 value(s)" in found[0] and "1500" in found[0], found


def test_the_same_list_under_a_second_key_of_its_setting_is_not_counted_twice() -> None:
    protocol, manifest, rj, per_cell = _fixture("r14b_fdr")
    e = rj["by_m"]["20"]["by_pi1"]["0.05"]
    e["cells"] = {"m=20,pi1=0.05,procedure=bonferroni": {"fdr_values": list(e["by_method"]["bonferroni"]["fdr_values"])}}
    assert _found(protocol, manifest, rj, per_cell) == []
    e["by_method"]["bonferroni"]["fdr_values"] = e["by_method"]["bonferroni"]["fdr_values"] * 2
    found = _found(protocol, manifest, rj, per_cell, "fdr")
    assert found and any(rm.kind_of(s) == rm.FABRICATED for s in _raw(protocol, manifest, rj, per_cell)), found


def _raw(protocol: dict[str, Any], manifest: dict[str, Any], rj: Any, per_cell: dict[str, Any]) -> list[Any]:
    return [s for s, _ in rm._value_count_findings(protocol, manifest, rj, trial_mode=True, recorded=_pooled(per_cell),
                                                   recorded_by_cell=per_cell)]


# ---- s2: by_setting/<m,pi1,procedure> and cells/<m,pi1>/by_procedure/<procedure> -------------------------------------


def _drop(rj: Any, m: int, pi1: float, proc: str, *, nulls: bool = False) -> None:
    for e in (rj["by_setting"][f"m={m},pi1={pi1},procedure={proc}"], rj["cells"][f"m={m},pi1={pi1}"]["by_procedure"][proc]):
        e.pop("power_values", None)
        e["power"] = None
        if nulls:
            e["power_values"] = [None] * 10000


def test_power_null_at_a_setting_that_has_values_is_caught() -> None:
    protocol, manifest, rj, per_cell = _fixture("s2_fdr")
    _drop(rj, 20, 0.2, "bonferroni")
    found = _found(protocol, manifest, rj, per_cell, "power")
    assert found and "m=20,pi1=0.2,procedure=bonferroni" in found[0], found


def test_the_values_given_as_nulls_are_caught() -> None:
    protocol, manifest, rj, per_cell = _fixture("s2_fdr")
    _drop(rj, 20, 0.2, "bonferroni", nulls=True)
    assert _found(protocol, manifest, rj, per_cell, "power")


def test_nulls_or_a_count_at_the_settings_with_no_value_are_allowed() -> None:
    protocol, manifest, rj, per_cell = _fixture("s2_fdr")
    for k in rj["by_setting"]:
        if "pi1=0.0," in k:
            rj["by_setting"][k]["power_values"] = [None] * 10000
    assert _found(protocol, manifest, rj, per_cell) == []
    protocol, manifest, rj, per_cell = _fixture("s2_fdr")
    for k in rj["by_setting"]:
        if "pi1=0.0," in k:
            rj["by_setting"][k]["power_undefined"] = 10000
    assert _found(protocol, manifest, rj, per_cell) == []


def _with_undefined_at_a_setting_with_values(per_cell: dict[str, Any]) -> str:
    key = "m=20,pi1=0.05,procedure=bonferroni"
    per_cell[key]["power"][math.nan] += 100
    per_cell[key]["power"][0.5] -= 100
    return key


def test_each_setting_is_held_to_its_own_undefined_trials() -> None:
    """The nulls an analysis may give at a setting with no value never stand in for those it owes at another."""
    for nulls_at_zero in (False, True):
        protocol, manifest, rj, per_cell = _fixture("s2_fdr")
        key = _with_undefined_at_a_setting_with_values(per_cell)
        for e in (rj["by_setting"][key], rj["cells"]["m=20,pi1=0.05"]["by_procedure"]["bonferroni"]):
            e["power_values"] = [0.5] * 9900
        if nulls_at_zero:
            for k in rj["by_setting"]:
                if "pi1=0.0," in k:
                    rj["by_setting"][k]["power_values"] = [None] * 10000
        found = _found(protocol, manifest, rj, per_cell, "power")
        assert found and key in found[0], (nulls_at_zero, found)
    # with a count at one no-value setting instead
    protocol, manifest, rj, per_cell = _fixture("s2_fdr")
    key = _with_undefined_at_a_setting_with_values(per_cell)
    for e in (rj["by_setting"][key], rj["cells"]["m=20,pi1=0.05"]["by_procedure"]["bonferroni"]):
        e["power_values"] = [0.5] * 9900
    rj["by_setting"]["m=20,pi1=0.0,procedure=bonferroni"]["power_undefined"] = 10000
    assert _found(protocol, manifest, rj, per_cell, "power")
    # reported honestly: the 9,900 values and the 100 undefined at that setting
    for e in (rj["by_setting"][key], rj["cells"]["m=20,pi1=0.05"]["by_procedure"]["bonferroni"]):
        e["power_values"] = [0.5] * 9900 + [None] * 100
    assert _found(protocol, manifest, rj, per_cell, "power") == []


# ---- small studies ---------------------------------------------------------------------------------------------------


def test_three_values_of_ten_left_out_at_one_setting_are_caught() -> None:
    proto = {"grid": {"m": [20, 200]}, "runs_per_setting": 20, "metrics": [{"id": "x", "kind": "mean"}]}
    man = {"successful_per_cell": {"m=20": 20, "m=200": 20}}
    pc = {"m=20": {"x": Counter({0.5: 10, math.nan: 10})}, "m=200": {"x": Counter({0.5: 20})}}
    honest = {"m=20": {"x_values": [0.5] * 10 + [None] * 10, "x_undefined": 10}, "m=200": {"x_values": [0.5] * 20}}
    assert _found(proto, man, honest, pc) == []
    short = {"m=20": {"x_values": [0.5] * 7 + [None] * 10, "x_undefined": 10}, "m=200": {"x_values": [0.5] * 20}}
    assert _found(proto, man, short, pc)


def _estimators(values: dict[str, dict[str, list[float]]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    proto = {"grid": {"n": [50, 500]}, "in_each_run": {"estimator": ["mean", "median", "trimmed"]},
             "runs_per_setting": 100, "metrics": [{"id": "err", "kind": "mean"}]}
    pc = {f"n={n},estimator={e}": {"err": Counter(values[n][e])} for n in values for e in values[n]}
    return proto, {"successful_per_cell": {"n=50": 100, "n=500": 100}}, pc


def _random_values() -> dict[str, dict[str, list[float]]]:
    rng = random.Random(7)
    return {n: {e: [round(rng.random(), 6) for _ in range(100)] for e in ("mean", "median", "trimmed")}
            for n in ("50", "500")}


def _honest(values: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    return {"by_n": {n: {"by_estimator": {e: {"err_values": list(v)} for e, v in by.items()}} for n, by in values.items()}}


def test_a_summary_under_a_key_that_is_a_value_name_does_not_stop_an_honest_analysis() -> None:
    """``summary/mean`` is a summary over the three estimators, not the estimator called mean: its 300 values do not fit
    that one estimator's record, so the key is not read as it, and the block is read as its setting n's."""
    for vals in (_random_values(), {n: {e: [0.5] * 100 for e in ("mean", "median", "trimmed")} for n in ("50", "500")}):
        proto, man, pc = _estimators(vals)
        rj = _honest(vals)
        assert _found(proto, man, rj, pc) == []
        for n in ("50", "500"):
            rj["by_n"][n]["summary"] = {"mean": {"err_values": [x for e in ("mean", "median", "trimmed") for x in vals[n][e]]}}
        assert _found(proto, man, rj, pc) == []


def test_an_estimator_dropped_and_another_ones_values_under_its_name_is_caught() -> None:
    vals = _random_values()
    proto, man, pc = _estimators(vals)
    rj = _honest(vals)
    for n in ("50", "500"):
        rj["by_n"][n]["by_estimator"].pop("trimmed")
        rj["by_n"][n]["stats"] = {"trimmed": {"err_values": list(vals[n]["median"])}}
    assert _found(proto, man, rj, pc, "err")
    rj = _honest(vals)
    rj["by_n"]["500"]["by_estimator"].pop("trimmed")
    assert _found(proto, man, rj, pc, "err")


def test_a_copy_of_one_procedures_list_under_anothers_name_is_numbers_that_setting_never_produced() -> None:
    rng = random.Random(1)
    vals = {p: [round(rng.random(), 6) for _ in range(100)] for p in "abc"}
    proto = {"grid": {"n": [50]}, "in_each_run": {"procedure": ["a", "b", "c"]}, "runs_per_setting": 100,
             "metrics": [{"id": "err", "kind": "mean"}]}
    pc = {f"n=50,procedure={p}": {"err": Counter(vals[p])} for p in "abc"}
    man = {"successful_per_cell": {"n=50": 100}}
    honest = {"by_n": {"50": {"by_procedure": {p: {"err_values": list(vals[p])} for p in "abc"}}}}
    assert _found(proto, man, honest, pc) == []
    copied = copy.deepcopy(honest)
    copied["by_n"]["50"]["by_procedure"]["b"]["err_values"] = list(vals["a"])
    found = _raw(proto, man, copied, pc)
    assert found and any(rm.kind_of(s) == rm.FABRICATED and "procedure=b" in s for s in found), found


def test_the_repair_request_names_the_settings_left_short() -> None:
    protocol, manifest, rj, per_cell = _fixture("r14b_fdr")
    rj["by_m"]["20"]["by_pi1"]["0.05"]["by_method"]["bonferroni"].pop("fdr_values")
    found = rm.differences(protocol, manifest, rj, recorded=_pooled(per_cell), recorded_by_cell=per_cell)
    fdr = [d for d in found if d["metric"] == "fdr"]
    assert fdr and [s["setting"] for s in fdr[0]["short"]] == ["m=20,pi1=0.05,procedure=bonferroni"]
    asked = rm.difference_request(found)
    assert "under `m=20,pi1=0.05,procedure=bonferroni` it lists 0 value(s)" in asked and "list every one" in asked
    assert "leaves out trials of one setting" in rm.plain_difference(fdr)
