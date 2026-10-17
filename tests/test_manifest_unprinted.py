"""An analysis that prints no per-trial list for a mean FI holds trial by trial is not "leaving trials out" of FI's record
(s3: both papers said "FI's record of the trials that ran does not match what the analysis reports" while every trial had
its value). No model is called."""

from __future__ import annotations

import copy

from core import run_manifest as rm
from tests.test_manifest_count_s2 import _inputs


def _strip_lists(rj, metric):
    rj = copy.deepcopy(rj)

    def walk(node):
        if isinstance(node, dict):
            for k in [k for k in node if k == f"{metric}_values"]:
                del node[k]
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(rj)
    return rj


def _run(rj):
    protocol, manifest, _rj, recorded, per_cell = _inputs()
    found = rm.problems(protocol, manifest, result_json=rj, trial_mode=True, recorded=recorded, recorded_by_cell=per_cell)
    return found, rm.differences(protocol, manifest, rj, recorded=recorded, recorded_by_cell=per_cell)


def test_no_list_at_all_is_unprinted_and_no_difference_of_the_record() -> None:
    _p, _m, rj, _r, _c = _inputs()
    found, diff = _run(_strip_lists(rj, "power"))
    mine = [f for f in found if "`power`" in f]
    assert mine and all(rm.kind_of(f) == rm.UNPRINTED for f in mine)
    assert not [d for d in diff if d["metric"] == "power"]


def test_a_list_that_does_not_account_for_a_setting_is_still_a_difference() -> None:
    """The same quest with 100 trials of one setting that returned no power, which the analysis does not say."""
    import math
    from collections import Counter

    protocol, manifest, rj, _recorded, per_cell = _inputs()
    key = "m=20,pi1=0.05,procedure=bonferroni"
    by = per_cell[key]["power"]
    by[math.nan] += 100
    by[0.5] -= 100
    rj["by_setting"][key]["power_values"] = rj["by_setting"][key]["power_values"][:-100]
    recorded: dict[str, Counter] = {}
    for b in per_cell.values():
        for name, c in b.items():
            recorded.setdefault(name, Counter()).update(c)
    found = rm.problems(protocol, manifest, result_json=rj, trial_mode=True, recorded=recorded, recorded_by_cell=per_cell)
    diff = rm.differences(protocol, manifest, rj, recorded=recorded, recorded_by_cell=per_cell)
    assert any(d["metric"] == "power" and d["short"] for d in diff)
    assert any(rm.kind_of(f) == rm.UNACCOUNTED for f in found if "`power`" in f)
