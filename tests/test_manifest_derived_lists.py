"""A list drawn from a setting's own record (a bootstrap resample, a duplicated subset) is derived, not fabricated.

The review's probe S7 on the real r14b record put a bootstrap resample of BH's fdr values under
``by_method/benjamini_hochberg/fdr_bootstrap/fdr_values`` and FI called it numbers the trials never produced: a resample
holds some trials' values more than once. The rule is read from the record, not from the key's name: a list whose every
value is one of that output's recorded values at its setting is derived from the record. It is never fabricated and never
counts toward the trials, so a duplicated subset standing in for a setting's list still leaves that setting short (a
counting difference, recorded under ``warn``). A list that is another procedure's whole record at that setting is that
procedure's values, and stays fabricated even when every value also occurs in this procedure's record (fdr at pi1 = 0
is only ever 0 or 1).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from core import run_manifest as rm
from core import trial_runner as tr

GRID = {"m": [20, 200], "procedure": ["uncorrected", "bonferroni", "benjamini_hochberg"]}
PROTO = {"grid": GRID, "runs_per_setting": 50, "metrics": [{"id": "fdr", "kind": "mean"}]}
MANIFEST = {"successful_per_cell": {f"m={m},procedure={p}": 50 for m in GRID["m"] for p in GRID["procedure"]}}
PROCS = GRID["procedure"]


def _study(zero_one: bool = False) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Each setting's fdr: distinct values, or (``zero_one``) only 0 and 1, as fdr where there is no real effect."""
    vals: dict[str, list[float]] = {}
    for i, m in enumerate(GRID["m"]):
        for j, p in enumerate(PROCS):
            rng = random.Random(100 * i + j)
            vals[f"m={m},procedure={p}"] = ([0.0] * (44 + j) + [1.0] * (6 - j) if zero_one
                                            else [round(rng.random(), 9) for _ in range(50)])
    per_cell = {k: {"fdr": Counter(tr._value_key(x) for x in v)} for k, v in vals.items()}
    return vals, per_cell


def _honest(vals: dict[str, list[float]]) -> dict[str, Any]:
    return {"by_m": {str(m): {"by_method": {p: {"fdr_values": list(vals[f"m={m},procedure={p}"])} for p in PROCS}}
                     for m in GRID["m"]}}


def _found(rj: Any, per_cell: dict[str, Any]) -> list[Any]:
    recorded = tr.pooled(per_cell)
    return ([s for s, _ in rm._value_count_findings(PROTO, MANIFEST, rj, trial_mode=True, recorded=recorded,
                                                    recorded_by_cell=per_cell)]
            + tr.reported_values_not_run(recorded, rj))


def _fabricated(found: list[Any]) -> list[Any]:
    return [f for f in found if rm.kind_of(f) == rm.FABRICATED]


def test_a_bootstrap_resample_beside_the_honest_lists_passes() -> None:
    """S7: a resample of each setting's BH values, under a key of its own beside them."""
    vals, per_cell = _study()
    rj = _honest(vals)
    rng = random.Random(0)
    for m in rj["by_m"].values():
        bh = m["by_method"]["benjamini_hochberg"]
        bh["fdr_bootstrap"] = {"fdr_values": [rng.choice(bh["fdr_values"]) for _ in bh["fdr_values"]]}
    assert _found(rj, per_cell) == []


def test_a_duplicated_subset_in_place_of_a_settings_list_is_a_counting_difference_not_fabricated() -> None:
    """F6 without a mean beside it: half of BH's values, each twice, as BH's list at m = 20."""
    vals, per_cell = _study()
    rj = _honest(vals)
    own = rj["by_m"]["20"]["by_method"]["benjamini_hochberg"]
    own["fdr_values"] = own["fdr_values"][:25] * 2
    found = _found(rj, per_cell)
    assert not _fabricated(found), found
    assert any(rm.kind_of(f) == rm.UNACCOUNTED and "m=20,procedure=benjamini_hochberg" in str(f) for f in found), found


def test_another_procedures_whole_record_stays_fabricated_even_when_its_values_occur_in_this_ones() -> None:
    """F1 at pi1 = 0: Bonferroni's 0s and 1s under BH's name, every value also one of BH's."""
    vals, per_cell = _study(zero_one=True)
    rj = _honest(vals)
    assert _found(rj, per_cell) == []
    by = rj["by_m"]["20"]["by_method"]
    by["benjamini_hochberg"]["fdr_values"] = list(by["uncorrected"]["fdr_values"])
    found = _fabricated(_found(rj, per_cell))
    assert found and any("benjamini_hochberg" in str(f) for f in found), _found(rj, per_cell)
