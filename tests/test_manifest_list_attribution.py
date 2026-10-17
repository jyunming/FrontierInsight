"""Which trials a list of values belongs to, and numbers beside a list that are not its mean.

The review's probes on the real r14b record (C:/fi_compare/reviews/scratch_size2), rebuilt on a small study with distinct
values so a copy is never the same numbers by chance:

* a list under a key that names one procedure must be that procedure's trials at that setting. Bonferroni's values under
  ``stats/benjamini_hochberg`` (F3, F9) or under ``by_method/benjamini_hochberg`` beside a ``notes`` key (F2) were read at
  the coarser setting, whose record holds them, and passed or were only "unaccounted". A summary over all three
  procedures under such a key (K1) is that setting's whole record and still passes.
* a ``<m>`` printed beside its ``<m>_values`` is that list's mean (F8: a made-up 0.001 beside honest values passed).
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from typing import Any

from core import run_manifest as rm
from core import trial_runner as tr

GRID = {"m": [20, 200], "procedure": ["uncorrected", "bonferroni", "benjamini_hochberg"]}
PROTO = {"grid": GRID, "runs_per_setting": 50, "metrics": [{"id": "fdr", "kind": "mean"}, {"id": "power", "kind": "mean"}]}
MANIFEST = {"successful_per_cell": {f"m={m},procedure={p}": 50 for m in GRID["m"] for p in GRID["procedure"]}}
PROCS = GRID["procedure"]


def _values(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [round(rng.random(), 9) for _ in range(50)]


def _study() -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
    """Each setting's fdr and power, and an extra output x1_power at m=200 only (r13's probe)."""
    vals: dict[str, dict[str, list[float]]] = {}
    per_cell: dict[str, Any] = {}
    for i, m in enumerate(GRID["m"]):
        for j, p in enumerate(PROCS):
            key = f"m={m},procedure={p}"
            vals[key] = {"fdr": _values(100 * i + j), "power": _values(1000 + 100 * i + j)}
            if m == 200:
                vals[key]["x1_power"] = _values(5000 + j)
            per_cell[key] = {n: Counter(tr._value_key(x) for x in v) for n, v in vals[key].items()}
    return vals, per_cell


def _honest(vals: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_m": {}}
    for m in GRID["m"]:
        out["by_m"][str(m)] = {"by_method": {
            p: {"fdr": sum(vals[f"m={m},procedure={p}"]["fdr"]) / 50, "fdr_values": list(vals[f"m={m},procedure={p}"]["fdr"]),
                "power_values": list(vals[f"m={m},procedure={p}"]["power"])} for p in PROCS}}
    return out


def _found(rj: Any, per_cell: dict[str, Any]) -> list[Any]:
    recorded = tr.pooled(per_cell)
    return ([s for s, _ in rm._value_count_findings(PROTO, MANIFEST, rj, trial_mode=True, recorded=recorded,
                                                    recorded_by_cell=per_cell)]
            + tr.reported_values_not_run(recorded, rj) + tr.given_values_not_run(PROTO, per_cell, rj))


def _fabricated(found: list[Any]) -> list[Any]:
    return [f for f in found if rm.kind_of(f) == rm.FABRICATED]


def _at(rj: dict[str, Any], m: str = "20") -> dict[str, Any]:
    return rj["by_m"][m]["by_method"]


def test_the_honest_study_passes() -> None:
    vals, per_cell = _study()
    assert _found(_honest(vals), per_cell) == []


def test_another_procedures_values_beside_a_notes_key_are_numbers_that_setting_never_produced() -> None:
    """F2: a ``notes`` key in ``by_method`` turned the grouping off and the copy fell back to the setting's pooled record."""
    vals, per_cell = _study()
    rj = _honest(vals)
    _at(rj)["notes"] = "three procedures on the same data"
    _at(rj)["benjamini_hochberg"]["fdr_values"] = list(_at(rj)["bonferroni"]["fdr_values"])
    found = _fabricated(_found(rj, per_cell))
    assert found and any("benjamini_hochberg" in f for f in found), _found(rj, per_cell)


def test_another_procedures_values_under_a_lone_key_naming_a_procedure_are_fabricated() -> None:
    """F3 and F9: Bonferroni's values (and mean) under ``stats/benjamini_hochberg``, beside the honest lists or instead
    of BH's own."""
    vals, per_cell = _study()
    for drop in (False, True):
        rj = _honest(vals)
        e = rj["by_m"]["20"]
        bon = e["by_method"]["bonferroni"]
        e["stats"] = {"benjamini_hochberg": {"fdr_values": list(bon["fdr_values"]), "fdr": bon["fdr"]}}
        if drop:
            e["by_method"]["benjamini_hochberg"].pop("fdr_values")
        found = _fabricated(_found(rj, per_cell))
        assert found and any("stats/benjamini_hochberg" in f for f in found), (drop, _found(rj, per_cell))


def test_a_summary_over_every_procedure_under_a_procedures_name_still_passes() -> None:
    """K1: the whole setting's record under a key that happens to be a value's name is a summary, not a copy."""
    vals, per_cell = _study()
    rj = _honest(vals)
    for m in ("20", "200"):
        everything = [x for p in PROCS for x in vals[f"m={m},procedure={p}"]["fdr"]]
        rj["by_m"][m]["summary"] = {"benjamini_hochberg": {"fdr_values": everything}}
    assert _found(rj, per_cell) == []
    # A part of it is not the whole: still another setting's values under that name.
    rj["by_m"]["20"]["summary"]["benjamini_hochberg"]["fdr_values"] = everything[:120]
    assert _fabricated(_found(rj, per_cell))


def test_a_mean_beside_its_values_that_is_not_their_mean_is_fabricated() -> None:
    """F8: every list honest, one printed mean made up."""
    vals, per_cell = _study()
    rj = _honest(vals)
    _at(rj)["benjamini_hochberg"]["fdr"] = 0.001
    found = _fabricated(_found(rj, per_cell))
    assert found and any("0.001" in f and "benjamini_hochberg" in f for f in found), _found(rj, per_cell)
    # rounded to four places, or computed in another order, it is the same mean
    rj = _honest(vals)
    _at(rj)["benjamini_hochberg"]["fdr"] = round(_at(rj)["benjamini_hochberg"]["fdr"], 4)
    _at(rj)["bonferroni"]["fdr_mean"] = sum(reversed(_at(rj)["bonferroni"]["fdr_values"])) / 50
    assert _found(rj, per_cell) == []
    # a metric that is the spread of the per-trial values (s2 kinetics: SD(Ea_hat)/Ea_true beside each Ea_hat/Ea_true)
    _at(rj)["uncorrected"]["fdr"] = statistics.stdev(_at(rj)["uncorrected"]["fdr_values"])
    assert _found(rj, per_cell) == []
