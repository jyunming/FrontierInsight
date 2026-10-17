"""The checks that read a run's numbers beyond the study itself understand what each run compares inside it
(``protocol.in_each_run``, core/in_run.py): the constant-value check, the pilot's flat-output
check. A number per run is judged over the settings, a number per value over the settings and values."""
from __future__ import annotations

from typing import Any

from core import degenerate_results as dr
from core import flat_output, in_run, reported_settings as rs, run_estimate

INSIDE = {"procedure": ["a", "b", "c"]}


def _cells() -> list[dict[str, Any]]:
    """A trials.json as FI writes it for 4 settings x 3 values: ``total`` per run, ``power`` per value."""
    def metric(vals: list[float]) -> dict[str, Any]:
        return {"values": vals, "trials": list(range(len(vals))), "count": len(vals), "total": sum(vals), "non_finite": 0}

    out = []
    for m in (10, 20, 30, 40):
        out.append({"cell": {"m": m}, "key": f"m={m}", "metrics": {"total": metric([float(m), m + 0.1])},
                    "compared_in_each_run": ["procedure=a", "procedure=b", "procedure=c"]})
        for p in "abc":
            out.append({"cell": {"m": m, "procedure": p}, "key": f"m={m},procedure={p}", "level": {"procedure": p},
                        "metrics": {"power": metric([0.0, 0.0, 0.0, 0.0])}})
    return out


def _cells_with(power: Any) -> list[dict[str, Any]]:
    out = _cells()
    for c in out:
        if "level" in c:
            vals = power(c["cell"])
            c["metrics"]["power"] = {**c["metrics"]["power"], "values": vals, "count": len(vals)}
    return out


def test_the_constant_check_judges_each_number_over_the_records_that_hold_it() -> None:
    def cells(power: Any, other: Any) -> list[dict[str, Any]]:
        out = _cells_with(power)
        for c in out:
            if "level" in c:
                c["metrics"]["other"] = {**c["metrics"]["power"], "values": other(c["cell"])}
        return rs.cells_of({"cells": out})

    reported = {"power", "other", "total"}
    # a number per value that never changes while another per value does: caught, though only the value records hold it
    found = dr.quantity_problems(cells(lambda c: [0.3, 0.3], lambda c: [c["m"] * 1.0] * 2), reported)
    assert [(q.name, q.kind) for q in found] == [("power", "constant")]
    # a number per value that moves with the setting, and a number per run that moves: fine
    assert dr.quantity_problems(cells(lambda c: [c["m"] * 0.1] * 2, lambda c: [c["m"] * 1.0] * 2), reported) == []
    # a number that only differs between the values is moving too
    assert dr.quantity_problems(cells(lambda c: [ord(c["procedure"]) * 1.0] * 2, lambda c: [ord(c["procedure"]) * 2.0] * 2),
                                reported) == []


def test_a_plain_study_is_one_group_as_it_was() -> None:
    cells = [{"cell": {"m": m}, "metrics": {"x": [1.0, 2.0]}} for m in (1, 2, 3)]
    assert in_run.groups(cells) == [cells]
    assert in_run.groups([]) == [[]]
    assert dr.quantity_problems(cells, {"x"}) == []


# --- the pilot ----------------------------------------------------------------------------------------------------------


class _Cell:
    def __init__(self, cell: dict[str, Any], values: list[dict[str, Any]], levels: list[dict[str, Any]]) -> None:
        self.cell, self.levels = cell, levels
        self.rows = [{"status": "ok", "values": v} for v in values]


def test_flat_output_judges_a_number_per_run_and_a_number_per_value_each_over_its_own_cells() -> None:
    levels = in_run.level_cells(INSIDE)
    flat_all = [_Cell({"m": m}, [{"total": 3.0, "power": {"procedure=a": 1.0, "procedure=b": 1.0, "procedure=c": 1.0}}] * 2,
                      levels) for m in (1, 2, 3)]
    seen = flat_output.observed_flat_sweep(flat_all)
    assert seen is not None and seen.names == ["power", "total"] and seen.varied == ["m", "procedure"]
    moves = [_Cell({"m": m}, [{"total": 3.0, "power": {"procedure=a": m * 1.0, "procedure=b": 1.0, "procedure=c": 1.0}}] * 2,
                   levels) for m in (1, 2, 3)]
    assert flat_output.observed_flat_sweep(moves) is None       # a number per value moves: not flat


def test_the_pilot_counts_settings_not_values_and_names_a_level_that_reads_as_a_number() -> None:
    obs = [({"m": m}, {"total": 1.0}) for m in (1, 2)] + [({"m": m, "procedure": p}, {"power": 1.0}) for m in (1, 2) for p in "ab"]
    assert run_estimate._setting_count(obs) == 6 and run_estimate._setting_count(obs, ["procedure"]) == 2
    assert in_run.parse_level_key("method=nan,T=300.5") == {"method": "nan", "T": 300.5}
