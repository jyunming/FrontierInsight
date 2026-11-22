"""A derived result the record could not have moved is not delivered.

* A result a function of the study's worked out along a setting no trial read is missing with a plain reason (FI's record of what
  the code asked for, not the function's text): the same result with the setting read, one that only filters by the setting, a
  constant the model makes while the trials read the setting, a function that never looked at the settings or took them whole
  is kept.

Made-up topic (a colony), on the real analysis path: the frame FI writes is run in a fresh Python.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from core import trial_runner
from tests.test_analysis_frame import RECORD, _derived, _design, _quest, _run

# ---- a result the record could not have moved --------------------------------------------------------------------------

AXIS_FIT = """
def derive_slope_rate(table):
    pts = sorted((table.setting(k)["rate"], table.mean("growth", k)) for k in table.at(dose=1))
    return pts[-1][1] - pts[0][1]
"""
SUMMARY = """
def derive_largest_growth(table):
    return max(table.mean("growth", k) for k in table.keys())
"""


def _constant(record: dict[str, Any], value: float = 0.0) -> dict[str, Any]:
    out = copy.deepcopy(record)
    for c in out["cells"]:
        c["metrics"]["growth"]["values"] = [value] * len(c["metrics"]["growth"]["values"])
    return out


def _derived_of(tmp_path: Path, record: dict[str, Any], bodies: str, ident: str) -> dict[str, Any]:
    design = _design([_derived(ident, inputs=("growth",))])
    quest = _quest(tmp_path, design, bodies, record)
    code, printed, derived, err = _run(quest)
    assert derived and derived["derived"][ident]["status"] == "ok", (code, err)
    return derived


def test_a_result_worked_out_along_a_setting_no_trial_read_is_missing_with_the_reason(tmp_path: Path) -> None:
    derived = _derived_of(tmp_path, {**RECORD, "never_read": ["rate"]}, AXIS_FIT, "slope_rate")
    rec = derived["derived"]["slope_rate"]
    assert rec["value"] is None and "`rate`" in rec["held_back"] and "no trial of the run read" in rec["held_back"]
    assert rec["missing"] == {"": rec["held_back"]}
    assert {"id": "slope_rate", "status": "held_back", "why": rec["held_back"]} in derived["not_computed"]
    assert not derived["complete"]


def test_the_same_result_with_the_setting_read_is_kept(tmp_path: Path) -> None:
    for n, record in enumerate((RECORD, {**RECORD, "never_read": ["colour"]})):  # nothing unread, or a setting it never used
        rec = _derived_of(tmp_path / str(n), record, AXIS_FIT, "slope_rate")["derived"]["slope_rate"]
        assert isinstance(rec["value"], float) and "held_back" not in rec and rec["axes"] == ["rate"]


def test_a_setting_a_function_only_filters_by_is_not_one_it_worked_out_along(tmp_path: Path) -> None:
    """``at(dose=1)`` picks settings that hold a value; the trials never read ``dose``, but the result is worked out along ``rate``."""
    rec = _derived_of(tmp_path, {**RECORD, "never_read": ["dose"]}, AXIS_FIT, "slope_rate")["derived"]["slope_rate"]
    assert isinstance(rec["value"], float) and "held_back" not in rec and rec["axes"] == ["rate"]


@pytest.mark.parametrize("record", [_constant(RECORD), {**_constant(RECORD), "drew": True}])
def test_a_constant_the_model_makes_while_the_trials_read_the_setting_is_kept(tmp_path: Path, record: dict[str, Any]) -> None:
    """Results that are the same at every setting (a conserved quantity) with a function that follows ``rate``, which the trials
    read: nothing in FI's record says the record could not have moved it, so it stays (random trials or not)."""
    rec = _derived_of(tmp_path, record, AXIS_FIT, "slope_rate")["derived"]["slope_rate"]
    assert rec["value"] == 0.0 and "held_back" not in rec


def test_a_fit_along_a_setting_no_trial_read_is_missing_whatever_the_record_holds(tmp_path: Path) -> None:
    flat = {**_constant(RECORD), "never_read": ["rate"]}
    rec = _derived_of(tmp_path, flat, AXIS_FIT, "slope_rate")["derived"]["slope_rate"]
    assert rec["value"] is None and "`rate`" in rec["held_back"]


def test_a_summary_that_never_looked_at_the_settings_stays(tmp_path: Path) -> None:
    rec = _derived_of(tmp_path, {**RECORD, "never_read": ["rate"]}, SUMMARY, "largest_growth")["derived"]["largest_growth"]
    assert isinstance(rec["value"], float) and "held_back" not in rec and rec["axes"] == []


def test_a_function_that_takes_the_settings_whole_makes_no_claim(tmp_path: Path) -> None:
    whole = AXIS_FIT.replace('table.setting(k)["rate"]', 'dict(table.setting(k))["rate"]')
    rec = _derived_of(tmp_path, {**RECORD, "never_read": ["rate"]}, whole, "slope_rate")["derived"]["slope_rate"]
    assert isinstance(rec["value"], float) and rec["axes_all"] is True and "held_back" not in rec


# ---- FI's record of the trials names what no trial read -------------------------------------------------------------------


def _run_of(cell: dict[str, Any], rows: list[dict[str, Any]]) -> Any:
    return trial_runner.CellRun(key=trial_runner.cell_key(cell), cell=cell, planned=len(rows), rows=rows)


def test_the_summary_of_the_trials_carries_the_settings_no_trial_read() -> None:
    def row(i: int, read: list[str]) -> dict[str, Any]:
        return {"trial": i, "status": "ok", "values": {"y": float(i)}, "read": read}

    runs = [_run_of({"a": a, "b": b}, [row(0, ["a"]), row(1, ["a"])]) for a in (1, 2) for b in (10, 20)]
    assert trial_runner.summary_of(runs)["never_read"] == ["b"]
    assert "never_read" not in trial_runner.summary_of([_run_of({"a": a}, [row(0, ["a"])]) for a in (1, 2)])
