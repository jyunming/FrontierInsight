"""A measurement's numerical settings: FI runs the headline results again at finer values and judges them itself
(core/refinement_check.py). A value lost at a finer level is not converged; a refinement that changed nothing checked
nothing; an error is judged against the result's spread over the study, never against a value near zero."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from core import plan
from core import refinement_check as rc
from core import run_estimate

PROTOCOL: dict[str, Any] = {
    "grid": {"gap_mm": [1.0, 2.0, 3.0, 4.0]},
    "numerical_settings": {"step_mm": {"search": 0.1, "check": [0.05, 0.025]}},
}


def test_a_numerical_setting_is_a_setting_of_the_study_at_its_one_value() -> None:
    out, why = plan.normalize_protocol(dict(PROTOCOL))
    assert why is None, why
    assert out["grid"]["step_mm"] == [0.1]
    again, why = plan.normalize_protocol(out)
    assert why is None and again == out, "reading it twice changes nothing"


@pytest.mark.parametrize("change, words", [
    ({"grid": {"gap_mm": [1.0, 2.0], "step_mm": [0.1, 0.05]}}, "has one value in the study"),
    ({"numerical_settings": {"step_mm": {"search": 0.1, "check": [0.1]}}}, "is not finer than"),
    ({"numerical_settings": {"step_mm": {"search": -1}}}, "positive number"),
    ({"optimisation": {}}, "cannot stand beside `protocol.optimisation`"),
])
def test_what_cannot_be_a_numerical_setting_is_refused(change: dict[str, Any], words: str) -> None:
    _out, why = plan.normalize_protocol({**PROTOCOL, **change})
    assert why and words in why and "`protocol.numerical_settings`" in why, why


def test_the_levels_and_ratios_come_from_the_plan_or_fis_rule() -> None:
    s = rc.settings({"numerical_settings": {"step": {"search": 0.1, "check": [0.05, 0.025]},
                                            "points": {"search": 64, "finer": "larger"}}})
    assert s["step"]["levels"] == [0.1, 0.05, 0.025] and s["step"]["ratios"] == pytest.approx([2.0, 2.0])
    assert s["points"]["levels"] == [64, 128, 256] and s["points"]["from"] == "rule"
    assert rc.settings({"optimisation": {}, "numerical_settings": {"step": {"search": 0.1}}}) == {}


SETTING = {"levels": [0.1, 0.05, 0.025], "ratios": [2.0, 2.0], "tolerance": 0.05}


def test_a_value_lost_at_a_finer_level_is_not_converged() -> None:
    v = rc.judge(SETTING, {"gap=1": [{"width": 51.4}, {"width": float("nan")}, {"width": float("nan")}]},
                 {"width": 30.0})
    assert v["verdict"] == "not_converged" and v["results"][0]["status"] == "lost"


def test_a_refinement_that_changed_nothing_checked_nothing() -> None:
    same = [{"width": 12.0, "depth": 3.0}] * 3
    v = rc.judge(SETTING, {"gap=1": same, "gap=4": same}, {"width": 5.0, "depth": 1.0})
    assert v["verdict"] == "no_effect" and "checked nothing" in v["why"]


def test_a_value_near_zero_is_judged_against_its_spread_not_itself() -> None:
    # 0.02 -> 0.011 -> 0.0105 would be a 90% relative change, but the study's values span 0 to 40
    v = rc.judge(SETTING, {"gap=1": [{"width": 0.02}, {"width": 0.011}, {"width": 0.0105}]}, {"width": 40.0})
    assert v["verdict"] == "converged", v


def test_a_result_that_moves_with_the_step_is_not_converged() -> None:
    v = rc.judge(SETTING, {"gap=1": [{"width": 40.0}, {"width": 30.0}, {"width": 25.0}]}, {"width": 20.0})
    assert v["verdict"] == "not_converged" and v["results"][0]["share_of_spread"] > 0.05


def test_the_settings_run_again_are_where_each_headline_is_lowest_and_highest() -> None:
    cells = [{"cell": {"gap_mm": g, "step_mm": 0.1}, "metrics": {"width": {"values": [w]}, "other": {"values": [1.0]}}}
             for g, w in ((1.0, 5.0), (2.0, 9.0), (3.0, 1.0), (4.0, 7.0))]
    cells.append({"cell": {"gap_mm": 1.0, "step_mm": 0.1, "case": 1}, "level": {"case": 1},
                  "metrics": {"width": {"values": [-100.0]}}})
    picked = rc.pick_cells(cells, {"width"}, most=2)
    assert [c["gap_mm"] for c in picked] == [3.0, 2.0]


@pytest.mark.asyncio
async def test_fi_runs_the_settings_again_and_says_what_it_found(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {}\n", encoding="utf-8")
    (tmp_path / "raw").mkdir()
    cells = [{"cell": {"gap_mm": g, "step_mm": 0.1}, "metrics": {"width": {"values": [10.0 * g]}}}
             for g in (1.0, 2.0, 3.0, 4.0)]
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    asked: list[dict[str, Any]] = []

    async def fake_time_one(executor, python, quest_root, module, cell, **kw):  # noqa: ANN001
        asked.append(dict(cell))
        # the result at the smallest gap is lost at the finest step; every other is the same at every step
        width = float("nan") if cell["gap_mm"] == 1.0 and cell["step_mm"] == 0.025 else 10.0 * cell["gap_mm"]
        return run_estimate.Probe(None, None, dict(cell), 1.0, trials=[{"width": width}]), ""

    monkeypatch.setattr(run_estimate, "_time_one", fake_time_one)
    protocol, _why = plan.normalize_protocol(dict(PROTOCOL))
    record = await rc.run(None, "python", tmp_path, protocol, deterministic=True, timeout_s=60, env=None,
                          thresholds=None, in_each_run=None, names={"width"})
    rc.write(tmp_path, record)
    assert {c["step_mm"] for c in asked} == {0.1, 0.05, 0.025}
    step = record["settings"]["step_mm"]
    assert step["verdict"] == "not_converged" and step["settings_run"] >= 2
    gaps = rc.evidence_gaps(tmp_path, protocol)
    assert gaps and "not converged" in gaps[0] and "none at another" in gaps[0]


@pytest.mark.asyncio
async def test_a_study_with_random_numbers_is_not_judged_here(tmp_path: Path) -> None:
    protocol, _why = plan.normalize_protocol(dict(PROTOCOL))
    record = await rc.run(None, "python", tmp_path, protocol, deterministic=False, timeout_s=60, env=None,
                          thresholds=None, in_each_run=None, names=set())
    assert record["status"] == "not_judged" and rc.lines(record)


def test_no_record_on_a_plan_with_numerical_settings_is_a_gap_and_none_without_them(tmp_path: Path) -> None:
    protocol, _why = plan.normalize_protocol(dict(PROTOCOL))
    assert rc.evidence_gaps(tmp_path, protocol)
    assert rc.evidence_gaps(tmp_path, {"grid": {"a": [1, 2]}}) == []
    assert math.isfinite(rc.DEFAULT_TOLERANCE)


def test_a_search_s_first_scan_is_run_again_at_the_search_s_finer_settings() -> None:
    block = {"grid": {"gap_mm": [1.0, 2.0]}, "numerical_settings": {"step_mm": {"search": 0.1, "check": [0.05]}}}
    s = rc.settings({"optimisation": block})
    assert s["step_mm"]["levels"] == [0.1, 0.05]
    assert rc.settings({"optimisation": {**block, "grid": None}}) == {}, "a search with no scan: its own check covers it"
