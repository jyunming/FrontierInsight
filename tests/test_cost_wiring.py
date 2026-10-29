"""The size of a search comes from FI's own cost model, not from the plan's count (core/run_checks.py::_size_the_scan).

A neutral made-up search: 20,000 planned evaluations of a design that takes 15 s each, in a 1 h limit. The plan model's count
cannot be run; FI's measured cost per evaluation gives the number that fits after the coarse scan, the check of the best
design (its runs times one measured run) and the analysis."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

from core import cost_model, plan, run_estimate
from tests.test_plan_optimisation import HEAT_SINK
from tests.test_run_checks import _SCAN_BLOCK, _cut, _log, _search_engine


def _plan_budget(eng: Any) -> tuple[int, int]:
    block = plan.load_design(eng.quest_root)[0]["protocol"]["optimisation"]["evaluation_budget"]
    return int(block["starts"]), int(block["per_start"])


async def _sized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, per_eval: float = 15.0) -> tuple[Any, Any, dict[str, Any]]:
    block = copy.deepcopy(_SCAN_BLOCK)
    block["grid"] = {"axis0": [0, 1, 2]}
    block["evaluation_budget"] = {"starts": 8, "per_start": 2500, "seconds_per_evaluation": 0.3}
    eng, runner = await _search_engine(tmp_path, block)
    # the engine helper puts no plan.md in place: the plan is the one the model wrote, with the same protocol
    design = {**HEAT_SINK, "protocol": {"optimisation": block, "thresholds": {}}}
    (eng.quest_root / "plan.md").write_text(plan.render("a made-up search", None, design), encoding="utf-8")

    async def timed(*a: Any, **k: Any) -> Any:
        return run_estimate.Probe(None, None, {}, per_eval), ""

    monkeypatch.setattr(run_estimate, "_time_one", timed)
    assert await eng._size_the_run({"iteration": 0}, runner, sys.executable, None) is None
    return eng, runner, _cut(eng)


@pytest.mark.asyncio
async def test_the_search_is_given_the_evaluations_the_cost_model_says_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, cut = await _sized(tmp_path, monkeypatch)
    model = cost_model.read(eng.quest_root)
    assert model is not None
    want = cost_model.search_plan(model, 3600.0, planned_evaluations=20000, check_runs=10, scan_s=model.seconds("scan"),
                                  design=model.search_key)
    assert want.evaluations is not None and 0 < want.evaluations < 20000 / 10
    assert cut["evaluations_lowered_to"] == want.evaluations, cut
    assert runner.search_deadline_s == pytest.approx(want.deadline_s) and cut["search_deadline_s"] == pytest.approx(want.deadline_s, abs=0.1)


@pytest.mark.asyncio
async def test_the_check_keeps_its_reserve_and_the_analysis_keeps_its_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, cut = await _sized(tmp_path, monkeypatch)
    # 10 runs of the check at 15 s each (the time of one run at its finest settings, here the same 15 s), never more than half
    assert runner.check_reserve_s >= 10 * 15.0 and runner.check_reserve_s <= 3600 / 2
    search_s = min(3600 - runner.check_reserve_s, runner.search_deadline_s)  # as OptimisationRunner.run gives the search its time
    assert search_s <= 3600 - 10 * 15.0 - cost_model.ANALYSIS_RESERVE_S


@pytest.mark.asyncio
async def test_the_plan_is_given_the_count_as_a_version_of_the_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, cut = await _sized(tmp_path, monkeypatch)
    starts, per_start = _plan_budget(eng)
    assert starts * per_start <= cut["evaluations_lowered_to"] and starts * per_start > 0
    assert any(row.get("by") == "engine" for row in plan.history(eng.quest_root))
    assert "FI made the search smaller" in _log(eng)


@pytest.mark.asyncio
async def test_nothing_measured_keeps_the_old_size_and_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    def nothing(model: Any, limit_s: float, **kw: Any) -> Any:
        return cost_model.SearchPlan(None, None, 0.0, 10, cost_model.ANALYSIS_RESERVE_S, 0.0, None, cost_model.UNKNOWN,
                                     "no cost per evaluation was measured, so no number of evaluations is given")

    monkeypatch.setattr(cost_model, "search_plan", nothing)
    eng, runner, cut = await _sized(tmp_path, monkeypatch)
    assert "evaluations_lowered_to" not in cut and "search_deadline_s" not in cut and runner.search_deadline_s is None
    assert runner.check_reserve_s == 1.5 * 10 * 15.0  # FI's usual reserve stands
    assert "no cost per evaluation was measured" in _log(eng) and "FI could not work out how many evaluations" in capsys.readouterr().out


def test_the_paper_says_the_search_was_made_smaller_from_fis_record(tmp_path: Path) -> None:
    from core import disclosure

    (tmp_path / ".fi").mkdir()
    cut = {"kind": "search", "estimate_s": 4.5e5, "at_least": True, "limit_s": 3600.0, "settings": 3, "evaluations": 20000,
           "evaluations_lowered_to": 216}
    (tmp_path / ".fi" / "run_estimate.json").write_text(__import__("json").dumps({"truncated": cut}), encoding="utf-8")
    text = disclosure.cut_short_text(tmp_path)
    assert "asked for 20,000 search evaluations; FI lowered them to 216" in text
