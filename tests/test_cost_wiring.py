"""The size of a search comes from FI's own cost model, not from the plan's count (core/run_checks.py::_size_the_scan).

A neutral made-up search: 20,000 planned evaluations of a design that takes 15 s each, in a 1 h limit. The plan model's count
cannot be run; FI's measured cost per evaluation gives the number that fits after the coarse scan, the check of the best
design (its runs times one measured run, with FI's margin) and the analysis."""

from __future__ import annotations

import copy
import json
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


async def _engine(tmp_path: Path, per_start: int = 2500) -> tuple[Any, Any, dict[str, Any]]:
    block = copy.deepcopy(_SCAN_BLOCK)
    block["grid"] = {"axis0": [0, 1, 2]}
    block["evaluation_budget"] = {"starts": 8, "per_start": per_start, "seconds_per_evaluation": 0.3}
    eng, runner = await _search_engine(tmp_path, block)
    design = {**HEAT_SINK, "protocol": {"optimisation": block, "thresholds": {}}}
    (eng.quest_root / "plan.md").write_text(plan.render("a made-up search", None, design), encoding="utf-8")
    return eng, runner, block


async def _size(eng: Any, runner: Any, block: dict[str, Any], monkeypatch: pytest.MonkeyPatch, per_eval: float = 15.0,
                finer: Any = 15.0, freeze: bool = False, iteration: int = 0) -> tuple[dict[str, Any] | None, Any]:
    """Size the search again. ``finer``: what the check's one finest-settings run is timed at (a number; ``None`` for a timing
    that fails; an exception to raise). Returns FI's cut record (``None`` when the quest stopped) and the sizing's result."""

    async def timed(*a: Any, name: str = "pilot", **k: Any) -> Any:
        if name == "pilot_finer":
            if isinstance(finer, Exception):
                raise finer
            if finer is None:
                return None, "the simulation failed"
            return run_estimate.Probe(None, None, {}, finer), ""
        return run_estimate.Probe(None, None, {}, per_eval), ""

    monkeypatch.setattr(run_estimate, "_time_one", timed)
    if freeze:
        from core import frozen_protocol

        frozen_protocol.freeze(eng.quest_root, {"optimisation": block, "thresholds": {}}, approved_by="test", source="plan.md")
    result = await eng._size_the_run({"iteration": iteration}, runner, sys.executable, None)
    record = run_estimate.read_record(eng.quest_root).get("truncated")
    return (record if isinstance(record, dict) else None), result


async def _sized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, per_start: int = 2500, **kw: Any) -> tuple[Any, Any, Any, Any]:
    eng, runner, block = await _engine(tmp_path, per_start)
    cut, result = await _size(eng, runner, block, monkeypatch, **kw)
    return eng, runner, cut, result


@pytest.mark.asyncio
async def test_the_search_is_given_the_evaluations_the_cost_model_says_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, cut, _ = await _sized(tmp_path, monkeypatch)
    model = cost_model.read(eng.quest_root)
    assert model is not None
    want = cost_model.search_plan(model, 3600.0, planned_evaluations=20000, check_runs=10, scan_s=model.seconds("scan"),
                                  design=model.search_key, margin=run_estimate.SAFETY)
    assert want.evaluations is not None and 0 < want.evaluations < 20000 / 10
    starts, per = _plan_budget(eng)
    assert cut["evaluations_lowered_to"] == starts * per <= want.evaluations  # the count in the paper is the count in plan.md
    assert runner.search_deadline_s == pytest.approx(want.deadline_s) and cut["search_deadline_s"] == pytest.approx(want.deadline_s, abs=0.1)


@pytest.mark.asyncio
async def test_the_check_reserve_is_the_finer_run_times_the_safety_factor_not_the_search_design(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, cut, _ = await _sized(tmp_path, monkeypatch, per_eval=15.0, finer=45.0)
    assert runner.check_reserve_s == pytest.approx(10 * 45.0 * run_estimate.SAFETY)  # 675 s; the design's own 15 s would give 225
    assert runner.search_deadline_s == pytest.approx(3600 - runner.check_reserve_s - cost_model.ANALYSIS_RESERVE_S)
    model = cost_model.read(eng.quest_root)
    assert "includes the start" in model.term("finer_check_unit").basis  # the probe did not separate the start from the run
    search_s = min(3600 - runner.check_reserve_s, runner.search_deadline_s)  # as OptimisationRunner.run gives the search its time
    assert search_s <= 3600 - 10 * 45.0 - cost_model.ANALYSIS_RESERVE_S


@pytest.mark.asyncio
async def test_the_plan_is_given_the_count_as_a_version_of_the_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, cut, _ = await _sized(tmp_path, monkeypatch)
    starts, per_start = _plan_budget(eng)
    assert 0 < starts * per_start < 20000
    assert any(row.get("by") == "engine" for row in plan.history(eng.quest_root))
    assert "FI made the search smaller" in _log(eng)


@pytest.mark.asyncio
async def test_a_frozen_protocol_keeps_its_plan_and_still_gets_the_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, cut, _ = await _sized(tmp_path, monkeypatch, freeze=True)
    assert "FI made the search smaller" not in _log(eng) and "evaluations_lowered_to" not in cut
    assert _plan_budget(eng) == (8, 2500) and runner.search_deadline_s is not None
    assert "the protocol is frozen" in _log(eng)


@pytest.mark.asyncio
async def test_a_redesign_keeps_its_plan_and_still_gets_the_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, cut, _ = await _sized(tmp_path, monkeypatch, iteration=1)
    assert _plan_budget(eng) == (8, 2500) and "evaluations_lowered_to" not in cut and runner.search_deadline_s is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("finer", [None, RuntimeError("boom")])
async def test_a_finer_timing_that_fails_does_not_stop_the_run_and_the_check_stays_a_lower_bound(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finer: Any) -> None:
    eng, runner, cut, result = await _sized(tmp_path, monkeypatch, finer=finer)
    assert result is None and cut is not None
    assert cost_model.read(eng.quest_root).seconds("finer_check_unit") is None  # the search design's cost stands, as a lower bound
    assert runner.check_reserve_s >= 10 * 15.0
    assert "could not time one run of its check" in _log(eng)


@pytest.mark.asyncio
async def test_a_study_that_stops_as_too_long_leaves_plan_md_alone_and_says_nothing_of_a_smaller_search(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    eng, runner, cut, result = await _sized(tmp_path, monkeypatch, per_eval=4000.0, finer=4000.0)  # one design takes > 1 h
    assert result is not None and cut is None  # the quest's plain stop
    assert _plan_budget(eng) == (8, 2500)
    assert "made the search smaller" not in _log(eng) and "made the search smaller" not in capsys.readouterr().out
    assert not eng._size_edit_after_read()


@pytest.mark.asyncio
async def test_nothing_measured_through_the_real_cost_model_keeps_the_old_size_and_says_why(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    real = cost_model.keep_measured

    def without_the_cost_of_an_evaluation(root: Any, **kw: Any) -> bool:
        kw.pop("per_evaluation_s", None)
        return real(root, **kw)

    monkeypatch.setattr(cost_model, "keep_measured", without_the_cost_of_an_evaluation)
    eng, runner, cut, _ = await _sized(tmp_path, monkeypatch)
    assert "evaluations_lowered_to" not in cut and runner.search_deadline_s is None and _plan_budget(eng) == (8, 2500)
    assert "no cost per evaluation was measured" in _log(eng) and "FI could not work out how many evaluations" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_sizing_again_after_a_resume_keeps_what_the_plan_asked_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import disclosure

    eng, runner, block = await _engine(tmp_path)
    cut, _ = await _size(eng, runner, block, monkeypatch)
    to = cut["evaluations_lowered_to"]
    starts, per = _plan_budget(eng)
    block["evaluation_budget"] = {"starts": starts, "per_start": per, "seconds_per_evaluation": 0.3}  # the resume reads plan.md
    again, _ = await _size(eng, runner, block, monkeypatch)
    assert again["evaluations"] == 20000 and again["evaluations_lowered_to"] == to
    assert f"the plan asked for 20,000 search evaluations; FI lowered them to {to:,}" in disclosure.cut_short_text(eng.quest_root)


@pytest.mark.asyncio
async def test_the_record_of_who_approved_the_plan_learns_that_fi_changed_it_after_it_was_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, runner, cut, _ = await _sized(tmp_path, monkeypatch)
    assert eng._size_edit_after_read()


@pytest.mark.asyncio
async def test_a_fast_evaluation_is_said_as_under_a_second(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    eng, runner, cut, _ = await _sized(tmp_path, monkeypatch, per_start=5000, per_eval=0.2, finer=0.2)
    assert "one takes under a second" in capsys.readouterr().out


def test_a_margin_on_the_check_reserve_keeps_the_half_limit_cap() -> None:
    model = cost_model.CostModel(plan={"check_runs": 100}, terms={"wall_per_evaluation": cost_model.Term(1.0, cost_model.MEASURED),
                                                                  "finer_check_unit": cost_model.Term(40.0, cost_model.MEASURED)})
    assert cost_model.search_plan(model, 10_000.0, margin=1.5).check_reserve_s == 5000.0  # 6000 wanted, half the limit given
    assert cost_model.search_plan(model, 100_000.0, margin=1.5).check_reserve_s == 6000.0
    assert cost_model.search_plan(model, 100_000.0).check_reserve_s == 4000.0


def test_the_paper_says_the_search_was_made_smaller_from_fis_record(tmp_path: Path) -> None:
    from core import disclosure

    (tmp_path / ".fi").mkdir()
    cut = {"kind": "search", "estimate_s": 4.5e5, "at_least": True, "limit_s": 3600.0, "settings": 3, "evaluations": 20000,
           "evaluations_lowered_to": 216}
    (tmp_path / ".fi" / "run_estimate.json").write_text(json.dumps({"truncated": cut}), encoding="utf-8")
    assert "asked for 20,000 search evaluations; FI lowered them to 216" in disclosure.cut_short_text(tmp_path)
