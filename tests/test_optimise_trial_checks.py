"""FI's check of the failed trials (core/trial_checks.py, check 2a) on a search for the best design
(``OptimisationRunner._check_trials``): the search's evaluations are judged together by their failure rate, and each
setting of the coarse scan by its own trials. More than half failed is sent back to be repaired while a repair is left,
then stops the quest; more than 5% is recorded for the paper. A failed design is never chosen by the search, so nothing is
left out. The intervals and standard errors are not checked on a search, and the record says "calibration not checked"."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core import optimise, trial_checks


def _rows(ok: int, failed: int, *, stage: str = "start") -> list[dict[str, Any]]:
    return ([{"event": "evaluation", "stage": stage, "status": "ok", "design": {"x": n}} for n in range(ok)]
            + [{"event": "evaluation", "stage": stage, "status": "failed", "design": {"x": 100 + n},
                "reason": "ZeroDivisionError: division by zero (at simulate.py line 7)"} for n in range(failed)])


def _runner(tmp_path: Path, repairs_left: bool) -> optimise.OptimisationRunner:
    return optimise.OptimisationRunner(None, quest_root=tmp_path, protocol={}, simulate=tmp_path / "code" / "simulate.py",
                                       analysis=tmp_path / "code" / "experiment.py", repairs_left=lambda: repairs_left)


def test_the_evaluations_are_one_setting_and_the_scan_settings_their_own(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "trials.json").write_text(json.dumps({"cells": [
        {"key": "x=1", "planned": 10, "ok": 2, "failed": 8, "metrics": {}},
        {"key": "x=2", "planned": 10, "ok": 10, "failed": 0, "metrics": {}}]}), encoding="utf-8")
    run = optimise.SearchRun(record={}, rows=_rows(3, 1) + _rows(2, 0, stage="scan"), scan=True)
    cells, reasons = optimise.trial_cells(run, tmp_path)
    assert [(c["key"], c["planned"], c["failed"]) for c in cells] == [
        ("x=1", 10, 8), ("x=2", 10, 0), ("the search's evaluations", 4, 1)]
    assert reasons["the search's evaluations"].most_common(1)[0][0].startswith("ZeroDivisionError")


def test_most_evaluations_failed_is_sent_back_then_stops_and_calibration_is_not_checked(tmp_path: Path) -> None:
    run = optimise.SearchRun(record={}, rows=_rows(3, 7))
    runner = _runner(tmp_path, True)
    sent = runner._check_trials(run, time.monotonic())
    assert sent is not None and sent.returncode == 1 and runner.failed_script == "simulate.py"
    assert runner.trial_fault["action"] == trial_checks.REPAIR and not runner.trial_fault["partial"]
    assert "more than half of the trials failed" in sent.stderr and "ZeroDivisionError" in sent.stderr
    record = trial_checks.read_record(tmp_path)
    assert record["calibration"]["checked"] is False
    assert any("search for the best design" in n for n in record["calibration"]["not_checked"])
    runner = _runner(tmp_path, False)
    assert runner._check_trials(run, time.monotonic()) is not None
    assert runner.trial_fault["action"] == trial_checks.STOP and "7 of 10" in runner.trial_fault["say"]


def test_a_few_failed_designs_are_recorded_for_the_paper_and_nothing_is_sent_back(tmp_path: Path) -> None:
    runner = _runner(tmp_path, True)
    assert runner._check_trials(optimise.SearchRun(record={}, rows=_rows(9, 1)), time.monotonic()) is None
    record = trial_checks.read_record(tmp_path)
    assert record["failures"][0]["key"] == "the search's evaluations" and record["failures"][0]["fault"] is False
    assert record["left_out_settings"] == []
    assert runner._check_trials(optimise.SearchRun(record={}, rows=_rows(10, 0)), time.monotonic()) is None
