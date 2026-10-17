"""What the analysis reports for a setting must be what ran at that setting (core/reported_settings.py, made after the run
through core/result_checks.py). Neutral toy records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import reported_settings as rs
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.execution import ExecutionResult

RATES = [1.0, 2.0, 4.0, 8.0, 16.0]


def _record(rates: list[float] = RATES, sizes: tuple[int, ...] = (10,)) -> dict[str, Any]:
    cells = []
    for size in sizes:
        for r in rates:
            cells.append({"cell": {"rate": r, "size": size}, "key": f"rate={r}", "planned": 1, "ok": 1, "failed": 0,
                          "metrics": {"gain": {"values": [r * 3.0 + size]}, "loss": {"values": [100.0 / r]},
                                      "rate_hz": {"values": [r]}, "flat": {"values": [7.0]}}})
    return {"schema": "fi.trials/v1", "cells": cells}


def _rows(rate_of: Any) -> list[dict[str, float]]:
    return [{"rate_hz": rate_of(r), "gain": r * 3.0 + 10, "loss": 100.0 / r} for r in RATES]


def _cells() -> list[dict[str, Any]]:
    return rs.cells_of(_record())


def test_rows_that_carry_one_settings_results_under_another_are_caught_with_the_real_values() -> None:
    found = rs.problems(_cells(), [("", {"series": _rows(lambda r: 1.0)})])
    assert len(found) == 1
    m = found[0]
    assert (m.path, m.column, m.axis, m.wrong, m.rows) == ("series", "rate_hz", "rate", 4, 5)
    assert m.reported == [1.0] and m.ran[:3] == [2.0, 4.0, 8.0]
    assert "reports `rate_hz` as 1 in 4 of 5 rows whose results ran at `rate` = 2, 4, 8, 16" in m.says()
    assert m.top == "series"


def test_rows_that_carry_what_ran_pass_in_every_table_shape() -> None:
    assert rs.problems(_cells(), [("", {"series": _rows(lambda r: r)})]) == []
    columns = {"rate_hz": RATES, "gain": [r * 3.0 + 10 for r in RATES], "loss": [100.0 / r for r in RATES]}
    assert rs.problems(_cells(), [("summary", columns)]) == []
    bad = dict(columns, rate_hz=[1.0] * 5)
    assert [m.column for m in rs.problems(_cells(), [("summary", bad)])] == ["rate_hz"]


def test_a_slice_of_the_grid_reported_at_one_setting_is_not_a_collapsed_axis() -> None:
    cells = rs.cells_of(_record(RATES, sizes=(10, 20)))
    rows = [{"size": 20, "rate_hz": r, "gain": r * 3.0 + 20, "loss": 100.0 / r} for r in RATES]
    assert rs.problems(cells, [("", {"sweep": rows})]) == []


def test_rows_that_match_no_cell_are_not_judged_and_a_table_with_one_row_per_cell_is_a_collapsed_axis() -> None:
    derived = [{"rate_hz": 1.0, "gain": 123.0 + i, "other": 5.0 * i} for i in range(5)]
    found = rs.problems(_cells(), [("", {"d": derived})])
    assert len(found) == 1 and found[0].collapsed and "in all 5 rows" in found[0].says() and found[0].ran == RATES
    # fewer rows than cells: could be any slice
    assert rs.problems(_cells(), [("", {"d": derived[:3]})]) == []


def test_a_study_that_varies_nothing_or_a_report_with_no_table_has_nothing_to_compare() -> None:
    assert rs.problems(rs.cells_of(_record([1.0])), [("", {"series": _rows(lambda r: 1.0)})]) == []
    assert rs.problems(_cells(), [("", {"mean": 3.0, "name": "x"})]) == []
    assert rs.problems([], [("", {"series": _rows(lambda r: 1.0)})]) == []


def test_a_column_is_a_setting_only_when_named_for_one() -> None:
    assert rs.names_setting("rate", "rate") and rs.names_setting("rate_hz", "rate") and rs.names_setting("rate", "rate_hz")
    assert rs.names_setting("RateHz", "rate")
    assert not rs.names_setting("rate_error", "rate") and not rs.names_setting("max_rate_hz", "rate")
    assert not rs.names_setting("rateless", "rate") and not rs.names_setting("", "rate")


def test_the_request_says_what_differs_and_ends_with_the_instruction() -> None:
    found = rs.problems(_cells(), [("", {"series": _rows(lambda r: 1.0)})])
    text = rs.directive(found, _cells())
    assert text.startswith("FI compared what the analysis reports for each setting with the settings that really ran")
    assert "The study's settings are `rate`." in text
    assert text.rstrip().endswith("Change nothing else.") and 'entry["cell"]' in text
    assert "left out" in rs.writer_note(found) and "limitations" in rs.writer_note(found)
    assert rs.keys_to_leave_out(found) == ["series", "rate_hz"]


def test_the_summaries_under_data_results_are_read_with_the_printed_result(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "results"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"series": _rows(lambda r: 1.0)}), encoding="utf-8")
    (folder / "broken.json").write_text("{", encoding="utf-8")
    got = rs.reported_of(tmp_path, {"x": 1})
    assert [w for w, _ in got] == ["", "summary"]
    assert [m.top for m in rs.problems(_cells(), got)] == [""]


# ---- through the engine ---------------------------------------------------------------------------------------------------


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: Any) -> Engine:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    eng = Engine(Config(
        topic="a neutral topic", title="reported", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng._trial_mode = True
    (eng.quest_root / "raw").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "raw" / "trials.json").write_text(json.dumps(_record()), encoding="utf-8")
    eng._rows = rows  # type: ignore[attr-defined]
    return eng


def _run(eng: Engine) -> ExecutionResult:
    printed = {"series": eng._rows, "mean_gain": 5.0}  # type: ignore[attr-defined]
    return ExecutionResult(0, "log line\nRESULT_JSON: " + json.dumps(printed), "", 1.0, False)


@pytest.mark.asyncio
async def test_results_that_do_not_match_what_ran_are_sent_back_like_a_failed_run_twice_then_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine(tmp_path, monkeypatch, _rows(lambda r: 1.0))
    state: dict[str, Any] = {"iteration": 0}
    for n in (1, 2):
        result, script = await eng._results_match_run(state, _run(eng), None, True)
        assert result.returncode == 1 and script == "experiment.py", n
        assert "RESULT_JSON" not in result.stdout and "log line" in result.stdout
        assert "FI compared what the analysis reports" in result.stderr and "reports `rate_hz` as 1" in result.stderr
        assert not (eng.quest_root / "needs" / "RESULT_CHECK.json").exists()
    # the third time the repairs are spent: the run is kept as it is and what is wrong is left out
    result, script = await eng._results_match_run(state, _run(eng), None, True)
    assert result.returncode == 0 and script is None
    record = json.loads((eng.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))
    assert record["removed_quantities"] == ["rate_hz", "series"]
    assert eng._left_out_quantities() == {"rate_hz", "series"}
    assert eng._without_typed_results({"series": [1], "rate_hz": 1, "mean_gain": 5.0}) == {"mean_gain": 5.0}
    note = eng._run_checks_note()
    assert "left out" in note and "limitations" in note and "reports `rate_hz`" in note
    # a new design iteration has its own repairs, and a clean run removes the record
    again, _ = await eng._results_match_run({"iteration": 1}, _run(eng), None, True)
    assert again.returncode == 1
    eng._rows = _rows(lambda r: r)  # type: ignore[attr-defined]
    clean, script = await eng._results_match_run({"iteration": 1}, _run(eng), "experiment.py", True)
    assert clean.returncode == 0 and script == "experiment.py"
    assert not (eng.quest_root / "needs" / "RESULT_CHECK.json").exists() and eng._left_out_quantities() == set()
    assert eng._run_checks_note() == ""


@pytest.mark.asyncio
async def test_no_repair_is_asked_when_the_quest_has_none_left_or_the_run_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch, _rows(lambda r: 1.0))
    spent = {"iteration": 0, "exec_reflect_iter": eng.config.engine.exec_reflect_max_iterations}
    result, _ = await eng._results_match_run(spent, _run(eng), None, True)
    assert result.returncode == 0 and (eng.quest_root / "needs" / "RESULT_CHECK.json").is_file()
    failed = ExecutionResult(1, "", "Traceback", 1.0, False)
    got, script = await eng._results_match_run({"iteration": 0}, failed, "simulate.py", True)
    assert got is failed and script == "simulate.py"
    eng._trial_mode = False
    got, _ = await eng._results_match_run({"iteration": 0}, _run(eng), None, True)
    assert got.returncode == 0
