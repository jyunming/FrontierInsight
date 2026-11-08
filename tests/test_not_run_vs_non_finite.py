"""A setting that never ran and a setting whose every trial gave a non-finite number are different facts, and FI's own
trial record (raw/trials.json) tells them apart: the first has no trial at all, the second has trials whose values are
NaN or infinite. The model-written analysis returned None for both and the paper described a run that never happened."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import results_by_setting as rbs
from core import trial_table as tt

NAN = float("nan")


def _cells() -> list[dict[str, Any]]:
    return [
        {"cell": {"load": 1}, "key": "load=1", "planned": 3, "ok": 3, "failed": 0,
         "metrics": {"gain": {"values": [1.0, 2.0, 3.0]}}},
        # never ran: no trial succeeded or failed, no value
        {"cell": {"load": 2}, "key": "load=2", "planned": 3, "ok": 0, "failed": 0, "metrics": {"gain": {"values": []}}},
        # ran, but every value was non-finite
        {"cell": {"load": 3}, "key": "load=3", "planned": 3, "ok": 3, "failed": 0,
         "metrics": {"gain": {"values": [NAN, float("inf"), NAN], "why_missing": {"overflow": 3}}}},
    ]


def _by_setting(table: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["setting"]: r for r in table}


def test_never_run_and_non_finite_settings_are_told_apart() -> None:
    """Old code: both settings had no row at all (a row needed n > 0), so the table said nothing of either and the
    analysis model's None stood for both."""
    got = _by_setting(tt.rows_of(tt.summaries(_cells())))
    assert got["load=2"]["status"] == "not run" and got["load=2"]["n"] == 0 and got["load=2"]["mean"] is None
    assert got["load=3"]["status"] == "no finite value" and got["load=3"]["n"] == 0
    assert got["load=3"]["missing"] == 3 and got["load=3"]["why_missing"] == {"overflow": 3}
    assert "missing" not in got["load=2"]


def test_a_setting_with_no_metrics_at_all_is_not_run_when_its_counts_say_so() -> None:
    cells = _cells()
    cells[1]["metrics"] = {}
    got = _by_setting(tt.rows_of(tt.summaries(cells)))
    assert got["load=2"]["status"] == "not run"


def test_a_normal_setting_is_unchanged() -> None:
    """Known positive: a setting that gave numbers has no status and keeps its mean, SD and interval."""
    got = _by_setting(tt.rows_of(tt.summaries(_cells())))
    row = got["load=1"]
    assert row["status"] == "" and row["n"] == 3 and row["mean"] == 2.0 and row["sd"] == 1.0
    assert row["ci_lower"] is not None


def test_failed_trials_are_not_called_not_run() -> None:
    """Trials that were run and failed are not "never ran"."""
    cells = [{"cell": {"load": 1}, "key": "load=1", "ok": 2, "failed": 0, "metrics": {"gain": {"values": [1.0, 2.0]}}},
             {"cell": {"load": 2}, "key": "load=2", "ok": 0, "failed": 2, "metrics": {"gain": {"values": []}}}]
    got = _by_setting(tt.rows_of(tt.summaries(cells)))
    assert "load=2" not in got


def test_markdown_shows_both_words() -> None:
    text = tt.markdown(tt.rows_of(tt.summaries(_cells())))
    assert "not run" in text
    assert "no finite value (overflow: 3)" in text
    assert "nan" not in text.lower()


def test_the_trial_table_file_keeps_both_rows(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": _cells()}), encoding="utf-8")
    table = tt.write(tmp_path)
    assert {r["status"] for r in table} == {"", "not run", "no finite value"}
    assert "not run" in (tmp_path / "paper" / "results_table.md").read_text(encoding="utf-8")


def test_the_per_setting_table_says_not_run_and_the_reason() -> None:
    """Old code: the never-run setting read "no value", the same words as a run that gave nothing for a reason; the
    non-finite one carries the reason the run gave."""
    built = rbs.build(tt.summaries(_cells()), {"grid": {"load": [1, 2, 3]}}, {})
    table = next(t for t in built.tables if t.value == "gain")
    cells = {r[0]: r[2] for r in table.rows}
    assert cells["2"] == "not run"
    assert cells["3"].startswith("no value") and "overflow" in cells["3"]
    assert cells["1"].startswith("2")
    by = {str(r["load"]): r for r in built.rows}
    assert by["2"]["status"] == "not run" and by["2"]["n"] == 0 and by["2"]["mean"] is None
    assert by["3"]["status"] == "no finite value"
    assert by["1"]["status"] == ""


def test_no_values_and_no_counts_is_not_called_not_run() -> None:
    """"Not run" claims no trial ran, so only the record's counts may say it; a record without counts cannot tell it from
    trials that all failed and keeps the old "no value" wording."""
    cells = [{"cell": {"load": 1}, "key": "load=1", "metrics": {"gain": {"values": [1.0, 2.0, 3.0]}}},
             {"cell": {"load": 2}, "key": "load=2", "metrics": {"gain": {"values": []}}}]
    summarised = tt.summaries(cells)
    assert summarised[1]["metrics"]["gain"]["status"] == "" and not summarised[1]["not_run"]
    assert "load=2" not in _by_setting(tt.rows_of(summarised))
    built = rbs.build(summarised, {"grid": {"load": [1, 2]}}, {})
    table = next(t for t in built.tables if t.value == "gain")
    assert any("no value" in r for r in table.rows) and not any("not run" in r for r in table.rows)
