"""The settings table counts the plan's settings and writes no empty rows; a spread carries its own interval.

The fixture ``tests/fixtures/settings_table/trials.json`` is a trimmed copy of a real quest's trial record (a 2 x 2 slice
of its sigma x n_points grid): each grid setting has a record of ``Ea_hat`` and, because the study compares values
inside each run, one more record per (temperature, interval type) that carries only ``k_hat`` and ``k_rel_err``.
"""
from __future__ import annotations

import csv
import io
import json
import math
import shutil
from pathlib import Path

from core import paper_values as pv
from core import results_by_setting as rbs

FIXTURE = Path(__file__).parent / "fixtures" / "settings_table" / "trials.json"
PROTOCOL = {"grid": {"sigma": [0.005, 0.1], "n_points": [5, 40]}}


def _quest(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    shutil.copy(FIXTURE, tmp_path / "raw" / "trials.json")
    (tmp_path / "code").mkdir()
    return tmp_path


def test_the_count_is_the_plans_settings_not_every_record(tmp_path: Path) -> None:
    _rows, built = rbs.from_quest(_quest(tmp_path), PROTOCOL)
    assert built is not None
    assert len(built.distinct.settings) > 4  # the per-temperature records are kept ...
    assert built.distinct.n_settings == 4  # ... but they are not settings
    block = rbs.paper_block(built, cap=1)
    assert "all 4 settings" in block and "275" not in block
    assert "supplementary table `paper/results_by_setting.csv`" in block and "FI's tables" not in block
    assert "4 distinct settings" in rbs.writer_note([], built, cap=1)


def test_a_row_with_no_trials_is_not_written(tmp_path: Path) -> None:
    _rows, built = rbs.from_quest(_quest(tmp_path), PROTOCOL)
    assert built is not None
    assert built.rows and all(r["n"] > 0 and r["mean"] is not None for r in built.rows)
    by_value: dict[str, int] = {}
    for r in built.rows:
        by_value[r["value"]] = by_value.get(r["value"], 0) + 1
    assert by_value["Ea_hat"] == 4  # one per grid setting, none per temperature
    assert by_value["k_hat"] == 16  # the per-temperature records that do carry k_hat stay, with their values
    rows = list(csv.DictReader(io.StringIO(rbs.csv_text(built))))
    assert len(rows) == len(built.rows) and all(int(r["n"]) > 0 for r in rows)
    ea = next(t for t in built.tables if t.value == "Ea_hat")
    assert len(ea.rows) == 4 and "at each of the 4 settings" in ea.caption


def test_a_grid_setting_whose_trials_all_failed_still_shows_as_no_value() -> None:
    cells = [{"cell": {"m": 1}, "key": "m=1", "metrics": {"y": {"values": [1.0, 2.0, 3.0]}}},
             {"cell": {"m": 2}, "key": "m=2", "metrics": {"y": {"values": []}}}]
    from core import trial_table as tt

    built = rbs.build(tt.summaries(cells), {"grid": {"m": [1, 2]}}, {})
    table = next(t for t in built.tables if t.value == "y")
    assert len(table.rows) == 2 and any("no value" in r for r in table.rows)


# --- an SD carries its own interval ---------------------------------------------------------------------------------


def _records() -> pv.Records:
    return pv.from_cells(json.loads(FIXTURE.read_text())["cells"])


def test_an_sd_is_written_with_its_95_percent_interval_like_a_mean() -> None:
    rec = _records()
    sd = pv.resolve("Ea_hat:sd @ sigma=0.005, n_points=all, temperature_K=all, interval_type=all", rec)
    ci = pv.resolve("Ea_hat:sd_ci @ sigma=0.005, n_points=all, temperature_K=all, interval_type=all", rec)
    assert sd.filled and sd.text == ci.text
    assert "95% chi-square interval" in sd.text and "16 trials" in sd.text and " to " in sd.text
    value, _dof = pv._pooled_sd(pv._matching(rec, pv.parse("Ea_hat:sd @ sigma=0.005, n_points=all, temperature_K=all, interval_type=all").where, "Ea_hat"),
                                "Ea_hat")
    lo, hi = sd.numbers[1], sd.numbers[2]
    assert lo < value < hi and math.isclose(sd.numbers[0], value)
    mean = pv.resolve("Ea_hat:mean_ci @ sigma=0.005, n_points=all, temperature_K=all, interval_type=all", rec)
    assert "t interval of the mean" in mean.text  # the way a mean is written is unchanged
    assert "chi-square" not in pv.resolve("Ea_hat:mean @ sigma=0.005, n_points=all, temperature_K=all, interval_type=all", rec).text


def test_an_sd_used_in_arithmetic_or_a_reducer_stays_one_number() -> None:
    rec = _records()
    ratio = pv.resolve("Ea_hat:sd @ sigma=0.1, n_points=all, temperature_K=all, interval_type=all | ratio @ sigma=0.005, n_points=all, temperature_K=all, interval_type=all", rec)
    assert ratio.filled and "interval" not in ratio.text
    biggest = pv.resolve("Ea_hat:sd @ sigma=each, n_points=all, temperature_K=all, interval_type=all | max", rec)
    assert biggest.filled and "interval" not in biggest.text


def test_an_sd_of_one_trial_is_not_given_an_interval() -> None:
    rec = pv.from_cells([{"cell": {"m": 1}, "key": "m=1", "metrics": {"y": {"values": [1.5]}}}])
    got = pv.resolve("y:sd @ m=1", rec)
    assert not got.filled
