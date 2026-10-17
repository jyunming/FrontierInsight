"""The second review of what each run compares inside it (core/in_run.py) and of the plan's own sums: the stuck-output
check holds for a study whose numbers are all per value, a record's own field names are allowed only where the record
has them, and the plan's sums are read only where they are plainly written."""

from __future__ import annotations

from typing import Any

from core import flat_output, grid_keys, in_run, oracle_forms, run_estimate

INSIDE = {"procedure": ["a", "b"]}


class _Cell:
    def __init__(self, cell: dict[str, Any], values: list[dict[str, Any]], levels: list[dict[str, Any]]) -> None:
        self.cell, self.levels = cell, levels
        self.rows = [{"status": "ok", "values": v} for v in values]


def test_a_study_whose_numbers_are_all_per_value_is_still_checked_for_output_that_ignores_the_settings() -> None:
    levels = in_run.level_cells(INSIDE)
    flat = [_Cell({"m": m}, [{"power": {"procedure=a": 0.5, "procedure=b": 0.5}}] * 2, levels) for m in (10, 20, 30)]
    seen = flat_output.observed_flat_sweep(flat)
    assert seen is not None and seen.names == ["power"]
    moves = [_Cell({"m": m}, [{"power": {"procedure=a": m / 100, "procedure=b": 0.5}}] * 2, levels) for m in (10, 20, 30)]
    assert flat_output.observed_flat_sweep(moves) is None


def test_the_pilot_leaves_out_an_empty_number_per_run() -> None:
    probe = run_estimate.Probe(None, None, {"m": 10}, 0.0, trials=[{"power": {"procedure=a": 0.5, "procedure=b": 0.4}}])
    obs = run_estimate.observations([probe])
    assert all(values for _cell, values in obs) and len(obs) == 2


def test_a_record_field_is_a_setting_name_only_for_a_study_that_has_such_records() -> None:
    reads_level = {"simulate.py": "def run_trial(cell, trial_id, seed):\n    return {'x': cell['level']}\n"}
    assert grid_keys.problems(reads_level, {"grid": {"noise": [0.1, 0.2]}})
    assert grid_keys.problems(reads_level, {"grid": {"noise": [0.1, 0.2]}, "in_each_run": INSIDE}) == []


def _flagged(text: str) -> list[str]:
    return [f["expression"] for f in oracle_forms.stated_arithmetic(text)]


def test_the_plans_sums_are_read_only_where_they_are_plainly_written() -> None:
    for correct in ["10^3 + 5 = 1005", "2^3 * 4 = 32", "1.05^10 - 1 = 0.629", "10^-3 = 0.001", "e^-1 = 0.368",
                    "1/4 = 25%", "3/4 = 75 %", "1,000 * 2 = 2,000", "1000*2 = 2,000", "8.314 * 350 = 2.9e3",
                    "8.314 * 350 = 2900", "exp(10) = 22026", "100/3 = 33", "2/3 = 0.66", "log(100) = 2",
                    "runs 10-20 = 11 values", "2024-10-15 = 2024", "k = 3.45 s^-1", "2 * 3 = 6"]:
        assert _flagged(correct) == [], correct


def test_wrong_sums_are_still_caught() -> None:
    assert _flagged("k=100000000*exp(-50000/(8.314462618*350))=3.448570123e0 s^-1") == [
        "100000000*exp(-50000/(8.314462618*350))"]
    assert _flagged("2 * 3 = 7") == ["2 * 3"]
    assert _flagged("abs(3.4519 - 3.4486)/3.4486 = 0") == ["abs(3.4519 - 3.4486)/3.4486"]
    assert _flagged("100/3 = 34") == ["100/3"]


def test_a_long_text_before_a_sum_is_read_quickly() -> None:
    import time
    started = time.monotonic()
    oracle_forms.stated_arithmetic("word " * 6000 + "2 * 3 = 7")
    assert time.monotonic() - started < 1.0
