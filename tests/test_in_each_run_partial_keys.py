"""A result that varies over only some of the settings compared inside each run (``protocol.in_each_run`` with two or more
settings): one number per value of ``width`` while ``gap`` is also compared inside the run. It is keyed by those settings
alone (``width=74``), recorded as a result over them (not once per combination), and read by a check at the value its case
names. A bare value, or a key no setting has, is still refused, and the refusal names the keys that are accepted."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core import in_run, trial_runner
from core.execution import SharedInterpreterExecutor

INSIDE = {"width": [74, 80, 90], "gap": [80, 90]}   # 80 and 90 are values of both settings
LEVELS = in_run.level_cells(INSIDE)


def test_a_result_over_one_setting_is_keyed_by_that_setting() -> None:
    got, why = in_run.read_returned({"spread": {"width=74": 1.0, "width=80": 2.0, "width=90": 3.0}}, LEVELS, "run_cell")
    assert why == "" and got == {"spread": {"width=74": 1.0, "width=80": 2.0, "width=90": 3.0}}
    got, why = in_run.read_returned({"g": {"gap=80": 1.0, "gap=90": 2.0}, "both": {"width=74,gap=80": 1.0, **{
        in_run.level_key(lv): 0.0 for lv in LEVELS if in_run.level_key(lv) != "width=74,gap=80"}}}, LEVELS, "run_cell")
    assert why == "" and set(got["g"]) == {"gap=80", "gap=90"} and len(got["both"]) == 6


def test_a_partial_result_still_gives_every_value_of_its_settings_and_one_set_of_settings() -> None:
    assert "without width=80, width=90" in in_run.read_returned({"spread": {"width=74": 1.0}}, LEVELS, "run_cell")[1]
    mixed = {"spread": {"width=74": 1.0, "gap=80": 2.0}}
    assert "same settings" in in_run.read_returned(mixed, LEVELS, "run_cell")[1]


def test_a_bare_or_unknown_key_is_refused_and_the_message_says_how_to_key() -> None:
    for key in ("74", "p74", "gap=77", "height=74"):
        why = in_run.read_returned({"spread": {key: 1.0}}, LEVELS, "run_cell")[1]
        assert "not one of the values" in why and "`width=74` " not in why.split("key it")[0]
        assert "width=74,gap=80" in why and "(`width=74`)" in why and "bare value names none" in why
    # one setting compared inside the run: a bare value stays unambiguous and stays accepted
    one = in_run.level_cells({"width": [74, 80]})
    assert in_run.read_returned({"spread": {"74": 1.0, "80": 2.0}}, one, "run_cell")[0] == {"spread": {"width=74": 1.0, "width=80": 2.0}}


def test_a_partial_result_gets_one_record_per_value_of_its_settings() -> None:
    values = {"spread": {"width=74": 1.0, "width=80": 2.0, "width=90": 3.0}, "total": 5.0}
    rows = [{"trial": 0, "status": "ok", "values": values}]
    cells = in_run.summary_cells({"n": 1}, "n=1", 1, rows, LEVELS)
    assert [c["key"] for c in cells] == ["n=1", "n=1,width=74", "n=1,width=80", "n=1,width=90"]   # not 6 combinations
    assert cells[1]["level"] == {"width": 74} and cells[1]["metrics"]["spread"]["values"] == [1.0]
    assert cells[0]["metrics"] == {"total": {"values": [5.0], "trials": [0], "count": 1, "total": 5.0, "non_finite": 0}}
    assert ("n=1,width=80", "spread", 2.0) in in_run.flat("n=1", values)


def test_a_check_reads_a_partial_result_at_the_value_its_case_names() -> None:
    values = {"spread": {"width=74": 1.0, "width=80": 2.0, "width=90": 3.0}, "total": 5.0}
    assert in_run.for_case(values, LEVELS, {"width": 80})["spread"] == 2.0
    assert in_run.for_case(values, LEVELS, {"width": 80, "gap": 90})["spread"] == 2.0   # the other setting does not move it
    assert in_run.for_case(values, LEVELS, {"gap": 90})["spread"] == [1.0, 2.0, 3.0]    # the case leaves width to the check
    assert in_run.for_case(values, LEVELS, None)["spread"] == [1.0, 2.0, 3.0] and in_run.for_case(values, LEVELS, {"width": 80})["total"] == 5.0


SOURCE = '''
def run_cell(cell):
    return {"spread": {"width=%d" % w: float(w) for w in (74, 80, 90)}, "total": 1.0}
'''


def test_the_small_run_accepts_a_per_setting_result_and_records_it_over_that_setting(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SOURCE, encoding="utf-8")
    ex = SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")
    run = asyncio.run(trial_runner.run_trials(ex, sys.executable, root, "code/simulate.py", {"m": [1]}, runs_per_setting=1,
                                              base_seed=0, deterministic=True, timeout_s=60, run_id="r1", in_each_run=INSIDE))
    assert [r["status"] for c in run.cells for r in c.rows] == ["ok"]
    assert [c.key for c in run.record_cells()] == ["m=1", "m=1,width=74", "m=1,width=80", "m=1,width=90"]


# --- a search for the best design keeps its in_each_run in its own block ---------------------------------------------------

SEARCH_PROTOCOL = {"optimisation": {"in_each_run": INSIDE}}
CASE_SOURCE = '''
def run_cell(cell):
    assert cell["width"] == [74, 80, 90], cell     # the run is given the whole list, as in the study
    return {"spread": {"width=%d" % w: float(w) for w in cell["width"]}, "total": 1.0}
'''


def test_the_runs_of_a_search_are_given_the_values_its_block_compares_inside_each_run() -> None:
    assert in_run.of_study(SEARCH_PROTOCOL) == INSIDE
    assert in_run.of_study({"in_each_run": INSIDE}) == INSIDE and in_run.of_study({"optimisation": {}}) is None


def test_a_case_run_for_a_search_gets_the_values_and_its_per_setting_result_is_read(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(CASE_SOURCE, encoding="utf-8")
    ex = SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")
    case = {"m": 1}   # a known-answer case carries settings only, no list of what each run compares
    kept = asyncio.run(trial_runner.run_case(ex, sys.executable, root, "code/simulate.py", cell=case, timeout_s=60,
                                             in_each_run=in_run.of_study(SEARCH_PROTOCOL)))
    assert kept[1] == "" and kept[0]["spread"] == [74.0, 80.0, 90.0] and kept[0]["total"] == 1.0
    old = asyncio.run(trial_runner.run_case(ex, sys.executable, root, "code/simulate.py", cell=case, timeout_s=60,
                                            in_each_run=in_run.of(SEARCH_PROTOCOL)))
    assert old[0] is None   # what the checks passed before: no list, so the run could not be read


def test_a_key_naming_one_setting_twice_is_not_a_level() -> None:
    assert in_run._level_of("width=0,width=74", LEVELS) is None
    assert in_run.axes_of_key("width=0,width=74") is None
    got, why = in_run.read_returned({"spread": {"width=0,width=74": 1.0, "width=80": 2.0, "width=90": 3.0}}, LEVELS, "run_cell")
    assert got is None and "not one of the values" in why
