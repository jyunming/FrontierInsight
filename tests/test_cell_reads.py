"""The harness hands the simulation a ``Cell`` that remembers which settings the code asked for, FI's small run turns that
into a fact about the grid (a setting nothing ever asked for), and the gate treats it as it treats the reading of the code:
a setting the study compares goes back to be repaired, any other is one line in plan.md. Made-up topic: a colony's growth."""

from __future__ import annotations

import asyncio
import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pytest

from core import in_run, small_run
from core import trial_runner as T
from core.execution import SharedInterpreterExecutor

GRID = {"rate": [0.1, 0.2, 0.3], "dose": [1, 2, 3]}


# ---- the cell ------------------------------------------------------------------------------------------------------

def test_the_cell_is_a_dict_that_remembers_what_was_asked_for() -> None:
    cell = in_run.Cell({"rate": 0.1, "dose": 2})
    assert isinstance(cell, dict) and cell == {"rate": 0.1, "dose": 2}
    assert cell["rate"] == 0.1 and cell.get("dose") == 2 and cell.get("typo", 7) == 7
    assert "other" not in cell
    assert in_run.cell_reads(cell) == {"read": ["dose", "rate"], "unknown_reads": ["other", "typo"], "read_all": False}


def test_a_missing_key_is_recorded_before_it_raises() -> None:
    cell = in_run.Cell({"rate": 0.1})
    with pytest.raises(KeyError):
        cell["rat"]
    assert in_run.cell_reads(cell)["unknown_reads"] == ["rat"]


@pytest.mark.parametrize("take", [dict, lambda c: {**c}, lambda c: list(c), lambda c: c.items(), lambda c: c.keys(),
                                  lambda c: c.values(), lambda c: c.copy(), lambda c: json.dumps(c),
                                  lambda c: copy.deepcopy(c), lambda c: pickle.dumps(c), lambda c: c | {}])
def test_taking_the_cell_as_a_whole_is_recorded(take: Any) -> None:
    cell = in_run.Cell({"rate": 0.1, "dose": 2})
    take(cell)
    assert in_run.cell_reads(cell)["read_all"] is True


def test_a_copy_of_the_cell_is_a_plain_dict() -> None:
    again = pickle.loads(pickle.dumps(in_run.Cell({"rate": 0.1})))
    assert type(again) is dict and again == {"rate": 0.1}


# ---- what the rows say ---------------------------------------------------------------------------------------------

def _row(read: list[str], *, status: str = "ok", whole: bool = False, unknown: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "read": read, "unknown_reads": unknown or [], "read_all": whole}


def test_a_setting_no_successful_trial_asked_for_is_never_read() -> None:
    got = T.read_facts([_row(["dose"]), _row(["dose", "rate"], status="failed")], ["rate", "dose"])
    assert got["measurable"] and got["never"] == ["rate"], "a failed trial's reads stop where it crashed: not evidence"
    assert T.read_facts([_row(["dose"]), _row(["rate"])], ["rate", "dose"])["never"] == []


@pytest.mark.parametrize("rows", [
    [_row([], whole=False)],                    # nothing asked for by name: a C extension may have read the cell
    [_row(["dose"], whole=True)],               # dict(cell): which keys matter cannot be told
    [{"status": "ok", "values": {}}],           # an older run without the record
    [_row(["dose"], status="failed")],          # no trial ran
    [],
])
def test_no_claim_without_positive_evidence(rows: list[dict[str, Any]]) -> None:
    got = T.read_facts(rows, ["rate", "dose"])
    assert not got["measurable"] and got["never"] == [] and got["why"]


# ---- the small run on a real (made-up) simulation -----------------------------------------------------------------

SIM = '''
import random

def run_trial(cell, trial, seed):
    rng = random.Random(seed)
    rate = {rate}
    dose = cell["dose"]
    return {{"growth": rate * dose + rng.random() * 0.01}}
'''


def _small(tmp_path: Path, source: str) -> small_run.SmallRun:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return asyncio.run(small_run.run(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root,
                                     "code/simulate.py", GRID, deterministic=False, timeout_s=120))


@pytest.mark.parametrize("rate, never, asked", [
    ('cell["rate"]', [], []),                       # used
    ("0.2", ["rate"], []),                          # ignored: the value is fixed in the code
    ('cell.get("rate_x", 0.2)', ["rate"], ["rate_x"]),   # a misspelt key with a default
])
def test_the_small_run_says_which_setting_the_code_never_asked_for(tmp_path: Path, rate: str, never: list[str],
                                                                   asked: list[str]) -> None:
    got = _small(tmp_path, SIM.format(rate=rate))
    assert not got.crashed
    assert got.reads["never"] == never and got.reads["unknown"] == asked
    assert got.record()["reads"]["never"] == never
    assert bool(got.unread_note()) == bool(never) and ("`rate`" in got.unread_note() if never else True)
    assert (got.unread_note() in got.for_reader()) if never else True


def test_a_code_that_copies_the_cell_gets_no_never_read_verdict(tmp_path: Path) -> None:
    got = _small(tmp_path, SIM.format(rate="dict(cell)['rate']").replace('cell["dose"]', "2"))
    assert not got.crashed and got.reads["never"] == [] and not got.reads["measurable"] and not got.unread_note()


def test_a_small_run_that_did_not_finish_makes_no_claim(tmp_path: Path) -> None:
    run = _small(tmp_path, SIM.format(rate="0.2"))
    run.settings[-1].timed_out = True
    assert run.reads["never"] == [] and not run.reads["measurable"]


# ---- the gate ------------------------------------------------------------------------------------------------------

def _gate(tmp_path: Path, design: dict[str, Any], never: list[str], *, iteration: int = 0,
          source: str = "def run_trial(cell, t, seed):\n    return {}\n") -> tuple[Any, Any]:
    from tests.test_disclosure import _engine

    eng = _engine(tmp_path)
    eng._draft_protocol = lambda state: {"grid": GRID}  # type: ignore[assignment]
    (tmp_path / "code").mkdir(exist_ok=True)
    (tmp_path / "code" / "simulate.py").write_text(source, encoding="utf-8")
    (tmp_path / "plan.md").write_text("# Plan\n\n## Method\nx\n", encoding="utf-8")
    record = {"reads": {"never": never, "unknown": ["rate_x"], "measurable": True}}
    sent = eng._unread_in_small_run({"design": design, "iteration": iteration, "exec_reflect_iter": 0}, record,
                                    tmp_path / "code" / "simulate.py")
    return eng, sent


def test_a_compared_setting_the_code_never_asked_for_goes_back_to_be_repaired(tmp_path: Path) -> None:
    _eng, sent = _gate(tmp_path, {"comparisons": ["rate"]}, ["rate"])
    assert sent is not None and sent["exec_result"]["failed_script"] == "simulate.py"
    message = sent["exec_result"]["stderr_tail"]
    assert "ran the simulation on a small case" in message and "`rate`" in message and "`rate_x`" in message


def test_another_setting_is_one_line_and_nothing_is_sent(tmp_path: Path) -> None:
    eng, sent = _gate(tmp_path, {}, ["dose"])
    assert sent is None
    assert "the simulation never asked for `dose`" in (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "every setting is kept" in (tmp_path / "plan.md").read_text(encoding="utf-8")


def test_a_redesign_and_a_setting_outside_the_grid_are_left_alone(tmp_path: Path) -> None:
    assert _gate(tmp_path, {"comparisons": ["rate"]}, ["rate"], iteration=1)[1] is None
    other = tmp_path / "b"
    other.mkdir()
    assert _gate(other, {"comparisons": ["ghost"]}, ["ghost"])[1] is None


def test_a_setting_the_code_reads_under_a_condition_is_a_note_not_a_repair(tmp_path: Path) -> None:
    source = "def run_trial(cell, t, seed):\n    if cell['dose'] > 2:\n        return {'x': cell['rate']}\n    return {}\n"
    eng, sent = _gate(tmp_path, {"comparisons": ["rate"]}, ["rate"], source=source)
    assert sent is None, "the code looks `rate` up: the small run did not reach the branch that reads it"
    plan = (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "never asked for `rate`" in plan and "under a condition the small run did not reach" in plan


# ---- more ways to take the cell as a whole; a key built at run time ----------------------------------------------------

@pytest.mark.parametrize("take", [repr, str, lambda c: list(reversed(c))])
def test_a_printed_or_reversed_cell_counts_as_the_whole_cell(take: Any) -> None:
    cell = in_run.Cell({"rate": 0.1, "dose": 2})
    take(cell)
    assert in_run.cell_reads(cell)["read_all"] is True


def test_a_cell_can_be_dumped_as_yaml() -> None:
    yaml = pytest.importorskip("yaml")
    cell = in_run.Cell({"rate": 0.1, "dose": 2})
    assert yaml.safe_load(yaml.safe_dump(cell)) == {"rate": 0.1, "dose": 2}
    assert in_run.cell_reads(cell)["read_all"] is True


DYNAMIC = '''
def run_trial(cell, trial, seed):
    out = 0.0
    for name in ("rate", "dose"):
        out += cell[name + ""] * 1.0
    return {"growth": out}
'''


def test_a_key_built_at_run_time_is_seen_when_the_small_run_reaches_it(tmp_path: Path) -> None:
    got = _small(tmp_path, DYNAMIC)
    assert not got.crashed and got.reads["never"] == [] and got.reads["measurable"], "the text names no key; the run saw both"


def test_a_key_built_at_run_time_that_the_small_run_missed_is_a_known_gap(tmp_path: Path) -> None:
    """Known gap, kept on purpose (bench in the commit message of the hold-back): the hold-back reads literal keys only, so a
    compared setting read through a key built at run time, on a branch the small run did not reach, is sent back once."""
    source = ("def run_trial(cell, t, seed):\n    if cell['dose'] > 2:\n        return {'x': cell['ra' + 'te']}\n    return {}\n")
    _eng, sent = _gate(tmp_path, {"comparisons": ["rate"]}, ["rate"], source=source)
    assert sent is not None


def test_a_cell_dumped_as_yaml_reads_like_the_dict() -> None:
    yaml = pytest.importorskip("yaml")
    plain = {"b": 1, "a": 2.5}
    assert yaml.safe_dump(in_run.Cell(plain)) == yaml.safe_dump(plain)
