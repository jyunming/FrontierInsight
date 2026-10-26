"""The measures a repair is judged by (core/finding_repair.py) observe the code and change nothing: they run in a throwaway
copy of the quest; a case the clock cut off is undecided, not lost; the baseline is the code just before the repair being
judged; a number that is not finite makes the design comparison undecided. Neutral code."""
from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from typing import Any

from core import code_checks, finding_repair as fr
from core import optimise as optimise_mod
from core.execution import VenvExecutor
from tests.test_finding_repair import FAILING, Run, _block

KEEPS = "def quality(m, w, c):\n    return m * w / c + 0.5 * (m - 1)\n"

SIM = '''
import pathlib, time

def run_cell(cell):
    if cell.get("a") == 100.0:
        time.sleep(60)
    if cell.get("a") == -1.0:
        raise RuntimeError("the simulation fails here")
    if cell.get("a") == -2.0:
        raise RuntimeError("the study's time (execution.timeout_s) ran out; the case ran out of time")
    for rel in ("data/results/cache/v.json", "data/written.txt", "raw/x.txt", "stray.txt", "code/stray.txt"):
        p = pathlib.Path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("written by the run")
    return {"y": cell["a"] * 2.0, "obj": cell["a"] + 1.0}
'''
PROTOCOL = {"oracles": [
    {"name": "double", "expected": 2.0, "tolerance": 1e-6, "case": {"a": 1.0}, "measure": "y"},
    {"name": "slow", "expected": 200.0, "tolerance": 1e-6, "case": {"a": 100.0}, "measure": "y"},
    {"name": "crashes", "expected": -2.0, "tolerance": 1e-6, "case": {"a": -1.0}, "measure": "y"},
    {"name": "spoof", "expected": -4.0, "tolerance": 1e-6, "case": {"a": -2.0}, "measure": "y"}]}
BLOCK = {"objective": {"quantity": "obj", "direction": "maximise"},
         "design_variables": [{"name": "a", "low": 0.0, "high": 10.0, "kind": "continuous"}],
         "baseline": {"values": {"a": 1.0}}}


class _Ex(VenvExecutor):
    def python_path(self, root: Path) -> Path:
        return Path(sys.executable)


def _quest(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "input.txt").write_text("an input", encoding="utf-8")
    return root


def _state(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_measures_leave_the_quest_as_it_was_and_a_cut_off_case_is_undecided(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    before = _state(root)
    answers = asyncio.run(code_checks.known_answer_values(
        root, PROTOCOL, executor=_Ex(), python=sys.executable, env={}, timeout_s=8))
    assert _state(root) == before, "the known-answer cases wrote into the quest"
    assert answers["double"]["passed"] is True and answers["double"]["value"] == 2.0
    # the clock cut `slow` off: nothing is learned from it; `crashes` ran and gave nothing: that is behaviour
    assert answers["slow"]["undecided"] is True and answers["slow"]["passed"] is None
    assert answers["crashes"]["undecided"] is False and answers["crashes"]["passed"] is None and answers["crashes"]["not_run"]
    # a program that raises the very words of a time-out is a case that ran and failed, not one FI stopped
    assert answers["spoof"]["undecided"] is False and "ran out of time" in answers["spoof"]["not_run"]
    response = asyncio.run(optimise_mod.design_response(
        _Ex(), sys.executable, root, "code/simulate.py", BLOCK, timeout_s=60, env=None, thresholds=None))
    assert _state(root) == before, "the design measure wrote into the quest"
    assert response and response["quantity"] == "obj" and response["values"][0] == 2.0 and response["values"][1] != 2.0


def test_an_answer_the_clock_cut_off_is_not_held_against_a_repair_and_a_crash_is() -> None:
    before = {"a": {"passed": True, "value": 1.0}, "b": {"passed": True, "value": 2.0}}
    cut = {"a": {"passed": None, "value": None, "undecided": True, "not_run": "ran out of time"},
           "b": {"passed": None, "value": None, "undecided": False, "not_run": "load error"}}
    lost = fr.answers_lost(before, cut)
    assert len(lost) == 1 and "`b`" in lost[0]


def test_a_repair_with_a_check_cut_off_after_it_is_kept_and_says_so(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)],
                                          "code:repair": [_block("def quality(m, w, c):\n    return m * w / c + 0.5 * (m - 1)\n")]})
    log: list[str] = []

    class _Log:
        def __getattr__(self, name: str) -> Any:
            return lambda msg, *a, **k: log.append(msg % a if a else msg)

    run.env.log = _Log()
    calls: list[int] = []

    async def measure() -> dict[str, dict[str, Any]]:
        calls.append(1)
        if len(calls) == 1:
            return {"c": {"passed": True, "value": 1.0}}
        return {"c": {"passed": None, "value": None, "undecided": True, "not_run": "ran out of time"}}

    run.env.known_answers = measure
    assert run.go() == ["spring.py"]
    measured = run.record["issues"]["F-1"]["repairs"][0]["measured"]
    assert measured["not_decided"] == {"c": "ran out of time"}
    assert any("could not be decided" in line for line in log)


DAMPING_FAILING = "import spring\n\ndef test_d():\n    assert spring.damping(1, 1, 2) == 2.0, 'the factor is wrong'\n"


def test_the_baseline_is_the_code_just_before_the_repair_even_when_the_last_measure_was_missing(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1"}, {"id": "F-2", "function": "damping", "text": "damping is wrong"}],
              {"code:test": [_block(FAILING), _block(DAMPING_FAILING)],
               "code:repair": [_block("def quality(m, w, c):\n    return m * w / c + 0.5 * (m - 1)\n"),
                               _block("def damping(m, k, c):\n    return c / math.sqrt(m * k)\n")]})
    seen: list[float] = []

    async def measure() -> dict[str, dict[str, Any]] | None:
        ns: dict[str, Any] = {}
        exec((run.code / "spring.py").read_text(encoding="utf-8"), ns)
        seen.append(ns["quality"](2, 3, 2))
        # the measure after the first accepted repair is not available
        return None if len(seen) == 2 else {"c": {"passed": True, "value": 1.0}}

    run.env.known_answers = measure
    assert run.go() == ["spring.py"]
    assert [r["outcome"] for i in ("F-1", "F-2") for r in run.record["issues"][i]["repairs"]] == ["accepted", "accepted"]
    # baseline (code before F-1), after F-1 (not available), the baseline of F-2 measured fresh on the code after F-1, after F-2
    assert seen == [3.0, 3.5, 3.5, 3.5]


def test_a_number_that_is_not_finite_makes_the_design_comparison_undecided() -> None:
    differ = {"quantity": "o", "values": [1.0, 2.0], "numbers": [{"o": 1.0}, {"o": 2.0}]}
    nan = {"quantity": "o", "values": [math.nan, math.nan], "numbers": [{"o": math.nan}, {"o": math.nan}]}
    assert fr.design_blind(differ, nan) == "" and fr.design_blind(nan, differ) == ""
    assert math.isnan(optimise_mod._leaves({"k": [math.nan, 1.0]})["k[0]"])


def test_a_value_that_is_not_finite_is_not_evidence_either_way_but_the_finite_ones_still_decide() -> None:
    nan = math.nan
    before = {"quantity": "o", "values": [0.0, 0.0],
              "numbers": [{"o": 0.0, "x": nan, "flag": 0.0}, {"o": 0.0, "x": 1.5, "flag": 1.0}]}
    blind = {"quantity": "o", "values": [0.0, 0.0], "numbers": [{"o": 0.0, "x": 2.0, "y": nan}, {"o": 0.0, "x": 2.0, "y": 3.0}]}
    all_nan = {"quantity": "o", "values": [nan, nan], "numbers": [{"o": nan, "x": nan}, {"o": nan, "x": nan}]}
    assert "blind to the design" in fr.design_blind(before, blind), "finite values identical on both designs, after differing before"
    assert fr.design_blind(before, all_nan) == "", "nothing finite to compare after: undecided, not identical"


def _paths(a: Any, b: Any) -> dict[str, bool]:
    return fr._compared([optimise_mod._leaves(a), optimise_mod._leaves(b)])


def test_designs_are_compared_leaf_by_leaf_by_path() -> None:
    assert _paths({"m": [7]}, {"m": [7, 9]}) == {"m[0]": False}, "a longer list is not a different answer by itself"
    assert _paths({"m": {"left": 1}}, {"m": {"right": 2}}) == {}, "no leaf is on both"
    assert _paths({"m": [math.nan, 1]}, {"m": [0, 2]}) == {"m[1]": True}, "the finite leaf decides"
    assert _paths({"m": [1, 2]}, {"m": [2, 1]}) == {"m[0]": True, "m[1]": True}, "by path, not by the list as a whole"
    before = {"quantity": "m", "values": [None, None], "numbers": [optimise_mod._leaves({"m": [7]}), optimise_mod._leaves({"m": [7, 9]})]}
    assert fr.design_blind(before, before) == ""


def test_a_copy_that_cannot_be_made_or_deleted_is_said_and_kept_in_the_record(tmp_path: Path, monkeypatch: Any) -> None:
    from core import outside_run

    root = _quest(tmp_path)
    notes: list[str] = []
    monkeypatch.setattr(outside_run, "copy_code", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    try:
        with outside_run.observation_copy(root, notes=notes):
            raise AssertionError("not reached")
    except OSError:
        pass
    assert notes and "could not be copied" in notes[0] and "disk full" in notes[0]
    # through the loop: a measure that raises is undecided, and the reason is in the record of the try
    run = Run(tmp_path / "loop", [{"id": "F-1"}], {"code:test": [_block(FAILING)],
                                                   "code:repair": [_block(KEEPS)]})

    async def broken() -> dict[str, Any]:
        raise OSError("disk full")

    run.env.known_answers = broken
    assert run.go() == ["spring.py"]
    assert any("disk full" in n for n in run.record["issues"]["F-1"]["repairs"][0]["measured"]["not_measured"])
