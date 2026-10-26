"""A repair of a reader's finding is kept only when the code still behaves as it did where FI measures behaviour
(core/finding_repair.py): a known-answer check that passed before must still pass, and a study that searches for the best
design must not become blind to the design. Neutral code (a mass on a damped spring); the checks are scripted closures that
run the code as it is in the folder, as the engine's measures do."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core import finding_repair as fr
from tests.test_finding_repair import FAILING, GOOD, SOURCE, Run, _block

#: Passes the finding's test (quality(2, 3, 2) == 3.5) and keeps quality(1, 1, 1) == 1 and the dependence on the inputs.
KEEPS = "def quality(m, w, c):\n    return m * w / c + 0.5 * (m - 1)\n"
#: Passes the finding's test and ignores its inputs.
CONSTANT = "def quality(m, w, c):\n    return 3.5\n"


def _quality(run: Run):
    ns: dict[str, Any] = {}
    exec((run.code / "spring.py").read_text(encoding="utf-8"), ns)
    return ns["quality"]


def _answers(run: Run, calls: list[int]):
    """A known-answer check: quality(1, 1, 1) must be 1 (within 1e-6)."""
    async def measure() -> dict[str, dict[str, Any]]:
        calls.append(1)
        value = float(_quality(run)(1, 1, 1))
        return {"unit_quality": {"passed": abs(value - 1.0) <= 1e-6, "value": value},
                "never_passed": {"passed": False, "value": 99.0}}
    return measure


def _response(run: Run):
    """The study's entry on two designs: its objective and one more number."""
    async def measure() -> dict[str, Any]:
        q = _quality(run)
        a, b = float(q(1, 1, 1)), float(q(2, 3, 2))
        return {"quantity": "quality", "values": [a, b], "numbers": [{"quality": [a], "other": [a + 1]},
                                                                    {"quality": [b], "other": [b + 1]}]}
    return measure


def _go(tmp_path: Path, replies: list[str], *, answers: bool = False, response: bool = False):
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(r) for r in replies]})
    calls: list[int] = []
    if answers:
        run.env.known_answers = _answers(run, calls)
    if response:
        run.env.design_response = _response(run)
    return run, run.go(), calls


def test_a_repair_that_breaks_a_known_answer_that_passed_is_refused_and_undone(tmp_path: Path) -> None:
    run, changed, calls = _go(tmp_path, [GOOD, GOOD, GOOD], answers=True)  # GOOD gives quality(1, 1, 1) == 1.5
    assert changed == []
    issue = run.record["issues"]["F-1"]
    assert all(r["outcome"] == "rejected" for r in issue["repairs"])
    why = issue["repairs"][0]["why"]
    assert "known-answer checks that passed before" in why and "`unit_quality`" in why and "1.5" in why
    assert "never_passed" not in why, "a check that already failed before is not held against the repair"
    assert issue["repairs"][0]["measured"]["known_answers"]["unit_quality"] == [1.0, 1.5], "both values are in the record"
    assert (run.code / "spring.py").read_text(encoding="utf-8") == SOURCE


def test_a_repair_that_keeps_every_known_answer_is_still_kept(tmp_path: Path) -> None:
    run, changed, calls = _go(tmp_path, [KEEPS], answers=True, response=True)
    assert changed == ["spring.py"] and run.record["issues"]["F-1"]["outcome"] == "fixed"
    assert len(calls) == 2, "measured once before the first repair and once after it"
    assert run.record["issues"]["F-1"]["repairs"][0]["measured"]["seconds"] >= 0


def test_a_refused_repair_is_retried_and_the_next_one_is_judged_against_the_unchanged_code(tmp_path: Path) -> None:
    run, changed, calls = _go(tmp_path, [GOOD, KEEPS], answers=True)
    assert [r["outcome"] for r in run.record["issues"]["F-1"]["repairs"]] == ["rejected", "accepted"]
    assert len(calls) == 3, "one baseline, one per repair tried"


def test_a_check_that_failed_before_does_not_stop_a_repair_and_nothing_measured_judges_nothing(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(GOOD)]})

    async def failing_before() -> dict[str, dict[str, Any]]:
        return {"c": {"passed": False, "value": 0.0}}

    async def broken() -> dict[str, Any]:
        raise RuntimeError("could not be run")

    run.env.known_answers, run.env.design_response = failing_before, broken
    assert run.go() == ["spring.py"]


def test_a_repair_that_makes_the_study_blind_to_the_design_is_refused(tmp_path: Path) -> None:
    run, changed, _ = _go(tmp_path, [CONSTANT, CONSTANT, CONSTANT], response=True)
    assert changed == []
    why = run.record["issues"]["F-1"]["repairs"][0]["why"]
    assert "blind to the design" in why and "3.5" in why
    assert (run.code / "spring.py").read_text(encoding="utf-8") == SOURCE


def test_the_same_repair_is_kept_when_the_study_was_already_blind_before_it(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(CONSTANT)]})

    async def flat() -> dict[str, Any]:
        return {"quantity": "quality", "values": [1.0, 1.0], "numbers": [{"quality": [1.0]}, {"quality": [1.0]}]}

    run.env.design_response = flat
    assert run.go() == ["spring.py"], "a property of the code, not of the repair"


def test_answers_lost_names_only_what_passed_before_and_fails_after() -> None:
    before = {"a": {"passed": True, "value": 1.0}, "b": {"passed": False, "value": 5.0}, "c": {"passed": None, "value": None},
              "d": {"passed": True, "value": 2.0}, "e": {"passed": True, "value": 3.0}}
    after = {"a": {"passed": False, "value": 1.5}, "b": {"passed": False, "value": 6.0}, "c": {"passed": False, "value": 7.0},
             "d": {"passed": True, "value": 2.0}, "e": {"passed": None, "value": None, "undecided": True}}
    assert [s.split("`")[1] for s in fr.answers_lost(before, after)] == ["a"]
    assert fr.answers_lost(None, after) == [] and fr.answers_lost(before, None) == []
    assert fr.answers_lost(before, {**after, "e": {"passed": None, "value": None}})[-1].startswith("the known-answer check `e`")


def test_design_blind_looks_at_every_number_the_entry_returns_not_the_objective_alone() -> None:
    differ = {"quantity": "o", "values": [0.0, 0.0], "numbers": [{"o": [0.0], "x": [1.0]}, {"o": [0.0], "x": [2.0]}]}
    same = {"quantity": "o", "values": [0.0, 0.0], "numbers": [{"o": [0.0], "x": [1.0]}, {"o": [0.0], "x": [1.0]}]}
    assert fr.design_blind(differ, same) and "blind to the design" in fr.design_blind(differ, same)
    assert fr.design_blind(differ, differ) == "" and fr.design_blind(same, same) == "" and fr.design_blind(same, differ) == ""
    assert fr.design_blind(None, same) == "" and fr.design_blind(differ, None) == ""
