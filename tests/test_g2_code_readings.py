"""Readings of free code text (a loop's literal values, a generator built from a constant) neither stop a quest nor send
its code back; what can be measured from FI's record of the trials is measured there (core/stop_power.py)."""

from __future__ import annotations

from core import protocol_check as pc
from core import run_facts

PROTOCOL = {"grid": {"gain": [0.0, 0.1, 0.2]}, "runs_per_setting": 50}

# The grid is assigned under its own name; the acceptance function loops over two literal values because its property is
# undefined at gain = 0. Nothing about the grid is wrong.
ACCEPTANCE_LOOP = """
GAIN_LIST = [0.0, 0.1, 0.2]
RUNS = 50

def check_acceptance(table):
    for gain in (0.1, 0.2):
        assert table[gain] > 0

def main():
    for g in GAIN_LIST:
        pass
"""

# The same loop is the only place the script names the grid: a real difference, but only a reading of the code.
LOOP_ONLY_GRID = """
RUNS = 50

def main():
    for gain in (0.1, 0.3):
        pass
"""

ASSIGNED_WRONG_GRID = """
GAIN_LIST = [0.5, 0.9]
RUNS = 50
"""


def test_acceptance_loop_over_literals_is_not_a_repair() -> None:
    """Known positive for the pass side: the assigned list is the grid and it matches, so the acceptance loop adds nothing."""
    scripts = {"experiment.py": ACCEPTANCE_LOOP}
    assert pc.check_repairable(PROTOCOL, scripts) == []


def test_loop_literals_that_differ_are_told_not_repaired() -> None:
    """Old code: a difference found only in a loop's literal values was a repairable difference. It is now a note."""
    scripts = {"experiment.py": LOOP_ONLY_GRID}
    assert any(m.from_loop for m in pc.check(PROTOCOL, scripts))
    assert pc.check_repairable(PROTOCOL, scripts) == []
    (note,) = pc.loop_hints(PROTOCOL, scripts)
    assert "only a reading of the code" in note


def test_an_assigned_list_that_differs_is_still_repairable() -> None:
    """Known positive: a list assigned under the axis's name that contradicts the plan still sends the script back."""
    found = pc.check_repairable(PROTOCOL, {"experiment.py": ASSIGNED_WRONG_GRID})
    assert len(found) == 1 and found[0].kind == "grid" and not found[0].from_loop
    assert pc.loop_hints(PROTOCOL, {"experiment.py": ASSIGNED_WRONG_GRID}) == []


def _cell(gain: float, values: list[float]) -> dict:
    return {"cell": {"gain": gain}, "metrics": {"hit_rate": {"values": values, "trials": list(range(len(values)))}}}


def _noise(seed: int, n: int = 30) -> list[float]:
    import random
    r = random.Random(seed)
    return [r.gauss(0, 1) for _ in range(n)]


def test_settings_with_the_same_random_numbers_are_found_in_the_record() -> None:
    """Old code: only a reading of the code could say settings share random numbers. Here the per-trial values of three
    settings are the same noise plus an offset, which the record shows."""
    base = _noise(1)
    cells = [_cell(0.0, base), _cell(0.1, [v + 1 for v in base]), _cell(0.2, [2 * v for v in base])]
    (note,) = run_facts.shared_stream_notes({"cells": cells})
    assert "hit_rate" in note and "same random numbers" in note


def test_independent_settings_give_no_note() -> None:
    """Known negative: settings that drew their own random numbers are not reported."""
    cells = [_cell(0.0, _noise(1)), _cell(0.1, _noise(2)), _cell(0.2, _noise(3))]
    assert run_facts.shared_stream_notes({"cells": cells}) == []
    assert run_facts.shared_stream_notes(None) == [] and run_facts.shared_stream_notes([{"cell": {}}]) == []
