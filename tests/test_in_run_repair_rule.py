"""A repair of a simulation the model wrote whole (run_trial / run_cell) is told what each run is given when the plan
compares values inside each run, and the one form FI reads back, with a full example. Neutral; no model is called."""

from __future__ import annotations

from tests._engine_text import engine_text
from core import skeleton

PROTOCOL = {"in_each_run": {"step": [0.4, 0.2, 0.1]}}
WHOLE = "def run_trial(cell, trial_id, seed):\n    return {}\n"


def test_a_script_written_whole_gets_the_contract_with_a_full_example() -> None:
    rule = skeleton.level_rule(PROTOCOL, WHOLE)
    assert "[0.4, 0.2, 0.1]" in rule and '"step=0.4": <number>' in rule and '"step=0.1": <number>' in rule
    assert "Never take one value out of the list" in rule and "never key the results by the value first" in rule


def test_no_rule_when_nothing_is_compared_inside_each_run() -> None:
    assert skeleton.level_rule({}, WHOLE) == ""


def test_the_oracle_and_crash_repairs_carry_the_rule() -> None:
    import inspect

    from core import engine

    src = engine_text()
    assert src.count("_skeleton.level_rule(self._protocol_block(state)") >= 2
