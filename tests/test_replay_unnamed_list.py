"""Replay of two real quests that were stopped, after two repairs and with no paper, because a list in the script that is
not named after a protocol axis happened to share values with it (a loop over four output positions read as an axis of
five settings; the values of one axis read as a one-value axis). In tests/fixtures/replay/unnamed_list_false_match/ each
case holds the quest's scripts and protocol grid. Controls show a real contradiction is still caught. No model is called."""
from __future__ import annotations

import json
from pathlib import Path

from core import protocol_check

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "unnamed_list_false_match"


def _case(name: str) -> tuple[dict, dict[str, str]]:
    folder = FIXTURE / name
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    scripts = {f.name: f.read_text(encoding="utf-8") for f in folder.glob("*.py")}
    return protocol, scripts


def _grid(mismatches) -> list:
    return [m for m in mismatches if m.kind == "grid"]


def test_the_evidence_is_what_the_real_quests_held() -> None:
    protocol, scripts = _case("case_a")
    assert protocol["grid"]["temperature_index"] == [0, 1, 2, 3, 4]
    assert "for multiplier in (0, 1, 2, 3):" in scripts["simulate.py"]
    assert "def run_trial" in scripts["simulate.py"]
    protocol, scripts = _case("case_b")
    assert protocol["grid"]["alpha"] == [0.05]
    assert "PI1_VALUES = (0.0, 0.05, 0.2)" in scripts["experiment.py"]
    assert "def run_trial" in scripts["simulate.py"]


def test_positions_loop_is_not_an_axis() -> None:
    protocol, scripts = _case("case_a")
    assert [m for m in _grid(protocol_check.check(protocol, scripts)) if m.name == "temperature_index"] == []


def test_another_axis_values_are_not_a_one_value_axis() -> None:
    protocol, scripts = _case("case_b")
    assert _grid(protocol_check.check(protocol, scripts)) == []


def test_trial_contract_still_catches_a_list_named_after_the_axis() -> None:
    protocol, scripts = _case("case_a")
    scripts = {**scripts, "experiment.py": scripts["experiment.py"] + "\nTEMPERATURE_INDEX_VALUES = [0, 1, 2, 3, 7]\n"}
    found = [m for m in protocol_check.check(protocol, scripts) if m.name == "temperature_index"]
    assert len(found) == 1 and found[0].found == [0, 1, 2, 3, 7]


def test_outside_the_trial_contract_an_oddly_named_list_is_still_caught() -> None:
    protocol, scripts = _case("case_a")
    scripts = {k: v.replace("def run_trial", "def run_one") for k, v in scripts.items()}
    scripts["experiment.py"] += "\nSETTINGS_TO_RUN = [0, 1, 2, 3, 9]\n"
    found = [m for m in protocol_check.check(protocol, scripts) if m.name == "temperature_index"]
    assert len(found) == 1 and found[0].found == [0, 1, 2, 3, 9]


def test_outside_the_trial_contract_a_one_value_axis_is_not_matched_by_value() -> None:
    protocol, scripts = _case("case_b")
    scripts = {k: v.replace("def run_trial", "def run_one") for k, v in scripts.items()}
    assert [m for m in protocol_check.check(protocol, scripts) if m.name == "alpha"] == []


def test_outside_the_trial_contract_a_list_equal_to_another_axis_is_not_this_axis() -> None:
    protocol = {"grid": {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0]}}
    scripts = {"x.py": "ODD = [1, 2, 3, 4]\nOTHER = [1, 2, 3]\n"}
    assert [m for m in protocol_check.check(protocol, scripts) if m.found] == []  # no list is read as the axis
    scripts = {"x.py": "ODD = [1, 2, 3, 5]\nOTHER = [1, 2, 3]\n"}  # b's values with one changed: a real contradiction
    assert [m.name for m in protocol_check.check(protocol, scripts)] == ["b"]


def test_outside_the_trial_contract_a_loop_that_hands_its_value_on_is_still_an_axis() -> None:
    protocol = {"grid": {"a": [1.0, 2.0, 3.0, 4.0]}}
    scripts = {"x.py": "for v in (1, 2, 3, 9):\n    run(v)\n"}
    assert [m.name for m in protocol_check.check(protocol, scripts)] == ["a"]
    scripts = {"x.py": "for v in (1, 2, 3, 9):\n    out[f'k{v}'] = x[v]\n"}
    assert [m for m in protocol_check.check(protocol, scripts) if m.found] == []  # no list is read as the axis
