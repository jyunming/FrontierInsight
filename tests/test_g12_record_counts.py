"""Counting the trials the analysis lists against FI's own record: the very same list under two sibling keys is one list,
and a setting whose value is an array is compared by value, not by how it is spelled."""

from __future__ import annotations

from core import run_manifest, trial_runner

LIST = [0.1, 0.2, 0.3, 0.4, 0.5]


def test_the_same_list_under_two_sibling_keys_is_counted_once() -> None:
    """Old code added the sibling lists up as 'parts of one report', so 5 recorded trials listed under keys "0" and "1"
    were counted as 10 and an honest short run passed the lower bound (or a copy looked like an excess)."""
    result = {"m_values": {"0": LIST, "1": list(LIST)}}
    account = run_manifest._account(result, "m", {}, (5, 0), None)
    assert (account["acc_defined"], account["acc_undefined"]) == (5, 0)
    nested = {"a": {"m_values": LIST}, "b": {"m_values": list(LIST)}}
    assert run_manifest._account(nested, "m", {}, (5, 0), None)["acc_defined"] == 5


def test_two_arms_with_equal_values_are_two_sets_of_trials() -> None:
    """Two independent arms can give the same deterministic values: FI's record (10 trials) says they are 10, so equal
    values are never taken as a copy."""
    result = {"m_values": {"arm_a": [1, 2, 3, 4, 5], "arm_b": [1, 2, 3, 4, 5]}}
    assert run_manifest._account(result, "m", {}, (10, 0), None)["acc_defined"] == 10


def test_different_sibling_lists_still_add_up() -> None:
    """Known positive: two different lists side by side are two sets of trials (the rows of one table) and still add."""
    result = {"m_values": {"0": [1.0, 2.0, 3.0], "1": [4.0, 5.0, 6.0]}}
    assert run_manifest._account(result, "m", {}, (6, 0), None)["acc_defined"] == 6


def test_an_array_setting_is_read_by_value_whichever_way_it_is_spelled() -> None:
    """Old code read the grid's ``[0.5, 0.8]`` as the text '[0.5, 0.8]' and the record's cell key ``[0.5;0.8]`` as another,
    so a setting with an array value was never found as a copy."""
    key_text = trial_runner.cell_key({"w": [0.5, 0.8]}).split("=", 1)[1]
    assert key_text == "[0.5;0.8]"
    assert trial_runner._level_text([0.5, 0.8]) == trial_runner._level_text(key_text)
    assert trial_runner._level_text([1, 2]) == trial_runner._level_text("[1.0;2.0]")
    assert trial_runner._level_text([0.5, 0.8]) != trial_runner._level_text("[0.5;0.9]")
    assert trial_runner._level_text(0.0) == trial_runner._level_text("0")
