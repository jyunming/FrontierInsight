"""The code reader's findings about a result's name are matched to FI's results by FI's own record, and a reader's word
about a name alone never stops a quest (group G1).

* A quantity the plan names with one word fewer than the key the run reports it under (``loop_gain_error`` for
  ``loop_gain_empirical_error``) is that key when it is the only one: no synonym table.
* A plan name written as a sentence is no key; a reader's "missing" for it (or for a planned figure) is a part not done,
  which the paper states, even for the main quantity. A reader's "different" still stops on the main quantity.
* A quantity's own key holding no number (an empty list where the plan asked a count) is left out, not a stop; a fault
  whose quantity the results hold no key for still stops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core import code_findings
from core import code_review as cr
from tests.test_shown_fault_names import _gate_on

SENTENCE = "the mean time the loop takes to settle, for every setting"


def test_a_name_with_one_more_word_in_the_results_is_the_quantity_when_it_is_the_only_one() -> None:
    """Old code: ``loop_gain_error`` found nothing in ``loop_gain_empirical_error`` (only a key that begins with the name
    counted), so the reader's "missing" stood and could stop the quest."""
    result = {"loop_gain_empirical_error": [0.1, 0.2], "loop_gain_empirical_error_ci95": [0.05, 0.3], "n_trials": 5}
    assert cr.named_in_results("loop_gain_error", result) == {"loop_gain_empirical_error", "loop_gain_empirical_error_ci95"}


def test_two_keys_that_could_each_be_the_name_are_not_chosen_between() -> None:
    """Known negative: with two candidates FI does not guess; a one-word name never matches a longer key."""
    assert cr.named_in_results("loop_gain_error", {"loop_gain_empirical_error": 0.1, "loop_gain_raw_error": 0.2}) == set()
    assert cr.named_in_results("gain", {"open_loop_gain": 1.0}) == set()
    assert cr.named_in_results("loop_gain_error", {"loop_error": 0.1}) == set()  # words of the key not all in the name


def test_a_sentence_is_no_key_and_an_identifier_is() -> None:
    assert cr.names_a_key("settle_time") and cr.names_a_key("Hit-Rate")
    assert not cr.names_a_key(SENTENCE) and not cr.names_a_key("")


def _missing_record(kind: str, name: str, text: str, headline: bool) -> dict[str, Any]:
    return {"read_problems": [{"fid": "F-1", "kind": kind, "status": "missing", "name": name, "text": text,
                               "headline": headline}]}


def test_a_missing_figure_is_a_part_not_done_and_a_wrong_kind_of_finding_is_not() -> None:
    """Old code: only a quantity could be a part not done; a planned figure called missing (a file-name finding) was
    handed to the key search and stopped the quest as 'cannot find which results it changes'."""
    text = "the script draws the planned figure settle_curve.png"
    assert cr.missing_quantities(_missing_record("figure", "", text, False), {"id": "F-1"}) == [text]
    assert cr.missing_quantities(_missing_record("metric", "settle_time", "x", False), {"id": "F-1"}) == ["settle_time"]
    different = _missing_record("metric", "settle_time", "x", False)
    different["read_problems"][0]["status"] = "different"
    assert cr.missing_quantities(different, {"id": "F-1"}) == []
    logic = _missing_record("logic", "settle_time", "x", False)
    assert cr.missing_quantities(logic, {"id": "F-1"}) == []


def test_a_missing_main_quantity_is_not_about_the_headline_but_a_different_one_is() -> None:
    """Old code: ``issue["headline"]`` made every issue of the main quantity a stop, whatever the reader said of it."""
    issue = {"id": "F-1", "headline": True, "quantity": SENTENCE}
    missing = _missing_record("dependent variable", SENTENCE, "the dependent variable is computed", True)
    assert not cr.about_headline(missing, issue, {"gain": 1.0}, [SENTENCE])
    missing["read_problems"][0]["status"] = "different"
    assert cr.about_headline(missing, issue, {"gain": 1.0}, [SENTENCE])


def test_the_reader_is_contradicted_by_a_key_with_one_more_word_in_it(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("x = 1\n", encoding="utf-8")
    problem = {"id": "R1", "kind": "metric", "status": "missing", "headline": True, "name": "loop_gain_error",
               "text": "the result reports loop_gain_error"}
    cr.save(tmp_path, {"final": True, "version": cr.version(code), "problems": [dict(problem)]})
    assert cr.resolve_headline(tmp_path, code, {"loop_gain_empirical_error": [0.1, 0.2]}) == []
    assert cr.load(tmp_path)["problems"][0]["contradicted"] is True
    cr.save(tmp_path, {"final": True, "version": cr.version(code), "problems": [dict(problem)]})
    assert cr.resolve_headline(tmp_path, code, {"n_trials": 5}) == [problem["text"]]  # known positive: still absent


def _issue(**over: Any) -> dict[str, Any]:
    base = {"id": "F-1", "changes_result": True, "demonstrated": True, "failure_kind": code_findings.ASSERTION,
            "outcome": code_findings.GAVE_UP, "plan_conflict": False, "readings": [{"n": 1, "status": "open"}],
            "severity": "high", "first_seen": 1, "file": "experiment.py", "function": "main", "text": "a finding",
            "test": ".fi/finding_tests/F-1_1.py"}
    return {**base, **over}


def test_a_missing_sentence_named_main_quantity_does_not_stop_the_quest(tmp_path: Path, monkeypatch: Any) -> None:
    """Old code: the issue was about the headline, so a failing name test stopped the quest with no paper, on the
    reader's word that a sentence-named quantity is not there. Now it is a part not done, which the paper states."""
    cr.save(tmp_path, {**_missing_record("dependent variable", SENTENCE, "the dependent variable is computed", True),
                       "issues": {"F-1": _issue(headline=True, quantity=SENTENCE)}})
    _root, verdict, _eng = _gate_on(tmp_path, {"gain": 1.0}, monkeypatch)
    assert verdict is None
    assert cr.load(tmp_path)["issues"]["F-1"]["at_gate"]["outcome"] == "not_computed"


def test_a_different_main_quantity_still_stops(tmp_path: Path, monkeypatch: Any) -> None:
    """Known positive: the reader says the main quantity is computed differently and a test showed it: stop."""
    record = _missing_record("dependent variable", SENTENCE, "the dependent variable is computed", True)
    record["read_problems"][0]["status"] = "different"
    cr.save(tmp_path, {**record, "issues": {"F-1": _issue(headline=True, quantity=SENTENCE)}})
    _root, verdict, _eng = _gate_on(tmp_path, {"gain": 1.0}, monkeypatch)
    assert verdict is not None and verdict["stuck_reason"] == "shown_fault_open"


def test_a_count_reported_as_an_empty_list_is_left_out_not_a_stop(tmp_path: Path, monkeypatch: Any) -> None:
    """The quantity's own key holds no number (the plan asked a count; the code reported a list, here empty): it is the
    result the shown fault is about, so it is left out. A fault whose quantity has no key at all still stops."""
    cr.save(tmp_path, {"issues": {"F-1": _issue(quantity="blocked_count", kind="logic")}})
    _root, verdict, _eng = _gate_on(tmp_path, {"blocked_count": [], "gain": 1.0}, monkeypatch)
    assert verdict is None
    assert cr.load(tmp_path)["issues"]["F-1"]["at_gate"]["outcome"] == "left_out"


def test_a_fault_whose_quantity_has_no_key_still_stops(tmp_path: Path, monkeypatch: Any) -> None:
    """Known positive (a measured fault in a check's input FI cannot map to a result): the conservative stop stays."""
    cr.save(tmp_path, {"issues": {"F-1": _issue(quantity="printable_count", kind="logic")}})
    _root, verdict, _eng = _gate_on(tmp_path, {"blocked_count": [], "gain": 1.0}, monkeypatch)
    assert verdict is not None and "cannot find which of the results it changes" in verdict["rationale"]
