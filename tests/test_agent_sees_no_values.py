"""The protocol shown to a code writer or a repair (``_stripped``) holds no expected value and no tolerance of FI's checks,
and keeps every setting the code is given (review P1-1, review pass 2 N3 and N4)."""
from __future__ import annotations

import json
from pathlib import Path

from core.engine import _stripped

FIXTURES = Path(__file__).parent / "fixtures" / "agent_cheats"


def test_the_check_prose_keeps_its_meaning_without_the_numbers() -> None:
    protocol = json.loads((FIXTURES / "decay" / "protocol.json").read_text(encoding="utf-8"))
    shown = _stripped(protocol)
    o = next(x for x in shown["oracles"] if x["name"] == "noise_free_lnA")
    assert set(o) >= {"name", "kind", "check", "case", "measure"}
    assert not set(o) & {"expected", "expected_formula", "tolerance", "tolerance_mode", "limit"}
    assert "recover ln A exactly" in o["check"] and "1e-6" not in o["check"]


MASKED = "[the value FI checks it against]"


def test_a_number_at_the_end_of_a_sentence_is_masked_too() -> None:
    """review pass 2 N3: ``(?![\\w.])`` after the number left "FWER should be 0.6415." as written."""
    protocol = {"grid": {"m": [20]}, "oracles": [
        {"name": "fwer", "kind": "special_case", "case": {"m": 20}, "expected": 0.6415, "tolerance": 0.015,
         "check": "FWER should be 0.6415. Within 0.015.", "reference": "1-(1-0.05)^20 = 0.6415"}]}
    shown = _stripped(protocol)
    text = json.dumps(shown)
    assert "0.6415" not in text and "0.015" not in text, text
    assert shown["oracles"][0]["check"] == f"FWER should be {MASKED}. Within {MASKED}."


def test_a_setting_the_code_is_given_is_never_masked_even_when_it_equals_a_checks_number() -> None:
    """review pass 2 N4: a setting written as text (a grid value, a fixed condition, a check's case, a value compared
    inside each run) that equals a tolerance or an expected value was masked, so the code writer could not know it."""
    protocol = {
        "grid": {"noise": ["0.25"]},
        "fixed": {"q": "0.01", "note": "Apply BH at q = 0.01 to the m=20 p-values"},
        "in_each_run": {"alpha": ["0.2", "0.5"]},
        "oracles": [
            {"name": "a", "kind": "special_case", "case": {"pi1": "0.3"}, "expected": 0.25, "tolerance": 0.01,
             "check": "with noise sd 0.25 and q = 0.01, the rate is 0.25"},
            {"name": "b", "kind": "special_case", "case": {"m": 20}, "expected": 0.3, "tolerance": 0.2,
             "check": "at pi1 0.3 the power is 0.3 within 0.2"},
            {"name": "c", "kind": "special_case", "case": {"m": 20}, "expected": 0.7385, "tolerance": 0.0031,
             "check": "c is 0.7385 within 0.0031"}]}
    shown = _stripped(protocol)
    assert shown["fixed"]["note"] == "Apply BH at q = 0.01 to the m=20 p-values"
    assert shown["oracles"][0]["check"] == "with noise sd 0.25 and q = 0.01, the rate is 0.25"
    assert shown["oracles"][1]["check"] == "at pi1 0.3 the power is 0.3 within 0.2"
    assert shown["oracles"][2]["check"] == f"c is {MASKED} within {MASKED}", "a number that is no setting is masked"
