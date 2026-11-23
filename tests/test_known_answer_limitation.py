"""A check of the code against a known answer that the quest went on without (it failed, it could not be run on its case,
or the trials could not decide it) is an item of FI's own record, in the plan's words and FI's numbers, that QUALITY.md
lists. Before, saying it rested on the writer model alone, and recorded papers left out the checks that were not run or
not decided. Neutral made-up study: a pond's algae."""

from __future__ import annotations

import json
from pathlib import Path

from core import accepted_checks as ac
from core import paper_facts as pf

ORACLES = [
    {"name": "peak_closed_form", "check": "The peak algae count at nutrient 1 equals the logistic closed form. It uses 400 runs."},
    {"name": "bloom_rate", "check": "The bloom chance at the lowest nutrient is 0.2. The tolerance is three standard errors."},
    {"name": "growth_doubling", "check": "Doubling the nutrient doubles the growth rate"},
    {"name": "decay_working", "check": "The decay constant matches the plan's own working"},
]
WENT_ON = [
    {"name": "peak_closed_form", "expected": 120.0, "tolerance": 2.0, "tolerance_mode": "absolute", "measured": 131.5,
     "by": ac.AUTOMATIC},
    {"name": "bloom_rate", "expected": 0.2, "tolerance": 0.05, "tolerance_mode": "relative", "measured": 0.212,
     "undecided": True, "by": ac.AUTOMATIC},
    {"name": "growth_doubling", "expected": 2.0, "tolerance": 0.1, "measured": None, "unmeasured": "no case", "by": ac.AUTOMATIC},
    {"name": "decay_working", "expected": 0.3, "tolerance": 0.01, "measured": 0.5, "disagreement": "they disagree",
     "by": ac.AUTOMATIC},
]


def _quest(root: Path, went_on: list[dict], status: str = ac.WENT_ON, oracles: list[dict] = ORACLES) -> Path:
    (root / "needs").mkdir(parents=True, exist_ok=True)
    (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({"status": status, "went_on": went_on}), encoding="utf-8")
    (root / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps({"protocol": {"oracles": oracles}}), encoding="utf-8")
    return root


def _said(root: Path) -> list[str]:
    return [i.sentence for i in pf.required_items(root) if i.kind == pf.KNOWN_ANSWER]


def test_each_check_the_quest_went_on_without_is_said_by_fi_in_its_own_words(tmp_path: Path) -> None:
    said = _said(_quest(tmp_path, WENT_ON))
    assert said == [
        'The check of the code against a known answer "The peak algae count at nutrient 1 equals the logistic closed form" '
        "did not pass (the run gave 131.5, the known value is 120 within 2); the results are not validated against it.",
        'The check of the code against a known answer "The bloom chance at the lowest nutrient is 0.2" could not be decided '
        "with the trials the study ran (the run gave 0.212, the known value is 0.2 within 0.05 (relative)); the results are "
        "not validated against it.",
        'The check of the code against a known answer "Doubling the nutrient doubles the growth rate" could not be run on its '
        "case; the results are not validated against it.",
        'The check of the code against a known answer "The decay constant matches the plan\'s own working" could not be '
        "judged: the simulation and the plan's own working of the known value disagree, and neither is confirmed; the results "
        "are not validated against it.",
    ]
    assert sum("did not pass" in s for s in said) == 1, "a check not run or not decided is never said to have failed"


def test_nothing_is_said_when_every_check_passed_or_none_was_recorded(tmp_path: Path) -> None:
    assert _said(_quest(tmp_path / "ok", WENT_ON, status="ok")) == []
    assert _said(tmp_path / "none") == []


def test_a_check_the_plan_gave_no_words_for_is_named_and_many_are_one_item(tmp_path: Path) -> None:
    said = _said(_quest(tmp_path / "bare", WENT_ON[:1], oracles=[]))
    assert said and '"peak_closed_form" did not pass' in said[0]
    many = [{**WENT_ON[2], "name": f"c{i}"} for i in range(pf._MAX_SETTING_ITEMS + 1)]
    said = _said(_quest(tmp_path / "many", many, oracles=[]))
    assert len(said) == 1 and said[0].startswith(f"{len(many)} checks of the code against a known answer")


def test_a_record_that_no_check_could_be_judged_is_said_too(tmp_path: Path) -> None:
    # The gate writes an entry with no name when no check of the plan could be judged at all.
    said = _said(_quest(tmp_path, [{"name": "", "measured": None, "unmeasured": "no check could be judged"}]))
    assert said == ["No check of the code against a known answer could be judged, so the results are not validated "
                    "against one."]


def test_the_plans_first_sentence_is_kept_whole_through_a_semicolon(tmp_path: Path) -> None:
    oracles = [{"name": "peak_closed_form", "check": "The total algae is conserved; All inflows are zero. It uses 40 runs."}]
    said = _said(_quest(tmp_path, WENT_ON[:1], oracles=oracles))
    assert '"The total algae is conserved; All inflows are zero"' in said[0]


def test_no_check_judged_is_never_counted_or_folded_in_as_a_check(tmp_path: Path) -> None:
    unnamed = {"name": "", "measured": None, "unmeasured": "no check could be judged"}
    many = [{**WENT_ON[2], "name": f"c{i}"} for i in range(pf._MAX_SETTING_ITEMS + 1)]
    said = _said(_quest(tmp_path, [*many, unnamed], oracles=[]))
    assert len(said) == 2 and said[0].startswith(f"{len(many)} checks") and said[1].startswith("No check of the code")


def test_many_checks_folded_into_one_item_never_say_a_check_failed_that_was_not_run(tmp_path: Path) -> None:
    not_run = [{**WENT_ON[2], "name": f"c{i}"} for i in range(pf._MAX_SETTING_ITEMS + 1)]
    said = _said(_quest(tmp_path / "not_run", not_run, oracles=[]))
    assert len(said) == 1 and "fail" not in said[0] and "did not pass" not in said[0], said
    assert f"({len(not_run)} could not be run on their cases)" in said[0], said
    mixed = [*not_run, {**WENT_ON[0], "name": "p"}, {**WENT_ON[1], "name": "u"}]
    said = _said(_quest(tmp_path / "mixed", mixed, oracles=[]))
    assert (f"({len(not_run)} could not be run on their cases, 1 did not pass, 1 could not be decided with the trials the "
            "study ran)") in said[0], said
    one = [{**WENT_ON[2], "name": "c"}, *({**WENT_ON[0], "name": f"p{i}"} for i in range(pf._MAX_SETTING_ITEMS))]
    said = _said(_quest(tmp_path / "one_not_run", one, oracles=[]))
    assert "(1 could not be run on its case, " in said[0], said
