"""A name the plan appears to define twice (a dependent variable written with one formula, a protocol metric of the same name
with another), or a finding marked as one about the plan, is no exemption by name: a value a second model flagged stays only
when FI holds a check it measured itself that passed and reads the quantity, and what the paper says is only what FI knows.
Neutral made-up study; no model is called."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core import code_review as cr
from core import code_findings, frozen_protocol, paper_facts, plan
from core.result_checks import ResultChecksMixin

METRIC_A = {"id": "quantity_a", "kind": "mean", "unit": "one trial", "estimand": "(value_exact - base)/base at one setting, the closed form"}
METRIC_B = {"id": "quantity_b", "kind": "mean", "unit": "one trial", "estimand": "|value_run - value_exact|/value_exact at one setting"}


def _design(dependent: list[str], metrics: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"hypothesis": "a made-up hypothesis", "variables": {"independent": ["setting_a"], "dependent": dependent},
            "protocol": {"form": "contract", "metrics": metrics if metrics is not None else [METRIC_A, METRIC_B]}}


TWICE = ["quantity_a = (value_run - base)/base", "quantity_b = |value_run - value_exact|/value_exact"]
ONCE = ["quantity_a", "quantity_b = |value_run - value_exact|/value_exact"]


class Quest(ResultChecksMixin):
    _log = logging.getLogger("t")

    def __init__(self, root: Path) -> None:
        self.quest_root = root


def _quest(tmp_path: Path, dependent: list[str], *, issue: dict[str, Any] | None = None, check: str = "") -> Quest:
    """A quest whose reader flagged quantity_a. ``check``: "pass" or "fail" writes FI's record of a known-answer check it
    measured (``check_a``, whose measure reads quantity_a) with that verdict."""
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("def trial(cell, ctx):\n    return {'quantity_a': 1.0}\n", encoding="utf-8")
    design = _design(dependent)
    plan.plan_path(root).write_text(plan.render("a made-up topic", {}, design), encoding="utf-8")
    (root / "needs").mkdir()
    fid = "F-1"
    problem = {"id": "R6", "kind": "metric", "text": "the result reports quantity_a", "headline": False, "name": "quantity_a",
               "status": "different", "file": "simulate.py", "note": "reported another way", "fid": fid, "file_rel": "", "function": ""}
    record = {"version": cr.version(root / "code"), "final": True, "problems": [problem],
              "issues": {fid: {"id": fid, "demonstrated": None, "outcome": "open", **(issue or {})}}}
    cr.save(root, record)
    protocol = {**design["protocol"], "oracles": [{"name": "check_a", "measure": "quantity_a - 1"}, {"name": "check_b", "measure": "quantity_b"}]}
    frozen = {"schema": frozen_protocol.FROZEN_SCHEMA, "protocol": protocol, "sha256": frozen_protocol.sha256(protocol)}
    (root / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps(frozen), encoding="utf-8")
    if check:
        judged = [{"name": "check_a", "measured_by": "engine", "value": 0.0, "passed_by_engine": check == "pass"}]
        (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({"status": "ok", "attempts": [{"judged": judged}]}), encoding="utf-8")
    return Quest(root)


RESULT = {"quantity_a": [0.1, 0.2], "quantity_a_ci": [0.0, 0.3], "quantity_b": 0.5}


def test_the_fact_names_an_id_the_plan_defines_two_ways() -> None:
    found = list(cr.plan_defines_twice(_design(TWICE), _design(TWICE)["protocol"]).values())
    assert [f["id"] for f in found] == ["quantity_a"]
    assert "value_run" in found[0]["dependent"] and "value_exact" in found[0]["metric"]


def test_one_definition_is_not_a_double_definition() -> None:
    assert cr.plan_defines_twice(_design(ONCE), _design(ONCE)["protocol"]) == {}
    same = ["quantity_a = (value_exact - base)/base"]
    assert cr.plan_defines_twice(_design(same), _design(same)["protocol"]) == {}


def test_the_text_comparison_keeps_the_operators_and_reads_the_formula_sentence() -> None:
    def twice(dependent: str, estimand: str) -> bool:
        design = _design([dependent], [{"id": "quantity_a", "kind": "mean", "unit": "one trial", "estimand": estimand}])
        return bool(cr.plan_defines_twice(design, design["protocol"]))

    assert twice("quantity_a = a+b", "a - b at one setting")  # a+b and a - b are two formulas
    assert not twice("quantity_a = a+b", "a+b at one setting")
    assert twice("quantity_a = a/b", "The ratio is defined here. The formula is c/d.")  # the first sentence that is a formula
    assert not twice("quantity_a = mean(error)", "the typical size of the error")  # words: nothing to compare


def test_a_name_defined_twice_with_no_passing_check_is_withheld_and_said_so(tmp_path: Path) -> None:
    found = _quest(tmp_path, TWICE)._code_reader_findings(RESULT)
    assert len(found) == 1 and found[0].contested == "twice" and set(found[0].keys) == {"quantity_a", "quantity_a_ci"}
    assert "two ways" in found[0].note and "no passing check" in found[0].note


def test_a_failed_check_does_not_keep_a_flagged_value(tmp_path: Path) -> None:
    assert len(_quest(tmp_path, TWICE, check="fail")._code_reader_findings(RESULT)) == 1


def test_a_finding_marked_as_about_the_plan_with_no_passing_check_is_withheld(tmp_path: Path) -> None:
    for n, extra in enumerate(({"plan_conflict": True}, {"outcome": code_findings.PLAN_PROBLEM})):
        found = _quest(tmp_path / str(n), ONCE, issue=extra)._code_reader_findings(RESULT)
        assert len(found) == 1 and found[0].contested == "plan_problem", extra


def test_a_flagged_value_with_a_passing_measured_check_is_kept(tmp_path: Path) -> None:
    assert _quest(tmp_path / "a", TWICE, check="pass")._code_reader_findings(RESULT) == []
    assert _quest(tmp_path / "b", ONCE, issue={"plan_conflict": True}, check="pass")._code_reader_findings(RESULT) == []
    quest = _quest(tmp_path / "c", ONCE, check="pass")  # a check that reads another quantity keeps nothing
    assert cr.passing_checks(quest.quest_root, "quantity_b") == [] and cr.passing_checks(quest.quest_root, "quantity_a") == ["check_a"]


def test_a_single_definition_with_a_reader_finding_is_still_taken_out(tmp_path: Path) -> None:
    found = _quest(tmp_path / "a", ONCE, check="pass")._code_reader_findings(RESULT)  # not contested: a passing check does not overrule
    assert len(found) == 1 and found[0].kind == "code_reader" and found[0].contested == ""
    assert len(_quest(tmp_path / "d", ONCE, issue={"demonstrated": True})._code_reader_findings(RESULT)) == 1


def _items(root: Path) -> dict[str, str]:
    return {i.id: i.sentence for i in paper_facts.required_items(root)}


def _write_check(root: Path, *, contested: str = "") -> None:
    check = {"kind": "code_reader", "note": "reported another way", "quantity": "quantity_a", "keys": ["quantity_a"],
             **({"contested": contested} if contested else {})}
    rc = {"removed_quantities": ["quantity_a"], "removed_figures": [], "removed_values": {"quantity_a": [0.1]},
          "checks": [check], "note": "n", "say": "s"}
    (root / "needs" / "RESULT_CHECK.json").write_text(json.dumps(rc), encoding="utf-8")


def test_a_reader_removal_is_never_said_to_be_a_wrong_value_and_has_one_sentence(tmp_path: Path) -> None:
    quest = _quest(tmp_path, TWICE)
    _write_check(quest.quest_root, contested="twice")
    text = " ".join(_items(quest.quest_root).values())
    assert "second model" in text and "value is wrong" not in text and "two different formulas" in text and "holds no check" in text
    assert "follows" not in text and "could not tell" not in text  # left out: no "follows" sentence beside "not reported"
    assert text.count('"quantity_a"') == 1


def test_the_paper_never_claims_which_definition_the_value_follows(tmp_path: Path) -> None:
    # a passing check that reads the quantity does not show which formula every reported value follows: it is named, no more
    text = " ".join(_items(_quest(tmp_path / "a", TWICE, check="pass").quest_root).values())
    assert "two different formulas" in text and '"check_a"' in text and "passed" in text
    assert "follows the definition" not in text and "did not establish which formula" in text
    text = " ".join(_items(_quest(tmp_path / "b", TWICE).quest_root).values())
    assert "two different formulas" in text and "could not tell which of the two" in text and "follows the definition" not in text


def test_the_gap_in_the_plan_is_gone_only_when_the_double_definition_is_gone() -> None:
    proto = _design(TWICE)["protocol"]
    assert cr.double_definition_gaps(_design(TWICE), proto)
    assert cr.double_definition_gaps(_design(ONCE), proto) == []


def test_a_check_is_linked_to_the_names_fi_evaluates_in_its_measure(tmp_path: Path) -> None:
    # `measure` is FI's own executable formula: the quantity it reads is a name FI evaluates, not any word of the text
    for n, (measure, found) in enumerate((("quantity_a/(2*pi) - 1", ["check_a"]), ("the quantity_a of the swing", []),
                                         ("quantity_a +", []), ("quantity_a_ +", []))):
        quest = _quest(tmp_path / str(n), TWICE, check="pass")
        path = quest.quest_root / "needs" / "FROZEN_PROTOCOL.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        frozen["protocol"]["oracles"][0]["measure"] = measure
        frozen["sha256"] = frozen_protocol.sha256(frozen["protocol"])
        path.write_text(json.dumps(frozen), encoding="utf-8")
        assert cr.passing_checks(quest.quest_root, "quantity_a") == found, measure


def test_a_check_whose_measure_is_not_a_formula_or_reads_a_similar_name_is_linked_to_no_quantity(tmp_path: Path) -> None:
    # names are compared exactly: `quantity_a_x` / `quantity-a` differ from `quantity_a`, and a measure that does not parse links to nothing
    for n, measure in enumerate(("quantity-a", "quantity_ax + 1", "quantity_a +", "Quantity_a")):
        quest = _quest(tmp_path / str(n), TWICE, check="pass")
        path = quest.quest_root / "needs" / "FROZEN_PROTOCOL.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        frozen["protocol"]["oracles"][0]["measure"] = measure
        frozen["sha256"] = frozen_protocol.sha256(frozen["protocol"])
        path.write_text(json.dumps(frozen), encoding="utf-8")
        assert cr.passing_checks(quest.quest_root, "quantity_a") == [], measure


def test_a_test_failure_record_with_no_gate_record_is_not_said_to_show_a_wrong_value(tmp_path: Path) -> None:
    quest = _quest(tmp_path, ONCE)
    rc = {"removed_quantities": ["quantity_a"], "removed_figures": [], "removed_values": {"quantity_a": [0.1]},
          "checks": [{"kind": "shown_fault", "note": "n", "quantity": "quantity_a", "keys": ["quantity_a"]}], "note": "n", "say": "s"}
    (quest.quest_root / "needs" / "RESULT_CHECK.json").write_text(json.dumps(rc), encoding="utf-8")
    text = " ".join(_items(quest.quest_root).values())
    assert "did not establish whether the value is wrong or missing" in text and "showed the value is wrong" not in text
