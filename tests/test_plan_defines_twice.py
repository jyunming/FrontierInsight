"""A name the plan defines twice (a dependent variable written with one formula, a protocol metric of the same name with
another) is a fact of FI's own records, and a second model's word that the code computes it differently is then not a
reason to take its value out or to say the value is wrong. Neutral made-up study; no model is called."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core import code_review as cr
from core import code_findings, paper_facts, plan
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


def _quest(tmp_path: Path, dependent: list[str], *, issue: dict[str, Any] | None = None) -> Quest:
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
    return Quest(root)


RESULT = {"quantity_a": [0.1, 0.2], "quantity_a_ci": [0.0, 0.3], "quantity_b": 0.5}


def test_the_fact_names_an_id_the_plan_defines_two_ways() -> None:
    found = list(cr.plan_defines_twice(_design(TWICE), _design(TWICE)["protocol"]).values())
    assert [f["id"] for f in found] == ["quantity_a"]
    assert "value_run" in found[0]["dependent"] and "value_exact" in found[0]["metric"]


def test_one_definition_is_not_a_double_definition() -> None:
    assert cr.plan_defines_twice(_design(ONCE), _design(ONCE)["protocol"]) == {}
    same = ["quantity_a = (value_exact - base)/base"]  # the same formula in both places
    assert cr.plan_defines_twice(_design(same), _design(same)["protocol"]) == {}


def test_a_reader_finding_about_a_name_defined_twice_takes_nothing_out(tmp_path: Path) -> None:
    quest = _quest(tmp_path, TWICE)
    assert cr.differing_quantities(quest.quest_root, quest.quest_root / "code") == []
    assert quest._code_reader_findings(RESULT) == []


def test_a_single_definition_with_a_reader_finding_is_still_taken_out(tmp_path: Path) -> None:
    quest = _quest(tmp_path, ONCE)
    assert [q["name"] for q in cr.differing_quantities(quest.quest_root, quest.quest_root / "code")] == ["quantity_a"]
    found = quest._code_reader_findings(RESULT)
    assert len(found) == 1 and found[0].kind == "code_reader" and set(found[0].keys) == {"quantity_a", "quantity_a_ci"}


def test_a_demonstrated_finding_on_a_single_definition_is_still_taken_out(tmp_path: Path) -> None:
    quest = _quest(tmp_path, ONCE, issue={"demonstrated": True})
    assert len(quest._code_reader_findings(RESULT)) == 1


def test_a_plan_conflict_or_plan_problem_issue_takes_nothing_out(tmp_path: Path) -> None:
    for extra in ({"plan_conflict": True}, {"outcome": code_findings.PLAN_PROBLEM}):
        quest = _quest(tmp_path / ("a" if "plan_conflict" in extra else "b"), ONCE, issue=extra)
        assert quest._code_reader_findings(RESULT) == [], extra
        assert cr.differing_quantities(quest.quest_root, quest.quest_root / "code") == [], extra


def _paper_items(root: Path) -> list[Any]:
    return paper_facts.required_items(root)


def _write_check(root: Path) -> None:
    check = {"kind": "code_reader", "note": "reported another way", "quantity": "quantity_a", "keys": ["quantity_a"]}
    rc = {"removed_quantities": ["quantity_a"], "removed_figures": [], "removed_values": {"quantity_a": [0.1]},
          "checks": [check], "note": "n", "say": "s"}
    (root / "needs" / "RESULT_CHECK.json").write_text(json.dumps(rc), encoding="utf-8")


def test_a_reader_removal_is_never_said_to_be_a_wrong_value(tmp_path: Path) -> None:
    quest = _quest(tmp_path, ONCE)
    _write_check(quest.quest_root)
    text = " ".join(i.sentence for i in _paper_items(quest.quest_root))
    assert "quantity_a" in text and "value is wrong" not in text and "second model" in text


def test_the_paper_says_the_plan_defined_the_name_two_ways(tmp_path: Path) -> None:
    quest = _quest(tmp_path, TWICE)
    _write_check(quest.quest_root)
    text = " ".join(i.sentence for i in _paper_items(quest.quest_root))
    assert "two ways" in text and "frozen protocol" in text and "value is wrong" not in text


def test_the_gap_in_the_plan_is_gone_only_when_the_double_definition_is_gone() -> None:
    proto = _design(TWICE)["protocol"]
    assert cr.double_definition_gaps(_design(TWICE), proto)  # still there
    assert cr.double_definition_gaps(_design(ONCE), proto) == []  # rewritten to one


def test_a_metric_described_in_words_is_not_taken_for_a_second_definition() -> None:
    words = [{"id": "quantity_a", "kind": "mean", "unit": "one trial", "estimand": "the typical size of the error"}]
    design = _design(["quantity_a = mean(error)"], words)
    assert cr.plan_defines_twice(design, design["protocol"]) == {}
