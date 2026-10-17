"""Replay of a real quest whose known-answer check was reported as failed because a rewrite of the plan moved its expected
value away from the plan's own formula.

In tests/fixtures/replay/expected_moved/ one check had ``expected: 2.97934584`` in one version of the plan
(``oracles_v15.yaml``) and ``expected: 2.9803286656`` in the next (``oracles_v16.yaml``), written by the plan's model while
it was asked about another check; its ``expected_formula`` (a bare ``log`` of a setting minus a ratio of settings) and case
did not change and compute 2.979343. The code's own number (2.979343) was right and was reported as failing the check.
No model is called."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core import oracle_forms as forms
from core import plan
from tests.test_code_review import DESIGN, _engine

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "expected_moved"
MEASURED = 2.979343031616345  # what the code returned, and what the formula computes


def _oracles(version: int) -> list[dict[str, Any]]:
    return yaml.safe_load((FIXTURE / f"oracles_v{version}.yaml").read_text(encoding="utf-8"))["oracles"]


NAME = next(a["name"] for a, b in zip(_oracles(15), _oracles(16)) if a["expected"] != b["expected"])


def _floats(oracles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The checks as FI's later reading has them: `tolerance: 1e-06` is the text "1e-06" to YAML, a number to FI."""
    return [{**o, **{k: float(o[k]) for k in ("expected", "tolerance") if isinstance(o.get(k), str)}} for o in oracles]


def _one(oracles: list[dict[str, Any]]) -> dict[str, Any]:
    return next(o for o in oracles if o["name"] == NAME)


def test_the_evidence_is_what_the_real_quest_held() -> None:
    before, after = _one(_oracles(15)), _one(_oracles(16))
    assert (before["expected"], after["expected"]) == (2.97934584, 2.9803286656)
    assert all(before[k] == after[k] for k in ("expected_formula", "case", "measure", "tolerance"))
    others = {o["name"]: o for o in _oracles(15) if o["name"] != NAME}
    assert others and all(others[o["name"]] == o for o in _oracles(16) if o["name"] != NAME)
    assert abs(forms.formula_value(after)[0] - MEASURED) < 1e-9


def test_before_the_fix_the_moved_value_was_never_looked_at() -> None:
    # The pass that computes expected values from their formulas read this formula as one that can be read two ways
    # (a bare `log`), so it never used it: the check was left as the rewrite wrote it.
    [found] = [f for f in forms.formula_findings({"oracles": _floats(_oracles(16))}) if f["name"] == NAME]
    assert found["state"] == "ambiguous" and abs(found["value"] - MEASURED) < 1e-9


def test_the_number_the_rewrite_moved_is_found_and_the_earlier_one_comes_back() -> None:
    [moved] = forms.moved_expected(_oracles(15), _oracles(16))
    assert (moved["name"], moved["was"], moved["now"]) == (NAME, 2.9803286656, 2.97934584)
    limit = forms._oracle.limit_of(_one(_floats(_oracles(15))))[1]
    assert abs(MEASURED - moved["now"]) <= limit  # the code's number passes the check it was reported as failing
    assert abs(MEASURED - moved["was"]) > limit


def test_nothing_is_put_back_for_a_changed_definition_a_disagreeing_earlier_number_or_an_unchanged_check() -> None:
    before, after = _oracles(15), _oracles(16)
    first = next(iter(_one(after)["case"]))
    changed_case = [{**o, "case": {**o["case"], first: o["case"][first] + 1}} if o["name"] == NAME else o for o in after]
    assert forms.moved_expected(before, changed_case) == []
    changed_formula = [{**o, "expected_formula": o["expected_formula"] + " + 0"} if o["name"] == NAME else o for o in after]
    assert forms.moved_expected(before, changed_formula) == []
    wrong_before = [{**o, "expected": 3.5} if o["name"] == NAME else o for o in before]
    assert forms.moved_expected(wrong_before, after) == []  # the earlier number did not agree with the formula either
    assert forms.moved_expected(before, before) == []
    assert forms.moved_expected(after, after) == []
    no_formula = [{k: v for k, v in o.items() if k != "expected_formula"} for o in after]
    assert forms.moved_expected(before, no_formula) == []


def _quest(tmp_path: Path, oracles: list[dict[str, Any]]):
    engine = _engine(tmp_path)
    protocol = {"grid": {"x": [1, 2]}, "oracles": oracles}
    text = plan.render(engine.config.topic, {}, {**DESIGN, "protocol": protocol})
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="engine")
    return engine


def test_the_plan_has_the_earlier_number_back_and_says_so(tmp_path: Path) -> None:
    before = {o["name"]: o for o in _oracles(15)}
    engine = _quest(tmp_path, _oracles(16))  # the plan as the rewrite left it
    moved = engine._put_back_moved_expected(before)
    assert [m["name"] for m in moved] == [NAME]
    planned = engine._planned_oracles()
    assert planned[NAME]["expected"] == 2.97934584
    assert {n: o["expected"] for n, o in planned.items() if n != NAME} == {n: o["expected"] for n, o in before.items() if n != NAME}
    assert planned[NAME]["tolerance"] == before[NAME]["tolerance"]  # a tolerance is never touched
    text = plan.plan_path(engine.quest_root).read_text(encoding="utf-8")
    assert f"moved the expected value of '{NAME}'" in text
    # A second pass has nothing to do.
    assert engine._put_back_moved_expected(before) == []


def test_a_frozen_protocol_is_never_changed_by_this(tmp_path: Path) -> None:
    from core import frozen_protocol as frozen

    before = {o["name"]: o for o in _oracles(15)}
    engine = _quest(tmp_path, _oracles(16))
    frozen.freeze(engine.quest_root, {"grid": {"x": [1, 2]}, "oracles": _oracles(16)}, approved_by="test", source="test")
    assert engine._put_back_moved_expected(before) == []
    assert engine._planned_oracles()[NAME]["expected"] == 2.9803286656


def test_ln_is_a_function_of_the_calculator() -> None:
    # The request to the plan about a formula it can read two ways names `ln`; the calculator has to know it.
    value, why = forms.formula_value({"expected_formula": _one(_oracles(16))["expected_formula"].replace("log(", "ln("),
                                              "case": _one(_oracles(16))["case"]})
    assert why == "" and abs(value - MEASURED) < 1e-9
