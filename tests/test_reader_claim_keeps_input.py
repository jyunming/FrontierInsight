"""A second model's untested "computed differently from the plan" finding never takes a quantity out of the analysis input at
execute time: the evidence gate decides after a test shows it and a rerun moves a number. FI's own removals are known to the
range check. Neutral made-up study; no model is called."""
from __future__ import annotations

import json
from pathlib import Path

from core import result_checks
from core.engine import Engine
from tests.test_plan_defines_twice import ONCE, RESULT, _quest


def test_a_reader_claim_about_a_quantity_is_not_looked_for_in_the_results_and_leaves_nothing_out(tmp_path: Path) -> None:
    quest = _quest(tmp_path, ONCE)
    assert quest._code_reader_findings(RESULT) == [], "a second model's finding is its opinion: no check is made of it"
    path = quest.quest_root / "needs" / "RESULT_CHECK.json"
    quest._leave_out_results([], path, RESULT)
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    assert record.get("removed_quantities", []) == [] and record.get("removed_values", {}) == {}
    assert result_checks.left_out_by_record(quest.quest_root) == {}  # so the analysis input keeps every metric
    tested = _quest(tmp_path / "b", ONCE, issue={"demonstrated": True})
    assert tested._code_reader_findings(RESULT) == [], "not even one a test of the reader's showed"


def test_a_bounded_quantity_fi_took_out_is_not_asked_for_again(tmp_path: Path) -> None:
    from core.plausibility import Assertion, Violation

    root = tmp_path / "q"
    (root / "needs").mkdir(parents=True)
    (root / "needs" / "RESULT_CHECK.json").write_text(json.dumps({"removed_quantities": ["gone_a"]}), encoding="utf-8")
    engine = Engine.__new__(Engine)
    engine.quest_root = root
    missing = Violation("metrics.gone_a", float("nan"), Assertion("metrics.gone_a", min=0, max=1), kind="missing")
    other = Violation("metrics.other", float("nan"), Assertion("metrics.other", min=0, max=1), kind="missing")
    assert engine._removed_by_fi(missing) and not engine._removed_by_fi(other)
