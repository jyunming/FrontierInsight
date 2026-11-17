"""A second model's untested "computed differently from the plan" finding never takes a quantity out of the analysis input at
execute time: the evidence gate decides after a test shows it and a rerun moves a number. FI's own removals are known to the
range check. Neutral made-up study; no model is called."""
from __future__ import annotations

import json
from pathlib import Path

from core import result_checks
from core.engine import Engine
from tests.test_plan_defines_twice import ONCE, RESULT, _quest


def test_an_untested_reader_claim_is_recorded_but_leaves_nothing_out(tmp_path: Path) -> None:
    quest = _quest(tmp_path, ONCE)
    found = quest._code_reader_findings(RESULT)
    assert len(found) == 1 and found[0].keys == [] and set(found[0].claimed) == {"quantity_a", "quantity_a_ci"}
    path = quest.quest_root / "needs" / "RESULT_CHECK.json"
    quest._leave_out_results(found, path, RESULT)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["removed_quantities"] == [] and record["removed_values"] == {}
    assert record["checks"][0]["keys"] == ["quantity_a", "quantity_a_ci"]  # the gate measures it from this record
    assert result_checks.left_out_by_record(quest.quest_root) == {}  # so the analysis input keeps every metric
    assert "stays in the results" in record["say"] and "Do NOT" not in record["say"]


def test_a_demonstrated_claim_is_not_left_out_before_the_rerun_either(tmp_path: Path) -> None:
    quest = _quest(tmp_path, ONCE, issue={"demonstrated": True})
    found = quest._code_reader_findings(RESULT)
    assert len(found) == 1 and found[0].keys == []


def test_the_gate_keeps_a_claim_no_test_shows_and_removes_one_a_rerun_moved(tmp_path: Path) -> None:
    quest = _quest(tmp_path, ONCE)
    path = quest.quest_root / "needs" / "RESULT_CHECK.json"
    quest._leave_out_results(quest._code_reader_findings(RESULT), path, RESULT)
    record = json.loads(path.read_text(encoding="utf-8"))
    for c in record["checks"]:
        c["measured"] = {"status": "not_shown"}
    result_checks._keep_measured(record, list(record["checks"]))
    assert record["removed_quantities"] == [] and record["kept_differences"][0]["measured"]["status"] == "not_shown"
    assert "FI could not show it" in record["say"]


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
