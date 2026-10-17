"""A quantity the code reader called missing, which the run's results hold under another name, is a difference of names
and does not stop the quest (replay of suite r8 first_order_kinetics: the quest stopped without a paper on F-b5e6edb8,
"k_relative_bias is not reported", while the results held the same per-temperature values as
``k_relative_bias_by_temp``). The r7 EUV run, whose typed-in dose-to-size (a problem of logic) was a real fault, is
untouched."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from core import code_findings
from core import code_review

FIXTURES = Path(__file__).parent / "fixtures" / "shown_fault_names"


def _load(name: str, tmp_path: Path) -> tuple[Path, dict, object]:
    """The fixture's record, with each issue its test showed recorded as shown by an assertion about values (what FI
    records now; the fixtures predate ``failure_kind``, and an issue without it counts as not shown)."""
    shutil.copytree(FIXTURES / name, tmp_path, dirs_exist_ok=True)
    record = code_review.load(tmp_path)
    for issue in (record.get("issues") or {}).values():
        if issue.get("demonstrated") is True:
            issue["failure_kind"] = code_findings.ASSERTION
    code_review.save(tmp_path, record)
    return (tmp_path, code_review.load(tmp_path),
            json.loads((tmp_path / "result_json.json").read_text(encoding="utf-8")))


def test_an_issue_recorded_before_fi_kept_how_its_test_failed_is_not_shown(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "euv_r7", tmp_path, dirs_exist_ok=True)
    record = code_review.load(tmp_path)
    assert record["issues"]["F-bed59bff"]["demonstrated"] is True
    assert not code_findings.shown_open(record)


def test_r8_s_names_are_found_in_the_results(tmp_path: Path) -> None:
    _root, record, result = _load("kinetics_r8", tmp_path)
    issues = record["issues"]
    assert code_review.reported_under_another_name(record, issues["F-b5e6edb8"], result) == ["k_relative_bias_by_temp"]
    assert code_review.reported_under_another_name(record, issues["F-fe5d51ce"], result) == ["k_95_coverage_by_temp"]
    # not held under any name the check can tell: still a missing quantity
    assert code_review.reported_under_another_name(record, issues["F-650ad5f3"], result) == []
    # a problem of logic (a confidence interval set to null) is never a difference of names
    assert code_review.reported_under_another_name(record, issues["F-3af8cc05"], result) == []


def test_r7_s_shown_faults_are_untouched(tmp_path: Path) -> None:
    _root, record, result = _load("euv_r7", tmp_path)
    shown = code_findings.shown_open(record)
    assert "F-bed59bff" in {i["id"] for i in shown}  # the typed-in dose-to-size
    assert all(code_review.reported_under_another_name(record, i, result) == [] for i in shown)


def _gate(name: str, tmp_path: Path, monkeypatch: Any) -> tuple[Path, Any, Any]:
    """The evidence gate on a fixture, with every finding test still failing on the code (the tests and the code are not
    in the fixture: each test file is a stand-in, and its run is FI's record that it failed)."""
    root, _record, result = _load(name, tmp_path)
    return _gate_on(root, result, monkeypatch)


def _gate_on(root: Path, result: Any, monkeypatch: Any) -> tuple[Path, Any, Any]:
    from core import finding_repair
    from tests.test_disclosure import _engine

    record = code_review.load(root)
    for issue in code_findings.shown_open(record):
        test = root / str(issue["test"])
        test.parent.mkdir(parents=True, exist_ok=True)
        test.write_text("def test_x():\n    assert False\n", encoding="utf-8")
    eng = _engine(root)

    async def still_fails(rel: str) -> Any:
        return finding_repair.TestRun("fail", "test_x: the assertion failed")

    monkeypatch.setattr(eng, "_rerun_finding_test", still_fails)
    verdict = asyncio.run(eng._shown_fault_verdict(  # type: ignore[arg-type]
        {"exec_result": {"returncode": 0}, "result_json": result}))
    return root, verdict, eng


def test_r8_does_not_stop_and_the_paper_states_what_is_left_out(tmp_path: Path, monkeypatch: Any) -> None:
    from core import disclosure

    root, verdict, eng = _gate("kinetics_r8", tmp_path, monkeypatch)
    assert verdict is None
    saved = code_review.load(root)["issues"]
    # held under another name: a difference of names, not a fault
    assert saved["F-b5e6edb8"]["changes_result"] is False
    assert saved["F-b5e6edb8"]["reported_as"] == ["k_relative_bias_by_temp"]
    assert {"F-b5e6edb8", "F-fe5d51ce"}.isdisjoint({i["id"] for i in code_findings.shown_open(code_review.load(root))})
    # the plan's quantity the code does not compute: a part not done
    assert saved["F-650ad5f3"]["at_gate"]["outcome"] == "not_computed"
    assert saved["F-650ad5f3"]["at_gate"]["names"] == ["Ea_lnA_correlation"]
    # a fault a test showed in a quantity that is not the study's main one: left out of the results
    assert saved["F-3af8cc05"]["at_gate"]["outcome"] == "left_out"
    assert saved["F-3af8cc05"]["at_gate"]["keys"] == ["Ea_lnA_empirical_correlation"]
    checked = json.loads((root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))
    assert checked["removed_quantities"] == ["Ea_lnA_empirical_correlation"]
    assert checked["removed_values"]["Ea_lnA_empirical_correlation"]
    assert "Do NOT report" in checked["say"] and "`Ea_lnA_correlation` was not computed" in checked["say"]
    record = disclosure.run_record(root)
    assert "computes `Ea_lnA_empirical_correlation` wrongly" in record
    assert "does not compute `Ea_lnA_correlation`, which the plan asks for" in record
    # the gate run again on the same files adds nothing
    asyncio.run(eng._shown_fault_verdict(  # type: ignore[arg-type]
        {"exec_result": {"returncode": 0}, "result_json": _load("kinetics_r8", tmp_path / "again")[2]}))
    assert json.loads((root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8")) == checked


def test_r7_still_stops_on_its_typed_in_dose(tmp_path: Path, monkeypatch: Any) -> None:
    root, verdict, _eng = _gate("euv_r7", tmp_path, monkeypatch)
    assert verdict is not None and verdict["stuck_reason"] == "shown_fault_open"
    assert "dose_to_size_mJ_cm2" in verdict["rationale"]
    assert not (root / "needs" / "RESULT_CHECK.json").exists()


def test_a_fault_whose_quantity_the_results_do_not_hold_still_stops(tmp_path: Path, monkeypatch: Any) -> None:
    root, record, result = _load("kinetics_r8", tmp_path)
    record["issues"]["F-3af8cc05"]["quantity"] = "nothing_like_it"
    for v in record["read_problems"]:
        if v.get("fid") == "F-3af8cc05":
            v["name"], v["quantities"] = "nothing_like_it", []
    code_review.save(root, record)
    _root, verdict, _eng = _gate_on(root, result, monkeypatch)
    assert verdict is not None and "cannot find which of the results it changes" in verdict["rationale"]
