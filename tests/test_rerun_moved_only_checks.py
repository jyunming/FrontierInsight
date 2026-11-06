"""A rerun with a repair of a shown fault that moves only numbers FI measures its checks with (a trial measure that is no
declared result and no key of the results) changes no reported number: the finding does not stop the quest. Neutral
made-up study (tests/test_shown_reach.py); no model is called."""
from __future__ import annotations

from pathlib import Path

from core.rerun_check import Verdict
from tests.test_rerun_check import _gate
from tests.test_shown_reach import DERIVE_COVER, DERIVE_EST

CHECK_KEY = "trials/noise=0.1/oracle_gap/mean"


def _checks_only() -> Verdict:
    return Verdict("changed", metrics=["oracle_gap"], keys=[CHECK_KEY, "trials/noise=0.2/oracle_gap/mean"])


def test_a_rerun_that_moves_only_a_check_measure_does_not_stop_the_quest(tmp_path: Path) -> None:
    out, record = _gate(tmp_path, _checks_only(), headline=("est_b",), test=DERIVE_COVER)
    assert out is None
    issue = record["issues"]["F-1"]
    assert issue["rerun_unchanged"] is True and issue["changes_result"] is False and "at_gate" not in issue and issue["rerun_moved_checks"] is True
    logs = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("run.log"))
    assert "moves only the numbers FI checks the code with, not one result" in logs


def test_a_rerun_that_moves_a_declared_result_still_leaves_it_out_and_stops_for_the_main_result(tmp_path: Path) -> None:
    moved = Verdict("changed", metrics=["cover"], keys=["trials/noise=0.1/cover/mean"])
    out, record = _gate(tmp_path, moved, headline=("est_a",), test=DERIVE_COVER)
    assert out is None and record["issues"]["F-1"]["at_gate"]["outcome"] == "left_out"
    out, _ = _gate(tmp_path / "b", moved, headline=("cover",), test=DERIVE_COVER)
    assert out and out["stuck_reason"] == "shown_fault_open" and "main result" in out["rationale"]


def test_a_trial_measure_that_is_a_key_of_the_results_is_not_a_check_measure(tmp_path: Path) -> None:
    # se_a is printed in the results: its change reaches a reported number
    moved = Verdict("changed", metrics=["se_a"], keys=["trials/noise=0.1/se_a/mean"])
    out, record = _gate(tmp_path, moved, headline=("est_a",), test=DERIVE_COVER, quantity="se_a")
    assert out is None and record["issues"]["F-1"]["at_gate"]["outcome"] == "left_out"
    assert "rerun_unchanged" not in record["issues"]["F-1"]


def test_a_changed_number_that_cannot_be_shown_to_be_a_trial_measure_is_judged_as_before(tmp_path: Path) -> None:
    cases = [(Verdict("changed", metrics=["oracle_gap"]), DERIVE_COVER, "est_b", "cover"),  # no key says where it was measured
             (Verdict("changed", metrics=["oracle_gap"], keys=["derived/derived/oracle_gap"]), DERIVE_COVER, "est_b", "cover"),
             # the same number of another file also moved: the fault is also placed by what its test reaches (est_a, the main result)
             (Verdict("changed", metrics=["oracle_gap"], keys=[CHECK_KEY], unplaced=["file:t.json/x"]), DERIVE_EST, "est_a", "est_a"),
             (Verdict("changed", metrics=["failed trials", "oracle_gap"], keys=[CHECK_KEY]), DERIVE_EST, "est_a", "est_a")]
    for n, (verdict, test, headline, quantity) in enumerate(cases):
        out, _ = _gate(tmp_path / str(n), verdict, headline=(headline,), test=test, quantity=quantity)
        assert out and out["stuck_reason"] == "shown_fault_open", n


def test_a_rerun_that_did_not_write_a_delivered_result_file_is_not_cleared(tmp_path: Path) -> None:
    # a delivered result file the repaired run did not write was never compared: moving only check measures says nothing of it
    unwritten = Verdict("changed", metrics=["oracle_gap"], keys=[CHECK_KEY], unwritten=["data/results/table.json"])
    out, record = _gate(tmp_path, unwritten, headline=("est_b",), test=DERIVE_COVER)
    assert "rerun_moved_checks" not in record["issues"]["F-1"]


def test_the_quality_report_says_new_random_numbers_when_the_repair_redrew_them(tmp_path: Path) -> None:
    redrawn = Verdict("changed", metrics=["oracle_gap"], keys=[CHECK_KEY], redrawn=True)
    out, record = _gate(tmp_path, redrawn, headline=("est_b",), test=DERIVE_COVER)
    assert out is None and record["issues"]["F-1"]["rerun_redrawn"] is True
