"""A second model's "computed differently" claim is decided at the evidence gate, and the decision never depends on whether the
gate's own code ran to the end: a claim the gate could not decide is treated like one no rerun could measure (a claim whose
test failed stays out and is disclosed; a claim no test shows stays in and is disclosed). A quantity left out takes the figures
that draw it with it, the writer is never told both "kept" and "do not report", and a claimed-but-kept check is never listed as
left out. Neutral made-up study; no model is called."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core import code_review, paper_facts, quality_report
from core import result_checks
from core.rerun_check import Verdict
from tests.test_reader_findings_measured import ISSUE, NOTE, _measure, _rc
from tests.test_shown_reach import RESULT, _engine, _quest

CLAIMED = {"kind": "code_reader", "note": NOTE, "quantity": "cover", "keys": ["cover"]}
# What the results check writes at execute time since the claim no longer takes anything out before the gate has measured it.
SAY = ("A second model compared the code with the plan and found that `cover` is computed differently from the plan (x). No "
       "test of FI's has shown it yet, so `cover` stays in the results for now.")


def _claimed(tmp_path: Path, monkeypatch: Any, *, shown: bool, verdict: Verdict | None, figure: bool = False) -> tuple[Any, dict[str, Any]]:
    engine = _engine(tmp_path)
    root = _quest(engine)
    rc = {"removed_quantities": [], "removed_figures": [], "removed_values": {}, "checks": [dict(CLAIMED)], "note": NOTE, "say": SAY}
    (root / "needs" / "RESULT_CHECK.json").write_text(json.dumps(rc), encoding="utf-8")
    if figure:
        path = root / "data" / "results" / "derived.json"
        derived = json.loads(path.read_text(encoding="utf-8"))
        derived["figures"][0]["declared"] = {"shows": ["cover", "est_a"], "drawn": ["cover"], "not_drawn": ["est_a"]}
        path.write_text(json.dumps(derived), encoding="utf-8")
    monkeypatch.setattr(code_review, "differing_quantities", lambda *a, **k: [
        {"name": "cover", "fid": "F-9", "kind": "dependent variable", "status": "different", "note": "x"}])
    issue = dict(ISSUE)
    if shown:
        (root / ".fi" / "finding_tests").mkdir(parents=True, exist_ok=True)
        (root / ".fi" / "finding_tests" / "F-9_1.py").write_text("def test_x():\n    assert False\n", encoding="utf-8")
        issue.update(demonstrated=True, test=".fi/finding_tests/F-9_1.py")
    record = {"issues": {"F-9": issue}}
    code_review.save(root, record)

    async def demonstrate(i: Any, rec: Any, state: Any) -> bool:
        return shown

    async def answers(verdicts: Any, rec: Any, state: Any, strict: bool = False) -> dict[str, Any]:
        return {} if verdict is None else {str(i["id"]): verdict for i, _h in verdicts}

    engine._demonstrate_reader_finding = demonstrate  # type: ignore[method-assign,assignment]
    engine._rerun_shown = answers  # type: ignore[method-assign,assignment]
    return engine, record


def _gate_crashes(engine: Any, monkeypatch: Any) -> None:
    async def boom(state: Any) -> Any:
        raise RuntimeError("the gate's own code failed")

    monkeypatch.setattr(engine, "_shown_fault_verdict", boom)


def _run_gate(engine: Any) -> Any:
    state = {"design": None, "result_json": RESULT, "exec_result": {"returncode": 0}, "literature": [], "analysis": {}}
    return asyncio.run(engine._node_evidence_gate(state))


# ---- (1) a gate that raised decides like one that could not measure ---------------------------------------------------------
def test_a_gate_that_crashed_still_takes_out_a_claim_whose_test_failed(tmp_path, monkeypatch) -> None:
    engine, _ = _claimed(tmp_path, monkeypatch, shown=True, verdict=None)
    _gate_crashes(engine, monkeypatch)
    _run_gate(engine)
    rc = _rc(engine.quest_root)
    assert rc["removed_quantities"] == ["cover"] and rc["checks"][0]["measured"]["status"] == "could_not"
    assert "cover" in result_checks.left_out_by_record(engine.quest_root)


def test_a_gate_that_crashed_keeps_and_discloses_a_claim_no_test_shows(tmp_path, monkeypatch) -> None:
    engine, _ = _claimed(tmp_path, monkeypatch, shown=False, verdict=None)
    _gate_crashes(engine, monkeypatch)
    _run_gate(engine)
    rc = _rc(engine.quest_root)
    assert rc["removed_quantities"] == [] and rc["kept_differences"][0]["measured"]["status"] == "not_shown"
    assert "could not show it" in rc["say"]


# ---- (2) the figures that draw a quantity left out are left out with it -----------------------------------------------------
def test_a_figure_drawn_from_a_quantity_a_rerun_moved_is_taken_out(tmp_path, monkeypatch) -> None:
    engine, record = _claimed(tmp_path, monkeypatch, shown=True, verdict=Verdict("changed", metrics=["cover"]), figure=True)
    assert (engine.quest_root / "figures" / "fig1_made_up.png").exists()
    _measure(engine, record, ["est_a"])
    rc = _rc(engine.quest_root)
    assert "cover" in rc["removed_quantities"] and rc["removed_figures"] == ["fig1_made_up.png"]
    assert not (engine.quest_root / "figures" / "fig1_made_up.png").exists()


def test_a_figure_drawn_from_a_quantity_no_rerun_could_measure_is_taken_out(tmp_path, monkeypatch) -> None:
    engine, record = _claimed(tmp_path, monkeypatch, shown=True, verdict=Verdict("could_not", why="docker"), figure=True)
    _measure(engine, record, ["est_a"])
    rc = _rc(engine.quest_root)
    assert rc["removed_quantities"] == ["cover"] and rc["removed_figures"] == ["fig1_made_up.png"]
    assert not (engine.quest_root / "figures" / "fig1_made_up.png").exists()


def test_a_figure_stays_when_the_claim_is_kept(tmp_path, monkeypatch) -> None:
    engine, record = _claimed(tmp_path, monkeypatch, shown=False, verdict=None, figure=True)
    _measure(engine, record, ["est_a"])
    assert _rc(engine.quest_root)["removed_figures"] == [] and (engine.quest_root / "figures" / "fig1_made_up.png").exists()


# ---- (3) the writer is told one thing -------------------------------------------------------------------------------------
def test_the_writer_is_not_told_both_kept_and_do_not_report(tmp_path, monkeypatch) -> None:
    for number, verdict in enumerate((Verdict("could_not", why="docker"), Verdict("changed", metrics=["cover"]))):
        engine, record = _claimed(tmp_path / str(number), monkeypatch, shown=True, verdict=verdict)
        _measure(engine, record, ["est_a"])
        say = _rc(engine.quest_root)["say"]
        assert "stays in the results" not in say and "Do NOT report" in say and "cover" in say


# ---- (4) a claimed-but-kept check is not a left-out quantity ---------------------------------------------------------------
def test_a_claimed_check_that_leaves_nothing_out_is_not_listed_as_left_out(tmp_path, monkeypatch) -> None:
    engine, _ = _claimed(tmp_path, monkeypatch, shown=False, verdict=None)
    root = engine.quest_root
    assert not [i for i in paper_facts.required_items(root) if i.kind == paper_facts.LEFT_OUT and "cover" in i.sentence]
    q = quality_report._Quest.__new__(quality_report._Quest)
    q.result_check = _rc(root)
    assert quality_report._left_out_of_paper(q) == []


# ---- fail closed: when settling the claims itself fails ------------------------------------------------------------------------
def test_a_failure_in_settling_the_claims_leaves_every_undecided_claim_out(tmp_path, monkeypatch) -> None:
    engine, _ = _claimed(tmp_path, monkeypatch, shown=False, verdict=None, figure=True)
    _gate_crashes(engine, monkeypatch)

    def broken(state: Any) -> None:
        raise RuntimeError("settling failed")

    monkeypatch.setattr(engine, "_settle_undecided_reader_claims", broken)
    _run_gate(engine)
    rc = _rc(engine.quest_root)
    assert rc["removed_quantities"] == ["cover"] and rc["removed_figures"] == ["fig1_made_up.png"]
    assert "Do NOT report" in rc["say"] and "stays in the results" not in rc["say"]
    assert "could not decide" in rc["checks"][0]["note"]
    q = quality_report._Quest.__new__(quality_report._Quest)
    q.result_check = rc
    assert "could not decide" in quality_report._left_out_of_paper(q)[0]["why"]


def test_a_failed_lookup_of_the_finding_does_not_keep_a_claim_whose_test_failed(tmp_path, monkeypatch) -> None:
    engine, _ = _claimed(tmp_path, monkeypatch, shown=True, verdict=None)
    _gate_crashes(engine, monkeypatch)

    def lookup_fails(*a: Any, **k: Any) -> Any:
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(code_review, "differing_quantities", lookup_fails)
    _run_gate(engine)
    rc = _rc(engine.quest_root)
    assert rc["removed_quantities"] == ["cover"] and "kept_differences" not in rc


def test_a_leave_out_that_cannot_be_recorded_stops_the_quest(tmp_path, monkeypatch) -> None:
    engine, _ = _claimed(tmp_path, monkeypatch, shown=True, verdict=None)
    _gate_crashes(engine, monkeypatch)
    real = Path.write_text

    def write(self: Path, *a: Any, **k: Any) -> Any:
        if self.name == "RESULT_CHECK.json":
            raise OSError("disk full")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", write)
    out = _run_gate(engine)
    assessment = out["evidence_assessment"]
    assert assessment["verdict"] == "insufficient" and assessment["status"] == "ok" and "stops here" in json.dumps(assessment)
