"""A plan dependent variable that FI placed on no declared result: the one release the gate may give a finding about it
(its failing test reads only text FI wrote from the plan, with current records), and every case that must still stop."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from core import code_review, quality_report
from tests.test_shown_reach import FILES, RESULT, _engine, _quest

PLAN_TEXT_TEST = "import derive\n\n\ndef test_x():\n    assert 'drift' in derive.ROLES, 'not wired'\n"
SOURCE_TEST = ("import derive\nimport inspect\n\n\ndef test_x():\n"
               "    assert 'constraint' in inspect.getsource(derive.derive_cover), 'no constraint'\n")
TRIAL_TEST = "import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n    assert 'drift' in r, 'no drift'\n"
KEY_TEST = "def test_x():\n    import json\n    r = json.loads('{}')\n    assert r['est_a'] > 1, 'low'\n"
PLAN_HEAD = "# Plan"+chr(10)*2+"## The design (used as written)"+chr(10)*2+"```yaml"+chr(10)
PLAN_TAIL = chr(10)+"```"+chr(10)
UNTIED = "mean drift per segment (read from the cells)"


def _gate(engine: Any, test_src: str, *, name: str = UNTIED, headline: tuple[str, ...] = ("est_a",), result: Any = RESULT,
          kind: str = "dependent variable", is_headline: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    root = _quest(engine)
    rel = ".fi/finding_tests/F-1_1.py"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(test_src, encoding="utf-8")
    issue = {"id": "F-1", "file": "derive.py", "function": "", "quantity": name, "kind": "quantity", "changes_result": True,
             "demonstrated": True, "test": rel, "text": f"the dependent variable {name} is computed", "outcome": "not repaired",
             "failure_kind": "assertion", "readings": [{"n": 1, "status": "open"}], "severity": "medium", "first_seen": 1}
    req = {"id": "R2", "kind": kind, "name": name, "status": "missing", "fid": "F-1", "text": issue["text"], "headline": is_headline}
    record = {"issues": {"F-1": issue}, "read_problems": [req]}
    state = {"design": None, "result_json": result, "exec_result": {"returncode": 0}}
    return asyncio.run(engine._shown_fault_check([issue], record, result, list(headline), state)), record


def _released(out: Any, record: dict[str, Any]) -> bool:
    return out is None and bool(record["issues"]["F-1"].get("untied_plan_item"))


def test_a_failing_test_on_plan_text_about_an_untied_dependent_does_not_stop_and_is_kept_in_the_quality_report_only(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    out, record = _gate(engine, PLAN_TEXT_TEST)
    assert _released(out, record)  # on the old gate: stopped, "FI cannot find which of the results it changes"
    saved = record["issues"]["F-1"]
    assert saved["changes_result"] is False and not saved.get("about_plan_text")
    assert not (engine.quest_root / "needs" / "RESULT_CHECK.json").exists()  # nothing is left out
    from core import disclosure

    (engine.quest_root / ".fi").mkdir(exist_ok=True)
    code_review.save(engine.quest_root, record)
    assert UNTIED not in disclosure._code_faults_sentence(engine.quest_root)  # the paper says nothing
    rows = [r for r in quality_report._issue_rows(code_review.load(engine.quest_root), engine.quest_root) if r["id"] == "F-1"]
    assert rows and rows[0]["untied_plan_item"] == [UNTIED]


@pytest.mark.parametrize("src", [SOURCE_TEST, TRIAL_TEST, KEY_TEST], ids=["source-only", "trial", "held key"])
def test_a_test_that_reads_more_than_plan_text_still_stops(tmp_path: Path, src: str) -> None:
    out, record = _gate(_engine(tmp_path), src)
    assert not _released(out, record)
    assert not record["issues"]["F-1"].get("untied_plan_item")
    assert out and out["stuck_reason"] == "shown_fault_open"


def test_a_name_that_is_exactly_a_declared_id_is_not_judged_as_untied(tmp_path: Path) -> None:
    out, record = _gate(_engine(tmp_path), PLAN_TEXT_TEST, name="corr_ab")
    assert not record["issues"]["F-1"].get("untied_plan_item")


def test_a_name_that_only_starts_with_an_id_is_untied_and_never_tied_to_it(tmp_path: Path) -> None:
    name = "est_a compared with the baseline"
    assert code_review.untied_dependents({"read_problems": [{"fid": "F-1", "kind": "dependent variable", "name": name,
                                                             "status": "missing"}]}, {"id": "F-1"}, ["est_a", "est_b"], []) == [name]


def test_the_main_quantity_is_never_released(tmp_path: Path) -> None:
    out, record = _gate(_engine(tmp_path), PLAN_TEXT_TEST, headline=(UNTIED,))
    assert not _released(out, record)
    out, record = _gate(_engine(tmp_path / "b"), PLAN_TEXT_TEST, is_headline=True)
    assert not _released(out, record)


def test_a_mixed_finding_and_other_requirement_kinds_are_not_released() -> None:
    rec = {"read_problems": [{"fid": "F-1", "kind": "dependent variable", "name": UNTIED, "status": "missing"},
                             {"fid": "F-1", "kind": "metric", "name": "cover", "status": "missing"}]}
    assert code_review.untied_dependents(rec, {"id": "F-1"}, ["est_a"], []) == []
    rec = {"read_problems": [{"fid": "F-1", "kind": "metric", "name": UNTIED, "status": "missing"}]}
    assert code_review.untied_dependents(rec, {"id": "F-1"}, ["est_a"], []) == []
    rec = {"read_problems": [{"fid": "F-1", "kind": "dependent variable", "name": UNTIED, "status": "different"}]}
    assert code_review.untied_dependents(rec, {"id": "F-1"}, ["est_a"], []) == []


def test_missing_or_old_records_keep_todays_stop(tmp_path: Path) -> None:
    out, record = _gate(_engine(tmp_path / "a"), PLAN_TEXT_TEST, result={})
    assert not _released(out, record)  # no results
    # _quest writes the analysis record again, so the missing record is made through a patched reader
    import core.analysis_frame as af

    orig = af.read_record
    af.read_record = lambda root: {}
    try:
        out, record = _gate(_engine(tmp_path / "c"), PLAN_TEXT_TEST)
    finally:
        af.read_record = orig
    assert not _released(out, record)


def test_an_older_plan_form_keeps_todays_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import frozen_protocol
    from tests import test_shown_reach as t

    older = {**t.PROTOCOL, "form": "older"}
    monkeypatch.setattr(t, "PROTOCOL", older)
    out, record = _gate(_engine(tmp_path), PLAN_TEXT_TEST)
    assert not _released(out, record)
    assert frozen_protocol  # the record on disk was written from the older form


def test_the_quality_report_names_the_unplaced_entries_without_saying_the_study_does_not_report_them(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    root = _quest(engine)
    block = {"hypothesis": "h", "variables": {"dependent": ["est_a", UNTIED], "independent": ["x"]}}
    (root / "plan.md").write_text(PLAN_HEAD + json.dumps(block) + PLAN_TAIL, encoding="utf-8")
    from core import frozen_protocol, plan

    design = plan.parse((root / "plan.md").read_text(encoding="utf-8")).design or block
    names = code_review.unplaced_dependents(design, frozen_protocol.protocol_of(root))
    assert names == [UNTIED]  # an entry equal to an id is placed
    rows = quality_report._unplaced_problems(quality_report._Quest(root))
    text = json.dumps(rows)
    assert rows and "did not place it on any declared result" in text and "does not report" not in text
