"""A plan dependent variable that FI placed on no declared result: the one release the gate may give a finding about it
(its failing test reads only text FI wrote from the plan, with current records), and every case that must still stop."""
from __future__ import annotations

import json
from pathlib import Path


from core import code_review, quality_report
from tests.test_shown_reach import _engine, _quest


PLAN_HEAD = "# Plan"+chr(10)*2+"## The design (used as written)"+chr(10)*2+"```yaml"+chr(10)
PLAN_TAIL = chr(10)+"```"+chr(10)
UNTIED = "mean drift per segment (read from the cells)"


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
