"""A second model's word that a quantity is computed differently from the plan keeps a result out of the paper only after the
study has been run again with a repair of it and a reported number moved (core/engine.py::_measure_reader_findings). Neutral
made-up study; the rerun itself is replaced by the verdict it would give (what the gate does with each verdict is judged)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core import code_review
from core import paper_facts as pf
from core import result_checks
from core.rerun_check import Verdict
from tests.test_shown_reach import PROTOCOL, _engine, _quest

ISSUE = {"id": "F-9", "file": "derive.py", "function": "", "quantity": "cover", "kind": "quantity", "changes_result": True,
         "demonstrated": None, "outcome": "open", "text": "cover is computed another way", "severity": "high"}
CHECK = {"kind": "code_reader", "note": "the plan's dependent variable `cover` is computed differently from the plan in the code (x)",
         "quantity": "cover", "keys": ["cover"]}


def _setup(tmp_path: Path, monkeypatch: Any, verdict: Verdict | None, calls: list[Any]) -> tuple[Any, dict[str, Any]]:
    engine = _engine(tmp_path)
    root = _quest(engine)
    rc = {"removed_quantities": ["cover"], "removed_figures": [], "removed_values": {"cover": [0.94]}, "checks": [dict(CHECK)],
          "note": CHECK["note"], "say": "A second model compared the code with the plan and found that `cover` is computed differently "
                                        "from the plan (x); the repairs did not remove it. Do NOT report `cover`."}
    (root / "needs" / "RESULT_CHECK.json").write_text(json.dumps(rc), encoding="utf-8")
    monkeypatch.setattr(code_review, "differing_quantities", lambda *a, **k: [{"name": "cover", "fid": "F-9", "kind": "dependent variable",
                                                                              "status": "different", "note": "x"}])

    async def answers(verdicts: Any, record: Any, state: Any, held_to_test: bool = True) -> dict[str, Any]:
        calls.append(held_to_test)
        return {} if verdict is None else {str(i["id"]): verdict for i, _h in verdicts}

    engine._rerun_shown = answers  # type: ignore[method-assign,assignment]
    return engine, {"issues": {"F-9": dict(ISSUE)}}


def _measure(engine: Any, record: dict[str, Any], headline: list[str]) -> Any:
    from core import frozen_protocol, shown_reach

    root = engine.quest_root
    reach = shown_reach.Reach(root, frozen_protocol.protocol_of(root), headline, [])
    return asyncio.run(engine._measure_reader_findings(record, {}, reach))


def _left(root: Path) -> dict[str, Any]:
    return json.loads((root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))


def test_a_reader_finding_whose_repair_moves_no_number_gives_the_result_back_and_says_the_method_differs(tmp_path, monkeypatch) -> None:
    """Before: the quantity stayed out on the reader's word and the paper lost it, with no run of anything."""
    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, Verdict("unchanged", mode="analysis only"), calls)
    assert _measure(engine, record, ["est_a"]) is None
    assert calls == [False]  # asked as a reader's finding: no test holds the repair
    rc = _left(engine.quest_root)
    assert rc["removed_quantities"] == [] and rc["checks"] == [] and "Do NOT report" not in rc["say"]
    assert rc["kept_differences"][0]["quantity"] == "cover" and rc["kept_differences"][0]["measured"]["status"] == "unchanged"
    assert "stays in the results" in rc["say"]
    items = [i for i in pf.required_items(engine.quest_root) if i.kind == pf.METHOD_DIFFERS]
    assert items and "differs from the plan's definition" in items[0].sentence
    assert not [i for i in pf.required_items(engine.quest_root) if i.kind == pf.LEFT_OUT]
    assert "cover" not in result_checks.left_out_by_record(engine.quest_root)


def test_a_reader_finding_whose_repair_moves_a_number_keeps_the_result_out_and_stops_for_the_main_result(tmp_path, monkeypatch) -> None:
    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, Verdict("changed", metrics=["cover"]), calls)
    assert _measure(engine, record, ["est_a"]) is None  # not the main result: stays out
    rc = _left(engine.quest_root)
    assert rc["removed_quantities"] == ["cover"] and rc["checks"][0]["measured"]["status"] == "changed" and "kept_differences" not in rc
    out = _measure(engine, record, ["cover"])
    assert out and out["stuck_reason"] == "shown_fault_open" and "main result" in out["rationale"]


def test_a_reader_finding_that_could_not_be_rerun_stays_out_and_the_record_says_nothing_measured_it(tmp_path, monkeypatch) -> None:
    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, Verdict("could_not", why="the study runs in Docker"), calls)
    assert _measure(engine, record, ["cover"]) is None  # no rerun: never a stop on a reading alone
    rc = _left(engine.quest_root)
    assert rc["removed_quantities"] == ["cover"] and rc["checks"][0]["measured"] == {
        "status": "could_not", "why": "the study runs in Docker", "metrics": [], "notes": [], "mode": ""}


def test_other_checks_that_took_the_same_key_out_keep_it_out() -> None:
    other = {"kind": "shown_fault", "note": "n", "quantity": "other", "keys": ["cover"]}
    record = {"removed_quantities": ["cover"], "checks": [dict(CHECK), other], "removed_values": {"cover": [1.0]}, "say": "x"}
    result_checks._keep_measured(record, [record["checks"][0]])
    assert record["removed_quantities"] == ["cover"] and record["checks"] == [other]


def test_a_file_marker_on_the_line_before_the_fence_is_read_as_the_files_marker() -> None:
    """Before: the model's reply `# file: derive.py` then a fence gave no file, and the rerun said no repair could be written."""
    for reply in ("# file: derive.py\n```python\nx = 1\n```\n", "```python\n# file: derive.py\nx = 1\n```\n"):
        assert code_review.parse_files(reply, ["derive.py"]) == {"derive.py": "x = 1\n"}
    assert code_review.parse_files("```python\nx = 1\n```\n", ["derive.py"]) == {}


def test_a_precision_on_a_result_worked_out_afterwards_is_the_plans_precision() -> None:
    """Before: only a metric the trials return carried the plan's precision, so a move of 1e-7 in a derived result with a
    1e-4 target was a changed number."""
    from core import rerun_check

    plan = {"metrics": [{"id": "est", "kind": "mean"}], "derived": [{"id": "limit"}],
            "precision": {"target_half_width": 1e-4, "metric": "limit"}}
    assert rerun_check.declared_precision(plan) == ("limit", 1e-4)
    base = {"derived/derived/limit": 22.81384918}
    changes = rerun_check.compare(base, {"derived/derived/limit": 22.81384926}, 1e-9)
    assert changes and rerun_check.counted(changes, plan, None, False)[0] == []
    moved = rerun_check.compare(base, {"derived/derived/limit": 22.9}, 1e-9)
    assert len(rerun_check.counted(moved, plan, None, False)[0]) == 1  # a real move still counts


def test_a_repair_that_loses_a_number_is_not_a_measurement_when_no_test_held_it_to_the_finding(tmp_path) -> None:
    import sys

    from tests.test_rerun_check import PROTOCOL as P, _Local, _delivered
    from core import rerun_check

    root = tmp_path / "quest"
    _delivered(root, 1.0)
    copy = rerun_check.copy_code(root, tmp_path / "w")
    text = (copy / "run.py").read_text(encoding="utf-8").replace('{"est": {"by_setting": {"s": 4.5 * scale}}}', "{}")
    (copy / "run.py").write_text(text, encoding="utf-8")
    old = asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, P, 60, None))
    new = asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, P, 60, None, unmeasured_if_dropped=True))
    assert old.status == "changed"
    assert new.status == "could_not" and "gave no number" in new.why and new.crashed
