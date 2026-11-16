"""A question asked in another unit than the result the plan links to it is answered by that result, with both units stated
and nothing converted; it is never marked "not answered" for the units alone. The flow is the real one: the answers part asks
the plan's model (a fake function) until its last ask, the plan's map it leaves is read by the paper's answers block, and the
quality report lists the gap once as disclosed. Neutral made-up study: a pond's algae."""

from __future__ import annotations

import json
from typing import Any

from core import contract_parts as cp
from core import paper_facts as pf
from core import quality_report as qr
from tests.test_answers_map_part import DESIGN, PROTOCOL, REQ, _ask, _quest

ASKED = {"Q1": {"name": "peak algae", "stated": "report the peak", "by": "metric:algae_peak"}}
OTHER_UNIT = {"answers": {"Q1": {"unit": "m", "by": "none", "plan_unit": "1"}}}
SAME_UNIT = {"answers": {"Q1": {"unit": "1", "by": "none", "plan_unit": "1"}}}


def _after_asks(reply: dict[str, Any], protocol: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    design = {**DESIGN, "protocol": {**(protocol or PROTOCOL), "topic_map": [{"item": "Q1", "by": "metric:algae_peak"}]}}
    out, _seen = _ask(design, [reply] * cp.MAX_ASKS, ASKED)
    assert out.parts["answers"]["valid"]
    return out.design["protocol"]["topic_map"]


def _text(tmp_path, monkeypatch, rows: list[dict[str, Any]], protocol: dict[str, Any] | None = None) -> str:
    root = _quest(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: {**(protocol or PROTOCOL), "topic_map": rows})
    (root / ".fi" / "topic_requirements.json").write_text(json.dumps(REQ), encoding="utf-8")
    return pf.answers_text(root)


def _line(text: str, start: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(start))


def test_a_result_in_another_unit_than_the_question_is_answered_with_both_units_stated(tmp_path, monkeypatch) -> None:
    rows = _after_asks(OTHER_UNIT)
    assert rows[0]["by"] == "metric:algae_peak" and rows[0]["unit_conflict"] == {"asked": "m", "result": "1", "declared": False}
    line = _line(_text(tmp_path, monkeypatch, rows), "- peak algae")
    assert "not answered" not in line and "{{v:algae_peak" in line
    assert "the plan states that the result used, \"algae_peak\", is in a pure number (no unit); the question asks in m, and FI did not convert it)" in line


def test_the_plans_own_declared_unit_is_said_as_the_results_unit(tmp_path, monkeypatch) -> None:
    protocol = {**PROTOCOL, "metrics": [{**PROTOCOL["metrics"][0], "physical_unit": "count"}, PROTOCOL["metrics"][1]]}
    rows = _after_asks({"answers": {"Q1": {"unit": "m", "by": "none"}}}, protocol)
    assert rows[0]["unit_conflict"] == {"asked": "m", "result": "count", "declared": True}
    line = _line(_text(tmp_path, monkeypatch, rows, protocol), "- peak algae")
    assert "(the result used, \"algae_peak\", is in count; the question asks in m, and FI did not convert it)" in line


def test_a_question_in_the_results_own_unit_has_no_unit_sentence(tmp_path, monkeypatch) -> None:
    """The known positive: the units agree, so the line is the answer alone."""
    rows = _after_asks(SAME_UNIT)
    assert "unit_conflict" not in rows[0]
    line = _line(_text(tmp_path, monkeypatch, rows), "- peak algae")
    assert "{{v:algae_peak" in line and "did not convert" not in line


def test_a_result_with_no_value_is_still_not_answered_for_that_reason(tmp_path, monkeypatch) -> None:
    protocol = {**PROTOCOL, "metrics": [*PROTOCOL["metrics"], {"id": "depth", "kind": "mean", "estimand": "pond depth", "unit": "one pond run"}]}
    rows = [{"item": "Q3", "by": "metric:depth", "at": {}, "unit_conflict": {"asked": "m", "result": "1", "declared": False}}]
    line = _line(_text(tmp_path, monkeypatch, rows, protocol), "- water clarity")
    assert line.startswith("- water clarity: not answered (FI holds no value for") and "did not convert" not in line


def _quality(tmp_path, monkeypatch, rows: list[dict[str, Any]], protocol: dict[str, Any] | None = None) -> list[dict[str, str]]:
    from core import frozen_protocol as frozen

    root = _quest(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: {**(protocol or PROTOCOL), "topic_map": rows})
    monkeypatch.setattr(frozen, "protocol_of", lambda r: {**(protocol or PROTOCOL), "topic_map": rows})
    return qr._unit_problems(qr._Quest(root))


def test_the_quality_report_lists_the_unit_gap_once_as_disclosed(tmp_path, monkeypatch) -> None:
    rows = _after_asks(OTHER_UNIT) + [{"item": "Q1", "by": "metric:algae_peak", "unit_conflict": {"asked": "m", "result": "1"}}]
    problems = _quality(tmp_path, monkeypatch, rows)
    assert [(p["status"], p["stage"]) for p in problems] == [("disclosed", "plan")]
    assert "asks in m" in problems[0]["text"] and "did not convert" in problems[0]["text"]


def test_a_unit_gap_on_a_result_with_no_value_is_listed_as_left_out_not_disclosed(tmp_path, monkeypatch) -> None:
    protocol = {**PROTOCOL, "metrics": [*PROTOCOL["metrics"], {"id": "depth", "kind": "mean", "estimand": "pond depth", "unit": "one pond run"}]}
    rows = [{"item": "Q3", "by": "metric:depth", "at": {}, "unit_conflict": {"asked": "m", "result": "1", "declared": False}}]
    problems = _quality(tmp_path, monkeypatch, rows, protocol)
    assert [p["status"] for p in problems] == ["left out"] and "water clarity" in problems[0]["text"]


def test_a_plan_of_the_older_form_keeps_its_reading_in_the_paper_and_the_quality_report(tmp_path, monkeypatch) -> None:
    old = "the result the plan names is in a pure number (no unit), the question asks for m"
    rows = [{"item": "Q1", "by": "metric:algae_peak", "at": {}, "unit_conflict": old}]
    root = tmp_path / "q"
    root.mkdir()
    text = _text(root, monkeypatch, rows)
    assert f"- peak algae: not answered ({old})" in text and "did not convert" not in text
    problems = _quality(tmp_path / "r", monkeypatch, rows)
    assert [p["status"] for p in problems] == ["left out"] and "peak algae" in problems[0]["text"]


def test_an_answer_naming_a_result_worked_out_from_the_plans_does_not_move_the_map_or_its_unit_note() -> None:
    """The plan links a result in the question's unit; the answer names a result worked out from it, in another unit. The plan's
    result stays, with no unit note (it is in the question's unit), and the other result is only a disclosed candidate."""
    p = {**PROTOCOL, "derived": [{"id": "log_peak", "role": "result", "inputs": ["algae_peak"]}],
         "topic_map": [{"item": "Q1", "by": "metric:algae_peak"}]}
    design = {**DESIGN, "protocol": p}
    row = {"Q1": {"unit": "count", "plan_unit": "count", "by": "metric:log_peak", "by_unit": "1"}}
    entry = cp._apply_answers(p, design, row, ASKED)["topic_map"][0]
    assert entry["by"] == "metric:algae_peak" and entry["other_candidate"] == "metric:log_peak" and "unit_conflict" not in entry


def test_the_quality_report_lists_a_candidate_the_answers_part_named_as_disclosed(tmp_path, monkeypatch) -> None:
    rows = [{"item": "Q1", "by": "metric:algae_peak", "at": {}, "other_candidate": "metric:bloom"}]
    from core import frozen_protocol as frozen

    root = _quest(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: {**PROTOCOL, "topic_map": rows})
    monkeypatch.setattr(frozen, "protocol_of", lambda r: {**PROTOCOL, "topic_map": rows})
    (root / ".fi" / "topic_requirements.json").write_text(json.dumps(REQ), encoding="utf-8")
    found = qr._candidate_problems(qr._Quest(root))
    assert [p["status"] for p in found] == ["disclosed"]
    assert "`metric:algae_peak`" in found[0]["text"] and "`metric:bloom`" in found[0]["text"] and "peak algae" in found[0]["text"]
    rows[0].pop("other_candidate")  # the known positive: no candidate, no row
    assert qr._candidate_problems(qr._Quest(root)) == []


def test_every_writer_is_handed_the_units_note_from_the_same_record(tmp_path, monkeypatch) -> None:
    from generation import _paper_context as ctx

    rows = _after_asks(OTHER_UNIT)
    text = _text(tmp_path, monkeypatch, rows)  # the quest on disk and the plan the writers read
    assert "did not convert" in text
    note = "which is in a pure number (no unit); the question asks in m, and FI did not convert it"
    assert any(note in n for n in pf.unit_notes(tmp_path))
    assert note in pf.writer_section(tmp_path)  # the paper's writer
    assert note in ctx.paper_and_claims("paper", tmp_path).claims  # the slides, poster and speech writers
