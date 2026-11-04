"""FI works every value mark of a paper out again from its own record at the final pass: a digit a later round changed
inside a mark is put back and logged; a key FI can no longer work out is listed as unverified, never kept as FI-filled."""
from __future__ import annotations

from pathlib import Path

from core import paper_facts, paper_values, quality_report
from tests.test_paper_keys_wiring import KEY_MISSING, KEY_SD, PROTOCOL, _engine, _quest


def _filled(tmp_path: Path) -> tuple[Path, str, object]:
    root = _quest(tmp_path)
    rec = paper_values.records_of(root, PROTOCOL)
    text, _v = paper_values.fill_values(f"The SD was {KEY_SD} at noise 0.01.", rec, None)
    return root, text, rec


def test_a_clean_paper_is_not_changed(tmp_path: Path) -> None:
    _root, text, rec = _filled(tmp_path)
    check = paper_values.verify_marks(text, rec)  # type: ignore[arg-type]
    assert check.checked == 1 and check.text == text and not check.repaired and not check.unverified


def test_a_digit_changed_inside_the_mark_is_put_back(tmp_path: Path) -> None:
    _root, text, rec = _filled(tmp_path)
    bad = text.replace("1.22%", "1.92%")
    assert bad != text and paper_facts.value_hits(bad, tmp_path) == []  # the older checks do not see it
    check = paper_values.verify_marks(bad, rec)  # type: ignore[arg-type]
    assert check.text == text
    assert check.repaired and check.repaired[0]["was"].startswith("1.92%") and check.repaired[0]["now"].startswith("1.22%")


def test_a_key_fi_can_no_longer_work_out_is_listed_not_trusted(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    rec = paper_values.records_of(root, PROTOCOL)
    text = f"It was <!-- fi:v {KEY_MISSING} -->0.5<!-- /fi:v -->."
    check = paper_values.verify_marks(text, rec)
    assert check.text == text and not check.repaired
    assert [u["text"] for u in check.unverified] == ["0.5"] and check.unverified[0]["why"]


def test_a_mark_that_already_reads_not_recorded_is_not_an_unverified_value(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    rec = paper_values.records_of(root, PROTOCOL)
    text = f"It was <!-- fi:v {KEY_MISSING} -->{paper_values.NOT_RECORDED}<!-- /fi:v -->."
    assert not paper_values.verify_marks(text, rec).unverified


def test_the_engine_puts_the_value_back_logs_it_and_quality_md_lists_it(tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
    root, text, _rec = _filled(tmp_path)
    paper = root / "paper.md"
    paper.write_text(text.replace("1.22%", "1.92%"), encoding="utf-8")
    eng = _engine(root)
    eng._log.propagate = True
    with caplog.at_level("WARNING"):
        eng._refresh_paper_facts({"paper_md": str(paper), "exec_result": {"duration_s": 3.0}})  # type: ignore[arg-type]
    assert "1.92%" not in paper.read_text(encoding="utf-8") and "1.22%" in paper.read_text(encoding="utf-8")
    assert any("put back" in r.getMessage() for r in caplog.records)
    kept = paper_values.read_marks_record(root)
    assert kept["repaired"][0]["was"].startswith("1.92%")
    problems = quality_report._paper_problems(quality_report._Quest(root))
    assert any("FI put its own value back" in p["text"] and p["source"] == ".fi/value_marks.json" for p in problems)
