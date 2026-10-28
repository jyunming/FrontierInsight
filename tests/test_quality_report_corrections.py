"""QUALITY.md tells a reader that FI changed a known answer (``.fi/oracle_corrections.json``)."""

from __future__ import annotations

import json
from pathlib import Path

from core import quality_report as qr


def _plan(expected: float, reference: str) -> str:
    return ("# Plan\n\n## The design\n\n```yaml\nprotocol:\n  oracles:\n  - name: spring_period\n"
            f"    expected: {expected}\n    tolerance: 0.001\n    reference: '{reference}'\n```\n")


def _quest(root: Path, with_record: bool = True, with_versions: bool = True) -> Path:
    fi = root / ".fi"
    fi.mkdir(parents=True)
    if with_versions:
        (fi / "plan_versions").mkdir()
        (fi / "plan_versions" / "plan.v1.md").write_text(_plan(1.5, "handbook table 3"), encoding="utf-8")
        (fi / "plan_versions" / "plan.v2.md").write_text(_plan(2.5, "handbook table 3 (FI corrected it)"),
                                                         encoding="utf-8")
    if with_record:
        (fi / "oracle_corrections.json").write_text(json.dumps({"spring_period": {
            "from": 1.5, "to": 2.5, "source": "arithmetic",
            "confirmed_by": {"source": "recompute", "model": "m2", "value": 2.5},
            "why": "the plan's own formula comes to 2.5", "at": "2026-01-01T00:00:00+00:00"}}), encoding="utf-8")
    return root


def _texts(root: Path) -> list[str]:
    return [p["text"] for p in qr._checks_problems(qr._Quest(root))
            if p["source"] == ".fi/oracle_corrections.json"]


def test_a_changed_known_answer_is_listed_with_its_original_source(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q")
    rows = _texts(root)
    assert len(rows) == 1
    assert "known answer 'spring_period' changed by FI: 1.5 -> 2.5, not validated" in rows[0]
    assert "another model (m2)" in rows[0]
    assert "handbook table 3" in rows[0]
    assert "not independently validated" in rows[0]
    qr.write(root)
    assert "spring_period" in (root / "QUALITY.md").read_text(encoding="utf-8")


def test_without_kept_plan_versions_the_source_is_left_out_not_guessed(tmp_path: Path) -> None:
    rows = _texts(_quest(tmp_path / "q", with_versions=False))
    assert len(rows) == 1 and "original source" not in rows[0]


def test_no_corrections_file_no_row(tmp_path: Path) -> None:
    assert _texts(_quest(tmp_path / "q", with_record=False)) == []


def test_the_visible_table_cell_says_not_validated_even_for_a_long_name(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q")
    rec = json.loads((root / ".fi" / "oracle_corrections.json").read_text(encoding="utf-8"))
    long_name = "a_very_long_check_name_" * 4
    rec[long_name] = {**rec.pop("spring_period"), "from": 1.23456789e-05, "to": 98765.4321}
    (root / ".fi" / "oracle_corrections.json").write_text(json.dumps(rec), encoding="utf-8")
    qr.write(root)
    line = next(ln for ln in (root / "QUALITY.md").read_text(encoding="utf-8").splitlines()
                if ln.startswith("| Checks |") and "known answer" in ln)
    assert "not validated" in line and "1.23457e-05" in line and "98765.4" in line


def _row(root: Path) -> dict[str, str]:
    return next(p for p in qr._checks_problems(qr._Quest(root)) if p["source"] == ".fi/oracle_corrections.json")


def test_it_is_among_the_first_three_items_and_counts_neither_fixed_nor_failed(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q")
    with_row = qr.build(root)
    (root / ".fi" / "oracle_corrections.json").rename(root / ".fi" / "gone.json")
    without = qr.build(root)
    (root / ".fi" / "gone.json").rename(root / ".fi" / "oracle_corrections.json")
    (root / ".fi" / "equation_tests.json").write_text(json.dumps({
        "failed": [{"id": f"E{i}", "function": "f"} for i in range(4)], "equations": {}}), encoding="utf-8")
    rep = qr.build(root)
    checks = next(s for s in rep["stages"] if s["stage"] == "checks")
    assert checks["problems"]["total"] == 5 and checks["problems"]["items"][0]["status"] == "disclosed"
    qr.write(root)
    line = next(ln for ln in (root / "QUALITY.md").read_text(encoding="utf-8").splitlines() if ln.startswith("| Checks |"))
    assert "spring_period" in line and "not validated" in line
    # The disclosure changes neither the fixed number nor the rate, and is not a failure.
    assert with_row["problems"]["fixed"] == without["problems"]["fixed"] == 0
    assert with_row["problems"]["fixed_rate"] == without["problems"]["fixed_rate"]
    assert with_row["problems"]["total"] == without["problems"]["total"] and with_row["problems"]["disclosed"] == 1
    assert with_row["failure"] is None
    assert qr._failure(qr._Quest(root), {"kind": "finished", "line": ""}, _row_list(root)) is None


def _row_list(root: Path) -> list[dict[str, str]]:
    return [p for p in qr._checks_problems(qr._Quest(root)) if p["source"] == ".fi/oracle_corrections.json"]


def test_the_original_source_is_the_version_before_the_correction(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q")
    # A later redesign puts the old value back, copying the reference with FI's note.
    (root / ".fi" / "plan_versions" / "plan.v3.md").write_text(
        _plan(1.5, "handbook table 3 (FI corrected the expected value)"), encoding="utf-8")
    text = _row(root)["text"]
    assert "original source was: handbook table 3" in text and "FI corrected" not in text


def test_a_version_that_cannot_be_read_is_skipped(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q")
    (root / ".fi" / "plan_versions" / "plan.v0.md").write_bytes(b"\xff\xfe\x00bad \x80")
    assert len(_texts(root)) == 1


def test_a_check_that_now_matches_the_measurement_says_so_and_odd_values_keep_the_visible_part(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q")
    (root / ".fi" / "oracle_corrections.json").write_text(json.dumps({"c": {
        "from": "a long text value " * 5, "to": ["x"] * 30, "source": "numeric", "agrees_with_measurement": True,
        "why": "w"}}), encoding="utf-8")
    text = _row(root)["text"]
    assert "not validated" in text[:90] and "set to what was measured" in text


def test_the_other_sources_are_worded(tmp_path: Path) -> None:
    root = _quest(tmp_path / "q")
    for source, words in (("challenge", "corrected formula"), ("numeric", "definitions alone"),
                          ("arithmetic", "plan's formula")):
        (root / ".fi" / "oracle_corrections.json").write_text(json.dumps({"c": {
            "from": 1, "to": 2, "source": source, "why": "w"}}), encoding="utf-8")
        assert words in _row(root)["text"]
