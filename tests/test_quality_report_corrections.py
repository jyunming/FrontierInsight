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
    assert "changed the known answer of 'spring_period' from 1.5 to 2.5" in rows[0]
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
