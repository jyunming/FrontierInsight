"""The numeric oracle must reach ``must_flag_hits``, not just compute.

``test_numeric_oracle.py`` pins the arithmetic. These pin the wiring: a
correct checker that never reaches the router changes nothing, and the
router is what makes a finding blocking (``engine.py:_route_after_review``
forces ``revise`` when ``must_flag_hits`` is non-empty, overriding
``review_loop=False``).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.engine import Engine


class _Recorder:
    """Minimal stand-in exposing only what the oracle helper touches."""

    def __init__(self, tmp: Path, result_json: dict) -> None:
        self.quest_root = tmp
        self._log = SimpleNamespace(
            warning=lambda *a, **k: self.msgs.append(("warn", a)),
            info=lambda *a, **k: self.msgs.append(("info", a)),
        )
        self.msgs: list = []
        self.state = {"result_json": result_json}

    def hits(self, paper: str) -> list[str]:
        return Engine._numeric_oracle_hits(self, paper, self.state)  # type: ignore[arg-type]


def test_transcription_error_becomes_a_must_flag(tmp_path: Path) -> None:
    """The end the whole feature exists for: computed 2.14, written 2.41."""
    rec = _Recorder(tmp_path, {"nils_dipole": 2.14})
    hits = rec.hits("Dipole illumination raised NILS to 2.41 at best focus.")
    assert len(hits) == 1
    assert hits[0].startswith("unverified_number:")
    # The hit names the value and the path, so the rewrite knows what to fix.
    assert "2.41" in hits[0] and "nils_dipole" in hits[0]


def test_consistent_paper_produces_no_hits(tmp_path: Path) -> None:
    rec = _Recorder(tmp_path, {"nils_dipole": 2.14})
    assert rec.hits("Dipole illumination raised NILS to 2.14.") == []


def test_audit_is_written_even_when_clean(tmp_path: Path) -> None:
    """A clean run must leave evidence the check ran — silence would be
    indistinguishable from the check being skipped."""
    rec = _Recorder(tmp_path, {"nils": 2.14})
    rec.hits("NILS was 2.14.")
    audit = tmp_path / "paper" / "numeric_audit.json"
    assert audit.is_file()
    data = json.loads(audit.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert data["result_numbers"] == 1


def test_audit_records_the_finding_when_dirty(tmp_path: Path) -> None:
    rec = _Recorder(tmp_path, {"nils": 2.14})
    rec.hits("NILS was 2.41.")
    data = json.loads((tmp_path / "paper" / "numeric_audit.json").read_text("utf-8"))
    assert data["ok"] is False
    assert data["findings"][0]["kind"] == "transposed"
    assert data["findings"][0]["result_path"] == "nils"


def test_survey_quest_with_no_results_is_not_blocked(tmp_path: Path) -> None:
    """A literature synthesis runs no experiment. The oracle must stay out
    of its way rather than flag every number in it."""
    rec = _Recorder(tmp_path, {})
    hits = rec.hits("The literature reports NILS values between 1.8 and 2.41.")
    assert hits == []
    data = json.loads((tmp_path / "paper" / "numeric_audit.json").read_text("utf-8"))
    assert data["skipped"] is True


def test_checker_failure_is_never_quest_fatal(tmp_path: Path, monkeypatch) -> None:
    """A bug in the oracle must not block a paper. Blocking publication over
    a defect in the checker is strictly worse than not checking."""
    from core import numeric_oracle

    def boom(*_a, **_k):
        raise RuntimeError("oracle exploded")

    monkeypatch.setattr(numeric_oracle, "check", boom)
    rec = _Recorder(tmp_path, {"nils": 2.14})
    assert rec.hits("NILS was 2.41.") == []
    assert any(kind == "warn" for kind, _ in rec.msgs)


def test_router_treats_the_hit_as_blocking() -> None:
    """Pin the contract the hit depends on: a non-empty must_flag_hits
    forces revise even when review_loop is off."""
    engine = SimpleNamespace(
        config=SimpleNamespace(
            engine=SimpleNamespace(max_iterations=4, review_loop=False),
            pauses=SimpleNamespace(review="off"),
        ),
        _log=SimpleNamespace(info=lambda *a, **k: None),
    )
    state = {
        "review": {"verdict": "accept", "must_flag_hits": ["unverified_number: x"]},
        "iteration": 0,
    }
    assert Engine._route_after_review(engine, state) == "revise"  # type: ignore[arg-type]


def test_max_iterations_zero_still_bypasses_the_gate() -> None:
    """Known limitation, pinned so it is a decision rather than a surprise.

    ``_route_after_review`` gates must-flag on ``iteration < max_iterations``,
    so a zero budget disables the blocking behaviour entirely. Fixing that is
    roadmap item A4; this test documents the hole until then.
    """
    engine = SimpleNamespace(
        config=SimpleNamespace(
            engine=SimpleNamespace(max_iterations=0, review_loop=False),
            pauses=SimpleNamespace(review="off"),
        ),
        _log=SimpleNamespace(info=lambda *a, **k: None),
    )
    state = {
        "review": {"verdict": "accept", "must_flag_hits": ["unverified_number: x"]},
        "iteration": 0,
    }
    assert Engine._route_after_review(engine, state) != "revise"
