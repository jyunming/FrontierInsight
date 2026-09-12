"""Design-revision provenance (the HARKing record).

`review` and `cross_check` can both route back to `design`, so a hypothesis
CAN legitimately be rewritten after its results are known. The finished paper
looks identical either way, which is what makes the record necessary: "stated
up front" and "rewritten after seeing the numbers" are different scientific
claims.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.engine import _append_design_revision

LOG = logging.getLogger("test")


def test_first_design_is_the_preregistration(tmp_path: Path) -> None:
    h = _append_design_revision({}, {"hypothesis": "A beats B"}, tmp_path, LOG)
    assert len(h) == 1
    assert h[0]["revision"] == 0
    assert h[0]["post_hoc"] is False
    assert "before any result" in h[0]["reason"]
    assert h[0]["hypothesis"] == "A beats B"


def test_reentry_is_flagged_post_hoc(tmp_path: Path) -> None:
    h1 = _append_design_revision({}, {"hypothesis": "A beats B"}, tmp_path, LOG)
    h2 = _append_design_revision(
        {"design_history": h1, "analysis": {"next_step": "re_experiment"}},
        {"hypothesis": "A beats B only at small dt"}, tmp_path, LOG,
    )
    assert h2[1]["post_hoc"] is True
    assert "re_experiment" in h2[1]["reason"]
    assert "results were seen" in h2[1]["reason"]


def test_review_loop_reentry_names_the_verdict(tmp_path: Path) -> None:
    h1 = _append_design_revision({}, {"hypothesis": "H"}, tmp_path, LOG)
    h2 = _append_design_revision(
        {"design_history": h1, "review": {"verdict": "revise"}},
        {"hypothesis": "H"}, tmp_path, LOG,
    )
    assert "verdict=revise" in h2[1]["reason"]


def test_reentry_without_a_known_trigger_still_records(tmp_path: Path) -> None:
    """An unattributed revision must still be visible rather than dropped."""
    h1 = _append_design_revision({}, {"hypothesis": "H"}, tmp_path, LOG)
    h2 = _append_design_revision(
        {"design_history": h1}, {"hypothesis": "H2"}, tmp_path, LOG,
    )
    assert h2[1]["post_hoc"] is True
    assert h2[1]["reason"]


def test_history_is_persisted_next_to_the_quest(tmp_path: Path) -> None:
    h1 = _append_design_revision({}, {"hypothesis": "first"}, tmp_path, LOG)
    _append_design_revision(
        {"design_history": h1, "iteration": 2}, {"hypothesis": "second"},
        tmp_path, LOG,
    )
    on_disk = json.loads(
        (tmp_path / "needs" / "DESIGN_HISTORY.json").read_text(encoding="utf-8"))
    assert [e["revision"] for e in on_disk] == [0, 1]
    assert [e["hypothesis"] for e in on_disk] == ["first", "second"]
    assert on_disk[1]["iteration"] == 2


def test_a_changed_hypothesis_is_reported(tmp_path: Path, caplog) -> None:
    """The warning has to distinguish 'came back but kept the hypothesis'
    from 'rewrote the claim', because only the second is the HARKing risk."""
    h1 = _append_design_revision({}, {"hypothesis": "original"}, tmp_path, LOG)
    with caplog.at_level(logging.WARNING):
        _append_design_revision(
            {"design_history": h1}, {"hypothesis": "different"}, tmp_path, LOG)
    assert "TEXT CHANGED" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _append_design_revision(
            {"design_history": h1}, {"hypothesis": "original"}, tmp_path, LOG)
    assert "hypothesis unchanged" in caplog.text


def test_unwritable_quest_root_does_not_raise(tmp_path: Path) -> None:
    """Provenance is a record, not a gate: a disk problem must not stall a
    quest that is otherwise fine."""
    blocker = tmp_path / "needs"
    blocker.write_text("not a directory", encoding="utf-8")
    h = _append_design_revision({}, {"hypothesis": "H"}, tmp_path, LOG)
    assert h[0]["hypothesis"] == "H"       # still returned in-memory


def test_missing_hypothesis_is_tolerated(tmp_path: Path) -> None:
    h = _append_design_revision({}, {}, tmp_path, LOG)
    assert h[0]["hypothesis"] == ""
