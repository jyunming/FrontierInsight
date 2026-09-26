"""What the user-facing docs say about behaviour the engine enforces, checked against the engine.

A doc that still describes an older behaviour contradicts the rigor it documents (the 2026-09-26 re-audit found the
capabilities reference saying a failed reviewer "degrades to a neutral accept" while the engine stops instead).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _user_docs() -> dict[str, str]:
    paths = [ROOT / "README.md", ROOT / "vscode-frontier-insight" / "README.md",
             *(p for p in (ROOT / "docs").rglob("*.md") if "audits" not in p.parts)]
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths if p.is_file()}


def test_no_doc_says_a_failed_review_becomes_an_accept() -> None:
    import core.engine as engine

    # What the engine does: a reviewer that cannot be asked stops the quest.
    assert hasattr(engine.Engine, "_pause_for_review_unavailable")
    for name, text in _user_docs().items():
        assert "degrades to a neutral accept" not in text, name
        assert "falls back to a neutral verdict" not in text, name
    for name in ("docs/rigor.md", "docs/capabilities-reference.md"):
        assert "stops the quest" in _user_docs()[name], name
